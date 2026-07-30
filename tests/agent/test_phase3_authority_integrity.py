"""Phase 3 authority durability blockers (fix/phase3-lifecycle-closure).

Independent verification of 3052052ee reproduced two blockers that survive a
restart, plus one exactness gap. These tests pin the corrected contract:

1. ``_append_rejection`` persisted the caller's claimed ``fence`` verbatim. The
   ``fence`` column has INTEGER affinity, so SQLite silently rewrote ``"1"``,
   ``1.0`` and ``True`` to the integer ``1`` while the event hash had been
   computed over the original Python value — every later ``recover()`` failed
   with "authority event hash chain is invalid", permanently, because the table
   is append-only. ``[1]`` and ``{}`` could not bind at all and aborted the write
   with an untyped ``sqlite3.ProgrammingError``, losing the audit record. Any
   caller whose completion was *rejected* could therefore brick the store.

2. ``_ensure_schema`` created new partial unique indexes on every ``_connect()``.
   A store written by 66d96c8b4 may hold cross-node duplicate ``attempt_id``s and
   more than one terminal completion row per node (that base bound neither), so
   ``CREATE UNIQUE INDEX`` raised a raw ``sqlite3.IntegrityError`` from inside
   ``_connect`` and took out every authority-serving call, unactionably.

3. ``reserve_budget`` — an authoritative spend transition — bound only fence and
   attempt, not holder, canonical envelope identity, or the node deadline.

Authoritative events are never deleted or rewritten to resolve any of this.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest

from agent.phase2_authority import (
    AuthorityError,
    AuthorityMigrationRequired,
    MalformedEnvelopeFence,
    Phase2AuthorityStore,
    ResultRejection,
    _canonical_hash,
)

NOW = datetime(2026, 7, 30, tzinfo=timezone.utc)

# Every value a caller could put in ``fence`` that is not an integer. The first
# three are the dangerous ones: they compare or coerce close enough to an int to
# reach the durable write, then change representation inside SQLite.
MALFORMED_FENCES = [
    ("1", "str", "'1'"),
    (1.0, "float", "1.0"),
    (True, "bool", "True"),
    ([1], "list", "[1]"),
    ({}, "dict", "{}"),
]

INTEGRITY_INDEXES = (
    "one_accepted_result_per_node",
    "one_terminal_result_per_node",
    "one_grant_per_attempt",
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _node(node_id: str = "n-tool", *, deadline: str = "2030-01-01T00:00:00+00:00") -> dict:
    return {
        "node_id": node_id,
        "objective": f"execute {node_id}",
        "idempotency_key": "b" * 64,
        "execution_surface": "local_tool",
        "lane": "hermes",
        "roots": ["/tmp"],
        "permissions": {"read": ["/tmp"], "write": [], "spawn": []},
        "network_policy": "none",
        "exec_policy": "none",
        "destructive_policy": "forbid",
        "budgets": {"usd_max": 1.0, "tokens_max": 1000, "wall_clock_s_max": 60},
        "deadline_utc": deadline,
        "retry_policy": {"max_attempts": 1, "retry_eligible": False, "backoff_s": [0]},
        "cancellation": {"mode": "cooperative", "grace_s": 30},
        "evidence_policy": {"required_receipt_kinds": ["tool_exec"], "min_receipts": 1},
        "verifier_policy": {
            "required": False,
            "verifier_lane_must_differ": False,
            "clean_workspace_required": False,
        },
        "input_hash": "c" * 64,
    }


def _plan(*nodes: dict) -> dict:
    return {
        "contract_version": 2,
        "graph_id": "g-authority",
        "nodes": list(nodes or (_node("n-tool"),)),
    }


def _granted(
    store: Phase2AuthorityStore,
    now: datetime = NOW,
    *,
    ttl_s: float = 300,
    node: dict | None = None,
    attempt_id: str = "at-1",
    holder: str = "hermes:one",
) -> dict:
    store.seal_plan(_plan(node or _node("n-tool")), policy_hash="d" * 64)
    return store.grant_node(
        "g-authority",
        "n-tool",
        holder=holder,
        ttl_s=ttl_s,
        attempt_id=attempt_id,
        now=now,
    )


def _raw_rows(path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM authority_events ORDER BY id").fetchall()
    finally:
        conn.close()


# ── Blocker 1: the persisted fence must be what the hash chain verifies ─────


@pytest.mark.parametrize("bad_fence,type_name,repr_text", MALFORMED_FENCES)
def test_malformed_fence_completion_is_typed_and_leaves_chain_verifiable(
    tmp_path, bad_fence, type_name, repr_text
):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store)
    forged = {**envelope, "fence": bad_fence}

    with pytest.raises(MalformedEnvelopeFence) as excinfo:
        store.complete_node(forged, result_hash=_sha("r"), now=NOW + timedelta(seconds=5))

    # Typed and fail-closed: still a ResultRejection, so existing handlers hold.
    assert isinstance(excinfo.value, ResultRejection)
    assert isinstance(excinfo.value, AuthorityError)
    assert excinfo.value.reason == "malformed_fence"
    assert excinfo.value.result_hash == _sha("r")
    assert excinfo.value.claimed_fence_type == type_name
    assert excinfo.value.claimed_fence_repr == repr_text

    # The rejection is still auditable: exactly one typed RESULT_REJECTED.
    rejected = [e for e in store.read_events("g-authority") if e["kind"] == "RESULT_REJECTED"]
    assert len(rejected) == 1
    payload = rejected[0]["payload"]
    assert payload["reason"] == "malformed_fence"
    assert payload["result_hash"] == _sha("r")
    assert payload["claimed_fence_malformed"] is True
    assert payload["claimed_fence"] is None
    assert payload["claimed_fence_type"] == type_name
    assert payload["claimed_fence_repr"] == repr_text
    # The authoritative fence is still recorded for reconciliation.
    assert payload["current_fence"] == 1

    # The indexed column holds the normalized value, never the caller's claim.
    assert rejected[0]["fence"] is None

    # recover() verifies the chain including the rejection, and stays clean.
    assert store.recover(now=NOW + timedelta(seconds=10)) == {
        "orphaned_reservations_reconciled": 0
    }
    # A fresh process reaches the same conclusion after restart.
    assert Phase2AuthorityStore(tmp_path / "a.db").recover(
        now=NOW + timedelta(seconds=10)
    ) == {"orphaned_reservations_reconciled": 0}

    # Nothing was accepted, and the store is still fully usable afterwards.
    assert not [e for e in store.read_events("g-authority") if e["kind"] == "RESULT_ACCEPTED"]
    accepted = store.complete_node(
        envelope, result_hash=_sha("r"), now=NOW + timedelta(seconds=6)
    )
    assert accepted["fence"] == 1


@pytest.mark.parametrize("bad_fence", [value for value, _, _ in MALFORMED_FENCES])
def test_stored_fence_representation_equals_the_hashed_representation(tmp_path, bad_fence):
    # The invariant itself: for every durable row, the value SQLite hands back is
    # the value the event hash was computed over. Pre-fix, "1"/1.0/True were
    # stored as the integer 1 while the hash covered the original Python object.
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store)
    with pytest.raises(MalformedEnvelopeFence):
        store.complete_node(
            {**envelope, "fence": bad_fence},
            result_hash=_sha("r"),
            now=NOW + timedelta(seconds=5),
        )

    previous_hash = None
    for row in _raw_rows(store.db_path):
        assert row["fence"] is None or (
            isinstance(row["fence"], int) and not isinstance(row["fence"], bool)
        )
        assert isinstance(row["graph_id"], str)
        assert row["node_id"] is None or isinstance(row["node_id"], str)
        assert row["attempt_id"] is None or isinstance(row["attempt_id"], str)
        recomputed = _canonical_hash(
            {
                "event_id": row["event_id"],
                "ts_utc": row["ts_utc"],
                "kind": row["kind"],
                "graph_id": row["graph_id"],
                "node_id": row["node_id"],
                "attempt_id": row["attempt_id"],
                "fence": row["fence"],
                "payload": json.loads(row["payload_json"]),
                "prev_event_hash": row["prev_event_hash"],
            }
        )
        assert row["event_hash"] == recomputed
        assert row["prev_event_hash"] == previous_hash
        previous_hash = row["event_hash"]


@pytest.mark.parametrize("bad_fence", [1.0, True])
def test_malformed_fence_never_passes_as_an_idempotent_redelivery(tmp_path, bad_fence):
    # 1.0 == 1 and True == 1 in Python, so a naive fence comparison lets a
    # non-canonical envelope through. It must never be read as a redelivery of
    # the accepted result, even with the correct accepted result_hash.
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store)
    store.complete_node(envelope, result_hash=_sha("done"), now=NOW + timedelta(seconds=5))

    with pytest.raises(MalformedEnvelopeFence) as excinfo:
        store.complete_node(
            {**envelope, "fence": bad_fence},
            result_hash=_sha("done"),
            now=NOW + timedelta(seconds=6),
        )
    assert excinfo.value.reason == "malformed_fence"
    events = store.read_events("g-authority")
    assert len([e for e in events if e["kind"] == "RESULT_ACCEPTED"]) == 1
    assert store.recover(now=NOW + timedelta(seconds=7)) == {
        "orphaned_reservations_reconciled": 0
    }


def test_repeated_malformed_rejections_keep_the_store_recoverable(tmp_path):
    # A hostile caller cannot brick the store by looping rejected completions.
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store)
    for index, (bad_fence, _, _) in enumerate(MALFORMED_FENCES):
        with pytest.raises(MalformedEnvelopeFence):
            store.complete_node(
                {**envelope, "fence": bad_fence},
                result_hash=_sha(f"r{index}"),
                now=NOW + timedelta(seconds=5 + index),
            )
    # Interleave a well-formed stale rejection, which keeps its integer claim.
    with pytest.raises(ResultRejection) as excinfo:
        store.complete_node(
            {**envelope, "fence": 99}, result_hash=_sha("r-stale"), now=NOW + timedelta(seconds=20)
        )
    assert excinfo.value.reason == "stale_fence"
    assert not isinstance(excinfo.value, MalformedEnvelopeFence)

    rejected = [e for e in store.read_events("g-authority") if e["kind"] == "RESULT_REJECTED"]
    assert len(rejected) == len(MALFORMED_FENCES) + 1
    assert [e["fence"] for e in rejected] == [None] * len(MALFORMED_FENCES) + [99]
    assert store.recover(now=NOW + timedelta(seconds=30)) == {
        "orphaned_reservations_reconciled": 0
    }


def test_append_event_refuses_column_types_that_do_not_round_trip(tmp_path):
    # The chokepoint invariant, so no future call site can reintroduce the bug.
    store = Phase2AuthorityStore(tmp_path / "a.db")
    conn = store._connect()
    try:
        base = dict(
            kind="RESULT_REJECTED",
            graph_id="g",
            node_id="n",
            attempt_id="a",
            fence=1,
            payload={"reason": "x"},
            now=NOW,
        )
        for field, value, message in (
            ("fence", "1", "fence"),
            ("fence", 1.0, "fence"),
            ("fence", True, "fence"),
            ("fence", [1], "fence"),
            ("graph_id", 7, "graph_id"),
            ("node_id", 7, "node_id"),
            ("attempt_id", 7, "attempt_id"),
        ):
            with pytest.raises(AuthorityError, match=message):
                store._append_event(conn, **{**base, field: value})
        # A non-serializable payload is a typed refusal, not a raw TypeError.
        with pytest.raises(AuthorityError, match="serializable"):
            store._append_event(conn, **{**base, "payload": {"bad": {1, 2}}})
        # Valid types still append.
        assert store._append_event(conn, **base).startswith("ev-")
        assert store._append_event(conn, **{**base, "fence": None, "node_id": None})
    finally:
        conn.close()


# ── Blocker 2: legacy stores must fail closed with an actionable migration ──


def _legacy_schema(path) -> None:
    """Reproduce the 66d96c8b4 schema: tables and triggers, no integrity indexes."""

    with sqlite3.connect(path) as conn:
        for name in INTEGRITY_INDEXES:
            conn.execute(f"DROP INDEX IF EXISTS {name}")


def _append_legacy_event(path, *, kind, node_id, attempt_id, fence, payload) -> None:
    """Append a chain-valid event the prior base could legitimately have written."""

    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        previous = conn.execute(
            "SELECT event_hash FROM authority_events ORDER BY id DESC LIMIT 1"
        ).fetchone()
        prev_hash = previous["event_hash"] if previous else None
        event_id = f"ev-legacy-{kind}-{node_id}-{attempt_id}-{fence}"
        row = {
            "event_id": event_id,
            "ts_utc": NOW.isoformat(),
            "kind": kind,
            "graph_id": "g-authority",
            "node_id": node_id,
            "attempt_id": attempt_id,
            "fence": fence,
            "payload": payload,
            "prev_event_hash": prev_hash,
        }
        conn.execute(
            """INSERT INTO authority_events(
                   event_id, ts_utc, kind, graph_id, node_id, attempt_id, fence,
                   payload_json, prev_event_hash, event_hash
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                row["ts_utc"],
                kind,
                "g-authority",
                node_id,
                attempt_id,
                fence,
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                prev_hash,
                _canonical_hash(row),
            ),
        )


def _legacy_duplicate_attempt_db(tmp_path):
    """Prior base never bound attempt_id globally: same id granted on two nodes."""

    path = tmp_path / "legacy-attempt.db"
    store = Phase2AuthorityStore(path)
    store.seal_plan(_plan(_node("n-one"), _node("n-two")), policy_hash="d" * 64)
    store.grant_node(
        "g-authority", "n-one", holder="h:1", ttl_s=300, attempt_id="at-dup", now=NOW
    )
    _legacy_schema(path)
    _append_legacy_event(
        path, kind="LEASE_GRANTED", node_id="n-two", attempt_id="at-dup", fence=1,
        payload={"envelope": {"attempt_id": "at-dup", "fence": 1}},
    )
    return path


def _legacy_duplicate_terminal_db(tmp_path):
    """Prior base allowed regrant after completion: one terminal row per fence."""

    path = tmp_path / "legacy-terminal.db"
    store = Phase2AuthorityStore(path)
    envelope = _granted(store)
    store.complete_node(envelope, result_hash=_sha("first"), now=NOW + timedelta(seconds=5))
    _legacy_schema(path)
    _append_legacy_event(
        path, kind="NODE_COMPLETED", node_id="n-tool", attempt_id="at-2", fence=2,
        payload={"holder": "hermes:two"},
    )
    return path


def test_legacy_duplicate_attempt_id_surfaces_typed_migration_error(tmp_path):
    path = _legacy_duplicate_attempt_db(tmp_path)

    with pytest.raises(AuthorityMigrationRequired) as excinfo:
        Phase2AuthorityStore(path).recover(now=NOW + timedelta(seconds=60))

    error = excinfo.value
    assert isinstance(error, AuthorityError)  # existing handlers still fail closed
    assert error.db_path == str(path)
    assert [v["invariant"] for v in error.violations] == ["duplicate_grant_attempt_id"]
    violation = error.violations[0]
    assert violation["index"] == "one_grant_per_attempt"
    assert violation["violating_groups"] == 1
    assert violation["sample"] == [{"key": "at-dup", "events": 2}]
    # The message names the offending rows and the operator's next step.
    assert "at-dup" in str(error)
    assert "never auto-deleted or rewritten" in str(error)
    assert "migration_report()" in str(error)


def test_legacy_duplicate_terminal_rows_surface_typed_migration_error(tmp_path):
    path = _legacy_duplicate_terminal_db(tmp_path)

    with pytest.raises(AuthorityMigrationRequired) as excinfo:
        Phase2AuthorityStore(path).recover(now=NOW + timedelta(seconds=60))

    violations = {v["invariant"]: v for v in excinfo.value.violations}
    assert "duplicate_terminal_result" in violations
    assert violations["duplicate_terminal_result"]["sample"] == [
        {"key": "g-authority/n-tool", "events": 2}
    ]
    # A legacy NODE_COMPLETED alongside a new RESULT_ACCEPTED violates only the
    # terminal invariant, not the RESULT_ACCEPTED-only one.
    assert "duplicate_accepted_result" not in violations


def test_two_accepted_results_violate_both_terminal_invariants(tmp_path):
    path = tmp_path / "legacy-accepted.db"
    store = Phase2AuthorityStore(path)
    envelope = _granted(store)
    store.complete_node(envelope, result_hash=_sha("first"), now=NOW + timedelta(seconds=5))
    _legacy_schema(path)
    _append_legacy_event(
        path, kind="RESULT_ACCEPTED", node_id="n-tool", attempt_id="at-2", fence=2,
        payload={"holder": "hermes:two", "result_hash": _sha("second")},
    )

    with pytest.raises(AuthorityMigrationRequired) as excinfo:
        Phase2AuthorityStore(path).current_fence("g-authority", "n-tool")
    assert {v["invariant"] for v in excinfo.value.violations} == {
        "duplicate_accepted_result",
        "duplicate_terminal_result",
    }


def test_every_authority_entry_point_fails_closed_on_a_violating_store(tmp_path):
    path = _legacy_duplicate_attempt_db(tmp_path)
    store = Phase2AuthorityStore(path)
    envelope = {"graph_id": "g-authority", "node_id": "n-one", "fence": 1, "attempt_id": "at-dup"}

    for label, operation in (
        ("recover", lambda: store.recover(now=NOW)),
        ("current_fence", lambda: store.current_fence("g-authority", "n-one")),
        ("current_authority", lambda: store.current_authority("g-authority", "n-one")),
        ("validate_current", lambda: store.validate_current(envelope, now=NOW)),
        ("get_planner_hash", lambda: store.get_planner_hash("g-authority")),
        ("budget_usage", lambda: store.budget_usage("g-authority", "n-one", fence=1)),
        ("seal_plan", lambda: store.seal_plan(_plan(), policy_hash="d" * 64, now=NOW)),
        (
            "grant_node",
            lambda: store.grant_node(
                "g-authority", "n-one", holder="h", ttl_s=1, attempt_id="at-new", now=NOW
            ),
        ),
        ("reserve_budget", lambda: store.reserve_budget(envelope, tokens=1, usd=0.1, now=NOW)),
        (
            "complete_node",
            lambda: store.complete_node(envelope, result_hash=_sha("r"), now=NOW),
        ),
    ):
        with pytest.raises(AuthorityMigrationRequired):
            operation()
        assert label  # each entry point is covered by name


def test_migration_refusal_never_deletes_or_rewrites_events(tmp_path):
    path = _legacy_duplicate_attempt_db(tmp_path)
    before = [dict(row) for row in _raw_rows(path)]

    for _ in range(3):
        with pytest.raises(AuthorityMigrationRequired):
            Phase2AuthorityStore(path).recover(now=NOW)

    after = [dict(row) for row in _raw_rows(path)]
    assert after == before
    # And no index was partially created behind the refusal.
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        names = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }
    finally:
        conn.close()
    assert not names & set(INTEGRITY_INDEXES)


def test_migration_report_is_read_only_and_actionable(tmp_path):
    path = _legacy_duplicate_attempt_db(tmp_path)
    before = path.read_bytes()

    report = Phase2AuthorityStore(path).migration_report()

    assert report["exists"] is True
    assert report["compatible"] is False
    assert set(report["missing_indexes"]) == set(INTEGRITY_INDEXES)
    violation = report["violations"][0]
    assert violation["invariant"] == "duplicate_grant_attempt_id"
    assert violation["index"] == "one_grant_per_attempt"
    assert violation["sample"] == [{"key": "at-dup", "events": 2}]
    assert "operator" in violation["remediation"]
    # Diagnosing must not mutate the quarantined store.
    assert path.read_bytes() == before


def test_quarantined_store_stays_inspectable_for_recovery(tmp_path):
    path = _legacy_duplicate_attempt_db(tmp_path)
    with pytest.raises(AuthorityMigrationRequired):
        Phase2AuthorityStore(path).recover(now=NOW)

    # read_events is the read-only audit surface and keeps working, so an
    # operator can inspect the authoritative history the refusal is protecting.
    events = Phase2AuthorityStore(path).read_events("g-authority")
    kinds = [event["kind"] for event in events]
    assert kinds == ["PLAN_SEALED", "LEASE_GRANTED", "LEASE_GRANTED"]
    assert [e["attempt_id"] for e in events if e["kind"] == "LEASE_GRANTED"] == [
        "at-dup",
        "at-dup",
    ]


def test_migration_report_on_missing_and_empty_stores(tmp_path):
    missing = Phase2AuthorityStore(tmp_path / "absent.db").migration_report()
    assert missing == {
        "db_path": str(tmp_path / "absent.db"),
        "exists": False,
        "compatible": True,
        "missing_indexes": [],
        "violations": [],
    }

    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    report = Phase2AuthorityStore(empty).migration_report()
    assert report["exists"] is True
    assert report["compatible"] is True
    assert set(report["missing_indexes"]) == set(INTEGRITY_INDEXES)


def test_clean_legacy_store_migrates_forward_and_keeps_serving(tmp_path):
    # The refusal must be narrow: a legacy store with no violating rows upgrades
    # in place, keeps its events, and serves authority normally afterwards.
    path = tmp_path / "clean-legacy.db"
    store = Phase2AuthorityStore(path)
    envelope = _granted(store, ttl_s=300)
    before = [dict(row) for row in _raw_rows(path)]
    _legacy_schema(path)
    assert Phase2AuthorityStore(path).migration_report()["missing_indexes"]

    upgraded = Phase2AuthorityStore(path)
    assert upgraded.recover(now=NOW + timedelta(seconds=10)) == {
        "orphaned_reservations_reconciled": 0
    }
    assert upgraded.migration_report() == {
        "db_path": str(path),
        "exists": True,
        "compatible": True,
        "missing_indexes": [],
        "violations": [],
    }
    assert [dict(row) for row in _raw_rows(path)] == before
    record = upgraded.complete_node(
        envelope, result_hash=_sha("r"), now=NOW + timedelta(seconds=11)
    )
    assert record["fence"] == 1


def test_violation_scan_matches_index_null_semantics(tmp_path):
    # A unique index treats every NULL as distinct, so rows with a NULL indexed
    # column can never conflict. A scan that grouped them would refuse a store
    # the index would have accepted.
    path = tmp_path / "nulls.db"
    store = Phase2AuthorityStore(path)
    _granted(store)
    _legacy_schema(path)
    for suffix in ("a", "b"):
        _append_legacy_event(
            path, kind="RESULT_ACCEPTED", node_id=None, attempt_id=f"at-null-{suffix}",
            fence=1, payload={"result_hash": _sha(suffix)},
        )
        _append_legacy_event(
            path, kind="LEASE_GRANTED", node_id=f"n-null-{suffix}", attempt_id=None,
            fence=1, payload={"envelope": {}},
        )

    report = Phase2AuthorityStore(path).migration_report()
    assert report["compatible"] is True
    assert report["violations"] == []
    # And the indexes really do accept these rows.
    assert Phase2AuthorityStore(path).current_fence("g-authority", "n-tool") == 1


def test_migration_violation_sample_is_bounded_but_count_is_exact(tmp_path):
    path = tmp_path / "many.db"
    store = Phase2AuthorityStore(path)
    _granted(store)
    _legacy_schema(path)
    for index in range(8):
        for copy in range(2):
            _append_legacy_event(
                path, kind="LEASE_GRANTED", node_id=f"n-{index}-{copy}",
                attempt_id=f"at-many-{index}", fence=1, payload={"envelope": {}},
            )

    with pytest.raises(AuthorityMigrationRequired) as excinfo:
        Phase2AuthorityStore(path).recover(now=NOW)
    violation = excinfo.value.violations[0]
    assert violation["violating_groups"] == 8
    assert len(violation["sample"]) == 5
    # Deterministic ordering, so the same store always reports the same sample.
    assert [item["key"] for item in violation["sample"]] == [
        f"at-many-{index}" for index in range(5)
    ]


def test_concurrent_first_connect_migrates_exactly_once(tmp_path):
    # _connect() takes the write lock only when an index is missing, and
    # re-checks under it, so racing threads must not collide on CREATE INDEX.
    path = tmp_path / "race.db"
    store = Phase2AuthorityStore(path)
    _granted(store)
    _legacy_schema(path)

    barrier = threading.Barrier(8)
    failures: list[str] = []
    fences: list[int | None] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        try:
            fence = Phase2AuthorityStore(path).current_fence("g-authority", "n-tool")
            with lock:
                fences.append(fence)
        except Exception as exc:  # pragma: no cover - failure detail on regression
            with lock:
                failures.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    assert fences == [1] * 8
    assert Phase2AuthorityStore(path).migration_report()["compatible"] is True


# ── reserve_budget binds exact live authority (contract v2 §9) ──────────────


def test_reserve_budget_rejects_forged_holder(tmp_path):
    # A non-holder that knows graph/node/fence/attempt could otherwise burn the
    # real holder's budget to exhaustion.
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, ttl_s=300)
    forged = {**envelope, "lease": {**envelope["lease"], "holder": "hermes:attacker"}}

    with pytest.raises(AuthorityError, match="holder"):
        store.reserve_budget(forged, tokens=100, usd=0.1, now=NOW)
    assert store.budget_usage("g-authority", "n-tool", fence=1)["reserved_tokens"] == 0


def test_reserve_budget_rejects_non_authoritative_envelope(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, ttl_s=300)

    tampered = {**envelope, "objective": "attacker controlled objective"}
    with pytest.raises(AuthorityError, match="authoritative"):
        store.reserve_budget(tampered, tokens=100, usd=0.1, now=NOW)

    # A raised ceiling is not authoritative either: limits come from the sealed
    # envelope, and the forgery itself is refused.
    inflated = {**envelope, "budgets": {**envelope["budgets"], "tokens_max": 10**9}}
    with pytest.raises(AuthorityError, match="authoritative"):
        store.reserve_budget(inflated, tokens=5000, usd=0.1, now=NOW)
    assert store.budget_usage("g-authority", "n-tool", fence=1)["reserved_tokens"] == 0


@pytest.mark.parametrize("equal_but_wrong_fence", [1.0, True])
def test_reserve_budget_rejects_equal_but_non_canonical_fence(tmp_path, equal_but_wrong_fence):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, ttl_s=300)

    with pytest.raises(AuthorityError, match="authoritative"):
        store.reserve_budget(
            {**envelope, "fence": equal_but_wrong_fence}, tokens=10, usd=0.01, now=NOW
        )
    assert store.budget_usage("g-authority", "n-tool", fence=1)["reserved_tokens"] == 0


def test_reserve_budget_refuses_spend_past_the_node_deadline(tmp_path):
    # A grant's TTL is not capped by deadline_utc, so a lease can outlive the
    # deadline. Spend must not be authorized for a node whose result
    # complete_node can no longer accept.
    store = Phase2AuthorityStore(tmp_path / "a.db")
    node = _node("n-tool", deadline=(NOW + timedelta(seconds=60)).isoformat())
    envelope = _granted(store, ttl_s=600, node=node)
    past_deadline = NOW + timedelta(seconds=90)

    # The lease itself is still unexpired at this moment.
    authority = store.current_authority("g-authority", "n-tool")
    assert authority["effective_expires_utc"] == (NOW + timedelta(seconds=600)).isoformat()

    with pytest.raises(AuthorityError, match="deadline"):
        store.reserve_budget(envelope, tokens=100, usd=0.1, now=past_deadline)
    # The same moment is already refused for completion; the two now agree.
    with pytest.raises(ResultRejection, match="deadline_expired"):
        store.complete_node(envelope, result_hash=_sha("r"), now=past_deadline)


def test_reserve_budget_rejects_revoked_completed_expired_and_stale(tmp_path):
    revoked_store = Phase2AuthorityStore(tmp_path / "revoked.db")
    revoked = _granted(revoked_store, ttl_s=300)
    revoked_store.revoke_node(revoked, now=NOW + timedelta(seconds=5))
    with pytest.raises(AuthorityError, match="revoked"):
        revoked_store.reserve_budget(revoked, tokens=1, usd=0.01, now=NOW + timedelta(seconds=6))

    done_store = Phase2AuthorityStore(tmp_path / "done.db")
    done = _granted(done_store, ttl_s=300)
    done_store.complete_node(done, result_hash=_sha("r"), now=NOW + timedelta(seconds=5))
    with pytest.raises(AuthorityError, match="completed"):
        done_store.reserve_budget(done, tokens=1, usd=0.01, now=NOW + timedelta(seconds=6))

    expired_store = Phase2AuthorityStore(tmp_path / "expired.db")
    expired = _granted(expired_store, ttl_s=30)
    with pytest.raises(AuthorityError, match="expired"):
        expired_store.reserve_budget(
            expired, tokens=1, usd=0.01, now=NOW + timedelta(seconds=31)
        )

    stale_store = Phase2AuthorityStore(tmp_path / "stale.db")
    stale = _granted(stale_store, ttl_s=30)
    stale_store.grant_node(
        "g-authority", "n-tool", holder="hermes:two", ttl_s=300, attempt_id="at-2",
        now=NOW + timedelta(seconds=31),
    )
    with pytest.raises(AuthorityError, match="stale fence"):
        stale_store.reserve_budget(stale, tokens=1, usd=0.01, now=NOW + timedelta(seconds=32))


def test_reserve_budget_still_serves_the_real_holder(tmp_path):
    # The gate is strictly stricter: legitimate reservation, renewal-extended
    # authority, and the budget ceiling all behave exactly as before.
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, ttl_s=30)

    first = store.reserve_budget(envelope, tokens=600, usd=0.6, now=NOW)
    assert first.startswith("res-")
    with pytest.raises(AuthorityError, match="budget"):
        store.reserve_budget(envelope, tokens=500, usd=0.5, now=NOW)

    # A renewal extends the effective expiry, and the original sealed envelope
    # keeps reserving past its own TTL.
    store.renew_node(envelope, ttl_s=300, now=NOW + timedelta(seconds=10))
    store.reconcile_budget(first, actual_tokens=100, actual_usd=0.1, now=NOW)
    second = store.reserve_budget(envelope, tokens=100, usd=0.1, now=NOW + timedelta(seconds=100))
    assert second.startswith("res-")
    usage = store.budget_usage("g-authority", "n-tool", fence=1)
    assert usage == {
        "reserved_tokens": 100,
        "reserved_usd": 0.1,
        "charged_tokens": 100,
        "charged_usd": 0.1,
        "cost_unknown": False,
    }
    assert store.recover(now=NOW + timedelta(seconds=110)) == {
        "orphaned_reservations_reconciled": 0
    }


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_budget_amounts_fail_closed(tmp_path, bad):
    # nan defeats every ceiling comparison (nan > limit is False) and neither
    # nan nor inf is representable in the JSON audit payload.
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, ttl_s=300)

    with pytest.raises(AuthorityError, match="finite"):
        store.reserve_budget(envelope, tokens=1, usd=bad, now=NOW)
    assert store.budget_usage("g-authority", "n-tool", fence=1)["reserved_usd"] == 0.0

    reservation = store.reserve_budget(envelope, tokens=1, usd=0.01, now=NOW)
    with pytest.raises(AuthorityError, match="finite"):
        store.reconcile_budget(reservation, actual_tokens=1, actual_usd=bad, now=NOW)
    with pytest.raises(AuthorityError, match="finite"):
        store.renew_node(envelope, ttl_s=bad, now=NOW + timedelta(seconds=5))
    with pytest.raises(AuthorityError, match="finite"):
        store.grant_node(
            "g-authority", "n-tool", holder="h", ttl_s=bad, attempt_id="at-inf", now=NOW
        )
