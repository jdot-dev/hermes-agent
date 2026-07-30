"""Phase 3 lease/fence lifecycle corrective blockers (fix/phase3-lifecycle-closure).

Independent verification reproduced six blockers in commit 66d96c8b4. These
tests pin the corrected contract (contract v2 §4, §5, §9 and
phase3-lease-fence-lifecycle.md):

1. ``renew_node`` must reject a ttl whose effective expiry exceeds the node
   deadline (ttl is a duration from the renewal moment, capped by deadline).
2. ``revoke_node`` must bind the holder AND canonical envelope identity, not
   just fence/attempt.
3. ``grant_node`` must refuse ``attempt_id`` reuse anywhere in the graph/node
   history, including after revocation.
4. ``grant_node`` must refuse any regrant after terminal completion.
5. Stale/expired/revoked/superseded/conflicting-duplicate completions must
   append exactly one typed ``RESULT_REJECTED`` event in the same decision
   transaction before failing; same-hash redelivery is an idempotent lookup.
6. ``LEASE_RENEWED`` must store bounded external metadata only and never
   re-seal or store an envelope; effective expiry is derived from grant +
   latest valid renewal and the original sealed envelope stays immutable.

Terminal acceptance requires a SHA-256 ``result_hash``; a partial unique index
backstops at most one accepted result per graph/node. Raw results are never
stored.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest

from agent import phase2_enforcement
from agent.phase2_authority import (
    AuthorityError,
    Phase2AuthorityStore,
    ResultRejection,
)

NOW = datetime(2026, 7, 30, tzinfo=timezone.utc)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _node(
    node_id: str = "n-tool",
    surface: str = "local_tool",
    *,
    deadline: str = "2030-01-01T00:00:00+00:00",
) -> dict:
    return {
        "node_id": node_id,
        "objective": f"execute {node_id}",
        "idempotency_key": ("a" if node_id.endswith("model") else "b") * 64,
        "execution_surface": surface,
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
    now: datetime,
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


def _kinds(events: list[dict]) -> list[str]:
    return [e["kind"] for e in events]


# ── Blocker 1: renewal effective expiry must respect node deadline ──────────


def test_renew_rejects_ttl_extending_past_node_deadline(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    node = _node("n-tool", deadline=(NOW + timedelta(seconds=100)).isoformat())
    envelope = _granted(store, NOW, ttl_s=80, node=node)

    # ttl is a duration from the renewal moment; renewal at +20 with ttl=85
    # would put effective expiry at +105, past the +100 deadline -> fail closed.
    with pytest.raises(AuthorityError, match="deadline"):
        store.renew_node(envelope, ttl_s=85, now=NOW + timedelta(seconds=20))

    # A renewal that stays within the deadline is legal while the lease is still live.
    renewed = store.renew_node(envelope, ttl_s=70, now=NOW + timedelta(seconds=20))
    assert renewed["new_expires_utc"] == (NOW + timedelta(seconds=90)).isoformat()


def test_renew_ttl_is_duration_from_renewal_moment(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, NOW, ttl_s=30)

    record = store.renew_node(envelope, ttl_s=50, now=NOW + timedelta(seconds=10))
    # Effective expiry = renewal moment (T+10) + 50s = T+60.
    assert record["new_expires_utc"] == (NOW + timedelta(seconds=60)).isoformat()
    assert record["prior_expires_utc"] == (NOW + timedelta(seconds=30)).isoformat()
    # The original sealed envelope is untouched.
    assert record["envelope"]["lease"]["granted_utc"] == NOW.isoformat()
    assert record["envelope"]["lease"]["ttl_s"] == 30


# ── Blocker 2: revoke must bind holder and canonical envelope identity ──────


def test_revoke_rejects_forged_holder(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, NOW, ttl_s=300)

    forged = {**envelope, "lease": {**envelope["lease"], "holder": "hermes:attacker"}}
    with pytest.raises(AuthorityError, match="holder"):
        store.revoke_node(forged, now=NOW + timedelta(seconds=5))


def test_revoke_rejects_non_authoritative_envelope_bytes(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, NOW, ttl_s=300)

    # Correct fence/attempt/holder but tampered objective => not the canonical
    # sealed envelope. Fail closed.
    tampered = {**envelope, "objective": "attacker controlled objective"}
    with pytest.raises(AuthorityError, match="authoritative"):
        store.revoke_node(tampered, now=NOW + timedelta(seconds=5))


def test_revoke_binds_canonical_envelope_identity(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, NOW, ttl_s=300)
    record = store.revoke_node(envelope, now=NOW + timedelta(seconds=5))
    assert record["fence"] == 1
    assert "lease_revoked" in store.validate_current(envelope, now=NOW + timedelta(seconds=5))


# ── Blocker 3: attempt_id uniqueness across the graph/node ──────────────────


def test_grant_rejects_attempt_id_reuse_after_revocation(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, NOW, ttl_s=300, attempt_id="at-1")
    store.revoke_node(envelope, now=NOW + timedelta(seconds=5))

    with pytest.raises(AuthorityError, match="attempt"):
        store.grant_node(
            "g-authority",
            "n-tool",
            holder="hermes:two",
            ttl_s=300,
            attempt_id="at-1",  # reused
            now=NOW + timedelta(seconds=5),
        )

    # A fresh attempt_id regrants cleanly at the next fence.
    second = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:two",
        ttl_s=300,
        attempt_id="at-2",
        now=NOW + timedelta(seconds=5),
    )
    assert second["fence"] == 2


def test_grant_rejects_attempt_id_reuse_after_natural_expiry(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    _granted(store, NOW, ttl_s=30, attempt_id="at-1")

    # Lease expired naturally; attempt_id still may not be reused.
    with pytest.raises(AuthorityError, match="attempt"):
        store.grant_node(
            "g-authority",
            "n-tool",
            holder="hermes:two",
            ttl_s=300,
            attempt_id="at-1",
            now=NOW + timedelta(seconds=31),
        )


# ── Blocker 4: terminal completion permanently closes the node ──────────────


def test_grant_rejected_after_terminal_completion(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, NOW, ttl_s=300)
    store.complete_node(envelope, result_hash=_sha("result"), now=NOW + timedelta(seconds=5))

    with pytest.raises(AuthorityError, match="terminal|completed"):
        store.grant_node(
            "g-authority",
            "n-tool",
            holder="hermes:two",
            ttl_s=300,
            attempt_id="at-2",
            now=NOW + timedelta(seconds=6),
        )
    # Fence is unchanged by the failed regrant.
    assert store.current_fence("g-authority", "n-tool") == 1


# ── Blocker 5: typed auditable rejection events in the same transaction ─────


def test_stale_completion_appends_result_rejected(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    first = _granted(store, NOW, ttl_s=30, attempt_id="at-1")
    # Expire naturally then regrant at fence 2.
    store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:two",
        ttl_s=300,
        attempt_id="at-2",
        now=NOW + timedelta(seconds=31),
    )

    with pytest.raises(ResultRejection) as excinfo:
        store.complete_node(first, result_hash=_sha("old-result"), now=NOW + timedelta(seconds=32))
    assert excinfo.value.reason == "stale_fence"

    rejected = [e for e in store.read_events("g-authority") if e["kind"] == "RESULT_REJECTED"]
    assert len(rejected) == 1
    payload = rejected[0]["payload"]
    if isinstance(payload, str):
        import json

        payload = json.loads(payload)
    assert payload["reason"] == "stale_fence"
    assert payload["result_hash"] == _sha("old-result")
    assert rejected[0]["fence"] == 1


def test_expired_completion_appends_result_rejected(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, NOW, ttl_s=30)

    with pytest.raises(ResultRejection) as excinfo:
        store.complete_node(envelope, result_hash=_sha("r"), now=NOW + timedelta(seconds=31))
    assert excinfo.value.reason == "lease_expired"
    rejected = [e for e in store.read_events("g-authority") if e["kind"] == "RESULT_REJECTED"]
    assert len(rejected) == 1


def test_revoked_completion_appends_result_rejected(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, NOW, ttl_s=300)
    store.revoke_node(envelope, now=NOW + timedelta(seconds=5))

    with pytest.raises(ResultRejection) as excinfo:
        store.complete_node(envelope, result_hash=_sha("r"), now=NOW + timedelta(seconds=6))
    assert excinfo.value.reason == "lease_revoked"
    rejected = [e for e in store.read_events("g-authority") if e["kind"] == "RESULT_REJECTED"]
    assert len(rejected) == 1


def test_conflicting_duplicate_after_acceptance_appends_result_rejected(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, NOW, ttl_s=300)
    accepted = store.complete_node(
        envelope, result_hash=_sha("result-A"), now=NOW + timedelta(seconds=5)
    )
    assert accepted["result_hash"] == _sha("result-A")

    # A DIFFERENT result hash after acceptance is a conflicting duplicate.
    with pytest.raises(ResultRejection) as excinfo:
        store.complete_node(
            envelope, result_hash=_sha("result-B"), now=NOW + timedelta(seconds=6)
        )
    assert excinfo.value.reason in {"node_completed", "conflicting_duplicate"}

    # Exactly one RESULT_ACCEPTED, exactly one RESULT_REJECTED.
    events = store.read_events("g-authority")
    accepted_events = [e for e in events if e["kind"] == "RESULT_ACCEPTED"]
    rejected_events = [e for e in events if e["kind"] == "RESULT_REJECTED"]
    assert len(accepted_events) == 1
    assert len(rejected_events) == 1


def test_rejection_is_atomic_with_decision_transaction(tmp_path):
    # A failed completion must append its rejection event AND roll back nothing
    # else; the event must be durable for audit. Verify the store is still
    # usable and the chain remains valid after a rejection.
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, NOW, ttl_s=30)
    with pytest.raises(ResultRejection):
        store.complete_node(envelope, result_hash=_sha("r"), now=NOW + timedelta(seconds=31))
    # Chain still verifies.
    assert store.recover(now=NOW + timedelta(seconds=40)) == {
        "orphaned_reservations_reconciled": 0
    }


# ── Result identity: hash validation, idempotent redelivery, backstop ───────


def test_complete_requires_valid_sha256_result_hash(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, NOW, ttl_s=300)

    for bad in ("", "notahex", "zz" * 32, "A" * 64, 123, None):
        with pytest.raises(AuthorityError, match="result_hash"):
            store.complete_node(envelope, result_hash=bad, now=NOW + timedelta(seconds=5))


def test_same_hash_redelivery_is_idempotent_lookup(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, NOW, ttl_s=300)
    first = store.complete_node(
        envelope, result_hash=_sha("same"), now=NOW + timedelta(seconds=5)
    )

    # Same hash redelivered: idempotent prior acceptance, NO new event.
    events_before = store.read_events("g-authority")
    again = store.complete_node(
        envelope, result_hash=_sha("same"), now=NOW + timedelta(seconds=6)
    )
    events_after = store.read_events("g-authority")

    assert again["result_hash"] == first["result_hash"]
    assert again["fence"] == first["fence"]
    assert again.get("idempotent") is True
    assert len(events_after) == len(events_before)
    assert len([e for e in events_after if e["kind"] == "RESULT_ACCEPTED"]) == 1


def test_at_most_one_accepted_result_backed_by_schema(tmp_path):
    # The partial unique index must exist and enforce one RESULT_ACCEPTED per
    # graph/node even if the application guard is bypassed.
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, NOW, ttl_s=300)
    store.complete_node(envelope, result_hash=_sha("r"), now=NOW + timedelta(seconds=5))

    path = store.db_path
    with sqlite3.connect(path) as conn:
        idx = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='index' AND sql LIKE '%RESULT_ACCEPTED%'"
        ).fetchall()
        assert idx, "expected a partial unique index backstopping RESULT_ACCEPTED"

        # Attempt to bypass the app guard and insert a second acceptance raw.
        last = conn.execute(
            "SELECT event_hash FROM authority_events ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO authority_events(
                       event_id, ts_utc, kind, graph_id, node_id, attempt_id, fence,
                       payload_json, prev_event_hash, event_hash
                   ) VALUES ('ev-x', '2030-01-01T00:00:00+00:00', 'RESULT_ACCEPTED',
                             'g-authority', 'n-tool', 'at-1', 1,
                             '{"result_hash":"aa"}', ?, 'ev-x-hash')""",
                (last,),
            )


def test_result_hash_is_bounded_and_raw_result_not_stored(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, NOW, ttl_s=300)
    store.complete_node(envelope, result_hash=_sha("the actual result body"), now=NOW + timedelta(seconds=5))
    for event in store.read_events("g-authority"):
        assert "the actual result body" not in str(event["payload_json"])


# ── Blocker 6: immutable envelope, external renewal metadata ────────────────


def test_renewed_event_carries_bounded_metadata_not_envelope(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, NOW, ttl_s=30)
    store.renew_node(envelope, ttl_s=60, now=NOW + timedelta(seconds=10))

    renewed_events = [e for e in store.read_events("g-authority") if e["kind"] == "LEASE_RENEWED"]
    assert len(renewed_events) == 1
    import json

    payload = json.loads(renewed_events[0]["payload_json"])
    # No sealed envelope embedded in the renewal event.
    assert "envelope" not in payload
    # Bounded external metadata only.
    assert payload["prior_expires_utc"] == (NOW + timedelta(seconds=30)).isoformat()
    assert payload["new_expires_utc"] == (NOW + timedelta(seconds=70)).isoformat()
    assert payload["holder"] == "hermes:one"


def test_grant_event_envelope_is_immutable_across_renewal(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, NOW, ttl_s=30)
    original_grant_hash = [
        e["event_hash"] for e in store.read_events("g-authority") if e["kind"] == "LEASE_GRANTED"
    ][0]

    store.renew_node(envelope, ttl_s=60, now=NOW + timedelta(seconds=10))

    # The sealed LEASE_GRANTED envelope bytes/event are unchanged.
    grants = [e for e in store.read_events("g-authority") if e["kind"] == "LEASE_GRANTED"]
    assert len(grants) == 1
    assert grants[0]["event_hash"] == original_grant_hash
    import json

    sealed = json.loads(grants[0]["payload_json"])["envelope"]
    assert sealed["lease"]["granted_utc"] == NOW.isoformat()
    assert sealed["lease"]["ttl_s"] == 30


def test_effective_expiry_derived_from_grant_plus_latest_renewal(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, NOW, ttl_s=30)
    store.renew_node(envelope, ttl_s=60, now=NOW + timedelta(seconds=10))
    store.renew_node(envelope, ttl_s=100, now=NOW + timedelta(seconds=20))

    # Latest renewal: T+20 + 100 = T+120. Original envelope (TTL 30) still
    # authorizes execution up to the derived effective expiry.
    authority = store.current_authority("g-authority", "n-tool")
    assert authority["effective_expires_utc"] == (NOW + timedelta(seconds=120)).isoformat()
    # Execution authorization uses the ORIGINAL sealed envelope.
    assert authority["envelope"]["lease"]["ttl_s"] == 30
    assert store.validate_current(envelope, now=NOW + timedelta(seconds=110)) == []
    assert "lease_expired" in store.validate_current(envelope, now=NOW + timedelta(seconds=121))


def test_completion_with_original_envelope_uses_derived_expiry(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, NOW, ttl_s=30)
    store.renew_node(envelope, ttl_s=100, now=NOW + timedelta(seconds=10))

    # Past the original 30s TTL but within the derived effective expiry: the
    # ORIGINAL sealed envelope still completes authoritatively.
    record = store.complete_node(
        envelope, result_hash=_sha("done"), now=NOW + timedelta(seconds=50)
    )
    assert record["fence"] == 1


# ── Hash-chain tamper coverage for each new event kind ──────────────────────


def _tamper(path, kind):
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TRIGGER authority_events_no_update")
        conn.execute(
            "UPDATE authority_events SET payload_json = '{}' WHERE kind = ?", (kind,)
        )


@pytest.mark.parametrize("kind", ["LEASE_RENEWED", "LEASE_REVOKED", "RESULT_ACCEPTED", "RESULT_REJECTED"])
def test_recovery_detects_tamper_on_each_lifecycle_event(tmp_path, kind):
    path = tmp_path / "a.db"
    store = Phase2AuthorityStore(path)
    envelope = _granted(store, NOW, ttl_s=30)
    store.renew_node(envelope, ttl_s=100, now=NOW + timedelta(seconds=10))
    store.complete_node(envelope, result_hash=_sha("r"), now=NOW + timedelta(seconds=20))
    # Force a rejection event too (redeliver conflicting after completion).
    with pytest.raises(ResultRejection):
        store.complete_node(envelope, result_hash=_sha("other"), now=NOW + timedelta(seconds=21))
    # Add a revoked lease on a second node to ensure a LEASE_REVOKED row exists.
    store.seal_plan(
        {"contract_version": 2, "graph_id": "g-two", "nodes": [_node("n-x")]},
        policy_hash="d" * 64,
    )
    e2 = store.grant_node(
        "g-two", "n-x", holder="hermes:one", ttl_s=300, attempt_id="ax-1", now=NOW
    )
    store.revoke_node(e2, now=NOW + timedelta(seconds=1))

    # Sanity: recovery is clean before tampering.
    assert Phase2AuthorityStore(path).recover(now=NOW + timedelta(seconds=25)) == {
        "orphaned_reservations_reconciled": 0
    }
    _tamper(path, kind)
    with pytest.raises(AuthorityError, match="hash chain"):
        Phase2AuthorityStore(path).recover(now=NOW + timedelta(seconds=25))


# ── Cross-process contention: two independent store connections ─────────────


def test_cross_process_completion_exactly_one_winner(tmp_path):
    # Two independent store instances (separate connections, as in separate
    # processes) race the same completion. SQLite BEGIN IMMEDIATE serializes;
    # exactly one wins, the loser gets a typed rejection.
    path = tmp_path / "a.db"
    Phase2AuthorityStore(path).seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    store_a = Phase2AuthorityStore(path)
    store_b = Phase2AuthorityStore(path)
    envelope = store_a.grant_node(
        "g-authority", "n-tool", holder="hermes:one", ttl_s=300, attempt_id="at-1", now=NOW
    )

    barrier = threading.Barrier(2)
    outcomes: list[tuple[str, object]] = []
    lock = threading.Lock()

    def racer(store):
        barrier.wait()
        try:
            rec = store.complete_node(
                envelope, result_hash=_sha("same-result"), now=NOW + timedelta(seconds=5)
            )
            with lock:
                outcomes.append(("accepted", rec))
        except ResultRejection as exc:
            with lock:
                outcomes.append(("rejected", exc.reason))

    threads = [threading.Thread(target=racer, args=(s,)) for s in (store_a, store_b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Same hash raced concurrently: one is the original acceptance; the other
    # is an idempotent lookup (same hash, idempotent=True) or a typed rejection.
    # Never two original acceptances — the partial unique index backstops that.
    original = [o for o in outcomes if o[0] == "accepted" and not o[1].get("idempotent")]
    assert len(original) == 1
    events = store_a.read_events("g-authority")
    assert len([e for e in events if e["kind"] == "RESULT_ACCEPTED"]) == 1


def test_cross_process_conflicting_results_exactly_one_acceptance(tmp_path):
    path = tmp_path / "a.db"
    Phase2AuthorityStore(path).seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    store_a = Phase2AuthorityStore(path)
    store_b = Phase2AuthorityStore(path)
    envelope = store_a.grant_node(
        "g-authority", "n-tool", holder="hermes:one", ttl_s=300, attempt_id="at-1", now=NOW
    )

    barrier = threading.Barrier(2)
    outcomes: list[tuple[str, object]] = []
    lock = threading.Lock()

    def racer(store, result):
        barrier.wait()
        try:
            rec = store.complete_node(
                envelope, result_hash=_sha(result), now=NOW + timedelta(seconds=5)
            )
            with lock:
                outcomes.append(("accepted", rec))
        except ResultRejection as exc:
            with lock:
                outcomes.append(("rejected", exc.reason))

    t1 = threading.Thread(target=racer, args=(store_a, "result-A"))
    t2 = threading.Thread(target=racer, args=(store_b, "result-B"))
    t1.start(); t2.start(); t1.join(); t2.join()

    accepted = [o for o in outcomes if o[0] == "accepted"]
    rejected = [o for o in outcomes if o[0] == "rejected"]
    assert len(accepted) == 1
    assert len(rejected) == 1
    events = store_a.read_events("g-authority")
    assert len([e for e in events if e["kind"] == "RESULT_ACCEPTED"]) == 1
    assert len([e for e in events if e["kind"] == "RESULT_REJECTED"]) == 1


# ── Forged holder / attempt / fence across all gated transitions ────────────


def test_complete_rejects_forged_holder_attempt_and_fence(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, NOW, ttl_s=300)
    rh = _sha("r")

    wrong_holder = {**envelope, "lease": {**envelope["lease"], "holder": "hermes:evil"}}
    with pytest.raises(ResultRejection):
        store.complete_node(wrong_holder, result_hash=rh, now=NOW + timedelta(seconds=5))

    wrong_attempt = {**envelope, "attempt_id": "at-forged"}
    with pytest.raises(ResultRejection):
        store.complete_node(wrong_attempt, result_hash=rh, now=NOW + timedelta(seconds=5))

    wrong_fence = {**envelope, "fence": 99}
    with pytest.raises(ResultRejection):
        store.complete_node(wrong_fence, result_hash=rh, now=NOW + timedelta(seconds=5))

    # Each forged attempt appended exactly one typed rejection; none accepted.
    events = store.read_events("g-authority")
    assert len([e for e in events if e["kind"] == "RESULT_REJECTED"]) == 3
    assert len([e for e in events if e["kind"] == "RESULT_ACCEPTED"]) == 0


# ── Exact-expiry boundaries ─────────────────────────────────────────────────


def test_exact_expiry_renewal_and_completion_fail_closed(tmp_path):
    # Renewal strictly before expiry: now >= expires is already expired.
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, NOW, ttl_s=30)
    with pytest.raises(AuthorityError, match="expired"):
        store.renew_node(envelope, ttl_s=30, now=NOW + timedelta(seconds=30))

    # Completion at exactly the lease expiry fails closed.
    store2 = Phase2AuthorityStore(tmp_path / "b.db")
    env2 = _granted(store2, NOW, ttl_s=30)
    with pytest.raises(ResultRejection) as excinfo:
        store2.complete_node(env2, result_hash=_sha("r"), now=NOW + timedelta(seconds=30))
    assert excinfo.value.reason == "lease_expired"

    # One microsecond before expiry still authorizes.
    store3 = Phase2AuthorityStore(tmp_path / "c.db")
    env3 = _granted(store3, NOW, ttl_s=30)
    just_before = NOW + timedelta(seconds=30) - timedelta(microseconds=1)
    rec = store3.complete_node(env3, result_hash=_sha("r"), now=just_before)
    assert rec["fence"] == 1


def test_exact_deadline_renewal_fails_closed(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    node = _node("n-tool", deadline=(NOW + timedelta(seconds=100)).isoformat())
    envelope = _granted(store, NOW, ttl_s=30, node=node)
    # Renewal whose effective expiry lands exactly on the deadline fails closed.
    with pytest.raises(AuthorityError, match="deadline"):
        store.renew_node(envelope, ttl_s=90, now=NOW + timedelta(seconds=10))


# ── Recovery folds every lifecycle event and derived state ──────────────────


def test_recovery_folds_renewal_revoke_and_terminal_events(tmp_path):
    path = tmp_path / "a.db"
    store = Phase2AuthorityStore(path)
    envelope = _granted(store, NOW, ttl_s=30)
    store.renew_node(envelope, ttl_s=100, now=NOW + timedelta(seconds=10))
    store.complete_node(envelope, result_hash=_sha("done"), now=NOW + timedelta(seconds=50))

    recovered = Phase2AuthorityStore(path)
    assert recovered.recover(now=NOW + timedelta(seconds=60)) == {
        "orphaned_reservations_reconciled": 0
    }
    # Terminal state survives restart: derived from RESULT_ACCEPTED.
    authority = recovered.current_authority("g-authority", "n-tool")
    assert authority["completed"] is True
    assert authority["accepted_result_hash"] == _sha("done")
    # Regrant after restart still refused.
    with pytest.raises(AuthorityError, match="terminal|completed"):
        recovered.grant_node(
            "g-authority", "n-tool", holder="h", ttl_s=300, attempt_id="at-9",
            now=NOW + timedelta(seconds=60),
        )


def test_recovery_derives_effective_expiry_from_events_not_envelope(tmp_path):
    path = tmp_path / "a.db"
    store = Phase2AuthorityStore(path)
    envelope = _granted(store, NOW, ttl_s=30)
    store.reserve_budget(envelope, tokens=100, usd=0.1, now=NOW)
    store.renew_node(envelope, ttl_s=300, now=NOW + timedelta(seconds=10))

    # Effective expiry = T+10 + 300 = T+310. At T+200 lease is still live.
    live = Phase2AuthorityStore(path)
    assert live.recover(now=NOW + timedelta(seconds=200))["orphaned_reservations_reconciled"] == 0
    # At T+400 the lease is dead; the orphan is poisoned once.
    dead = Phase2AuthorityStore(path)
    assert dead.recover(now=NOW + timedelta(seconds=400))["orphaned_reservations_reconciled"] == 1


def test_monotonic_fence_preserved_across_events(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "a.db")
    envelope = _granted(store, NOW, ttl_s=30)
    store.revoke_node(envelope, now=NOW + timedelta(seconds=5))
    second = store.grant_node(
        "g-authority", "n-tool", holder="hermes:two", ttl_s=300, attempt_id="at-2",
        now=NOW + timedelta(seconds=5),
    )
    assert second["fence"] == 2
    store.revoke_node(second, now=NOW + timedelta(seconds=6))
    third = store.grant_node(
        "g-authority", "n-tool", holder="hermes:three", ttl_s=300, attempt_id="at-3",
        now=NOW + timedelta(seconds=6),
    )
    assert third["fence"] == 3
    assert store.current_fence("g-authority", "n-tool") == 3
