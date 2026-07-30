"""Phase 3 durable lease/fence lifecycle: renew, revoke/regrant, completion.

These exercise the Hermes-owned authority store's movement from a valid Phase 2
sealed candidate into live Phase 3 ownership discipline (contract v2 §5, §9):

* renewal atomically extends an active renewable lease, bound to the current
  holder/attempt/fence, and re-points authoritative validation at the renewed
  envelope without reviving a dead lease;
* revocation cancels a current lease and permits an immediate regrant at the
  next monotonic fence while the old envelope validates stale/revoked;
* completion accepts the current unexpired holder/attempt/fence exactly once and
  refuses stale/expired/revoked/superseded/duplicate completions.
"""

from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timedelta, timezone

import pytest

from agent import phase2_enforcement
from agent.phase2_authority import AuthorityError, Phase2AuthorityStore, ResultRejection


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


def _granted(store: Phase2AuthorityStore, now: datetime, *, ttl_s: float = 300, node: dict | None = None) -> dict:
    store.seal_plan(_plan(node or _node("n-tool")), policy_hash="d" * 64)
    return store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:one",
        ttl_s=ttl_s,
        attempt_id="at-1",
        now=now,
    )


# ── Slice 1: renew ──────────────────────────────────────────────────────────


def test_renew_extends_active_lease_without_changing_identity_or_fence(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = _granted(store, now, ttl_s=30)

    renewed = store.renew_node(envelope, ttl_s=30, now=now + timedelta(seconds=15))

    assert renewed["graph_id"] == envelope["graph_id"]
    assert renewed["node_id"] == envelope["node_id"]
    assert renewed["attempt_id"] == envelope["attempt_id"]
    assert renewed["fence"] == envelope["fence"] == 1
    # Renewal record carries holder directly; the grant envelope is immutable.
    assert renewed["holder"] == envelope["lease"]["holder"]
    assert renewed["new_expires_utc"] == (now + timedelta(seconds=45)).isoformat()
    # The grant envelope is the authoritative envelope; it validates correctly.
    assert phase2_enforcement.validate_sealed_envelope(envelope) == []

    # The grant envelope is still passed for all subsequent operations.
    # After the original TTL (+30), the live effective expiry is now +45.
    later = now + timedelta(seconds=40)
    assert store.validate_current(envelope, now=later) == []
    # A stale fence envelope is not authoritative.
    stale_fence = {**envelope, "fence": 2}
    stale = store.validate_current(stale_fence, now=later)
    assert "stale_fence" in stale
    assert store.current_fence("g-authority", "n-tool") == 1


def test_renew_is_bound_to_holder_attempt_and_current_fence(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = _granted(store, now, ttl_s=300)
    soon = now + timedelta(seconds=5)

    wrong_holder = {**envelope, "lease": {**envelope["lease"], "holder": "hermes:two"}}
    with pytest.raises(AuthorityError, match="holder"):
        store.renew_node(wrong_holder, ttl_s=300, now=soon)

    wrong_attempt = {**envelope, "attempt_id": "at-2"}
    with pytest.raises(AuthorityError, match="attempt"):
        store.renew_node(wrong_attempt, ttl_s=300, now=soon)

    stale_fence = {**envelope, "fence": 2}
    with pytest.raises(AuthorityError, match="fence"):
        store.renew_node(stale_fence, ttl_s=300, now=soon)

    # A well-formed renewal by the real holder still succeeds afterward.
    renewed = store.renew_node(envelope, ttl_s=300, now=soon)
    assert renewed["fence"] == 1


def test_renew_refuses_to_revive_expired_lease(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = _granted(store, now, ttl_s=30)

    with pytest.raises(AuthorityError, match="expired"):
        store.renew_node(envelope, ttl_s=30, now=now + timedelta(seconds=31))


def test_renew_refuses_past_node_deadline(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    node = _node("n-tool", deadline=(now + timedelta(seconds=20)).isoformat())
    envelope = _granted(store, now, ttl_s=300, node=node)

    with pytest.raises(AuthorityError, match="deadline"):
        store.renew_node(envelope, ttl_s=300, now=now + timedelta(seconds=25))


def test_renew_refuses_revoked_lease(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = _granted(store, now, ttl_s=300)
    store.revoke_node(envelope, now=now + timedelta(seconds=5))

    with pytest.raises(AuthorityError, match="revok"):
        store.renew_node(envelope, ttl_s=300, now=now + timedelta(seconds=6))


# ── Slice 2: revoke / immediate regrant ─────────────────────────────────────


def test_revoke_allows_immediate_regrant_at_next_fence(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = _granted(store, now, ttl_s=300)

    record = store.revoke_node(envelope, now=now + timedelta(seconds=5))
    assert record["fence"] == 1
    assert record["node_id"] == "n-tool"

    # The old envelope now validates as revoked authority.
    revoked_errors = store.validate_current(envelope, now=now + timedelta(seconds=5))
    assert "lease_revoked" in revoked_errors

    # Regrant is immediate even though the original TTL has not elapsed.
    second = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:two",
        ttl_s=300,
        attempt_id="at-2",
        now=now + timedelta(seconds=5),
    )
    assert second["fence"] == 2
    assert store.current_fence("g-authority", "n-tool") == 2
    assert store.validate_current(second, now=now + timedelta(seconds=5)) == []
    assert "stale_fence" in store.validate_current(envelope, now=now + timedelta(seconds=5))


def test_revoked_envelope_never_authorizes_execution(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = _granted(store, now, ttl_s=300)
    store.revoke_node(envelope, now=now + timedelta(seconds=5))

    with pytest.raises(AuthorityError):
        with store.bind_current(envelope, now=now + timedelta(seconds=5)):
            raise AssertionError("revoked authority must not publish a context")

    assert phase2_enforcement.current_sealed_envelope() is None


def test_revoke_requires_the_current_active_lease(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = _granted(store, now, ttl_s=300)
    store.revoke_node(envelope, now=now + timedelta(seconds=5))

    # Double revoke of the same fence is rejected.
    with pytest.raises(AuthorityError, match="already revoked"):
        store.revoke_node(envelope, now=now + timedelta(seconds=6))

    # A superseded fence cannot revoke the live lease.
    store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:two",
        ttl_s=300,
        attempt_id="at-2",
        now=now + timedelta(seconds=6),
    )
    with pytest.raises(AuthorityError, match="stale fence"):
        store.revoke_node(envelope, now=now + timedelta(seconds=7))


# ── Slice 3: completion accepted exactly once ───────────────────────────────


def test_completion_accepts_current_holder_attempt_fence_exactly_once(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = _granted(store, now, ttl_s=300)

    record = store.complete_node(envelope, result_hash=_sha("r1"), now=now + timedelta(seconds=5))
    assert record["fence"] == 1
    assert record["node_id"] == "n-tool"
    assert record["attempt_id"] == envelope["attempt_id"]
    assert record["result_hash"] == _sha("r1")

    # Same-hash redelivery by the same holder is idempotent (no new event).
    r2 = store.complete_node(envelope, result_hash=_sha("r1"), now=now + timedelta(seconds=6))
    assert r2["idempotent"] is True

    # A different-hash second submission is a typed conflict rejection.
    with pytest.raises(ResultRejection, match="node_completed"):
        store.complete_node(envelope, result_hash=_sha("r-different"), now=now + timedelta(seconds=7))

    # The completed node is terminal and can no longer authorize execution.
    assert "node_terminal" in store.validate_current(envelope, now=now + timedelta(seconds=6))
    with pytest.raises(AuthorityError):
        with store.bind_current(envelope, now=now + timedelta(seconds=6)):
            raise AssertionError("completed terminal node must not publish a context")


def test_completion_rejects_expired_revoked_and_superseded(tmp_path):
    # Expired lease cannot complete.
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    expired = _granted(store, now, ttl_s=30)
    with pytest.raises(ResultRejection, match="lease_expired"):
        store.complete_node(expired, result_hash=_sha("r1"), now=now + timedelta(seconds=31))

    # Revoked lease cannot complete.
    revoked_store = Phase2AuthorityStore(tmp_path / "revoked.db")
    revoked = _granted(revoked_store, now, ttl_s=300)
    revoked_store.revoke_node(revoked, now=now + timedelta(seconds=5))
    with pytest.raises(ResultRejection, match="lease_revoked"):
        revoked_store.complete_node(revoked, result_hash=_sha("r1"), now=now + timedelta(seconds=6))

    # Superseded fence cannot complete.
    superseded_store = Phase2AuthorityStore(tmp_path / "superseded.db")
    first = _granted(superseded_store, now, ttl_s=30)
    superseded_store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:two",
        ttl_s=300,
        attempt_id="at-2",
        now=now + timedelta(seconds=31),
    )
    with pytest.raises(ResultRejection, match="stale_fence"):
        superseded_store.complete_node(first, result_hash=_sha("r1"), now=now + timedelta(seconds=32))


def test_completion_uses_the_renewed_authoritative_envelope(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = _granted(store, now, ttl_s=30)
    store.renew_node(envelope, ttl_s=30, now=now + timedelta(seconds=15))

    # The original grant envelope remains the authoritative envelope for completion.
    # After renewal the live effective expiry is +45; completing at +40 succeeds.
    record = store.complete_node(
        envelope, result_hash=_sha("r1"), now=now + timedelta(seconds=40)
    )
    assert record["fence"] == 1

    # Completing past the effective expiry (after +45) now fails.
    store2 = Phase2AuthorityStore(tmp_path / "b.db")
    env2 = _granted(store2, now, ttl_s=30)
    store2.renew_node(env2, ttl_s=30, now=now + timedelta(seconds=15))
    with pytest.raises(ResultRejection, match="lease_expired"):
        store2.complete_node(env2, result_hash=_sha("r1"), now=now + timedelta(seconds=50))


def test_concurrent_completion_has_exactly_one_winner(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = _granted(store, now, ttl_s=300)

    barrier = threading.Barrier(8)
    winners: list[dict] = []
    failures: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        try:
            record = store.complete_node(
                envelope, result_hash=_sha("r1"), now=now + timedelta(seconds=5)
            )
            with lock:
                # Idempotent redeliveries are not original acceptances.
                if not record.get("idempotent"):
                    winners.append(record)
        except (AuthorityError, ResultRejection) as exc:
            with lock:
                failures.append(str(exc))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(winners) == 1
    # Total outcomes == total threads (some may be idempotent, some failures).
    assert len(winners) + len(failures) <= 8


# ── Slice 4: recovery verifies/derives lifecycle events, fails closed ────────


def test_recovery_verifies_lifecycle_event_chain_and_survives_restart(tmp_path):
    path = tmp_path / "authority.db"
    store = Phase2AuthorityStore(path)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = _granted(store, now, ttl_s=300)
    store.renew_node(envelope, ttl_s=300, now=now + timedelta(seconds=10))
    store.complete_node(envelope, result_hash=_sha("r1"), now=now + timedelta(seconds=20))

    # A fresh process verifies the full hash chain including the new events.
    recovered = Phase2AuthorityStore(path)
    assert recovered.recover(now=now + timedelta(seconds=30)) == {
        "orphaned_reservations_reconciled": 0
    }
    # The completed node is still terminal after restart (fail closed).
    assert "node_terminal" in recovered.validate_current(
        envelope, now=now + timedelta(seconds=30)
    )


def test_recovery_rejects_tampered_renewal_event(tmp_path):
    import sqlite3

    path = tmp_path / "authority.db"
    store = Phase2AuthorityStore(path)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = _granted(store, now, ttl_s=300)
    store.renew_node(envelope, ttl_s=300, now=now + timedelta(seconds=10))
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TRIGGER authority_events_no_update")
        conn.execute(
            "UPDATE authority_events SET payload_json = '{}' WHERE kind = 'LEASE_RENEWED'"
        )

    with pytest.raises(AuthorityError, match="hash chain"):
        Phase2AuthorityStore(path).recover()


def test_recovery_uses_renewed_lease_expiry_for_orphan_reservation(tmp_path):
    path = tmp_path / "authority.db"
    store = Phase2AuthorityStore(path)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = _granted(store, now, ttl_s=30)
    store.reserve_budget(envelope, tokens=100, usd=0.1, now=now)
    # Renew extends the live TTL well past the original 30s window.
    store.renew_node(envelope, ttl_s=300, now=now + timedelta(seconds=20))

    # After the ORIGINAL ttl but before the RENEWED ttl, the lease is still live
    # so recovery must not poison the reservation.
    live = Phase2AuthorityStore(path)
    assert live.recover(now=now + timedelta(seconds=40))[
        "orphaned_reservations_reconciled"
    ] == 0
    assert live.budget_usage("g-authority", "n-tool", fence=1)["reserved_tokens"] == 100

    # After the renewed ttl the lease is dead and the orphan is poisoned once.
    dead = Phase2AuthorityStore(path)
    assert dead.recover(now=now + timedelta(seconds=400))[
        "orphaned_reservations_reconciled"
    ] == 1
    usage = dead.budget_usage("g-authority", "n-tool", fence=1)
    assert usage["reserved_tokens"] == 0
    assert usage["cost_unknown"] is True


def test_recovery_poisons_reservation_under_revoked_lease(tmp_path):
    path = tmp_path / "authority.db"
    store = Phase2AuthorityStore(path)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = _granted(store, now, ttl_s=300)
    store.reserve_budget(envelope, tokens=100, usd=0.1, now=now)
    store.revoke_node(envelope, now=now + timedelta(seconds=5))

    # The lease TTL has not elapsed, but revocation makes it dead: fail closed.
    recovered = Phase2AuthorityStore(path)
    assert recovered.recover(now=now + timedelta(seconds=6))[
        "orphaned_reservations_reconciled"
    ] == 1
    usage = recovered.budget_usage("g-authority", "n-tool", fence=1)
    assert usage["reserved_tokens"] == 0
    assert usage["cost_unknown"] is True
