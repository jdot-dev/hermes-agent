"""Durable Hermes-owned authority for sealed Phase 2 graph nodes."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest

from agent import phase2_enforcement
from agent.phase2_authority import AuthorityError, Phase2AuthorityStore


def _node(node_id: str, surface: str = "local_tool") -> dict:
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
        "deadline_utc": "2030-01-01T00:00:00+00:00",
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


def test_seal_and_grant_produces_full_nested_v2_envelope(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    planner_hash = store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    envelope = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:session-1",
        ttl_s=300,
        attempt_id="at-1",
        now=datetime(2026, 7, 30, tzinfo=timezone.utc),
    )

    assert len(planner_hash) == 64
    assert envelope["envelope_version"] == 2
    assert envelope["graph_id"] == "g-authority"
    assert envelope["node_id"] == "n-tool"
    assert envelope["execution_surface"] == "local_tool"
    assert envelope["planner_hash"] == planner_hash
    assert envelope["policy_hash"] == "d" * 64
    assert envelope["permissions"]["read"] == ["/tmp"]
    assert envelope["lease"]["holder"] == "hermes:session-1"
    assert envelope["fence"] == 1
    assert phase2_enforcement.validate_sealed_envelope(envelope) == []


def test_model_and_tool_work_are_distinct_nodes_and_envelopes(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(
        _plan(_node("n-model", "direct_model"), _node("n-tool", "local_tool")),
        policy_hash="d" * 64,
    )
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    model = store.grant_node(
        "g-authority", "n-model", holder="hermes:model", ttl_s=60, attempt_id="at-model", now=now
    )
    tool = store.grant_node(
        "g-authority", "n-tool", holder="hermes:tool", ttl_s=60, attempt_id="at-tool", now=now
    )

    assert model["node_id"] != tool["node_id"]
    assert model["execution_surface"] == "direct_model"
    assert tool["execution_surface"] == "local_tool"


def test_invalid_or_duplicate_plan_nodes_fail_before_persistence(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    duplicate = _plan(_node("n-tool"), _node("n-tool"))
    with pytest.raises(AuthorityError, match="duplicate node_id"):
        store.seal_plan(duplicate, policy_hash="d" * 64)
    bad_surface = _plan(_node("n-bad", "local_tool_alias"))
    with pytest.raises(AuthorityError, match="execution_surface"):
        store.seal_plan(bad_surface, policy_hash="d" * 64)
    assert store.read_events("g-authority") == []


def test_sealed_plan_is_immutable_and_reseal_is_rejected(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    plan = _plan(_node("n-tool"))
    first = store.seal_plan(plan, policy_hash="d" * 64)
    with pytest.raises(AuthorityError, match="already sealed"):
        store.seal_plan(plan, policy_hash="d" * 64)
    reopened = Phase2AuthorityStore(tmp_path / "authority.db")
    assert reopened.get_planner_hash("g-authority") == first


def test_active_lease_blocks_regrant_and_expired_regrant_increments_fence(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    first = store.grant_node(
        "g-authority", "n-tool", holder="hermes:one", ttl_s=30, attempt_id="at-1", now=now
    )
    with pytest.raises(AuthorityError, match="active lease"):
        store.grant_node(
            "g-authority", "n-tool", holder="hermes:two", ttl_s=30, attempt_id="at-2", now=now
        )

    second = store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:two",
        ttl_s=30,
        attempt_id="at-2",
        now=now + timedelta(seconds=31),
    )
    assert first["fence"] == 1
    assert second["fence"] == 2
    assert store.current_fence("g-authority", "n-tool") == 2
    assert "stale_fence" in store.validate_current(first, now=now + timedelta(seconds=31))
    assert store.validate_current(second, now=now + timedelta(seconds=31)) == []


def test_bind_current_node_sources_authoritative_fence_from_durable_store(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = store.grant_node(
        "g-authority", "n-tool", holder="hermes:one", ttl_s=60, attempt_id="at-1", now=now
    )

    with store.bind_current(envelope, now=now):
        assert phase2_enforcement.current_sealed_envelope() == envelope
        assert phase2_enforcement.current_authoritative_fence() == 1

    assert phase2_enforcement.current_sealed_envelope() is None
    assert phase2_enforcement.current_authoritative_fence() is None


def test_bind_current_rejects_stale_envelope_before_context_is_published(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    stale = store.grant_node(
        "g-authority", "n-tool", holder="hermes:one", ttl_s=1, attempt_id="at-1", now=now
    )
    store.grant_node(
        "g-authority",
        "n-tool",
        holder="hermes:two",
        ttl_s=60,
        attempt_id="at-2",
        now=now + timedelta(seconds=2),
    )

    with pytest.raises(AuthorityError, match="stale_fence"):
        with store.bind_current(stale, now=now + timedelta(seconds=2)):
            raise AssertionError("stale authority must not publish a context")

    assert phase2_enforcement.current_sealed_envelope() is None


def test_concurrent_grant_has_exactly_one_winner(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    barrier = threading.Barrier(8)
    winners: list[dict] = []
    failures: list[str] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        barrier.wait()
        try:
            envelope = store.grant_node(
                "g-authority",
                "n-tool",
                holder=f"hermes:{index}",
                ttl_s=60,
                attempt_id=f"at-{index}",
                now=now,
            )
            with lock:
                winners.append(envelope)
        except AuthorityError as exc:
            with lock:
                failures.append(str(exc))

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(winners) == 1
    assert len(failures) == 7
    assert winners[0]["fence"] == 1


def test_budget_reservation_reconciliation_is_durable_and_fail_closed(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = store.grant_node(
        "g-authority", "n-tool", holder="hermes:one", ttl_s=60, attempt_id="at-1", now=now
    )

    reservation = store.reserve_budget(envelope, tokens=600, usd=0.6, now=now)
    with pytest.raises(AuthorityError, match="budget"):
        store.reserve_budget(envelope, tokens=500, usd=0.5, now=now)
    store.reconcile_budget(reservation, actual_tokens=400, actual_usd=0.4, now=now)

    reopened = Phase2AuthorityStore(tmp_path / "authority.db")
    usage = reopened.budget_usage("g-authority", "n-tool", fence=1)
    assert usage == {
        "reserved_tokens": 0,
        "reserved_usd": 0.0,
        "charged_tokens": 400,
        "charged_usd": 0.4,
        "cost_unknown": False,
    }
    reopened.reserve_budget(envelope, tokens=500, usd=0.5, now=now)


def test_authority_startup_rejects_tampered_event_chain(tmp_path):
    path = tmp_path / "authority.db"
    store = Phase2AuthorityStore(path)
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    store.grant_node(
        "g-authority", "n-tool", holder="hermes:one", ttl_s=60, attempt_id="at-1", now=now
    )
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TRIGGER authority_events_no_update")
        conn.execute("UPDATE authority_events SET payload_json = '{}' WHERE id = 1")

    with pytest.raises(AuthorityError, match="hash chain"):
        Phase2AuthorityStore(path).recover()


def test_recovery_reconciles_orphaned_budget_reservation_exactly_once(tmp_path):
    path = tmp_path / "authority.db"
    store = Phase2AuthorityStore(path)
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = store.grant_node(
        "g-authority", "n-tool", holder="hermes:one", ttl_s=1, attempt_id="at-1", now=now
    )
    reservation = store.reserve_budget(envelope, tokens=100, usd=0.1, now=now)

    recovered = Phase2AuthorityStore(path)
    result = recovered.recover(now=now + timedelta(seconds=2))
    assert result["orphaned_reservations_reconciled"] == 1
    assert recovered.budget_usage("g-authority", "n-tool", fence=1) == {
        "reserved_tokens": 0,
        "reserved_usd": 0.0,
        "charged_tokens": 0,
        "charged_usd": 0.0,
        "cost_unknown": True,
    }
    assert recovered.recover(now=now + timedelta(seconds=3))["orphaned_reservations_reconciled"] == 0
    with pytest.raises(AuthorityError, match="already reconciled"):
        recovered.reconcile_budget(reservation, actual_tokens=1, actual_usd=0.01)


def test_unknown_reconciled_cost_poisons_future_reservations(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    store.seal_plan(_plan(_node("n-tool")), policy_hash="d" * 64)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    envelope = store.grant_node(
        "g-authority", "n-tool", holder="hermes:one", ttl_s=60, attempt_id="at-1", now=now
    )
    reservation = store.reserve_budget(envelope, tokens=100, usd=0.1, now=now)
    store.reconcile_budget(reservation, actual_tokens=80, actual_usd=None, now=now)

    assert store.budget_usage("g-authority", "n-tool", fence=1)["cost_unknown"] is True
    with pytest.raises(AuthorityError, match="unknown"):
        store.reserve_budget(envelope, tokens=1, usd=0.01, now=now)
