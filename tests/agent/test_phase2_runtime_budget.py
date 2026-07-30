"""Runtime budget accounting for bound Phase 2 envelopes."""

from __future__ import annotations

from contextvars import copy_context
from datetime import datetime, timedelta, timezone
import threading

from agent import phase2_enforcement


def _envelope(**overrides) -> dict:
    now = datetime.now(timezone.utc)
    base = {
        "envelope_version": 2,
        "graph_id": "g-budget",
        "node_id": "n-budget",
        "attempt_id": "at-budget",
        "idempotency_key": "d" * 64,
        "objective": "enforce node budgets",
        "execution_surface": "local_tool",
        "lane": "hermes",
        "roots": ["/tmp"],
        "permissions": {"read": ["/tmp"], "write": [], "spawn": []},
        "network_policy": "none",
        "exec_policy": "none",
        "destructive_policy": "forbid",
        "budgets": {"usd_max": 1.0, "tokens_max": 100, "wall_clock_s_max": 10},
        "deadline_utc": (now + timedelta(minutes=5)).isoformat(),
        "retry_policy": {"max_attempts": 1, "retry_eligible": False, "backoff_s": [0]},
        "cancellation": {"mode": "cooperative", "grace_s": 30},
        "evidence_policy": {"required_receipt_kinds": ["tool_exec"], "min_receipts": 1},
        "verifier_policy": {
            "required": False,
            "verifier_lane_must_differ": False,
            "clean_workspace_required": False,
        },
        "planner_hash": "a" * 64,
        "policy_hash": "b" * 64,
        "input_hash": "c" * 64,
        "lease": {
            "holder": "hermes",
            "granted_utc": now.isoformat(),
            "ttl_s": 300,
            "renewable": True,
        },
        "fence": 1,
    }
    base.update(overrides)
    return base


def test_bound_runtime_starts_with_zero_usage():
    with phase2_enforcement.bind_sealed_envelope(_envelope(), current_fence=1):
        usage = phase2_enforcement.current_budget_usage()
    assert usage is not None
    assert usage["usd"] == 0.0
    assert usage["tokens"] == 0
    assert 0.0 <= usage["wall_clock_s"] < 0.1


def test_recorded_token_and_cost_usage_blocks_when_ceiling_is_reached(monkeypatch):
    monkeypatch.setattr(
        phase2_enforcement,
        "_load_config",
        lambda: {"enforcement": {"task_envelopes": {"enabled": True}}},
    )
    envelope = _envelope(budgets={"usd_max": 0.5, "tokens_max": 10, "wall_clock_s_max": 10})
    with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
        phase2_enforcement.record_budget_usage(tokens=10, usd=0.5)
        decision = phase2_enforcement.evaluate_tool_call(
            "read_file", {"path": "/tmp/inside.txt"}, cwd="/tmp"
        )
    assert decision is not None
    assert decision.code == "budget_exhausted"
    assert "tokens_max" in decision.details
    assert "usd_max" in decision.details


def test_wall_clock_budget_is_measured_from_binding_time(monkeypatch):
    monkeypatch.setattr(
        phase2_enforcement,
        "_load_config",
        lambda: {"enforcement": {"task_envelopes": {"enabled": True}}},
    )
    clock = iter((100.0, 111.0))
    monkeypatch.setattr(phase2_enforcement.time, "monotonic", lambda: next(clock))
    with phase2_enforcement.bind_sealed_envelope(_envelope(), current_fence=1):
        decision = phase2_enforcement.evaluate_tool_call(
            "read_file", {"path": "/tmp/inside.txt"}, cwd="/tmp"
        )
    assert decision is not None
    assert decision.code == "budget_exhausted"
    assert decision.details == ("wall_clock_s_max",)


def test_unknown_cost_blocks_further_execution_when_usd_budget_is_authoritative(monkeypatch):
    monkeypatch.setattr(
        phase2_enforcement,
        "_load_config",
        lambda: {"enforcement": {"task_envelopes": {"enabled": True}}},
    )
    with phase2_enforcement.bind_sealed_envelope(
        _envelope(execution_surface="direct_model"), current_fence=1
    ):
        phase2_enforcement.record_budget_usage(tokens=1, usd=None)
        decision = phase2_enforcement.evaluate_runtime_authority()
    assert decision is not None
    assert decision.code == "budget_cost_unknown"


def test_runtime_authority_blocks_missing_envelope_before_model_dispatch(monkeypatch):
    monkeypatch.setattr(
        phase2_enforcement,
        "_load_config",
        lambda: {"enforcement": {"task_envelopes": {"enabled": True}}},
    )
    decision = phase2_enforcement.evaluate_runtime_authority()
    assert decision is not None
    assert decision.code == "missing_sealed_envelope"


def test_concurrent_worker_contexts_charge_one_shared_budget_meter():
    with phase2_enforcement.bind_sealed_envelope(_envelope(), current_fence=1):
        barrier = threading.Barrier(8)
        threads = []

        def charge() -> None:
            barrier.wait()
            phase2_enforcement.record_budget_usage(tokens=1, usd=0.01)

        for _ in range(8):
            context = copy_context()
            thread = threading.Thread(target=context.run, args=(charge,))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

        usage = phase2_enforcement.current_budget_usage()

    assert usage is not None
    assert usage["tokens"] == 8
    assert round(float(usage["usd"]), 2) == 0.08


def test_budget_context_is_reset_after_scope():
    with phase2_enforcement.bind_sealed_envelope(_envelope(), current_fence=1):
        phase2_enforcement.record_budget_usage(tokens=1, usd=0.1)
        assert phase2_enforcement.current_budget_usage()["tokens"] == 1
    assert phase2_enforcement.current_budget_usage() is None
