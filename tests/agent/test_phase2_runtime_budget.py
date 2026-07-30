"""Runtime budget accounting for bound Phase 2 envelopes."""

from __future__ import annotations

from contextvars import copy_context
from datetime import datetime, timedelta, timezone
import threading

import pytest

from agent import phase2_enforcement

NON_FINITE = [("nan", float("nan")), ("+inf", float("inf")), ("-inf", float("-inf"))]

# Every numeric field ``validate_sealed_envelope`` bounds. A non-finite value in
# any of them makes the corresponding ceiling meaningless, so the envelope must
# be refused as invalid rather than enforced against.
ENVELOPE_NUMERIC_FIELDS = [
    ("budgets", "usd_max"),
    ("budgets", "tokens_max"),
    ("budgets", "wall_clock_s_max"),
    ("cancellation", "grace_s"),
    ("lease", "ttl_s"),
    ("retry_policy", "backoff_s"),
]


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


# ── non-finite numbers must fail closed at every runtime seam (v2 §9) ───────


@pytest.fixture
def enforcing(monkeypatch):
    monkeypatch.setattr(
        phase2_enforcement,
        "_load_config",
        lambda: {"enforcement": {"task_envelopes": {"enabled": True}}},
    )


@pytest.mark.parametrize(
    "group,key",
    ENVELOPE_NUMERIC_FIELDS,
    ids=[f"{g}.{k}" for g, k in ENVELOPE_NUMERIC_FIELDS],
)
@pytest.mark.parametrize("label,bad", NON_FINITE, ids=[label for label, _ in NON_FINITE])
def test_non_finite_envelope_ceilings_block_as_invalid_not_as_driver_errors(
    enforcing, group, key, label, bad
):
    envelope = _envelope()
    envelope[group] = {**envelope[group], key: [0, bad] if key == "backoff_s" else bad}

    assert phase2_enforcement.validate_sealed_envelope(envelope) == [f"{group}.{key}"]

    with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
        decision = phase2_enforcement.evaluate_tool_call(
            "read_file", {"path": "/tmp/inside.txt"}, cwd="/tmp"
        )

    # Typed and field-naming: the caller learns which ceiling is unusable.
    # Unguarded, these values passed validation and reached the int() coercion
    # inside the enforcement path, crashing it untypedly instead of blocking.
    assert decision is not None
    assert decision.code == "invalid_sealed_envelope"
    assert decision.details == (f"{group}.{key}",)


@pytest.mark.parametrize("label,bad", NON_FINITE, ids=[label for label, _ in NON_FINITE])
def test_non_finite_envelope_also_blocks_direct_model_dispatch(enforcing, label, bad):
    envelope = _envelope(execution_surface="direct_model")
    envelope["budgets"] = {**envelope["budgets"], "tokens_max": bad}

    with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
        decision = phase2_enforcement.evaluate_runtime_authority()

    assert decision is not None
    assert decision.code == "invalid_sealed_envelope"
    assert decision.details == ("budgets.tokens_max",)


@pytest.mark.parametrize("label,bad", NON_FINITE, ids=[label for label, _ in NON_FINITE])
def test_non_finite_charge_is_refused_and_leaves_the_meter_enforceable(enforcing, label, bad):
    # A single nan charge used to poison the running total permanently: every
    # later `usd >= usd_max` comparison against nan is False, so the ceiling
    # could never be breached again no matter how much was really spent.
    envelope = _envelope(budgets={"usd_max": 0.5, "tokens_max": 100, "wall_clock_s_max": 10})
    with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
        with pytest.raises(ValueError, match="finite"):
            phase2_enforcement.record_budget_usage(tokens=1, usd=bad)

        # The refusal is total: neither the cost nor the token side is charged.
        usage = phase2_enforcement.current_budget_usage()
        assert usage["usd"] == 0.0
        assert usage["tokens"] == 0
        assert usage["cost_unknown"] is False

        # The meter still works, and the ceiling it guards still fires.
        phase2_enforcement.record_budget_usage(tokens=1, usd=0.5)
        decision = phase2_enforcement.evaluate_tool_call(
            "read_file", {"path": "/tmp/inside.txt"}, cwd="/tmp"
        )

    assert decision is not None
    assert decision.code == "budget_exhausted"
    assert decision.details == ("usd_max",)


def test_non_finite_charge_refusal_does_not_disturb_a_concurrent_meter():
    # The guard raises before taking the lock, so a rejected charge on one
    # worker cannot corrupt or deadlock the shared meter for the others.
    with phase2_enforcement.bind_sealed_envelope(_envelope(), current_fence=1):
        barrier = threading.Barrier(8)
        refused: list[bool] = []
        lock = threading.Lock()

        def charge(index: int) -> None:
            barrier.wait()
            if index % 2:
                try:
                    phase2_enforcement.record_budget_usage(tokens=1, usd=float("nan"))
                except ValueError:
                    with lock:
                        refused.append(True)
            else:
                phase2_enforcement.record_budget_usage(tokens=1, usd=0.01)

        threads = []
        for index in range(8):
            context = copy_context()
            thread = threading.Thread(target=context.run, args=(charge, index))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

        usage = phase2_enforcement.current_budget_usage()

    assert len(refused) == 4
    assert usage["tokens"] == 4
    assert round(float(usage["usd"]), 2) == 0.04
