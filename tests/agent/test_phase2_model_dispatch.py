"""Phase 2 model-dispatch authority and usage accounting seams."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from agent import conversation_loop, phase2_enforcement


def _envelope(**overrides) -> dict:
    now = datetime.now(timezone.utc)
    envelope = {
        "envelope_version": 2,
        "graph_id": "g-model-dispatch",
        "node_id": "n-model-dispatch",
        "attempt_id": "at-model-dispatch",
        "idempotency_key": "d" * 64,
        "objective": "execute one model inference",
        "execution_surface": "direct_model",
        "lane": "hermes",
        "roots": ["/tmp"],
        "permissions": {"read": ["/tmp"], "write": [], "spawn": []},
        "network_policy": "none",
        "exec_policy": "none",
        "destructive_policy": "forbid",
        "budgets": {"usd_max": 1.0, "tokens_max": 1000, "wall_clock_s_max": 60},
        "deadline_utc": (now + timedelta(minutes=5)).isoformat(),
        "retry_policy": {"max_attempts": 1, "retry_eligible": False, "backoff_s": [0]},
        "cancellation": {"mode": "cooperative", "grace_s": 30},
        "evidence_policy": {"required_receipt_kinds": ["model_call"], "min_receipts": 1},
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
    envelope.update(overrides)
    return envelope


def _enabled(monkeypatch) -> None:
    monkeypatch.setattr(
        phase2_enforcement,
        "_load_config",
        lambda: {"enforcement": {"task_envelopes": {"enabled": True}}},
    )


def test_model_dispatch_blocks_before_provider_when_authority_missing(monkeypatch):
    _enabled(monkeypatch)
    dispatch = SimpleNamespace(called=False)

    def provider():
        dispatch.called = True
        return "must-not-run"

    try:
        conversation_loop._phase2_authorized_model_dispatch(provider)
    except conversation_loop.Phase2ModelDispatchBlocked as exc:
        assert exc.decision.code == "missing_sealed_envelope"
    else:
        raise AssertionError("missing Phase 2 authority did not block")

    assert dispatch.called is False


def test_model_dispatch_allows_direct_model_node(monkeypatch):
    _enabled(monkeypatch)
    with phase2_enforcement.bind_sealed_envelope(_envelope(), current_fence=1):
        result = conversation_loop._phase2_authorized_model_dispatch(lambda: "ok")
    assert result == "ok"


def test_model_dispatch_blocks_local_tool_node(monkeypatch):
    _enabled(monkeypatch)
    with phase2_enforcement.bind_sealed_envelope(
        _envelope(execution_surface="local_tool"), current_fence=1
    ):
        try:
            conversation_loop._phase2_authorized_model_dispatch(lambda: "must-not-run")
        except conversation_loop.Phase2ModelDispatchBlocked as exc:
            assert exc.decision.code == "execution_surface_mismatch"
        else:
            raise AssertionError("local_tool node authorized model dispatch")
