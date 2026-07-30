"""Node-scoped binding contracts for sealed Phase 2 task envelopes."""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone

from agent import phase2_enforcement
from run_agent import AIAgent


def _envelope(*, node_id: str = "n-bound-node", surface: str = "local_tool") -> dict:
    now = datetime.now(timezone.utc)
    return {
        "envelope_version": 2,
        "graph_id": "g-bound-graph",
        "node_id": node_id,
        "attempt_id": f"at-{node_id}",
        "idempotency_key": "d" * 64,
        "objective": "bind one sealed graph node to one execution surface",
        "execution_surface": surface,
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


def test_binding_is_node_scoped_and_resets_after_success():
    with phase2_enforcement.bind_sealed_envelope(_envelope(), current_fence=1):
        current = phase2_enforcement.current_sealed_envelope()
        assert current is not None
        assert current["graph_id"] == "g-bound-graph"
        assert current["node_id"] == "n-bound-node"
        assert current["execution_surface"] == "local_tool"
        assert phase2_enforcement.current_authoritative_fence() == 1

    assert phase2_enforcement.current_sealed_envelope() is None
    assert phase2_enforcement.current_authoritative_fence() is None


def test_binding_resets_after_exception():
    try:
        with phase2_enforcement.bind_sealed_envelope(_envelope(), current_fence=1):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert phase2_enforcement.current_sealed_envelope() is None
    assert phase2_enforcement.current_authoritative_fence() is None


def test_whole_conversation_binding_is_not_a_public_runtime_api():
    signature = inspect.signature(AIAgent.run_conversation)
    assert "sealed_envelope" not in signature.parameters
    assert "current_fence" not in signature.parameters


def test_model_and_tool_nodes_require_distinct_envelopes():
    model = _envelope(node_id="n-model", surface="direct_model")
    tool = _envelope(node_id="n-tool", surface="local_tool")

    assert model["node_id"] != tool["node_id"]
    assert model["execution_surface"] == "direct_model"
    assert tool["execution_surface"] == "local_tool"
    assert phase2_enforcement.validate_sealed_envelope(model) == []
    assert phase2_enforcement.validate_sealed_envelope(tool) == []
