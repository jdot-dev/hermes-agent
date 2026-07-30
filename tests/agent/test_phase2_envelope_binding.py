"""Hermes-owned conversation binding for sealed Phase 2 task envelopes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from agent import phase2_enforcement
from run_agent import AIAgent


def _envelope() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "envelope_version": 1,
        "graph_id": "g-bound-turn",
        "node_id": "n-bound-turn",
        "attempt_id": "at-bound-turn",
        "idempotency_key": "d" * 64,
        "objective": "bind one sealed graph node to one Hermes turn",
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


def _agent() -> SimpleNamespace:
    return SimpleNamespace(
        _session_db=None,
        session_id="s-bound-turn",
        _conversation_root_id=lambda: "s-bound-turn",
    )


def test_run_conversation_binds_sealed_envelope_for_the_whole_turn():
    observed: dict = {}

    def fake_turn(*args, **kwargs):
        current = phase2_enforcement.current_sealed_envelope()
        observed["graph_id"] = current["graph_id"] if current else None
        observed["fence"] = phase2_enforcement.current_authoritative_fence()
        return {"final_response": "ok"}

    with patch("agent.conversation_loop.run_conversation", side_effect=fake_turn):
        result = AIAgent.run_conversation(
            _agent(),
            "hello",
            sealed_envelope=_envelope(),
            current_fence=1,
        )

    assert result["final_response"] == "ok"
    assert observed == {"graph_id": "g-bound-turn", "fence": 1}
    assert phase2_enforcement.current_sealed_envelope() is None
    assert phase2_enforcement.current_authoritative_fence() is None


def test_run_conversation_without_envelope_preserves_unbound_context():
    observed: dict = {}

    def fake_turn(*args, **kwargs):
        observed["envelope"] = phase2_enforcement.current_sealed_envelope()
        observed["fence"] = phase2_enforcement.current_authoritative_fence()
        return {"final_response": "ok"}

    with patch("agent.conversation_loop.run_conversation", side_effect=fake_turn):
        result = AIAgent.run_conversation(_agent(), "hello")

    assert result["final_response"] == "ok"
    assert observed == {"envelope": None, "fence": None}
