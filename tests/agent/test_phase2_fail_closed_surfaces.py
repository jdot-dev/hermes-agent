"""Unsupported Phase 2 execution surfaces and transports fail closed."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent import phase2_enforcement


def _envelope(root: Path, *, surface: str) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "envelope_version": 2,
        "graph_id": "g-fail-closed",
        "node_id": f"n-{surface}",
        "attempt_id": "at-fail-closed",
        "idempotency_key": "d" * 64,
        "objective": "prove unsupported execution surfaces fail closed",
        "execution_surface": surface,
        "lane": "hermes",
        "roots": [str(root)],
        "permissions": {"read": [str(root)], "write": [], "spawn": []},
        "network_policy": "none",
        "exec_policy": "none",
        "destructive_policy": "forbid",
        "budgets": {"usd_max": 1.0, "tokens_max": 1000, "wall_clock_s_max": 60},
        "deadline_utc": (now + timedelta(minutes=5)).isoformat(),
        "retry_policy": {"max_attempts": 1, "retry_eligible": False, "backoff_s": [0]},
        "cancellation": {"mode": "cooperative", "grace_s": 30},
        "evidence_policy": {"required_receipt_kinds": ["rejection"], "min_receipts": 1},
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


def _enable(monkeypatch) -> None:
    monkeypatch.setattr(
        phase2_enforcement,
        "_load_config",
        lambda: {"enforcement": {"task_envelopes": {"enabled": True}}},
    )


@pytest.mark.parametrize("surface", ["combo", "acp_worker", "a2a", "cloud"])
def test_unsupported_surface_cannot_authorize_local_tool_dispatch(monkeypatch, tmp_path, surface):
    _enable(monkeypatch)
    with phase2_enforcement.bind_sealed_envelope(
        _envelope(tmp_path, surface=surface), current_fence=1
    ):
        decision = phase2_enforcement.evaluate_tool_call(
            "read_file", {"path": str(tmp_path / "inside.txt")}, cwd=str(tmp_path)
        )

    assert decision is not None
    assert decision.code == "execution_surface_mismatch"


@pytest.mark.parametrize("surface", ["combo", "acp_worker", "a2a", "cloud"])
def test_unsupported_surface_cannot_authorize_direct_model_dispatch(monkeypatch, tmp_path, surface):
    _enable(monkeypatch)
    with phase2_enforcement.bind_sealed_envelope(
        _envelope(tmp_path, surface=surface), current_fence=1
    ):
        decision = phase2_enforcement.evaluate_runtime_authority()

    assert decision is not None
    assert decision.code == "execution_surface_mismatch"
