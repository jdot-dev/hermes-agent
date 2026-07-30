"""Phase 2 control-plane ownership and binding at agent startup."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from agent import phase2_enforcement
from agent.phase2_authority import Phase2AuthorityStore
from run_agent import AIAgent


def _node() -> dict:
    return {
        "node_id": "n-startup",
        "objective": "prove startup-owned phase2 authority",
        "idempotency_key": "a" * 64,
        "execution_surface": "local_tool",
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
        "input_hash": "b" * 64,
    }


def _agent(hermes_home: Path) -> AIAgent:
    tool_defs: list[dict] = []
    with (
        patch("run_agent.get_tool_definitions", return_value=tool_defs),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.agent_init.get_hermes_home", return_value=hermes_home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        return AIAgent(
            api_key="test-key",
            base_url="http://127.0.0.1:1/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_id="session-phase2-startup",
        )


def test_agent_startup_owns_one_profile_scoped_durable_authority_store():
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-phase2-startup-"))

    agent = _agent(hermes_home)

    assert isinstance(agent.phase2_authority, Phase2AuthorityStore)
    assert agent.phase2_authority.db_path == hermes_home / "phase2_authority.db"
    assert agent.phase2_authority is agent.phase2_authority


def test_agent_control_plane_binds_durable_current_node_and_cleans_context():
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-phase2-bind-"))
    agent = _agent(hermes_home)
    now = datetime(2026, 7, 30, tzinfo=timezone.utc)
    agent.phase2_authority.seal_plan(
        {"contract_version": 2, "graph_id": "g-startup", "nodes": [_node()]},
        policy_hash="c" * 64,
        now=now,
    )
    envelope = agent.phase2_authority.grant_node(
        "g-startup",
        "n-startup",
        holder="hermes:session-phase2-startup",
        ttl_s=60,
        attempt_id="at-startup",
        now=now,
    )

    with agent.bind_phase2_node(envelope, now=now):
        assert phase2_enforcement.current_sealed_envelope() == envelope
        assert phase2_enforcement.current_authoritative_fence() == 1

    assert phase2_enforcement.current_sealed_envelope() is None
    assert phase2_enforcement.current_authoritative_fence() is None
