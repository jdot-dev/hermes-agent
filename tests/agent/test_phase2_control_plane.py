"""Phase 2 control-plane ownership and binding at agent startup."""

from __future__ import annotations

import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from agent import phase2_enforcement
from agent.phase2_authority import AuthorityMigrationRequired, Phase2AuthorityStore
from run_agent import AIAgent
from tests.agent.test_phase3_authority_integrity import _legacy_duplicate_attempt_db


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


def _agent(hermes_home: Path, *, quiet_mode: bool = True) -> AIAgent:
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
            quiet_mode=quiet_mode,
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


# ── a legacy store must quarantine at startup, never silently degrade ───────


@pytest.fixture
def quarantined_home(tmp_path) -> Path:
    """A profile whose authority store carries a pre-index invariant violation."""

    home = tmp_path / "home"
    home.mkdir()
    _legacy_duplicate_attempt_db(home, db_name="phase2_authority.db")
    return home


def test_startup_quarantines_a_legacy_store_and_keeps_enforcement_on(
    quarantined_home, caplog
):
    with (
        patch.object(phase2_enforcement, "is_enabled", return_value=True),
        caplog.at_level(logging.ERROR, logger="run_agent"),
    ):
        agent = _agent(quarantined_home)

        # Startup completes -- the agent must not crash on a legacy profile --
        # but the store is recorded as unusable, with the operator's next step.
        degraded = agent.phase2_authority_degraded
        assert degraded is not None
        assert degraded["reason"] == "authority_migration_required"
        assert degraded["db_path"] == str(quarantined_home / "phase2_authority.db")
        assert degraded["enforcement_enabled"] is True
        assert [v["invariant"] for v in degraded["violations"]] == [
            "duplicate_grant_attempt_id"
        ]
        violation = degraded["violations"][0]
        assert violation["index"] == "one_grant_per_attempt"
        assert violation["violating_groups"] == 1
        assert violation["sample"] == [{"key": "at-dup", "events": 2}]
        assert "at-dup" in degraded["detail"]
        assert "migration_report()" in degraded["detail"]

        # Fail closed, not open: enforcement is still on for this session.
        assert phase2_enforcement.is_enabled() is True

        # The quarantine is actionable in the log, naming path and remedy.
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1
        message = errors[0].getMessage()
        assert "migration" in message.lower()
        assert str(quarantined_home / "phase2_authority.db") in message

        # No fresh authority is issued from a quarantined store.
        with pytest.raises(AuthorityMigrationRequired):
            agent.phase2_authority.grant_node(
                "g-authority",
                "n-one",
                holder="hermes:session-phase2-startup",
                ttl_s=60,
                attempt_id="at-new",
                now=datetime(2026, 7, 30, tzinfo=timezone.utc),
            )

    # The legacy store is bound as-is: no sidestep copy, no parallel database.
    assert agent.phase2_authority.db_path == quarantined_home / "phase2_authority.db"
    assert sorted(p.name for p in quarantined_home.glob("*.db")) == [
        "phase2_authority.db"
    ]

    # Its history stays readable and intact for the operator's migration.
    events = agent.phase2_authority.read_events("g-authority")
    assert [e["kind"] for e in events] == [
        "PLAN_SEALED",
        "LEASE_GRANTED",
        "LEASE_GRANTED",
    ]
    assert agent.phase2_authority.migration_report()["compatible"] is False


def test_quarantine_is_announced_to_a_non_quiet_operator(quarantined_home, capsys):
    with patch.object(phase2_enforcement, "is_enabled", return_value=True):
        agent = _agent(quarantined_home, quiet_mode=False)

    assert agent.phase2_authority_degraded is not None
    out = capsys.readouterr().out
    assert "QUARANTINED" in out
    assert "duplicate_grant_attempt_id" in out


def test_clean_store_starts_undegraded_under_the_same_enforcement_flag(tmp_path):
    home = tmp_path / "home"
    home.mkdir()

    with patch.object(phase2_enforcement, "is_enabled", return_value=True):
        agent = _agent(home)

        # The positive control: same code path, same flag, no false quarantine.
        assert agent.phase2_authority_degraded is None
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
        assert envelope["fence"] == 1
