"""Representative isolated flags-on Phase 2 dispatch session."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent import phase2_enforcement
from hermes_cli.config_defaults import DEFAULT_CONFIG
from run_agent import AIAgent


def _agent() -> AIAgent:
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-phase2-flags-on-"))
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    tool_defs = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "read one file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    with (
        patch("run_agent.get_tool_definitions", return_value=tool_defs),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="http://127.0.0.1:1/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    for name, value in (
        ("_cached_system_prompt", "You are helpful."),
        ("_use_prompt_caching", False),
        ("tool_delay", 0),
        ("compression_enabled", False),
        ("save_trajectories", False),
        ("_flush_messages_to_session_db", MagicMock()),
    ):
        setattr(agent, name, value)
    return agent


def _envelope(root: Path) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "envelope_version": 2,
        "graph_id": "g-flags-on",
        "node_id": "n-flags-on",
        "attempt_id": "at-flags-on",
        "idempotency_key": "d" * 64,
        "objective": "representative flags-on read-only dispatch",
        "execution_surface": "local_tool",
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


def _call() -> SimpleNamespace:
    return SimpleNamespace(
        id="tc-flags-on",
        type="function",
        function=SimpleNamespace(
            name="read_file", arguments=json.dumps({"path": str(Path(__file__).resolve())})
        ),
    )


def test_task_envelope_enforcement_default_is_registered_and_off():
    assert DEFAULT_CONFIG["enforcement"]["task_envelopes"] == {"enabled": False}


def _run(agent: AIAgent, messages: list, dispatch: MagicMock) -> None:
    with (
        patch("run_agent.handle_function_call", side_effect=dispatch),
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        agent._execute_tool_calls_sequential(
            SimpleNamespace(content="", tool_calls=[_call()]), messages, "task-flags-on"
        )


def test_flags_on_without_authority_fails_before_dispatch(monkeypatch):
    monkeypatch.setenv("HERMES_ENVELOPE_ENFORCE", "1")
    dispatch = MagicMock(return_value="must-not-run")
    messages: list = []

    _run(_agent(), messages, dispatch)

    dispatch.assert_not_called()
    assert json.loads(messages[0]["content"])["code"] == "missing_sealed_envelope"


def test_flags_on_bound_read_node_executes_and_receipts_exact_v2(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_ENVELOPE_ENFORCE", "1")
    monkeypatch.setattr(
        "agent.action_receipts._load_config",
        lambda: {"observability": {"action_receipts": {"enabled": True}}},
    )
    receipts_path = tmp_path / "action_receipts.db"
    monkeypatch.setattr("agent.action_receipts.default_db_path", lambda: receipts_path)
    dispatch = MagicMock(return_value="read-ok")
    messages: list = []
    envelope = _envelope(Path(__file__).resolve().parent)

    with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
        _run(_agent(), messages, dispatch)

    from agent.action_receipts import ActionReceiptLedger

    recorded = ActionReceiptLedger(receipts_path).read_all()
    dispatch.assert_called_once()
    assert messages[0]["content"] == "read-ok"
    assert len(recorded) == 1
    assert json.loads(recorded[0]["envelope_json"]) == envelope
    assert recorded[0]["action_id"] == "g-flags-on:n-flags-on:at-flags-on"
    assert recorded[0]["execution_surface"] == "local_tool"
