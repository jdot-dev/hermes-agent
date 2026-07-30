"""Phase 2 enforcement wired at both production tool-execution seams."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent import phase2_enforcement, phase2_idempotency
from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _make_agent():
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-phase2-test-home-"))
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("read_file")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._flush_messages_to_session_db = MagicMock()
    return agent


def _tool_call(call_id: str, *, name: str = "read_file", args: dict | None = None):
    payload = args if args is not None else {"path": "inside.txt"}
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(payload)),
    )


def _run_sequential(agent, tool_calls, messages, dispatch=None):
    def _dispatch(function_name, function_args, effective_task_id, **kwargs):
        return f"result-{kwargs.get('tool_call_id')}"

    with (
        patch("run_agent.handle_function_call", side_effect=dispatch or _dispatch),
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        agent._execute_tool_calls_sequential(
            SimpleNamespace(content="", tool_calls=tool_calls), messages, "task-1"
        )


def _run_concurrent(agent, tool_calls, messages, invoke=None):
    def _invoke(function_name, function_args, effective_task_id, tool_call_id, **kwargs):
        return f"result-{tool_call_id}"

    with (
        patch.object(agent, "_invoke_tool", side_effect=invoke or _invoke),
        patch(
            "agent.tool_executor.maybe_persist_tool_result",
            side_effect=lambda **kwargs: kwargs["content"],
        ),
    ):
        agent._execute_tool_calls_concurrent(
            SimpleNamespace(content="", tool_calls=tool_calls), messages, "task-1"
        )


def _enable(monkeypatch):
    monkeypatch.setattr(
        phase2_enforcement,
        "_load_config",
        lambda: {"enforcement": {"task_envelopes": {"enabled": True}}},
    )


def _envelope(root: Path, **overrides) -> dict:
    now = datetime.now(timezone.utc)
    envelope = {
        "envelope_version": 1,
        "graph_id": "g-phase2-seam",
        "node_id": "n-phase2-seam",
        "attempt_id": "at-phase2-seam",
        "idempotency_key": "d" * 64,
        "objective": "exercise both production tool dispatch seams",
        "execution_surface": "direct_model",
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
    envelope.update(overrides)
    return envelope


def test_sequential_valid_envelope_allows_dispatch(monkeypatch):
    _enable(monkeypatch)
    agent = _make_agent()
    root = Path.cwd().resolve()
    dispatch = MagicMock(return_value="allowed-sequential")
    messages: list = []

    with phase2_enforcement.bind_sealed_envelope(_envelope(root), current_fence=1):
        _run_sequential(agent, [_tool_call(call_id="seq-valid")], messages, dispatch=dispatch)

    dispatch.assert_called_once()
    assert messages[0]["content"] == "allowed-sequential"


def test_concurrent_valid_envelope_propagates_context_and_allows_dispatch(monkeypatch):
    _enable(monkeypatch)
    agent = _make_agent()
    root = Path.cwd().resolve()
    observed: dict = {}

    def _invoke(function_name, function_args, effective_task_id, tool_call_id, **kwargs):
        current = phase2_enforcement.current_sealed_envelope()
        observed["graph_id"] = current["graph_id"] if current else None
        return "allowed-concurrent"

    messages: list = []
    with phase2_enforcement.bind_sealed_envelope(_envelope(root), current_fence=1):
        _run_concurrent(agent, [_tool_call(call_id="conc-valid")], messages, invoke=_invoke)

    assert observed == {"graph_id": "g-phase2-seam"}
    assert messages[0]["content"] == "allowed-concurrent"


def test_sequential_duplicate_mutation_claim_blocks_second_dispatch(monkeypatch, tmp_path):
    _enable(monkeypatch)
    monkeypatch.setattr(
        phase2_idempotency,
        "default_db_path",
        lambda: tmp_path / "phase2-enforcement.db",
    )
    agent = _make_agent()
    root = tmp_path.resolve()
    envelope = _envelope(
        root,
        permissions={"read": [str(root)], "write": [str(root)], "spawn": []},
        verifier_policy={
            "required": True,
            "verifier_lane_must_differ": True,
            "clean_workspace_required": True,
        },
    )
    target = root / "out.txt"
    target.touch(exist_ok=True)
    call = _tool_call(
        "seq-mutation",
        name="write_file",
        args={"path": str(target), "content": "one"},
    )
    dispatch = MagicMock(return_value="mutated")
    messages: list = []

    with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
        _run_sequential(agent, [call, call], messages, dispatch=dispatch)

    dispatch.assert_called_once()
    assert json.loads(messages[1]["content"])["code"] == "duplicate_idempotency_key"


def test_concurrent_duplicate_mutation_claim_has_one_dispatch_winner(monkeypatch, tmp_path):
    _enable(monkeypatch)
    monkeypatch.setattr(
        phase2_idempotency,
        "default_db_path",
        lambda: tmp_path / "phase2-enforcement.db",
    )
    agent = _make_agent()
    root = tmp_path.resolve()
    envelope = _envelope(
        root,
        permissions={"read": [str(root)], "write": [str(root)], "spawn": []},
        verifier_policy={
            "required": True,
            "verifier_lane_must_differ": True,
            "clean_workspace_required": True,
        },
    )
    target = root / "out.txt"
    target.touch(exist_ok=True)
    calls = [
        _tool_call(
            f"conc-mutation-{index}",
            name="write_file",
            args={"path": str(target), "content": "one"},
        )
        for index in range(8)
    ]
    invoke = MagicMock(return_value="mutated")
    messages: list = []

    with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
        _run_concurrent(agent, calls, messages, invoke=invoke)

    assert invoke.call_count == 1
    codes = [json.loads(message["content"]).get("code") for message in messages[1:]]
    assert codes == ["duplicate_idempotency_key"] * 7


def test_sequential_missing_envelope_blocks_before_dispatch(monkeypatch):
    _enable(monkeypatch)
    agent = _make_agent()
    dispatch = MagicMock(return_value="must-not-run")
    messages: list = []

    _run_sequential(agent, [_tool_call(call_id="seq-phase2")], messages, dispatch=dispatch)

    dispatch.assert_not_called()
    payload = json.loads(messages[0]["content"])
    assert payload["status"] == "blocked"
    assert payload["code"] == "missing_sealed_envelope"


def test_concurrent_missing_envelope_blocks_before_dispatch(monkeypatch):
    _enable(monkeypatch)
    agent = _make_agent()
    invoke = MagicMock(return_value="must-not-run")
    messages: list = []

    _run_concurrent(agent, [_tool_call(call_id="conc-phase2")], messages, invoke=invoke)

    invoke.assert_not_called()
    payload = json.loads(messages[0]["content"])
    assert payload["status"] == "blocked"
    assert payload["code"] == "missing_sealed_envelope"


def test_disabled_seam_preserves_dispatch(monkeypatch):
    monkeypatch.setattr(
        phase2_enforcement,
        "_load_config",
        lambda: {"enforcement": {"task_envelopes": {"enabled": False}}},
    )
    agent = _make_agent()
    messages: list = []

    _run_sequential(agent, [_tool_call(call_id="phase2-off")], messages)

    assert messages[0]["content"] == "result-phase2-off"
