from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import model_call_shadow, phase2_enforcement
from agent.action_receipts import ActionReceiptLedger
from run_agent import AIAgent


def test_flag_off_leaves_request_object_untouched_and_does_not_create_db(
    monkeypatch, tmp_path
):
    db_path = tmp_path / "action_receipts.db"
    monkeypatch.setattr(model_call_shadow, "_load_config", lambda: {})
    monkeypatch.setattr(model_call_shadow, "default_db_path", lambda: db_path)
    request = {
        "model": "codex/gpt-5.6-sol-xhigh",
        "messages": [{"role": "user", "content": "private prompt"}],
        "extra_headers": {"x-existing": "keep"},
    }

    prepared, shadow = model_call_shadow.prepare_model_call_shadow(
        request,
        base_url="http://127.0.0.1:20128/v1",
        provider="custom:omniroute-responses",
        model=request["model"],
        api_mode="codex_responses",
        session_id="session-1",
        task_id="task-1",
        api_request_id="turn-1:api:1",
    )

    assert prepared is request
    assert shadow is None
    assert request["extra_headers"] == {"x-existing": "keep"}
    assert not db_path.exists()


def _enable_shadow(monkeypatch):
    monkeypatch.setattr(
        model_call_shadow,
        "_load_config",
        lambda: {"observability": {"envelope_shadow": {"enabled": True}}},
    )


def test_enabled_omniroute_call_preserves_headers_and_injects_action_id(monkeypatch):
    _enable_shadow(monkeypatch)
    request = {
        "model": "codex/gpt-5.6-sol-xhigh",
        "input": [{"role": "user", "content": "private prompt"}],
        "extra_headers": {"x-existing": "keep"},
    }

    prepared, shadow = model_call_shadow.prepare_model_call_shadow(
        request,
        base_url="http://127.0.0.1:20128/v1",
        provider="custom:omniroute-responses",
        model=request["model"],
        api_mode="codex_responses",
        session_id="session-1",
        task_id="task-1",
        api_request_id="turn-1:api:1",
    )

    assert prepared is not request
    assert shadow is not None
    assert prepared["extra_headers"]["x-existing"] == "keep"
    assert prepared["extra_headers"]["x-request-id"] == shadow.envelope.action_id
    assert request["extra_headers"] == {"x-existing": "keep"}
    assert shadow.envelope.execution_surface == "direct_model"
    assert shadow.envelope.lane == "hermes"
    assert "budget_usd_max" in shadow.validation_errors


def test_enabled_omniroute_call_replaces_case_insensitive_request_id_duplicates(monkeypatch):
    _enable_shadow(monkeypatch)
    request = {
        "model": "auto/coding",
        "messages": [],
        "extra_headers": {
            "X-Request-Id": "stale-correlation",
            "x-existing": "keep",
        },
    }

    prepared, shadow = model_call_shadow.prepare_model_call_shadow(
        request,
        base_url="http://127.0.0.1:20128/v1",
        provider="custom:omniroute",
        model=request["model"],
        api_mode="chat_completions",
        session_id="session-1",
        task_id="task-1",
        api_request_id="turn-1:api:duplicate",
    )

    assert shadow is not None
    request_id_headers = [
        (name, value)
        for name, value in prepared["extra_headers"].items()
        if name.lower() == "x-request-id"
    ]
    assert request_id_headers == [("x-request-id", shadow.envelope.action_id)]
    assert prepared["extra_headers"]["x-existing"] == "keep"
    assert request["extra_headers"]["X-Request-Id"] == "stale-correlation"


def test_enabled_non_omniroute_request_is_not_tagged_or_recorded(monkeypatch, tmp_path):
    _enable_shadow(monkeypatch)
    db_path = tmp_path / "action_receipts.db"
    monkeypatch.setattr(model_call_shadow, "default_db_path", lambda: db_path)
    request = {"model": "gpt-test", "messages": [], "extra_headers": {"x-existing": "keep"}}

    prepared, shadow = model_call_shadow.prepare_model_call_shadow(
        request,
        base_url="https://api.openai.com/v1",
        provider="openai",
        model="gpt-test",
        api_mode="chat_completions",
        session_id="session-1",
        task_id="task-1",
        api_request_id="turn-1:api:1",
    )

    assert prepared is request
    assert shadow is None
    assert prepared["extra_headers"] == {"x-existing": "keep"}
    assert not db_path.exists()


@pytest.mark.parametrize("failure_site", ["builder", "validator"])
def test_shadow_setup_failures_leave_dispatch_untagged_and_do_not_create_receipt_db(
    monkeypatch, tmp_path, failure_site
):
    _enable_shadow(monkeypatch)
    db_path = tmp_path / "action_receipts.db"
    monkeypatch.setattr(model_call_shadow, "default_db_path", lambda: db_path)
    request = {
        "model": "auto/coding",
        "messages": [],
        "extra_headers": {"x-existing": "keep"},
    }

    if failure_site == "builder":
        monkeypatch.setattr(
            model_call_shadow,
            "build_shadow_envelope",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("builder failed")),
        )
    else:
        monkeypatch.setattr(
            model_call_shadow,
            "validate_envelope",
            lambda _envelope: (_ for _ in ()).throw(RuntimeError("validator failed")),
        )

    captured = {}
    response = SimpleNamespace(model="test", usage=None)
    result = model_call_shadow.run_model_call_shadow(
        request,
        lambda prepared: captured.update(prepared) or response,
        base_url="http://127.0.0.1:20128/v1",
        provider="custom:omniroute",
        model="auto/coding",
        api_mode="chat_completions",
        session_id="session-1",
        task_id="task-1",
        api_request_id="turn-1:api:setup-failure",
    )

    assert result is response
    assert captured == request
    assert "x-request-id" not in captured["extra_headers"]
    assert not db_path.exists()


def test_success_receipt_matches_injected_action_id_without_persisting_bodies(
    monkeypatch, tmp_path
):
    _enable_shadow(monkeypatch)
    db_path = tmp_path / "action_receipts.db"
    monkeypatch.setattr(model_call_shadow, "default_db_path", lambda: db_path)
    captured = {}
    response = SimpleNamespace(
        model="codex/gpt-5.6-sol-xhigh",
        status="completed",
        usage=SimpleNamespace(input_tokens=11, output_tokens=7, total_tokens=18),
    )

    def execute(prepared):
        captured.update(prepared)
        return response

    result = model_call_shadow.run_model_call_shadow(
        {
            "model": "codex/gpt-5.6-sol-xhigh",
            "input": [{"role": "user", "content": "private prompt SENSITIVE_MARKER"}],
            "api_key": "PRIVATE_API_KEY_MARKER",
        },
        execute,
        base_url="http://127.0.0.1:20128/v1",
        provider="custom:omniroute-responses",
        model="codex/gpt-5.6-sol-xhigh",
        api_mode="codex_responses",
        session_id="session-1",
        task_id="task-1",
        api_request_id="turn-1:api:1",
    )

    assert result is response
    action_id = captured["extra_headers"]["x-request-id"]
    rows = ActionReceiptLedger(db_path).read_all()
    assert len(rows) == 1
    assert rows[0]["kind"] == "model_call"
    assert rows[0]["action_id"] == action_id
    assert rows[0]["tool_call_id"] == "turn-1:api:1"
    assert rows[0]["exit_status"] == "ok"
    assert "budget_usd_max" in rows[0]["args_redacted_summary"]
    db_bytes = db_path.read_bytes()
    assert b"private prompt" not in db_bytes
    assert b"SENSITIVE_MARKER" not in db_bytes
    assert b"PRIVATE_API_KEY_MARKER" not in db_bytes


def _sealed_model_envelope(tmp_path):
    now = datetime.now(timezone.utc)
    return {
        "envelope_version": 2,
        "graph_id": "g-model-receipt",
        "node_id": "n-model-receipt",
        "attempt_id": "at-model-receipt",
        "idempotency_key": "e" * 64,
        "objective": "record one authoritative model call",
        "execution_surface": "direct_model",
        "lane": "hermes",
        "roots": [str(tmp_path)],
        "permissions": {"read": [str(tmp_path)], "write": [], "spawn": []},
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
            "holder": "hermes:test",
            "granted_utc": now.isoformat(),
            "ttl_s": 300,
            "renewable": True,
        },
        "fence": 1,
    }


def test_flags_on_bound_direct_model_call_records_exact_sealed_v2(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_ENVELOPE_ENFORCE", "1")
    db_path = tmp_path / "action_receipts.db"
    monkeypatch.setattr(model_call_shadow, "default_db_path", lambda: db_path)
    monkeypatch.setattr(
        model_call_shadow,
        "_load_config",
        lambda: {"observability": {"envelope_shadow": {"enabled": True}}},
    )
    envelope = _sealed_model_envelope(tmp_path)
    response = SimpleNamespace(
        model="codex/gpt-5.6-sol-xhigh",
        status="completed",
        usage=SimpleNamespace(input_tokens=11, output_tokens=7, total_tokens=18),
    )

    with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
        result = model_call_shadow.run_model_call_shadow(
            {"model": "codex/gpt-5.6-sol-xhigh", "input": []},
            lambda _prepared: response,
            base_url="http://127.0.0.1:20128/v1",
            provider="custom:omniroute-responses",
            model="codex/gpt-5.6-sol-xhigh",
            api_mode="codex_responses",
            session_id="session-1",
            task_id="task-1",
            api_request_id="turn-1:api:sealed-v2",
        )

    rows = ActionReceiptLedger(db_path).read_all()
    assert result is response
    assert len(rows) == 1
    assert rows[0]["kind"] == "model_call"
    assert rows[0]["action_id"] == "g-model-receipt:n-model-receipt:at-model-receipt"
    assert rows[0]["execution_surface"] == "direct_model"
    assert json.loads(rows[0]["envelope_json"]) == envelope


def test_provider_error_is_recorded_and_original_exception_is_raised(monkeypatch, tmp_path):
    _enable_shadow(monkeypatch)
    db_path = tmp_path / "action_receipts.db"
    monkeypatch.setattr(model_call_shadow, "default_db_path", lambda: db_path)

    def execute(_prepared):
        raise RuntimeError("provider exploded")

    with pytest.raises(RuntimeError, match="provider exploded"):
        model_call_shadow.run_model_call_shadow(
            {"model": "auto/coding", "messages": []},
            execute,
            base_url="http://localhost:20128/v1",
            provider="custom:omniroute",
            model="auto/coding",
            api_mode="chat_completions",
            session_id="session-1",
            task_id="task-1",
            api_request_id="turn-1:api:2",
        )

    rows = ActionReceiptLedger(db_path).read_all()
    assert len(rows) == 1
    assert rows[0]["exit_status"] == "error"
    assert rows[0]["execution_surface"] == "combo"
    assert rows[0]["tool_call_id"] == "turn-1:api:2"
    assert b"provider exploded" not in db_path.read_bytes()


@pytest.mark.parametrize("exc", [InterruptedError("stopped"), KeyboardInterrupt()])
def test_started_cancellation_is_recorded_as_cancelled_and_reraised(
    monkeypatch, tmp_path, exc
):
    _enable_shadow(monkeypatch)
    db_path = tmp_path / "action_receipts.db"
    monkeypatch.setattr(model_call_shadow, "default_db_path", lambda: db_path)

    def execute(_prepared):
        raise exc

    with pytest.raises(type(exc)):
        model_call_shadow.run_model_call_shadow(
            {"model": "auto/coding", "messages": []},
            execute,
            base_url="http://localhost:20128/v1",
            provider="custom:omniroute",
            model="auto/coding",
            api_mode="chat_completions",
            session_id="session-1",
            task_id="task-1",
            api_request_id="turn-1:api:cancelled",
        )

    rows = ActionReceiptLedger(db_path).read_all()
    assert len(rows) == 1
    assert rows[0]["exit_status"] == "cancelled"


def test_ledger_failure_does_not_change_model_result(monkeypatch):
    _enable_shadow(monkeypatch)
    response = SimpleNamespace(model="test", usage=None)

    def fail_receipt(*_args, **_kwargs):
        raise sqlite3.OperationalError("disk unavailable")

    monkeypatch.setattr(ActionReceiptLedger, "record_receipt", fail_receipt)

    result = model_call_shadow.run_model_call_shadow(
        {"model": "test", "messages": []},
        lambda _prepared: response,
        base_url="http://127.0.0.1:20128/v1",
        provider="custom:omniroute",
        model="test",
        api_mode="chat_completions",
        session_id="session-1",
        task_id="task-1",
        api_request_id="turn-1:api:3",
    )

    assert result is response


def test_conversation_loop_injects_action_id_at_real_dispatch_seam(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    _enable_shadow(monkeypatch)
    db_path = tmp_path / "action_receipts.db"
    monkeypatch.setattr(model_call_shadow, "default_db_path", lambda: db_path)
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            session_id="shadow-dispatch-test",
            api_key="test-key",
            base_url="http://127.0.0.1:20128/v1",
            provider="custom:omniroute",
            model="auto/coding",
            api_mode="chat_completions",
            max_iterations=1,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._cached_system_prompt = "stable test prompt"
    agent._session_db = None
    agent._session_json_enabled = False
    agent.save_trajectories = False
    agent.compression_enabled = False
    agent._cleanup_task_resources = lambda *_a, **_kw: None
    agent._save_trajectory = lambda *_a, **_kw: None
    captured = {}

    def model_call(api_kwargs):
        captured.update(api_kwargs)
        message = SimpleNamespace(content="done", tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            model="codex/gpt-5.6-sol-xhigh",
            usage=None,
        )

    agent._interruptible_api_call = model_call
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        result = agent.run_conversation("private user prompt", task_id="task-1")

    assert result["final_response"] == "done"
    action_id = captured["extra_headers"]["x-request-id"]
    rows = ActionReceiptLedger(db_path).read_all()
    assert len(rows) == 1
    assert rows[0]["action_id"] == action_id
    assert rows[0]["kind"] == "model_call"
