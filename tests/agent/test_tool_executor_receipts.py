"""Shadow receipt instrumentation at the real tool-execution seams (Phase 0).

These exercise the production dispatch surfaces used by
``tests/run_agent/test_tool_call_incremental_persistence.py``:

    * sequential -> ``run_agent.handle_function_call``
    * concurrent -> ``agent._invoke_tool``

Contract under test:
    G0  flag off  => no database, no envelope, byte-identical tool results
    G1  flag on   => exactly one receipt per *actually executed* tool call,
                     on both paths; a ledger failure never reaches the tool
    G1  blocked / malformed / skipped-before-start calls produce no receipt
"""

import hashlib
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import action_receipts
from run_agent import AIAgent


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
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
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-test-home-"))
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
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


def _tool_call(name="web_search", arguments="{}", call_id="c1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    """Point the ledger at a test-scoped database file."""
    path = tmp_path / "action_receipts.db"
    monkeypatch.setattr(action_receipts, "default_db_path", lambda: path)
    return path


def _set_receipts_enabled(monkeypatch, enabled):
    monkeypatch.setattr(
        action_receipts,
        "_load_config",
        lambda: {"observability": {"action_receipts": {"enabled": enabled}}},
    )


def _receipts(db_path):
    return action_receipts.ActionReceiptLedger(db_path=db_path).read_all()


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


# ---------------------------------------------------------------------------
# G0 — flag off: no database, unchanged results
# ---------------------------------------------------------------------------
def test_disabled_sequential_creates_no_db_and_preserves_results(db_path, monkeypatch):
    _set_receipts_enabled(monkeypatch, False)
    agent = _make_agent()
    messages: list = []

    _run_sequential(agent, [_tool_call(call_id="c1"), _tool_call(call_id="c2")], messages)

    assert not db_path.exists()
    assert [m["tool_call_id"] for m in messages] == ["c1", "c2"]
    assert [m["content"] for m in messages] == ["result-c1", "result-c2"]


def test_disabled_concurrent_creates_no_db_and_preserves_results(db_path, monkeypatch):
    _set_receipts_enabled(monkeypatch, False)
    agent = _make_agent()
    messages: list = []

    _run_concurrent(agent, [_tool_call(call_id="c1"), _tool_call(call_id="c2")], messages)

    assert not db_path.exists()
    assert [m["tool_call_id"] for m in messages] == ["c1", "c2"]
    assert [m["content"] for m in messages] == ["result-c1", "result-c2"]


def test_disabled_never_builds_an_envelope(db_path, monkeypatch):
    _set_receipts_enabled(monkeypatch, False)
    agent = _make_agent()

    with patch("agent.task_envelope.build_shadow_envelope") as build:
        _run_sequential(agent, [_tool_call(call_id="c1")], [])
        _run_concurrent(agent, [_tool_call(call_id="c2")], [])

    assert build.call_count == 0


# ---------------------------------------------------------------------------
# G1 — flag on: exactly one receipt per executed call
# ---------------------------------------------------------------------------
def test_enabled_sequential_writes_exactly_one_receipt_per_call(db_path, monkeypatch):
    _set_receipts_enabled(monkeypatch, True)
    agent = _make_agent()
    messages: list = []

    _run_sequential(agent, [_tool_call(call_id="c1"), _tool_call(call_id="c2")], messages)

    rows = _receipts(db_path)
    assert len(rows) == 2
    assert sorted(r["tool_call_id"] for r in rows) == ["c1", "c2"]
    assert {r["exit_status"] for r in rows} == {"ok"}
    assert {r["execution_surface"] for r in rows} == {"local_tool"}
    assert all(r["envelope_hash"] and len(r["envelope_hash"]) == 64 for r in rows)
    # Tool results are untouched by instrumentation.
    assert [m["content"] for m in messages] == ["result-c1", "result-c2"]


def test_receipt_hashes_raw_tool_output_before_guardrail_annotation(db_path, monkeypatch):
    _set_receipts_enabled(monkeypatch, True)
    agent = _make_agent()
    def annotate(function_name, function_args, result, *, failed, tool_call_id=""):
        return result + "-ANNOTATED"

    monkeypatch.setattr(agent, "_append_guardrail_observation", annotate)
    messages: list = []

    _run_sequential(agent, [_tool_call(call_id="c1")], messages)

    row = _receipts(db_path)[0]
    assert row["output_hash"] == hashlib.sha256(b"result-c1").hexdigest()
    assert messages[0]["content"] == "result-c1-ANNOTATED"


def test_enabled_concurrent_writes_exactly_one_receipt_per_call(db_path, monkeypatch):
    _set_receipts_enabled(monkeypatch, True)
    agent = _make_agent()
    messages: list = []
    calls = [_tool_call(call_id=f"c{i}") for i in range(4)]

    _run_concurrent(agent, calls, messages)

    rows = _receipts(db_path)
    assert len(rows) == 4
    assert sorted(r["tool_call_id"] for r in rows) == ["c0", "c1", "c2", "c3"]
    assert action_receipts.ActionReceiptLedger(db_path=db_path).verify_chain() == []
    assert [m["tool_call_id"] for m in messages] == ["c0", "c1", "c2", "c3"]


def test_enabled_records_error_outcome_for_a_failing_tool(db_path, monkeypatch):
    _set_receipts_enabled(monkeypatch, True)
    agent = _make_agent()

    def _boom(function_name, function_args, effective_task_id, **kwargs):
        raise RuntimeError("tool exploded")

    messages: list = []
    _run_sequential(agent, [_tool_call(call_id="c1")], messages, dispatch=_boom)

    rows = _receipts(db_path)
    assert len(rows) == 1
    assert rows[0]["exit_status"] == "error"
    # The tool's own error text still reaches the model unchanged.
    assert "tool exploded" in messages[0]["content"]


def test_enabled_records_cancelled_outcome_for_started_sequential_call(
    db_path, monkeypatch
):
    _set_receipts_enabled(monkeypatch, True)
    agent = _make_agent()

    def _interrupt(function_name, function_args, effective_task_id, **kwargs):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _run_sequential(agent, [_tool_call(call_id="cancelled")], [], dispatch=_interrupt)

    rows = _receipts(db_path)
    assert len(rows) == 1
    assert rows[0]["tool_call_id"] == "cancelled"
    assert rows[0]["exit_status"] == "cancelled"


def test_started_inline_runtime_call_records_cancelled_receipt(db_path, monkeypatch):
    _set_receipts_enabled(monkeypatch, True)
    agent = _make_agent()
    setattr(agent, "_context_engine_tool_names", {"context_lookup"})
    setattr(agent, "context_compressor", MagicMock())
    agent.context_compressor.handle_tool_call.side_effect = KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _run_sequential(
            agent,
            [_tool_call(name="context_lookup", call_id="inline-cancelled")],
            [],
        )

    rows = _receipts(db_path)
    assert len(rows) == 1
    assert rows[0]["tool_call_id"] == "inline-cancelled"
    assert rows[0]["exit_status"] == "cancelled"


def test_inline_middleware_cancelled_before_execution_produces_no_receipt(
    db_path, monkeypatch
):
    _set_receipts_enabled(monkeypatch, True)
    agent = _make_agent()
    setattr(agent, "_context_engine_tool_names", {"context_lookup"})
    setattr(agent, "context_compressor", MagicMock())

    def cancel_before_execute(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "hermes_cli.middleware.run_tool_execution_middleware",
        cancel_before_execute,
    )

    with pytest.raises(KeyboardInterrupt):
        _run_sequential(
            agent,
            [_tool_call(name="context_lookup", call_id="inline-never-started")],
            [],
        )

    assert _receipts(db_path) == []


# ---------------------------------------------------------------------------
# G1 — ledger failure is swallowed at the seam
# ---------------------------------------------------------------------------
def test_ledger_exception_is_swallowed_and_output_unchanged(db_path, monkeypatch):
    _set_receipts_enabled(monkeypatch, True)
    agent = _make_agent()

    def _explode(*args, **kwargs):
        raise RuntimeError("ledger is on fire")

    monkeypatch.setattr(
        action_receipts.ActionReceiptLedger, "record_receipt", _explode
    )

    seq_messages: list = []
    _run_sequential(agent, [_tool_call(call_id="c1")], seq_messages)
    conc_messages: list = []
    _run_concurrent(agent, [_tool_call(call_id="c2")], conc_messages)

    assert [m["content"] for m in seq_messages] == ["result-c1"]
    assert [m["content"] for m in conc_messages] == ["result-c2"]


# ---------------------------------------------------------------------------
# G1 — non-executed calls produce no receipt
# ---------------------------------------------------------------------------
def test_malformed_arguments_produce_no_receipt(db_path, monkeypatch):
    _set_receipts_enabled(monkeypatch, True)
    agent = _make_agent()
    messages: list = []

    _run_sequential(
        agent,
        [
            _tool_call(call_id="bad", arguments="{not json"),
            _tool_call(call_id="good"),
        ],
        messages,
    )

    rows = _receipts(db_path)
    assert [r["tool_call_id"] for r in rows] == ["good"]


def test_preflight_blocked_calls_produce_no_receipt(db_path, monkeypatch):
    _set_receipts_enabled(monkeypatch, True)
    agent = _make_agent()

    # ``_dispatch_pre_tool_call_hooks`` is the executor's single pre_tool_call
    # invocation point; ``resolve_pre_tool_block`` is a back-compat wrapper that
    # this dispatch path does not go through.
    with patch(
        "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
        return_value=("nope", None),
    ):
        seq_messages: list = []
        _run_sequential(agent, [_tool_call(call_id="c1")], seq_messages)
        conc_messages: list = []
        _run_concurrent(agent, [_tool_call(call_id="c2")], conc_messages)

    assert _receipts(db_path) == []
    # The block result still reaches the model on both paths.
    assert "nope" in seq_messages[0]["content"]
    assert "nope" in conc_messages[0]["content"]


def test_calls_skipped_before_start_produce_no_receipt(db_path, monkeypatch):
    _set_receipts_enabled(monkeypatch, True)
    agent = _make_agent()
    agent._interrupt_requested = True

    seq_messages: list = []
    _run_sequential(agent, [_tool_call(call_id="c1")], seq_messages)
    conc_messages: list = []
    _run_concurrent(agent, [_tool_call(call_id="c2")], conc_messages)

    assert _receipts(db_path) == []
    assert len(seq_messages) == 1 and len(conc_messages) == 1


def test_concurrent_submit_skipped_before_worker_start_produces_no_receipt(
    db_path, monkeypatch
):
    _set_receipts_enabled(monkeypatch, True)
    agent = _make_agent()

    class ShuttingDownExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def submit(self, *args, **kwargs):
            raise RuntimeError("cannot schedule new futures after interpreter shutdown")

        def shutdown(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        "tools.daemon_pool.DaemonThreadPoolExecutor", ShuttingDownExecutor
    )
    messages: list = []

    _run_concurrent(agent, [_tool_call(call_id="never-started")], messages)

    assert _receipts(db_path) == []
    assert "was not started" in messages[0]["content"]


def test_concurrent_worker_abandoned_at_start_order_gate_produces_no_receipt(
    db_path, monkeypatch
):
    """A worker whose thread starts after the batch deadline never dispatches.

    ``_begin_in_order`` returns False for a batch the turn already abandoned and
    the worker raises ``_BatchAbandoned`` *before* ``_authorized_dispatch``, so
    the tool provably did not run. Attesting it would be a false receipt. The
    started-then-wedged case — where the tool did run — is covered by
    ``test_concurrent_started_wedged_call_gets_timeout_receipt_before_worker_returns``.
    """
    _set_receipts_enabled(monkeypatch, True)
    agent = _make_agent()
    release_worker = threading.Event()
    worker_finished = threading.Event()

    def delayed_propagation(fn):
        def delayed(*args, **kwargs):
            release_worker.wait(2)
            try:
                return fn(*args, **kwargs)
            finally:
                worker_finished.set()

        return delayed

    monkeypatch.setattr(
        "agent.tool_executor.propagate_context_to_thread", delayed_propagation
    )
    monkeypatch.setattr("agent.tool_executor._resolve_concurrent_tool_timeout", lambda: 0.02)

    messages: list = []
    _run_concurrent(agent, [_tool_call(call_id="late-start")], messages)
    release_worker.set()
    assert worker_finished.wait(2)

    # Give any late writer a window to prove itself wrong.
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline and not _receipts(db_path):
        time.sleep(0.01)
    assert _receipts(db_path) == []
    assert "timed out" in messages[0]["content"]


def test_concurrent_started_wedged_call_gets_timeout_receipt_before_worker_returns(
    db_path, monkeypatch
):
    _set_receipts_enabled(monkeypatch, True)
    agent = _make_agent()
    worker_started = threading.Event()
    release_worker = threading.Event()

    def wedged(function_name, function_args, effective_task_id, tool_call_id, **kwargs):
        worker_started.set()
        release_worker.wait(2)
        return "late-result"

    monkeypatch.setattr("agent.tool_executor._resolve_concurrent_tool_timeout", lambda: 0.05)
    messages: list = []
    try:
        _run_concurrent(
            agent,
            [_tool_call(call_id="wedged")],
            messages,
            invoke=wedged,
        )
        assert worker_started.is_set()
        rows = _receipts(db_path)
        assert len(rows) == 1
        assert rows[0]["tool_call_id"] == "wedged"
        assert rows[0]["exit_status"] == "timeout"
    finally:
        release_worker.set()

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and getattr(agent, "_tool_worker_threads", set()):
        time.sleep(0.01)
    assert len(_receipts(db_path)) == 1
