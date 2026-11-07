"""Native executor observers see raw tool results before presentation transforms."""
import os
from pathlib import Path
import textwrap

import pytest

from hermes_cli.plugins import get_plugin_manager
from tests.agent.test_run_agent import (
    _mock_assistant_msg,
    _mock_tool_call,
)


pytest_plugins = ["tests.agent.test_run_agent"]


@pytest.mark.parametrize("mode", ["sequential", "concurrent"])
def test_executor_observes_raw_result_before_transform(agent, monkeypatch, mode):
    home = Path(os.environ["HERMES_HOME"])
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(home / "empty"))
    plugin = home / "plugins" / "observed_result"
    plugin.mkdir(parents=True)
    (home / "config.yaml").write_text("plugins:\n  enabled: [observed_result]\n")
    (plugin / "plugin.yaml").write_text("name: observed_result\nversion: 1.0.0\n")
    (plugin / "__init__.py").write_text(textwrap.dedent("""
        events = []
        def register(ctx):
            def post(**kw):
                events.append(("post", kw["tool_call_id"], kw["result"]))
            def transform(**kw):
                events.append(("transform", kw["tool_call_id"], kw["result"]))
                return "DISPLAY " + kw["result"]
            ctx.register_hook("post_tool_call", post)
            ctx.register_hook("transform_tool_result", transform)
            ctx.register_tool(
                name="observed_tool", toolset="observed_result",
                schema={"name": "observed_tool", "description": "fixture",
                        "parameters": {"type": "object", "properties": {}}},
                handler=lambda args, **kw: '{"task":{"id":"real-task"}}',
            )
    """))
    manager = get_plugin_manager()
    manager.discover_and_load(force=True)
    try:
        agent._memory_manager = None
        agent.valid_tool_names.add("observed_tool")
        calls = [_mock_tool_call(name="observed_tool", arguments="{}", call_id=call_id)
                 for call_id in ("one", "two")]
        messages = []
        execute = getattr(agent, "_execute_tool_calls_" + mode)
        execute(_mock_assistant_msg(content="", tool_calls=calls), messages, "task")
        events = manager._plugins["observed_result"].module.events
        raw = '{"task":{"id":"real-task"}}'
        for call_id in ("one", "two"):
            assert [(kind, result) for kind, cid, result in events if cid == call_id] == [
                ("post", raw), ("transform", raw)]
        assert [m["content"] for m in messages if m.get("role") == "tool"] == [
            "DISPLAY " + raw, "DISPLAY " + raw]
    finally:
        manager.unload()
