"""Dual-kind providers keep tool handlers paired with their owning hooks."""
import textwrap
from types import SimpleNamespace

import pytest

from agent.inline_tool_executors import resolve_invoke_tool_executor
from agent.memory_manager import MemoryManager
from hermes_cli.plugins import get_plugin_manager
from plugins.memory import load_memory_provider
from tools.registry import registry


@pytest.mark.parametrize("order", ["plugin-first", "memory-first", "memory-only"])
def test_memory_activation_preserves_tool_hook_identity(tmp_path, monkeypatch, order):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(tmp_path / "empty"))
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
    monkeypatch.chdir(tmp_path)
    enabled = "[]" if order == "memory-only" else "[paired]"
    (tmp_path / "config.yaml").write_text(
        f"plugins:\n  enabled: {enabled}\nmemory:\n  provider: paired\n"
    )
    plugin = tmp_path / "plugins" / "paired"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: paired\nversion: 1.0.0\nkind: standalone\n")
    (plugin / "__init__.py").write_text(textwrap.dedent("""
        from agent.memory_provider import MemoryProvider

        class Provider(MemoryProvider):
            name = "paired"
            def is_available(self): return True
            def initialize(self, session_id, **kwargs): pass
            def get_tool_schemas(self):
                return [{"name": name, "description": name, "parameters": {"type": "object"}}
                        for name in ("paired_tool", "memory_only_tool")]

        def register(ctx):
            pending = []
            def hook(**kwargs):
                pending.append(kwargs["tool_call_id"])
            def handler(args, **kwargs):
                return pending.pop() if pending else "missing"
            ctx.register_memory_provider(Provider())
            ctx.register_hook("pre_tool_call", hook)
            ctx.register_tool(
                name="paired_tool", toolset="paired",
                schema={"name": "paired_tool", "description": "correlated tool",
                        "parameters": {"type": "object", "properties": {}}},
                handler=handler,
            )
    """))
    manager = get_plugin_manager()
    try:
        if order == "plugin-first":
            manager.discover_and_load()
        assert load_memory_provider("paired") is not None
        if order == "memory-first":
            manager.discover_and_load()
        for call_id in ("call_one", "call_two"):
            provider = load_memory_provider("paired")
            assert provider is not None
            memory = MemoryManager()
            memory.add_provider(provider)
            assert memory.has_tool("memory_only_tool")
            assert not memory.has_tool("paired_tool")
            assert resolve_invoke_tool_executor(
                SimpleNamespace(_memory_manager=memory), "paired_tool") is None
            manager.invoke_hook("pre_tool_call", tool_name="paired_tool", tool_call_id=call_id)
            entry = registry.get_entry("paired_tool", scope=manager.scope_key)
            assert entry is not None
            assert entry.handler({}) == call_id
            assert entry.handler({}) == "missing"
        manager.unload()
        assert registry.get_entry("paired_tool", scope=manager.scope_key) is None
        assert manager.invoke_hook("pre_tool_call", tool_name="paired_tool", tool_call_id="after") == []
    finally:
        manager.unload()
