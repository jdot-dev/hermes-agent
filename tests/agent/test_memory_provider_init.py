"""Regression tests for memory provider selection during AIAgent init."""

from types import SimpleNamespace
import threading
from unittest.mock import patch

import pytest


class RecordingMemoryProvider:
    name = "recording"

    def __init__(self):
        self.init_kwargs = None
        self.init_session_id = None

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.init_session_id = session_id
        self.init_kwargs = dict(kwargs)

    def get_tool_schemas(self):
        return []

    def shutdown(self):
        pass


def test_shutdown_memory_provider_is_idempotent():
    from unittest.mock import MagicMock

    from run_agent import AIAgent

    manager = MagicMock()
    agent = object.__new__(AIAgent)
    agent._memory_manager = manager
    agent.context_compressor = None
    agent.session_id = "session-1"

    agent.shutdown_memory_provider([{"role": "user", "content": "one"}])
    agent.shutdown_memory_provider([{"role": "user", "content": "two"}])

    manager.on_session_end.assert_called_once()
    manager.shutdown_all.assert_called_once()


def test_blank_memory_provider_stays_opt_out():
    """Blank memory.provider loads no external provider and never rewrites the config."""
    cfg = {"memory": {"provider": ""}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("hermes_cli.config.save_config") as save_config,
        patch("plugins.memory.load_memory_provider") as load_memory_provider,
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
        )

    assert agent._memory_manager is None
    load_memory_provider.assert_not_called()
    save_config.assert_not_called()


def test_close_shuts_down_memory_provider():
    from unittest.mock import MagicMock

    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent._memory_manager = MagicMock()
    agent.context_compressor = None
    agent.session_id = ""
    agent._session_messages = []

    agent.close()

    agent._memory_manager.shutdown_all.assert_called_once()


def test_aiagent_forwards_user_id_alt_to_memory_provider():
    provider = RecordingMemoryProvider()
    cfg = {"memory": {"provider": "recording"}, "agent": {}}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg), patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
            session_id="sess-alt",
            platform="feishu",
            user_id="open-id",
            user_id_alt="union-id",
        )

    assert agent._memory_manager is not None
    assert provider.init_session_id == "sess-alt"
    assert provider.init_kwargs["user_id"] == "open-id"
    assert provider.init_kwargs["user_id_alt"] == "union-id"
    assert provider.init_kwargs["platform"] == "feishu"
    assert "warning_callback" not in provider.init_kwargs
    assert "status_callback" not in provider.init_kwargs


class CoreShadowProvider:
    """Provider that tries to register tools shadowing built-in core tools."""

    name = "core-shadow"

    def get_tool_schemas(self):
        return [
            {"name": "clarify", "description": "shadows built-in clarify"},
            {"name": "delegate_task", "description": "shadows built-in delegate"},
            {"name": "honcho_search", "description": "legit memory tool"},
        ]


def test_core_tool_names_rejected_from_memory_routing_table():
    """Memory tools shadowing core tool names are rejected at registration (#40466).

    Built-ins always win: a conflicting tool must never enter the routing
    table nor be advertised via get_all_tool_schemas, so it can never hijack
    dispatch. The non-conflicting tool is preserved.
    """
    from agent.memory_manager import MemoryManager

    mm = MemoryManager()
    mm.add_provider(CoreShadowProvider())

    # Reserved names never enter the routing table
    assert not mm.has_tool("clarify")
    assert not mm.has_tool("delegate_task")
    assert "clarify" not in mm._tool_to_provider
    assert "delegate_task" not in mm._tool_to_provider

    # Non-conflicting tool survives
    assert mm.has_tool("honcho_search")
    assert "honcho_search" in mm._tool_to_provider

    # Manager never advertises a schema it would refuse to route
    schema_names = {s.get("name") for s in mm.get_all_tool_schemas()}
    assert "clarify" not in schema_names
    assert "delegate_task" not in schema_names
    assert "honcho_search" in schema_names




def test_aiagent_reuses_handed_in_memory_manager_without_reinitializing(tmp_path, monkeypatch):
    """A caller that rebuilds the agent per turn (gateway api_server) hands back the session's manager:
    the provider keeps its state — no second load, no second initialize (#120116)."""
    provider = RecordingMemoryProvider()
    from hermes_cli.config import atomic_config_write

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    atomic_config_write(tmp_path / "config.yaml", {"memory": {
        "provider": "recording", "external_prefetch_timeout_seconds": 20,
    }})
    common = dict(
        api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
        skip_context_files=True, skip_memory=False, session_id="sess-api", platform="api_server",
    )
    with (
        patch("plugins.memory.load_memory_provider", return_value=provider) as load_memory_provider,
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        from run_agent import AIAgent

        first = AIAgent(**common)
        manager = first._memory_manager
        assert manager is not None and load_memory_provider.call_count == 1
        assert manager._external_prefetch_timeout == 20
        provider.init_session_id = None  # a re-initialize would set it again
        atomic_config_write(tmp_path / "config.yaml", {"memory": {
            "provider": "recording", "external_prefetch_timeout_seconds": 40,
        }})

        second = AIAgent(memory_manager=manager, **common)

    assert second._memory_manager is manager
    assert load_memory_provider.call_count == 1
    assert provider.init_session_id is None
    assert manager._external_prefetch_timeout == 20


@pytest.mark.parametrize("configured, expected", [
    (None, 8.0), (20, 20.0), (0.25, 0.25), ("12.5", 12.5),
    ("invalid", 8.0), (float("nan"), 8.0), (float("inf"), 8.0),
    (float("-inf"), 8.0), (0, 8.0), (-1, 8.0),
])
def test_profile_prefetch_timeout_reaches_external_wait(tmp_path, monkeypatch, configured, expected):
    """Profile YAML controls the real prefetch deadline; invalid values keep recall enabled."""
    from hermes_cli.config import atomic_config_write
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    memory = {"provider": "recording", "memory_enabled": False, "user_profile_enabled": False}
    if configured is not None:
        memory["external_prefetch_timeout_seconds"] = configured
    atomic_config_write(tmp_path / "config.yaml", {"memory": memory})
    provider = RecordingMemoryProvider()
    provider.prefetch = lambda query, **kwargs: "source-linked recall"
    with (
        patch("plugins.memory.load_memory_provider", return_value=provider),
        patch("agent.model_metadata.get_model_context_length", return_value=204_800),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890", base_url="http://127.0.0.1:1/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=False,
        )

    assert agent._memory_manager is not None
    deadlines = []
    join = threading.Thread.join

    def record_join(thread, timeout=None):
        deadlines.append(timeout)
        join(thread, timeout)

    with patch.object(threading.Thread, "join", record_join):
        result = agent._memory_manager.prefetch_all("operational fact")
    assert result == "source-linked recall"
    assert deadlines == [expected]
