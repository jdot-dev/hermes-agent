"""Bound automatic discovery without disabling native specialist routes."""
from contextlib import nullcontext

import pytest

from tools import mcp_tool_config as config
from tools import mcp_tool_discovery as discovery
from tools import mcp_tool

SERVERS = {'qmd': {'url': 'http://localhost/qmd'},
           'specialist': {'url': 'http://localhost/specialist'}}


@pytest.mark.parametrize('names,expected', [
    (None, {'qmd', 'specialist'}), (['qmd'], {'qmd'}), ([], set()),
    ('qmd', set()), ([False], set()), (['unknown'], set()),
])
def test_automatic_working_set(monkeypatch, names, expected):
    monkeypatch.setattr('hermes_cli.config.load_config_readonly',
                        lambda: {'mcp': {'automatic_servers': names}})
    assert set(config.automatic_mcp_servers(SERVERS)) == expected


def test_automatic_working_set_uses_current_profile(monkeypatch, tmp_path):
    from agent import secret_scope
    from gateway.run import _profile_runtime_scope

    for name in SERVERS:
        home = tmp_path / name
        home.mkdir()
        (home / 'config.yaml').write_text(f'mcp:\n  automatic_servers: [{name}]\n')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'qmd'))
    was_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        for name in ['qmd', 'specialist', 'qmd']:
            with _profile_runtime_scope(tmp_path / name, prepared_secret_scope={}):
                assert set(config.automatic_mcp_servers(SERVERS)) == {name}
    finally:
        secret_scope.set_multiplex_active(was_multiplex)


@pytest.mark.parametrize('explicit,expected', [(None, {'qmd'}), (['specialist'], {'specialist'})])
def test_explicit_selection_retains_native_specialist(monkeypatch, explicit, expected):
    core = mcp_tool
    monkeypatch.setattr('hermes_cli.config.load_config_readonly',
                        lambda: {'mcp': {'automatic_servers': ['qmd']}})
    monkeypatch.setattr(config, '_load_mcp_config', lambda: SERVERS)
    monkeypatch.setattr(discovery, '_owner_secret_scope', nullcontext)
    monkeypatch.setattr(core, '_ensure_mcp_sdk', lambda: True)
    monkeypatch.setattr(discovery, '_acquire_discovery_lock_with_retry', lambda: None)
    monkeypatch.setattr(discovery, '_log_summary', lambda *a, **kw: None)
    monkeypatch.setattr(discovery, 'register_mcp_servers', lambda servers: list(servers))
    assert set(discovery.discover_mcp_tools(allowed_mcp_names=explicit)) == expected


def test_reconcile_does_not_disconnect_explicit_enabled_specialist(monkeypatch):
    core = mcp_tool
    monkeypatch.setattr('hermes_cli.config.load_config_readonly',
                        lambda: {'mcp': {'automatic_servers': ['qmd']}})
    monkeypatch.setattr(config, '_load_mcp_config', lambda: SERVERS)
    monkeypatch.setattr(discovery, '_owner_secret_scope', nullcontext)
    monkeypatch.setattr(core, '_mcp_registry_scope', lambda: None)
    monkeypatch.setattr(core, '_server_scope_keys', {'specialist': None})
    monkeypatch.setattr(core, '_servers', {'specialist': object()})
    monkeypatch.setattr(core, '_server_connecting', set())
    monkeypatch.setattr(core, '_lazy_server_configs', {})
    seen = []
    monkeypatch.setattr(discovery, '_awaiting_connect', lambda wanted, scope: seen.append(wanted) or [])
    monkeypatch.setattr(discovery._lifecycle, 'shutdown_mcp_servers',
                        lambda **kwargs: pytest.fail('Explicit enabled server must stay connected'))
    assert discovery.reconcile_mcp_servers_with_config()['removed'] == []
    assert seen == [{'qmd'}]


@pytest.mark.parametrize('explicit,expected', [(None, []), (['specialist'], ['specialist'])])
def test_retry_only_considers_the_selected_working_set(monkeypatch, explicit, expected):
    from hermes_cli import mcp_startup

    monkeypatch.setattr('hermes_cli.config.load_config_readonly',
                        lambda: {'mcp': {'automatic_servers': ['qmd']}})
    monkeypatch.setattr(config, '_load_mcp_config', lambda: SERVERS)
    monkeypatch.setattr(discovery, '_owner_secret_scope', nullcontext)
    monkeypatch.setattr(mcp_tool, '_mcp_registry_scope', lambda: None)
    monkeypatch.setattr(mcp_tool, '_servers', {'qmd': object()})
    monkeypatch.setattr(mcp_tool, '_server_connecting', set())
    monkeypatch.setattr(mcp_tool, '_lazy_server_configs', {})
    monkeypatch.setattr(discovery, '_resolve_server_key', lambda name, *a, **kw: name)
    monkeypatch.setattr(discovery, '_connect_cooldown_active', lambda name: False)
    monkeypatch.setattr(mcp_startup, '_mcp_server_filter', explicit)

    assert mcp_startup._servers_awaiting_connect() == expected
