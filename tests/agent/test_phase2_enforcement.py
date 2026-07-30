"""Phase 2 task-envelope enforcement contracts.

This suite is intentionally fail-closed when enforcement is enabled. The live
profile remains default-off until a sealed-envelope producer binds v2 envelopes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent import phase2_enforcement


def _envelope(root: Path, **overrides):
    now = datetime.now(timezone.utc)
    base = {
        "envelope_version": 2,
        "graph_id": "g-phase2-1",
        "node_id": "n-phase2-1",
        "attempt_id": "at-phase2-1",
        "idempotency_key": "d" * 64,
        "objective": "exercise the Phase 2 enforcement contract",
        "execution_surface": "local_tool",
        "lane": "hermes",
        "roots": [str(root)],
        "permissions": {
            "read": [str(root)],
            "write": [str(root)],
            "spawn": [],
        },
        "network_policy": "none",
        "exec_policy": "host",
        "destructive_policy": "forbid",
        "budgets": {
            "usd_max": 1.0,
            "tokens_max": 10000,
            "wall_clock_s_max": 300,
        },
        "deadline_utc": (now + timedelta(minutes=5)).isoformat(),
        "retry_policy": {
            "max_attempts": 1,
            "retry_eligible": False,
            "backoff_s": [0],
        },
        "cancellation": {"mode": "cooperative", "grace_s": 30},
        "evidence_policy": {
            "required_receipt_kinds": ["tool_exec", "verification"],
            "min_receipts": 1,
        },
        "verifier_policy": {
            "required": True,
            "verifier_lane_must_differ": True,
            "clean_workspace_required": True,
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
    base.update(overrides)
    return base


def _set_enabled(monkeypatch, enabled: bool):
    monkeypatch.setattr(
        phase2_enforcement,
        "_load_config",
        lambda: {"enforcement": {"task_envelopes": {"enabled": enabled}}},
    )


def test_default_off_allows_without_envelope(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, False)
    assert (
        phase2_enforcement.evaluate_tool_call(
            "write_file", {"path": str(tmp_path / "x")}, cwd=str(tmp_path)
        )
        is None
    )


def test_enabled_without_bound_envelope_fails_closed(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    decision = phase2_enforcement.evaluate_tool_call(
        "write_file", {"path": str(tmp_path / "x")}, cwd=str(tmp_path)
    )
    assert decision is not None
    assert decision.code == "missing_sealed_envelope"


def test_local_tool_dispatch_rejects_direct_model_node(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    envelope = _envelope(tmp_path, execution_surface="direct_model")
    with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
        decision = phase2_enforcement.evaluate_tool_call(
            "read_file", {"path": str(tmp_path / "x")}, cwd=str(tmp_path)
        )
    assert decision is not None
    assert decision.code == "execution_surface_mismatch"


def test_model_dispatch_rejects_local_tool_node(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    envelope = _envelope(tmp_path, execution_surface="local_tool")
    with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
        decision = phase2_enforcement.evaluate_runtime_authority()
    assert decision is not None
    assert decision.code == "execution_surface_mismatch"


def test_contract_v1_local_tool_envelope_is_not_silently_reinterpreted(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    envelope = _envelope(tmp_path, envelope_version=1)
    with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
        decision = phase2_enforcement.evaluate_tool_call(
            "read_file", {"path": str(tmp_path / "inside.txt")}, cwd=str(tmp_path)
        )
    assert decision is not None
    assert decision.code == "invalid_sealed_envelope"
    assert "envelope_version" in decision.details


def test_contract_v2_local_tool_is_a_valid_execution_surface(tmp_path):
    assert phase2_enforcement.validate_sealed_envelope(_envelope(tmp_path)) == []


def test_contract_v2_still_rejects_unknown_execution_surfaces(tmp_path):
    errors = phase2_enforcement.validate_sealed_envelope(
        _envelope(tmp_path, execution_surface="local_tool_alias")
    )
    assert "execution_surface" in errors


def test_relative_permission_roots_are_rejected(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    envelope = _envelope(
        tmp_path,
        permissions={"read": ["relative"], "write": [], "spawn": []},
    )
    with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
        decision = phase2_enforcement.evaluate_tool_call(
            "read_file", {"path": str(tmp_path / "inside.txt")}, cwd=str(tmp_path)
        )
    assert decision is not None
    assert decision.code == "invalid_sealed_envelope"
    assert "permissions.read" in decision.details


def test_valid_bound_envelope_allows_declared_read(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    with phase2_enforcement.bind_sealed_envelope(_envelope(tmp_path), current_fence=1):
        assert (
            phase2_enforcement.evaluate_tool_call(
                "read_file", {"path": str(tmp_path / "inside.txt")}, cwd=str(tmp_path)
            )
            is None
        )


def test_write_outside_permission_root_is_blocked(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    with phase2_enforcement.bind_sealed_envelope(_envelope(allowed), current_fence=1):
        decision = phase2_enforcement.evaluate_tool_call(
            "write_file", {"path": str(tmp_path / "outside.txt")}, cwd=str(allowed)
        )
    assert decision is not None
    assert decision.code == "write_outside_permissions"


def test_symlinked_read_outside_permission_root_is_blocked(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("not authorized")
    link = allowed / "escape.txt"
    link.symlink_to(secret)

    with phase2_enforcement.bind_sealed_envelope(_envelope(allowed), current_fence=1):
        decision = phase2_enforcement.evaluate_tool_call(
            "read_file", {"path": str(link)}, cwd=str(allowed)
        )

    assert decision is not None
    assert decision.code == "read_outside_permissions"


def test_symlinked_write_parent_outside_permission_root_is_blocked(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    link = allowed / "escape"
    link.symlink_to(outside, target_is_directory=True)

    with phase2_enforcement.bind_sealed_envelope(_envelope(allowed), current_fence=1):
        decision = phase2_enforcement.evaluate_tool_call(
            "write_file", {"path": str(link / "created.txt")}, cwd=str(allowed)
        )

    assert decision is not None
    assert decision.code == "write_outside_permissions"


def test_nonexistent_write_target_is_still_blocked_without_race_safe_resolution(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    target = allowed / "new.txt"
    with phase2_enforcement.bind_sealed_envelope(_envelope(allowed), current_fence=1):
        decision = phase2_enforcement.evaluate_tool_call(
            "write_file", {"path": str(target), "content": "x"}, cwd=str(allowed)
        )
    assert decision is not None
    assert decision.code == "write_target_race_not_enforceable"


def test_expired_deadline_is_blocked(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    expired = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    with phase2_enforcement.bind_sealed_envelope(
        _envelope(tmp_path, deadline_utc=expired), current_fence=1
    ):
        decision = phase2_enforcement.evaluate_tool_call(
            "read_file", {"path": str(tmp_path / "x")}, cwd=str(tmp_path)
        )
    assert decision is not None
    assert decision.code == "deadline_expired"


def test_invalid_hash_or_fence_is_rejected(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    with phase2_enforcement.bind_sealed_envelope(
        _envelope(tmp_path, policy_hash="not-a-hash", fence=0), current_fence=0
    ):
        decision = phase2_enforcement.evaluate_tool_call(
            "read_file", {"path": str(tmp_path / "x")}, cwd=str(tmp_path)
        )
    assert decision is not None
    assert decision.code == "invalid_sealed_envelope"
    assert "policy_hash" in decision.details
    assert "fence" in decision.details


def test_web_extract_remains_blocked_until_redirect_and_dns_destination_checks_exist(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    envelope = _envelope(
        tmp_path,
        network_policy="allowlist",
        network_allowlist=["example.com"],
    )
    with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
        decision = phase2_enforcement.evaluate_tool_call(
            "web_extract", {"urls": ["https://example.com/docs"]}, cwd=str(tmp_path)
        )
    assert decision is not None
    assert decision.code == "network_transport_not_enforceable"


def test_web_extract_rejects_non_allowlisted_destination(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    envelope = _envelope(
        tmp_path,
        network_policy="allowlist",
        network_allowlist=["example.com"],
    )
    with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
        decision = phase2_enforcement.evaluate_tool_call(
            "web_extract", {"urls": ["https://evil.example.net/"]}, cwd=str(tmp_path)
        )
    assert decision is not None
    assert decision.code == "network_destination_denied"


def test_web_extract_loopback_policy_rejects_non_loopback_and_allows_loopback(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    envelope = _envelope(tmp_path, network_policy="loopback_only")
    with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
        allowed = phase2_enforcement.evaluate_tool_call(
            "web_extract", {"urls": ["http://127.0.0.1:8889/status"]}, cwd=str(tmp_path)
        )
        denied = phase2_enforcement.evaluate_tool_call(
            "web_extract", {"urls": ["https://example.com/"]}, cwd=str(tmp_path)
        )
    assert allowed is not None
    assert allowed.code == "network_transport_not_enforceable"
    assert denied is not None
    assert denied.code == "network_destination_denied"


def test_network_allowlist_does_not_match_subdomains_or_userinfo_spoofing(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    envelope = _envelope(
        tmp_path,
        network_policy="allowlist",
        network_allowlist=["example.com"],
    )
    for url in (
        "https://sub.example.com/",
        "https://example.com.evil.test/",
        "https://example.com@evil.test/",
    ):
        with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
            decision = phase2_enforcement.evaluate_tool_call(
                "web_extract", {"urls": [url]}, cwd=str(tmp_path)
            )
        assert decision is not None
        assert decision.code in {"network_destination_denied", "network_destination_ambiguous"}


def test_web_search_remains_blocked_because_query_does_not_bind_destination(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    envelope = _envelope(
        tmp_path,
        network_policy="allowlist",
        network_allowlist=["example.com"],
    )
    with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
        decision = phase2_enforcement.evaluate_tool_call(
            "web_search", {"query": "site:example.com phase 2"}, cwd=str(tmp_path)
        )
    assert decision is not None
    assert decision.code == "network_destination_ambiguous"


def test_delegate_task_remains_blocked_until_child_authority_is_bound(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    envelope = _envelope(
        tmp_path,
        permissions={"read": [str(tmp_path)], "write": [], "spawn": ["delegate_task:leaf"]},
    )
    with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
        decision = phase2_enforcement.evaluate_tool_call(
            "delegate_task", {"goal": "inspect only", "role": "leaf"}, cwd=str(tmp_path)
        )
    assert decision is not None
    assert decision.code == "spawn_child_authority_not_enforceable"


def test_delegate_task_rejects_unlisted_role_and_nested_batch(monkeypatch, tmp_path):
    """Every rejected spawn shape reports its own reason and fails closed.

    Phase 3 makes a *well-formed* batch enforceable — each task resolves to its
    own separately sealed child through a graph edge — so a batch is no longer
    categorically ambiguous. It is now rejected one step later, for the reason
    that actually applies: no sealed child exists for it (see
    ``tests/agent/test_phase3_delegate_authority.py`` for the enforced path).
    """

    _set_enabled(monkeypatch, True)
    envelope = _envelope(
        tmp_path,
        permissions={"read": [str(tmp_path)], "write": [], "spawn": ["delegate_task:leaf"]},
    )
    expected = {
        # Role outside the sealed spawn list — denied by policy, before any
        # graph lookup, so the operator sees the policy reason.
        "spawn_policy_denied": {"goal": "inspect", "role": "orchestrator"},
        # Batch that is not a bounded list of task objects.
        "spawn_request_ambiguous": {"tasks": []},
        # Well-formed batch with no separately sealed child behind it.
        "spawn_child_authority_not_enforceable": {
            "tasks": [{"goal": "one"}, {"goal": "two"}]
        },
    }
    for code, args in expected.items():
        with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
            decision = phase2_enforcement.evaluate_tool_call("delegate_task", args, cwd=str(tmp_path))
        assert decision is not None
        assert decision.code == code, (args, decision)


def test_terminal_destructive_command_respects_policy(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    with phase2_enforcement.bind_sealed_envelope(_envelope(tmp_path), current_fence=1):
        decision = phase2_enforcement.evaluate_tool_call(
            "terminal",
            {"command": "rm -rf build", "workdir": str(tmp_path)},
            cwd=str(tmp_path),
            destructive=True,
        )
    assert decision is not None
    assert decision.code == "destructive_policy_denied"


def test_terminal_non_destructive_command_requires_spawn_authority(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    with phase2_enforcement.bind_sealed_envelope(_envelope(tmp_path), current_fence=1):
        decision = phase2_enforcement.evaluate_tool_call(
            "terminal", {"command": "pwd", "workdir": str(tmp_path)}, cwd=str(tmp_path)
        )
    assert decision is not None
    assert decision.code == "spawn_policy_denied"


def test_terminal_spawn_authorized_still_blocks_shell_transport(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    envelope = _envelope(
        tmp_path,
        permissions={"read": [str(tmp_path)], "write": [], "spawn": ["pwd"]},
    )
    with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
        decision = phase2_enforcement.evaluate_tool_call(
            "terminal", {"command": "pwd", "workdir": str(tmp_path)}, cwd=str(tmp_path)
        )
    assert decision is not None
    assert decision.code == "terminal_transport_not_enforceable"


def test_terminal_absolute_executable_must_be_allowlisted_exactly(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    envelope = _envelope(
        tmp_path,
        permissions={"read": [str(tmp_path)], "write": [], "spawn": ["pwd"]},
    )
    with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
        decision = phase2_enforcement.evaluate_tool_call(
            "terminal", {"command": "/tmp/pwd", "workdir": str(tmp_path)}, cwd=str(tmp_path)
        )
    assert decision is not None
    assert decision.code == "spawn_policy_denied"


def test_terminal_arguments_remain_blocked_until_per_command_policy_exists(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    envelope = _envelope(
        tmp_path,
        permissions={"read": [str(tmp_path)], "write": [], "spawn": ["printf"]},
    )
    with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
        decision = phase2_enforcement.evaluate_tool_call(
            "terminal", {"command": "printf hello", "workdir": str(tmp_path)}, cwd=str(tmp_path)
        )
    assert decision is not None
    assert decision.code == "terminal_arguments_not_enforceable"


def test_terminal_shell_syntax_remains_blocked(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    envelope = _envelope(
        tmp_path,
        permissions={"read": [str(tmp_path)], "write": [], "spawn": ["pwd"]},
    )
    for command in ("pwd; id", "pwd && id", "echo $(id)", "pwd\nid"):
        with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
            decision = phase2_enforcement.evaluate_tool_call(
                "terminal", {"command": command, "workdir": str(tmp_path)}, cwd=str(tmp_path)
            )
        assert decision is not None
        assert decision.code == "terminal_command_ambiguous"


def test_terminal_argv0_requires_spawn_permission(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    envelope = _envelope(
        tmp_path,
        permissions={"read": [str(tmp_path)], "write": [], "spawn": ["pwd"]},
    )
    with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
        decision = phase2_enforcement.evaluate_tool_call(
            "terminal", {"command": "id", "workdir": str(tmp_path)}, cwd=str(tmp_path)
        )
    assert decision is not None
    assert decision.code == "spawn_policy_denied"


def test_terminal_workdir_outside_roots_is_blocked(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    envelope = _envelope(
        allowed,
        permissions={"read": [str(allowed)], "write": [], "spawn": ["pwd"]},
    )
    with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
        decision = phase2_enforcement.evaluate_tool_call(
            "terminal", {"command": "pwd", "workdir": str(outside)}, cwd=str(allowed)
        )
    assert decision is not None
    assert decision.code == "exec_workdir_outside_roots"


def test_terminal_background_execution_remains_blocked(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    envelope = _envelope(
        tmp_path,
        permissions={"read": [str(tmp_path)], "write": [], "spawn": ["pwd"]},
    )
    with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
        decision = phase2_enforcement.evaluate_tool_call(
            "terminal",
            {"command": "pwd", "workdir": str(tmp_path), "background": True},
            cwd=str(tmp_path),
        )
    assert decision is not None
    assert decision.code == "terminal_background_not_enforceable"


def test_multi_file_patch_fails_closed_until_all_targets_can_be_authorized(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    with phase2_enforcement.bind_sealed_envelope(_envelope(tmp_path), current_fence=1):
        decision = phase2_enforcement.evaluate_tool_call(
            "patch", {"mode": "patch", "patch": "*** Begin Patch\n*** End Patch"}, cwd=str(tmp_path)
        )
    assert decision is not None
    assert decision.code == "multi_file_patch_not_enforceable"


def test_bound_envelope_is_defensively_copied(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    envelope = _envelope(tmp_path)
    with phase2_enforcement.bind_sealed_envelope(envelope, current_fence=1):
        envelope["permissions"]["read"].clear()
        current = phase2_enforcement.current_sealed_envelope()
        assert current is not None
        assert str(tmp_path) in current["permissions"]["read"]


def test_stale_fence_is_blocked(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    with phase2_enforcement.bind_sealed_envelope(_envelope(tmp_path), current_fence=2):
        decision = phase2_enforcement.evaluate_tool_call(
            "read_file", {"path": str(tmp_path / "x")}, cwd=str(tmp_path)
        )
    assert decision is not None
    assert decision.code == "stale_fence"


def test_missing_authoritative_fence_is_blocked(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    with phase2_enforcement.bind_sealed_envelope(_envelope(tmp_path)):
        decision = phase2_enforcement.evaluate_tool_call(
            "read_file", {"path": str(tmp_path / "x")}, cwd=str(tmp_path)
        )
    assert decision is not None
    assert decision.code == "missing_current_fence"


def test_context_binding_is_reset_after_scope(monkeypatch, tmp_path):
    _set_enabled(monkeypatch, True)
    with phase2_enforcement.bind_sealed_envelope(_envelope(tmp_path), current_fence=1):
        assert phase2_enforcement.current_sealed_envelope() is not None
    assert phase2_enforcement.current_sealed_envelope() is None
