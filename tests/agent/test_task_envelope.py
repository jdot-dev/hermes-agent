"""Shadow task envelope contracts (Phase 0, §4/§12.3 of the quad-rework contract).

The envelope is *observational* in this phase: it records what the tool seam
can actually observe and marks every safety field it cannot know as ``None``.
It never invents a permissive default, and ``validate_envelope`` reports the
gaps without blocking anything.
"""

from agent.task_envelope import (
    ENVELOPE_SCHEMA_VERSION,
    build_shadow_envelope,
    envelope_hash,
    validate_envelope,
)


def test_shadow_envelope_leaves_unknown_safety_fields_null_and_reports_them():
    env = build_shadow_envelope(
        tool_name="terminal",
        args={"command": "ls"},
        cwd="/tmp/work",
        session_id="s-1",
        task_id="t-1",
        tool_call_id="c-1",
    )

    assert env.envelope_version == ENVELOPE_SCHEMA_VERSION == 1
    # Observable fields are recorded.
    assert env.tool_name == "terminal"
    assert env.cwd == "/tmp/work"
    assert env.execution_surface == "local_tool"
    assert env.lane == "hermes"

    # Unknown safety fields are explicitly null — never a permissive guess.
    assert env.permissions_write is None
    assert env.network_policy is None
    assert env.exec_policy is None
    assert env.destructive_policy is None

    # Validation reports the gaps without raising or blocking.
    missing = validate_envelope(env)
    assert isinstance(missing, list)
    for field in (
        "permissions_write",
        "network_policy",
        "exec_policy",
        "destructive_policy",
    ):
        assert field in missing


def test_envelope_hash_is_stable_for_equivalent_envelopes():
    kwargs = dict(
        tool_name="read_file",
        args={"b": 2, "a": 1},
        cwd="/tmp/work",
        session_id="s-1",
        task_id="t-1",
        tool_call_id="c-1",
        envelope_id="e-fixed",
        action_id="a-fixed",
        created_at="2026-07-29T00:00:00+00:00",
    )
    first = build_shadow_envelope(**kwargs)
    # Same logical inputs, different dict key order.
    second = build_shadow_envelope(**{**kwargs, "args": {"a": 1, "b": 2}})

    assert envelope_hash(first) == envelope_hash(second)
    assert len(envelope_hash(first)) == 64
