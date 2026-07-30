"""Phase 2 model-usage budget charging at the canonical accounting seam."""

from __future__ import annotations

import inspect

from agent import conversation_loop


def test_codex_app_server_transport_is_blocked_when_phase2_enforcement_is_enabled():
    source = inspect.getsource(conversation_loop.run_conversation)
    branch = source[source.index('if agent.api_mode == "codex_app_server":'):]
    branch = branch[: branch.index("while (")]
    assert "execution_surface_not_enforceable" in branch
    assert "is_enabled()" in branch


def test_conversation_loop_charges_canonical_usage_to_phase2_budget():
    source = inspect.getsource(conversation_loop.run_conversation)
    assert "record_budget_usage(" in source
    assert "tokens=total_tokens" in source
    assert "usd=_cost_delta" in source
    assert "usd=_cost_delta or 0.0" not in source
    assert "evaluate_runtime_authority()" in source
