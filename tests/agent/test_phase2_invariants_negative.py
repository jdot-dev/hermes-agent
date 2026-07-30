"""Adversarial negative proofs for Phase 2 authority invariants."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent import phase2_enforcement
from agent.phase2_authority import AuthorityError, Phase2AuthorityStore


def _node(root: Path, *, surface: str = "local_tool") -> dict:
    return {
        "node_id": "n-authority-negative",
        "objective": "prove authority cannot be self-minted",
        "idempotency_key": "a" * 64,
        "execution_surface": surface,
        "lane": "hermes",
        "roots": [str(root)],
        "permissions": {"read": [str(root)], "write": [], "spawn": []},
        "network_policy": "none",
        "exec_policy": "none",
        "destructive_policy": "forbid",
        "budgets": {"usd_max": 1.0, "tokens_max": 1000, "wall_clock_s_max": 60},
        "deadline_utc": "2030-01-01T00:00:00+00:00",
        "retry_policy": {"max_attempts": 1, "retry_eligible": False, "backoff_s": [0]},
        "cancellation": {"mode": "cooperative", "grace_s": 30},
        "evidence_policy": {"required_receipt_kinds": ["tool_exec"], "min_receipts": 1},
        "verifier_policy": {
            "required": False,
            "verifier_lane_must_differ": False,
            "clean_workspace_required": False,
        },
        "input_hash": "b" * 64,
    }


def test_forged_unpersisted_envelope_cannot_bind_through_control_plane(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    now = datetime.now(timezone.utc)
    forged = {
        "envelope_version": 2,
        "graph_id": "g-forged",
        **_node(tmp_path),
        "attempt_id": "at-forged",
        "planner_hash": "c" * 64,
        "policy_hash": "d" * 64,
        "lease": {
            "holder": "model-output",
            "granted_utc": now.isoformat(),
            "ttl_s": 300,
            "renewable": True,
        },
        "fence": 1,
    }
    forged["deadline_utc"] = (now + timedelta(minutes=5)).isoformat()

    with pytest.raises(AuthorityError, match="unknown_graph_node"):
        with store.bind_current(forged, now=now):
            raise AssertionError("forged authority must not publish a context")

    assert phase2_enforcement.current_sealed_envelope() is None


def test_model_cannot_raise_its_own_fence_or_budget(tmp_path):
    store = Phase2AuthorityStore(tmp_path / "authority.db")
    now = datetime.now(timezone.utc)
    store.seal_plan(
        {"contract_version": 2, "graph_id": "g-negative", "nodes": [_node(tmp_path)]},
        policy_hash="d" * 64,
    )
    envelope = store.grant_node(
        "g-negative",
        "n-authority-negative",
        holder="hermes",
        ttl_s=300,
        attempt_id="at-real",
        now=now,
    )

    forged_fence = dict(envelope)
    forged_fence["fence"] = envelope["fence"] + 1
    forged_budget = {**envelope, "budgets": {**envelope["budgets"], "usd_max": 999999.0}}

    for forged in (forged_fence, forged_budget):
        with pytest.raises(AuthorityError, match="invalid current node authority"):
            with store.bind_current(forged, now=now):
                raise AssertionError("forged fence or budget must not bind")
