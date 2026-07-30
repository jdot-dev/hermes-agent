"""Representative Phase 2 receipt window used by the committed readiness gate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent.action_receipts import ActionReceiptLedger


def _base_envelope(*, graph_id: str, node_id: str, attempt_id: str, surface: str) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "envelope_version": 2,
        "graph_id": graph_id,
        "node_id": node_id,
        "attempt_id": attempt_id,
        "idempotency_key": "e" * 64,
        "objective": f"representative {surface} readiness receipt",
        "execution_surface": surface,
        "lane": "hermes",
        "roots": ["/tmp"],
        "permissions": {"read": ["/tmp"], "write": [], "spawn": []},
        "network_policy": "none",
        "exec_policy": "none",
        "destructive_policy": "forbid",
        "budgets": {"usd_max": 1.0, "tokens_max": 1000, "wall_clock_s_max": 60},
        "deadline_utc": (now + timedelta(minutes=5)).isoformat(),
        "retry_policy": {"max_attempts": 1, "retry_eligible": False, "backoff_s": [0]},
        "cancellation": {"mode": "cooperative", "grace_s": 30},
        "evidence_policy": {
            "required_receipt_kinds": ["model_call" if surface == "direct_model" else "tool_exec"],
            "min_receipts": 1,
        },
        "verifier_policy": {
            "required": False,
            "verifier_lane_must_differ": False,
            "clean_workspace_required": False,
        },
        "planner_hash": "a" * 64,
        "policy_hash": "b" * 64,
        "input_hash": "c" * 64,
        "lease": {
            "holder": "hermes:readiness",
            "granted_utc": now.isoformat(),
            "ttl_s": 300,
            "renewable": True,
        },
        "fence": 1,
    }


def build_representative_receipts(path: Path) -> int:
    """Write one known-good direct-model and local-tool receipt; return baseline id."""

    ledger = ActionReceiptLedger(path)
    ledger.record_receipt(
        tool_name="codex/gpt-5.6-sol-xhigh",
        args={"model": "codex/gpt-5.6-sol-xhigh", "validation_errors": []},
        output=None,
        exit_status="ok",
        kind="model_call",
        envelope=_base_envelope(
            graph_id="g-readiness",
            node_id="n-model",
            attempt_id="at-model",
            surface="direct_model",
        ),
    )
    ledger.record_receipt(
        tool_name="read_file",
        args={"path": "/tmp/readiness.txt"},
        output="ok",
        exit_status="ok",
        kind="tool_exec",
        envelope=_base_envelope(
            graph_id="g-readiness",
            node_id="n-tool",
            attempt_id="at-tool",
            surface="local_tool",
        ),
    )
    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(build_representative_receipts(args.output))
