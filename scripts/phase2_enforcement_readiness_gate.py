#!/usr/bin/env python3
"""Evaluate whether Phase 2 envelope enforcement can safely be activated."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

MANDATORY_PATHS = (
    "graph_id",
    "node_id",
    "attempt_id",
    "idempotency_key",
    "objective",
    "execution_surface",
    "lane",
    "roots",
    "permissions.read",
    "permissions.write",
    "permissions.spawn",
    "network_policy",
    "exec_policy",
    "destructive_policy",
    "budgets.usd_max",
    "budgets.tokens_max",
    "budgets.wall_clock_s_max",
    "deadline_utc",
    "retry_policy.max_attempts",
    "retry_policy.retry_eligible",
    "retry_policy.backoff_s",
    "cancellation.mode",
    "cancellation.grace_s",
    "evidence_policy.required_receipt_kinds",
    "evidence_policy.min_receipts",
    "verifier_policy.required",
    "verifier_policy.verifier_lane_must_differ",
    "verifier_policy.clean_workspace_required",
    "planner_hash",
    "policy_hash",
    "input_hash",
    "lease.holder",
    "lease.granted_utc",
    "lease.ttl_s",
    "lease.renewable",
    "fence",
)
CONTRACT_SURFACES = {"direct_model", "combo", "acp_worker", "a2a", "cloud", "local_tool"}


def _load_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _value_at(document: dict[str, Any], dotted: str) -> Any:
    value: Any = document
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def evaluate(receipts_path: Path, after_id: int) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{receipts_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            """SELECT id, receipt_id, kind, action_id, exit_status, envelope_json,
                      args_redacted_summary
               FROM action_receipts
               WHERE id > ?
               ORDER BY id""",
            (after_id,),
        ).fetchall()
    finally:
        conn.close()

    missing_counts: Counter[str] = Counter()
    invalid_receipts: list[dict[str, Any]] = []
    schema_drifts: Counter[str] = Counter()
    known_good = 0
    valid_known_good_surfaces: Counter[str] = Counter()
    for row in rows:
        receipt_id_num, receipt_id, kind, action_id, exit_status, envelope_json, summary_json = row
        if exit_status == "ok":
            known_good += 1
        envelope = _load_json(envelope_json)
        summary = _load_json(summary_json)
        missing = [field for field in MANDATORY_PATHS if _value_at(envelope, field) is None]
        for field in missing:
            missing_counts[field] += 1
        receipt_drifts: list[str] = []
        if envelope.get("envelope_version") != 2:
            receipt_drifts.append("envelope_version_not_contract_v2")
        if envelope.get("execution_surface") not in CONTRACT_SURFACES:
            receipt_drifts.append("execution_surface_not_in_contract")
        if envelope.get("execution_surface") == "direct_model" and kind != "model_call":
            receipt_drifts.append("direct_model_receipt_kind_mismatch")
        if envelope.get("execution_surface") == "local_tool" and kind != "tool_exec":
            receipt_drifts.append("local_tool_receipt_kind_mismatch")
        if kind == "stale_rejection" and exit_status != "rejected":
            receipt_drifts.append("stale_rejection_status_mismatch")
        reported = summary.get("validation_errors")
        if reported is not None and not isinstance(reported, list):
            receipt_drifts.append("recorded_validation_errors_not_list")
        schema_drifts.update(set(receipt_drifts))
        if exit_status == "ok" and not missing and not receipt_drifts:
            valid_known_good_surfaces[str(envelope.get("execution_surface"))] += 1
        if missing or receipt_drifts:
            invalid_receipts.append(
                {
                    "id": receipt_id_num,
                    "receipt_id": receipt_id,
                    "kind": kind,
                    "action_id": action_id,
                    "exit_status": exit_status,
                    "missing": missing,
                    "execution_surface": envelope.get("execution_surface"),
                    "envelope_version": envelope.get("envelope_version"),
                }
            )

    invalid_known_good = sum(1 for item in invalid_receipts if item["exit_status"] == "ok")
    false_fail_rate = invalid_known_good / known_good if known_good else None
    blockers: list[str] = []
    if not rows:
        blockers.append("no representative receipts after the selected baseline")
    if invalid_known_good:
        blockers.append("known-good traffic still produces invalid sealed envelopes")
    if schema_drifts.get("envelope_version_not_contract_v2"):
        blockers.append("representative receipts do not carry the canonical sealed contract envelope v2")
    if schema_drifts.get("execution_surface_not_in_contract"):
        blockers.append("representative receipts include a non-contract execution surface")
    if missing_counts:
        blockers.append("representative receipts omit mandatory Phase 2 contract fields")
    required_surfaces = {"direct_model", "local_tool"}
    missing_surfaces = sorted(required_surfaces - set(valid_known_good_surfaces))
    if missing_surfaces:
        blockers.append(
            "representative receipt window lacks valid known-good coverage for: "
            + ", ".join(missing_surfaces)
        )

    required_next_changes: list[str] = []
    for surface in missing_surfaces:
        required_next_changes.append(
            f"produce at least one valid known-good sealed-v2 {surface} receipt after the explicit baseline"
        )
    if invalid_known_good:
        required_next_changes.append(
            "diagnose and eliminate invalid known-good receipts before activation"
        )

    return {
        "gate_version": 2,
        "phase": "2-enforcement-at-the-seam",
        "after_receipt_id": after_id,
        "receipt_count": len(rows),
        "known_good_receipts": known_good,
        "invalid_known_good_receipts": invalid_known_good,
        "valid_known_good_surface_counts": dict(sorted(valid_known_good_surfaces.items())),
        "known_good_invalid_rate": false_fail_rate,
        "phase1_stop_threshold": 0.05,
        "missing_field_counts": dict(sorted(missing_counts.items())),
        "schema_drift_counts": dict(sorted(schema_drifts.items())),
        "activation_ready": not blockers,
        "blockers": blockers,
        "required_next_changes": required_next_changes,
        "sample_invalid_receipts": invalid_receipts[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, default=Path("/home/jack/.hermes/action_receipts.db"))
    parser.add_argument(
        "--after-id",
        type=int,
        required=True,
        help="Explicit monotonic action_receipts.id baseline; only newer rows are scored",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.receipts, args.after_id)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    digest = hashlib.sha256(rendered.encode()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "activation_ready": result["activation_ready"],
                "blockers": result["blockers"],
                "sha256": digest,
            }
        )
    )
    return 0 if result["activation_ready"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
