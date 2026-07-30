from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "phase2_enforcement_readiness_gate.py"
spec = importlib.util.spec_from_file_location("phase2_gate", SCRIPT)
assert spec and spec.loader
phase2_gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase2_gate)


def _envelope(version: int = 2, surface: str = "local_tool") -> dict:
    return {
        "envelope_version": version,
        "graph_id": "g-gate",
        "node_id": "n-tool",
        "attempt_id": "at-1",
        "idempotency_key": "d" * 64,
        "objective": "score one sealed v2 receipt",
        "execution_surface": surface,
        "lane": "hermes",
        "roots": ["/tmp"],
        "permissions": {"read": ["/tmp"], "write": [], "spawn": []},
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
        "planner_hash": "a" * 64,
        "policy_hash": "b" * 64,
        "input_hash": "c" * 64,
        "lease": {
            "holder": "hermes:test",
            "granted_utc": "2026-07-30T00:00:00+00:00",
            "ttl_s": 300,
            "renewable": True,
        },
        "fence": 1,
    }


def _database(path: Path, envelope: dict, *, kind: str = "tool_exec") -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """CREATE TABLE action_receipts (
                   id INTEGER PRIMARY KEY,
                   receipt_id TEXT NOT NULL,
                   kind TEXT NOT NULL,
                   action_id TEXT NOT NULL,
                   exit_status TEXT NOT NULL,
                   envelope_json TEXT,
                   args_redacted_summary TEXT
               )"""
        )
        conn.execute(
            "INSERT INTO action_receipts VALUES (1, 'r-1', ?, 'a-1', 'ok', ?, '{}')",
            (kind, json.dumps(envelope)),
        )


def _append(path: Path, row_id: int, envelope: dict, *, kind: str = "tool_exec") -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO action_receipts VALUES (?, ?, ?, ?, 'ok', ?, '{}')",
            (row_id, f"r-{row_id}", kind, f"a-{row_id}", json.dumps(envelope)),
        )


def test_fresh_sealed_v2_local_tool_receipt_passes(tmp_path):
    path = tmp_path / "receipts.db"
    _database(path, _envelope())
    _append(path, 2, _envelope(surface="direct_model"), kind="model_call")

    result = phase2_gate.evaluate(path, after_id=0)

    assert result["activation_ready"] is True
    assert result["invalid_known_good_receipts"] == 0
    assert result["schema_drift_counts"] == {}
    assert result["valid_known_good_surface_counts"] == {
        "direct_model": 1,
        "local_tool": 1,
    }


def test_missing_direct_model_coverage_fails_closed(tmp_path):
    path = tmp_path / "receipts.db"
    _database(path, _envelope())

    result = phase2_gate.evaluate(path, after_id=0)

    assert result["activation_ready"] is False
    assert result["required_next_changes"] == [
        "produce at least one valid known-good sealed-v2 direct_model receipt after the explicit baseline"
    ]


def test_v1_observation_in_scoring_window_fails_closed(tmp_path):
    path = tmp_path / "receipts.db"
    _database(path, _envelope(version=1))

    result = phase2_gate.evaluate(path, after_id=0)

    assert result["activation_ready"] is False
    assert result["schema_drift_counts"]["envelope_version_not_contract_v2"] == 1


def test_cli_refuses_to_score_without_explicit_baseline(tmp_path):
    path = tmp_path / "receipts.db"
    _database(path, _envelope())
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--receipts",
            str(path),
            "--output",
            str(tmp_path / "result.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "--after-id" in result.stderr
    assert not (tmp_path / "result.json").exists()


def test_explicit_receipt_id_baseline_quarantines_historical_v1(tmp_path):
    path = tmp_path / "receipts.db"
    _database(path, _envelope(version=1))
    _append(path, 2, _envelope(version=2))
    _append(path, 3, _envelope(version=2, surface="direct_model"), kind="model_call")

    result = phase2_gate.evaluate(path, after_id=1)

    assert result["activation_ready"] is True
    assert result["after_receipt_id"] == 1
    assert result["receipt_count"] == 2
