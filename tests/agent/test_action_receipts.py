"""Append-only action receipt ledger contracts (Phase 0, §7/§12.3).

The ledger is default-off telemetry. It must never be reachable unless the
user opts in through ``config.yaml``, it must be tamper-evident once written,
and it must never store a raw secret.
"""

import sqlite3
import json

import pytest

from agent import action_receipts


def _enable(monkeypatch, value):
    """Point ``is_enabled`` at an injected config instead of the user's file."""
    monkeypatch.setattr(
        action_receipts,
        "_load_config",
        lambda: {"observability": {"action_receipts": {"enabled": value}}},
    )


# ---------------------------------------------------------------------------
# Default-off gate
# ---------------------------------------------------------------------------
def test_is_enabled_is_false_when_config_key_absent(monkeypatch):
    monkeypatch.setattr(action_receipts, "_load_config", dict)
    assert action_receipts.is_enabled() is False


def test_is_enabled_is_false_when_config_key_false(monkeypatch):
    _enable(monkeypatch, False)
    assert action_receipts.is_enabled() is False


def test_is_enabled_is_true_only_when_config_key_true(monkeypatch):
    _enable(monkeypatch, True)
    assert action_receipts.is_enabled() is True


def test_is_enabled_is_false_when_config_read_raises(monkeypatch):
    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(action_receipts, "_load_config", _boom)
    assert action_receipts.is_enabled() is False


# ---------------------------------------------------------------------------
# Ledger storage
# ---------------------------------------------------------------------------
@pytest.fixture()
def ledger(tmp_path):
    return action_receipts.ActionReceiptLedger(db_path=tmp_path / "action_receipts.db")


def _record(ledger, **overrides):
    fields = dict(
        tool_name="terminal",
        args={"command": "ls"},
        output="ok",
        exit_status="ok",
        duration_ms=12,
        session_id="s-1",
        task_id="t-1",
        tool_call_id="c-1",
    )
    fields.update(overrides)
    return ledger.record_receipt(**fields)


def _sealed_v2_envelope():
    return {
        "envelope_version": 2,
        "graph_id": "g-receipt",
        "node_id": "n-tool",
        "attempt_id": "at-1",
        "idempotency_key": "d" * 64,
        "objective": "record one sealed tool receipt",
        "execution_surface": "local_tool",
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


def test_sealed_v2_mapping_is_recorded_without_reinterpreting_it_as_v1(ledger):
    envelope = _sealed_v2_envelope()

    _record(ledger, envelope=envelope)

    row = ledger.read_all()[0]
    stored = json.loads(row["envelope_json"])
    assert stored == envelope
    assert row["action_id"] == "g-receipt:n-tool:at-1"
    assert row["execution_surface"] == "local_tool"
    assert row["envelope_hash"] == action_receipts.canonical_hash(envelope)
    assert ledger.verify_chain() == []


def test_default_db_path_lives_in_hermes_home(monkeypatch, tmp_path):
    monkeypatch.setattr(action_receipts, "get_hermes_home", lambda: tmp_path)
    assert action_receipts.default_db_path() == tmp_path / "action_receipts.db"


def test_reading_missing_ledger_does_not_create_database(tmp_path):
    path = tmp_path / "missing" / "action_receipts.db"
    missing = action_receipts.ActionReceiptLedger(db_path=path)

    assert missing.read_all() == []
    assert not path.exists()


def test_append_only_triggers_reject_update_and_delete(ledger):
    _record(ledger)

    with sqlite3.connect(ledger.db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("UPDATE action_receipts SET tool_name = 'tampered'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM action_receipts")

    rows = ledger.read_all()
    assert len(rows) == 1
    assert rows[0]["tool_name"] == "terminal"


def test_hash_chain_links_every_receipt_and_verifies(ledger):
    for i in range(3):
        _record(ledger, tool_call_id=f"c-{i}")

    rows = ledger.read_all()
    assert len(rows) == 3
    assert rows[0]["prev_receipt_hash"] is None
    assert rows[1]["prev_receipt_hash"] == rows[0]["receipt_hash"]
    assert rows[2]["prev_receipt_hash"] == rows[1]["receipt_hash"]
    assert ledger.verify_chain() == []


def test_verify_chain_detects_a_broken_link(ledger):
    _record(ledger)
    _record(ledger, tool_call_id="c-2")

    # Triggers block UPDATE/DELETE, so simulate tampering the only way an
    # attacker could: by rewriting the file out-of-band.
    with sqlite3.connect(ledger.db_path) as conn:
        conn.execute("DROP TRIGGER action_receipts_no_update")
        conn.execute("UPDATE action_receipts SET tool_name = 'tampered' WHERE id = 1")
        conn.commit()

    assert ledger.verify_chain() != []


def test_concurrent_writes_do_not_fork_the_chain(ledger):
    import threading

    def _worker(n):
        _record(ledger, tool_call_id=f"c-{n}")

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rows = ledger.read_all()
    assert len(rows) == 12
    # A forked chain shows up as a duplicate prev pointer or a duplicate hash.
    prevs = [r["prev_receipt_hash"] for r in rows[1:]]
    assert len(set(prevs)) == len(prevs)
    assert len({r["receipt_hash"] for r in rows}) == 12
    assert ledger.verify_chain() == []


# ---------------------------------------------------------------------------
# Redaction (G2)
# ---------------------------------------------------------------------------
def test_secret_shaped_nested_args_never_reach_stored_summary(ledger):
    _record(
        ledger,
        args={
            "api_key": "«redacted:sk-…»",
            "nested": {
                "headers": {"Authorization": "Bearer TOPSECRETTOKEN"},
                "list": [{"password": "hunter2"}, {"client_secret": "SHHH"}],
            },
            "credential": "CREDVALUE",
            "safe": "keep-me",
        },
    )

    row = ledger.read_all()[0]
    blob = "\n".join(str(v) for v in row.values())
    for secret in (
        "TOPSECRETTOKEN",
        "hunter2",
        "SHHH",
        "CREDVALUE",
    ):
        assert secret not in blob
    assert "keep-me" in row["args_redacted_summary"]
    assert len(row["args_redacted_summary"]) <= action_receipts.MAX_ARGS_SUMMARY_CHARS


def test_secret_shaped_values_inside_command_strings_are_redacted(ledger):
    _record(
        ledger,
        args={
            "command": (
                "curl -H 'Authorization: Basic BASIC-CREDENTIAL' "
                "--token TOKEN-CREDENTIAL --password=PASSWORD-CREDENTIAL "
                "https://user:URL-CREDENTIAL@example.test"
            ),
            "safe": "keep-me",
        },
    )

    summary = ledger.read_all()[0]["args_redacted_summary"]
    for secret in (
        "BASIC-CREDENTIAL",
        "TOKEN-CREDENTIAL",
        "PASSWORD-CREDENTIAL",
        "URL-CREDENTIAL",
    ):
        assert secret not in summary
    assert "keep-me" in summary


def test_receipt_summary_uses_canonical_redactor_for_command_credentials(ledger):
    secrets = {
        "github": "ghp_" + "A" * 30,
        "aws": "A" * 40,
        "db": "D" * 20,
        "json": "sk-" + "J" * 30,
    }
    _record(
        ledger,
        args={
            "command": (
                f"export GITHUB_TOKEN={secrets['github']} "
                f"AWS_SECRET_ACCESS_KEY={secrets['aws']} "
                f"postgresql://user:{secrets['db']}@db.example/test "
                f"--payload '{{\"api_key\": \"{secrets['json']}\"}}'"
            )
        },
    )

    summary = ledger.read_all()[0]["args_redacted_summary"]
    for secret in secrets.values():
        assert secret not in summary


@pytest.mark.parametrize(
    ("tool_name", "args", "body"),
    [
        ("write_file", {"path": "/tmp/x", "content": "PRIVATE-FILE-BODY"}, "PRIVATE-FILE-BODY"),
        ("memory", {"content": "PRIVATE-MESSAGE-BODY"}, "PRIVATE-MESSAGE-BODY"),
        ("send_message", {"message": "PRIVATE-SEND-BODY"}, "PRIVATE-SEND-BODY"),
        ("execute_code", {"code": "PRIVATE-CODE-BODY"}, "PRIVATE-CODE-BODY"),
        ("patch", {"old_string": "PRIVATE-OLD-BODY", "new_string": "PRIVATE-NEW-BODY"}, "PRIVATE-OLD-BODY"),
    ],
)
def test_raw_body_fields_are_hash_only_in_summary(ledger, tool_name, args, body):
    _record(ledger, tool_name=tool_name, args=args)

    summary = ledger.read_all()[0]["args_redacted_summary"]
    assert body not in summary
    assert "sha256" in summary


def test_raw_output_is_hashed_not_stored(ledger):
    _record(ledger, output="SENSITIVE-OUTPUT-BODY")

    row = ledger.read_all()[0]
    blob = "\n".join(str(v) for v in row.values())
    assert "SENSITIVE-OUTPUT-BODY" not in blob
    assert len(row["output_hash"]) == 64
    assert row["output_bytes"] == len("SENSITIVE-OUTPUT-BODY".encode("utf-8"))


def test_hashes_are_stable_for_canonically_equivalent_args(tmp_path):
    a = action_receipts.ActionReceiptLedger(db_path=tmp_path / "a.db")
    b = action_receipts.ActionReceiptLedger(db_path=tmp_path / "b.db")
    _record(a, args={"x": 1, "y": {"p": 2, "q": 3}})
    _record(b, args={"y": {"q": 3, "p": 2}, "x": 1})

    assert a.read_all()[0]["args_hash"] == b.read_all()[0]["args_hash"]


def test_module_exposes_no_mutation_api():
    import inspect

    public = {
        name
        for name, _ in inspect.getmembers(action_receipts.ActionReceiptLedger)
        if not name.startswith("_")
    }
    assert not {n for n in public if any(w in n for w in ("update", "delete", "vacuum", "prune"))}

    # Inspect executable code only — prose in docstrings/comments may well
    # mention the words this module refuses to implement.
    import ast

    tree = ast.parse(inspect.getsource(action_receipts))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node) is not None:
                node.body = node.body[1:]
    code = ast.unparse(tree).upper()
    # The only UPDATE/DELETE tokens allowed are inside the guard triggers.
    for statement in ("UPDATE ACTION_RECEIPTS SET", "DELETE FROM ACTION_RECEIPTS", "VACUUM"):
        assert statement not in code
