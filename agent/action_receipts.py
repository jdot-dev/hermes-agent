"""Append-only general action receipt ledger.

Phase 0 telemetry: one receipt per actually-executed tool call, written to a
dedicated SQLite database that has no update, delete, or vacuum path. Rows are
chained by hash so tampering is detectable after the fact.

The ledger is **default-off**. It is reachable only when the user opts in with

.. code-block:: yaml

    observability:
      action_receipts:
        enabled: true

in ``config.yaml``. With the key absent or false — and on any config read
error — :func:`is_enabled` returns ``False`` and nothing here opens a
database, builds an envelope, or touches the filesystem.

Conventions (WAL, busy timeout, module lock, schema table) mirror
``agent/verification_evidence.py`` so the two ledgers stay idiomatic.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agent.task_envelope import (
    TaskEnvelope,
    build_shadow_envelope,
    canonical_hash,
    canonical_json,
    envelope_hash,
)
from hermes_constants import get_hermes_home


RECEIPT_SCHEMA_VERSION = 1

MAX_ARGS_SUMMARY_CHARS = 1000

_DB_LOCK = threading.Lock()
_REDACTED = "[redacted]"
_BODY_FIELD_KEYS = frozenset(
    {
        "content",
        "code",
        "file_content",
        "information",
        "message",
        "new_string",
        "old_string",
        "patch",
        "prompt",
        "text",
    }
)

#: Argument keys whose values are treated as secrets wherever they appear,
#: at any nesting depth.
_SECRET_KEY_RE = re.compile(
    r"api[-_ ]?key|apikey|token|secret|password|passwd|pwd|auth|credential|"
    r"bearer|session[-_ ]?id_secret|private[-_ ]?key",
    re.IGNORECASE,
)

#: Values that look like credentials even under an innocuous key.
_SECRET_VALUE_RE = re.compile(
    r"(?:bearer\s+\S+)|"
    r"(?:authorization\s*[:=]\s*(?:basic|bearer)?\s*\S+)|"
    r"(?:--?(?:api[-_]?key|token|secret|password|passwd|pwd|auth|credential)(?:=|\s+)\S+)|"
    r"(?:https?://[^\s/@:]+:[^\s/@]+@)|"
    r"(?:\b(?:sk|pk|ghp|gho|ghu|ghs|xox[baprs])-[A-Za-z0-9_\-]{8,})|"
    r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----)",
    re.IGNORECASE,
)


def _load_config() -> Any:
    """Read the user config, read-only.

    Isolated behind a module-level function so tests inject a config dict
    instead of touching the operator's real ``config.yaml``.
    """
    from hermes_cli.config import load_config_readonly

    return load_config_readonly()


def is_enabled() -> bool:
    """True only when ``observability.action_receipts.enabled`` is exactly true.

    Any missing key, non-mapping section, or config read failure yields
    ``False``: telemetry never fails open into being on.
    """
    try:
        config = _load_config()
        if not isinstance(config, dict):
            return False
        section = config.get("observability")
        if not isinstance(section, dict):
            return False
        receipts = section.get("action_receipts")
        if not isinstance(receipts, dict):
            return False
        return receipts.get("enabled") is True
    except Exception:
        return False


def default_db_path() -> Path:
    """The ledger's own database — never any existing Hermes store."""
    return get_hermes_home() / "action_receipts.db"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _body_fingerprint(value: Any) -> dict[str, Any]:
    """Represent quarantine-class bodies without retaining their contents."""
    try:
        text = value if isinstance(value, str) else canonical_json(value)
    except Exception:
        text = str(value)
    encoded = text.encode("utf-8")
    return {"bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}


def _redact(value: Any, *, depth: int = 0) -> Any:
    """Recursively strip secret-shaped keys and values.

    Redaction happens *before* anything is written, so a secret never reaches
    the database in the first place.
    """
    if depth > 12:
        return "[truncated]"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _SECRET_KEY_RE.search(key_text):
                out[key_text] = _REDACTED
            elif key_text.lower() in _BODY_FIELD_KEYS:
                out[key_text] = _body_fingerprint(item)
            else:
                out[key_text] = _redact(item, depth=depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [_redact(item, depth=depth + 1) for item in value]
    if isinstance(value, str):
        from agent.redact import redact_sensitive_text

        redacted = redact_sensitive_text(
            value,
            force=True,
            redact_url_credentials=True,
        )
        return _SECRET_VALUE_RE.sub(_REDACTED, redacted)
    return value


def redacted_args_summary(args: Any) -> str:
    """A bounded, secrets-stripped rendering of the call arguments."""
    try:
        summary = canonical_json(_redact(args if args is not None else {}))
    except Exception:
        summary = "[unserializable]"
    if len(summary) > MAX_ARGS_SUMMARY_CHARS:
        return summary[: MAX_ARGS_SUMMARY_CHARS - 3] + "..."
    return summary


def _output_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    try:
        return canonical_json(output)
    except Exception:
        return str(output)


class ActionReceiptLedger:
    """Append-only, hash-chained ledger of executed actions.

    There is deliberately no update, delete, or vacuum method here, and the
    table carries ``BEFORE UPDATE`` / ``BEFORE DELETE`` triggers that abort —
    so a row, once written, is fixed for as long as the file is intact.
    """

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_db_path()

    # ── connection / schema ────────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        self._ensure_schema(conn)
        return conn

    def _connect_readonly(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            f"file:{self.db_path}?mode=ro",
            uri=True,
            isolation_level=None,
        )
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS action_receipts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                receipt_version INTEGER NOT NULL,
                receipt_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                kind TEXT NOT NULL,
                action_id TEXT NOT NULL,
                session_id TEXT,
                task_id TEXT,
                tool_call_id TEXT,
                envelope_id TEXT,
                actor_lane TEXT NOT NULL,
                execution_surface TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                exit_status TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                cwd TEXT,
                args_hash TEXT NOT NULL,
                args_redacted_summary TEXT NOT NULL,
                output_hash TEXT,
                output_bytes INTEGER NOT NULL,
                envelope_hash TEXT,
                envelope_json TEXT,
                prev_receipt_hash TEXT,
                receipt_hash TEXT NOT NULL
            )
            """
        )
        # Append-only guards. RAISE(ABORT) surfaces to Python as
        # sqlite3.IntegrityError, so a tamper attempt is loud, not silent.
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS action_receipts_no_update
            BEFORE UPDATE ON action_receipts
            BEGIN
                SELECT RAISE(ABORT, 'action_receipts is append-only');
            END
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS action_receipts_no_delete
            BEFORE DELETE ON action_receipts
            BEGIN
                SELECT RAISE(ABORT, 'action_receipts is append-only');
            END
            """
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(RECEIPT_SCHEMA_VERSION),),
        )

    # ── hashing ────────────────────────────────────────────────────────
    @staticmethod
    def _chain_payload(row: Any) -> dict[str, Any]:
        """The exact field set covered by ``receipt_hash``."""
        keys = (
            "receipt_version",
            "receipt_id",
            "created_at",
            "kind",
            "action_id",
            "session_id",
            "task_id",
            "tool_call_id",
            "envelope_id",
            "actor_lane",
            "execution_surface",
            "tool_name",
            "exit_status",
            "duration_ms",
            "cwd",
            "args_hash",
            "args_redacted_summary",
            "output_hash",
            "output_bytes",
            "envelope_hash",
            "envelope_json",
            "prev_receipt_hash",
        )
        return {key: row[key] for key in keys}

    # ── write ──────────────────────────────────────────────────────────
    def record_receipt(
        self,
        *,
        tool_name: str,
        args: Any = None,
        output: Any = None,
        exit_status: str = "ok",
        duration_ms: int = 0,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        cwd: Optional[str] = None,
        kind: str = "tool_exec",
        envelope: Optional[TaskEnvelope] = None,
        receipt_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> str:
        """Append exactly one receipt and return its ``receipt_id``.

        The tail read, hash computation, and insert all happen inside one
        ``BEGIN IMMEDIATE`` transaction under a process-wide lock, so
        concurrent callers extend a single chain instead of forking it.
        """
        if envelope is None:
            envelope = build_shadow_envelope(
                tool_name=tool_name,
                args=args,
                cwd=cwd,
                session_id=session_id,
                task_id=task_id,
                tool_call_id=tool_call_id,
            )

        output_text = _output_text(output)
        row: dict[str, Any] = {
            "receipt_version": RECEIPT_SCHEMA_VERSION,
            "receipt_id": receipt_id or f"r-{uuid.uuid4().hex}",
            "created_at": created_at or _utc_now(),
            "kind": kind,
            "action_id": envelope.action_id,
            "session_id": session_id,
            "task_id": task_id,
            "tool_call_id": tool_call_id,
            "envelope_id": envelope.envelope_id,
            "actor_lane": envelope.lane,
            "execution_surface": envelope.execution_surface,
            "tool_name": str(tool_name),
            "exit_status": str(exit_status),
            "duration_ms": int(duration_ms),
            "cwd": cwd,
            # §7: args_hash covers the canonical arguments as executed; only
            # the summary is redacted, and the raw args are never stored.
            "args_hash": canonical_hash(args if args is not None else {}),
            "args_redacted_summary": redacted_args_summary(args),
            "output_hash": (
                hashlib.sha256(output_text.encode("utf-8")).hexdigest()
                if output is not None
                else None
            ),
            "output_bytes": len(output_text.encode("utf-8")),
            "envelope_hash": envelope_hash(envelope),
            "envelope_json": canonical_json(envelope.to_dict()),
        }

        with _DB_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                tail = conn.execute(
                    "SELECT receipt_hash FROM action_receipts ORDER BY id DESC LIMIT 1"
                ).fetchone()
                row["prev_receipt_hash"] = tail["receipt_hash"] if tail else None
                row["receipt_hash"] = canonical_hash(self._chain_payload(row))
                columns = list(row.keys())
                conn.execute(
                    f"INSERT INTO action_receipts({', '.join(columns)}) "
                    f"VALUES ({', '.join('?' for _ in columns)})",
                    [row[c] for c in columns],
                )
                conn.execute("COMMIT")
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

        return str(row["receipt_id"])

    # ── read ───────────────────────────────────────────────────────────
    def read_all(self) -> list[dict[str, Any]]:
        """Every receipt, oldest first. Read-only."""
        if not self.db_path.exists():
            return []
        conn = self._connect_readonly()
        try:
            rows = conn.execute(
                "SELECT * FROM action_receipts ORDER BY id ASC"
            ).fetchall()
        finally:
            conn.close()
        return [dict(row) for row in rows]

    def verify_chain(self) -> list[str]:
        """Return one message per chain defect; empty list means intact."""
        problems: list[str] = []
        previous: Optional[str] = None
        for row in self.read_all():
            expected = canonical_hash(self._chain_payload(row))
            if row["receipt_hash"] != expected:
                problems.append(f"receipt {row['receipt_id']}: content hash mismatch")
            if row["prev_receipt_hash"] != previous:
                problems.append(f"receipt {row['receipt_id']}: broken chain link")
            previous = row["receipt_hash"]
        return problems


__all__ = [
    "ActionReceiptLedger",
    "MAX_ARGS_SUMMARY_CHARS",
    "RECEIPT_SCHEMA_VERSION",
    "default_db_path",
    "is_enabled",
    "redacted_args_summary",
]
