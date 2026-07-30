"""Atomic persistent claims for Phase 2 mutating logical operations."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from hermes_constants import get_hermes_home

_DB_LOCK = threading.Lock()


def default_db_path() -> Path:
    return get_hermes_home() / "phase2_enforcement.db"


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


class MutationClaimStore:
    """Own the durable first-writer claim for a mutating idempotency key."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_db_path()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mutation_claims (
                idempotency_key TEXT PRIMARY KEY,
                graph_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                attempt_id TEXT NOT NULL,
                fence INTEGER NOT NULL,
                tool_name TEXT NOT NULL,
                args_hash TEXT NOT NULL,
                claimed_utc TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS mutation_claims_no_update
            BEFORE UPDATE ON mutation_claims
            BEGIN
                SELECT RAISE(ABORT, 'mutation_claims is append-only');
            END
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS mutation_claims_no_delete
            BEFORE DELETE ON mutation_claims
            BEGIN
                SELECT RAISE(ABORT, 'mutation_claims is append-only');
            END
            """
        )
        return conn

    def try_claim(
        self,
        envelope: Mapping[str, Any],
        *,
        tool_name: str,
        args: Mapping[str, Any],
    ) -> bool:
        """Atomically claim one logical mutating operation; duplicate keys lose."""

        row = (
            str(envelope["idempotency_key"]),
            str(envelope["graph_id"]),
            str(envelope["node_id"]),
            str(envelope["attempt_id"]),
            int(envelope["fence"]),
            str(tool_name),
            _canonical_hash(args),
            datetime.now(timezone.utc).isoformat(),
        )
        with _DB_LOCK:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                inserted = conn.execute(
                    """
                    INSERT OR IGNORE INTO mutation_claims(
                        idempotency_key, graph_id, node_id, attempt_id, fence,
                        tool_name, args_hash, claimed_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    row,
                ).rowcount
                conn.execute("COMMIT")
                return inserted == 1
            except Exception:
                try:
                    conn.execute("ROLLBACK")
                except Exception:
                    pass
                raise
            finally:
                conn.close()

    def read_all(self) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(row) for row in conn.execute("SELECT * FROM mutation_claims ORDER BY rowid")]
        finally:
            conn.close()


__all__ = ["MutationClaimStore", "default_db_path"]
