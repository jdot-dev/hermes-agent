"""Opaque id of one canonical Hermes state database.

The id lives in ``state_meta`` so it follows the file. A replaced database does
not keep the previous id. Callers must not invent a stand-in when this read
fails.
"""

from __future__ import annotations

import secrets
import sqlite3
from pathlib import Path

_KEY = "hermes.talk.store_id"


def _path_of(db) -> Path:
    if isinstance(db, (str, Path)):
        return Path(db)
    for name in ("db_path", "database_path", "path"):
        value = getattr(db, name, None)
        if isinstance(value, (str, Path)) and str(value):
            return Path(value)
    raise ValueError("store path is not available")


def get_store_id(db) -> str:
    """Return the stable id stored inside this database, creating it once."""
    path = _path_of(db)
    if not path.is_file():
        raise ValueError("canonical store is not available")
    conn = sqlite3.connect(path, timeout=5)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT value FROM state_meta WHERE key = ?", (_KEY,)
        ).fetchone()
        if row and isinstance(row[0], str) and row[0]:
            conn.rollback()
            return row[0]
        store_id = secrets.token_hex(16)
        conn.execute(
            "INSERT INTO state_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value "
            "WHERE state_meta.value IS NULL OR state_meta.value = ''",
            (_KEY, store_id),
        )
        stored = conn.execute(
            "SELECT value FROM state_meta WHERE key = ?", (_KEY,)
        ).fetchone()
        conn.commit()
    except sqlite3.Error as exc:
        with __import__("contextlib").suppress(sqlite3.Error):
            conn.rollback()
        raise ValueError("canonical store id is not available") from exc
    finally:
        conn.close()
    if not stored or not isinstance(stored[0], str) or not stored[0]:
        raise ValueError("canonical store id is not available")
    return stored[0]
