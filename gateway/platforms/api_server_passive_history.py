"""Passive history v1 for an already-authenticated local API server.

Dialogue commits land in the canonical session transcript. Attachment and
receipt bookkeeping stays beside that database. This module has no execution
route and does not choose an actor.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import threading
from pathlib import Path

from aiohttp import web

_ID = re.compile(r"[A-Za-z0-9._-]{1,128}\Z")
_OPERATIONS = frozenset({"attach", "snapshot", "commit", "reconcile", "detach"})
_LIMITS = {
    "max_request_bytes": 160 * 1024,
    "max_message_bytes": 64 * 1024,
    "max_messages": 2,
    "max_snapshot_messages": 20,
    "max_snapshot_bytes": 32 * 1024,
}
_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def capabilities_document(store_id: str) -> dict:
    return {
        "version": 1,
        "passive_only": True,
        "origin_adoption": True,
        "operations": ["attach", "snapshot", "commit", "reconcile", "detach"],
        "restart_requires_reattach": True,
        "store_id": store_id,
        **_LIMITS,
    }


def _error(code: str, status: int, *, retryable: bool = False) -> web.Response:
    return web.json_response({"error": code, "retryable": retryable}, status=status)


def _identifier(value) -> str | None:
    if isinstance(value, str) and _ID.fullmatch(value):
        return value
    return None


def _session_lock(profile: str, session_id: str) -> threading.Lock:
    key = (profile, session_id)
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock


def _ledger_path(home: Path) -> Path:
    return Path(home) / "state" / "talk-passive.sqlite"


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5)
    try:
        path.chmod(0o600)
    except OSError:
        pass
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS attachments (
            profile TEXT NOT NULL,
            tab_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            attachment_id TEXT NOT NULL,
            generation INTEGER NOT NULL,
            PRIMARY KEY (profile, tab_id)
        );
        CREATE TABLE IF NOT EXISTS receipts (
            profile TEXT NOT NULL,
            event_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            origin_turn_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            message_ids TEXT NOT NULL,
            body_hash TEXT NOT NULL,
            revision INTEGER NOT NULL,
            PRIMARY KEY (profile, event_id)
        );
        CREATE TABLE IF NOT EXISTS revision_counter (
            profile TEXT PRIMARY KEY,
            value INTEGER NOT NULL
        );
        """
    )
    return conn


def _body_hash(session_id: str, origin_turn_id: str, messages: list) -> str:
    payload = {"session_id": session_id, "origin_turn_id": origin_turn_id, "messages": messages}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _receipt_wire(row, *, replayed: bool) -> dict:
    return {
        "producer": "passive.ingress.v1",
        "event_id": row["event_id"],
        "origin_turn_id": row["origin_turn_id"],
        "conversation_id": row["conversation_id"],
        "session_id": row["session_id"],
        "message_ids": json.loads(row["message_ids"]),
        "revision": row["revision"],
        "replayed": replayed,
    }


def _dialogue(messages) -> list[dict] | None:
    if not isinstance(messages, list) or not 1 <= len(messages) <= _LIMITS["max_messages"]:
        return None
    rows = []
    for row in messages:
        if not isinstance(row, dict) or set(row) != {"role", "content"}:
            return None
        if row["role"] not in {"user", "assistant"} or not isinstance(row["content"], str):
            return None
        if not row["content"].strip():
            return None
        if len(row["content"].encode("utf-8")) > _LIMITS["max_message_bytes"]:
            return None
        rows.append({"role": row["role"], "content": row["content"]})
    if len(rows) == 2 and [row["role"] for row in rows] != ["user", "assistant"]:
        return None
    return rows


def _snapshot(db, session_id: str, store_id: str) -> dict:
    rows = db.get_messages(session_id, latest=True, limit=200)
    candidates = []
    for row in rows:
        content = row.get("content")
        if row.get("role") not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
            continue
        if row.get("display_kind") == "hidden":
            continue
        message_id = row.get("id")
        if type(message_id) is not int or message_id <= 0:
            continue
        candidates.append({"id": message_id, "role": row["role"], "content": content})
    chosen = []
    used = 0
    truncated = False
    for row in reversed(candidates):
        size = len(row["content"].encode("utf-8"))
        if (
            len(chosen) >= _LIMITS["max_snapshot_messages"]
            or used + size > _LIMITS["max_snapshot_bytes"]
        ):
            truncated = True
            break
        chosen.append(row)
        used += size
    chosen.reverse()
    if len(candidates) > len(chosen):
        truncated = True
    return {
        "conversation_id": session_id,
        "session_id": session_id,
        "messages": chosen,
        "truncated": truncated,
        "capabilities": capabilities_document(store_id),
    }


def _stamped_ids(db, session_id: str, event_id: str, count: int) -> list[int] | None:
    rows = db._read_all(
        "SELECT id, platform_message_id FROM messages WHERE session_id = ? AND platform_message_id LIKE ?",
        (session_id, f"talk:{event_id}:%"),
    )
    found = {}
    for row in rows:
        marker = row[1]
        if not isinstance(marker, str) or not marker.startswith(f"talk:{event_id}:"):
            continue
        index = marker.rsplit(":", 1)[-1]
        if index.isdigit():
            found[int(index)] = int(row[0])
    if [index for index in range(count) if index not in found]:
        return None
    return [found[index] for index in range(count)]


def _next_revision(conn, profile: str) -> int:
    row = conn.execute("SELECT value FROM revision_counter WHERE profile = ?", (profile,)).fetchone()
    revision = 1 if row is None else int(row["value"]) + 1
    conn.execute(
        "INSERT INTO revision_counter (profile, value) VALUES (?, ?) "
        "ON CONFLICT(profile) DO UPDATE SET value = excluded.value",
        (profile, revision),
    )
    return revision


def receipt_matches(home: Path, profile: str, *, session_id: str, event_id: str, origin_turn_id: str, receipt_id: int) -> bool:
    path = _ledger_path(home)
    if not path.is_file():
        return False
    conn = _connect(path)
    try:
        row = conn.execute(
            "SELECT origin_turn_id, session_id, revision FROM receipts WHERE profile = ? AND event_id = ?",
            (profile, event_id),
        ).fetchone()
    finally:
        conn.close()
    return bool(
        row
        and row["session_id"] == session_id
        and row["origin_turn_id"] == origin_turn_id
        and int(row["revision"]) == receipt_id
    )


def _dispatch(db, home: Path, profile: str, store_id: str, operation: str, body) -> web.Response:
    if operation == "capabilities":
        return web.json_response(capabilities_document(store_id))
    if not isinstance(body, dict):
        return _error("invalid_input", 400)
    session_id = body.get("session_id")
    if not isinstance(session_id, str) or not 1 <= len(session_id) <= 256:
        return _error("invalid_input", 400)
    if db.get_session(session_id) is None:
        return _error("target_missing", 404)
    path = _ledger_path(home)
    if operation == "reconcile":
        event_id = _identifier(body.get("event_id"))
        if event_id is None or set(body) != {"session_id", "event_id"}:
            return _error("invalid_input", 400)
        conn = _connect(path)
        try:
            row = conn.execute(
                "SELECT * FROM receipts WHERE profile = ? AND event_id = ?",
                (profile, event_id),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return web.json_response({"profile": profile, "status": "unknown", "receipt": None})
        if row["session_id"] != session_id:
            return _error("event_conflict", 409)
        return web.json_response(
            {"profile": profile, "status": "saved", "receipt": _receipt_wire(row, replayed=False)}
        )
    if operation == "attach":
        tab_id = _identifier(body.get("tab_id"))
        if tab_id is None or set(body) != {"tab_id", "session_id"}:
            return _error("invalid_input", 400)
        conn = _connect(path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            current = conn.execute(
                "SELECT generation FROM attachments WHERE profile = ? AND tab_id = ?",
                (profile, tab_id),
            ).fetchone()
            generation = 1 if current is None else int(current["generation"]) + 1
            attachment_id = "a" + hashlib.sha256(f"{profile}:{tab_id}:{generation}".encode()).hexdigest()[:32]
            conn.execute(
                "INSERT INTO attachments (profile, tab_id, session_id, attachment_id, generation) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(profile, tab_id) DO UPDATE SET "
                "session_id = excluded.session_id, attachment_id = excluded.attachment_id, "
                "generation = excluded.generation",
                (profile, tab_id, session_id, attachment_id, generation),
            )
            conn.commit()
        finally:
            conn.close()
        return web.json_response({
            "profile": profile,
            "tab_id": tab_id,
            "session_id": session_id,
            "attachment_id": attachment_id,
            "generation": generation,
            "snapshot": _snapshot(db, session_id, store_id),
        })
    tab_id = _identifier(body.get("tab_id"))
    attachment_id = _identifier(body.get("attachment_id"))
    generation = body.get("generation")
    if tab_id is None or attachment_id is None or type(generation) is not int or generation < 1:
        return _error("invalid_input", 400)
    conn = _connect(path)
    try:
        authority = conn.execute(
            "SELECT * FROM attachments WHERE profile = ? AND tab_id = ?",
            (profile, tab_id),
        ).fetchone()
    finally:
        conn.close()
    if (
        authority is None
        or authority["session_id"] != session_id
        or authority["attachment_id"] != attachment_id
        or int(authority["generation"]) != generation
    ):
        return _error("stale_attachment", 409)
    if operation == "detach":
        if set(body) != {"tab_id", "attachment_id", "generation", "session_id"}:
            return _error("invalid_input", 400)
        conn = _connect(path)
        try:
            conn.execute("DELETE FROM attachments WHERE profile = ? AND tab_id = ?", (profile, tab_id))
            conn.commit()
        finally:
            conn.close()
        return web.json_response({"profile": profile, "status": "detached"})
    if operation == "snapshot":
        if set(body) != {"tab_id", "attachment_id", "generation", "session_id"}:
            return _error("invalid_input", 400)
        return web.json_response({"profile": profile, **_snapshot(db, session_id, store_id)})
    if operation != "commit":
        return _error("unsupported", 404)
    event_id = _identifier(body.get("event_id"))
    origin_turn_id = _identifier(body.get("origin_turn_id"))
    messages = _dialogue(body.get("messages"))
    expected = {"tab_id", "attachment_id", "generation", "session_id", "event_id", "origin_turn_id", "messages"}
    if event_id is None or origin_turn_id is None or messages is None or set(body) != expected:
        return _error("invalid_input", 400)
    digest = _body_hash(session_id, origin_turn_id, messages)
    lock = _session_lock(profile, session_id)
    if not lock.acquire(blocking=False):
        return _error("busy", 409, retryable=True)
    try:
        conn = _connect(path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM receipts WHERE profile = ? AND event_id = ?",
                (profile, event_id),
            ).fetchone()
            if existing is not None:
                conn.rollback()
                if existing["body_hash"] != digest or existing["session_id"] != session_id:
                    return _error("event_conflict", 409)
                return web.json_response({
                    "profile": profile,
                    "status": "already_saved",
                    "receipt": _receipt_wire(existing, replayed=True),
                })
            stamped = _stamped_ids(db, session_id, event_id, len(messages))
            if stamped is None:
                stamped = []
                for index, message in enumerate(messages):
                    stamped.append(db.append_message(
                        session_id,
                        message["role"],
                        message["content"],
                        platform_message_id=f"talk:{event_id}:{index}",
                    ))
            revision = _next_revision(conn, profile)
            conn.execute(
                "INSERT INTO receipts (profile, event_id, session_id, origin_turn_id, conversation_id, "
                "message_ids, body_hash, revision) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    profile, event_id, session_id, origin_turn_id, session_id,
                    json.dumps(stamped), digest, revision,
                ),
            )
            stored = conn.execute(
                "SELECT * FROM receipts WHERE profile = ? AND event_id = ?",
                (profile, event_id),
            ).fetchone()
            conn.commit()
        except Exception:
            with __import__("contextlib").suppress(Exception):
                conn.rollback()
            return _error("store_unavailable", 503, retryable=True)
        finally:
            conn.close()
    finally:
        lock.release()
    return web.json_response({
        "profile": profile,
        "status": "saved",
        "receipt": _receipt_wire(stored, replayed=False),
    })


def _http_routes(adapter) -> list[tuple]:
    return [
        ("GET", "/v1/passive-history/{operation}", adapter._handle_passive_history),
        ("POST", "/v1/passive-history/{operation}", adapter._handle_passive_history),
    ]


async def handle(adapter, request: web.Request) -> web.Response:
    operation = request.match_info.get("operation", "")
    if operation not in _OPERATIONS | {"capabilities"}:
        return _error("unsupported", 404)
    if operation == "capabilities" and request.method != "GET":
        return _error("unsupported", 405)
    if operation != "capabilities" and request.method != "POST":
        return _error("unsupported", 405)
    from hermes_constants import get_hermes_home
    from hermes_state_store_identity import get_store_id

    profile = "default"
    try:
        from gateway.platforms.api_server import _api_request_profile
        profile = _api_request_profile.get() or "default"
    except Exception:
        profile = "default"
    home = Path(get_hermes_home())
    db = adapter._ensure_session_db()
    if db is None:
        return _error("store_unavailable", 503, retryable=True)
    try:
        store_id = await asyncio.to_thread(get_store_id, db.db_path)
    except Exception:
        return _error("store_unavailable", 503, retryable=True)
    body = None
    if request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            return _error("invalid_input", 400)
    try:
        return await asyncio.to_thread(_dispatch, db, home, profile, store_id, operation, body)
    except Exception:
        return _error("store_unavailable", 503, retryable=True)
