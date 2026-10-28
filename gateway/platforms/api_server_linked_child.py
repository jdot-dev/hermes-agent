"""Linked-child dispatch v1.

A request whose body is exactly ``input``, ``session_id``, ``origin``, and
``child`` runs ``child.goal`` in a new hidden session. The parent session id
stays on the run record. The original ``input`` is not written here; passive
history owns that transcript.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path

from aiohttp import web

_ID = re.compile(r"[A-Za-z0-9._-]{1,128}\Z")
_BY_CHILD: dict[str, dict[str, str]] = {}


def linked_child_feature() -> dict:
    return {
        "version": 1,
        "supported": True,
        "separate_child_goal": True,
        "origin_sources": ["fresh", "passive_receipt"],
    }


def overlay_status(status: dict) -> None:
    """Publish the parent session on a run whose agent is bound to the child."""
    child = status.get("child_session_id") or status.get("session_id")
    info = _BY_CHILD.get(child) if isinstance(child, str) else None
    if info is None:
        return
    status["session_id"] = info["parent_session_id"]
    status["child_session_id"] = info["child_session_id"]


def _reject(code: str, status: int, *, retryable: bool = False) -> web.Response:
    return web.json_response({"error": code, "retryable": retryable}, status=status)


def _identifier(value) -> str | None:
    if isinstance(value, str) and _ID.fullmatch(value):
        return value
    return None


def child_session_id(parent: str, correlation_id: str) -> str:
    digest = hashlib.sha256(f"{parent}\n{correlation_id}".encode()).hexdigest()
    return "child" + digest[:40]


def classify(body):
    """Return a spec, an error code, or None when this is an ordinary run."""
    if not isinstance(body, dict) or set(body) != {"input", "session_id", "origin", "child"}:
        return None
    parent = body.get("session_id")
    origin = body.get("origin")
    child = body.get("child")
    if not isinstance(parent, str) or not 1 <= len(parent) <= 256:
        return "invalid_input"
    if not isinstance(body.get("input"), str) or not body["input"].strip():
        return "invalid_input"
    if not isinstance(origin, dict) or not {"event_id", "origin_turn_id"} <= set(origin) <= {
        "event_id", "origin_turn_id", "receipt_id"
    }:
        return "invalid_input"
    event_id = _identifier(origin.get("event_id"))
    origin_turn_id = _identifier(origin.get("origin_turn_id"))
    if event_id is None or origin_turn_id is None:
        return "invalid_input"
    receipt_id = origin.get("receipt_id")
    if receipt_id is not None and (type(receipt_id) is not int or isinstance(receipt_id, bool) or receipt_id < 1):
        return "invalid_input"
    if not isinstance(child, dict):
        return "invalid_input"
    allowed = {"goal", "context", "correlation_id", "worker"}
    if set(child) - allowed or not {"goal", "context", "correlation_id"} <= set(child):
        return "attachment_operation_unsupported" if "attachments" in child else "invalid_input"
    if child.get("worker") not in (None, "hermes"):
        return "child_denied"
    goal = child.get("goal")
    context = child.get("context")
    correlation_id = _identifier(child.get("correlation_id"))
    if not isinstance(goal, str) or not goal.strip() or len(goal) > 16000:
        return "invalid_input"
    if not isinstance(context, str) or len(context) > 32000 or correlation_id is None:
        return "invalid_input"
    return {
        "parent_session_id": parent,
        "event_id": event_id,
        "origin_turn_id": origin_turn_id,
        "receipt_id": receipt_id,
        "goal": goal,
        "context": context,
        "correlation_id": correlation_id,
    }


async def prepare_linked_child(adapter, body):
    spec = classify(body)
    if spec is None:
        return None
    if isinstance(spec, str):
        status = 404 if spec == "target_missing" else 409
        return _reject(spec, status)
    db = adapter._ensure_session_db()
    if db is None:
        return _reject("store_unavailable", 503, retryable=True)
    from hermes_constants import get_hermes_home
    from gateway.platforms.api_server import _api_request_profile
    from gateway.platforms.api_server_passive_history import receipt_matches

    home = Path(get_hermes_home())
    profile = _api_request_profile.get() or "default"
    parent = spec["parent_session_id"]
    try:
        session = await asyncio.to_thread(db.get_session, parent)
    except Exception:
        return _reject("store_unavailable", 503, retryable=True)
    if not session:
        return _reject("target_missing", 404)
    if spec["receipt_id"] is not None:
        try:
            matched = await asyncio.to_thread(
                receipt_matches,
                home,
                profile,
                session_id=parent,
                event_id=spec["event_id"],
                origin_turn_id=spec["origin_turn_id"],
                receipt_id=spec["receipt_id"],
            )
        except Exception:
            return _reject("store_unavailable", 503, retryable=True)
        if not matched:
            return _reject("child_denied", 409)
    child = child_session_id(parent, spec["correlation_id"])

    def create() -> None:
        db.create_session(child, "api_server", parent_session_id=parent, title="Linked child")
        db.set_session_hidden(child, True)

    try:
        await asyncio.to_thread(create)
    except Exception:
        return _reject("store_unavailable", 503, retryable=True)
    _BY_CHILD[child] = {"parent_session_id": parent, "child_session_id": child}
    if len(_BY_CHILD) > 1024:
        _BY_CHILD.pop(next(iter(_BY_CHILD)))
    rewritten = {"input": spec["goal"], "session_id": child}
    if spec["context"]:
        rewritten["instructions"] = spec["context"]
    return rewritten
