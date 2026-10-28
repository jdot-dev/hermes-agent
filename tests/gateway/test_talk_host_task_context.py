"""Host support the Talk target catalog requires: verified actor, store id, passive history."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.platforms.api_server_linked_child import (
    _BY_CHILD,
    child_session_id,
    classify,
    overlay_status,
)
from gateway.platforms.api_server_passive_history import _dispatch, capabilities_document
from hermes_cli.dashboard_task_context import resolve_dashboard_task_context
from hermes_state_store_identity import get_store_id

sys.path.insert(0, "/home/jack/.hermes/plugins/hermes-talk")
from talk_passive import HistoryCapabilities, HistorySnapshot


class _Headers(dict):
    def get(self, key, default=None):
        return super().get(key, default)


class _Db:
    def __init__(self):
        self.sessions = {"task-a"}
        self.messages = []
        self._reads = []

    def get_session(self, session_id):
        if session_id in self.sessions:
            return {"id": session_id}
        return None

    def get_messages(self, session_id, **_kwargs):
        return [row for row in self.messages if row["session_id"] == session_id]

    def append_message(self, session_id, role, content=None, **kwargs):
        message_id = len(self.messages) + 1
        self.messages.append({
            "id": message_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "platform_message_id": kwargs.get("platform_message_id"),
        })
        return message_id

    def _read_all(self, _sql, params):
        session_id, prefix = params
        needle = prefix[:-1]
        return [
            (row["id"], row["platform_message_id"])
            for row in self.messages
            if row["session_id"] == session_id and str(row.get("platform_message_id") or "").startswith(needle)
        ]


def _state_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()
    conn.close()


def _body(response) -> dict:
    return json.loads(response.body)


def test_store_id_follows_the_database_file(tmp_path):
    first = tmp_path / "one.sqlite"
    second = tmp_path / "two.sqlite"
    _state_db(first)
    _state_db(second)
    assert get_store_id(first) == get_store_id(SimpleNamespace(db_path=first))
    assert get_store_id(first) != get_store_id(second)
    with pytest.raises(ValueError):
        get_store_id(tmp_path / "missing.sqlite")


def test_resolver_requires_a_verified_actor_and_keeps_the_session(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    _state_db(home / "state.db")
    monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda name: name)
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: home)
    bare = SimpleNamespace(
        state=SimpleNamespace(session=None, token_principal=None, native_http_principal=None),
        headers=_Headers(),
    )
    with pytest.raises(PermissionError):
        resolve_dashboard_task_context(bare, profile="default")
    monkeypatch.setenv("TALK_DASHBOARD_TOKEN", "shared-token-value")
    shared = SimpleNamespace(
        state=bare.state,
        headers=_Headers({"x-talk-token": "shared-token-value"}),
    )
    shared_context = resolve_dashboard_task_context(shared)
    assert shared_context.principal_id == "shared-operator"
    assert shared_context.principal_kind == "dashboard_token"
    signed_in = SimpleNamespace(
        state=SimpleNamespace(
            session=SimpleNamespace(user_id="actor-1"),
            token_principal=SimpleNamespace(principal="other"),
            native_http_principal=None,
        ),
        headers=shared.headers,
    )
    context = resolve_dashboard_task_context(signed_in, profile="default")
    assert context.principal_id == "actor-1"
    assert context.principal_kind == "verified_subject"
    assert context.profile_home == home.resolve()
    assert context.store_id == get_store_id(home / "state.db")


def test_passive_history_round_trip_matches_the_talk_wire(tmp_path):
    HistoryCapabilities.parse(capabilities_document("store-1"))
    db = _Db()
    home = tmp_path
    attached = _body(_dispatch(db, home, "default", "store-1", "attach", {
        "tab_id": "tab-one", "session_id": "task-a",
    }))
    HistorySnapshot.parse(attached["snapshot"])
    assert attached["snapshot"]["capabilities"]["store_id"] == "store-1"
    authority = {
        key: attached[key] for key in ("tab_id", "attachment_id", "generation", "session_id")
    }
    saved = _body(_dispatch(db, home, "default", "store-1", "commit", {
        **authority,
        "event_id": "event-one",
        "origin_turn_id": "turn-one",
        "messages": [{"role": "user", "content": "Original words"}],
    }))
    assert saved["status"] == "saved"
    assert saved["receipt"]["message_ids"] == [1]
    assert len(db.messages) == 1
    replay = _body(_dispatch(db, home, "default", "store-1", "commit", {
        **authority,
        "event_id": "event-one",
        "origin_turn_id": "turn-one",
        "messages": [{"role": "user", "content": "Original words"}],
    }))
    assert replay["status"] == "already_saved"
    assert replay["receipt"]["replayed"] is True
    assert len(db.messages) == 1
    conflict = _dispatch(db, home, "default", "store-1", "commit", {
        **authority,
        "event_id": "event-one",
        "origin_turn_id": "turn-one",
        "messages": [{"role": "user", "content": "Different words"}],
    })
    assert conflict.status == 409
    assert _body(conflict)["error"] == "event_conflict"


def test_linked_child_keeps_the_goal_off_the_parent_session():
    _BY_CHILD.clear()
    assert classify({"input": "hello"}) is None
    spec = classify({
        "input": "Original words",
        "session_id": "task-a",
        "origin": {"event_id": "event-one", "origin_turn_id": "turn-one"},
        "child": {"goal": "Do the subtask", "context": "Bounded notes", "correlation_id": "action-1"},
    })
    assert spec["goal"] == "Do the subtask"
    assert spec["receipt_id"] is None
    refused = classify({
        "input": "Original words",
        "session_id": "task-a",
        "origin": {"event_id": "event-one", "origin_turn_id": "turn-one", "receipt_id": 4},
        "child": {
            "goal": "Do the subtask",
            "context": "",
            "correlation_id": "action-1",
            "worker": "hermes-talk-codex",
        },
    })
    assert refused == "child_denied"
    child = child_session_id("task-a", "action-1")
    _BY_CHILD[child] = {"parent_session_id": "task-a", "child_session_id": child}
    status = {"session_id": child, "status": "running"}
    overlay_status(status)
    assert status["session_id"] == "task-a"
    assert status["child_session_id"] == child
    assert child != "task-a"
