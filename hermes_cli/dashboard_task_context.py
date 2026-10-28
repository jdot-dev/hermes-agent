"""Verified actor, profile home, and canonical store for dashboard task continuity.

An individual session wins over the shared dashboard token. Nothing here guesses
a user from a profile name, a host header, or an unsigned body field.
"""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass
from pathlib import Path

from hermes_state_store_identity import get_store_id

_TALK_TOKEN_ENV = "TALK_DASHBOARD_TOKEN"
_TALK_TOKEN_HEADER = "x-talk-token"


@dataclass(frozen=True)
class DashboardTaskContext:
    principal_id: str
    principal_kind: str
    profile_name: str
    profile_home: Path
    store_id: str


def _nonempty(value) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _header(request, name: str) -> str:
    headers = getattr(request, "headers", None)
    getter = getattr(headers, "get", None)
    if getter is None:
        return ""
    value = getter(name)
    return value.strip() if isinstance(value, str) else ""


def _talk_token_matches(request) -> bool:
    configured = (os.environ.get(_TALK_TOKEN_ENV) or "").strip()
    if not configured:
        return False
    presented = _header(request, _TALK_TOKEN_HEADER)
    if not presented:
        authorization = _header(request, "authorization")
        if authorization.lower().startswith("bearer "):
            presented = authorization[7:].strip()
    if not presented:
        return False
    return hmac.compare_digest(presented.encode("utf-8"), configured.encode("utf-8"))


def _principal(request):
    state = getattr(request, "state", None)
    session = getattr(state, "session", None)
    user_id = _nonempty(getattr(session, "user_id", None))
    if user_id:
        return user_id, "verified_subject"
    token_principal = getattr(state, "token_principal", None)
    principal = _nonempty(getattr(token_principal, "principal", None))
    if principal:
        return principal, "token_principal"
    native = getattr(state, "native_http_principal", None)
    if isinstance(native, dict):
        native_id = _nonempty(native.get("principal")) or _nonempty(native.get("profile_id"))
        if native_id:
            return native_id, "native_grant"
    if _talk_token_matches(request):
        return "shared-operator", "dashboard_token"
    raise PermissionError("no verified dashboard actor")


def _profile(profile):
    from hermes_cli.profiles import get_profile_dir, normalize_profile_name, profile_exists

    if profile is None:
        profile = "default"
    if not isinstance(profile, str):
        raise PermissionError("profile")
    canon = normalize_profile_name(profile)
    if canon != "default" and not profile_exists(canon):
        raise PermissionError("profile")
    home = Path(get_profile_dir(canon)).resolve()
    if not home.is_absolute() or not home.is_dir():
        raise PermissionError("profile")
    return canon, home


def resolve_dashboard_task_context(request, profile=None) -> DashboardTaskContext:
    principal_id, principal_kind = _principal(request)
    profile_name, home = _profile(profile)
    database = home / "state.db"
    store_id = get_store_id(database)
    if not isinstance(store_id, str) or not store_id:
        raise PermissionError("store")
    return DashboardTaskContext(principal_id, principal_kind, profile_name, home, store_id)
