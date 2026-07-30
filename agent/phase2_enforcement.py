"""Default-off Phase 2 capability enforcement at the Hermes tool seam.

The canonical input is the sealed task-envelope v2 schema in the quad-system
contract. This module never manufactures authority. A Hermes-owned control plane
must bind an already-sealed envelope before enforcement can allow execution.
"""

from __future__ import annotations

import ipaddress
import math
import os
import re
import shlex
import threading
import time
from copy import deepcopy
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping
from urllib.parse import urlsplit

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_CURRENT_ENVELOPE: ContextVar[Mapping[str, Any] | None] = ContextVar(
    "phase2_current_sealed_envelope", default=None
)
_CURRENT_FENCE: ContextVar[int | None] = ContextVar("phase2_current_fence", default=None)
_CURRENT_BUDGET: ContextVar[dict[str, Any] | None] = ContextVar(
    "phase2_current_budget", default=None
)

_REQUIRED_FIELDS = (
    "graph_id",
    "node_id",
    "attempt_id",
    "idempotency_key",
    "objective",
    "execution_surface",
    "lane",
    "roots",
    "permissions",
    "network_policy",
    "exec_policy",
    "destructive_policy",
    "budgets",
    "deadline_utc",
    "retry_policy",
    "cancellation",
    "evidence_policy",
    "verifier_policy",
    "planner_hash",
    "policy_hash",
    "input_hash",
    "lease",
    "fence",
)
_ALLOWED_SURFACES = {"direct_model", "combo", "acp_worker", "a2a", "cloud", "local_tool"}
_ALLOWED_LANES = {"hermes", "fable", "claude_opus", "omx_codex", "omp", "orca"}
_ALLOWED_NETWORK = {"none", "loopback_only", "allowlist"}
_ALLOWED_EXEC = {"none", "sandboxed", "host"}
_ALLOWED_DESTRUCTIVE = {"forbid", "require_confirmation", "allow_within_roots"}
_STATELESS_SURFACES = {"direct_model", "combo"}
_NETWORK_TOOLS = {
    "web_search",
    "web_extract",
    "x_search",
    "image_generate",
    "video_generate",
    "text_to_speech",
}
_READ_PATH_FIELDS = {
    "read_file": ("path",),
    "search_files": ("path",),
    "vision_analyze": ("image_url",),
    "video_analyze": ("video_url",),
}
_WRITE_PATH_FIELDS = {
    "write_file": ("path",),
    "patch": ("path",),
}
_SHELL_CONTROL_RE = re.compile(r"(?:[;&|<>`\n\r]|\$\(|\$\{|\*|\?|\[|\]|\{|\})")


def _load_config() -> Any:
    from hermes_cli.config import load_config_readonly

    return load_config_readonly()


def is_enabled() -> bool:
    """True only for the explicit env flag or exact config boolean."""

    env_value = os.environ.get("HERMES_ENVELOPE_ENFORCE")
    if env_value is not None:
        return env_value.strip().lower() in {"1", "true", "yes", "on"}
    try:
        config = _load_config()
        block = ((config.get("enforcement") or {}).get("task_envelopes") or {})
        return block.get("enabled") is True
    except Exception:
        return False


@dataclass(frozen=True)
class EnforcementBlock:
    code: str
    message: str
    details: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.message,
            "status": "blocked",
            "error_type": "envelope_enforcement_block",
            "code": self.code,
            "details": list(self.details),
        }


@contextmanager
def bind_sealed_envelope(
    envelope: Mapping[str, Any], *, current_fence: int | None = None
) -> Iterator[None]:
    """Bind a defensive immutable copy to the current execution context."""

    frozen = MappingProxyType(deepcopy(dict(envelope)))
    envelope_token = _CURRENT_ENVELOPE.set(frozen)
    fence_token = _CURRENT_FENCE.set(current_fence)
    budget_token = _CURRENT_BUDGET.set(
        {
            "usd": 0.0,
            "tokens": 0,
            "cost_unknown": False,
            "started_monotonic": time.monotonic(),
            "lock": threading.Lock(),
        }
    )
    try:
        yield
    finally:
        _CURRENT_BUDGET.reset(budget_token)
        _CURRENT_FENCE.reset(fence_token)
        _CURRENT_ENVELOPE.reset(envelope_token)


def current_sealed_envelope() -> Mapping[str, Any] | None:
    return _CURRENT_ENVELOPE.get()


def current_authoritative_fence() -> int | None:
    """Return the fence bound by the Hermes control plane for this context."""

    return _CURRENT_FENCE.get()


def current_budget_usage() -> dict[str, float | int | bool] | None:
    """Return a defensive snapshot of usage charged to the bound node."""

    budget = _CURRENT_BUDGET.get()
    if budget is None:
        return None
    with budget["lock"]:
        usd = float(budget["usd"])
        tokens = int(budget["tokens"])
        cost_unknown = bool(budget["cost_unknown"])
        started = float(budget["started_monotonic"])
    return {
        "usd": usd,
        "tokens": tokens,
        "cost_unknown": cost_unknown,
        "wall_clock_s": max(0.0, time.monotonic() - started),
    }


def record_budget_usage(*, tokens: int = 0, usd: float | None = 0.0) -> None:
    """Charge observed model usage; unknown spend poisons further execution."""

    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        raise ValueError("tokens must be a non-negative integer")
    if usd is not None and (
        isinstance(usd, bool) or not isinstance(usd, (int, float)) or usd < 0
    ):
        raise ValueError("usd must be a non-negative number or None")
    budget = _CURRENT_BUDGET.get()
    if budget is None:
        return
    with budget["lock"]:
        budget["tokens"] = int(budget["tokens"]) + tokens
        if usd is None:
            budget["cost_unknown"] = True
        else:
            budget["usd"] = float(budget["usd"]) + float(usd)


def _aware_datetime(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _nonnegative_number(value: Any) -> bool:
    """True for a finite, non-negative, non-``bool`` number.

    Finiteness is part of the contract, not a nicety. A ``usd_max`` or
    ``tokens_max`` of ``inf`` is a ceiling that can never be breached, which
    contradicts §9 ("breach at either level → fail closed"), and ``nan`` defeats
    every comparison silently (``nan >= limit`` is always False). Neither is
    representable in JSON, so a sealed envelope carrying one cannot be durably
    audited or read back by a strict parser, and ``int(inf)`` aborts budget
    accounting with an untyped ``OverflowError``.
    """

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value >= 0
        and math.isfinite(value)
    )


def _nonempty_strings(value: Any, *, allow_empty: bool = True) -> bool:
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )


def validate_sealed_envelope(envelope: Mapping[str, Any]) -> list[str]:
    errors = [field for field in _REQUIRED_FIELDS if envelope.get(field) is None]
    if envelope.get("envelope_version") != 2:
        errors.append("envelope_version")
    for field in ("graph_id", "node_id", "attempt_id", "objective"):
        if not isinstance(envelope.get(field), str) or not envelope.get(field, "").strip():
            errors.append(field)
    if len(str(envelope.get("objective") or "")) > 2000:
        errors.append("objective")
    if envelope.get("execution_surface") not in _ALLOWED_SURFACES:
        errors.append("execution_surface")
    if envelope.get("lane") not in _ALLOWED_LANES:
        errors.append("lane")
    if not _nonempty_strings(envelope.get("roots"), allow_empty=False):
        errors.append("roots")
    elif any(not Path(root).expanduser().is_absolute() for root in envelope["roots"]):
        errors.append("roots")

    permissions = envelope.get("permissions")
    if not isinstance(permissions, Mapping):
        errors.append("permissions")
        permissions = {}
    for field in ("read", "write", "spawn"):
        values = permissions.get(field)
        if not _nonempty_strings(values):
            errors.append(f"permissions.{field}")
        elif field in {"read", "write"} and any(
            not Path(value).expanduser().is_absolute() for value in (values or [])
        ):
            errors.append(f"permissions.{field}")

    network_policy = envelope.get("network_policy")
    if network_policy not in _ALLOWED_NETWORK:
        errors.append("network_policy")
    if network_policy == "allowlist" and not _nonempty_strings(
        envelope.get("network_allowlist"), allow_empty=False
    ):
        errors.append("network_allowlist")
    if envelope.get("exec_policy") not in _ALLOWED_EXEC:
        errors.append("exec_policy")
    if envelope.get("destructive_policy") not in _ALLOWED_DESTRUCTIVE:
        errors.append("destructive_policy")

    budgets = envelope.get("budgets")
    if not isinstance(budgets, Mapping):
        errors.append("budgets")
        budgets = {}
    for field in ("usd_max", "tokens_max", "wall_clock_s_max"):
        if not _nonnegative_number(budgets.get(field)):
            errors.append(f"budgets.{field}")
    if _aware_datetime(envelope.get("deadline_utc")) is None:
        errors.append("deadline_utc")

    retry = envelope.get("retry_policy")
    if not isinstance(retry, Mapping):
        errors.append("retry_policy")
        retry = {}
    max_attempts = retry.get("max_attempts")
    retry_eligible = retry.get("retry_eligible")
    if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts < 1:
        errors.append("retry_policy.max_attempts")
    if not isinstance(retry_eligible, bool):
        errors.append("retry_policy.retry_eligible")
    if not isinstance(retry.get("backoff_s"), list) or not all(
        _nonnegative_number(item) for item in retry.get("backoff_s", [])
    ):
        errors.append("retry_policy.backoff_s")
    writes = permissions.get("write") if isinstance(permissions, Mapping) else []
    if writes and (max_attempts != 1 or retry_eligible is not False):
        errors.append("retry_policy.mutation")
    if max_attempts and max_attempts > 1 and envelope.get("execution_surface") not in _STATELESS_SURFACES:
        errors.append("retry_policy.execution_surface")

    cancellation = envelope.get("cancellation")
    if not isinstance(cancellation, Mapping):
        errors.append("cancellation")
        cancellation = {}
    if cancellation.get("mode") != "cooperative":
        errors.append("cancellation.mode")
    if not _nonnegative_number(cancellation.get("grace_s")):
        errors.append("cancellation.grace_s")

    evidence = envelope.get("evidence_policy")
    if not isinstance(evidence, Mapping):
        errors.append("evidence_policy")
        evidence = {}
    if not _nonempty_strings(evidence.get("required_receipt_kinds"), allow_empty=False):
        errors.append("evidence_policy.required_receipt_kinds")
    min_receipts = evidence.get("min_receipts")
    if not isinstance(min_receipts, int) or isinstance(min_receipts, bool) or min_receipts < 1:
        errors.append("evidence_policy.min_receipts")

    verifier = envelope.get("verifier_policy")
    if not isinstance(verifier, Mapping):
        errors.append("verifier_policy")
        verifier = {}
    for field in ("required", "verifier_lane_must_differ", "clean_workspace_required"):
        if not isinstance(verifier.get(field), bool):
            errors.append(f"verifier_policy.{field}")
    if writes and not all(
        verifier.get(field) is True
        for field in ("required", "verifier_lane_must_differ", "clean_workspace_required")
    ):
        errors.append("verifier_policy.mutation")

    for field in ("idempotency_key", "planner_hash", "policy_hash", "input_hash"):
        if not _HASH_RE.fullmatch(str(envelope.get(field) or "")):
            errors.append(field)

    lease = envelope.get("lease")
    if not isinstance(lease, Mapping):
        errors.append("lease")
        lease = {}
    if not isinstance(lease.get("holder"), str) or not lease.get("holder", "").strip():
        errors.append("lease.holder")
    granted = _aware_datetime(lease.get("granted_utc"))
    if granted is None:
        errors.append("lease.granted_utc")
    if not _nonnegative_number(lease.get("ttl_s")) or lease.get("ttl_s") == 0:
        errors.append("lease.ttl_s")
    if not isinstance(lease.get("renewable"), bool):
        errors.append("lease.renewable")
    fence = envelope.get("fence")
    if not isinstance(fence, int) or isinstance(fence, bool) or fence < 1:
        errors.append("fence")
    return sorted(set(errors))


def _inside(path_value: str, roots: list[str], cwd: str | None) -> bool:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = Path(cwd or Path.cwd()) / path
    candidate = path.resolve(strict=False)
    for root_value in roots:
        root = Path(root_value).expanduser().resolve(strict=False)
        if candidate == root or root in candidate.parents:
            return True
    return False


def _path_values(
    function_name: str,
    function_args: Mapping[str, Any],
    fields: Mapping[str, tuple[str, ...]],
) -> list[str]:
    values: list[str] = []
    for field in fields.get(function_name, ()):
        value = function_args.get(field)
        if isinstance(value, str) and value and not value.startswith(("http://", "https://", "data:")):
            values.append(value)
    return values


def _is_network_tool(function_name: str) -> bool:
    return (
        function_name in _NETWORK_TOOLS
        or function_name.startswith("browser_")
        or function_name.startswith("mcp__scrapling__")
    )


def _terminal_argv(command: Any) -> list[str] | None:
    if not isinstance(command, str) or not command.strip() or _SHELL_CONTROL_RE.search(command):
        return None
    try:
        argv = shlex.split(command, posix=True)
    except ValueError:
        return None
    if not argv or any("\x00" in item for item in argv):
        return None
    return argv


def _network_urls(function_name: str, function_args: Mapping[str, Any]) -> list[str] | None:
    if function_name == "web_extract":
        values = function_args.get("urls")
    elif function_name in {"browser_navigate", "vision_analyze", "video_analyze"}:
        values = [function_args.get("url") or function_args.get("image_url") or function_args.get("video_url")]
    elif function_name.startswith("mcp__scrapling__"):
        values = function_args.get("urls") or [function_args.get("url")]
    else:
        return None
    if not isinstance(values, list) or not values:
        return None
    urls = [value for value in values if isinstance(value, str) and value.strip()]
    return urls if len(urls) == len(values) else None


def _url_host(value: str) -> str | None:
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
        return None
    return host.rstrip(".").lower()


def _loopback_host(host: str) -> bool:
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def evaluate_runtime_authority(*, now: datetime | None = None) -> EnforcementBlock | None:
    """Validate bound authority and runtime budget before provider dispatch."""

    return evaluate_tool_call("__phase2_model_dispatch__", {}, now=now)


def claim_mutating_tool(
    function_name: str,
    function_args: Mapping[str, Any],
) -> EnforcementBlock | None:
    """Atomically reserve a mutating envelope's idempotency key before dispatch."""

    if not is_enabled():
        return None
    if function_name not in {"write_file", "patch", "terminal", "delegate_task"}:
        return None
    envelope = current_sealed_envelope()
    if envelope is None:
        return EnforcementBlock(
            "missing_sealed_envelope",
            "Phase 2 enforcement is enabled but no sealed task envelope is bound",
        )
    try:
        from agent.phase2_idempotency import MutationClaimStore

        claimed = MutationClaimStore().try_claim(
            envelope,
            tool_name=function_name,
            args=function_args,
        )
    except Exception:
        return EnforcementBlock(
            "idempotency_store_unavailable",
            "The authoritative mutation claim store could not reserve this operation",
        )
    if not claimed:
        return EnforcementBlock(
            "duplicate_idempotency_key",
            "This mutating logical operation already has a durable execution claim",
            (str(envelope["idempotency_key"]),),
        )
    return None


def evaluate_tool_call(
    function_name: str,
    function_args: Mapping[str, Any],
    *,
    cwd: str | None = None,
    destructive: bool = False,
    now: datetime | None = None,
) -> EnforcementBlock | None:
    """Return a fail-closed block, or ``None`` when the call may proceed."""

    if not is_enabled():
        return None
    envelope = current_sealed_envelope()
    if envelope is None:
        return EnforcementBlock(
            "missing_sealed_envelope",
            "Phase 2 enforcement is enabled but no sealed task envelope is bound",
        )
    errors = validate_sealed_envelope(envelope)
    if errors:
        return EnforcementBlock(
            "invalid_sealed_envelope",
            "The bound task envelope is incomplete or invalid",
            tuple(errors),
        )

    expected_surface = (
        "direct_model" if function_name == "__phase2_model_dispatch__" else "local_tool"
    )
    if envelope["execution_surface"] != expected_surface:
        return EnforcementBlock(
            "execution_surface_mismatch",
            "The bound graph node does not authorize this execution surface",
            (expected_surface, str(envelope["execution_surface"])),
        )

    usage = current_budget_usage()
    if usage is None:
        return EnforcementBlock(
            "missing_budget_meter",
            "No authoritative runtime budget meter is bound for this task envelope",
        )
    if bool(usage["cost_unknown"]):
        return EnforcementBlock(
            "budget_cost_unknown",
            "Observed provider usage could not be charged to the sealed USD budget",
        )
    exhausted: list[str] = []
    budgets = envelope["budgets"]
    if float(usage["usd"]) >= float(budgets["usd_max"]):
        exhausted.append("usd_max")
    if int(usage["tokens"]) >= int(budgets["tokens_max"]):
        exhausted.append("tokens_max")
    if float(usage["wall_clock_s"]) >= float(budgets["wall_clock_s_max"]):
        exhausted.append("wall_clock_s_max")
    if exhausted:
        return EnforcementBlock(
            "budget_exhausted",
            "The task-envelope runtime budget is exhausted",
            tuple(sorted(exhausted)),
        )

    current_fence = _CURRENT_FENCE.get()
    if current_fence is None:
        return EnforcementBlock(
            "missing_current_fence",
            "No authoritative current fence is bound for stale-attempt rejection",
        )
    if envelope["fence"] != current_fence:
        return EnforcementBlock(
            "stale_fence",
            "The task-envelope fence is not the authoritative current fence",
        )

    current = now or datetime.now(timezone.utc)
    deadline = _aware_datetime(envelope["deadline_utc"])
    assert deadline is not None
    if current >= deadline.astimezone(timezone.utc):
        return EnforcementBlock("deadline_expired", "The task envelope deadline has expired")
    lease = envelope["lease"]
    granted = _aware_datetime(lease["granted_utc"])
    assert granted is not None
    if current >= granted.astimezone(timezone.utc) + timedelta(seconds=float(lease["ttl_s"])):
        return EnforcementBlock("lease_expired", "The task-envelope lease has expired")

    if function_name == "__phase2_model_dispatch__":
        return None

    permissions = envelope["permissions"]
    if _is_network_tool(function_name):
        if envelope["network_policy"] == "none":
            return EnforcementBlock("network_policy_denied", "Network access is denied by the sealed envelope")
        urls = _network_urls(function_name, function_args)
        if urls is None:
            return EnforcementBlock(
                "network_destination_ambiguous",
                "The network tool does not expose enforceable request destinations",
            )
        hosts = [_url_host(value) for value in urls]
        if any(host is None for host in hosts):
            return EnforcementBlock(
                "network_destination_ambiguous",
                "One or more network destinations are not canonical HTTP(S) URLs",
            )
        if envelope["network_policy"] == "loopback_only":
            if not all(_loopback_host(host) for host in hosts if host is not None):
                return EnforcementBlock(
                    "network_destination_denied",
                    "A network destination is outside the loopback-only policy",
                )
        else:
            allowlist = {str(host).rstrip(".").lower() for host in envelope["network_allowlist"]}
            if not all(host in allowlist for host in hosts):
                return EnforcementBlock(
                    "network_destination_denied",
                    "A network destination is absent from the sealed envelope allowlist",
                )
        return EnforcementBlock(
            "network_transport_not_enforceable",
            "Network execution remains blocked until every redirect hop and resolved address is re-authorized",
        )
    if function_name == "delegate_task":
        tasks = function_args.get("tasks")
        if tasks is not None:
            return EnforcementBlock(
                "spawn_request_ambiguous",
                "Batch or nested worker spawning is not a single enforceable graph-node target",
            )
        goal = function_args.get("goal")
        role = function_args.get("role") or "leaf"
        if not isinstance(goal, str) or not goal.strip() or role not in {"leaf", "orchestrator"}:
            return EnforcementBlock(
                "spawn_request_ambiguous",
                "Worker spawn request lacks one bounded goal and a recognized role",
            )
        target = f"delegate_task:{role}"
        if target not in permissions["spawn"]:
            return EnforcementBlock(
                "spawn_policy_denied",
                "Worker target is not authorized by the sealed envelope spawn list",
                (target,),
            )
        return EnforcementBlock(
            "spawn_child_authority_not_enforceable",
            "Worker spawning remains blocked until the child receives its own sealed envelope, fence, budget, and receipt identity",
        )
    if function_name == "terminal":
        if envelope["exec_policy"] == "none":
            return EnforcementBlock("exec_policy_denied", "Command execution is denied by the sealed envelope")
        if destructive and envelope["destructive_policy"] == "forbid":
            return EnforcementBlock(
                "destructive_policy_denied",
                "Destructive execution is denied by the sealed envelope",
            )
        if function_args.get("background") is True:
            return EnforcementBlock(
                "terminal_background_not_enforceable",
                "Background terminal execution remains blocked until its lifetime is budgeted and receipted",
            )
        argv = _terminal_argv(function_args.get("command"))
        if argv is None:
            return EnforcementBlock(
                "terminal_command_ambiguous",
                "Terminal command contains shell control syntax or cannot be parsed as one argv",
            )
        executable = argv[0]
        if executable not in permissions["spawn"]:
            return EnforcementBlock(
                "spawn_policy_denied",
                "Terminal argv[0] is not authorized by the sealed envelope spawn list",
                (executable,),
            )
        if len(argv) != 1:
            return EnforcementBlock(
                "terminal_arguments_not_enforceable",
                "Terminal arguments remain blocked until command-specific argv policy exists",
            )
        workdir = function_args.get("workdir") or cwd
        if not isinstance(workdir, str) or not workdir.strip():
            return EnforcementBlock(
                "exec_workdir_ambiguous",
                "Terminal working directory cannot be authorized",
            )
        roots = [str(value) for value in envelope["roots"]]
        if not _inside(workdir, roots, cwd):
            return EnforcementBlock(
                "exec_workdir_outside_roots",
                "Terminal working directory is outside the sealed envelope roots",
            )
        return EnforcementBlock(
            "terminal_transport_not_enforceable",
            "Terminal execution remains blocked until the backend consumes a structured argv without a shell",
        )
    if function_name == "patch" and not isinstance(function_args.get("path"), str):
        return EnforcementBlock(
            "multi_file_patch_not_enforceable",
            "Multi-file patch execution remains blocked until every patch target is parsed and authorized",
        )

    roots = [str(value) for value in envelope["roots"]]
    read_roots = [str(value) for value in permissions["read"]]
    write_roots = [str(value) for value in permissions["write"]]
    read_values = _path_values(function_name, function_args, _READ_PATH_FIELDS)
    write_values = _path_values(function_name, function_args, _WRITE_PATH_FIELDS)
    if function_name in _READ_PATH_FIELDS and not read_values:
        return EnforcementBlock("read_path_ambiguous", "The read path cannot be authorized")
    if function_name in _WRITE_PATH_FIELDS and not write_values:
        return EnforcementBlock("write_path_ambiguous", "The write path cannot be authorized")
    for value in read_values:
        if not _inside(value, roots, cwd) or not _inside(value, read_roots, cwd):
            return EnforcementBlock(
                "read_outside_permissions",
                "Read path is outside the sealed envelope permissions",
            )
    for value in write_values:
        if not _inside(value, roots, cwd) or not _inside(value, write_roots, cwd):
            return EnforcementBlock(
                "write_outside_permissions",
                "Write path is outside the sealed envelope permissions",
            )
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(cwd or Path.cwd()) / path
        if not path.exists():
            return EnforcementBlock(
                "write_target_race_not_enforceable",
                "Creating a new filesystem target remains blocked until authorization and open are race-safe",
            )
    if not read_values and not write_values:
        return EnforcementBlock(
            "unclassified_tool_capability",
            f"Tool '{function_name}' has no enforceable Phase 2 capability classification",
        )
    return None
