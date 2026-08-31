"""Default-off Phase 1 model-call correlation for explicitly configured routers."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Optional, TypeVar

from agent.action_receipts import ActionReceiptLedger, default_db_path
from agent.task_envelope import TaskEnvelope, build_shadow_envelope, validate_envelope


@dataclass(frozen=True)
class ModelCallShadow:
    envelope: TaskEnvelope
    validation_errors: tuple[str, ...]


_ResultT = TypeVar("_ResultT")


def _load_config() -> Any:
    from hermes_cli.config import load_config_readonly

    return load_config_readonly()


def _shadow_config() -> dict[str, Any]:
    try:
        config = _load_config()
        if not isinstance(config, dict):
            return {}
        observability = config.get("observability")
        if not isinstance(observability, dict):
            return {}
        section = observability.get("envelope_shadow")
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def is_enabled() -> bool:
    return _shadow_config().get("enabled") is True


def _normalized_config_values(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        normalized
        for item in value
        if isinstance(item, str) and (normalized := item.strip().lower().rstrip("/"))
    )


def _is_configured_endpoint(base_url: Optional[str], section: dict[str, Any]) -> bool:
    normalized = str(base_url or "").strip().lower().rstrip("/")
    return bool(normalized) and normalized in _normalized_config_values(
        section.get("base_urls")
    )


def _is_combo_model(model: Optional[str], section: dict[str, Any]) -> bool:
    normalized = str(model or "").strip().lower()
    if not normalized:
        return False
    if normalized in _normalized_config_values(section.get("combo_models")):
        return True
    return any(
        normalized.startswith(prefix)
        for prefix in _normalized_config_values(section.get("combo_model_prefixes"))
    )


def _request_fingerprint_material(
    request: dict[str, Any], *, model: Optional[str], api_mode: Optional[str]
) -> dict[str, Any]:
    messages = request.get("messages")
    if not isinstance(messages, list):
        messages = request.get("input")
    tools = request.get("tools")
    return {
        "model": model,
        "api_mode": api_mode,
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "tool_count": len(tools) if isinstance(tools, list) else 0,
    }


def prepare_model_call_shadow(
    request: dict[str, Any],
    *,
    base_url: Optional[str],
    provider: Optional[str],
    model: Optional[str],
    api_mode: Optional[str],
    session_id: Optional[str],
    task_id: Optional[str],
    api_request_id: Optional[str],
) -> tuple[dict[str, Any], Optional[ModelCallShadow]]:
    """Inject correlation only for an explicitly allow-listed router endpoint."""
    section = _shadow_config()
    if section.get("enabled") is not True:
        return request, None
    if not _is_configured_endpoint(base_url, section):
        return request, None

    try:
        envelope = build_shadow_envelope(
            tool_name="model_call",
            args=_request_fingerprint_material(request, model=model, api_mode=api_mode),
            session_id=session_id,
            task_id=task_id,
            tool_call_id=api_request_id,
            objective=f"request model {model or 'unknown'}",
        )
        envelope = TaskEnvelope(
            **{
                **envelope.to_dict(),
                "execution_surface": (
                    "combo" if _is_combo_model(model, section) else "direct_model"
                ),
                "lane": "hermes",
            }
        )
        validation_errors = tuple(validate_envelope(envelope))
        prepared = dict(request)
        existing_headers = request.get("extra_headers")
        headers = (
            {
                name: value
                for name, value in existing_headers.items()
                if str(name).lower() != "x-request-id"
            }
            if isinstance(existing_headers, dict)
            else {}
        )
        headers["x-request-id"] = envelope.action_id
        prepared["extra_headers"] = headers
    except Exception:
        return request, None

    return prepared, ModelCallShadow(
        envelope=envelope,
        validation_errors=validation_errors,
    )


def _usage_summary(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    def read(*names: str) -> Any:
        for name in names:
            if isinstance(usage, dict) and name in usage:
                return usage[name]
            value = getattr(usage, name, None)
            if value is not None:
                return value
        return None

    return {
        "input_tokens": read("input_tokens", "prompt_tokens"),
        "output_tokens": read("output_tokens", "completion_tokens"),
        "total_tokens": read("total_tokens"),
    }


def _record_model_call(
    shadow: ModelCallShadow,
    *,
    model: Optional[str],
    provider: Optional[str],
    api_mode: Optional[str],
    response: Any,
    exit_status: str,
    duration_ms: int,
    session_id: Optional[str],
    task_id: Optional[str],
    api_request_id: Optional[str],
) -> None:
    summary = {
        "model": model,
        "provider": provider,
        "api_mode": api_mode,
        "response_model": getattr(response, "model", None) if response is not None else None,
        "validation_errors": list(shadow.validation_errors),
        "usage": _usage_summary(response),
    }
    try:
        ActionReceiptLedger(default_db_path()).record_receipt(
            tool_name=str(model or "model_call"),
            args=summary,
            output=None,
            exit_status=exit_status,
            duration_ms=duration_ms,
            session_id=session_id,
            task_id=task_id,
            tool_call_id=api_request_id,
            kind="model_call",
            envelope=shadow.envelope,
        )
    except Exception:
        pass


def run_model_call_shadow(
    request: dict[str, Any],
    execute: Callable[[dict[str, Any]], _ResultT],
    *,
    base_url: Optional[str],
    provider: Optional[str],
    model: Optional[str],
    api_mode: Optional[str],
    session_id: Optional[str],
    task_id: Optional[str],
    api_request_id: Optional[str],
) -> _ResultT:
    prepared, shadow = prepare_model_call_shadow(
        request,
        base_url=base_url,
        provider=provider,
        model=model,
        api_mode=api_mode,
        session_id=session_id,
        task_id=task_id,
        api_request_id=api_request_id,
    )
    if shadow is None:
        return execute(prepared)

    started = time.monotonic()
    try:
        response = execute(prepared)
    except BaseException as exc:
        _record_model_call(
            shadow,
            model=model,
            provider=provider,
            api_mode=api_mode,
            response=None,
            exit_status=(
                "cancelled"
                if isinstance(exc, (InterruptedError, KeyboardInterrupt))
                else "error"
            ),
            duration_ms=max(0, int((time.monotonic() - started) * 1000)),
            session_id=session_id,
            task_id=task_id,
            api_request_id=api_request_id,
        )
        raise

    _record_model_call(
        shadow,
        model=model,
        provider=provider,
        api_mode=api_mode,
        response=response,
        exit_status="ok",
        duration_ms=max(0, int((time.monotonic() - started) * 1000)),
        session_id=session_id,
        task_id=task_id,
        api_request_id=api_request_id,
    )
    return response


__all__ = [
    "ModelCallShadow",
    "is_enabled",
    "prepare_model_call_shadow",
    "run_model_call_shadow",
]
