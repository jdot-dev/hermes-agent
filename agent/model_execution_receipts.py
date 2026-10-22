"""Default-off model request receipts with explicit router correlation.

Only an enabled receipt profile records new requests. Telemetry never changes
provider selection, retries or error propagation. No historical backfilling.
"""
from contextlib import contextmanager
import logging
import os
import time
from urllib.parse import urlsplit
import uuid

logger = logging.getLogger(__name__)


def receipt_task_id(fallback):
    return os.environ.get("HERMES_KANBAN_TASK") or fallback or None


@contextmanager
def model_request_receipt(agent, request, effective_task_id, api_request_id):
    from agent.action_receipts import is_enabled

    if not is_enabled():
        yield request
        return

    from agent.action_receipts import ActionReceiptLedger
    from agent.task_envelope import build_shadow_envelope

    started = time.monotonic()
    action_id = str(uuid.uuid4())
    task_id = receipt_task_id(effective_task_id)
    session_id = getattr(agent, "session_id", None)
    # Bind run identity in the hashed envelope without adding a schema migration.
    run_id = os.environ.get("HERMES_KANBAN_RUN_ID")
    call_id = f"run:{run_id}:request:{api_request_id}" if run_id else str(api_request_id)
    args = {"model": str(getattr(agent, "model", "")),
            "provider": str(getattr(agent, "provider", "")), "run_id": run_id}
    envelope = None
    outgoing = request
    try:
        envelope = build_shadow_envelope(
            tool_name="model_call", args=args, session_id=session_id,
            task_id=task_id, tool_call_id=call_id, action_id=action_id,
            objective="Execute one model request; retries remain native",
        )
        # Only the configured loopback router receives this correlation header.
        # Native OAuth/direct providers and their request schemas stay untouched.
        base_url = str(getattr(getattr(agent, "client", None), "base_url", ""))
        url = urlsplit(base_url)
        if url.hostname in {"127.0.0.1", "localhost", "::1"} and url.port == 20128:
            outgoing = dict(request)
            headers = dict(request.get("extra_headers") or {})
            existing = next((v for k, v in headers.items() if k.lower() == "x-correlation-id"), None)
            if existing:
                from dataclasses import replace
                envelope = replace(envelope, action_id=str(existing))
            else:
                headers["x-correlation-id"] = action_id
            outgoing["extra_headers"] = headers
    except Exception:
        logger.warning("model receipt preparation unavailable", exc_info=False)
        envelope = None
        outgoing = request

    status = "ok"
    try:
        yield outgoing
    except BaseException:
        status = "error"
        raise
    finally:
        if envelope is not None:
            try:
                ActionReceiptLedger().record_receipt(
                    tool_name="model_call", args=args, output={"status": status},
                    exit_status=status, duration_ms=int((time.monotonic() - started) * 1000),
                    session_id=session_id, task_id=task_id, tool_call_id=call_id,
                    envelope=envelope, kind="model_call",
                )
            except Exception:
                logger.warning("model receipt write unavailable", exc_info=False)
