# FILE: app/llm/routing/command_dispatch.py
# Purpose: Command dispatch for stream routing — routes intents to stream handlers.
# Called-by: app.llm._stream_router_utils, app.llm.routing.weaver_reweave_handler, app.llm.stream_router
# Depends-on: app.builds.stage_hooks, app.llm.routing._cd_dispatch_table, app.llm.routing._command_dispatch_utils, app.llm.routing.handler_registry (+1 more)
# Last-renovated: 2026-06-11
"""
Command dispatch for stream routing — routes intents to stream handlers.

v1.2 (2026-02-24): Refactored to table-driven dispatch via _cd_dispatch_table.py
v1.1 (2026-01-28): Added MULTI_FILE_SEARCH and MULTI_FILE_REFACTOR handlers
v1.0 (2026-01-20): Extracted from stream_router.py for modularity.
"""

from __future__ import annotations

import logging
from typing import Optional, Any

from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.llm.translation_routing import (
    CanonicalIntent,
    intent_to_routing_info,
)

from .handler_registry import (
    StageTrace,
    log_handler_availability,
)
import app.llm.routing.handler_registry as _registry
from app.llm.routing._cd_dispatch_table import build_dispatch_table
from app.llm.routing._command_dispatch_utils import (
    _make_unavailable_error,
    create_error_stream,
    create_stage_trace,
    log_routing_failure,
)

logger = logging.getLogger(__name__)

# Build project lifecycle hooks (safe import)
try:
    from app.builds.stage_hooks import (
        is_tracked_stage,
        wrap_with_build_tracking,
    )
    _BUILD_TRACKING_AVAILABLE = True
except Exception as _e:
    _BUILD_TRACKING_AVAILABLE = False
    is_tracked_stage = None
    wrap_with_build_tracking = None
    logger.warning("[command_dispatch] Build tracking not available: %s", _e)

# Build table once at import time
_DISPATCH_TABLE = build_dispatch_table(_registry)

# Pre-index: intent → table entry (handles tuple-of-intents entries)
_INTENT_INDEX: dict[CanonicalIntent, dict] = {}
for _entry in _DISPATCH_TABLE:
    _intents = _entry["intent"]
    if isinstance(_intents, tuple):
        for _i in _intents:
            _INTENT_INDEX[_i] = _entry
    else:
        _INTENT_INDEX[_intents] = _entry

# SSE headers constant
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def handle_command_execution(
    req: Any,
    translation_result: Any,
    db: Session,
    trace: Any,
    conversation_id: str,
    stage_trace: Optional[StageTrace] = None,
) -> Optional[StreamingResponse]:
    """
    Handle approved command execution by dispatching to the correct stream handler.

    Returns StreamingResponse if command handled, None if should fall through.
    """
    intent = translation_result.resolved_intent
    routing_info = intent_to_routing_info(intent)

    if not routing_info:
        log_routing_failure(
            stage_trace,
            f"No routing info for intent: {intent.value if intent else 'None'}",
            "intent_to_routing_info",
            "falling through to normal routing",
        )
        return None

    # Log the resolved routing
    print(f"[COMMAND_ROUTE] Intent: {intent.value if intent else 'None'}")
    print(f"[COMMAND_ROUTE] Routing: type={routing_info['type']}, provider={routing_info['provider']}, model={routing_info['model']}")

    if trace:
        trace.log_request_start(
            job_type=req.job_type or "",
            resolved_job_type=routing_info["type"],
            provider=routing_info["provider"],
            model=routing_info["model"],
            reason=routing_info["reason"],
            frontier_override=False,
            file_map_injected=False,
            attachments=None,
        )

    # Look up in dispatch table
    entry = _INTENT_INDEX.get(intent)
    if entry is None:
        return None

    if not entry["available"]:
        error_name = entry.get("error_name")
        error_module = entry.get("error_module")
        log_routing_failure(
            stage_trace,
            f"{error_name or entry['stage']} handler not available",
            entry["stage"],
        )
        log_handler_availability()
        if error_name and error_module:
            error_msg = _make_unavailable_error(error_name, error_module)
            return StreamingResponse(
                create_error_stream(f"{error_name} not available", error_msg),
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )
        return None

    # Resolve provider/model for tracing
    if entry.get("provider_override"):
        prov_result = entry["provider_override"]()
        if isinstance(prov_result, tuple) and len(prov_result) == 2:
            trace_provider, trace_model = prov_result
        else:
            trace_provider = getattr(prov_result, 'provider', None) or routing_info["provider"]
            trace_model = getattr(prov_result, 'model', None) or routing_info["model"]
    else:
        trace_provider = routing_info["provider"]
        trace_model = routing_info["model"]

    if stage_trace:
        stage_trace.enter_stage(entry["stage"], provider=trace_provider, model=trace_model)

    # Build handler kwargs
    handler = entry["handler"]
    if entry.get("no_standard_args"):
        return StreamingResponse(
            handler(),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    kwargs: dict = {
        "project_id": req.project_id,
        "db": db,
        "trace": trace,
    }

    if not entry.get("skip_message"):
        kwargs["message"] = req.message

    if entry.get("needs_conversation_id"):
        kwargs["conversation_id"] = str(req.project_id)

    # Extra kwargs from handler_kwargs_fn
    if entry.get("handler_kwargs_fn"):
        extra = entry["handler_kwargs_fn"](req, translation_result)
        if extra:
            kwargs.update(extra)

    # Wrap with build project tracking if this is a tracked pipeline stage
    generator = handler(**kwargs)

    if (
        _BUILD_TRACKING_AVAILABLE
        and is_tracked_stage(entry["stage"])
        and hasattr(req, "project_id")
        and req.project_id
    ):
        generator = wrap_with_build_tracking(
            stream=generator,
            db=db,
            chat_project_id=req.project_id,
            dispatch_stage=entry["stage"],
            message=req.message,
            provider=trace_provider if "trace_provider" in dir() else None,
            model=trace_model if "trace_model" in dir() else None,
        )

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


__all__ = [
    "handle_command_execution",
    "create_stage_trace",
    "log_routing_failure",
    "create_error_stream",
]
