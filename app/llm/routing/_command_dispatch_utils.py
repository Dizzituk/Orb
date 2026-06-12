# Purpose: command dispatch utils
# Called-by: app.llm.routing.command_dispatch
# Depends-on: app.llm.routing.handler_registry
# Last-renovated: 2026-06-11
from __future__ import annotations
import json
import logging
from .handler_registry import StageTrace, _STAGE_TRACE_AVAILABLE
from typing import AsyncIterator, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


def create_stage_trace(
    command_type: str,
    project_id: int,
    job_id: Optional[str] = None,
) -> Optional[StageTrace]:
    """Create a stage trace if available."""
    if _STAGE_TRACE_AVAILABLE and StageTrace:
        return StageTrace.start(command_type, project_id=project_id, job_id=job_id)
    return None

def log_routing_failure(
    stage_trace: Optional[StageTrace],
    reason: str,
    expected_handler: str,
    fallback_action: Optional[str] = None,
) -> None:
    """Log a routing failure."""
    # Always log to console for visibility
    print(f"[ROUTING_FAILURE] {reason}")
    print(f"[ROUTING_FAILURE]   expected_handler={expected_handler}")
    if fallback_action:
        print(f"[ROUTING_FAILURE]   fallback={fallback_action}")
    
    logger.warning(f"[command_dispatch] ROUTING_FAILURE: {reason} (handler={expected_handler})")
    
    if stage_trace:
        stage_trace.record_routing_failure(reason, expected_handler, fallback_action)

async def create_error_stream(
    error_type: str,
    error_message: str,
    intent: Optional[str] = None,
    reason: Optional[str] = None,
) -> AsyncIterator[str]:
    """Generate an error SSE stream."""
    yield "data: " + json.dumps({'type': 'error', 'error': error_type}) + "\n\n"
    yield "data: " + json.dumps({'type': 'token', 'content': error_message}) + "\n\n"
    
    if intent and reason:
        yield "data: " + json.dumps({
            'type': 'command_blocked',
            'intent': intent,
            'reason': reason,
        }) + "\n\n"
    
    yield "data: " + json.dumps({'type': 'done', 'provider': 'system', 'model': 'command_router'}) + "\n\n"

def _make_unavailable_error(handler_name: str, module_path: str) -> str:
    """Generate standard handler unavailable error message."""
    return (
        f"⚠️ **{handler_name} Not Available**\n\n"
        f"The {handler_name.lower()} module failed to import.\n"
        f"Check server logs for details.\n\n"
        f"**Possible solutions:**\n"
        f"1. Ensure `{module_path}` exists\n"
        f"2. Check for import errors in the module\n"
        f"3. Restart the backend server"
    )
