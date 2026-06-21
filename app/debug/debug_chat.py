# FILE: app/debug/debug_chat.py
# Purpose: Debug Chat (re-export shim).
# Called-by: app.llm.stream_router, app.router_registry (main)
# Depends-on: app.debug.debug_chat_endpoint, app.debug.debug_lock_stream
# Last-renovated: 2026-06-21
"""Debug Chat shim. Split 2026-06-21 (BATCH 7): the read-only /api/debug/chat endpoint
surface -> debug_chat_endpoint.py; the Debug-Lock stream -> debug_lock_stream.py.
All public names re-exported so importers (router_registry, stream_router) resolve unchanged."""
from __future__ import annotations

from app.debug.debug_chat_endpoint import (
    router,
    MAX_TOOL_ITERATIONS,
    _conversations,
    MAX_CONVERSATION_LENGTH,
    DebugChatRequest,
    _sse_event,
    _call_llm_simple,
    debug_chat,
    get_debug_history,
    clear_debug_history,
    debug_status,
)
from app.debug.debug_lock_stream import stream_debug_locked
