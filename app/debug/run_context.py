# FILE: app/debug/run_context.py
# Purpose: Ambient per-run signals (cancel + phone-origin) threaded via contextvars.
#          Lets cooperative cancellation and the Android-write confirmation gate
#          reach deep call chains (loop_controller, spawn_tool, filesystem executor)
#          without adding a parameter to every intermediate function signature on
#          both the orchestrator path and the chat-tool-loop path. Default state is
#          a no-op, so desktop-originated runs (which never set these) are unchanged.
# Called-by: app.bridge.debug_run_registry, app.debug.orchestrator.loop_controller,
#            app.debug.orchestrator.spawn_tool, app.debug.executors.filesystem
# Depends-on: contextvars (stdlib only)
# Last-renovated: 2026-07-01
from __future__ import annotations

import contextvars
from typing import Optional

# Holds the asyncio.Event for the run currently executing in this context, if any.
# Typed as object (not asyncio.Event) to avoid importing asyncio here for a type-only need.
CANCEL_EVENT: "contextvars.ContextVar[Optional[object]]" = contextvars.ContextVar(
    "debug_run_cancel_event", default=None
)

# True only inside a phone-initiated (/bridge/debug-and-speak) run. Desktop debug
# runs (via /stream/chat debug_locked or /orchestrate) never set this.
PHONE_INITIATED: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
    "debug_run_phone_initiated", default=False
)


def is_cancelled() -> bool:
    """Cheap, safe to call from any depth. False whenever no run has set a signal
    (i.e. every desktop call path, unchanged from before this module existed)."""
    ev = CANCEL_EVENT.get()
    try:
        return bool(ev is not None and ev.is_set())
    except Exception:
        return False


def is_phone_initiated() -> bool:
    return PHONE_INITIATED.get()
