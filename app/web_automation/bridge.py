# FILE: app/web_automation/bridge.py
"""
Public Python bridge for web automation.

Callers (tool handlers, composite action scripts) use this module to
dispatch a primitive action to Electron and get the result back.

  result = await bridge.execute_action(session_id, action_type, payload)

All DB ceremony is hidden — the caller just needs a session_id or
a platform key. A fresh SessionLocal is used internally.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.db import SessionLocal
from app.web_automation import action_queue, session_registry
from app.web_automation.models import SessionStatus

logger = logging.getLogger(__name__)


async def execute_action(
    session_id: str,
    action_type: str,
    payload: Optional[dict] = None,
    *,
    timeout_seconds: float = 30.0,
) -> dict:
    """
    Enqueue an action for a session and wait for the result.

    Returns:
      {"ok": True,  "result": {...}}       on success
      {"ok": False, "error": "..."}        on failure or timeout
    """
    db = SessionLocal()
    try:
        sess = session_registry.get_session(db, session_id)
        if not sess:
            return {"ok": False, "error": f"session {session_id} not found"}
        action = action_queue.enqueue_action(db, session_id, action_type, payload or {})
        session_registry.touch_last_used(db, session_id)
        action_id = action.id
    finally:
        db.close()

    logger.info("[web_automation] dispatch %s -> session %s (action %s)",
                action_type, session_id[:8], action_id[:8])

    return await action_queue.await_result(action_id, timeout_seconds=timeout_seconds)


async def execute_by_platform(
    platform: str,
    action_type: str,
    payload: Optional[dict] = None,
    *,
    timeout_seconds: float = 30.0,
) -> dict:
    """Convenience: look up a session by platform name, then dispatch."""
    db = SessionLocal()
    try:
        sess = session_registry.get_session_by_platform(db, platform)
        if not sess:
            return {"ok": False, "error": f"no session registered for platform '{platform}'"}
        sid = sess.id
    finally:
        db.close()

    return await execute_action(sid, action_type, payload, timeout_seconds=timeout_seconds)


async def ensure_session_open(session_id: str, timeout_seconds: float = 10.0) -> dict:
    """
    Ask Electron to spawn the session's WebContentsView if not already live.

    Mechanism: dispatches a lightweight 'ensure_view' action.
    • If Electron already has the view: returns immediately with current_state.
    • If the view is brand-new: ensureView creates it, kicks off loadURL on
      the stored landing_url, waits up to ~5s for the navigation to commit,
      and returns. Full page load is NOT awaited — that's the job of an
      explicit navigate action when the LLM actually wants to change page.

    Why not just dispatch navigate(url=null) like before?  Because that did
    a full wc.loadURL + waitForLoad cycle, which on heavyweight SPAs like
    business.facebook.com routinely takes 30+ seconds. The bridge timeout
    (15s) fires first, the LLM hears \"timeout\", and the user gets a
    bewildering \"the session timed out\" message while the navigation is
    actually progressing fine. ensure_view exists precisely to break that
    fast/slow conflation.

    We always dispatch (no DB-state short-circuit). Electron's view
    registry is the source of truth for whether a view exists; the DB row's
    status field is informational only and can drift across Electron
    restarts. The cost of always dispatching is one cheap IPC round-trip;
    the cost of a stale short-circuit is lying to the user about browser
    state. Not worth the optimisation.
    """
    db = SessionLocal()
    try:
        sess = session_registry.get_session(db, session_id)
        if not sess:
            return {"ok": False, "error": f"session {session_id} not found"}
    finally:
        db.close()

    return await execute_action(
        session_id,
        "ensure_view",
        {},
        timeout_seconds=timeout_seconds,
    )
