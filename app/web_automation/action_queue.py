# FILE: app/web_automation/action_queue.py
# Purpose: Per-session in-memory action queue with asyncio Futures for result delivery.
# Called-by: app.web_automation.bridge, app.web_automation.router
# Depends-on: app.db, app.web_automation.models
# Last-renovated: 2026-07-01
"""
Per-session in-memory action queue with asyncio Futures for result delivery.

The DB is authoritative for action identity + audit.
The in-memory queue is the live dispatch channel to Electron.

Flow:
  1. bridge.execute_action  -> DB row created (status=pending) + enqueued
  2. Electron long-polls     -> dequeue & mark in_flight
  3. Electron POSTs result   -> mark completed, set the Future
  4. bridge.execute_action   -> awaited Future resolves -> caller gets result

If Electron never responds, await_result times out and marks the action
as timed_out in the DB.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Deque, Dict, Optional

from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.web_automation.models import WebAction, ActionStatus

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Module-level in-memory state ─────────────────────────────────────
# session_id -> deque[action_id]
_pending: Dict[str, Deque[str]] = defaultdict(deque)
# action_id -> asyncio.Future (resolves to {"ok", "result", "error"})
_result_futures: Dict[str, asyncio.Future] = {}


def enqueue_action(db: Session, session_id: str, action_type: str, payload: dict) -> WebAction:
    """Persist the action and add it to the in-memory queue."""
    action = WebAction(
        session_id=session_id,
        action_type=action_type,
        payload=payload or {},
        status=ActionStatus.pending,
    )
    db.add(action)
    db.commit()
    db.refresh(action)

    _pending[session_id].append(action.id)
    loop = asyncio.get_event_loop()
    _result_futures[action.id] = loop.create_future()
    logger.debug("[web_automation] enqueued %s (%s) for session %s",
                 action_type, action.id[:8], session_id[:8])
    return action


async def wait_for_next(session_id: str, timeout_seconds: float = 5.0) -> Optional[str]:
    """
    Long-poll helper for Electron.
    Returns the next action_id for this session, or None if the wait expires.
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_seconds
    while loop.time() < deadline:
        queue = _pending.get(session_id)
        if queue:
            try:
                return queue.popleft()
            except IndexError:
                pass
        try:
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            return None
    return None


def mark_in_flight(db: Session, action_id: str) -> Optional[WebAction]:
    action = db.query(WebAction).filter(WebAction.id == action_id).one_or_none()
    if not action:
        return None
    action.status = ActionStatus.in_flight
    action.delivered_at = _now()
    db.commit()
    return action


def resolve_action(
    db: Session,
    action_id: str,
    *,
    ok: bool,
    result: Optional[dict] = None,
    error: Optional[str] = None,
) -> Optional[WebAction]:
    action = db.query(WebAction).filter(WebAction.id == action_id).one_or_none()
    if not action:
        logger.warning("[web_automation] resolve_action: unknown action_id %s", action_id)
        return None

    action.status = ActionStatus.completed if ok else ActionStatus.failed
    action.result = result if ok else None
    action.error = error
    action.completed_at = _now()
    db.commit()

    fut = _result_futures.pop(action_id, None)
    if fut and not fut.done():
        fut.set_result({"ok": ok, "result": result, "error": error})
    return action


# Stable prefix other layers key on (composite helpers, tool descriptions,
# tests). An error starting with this means "Electron never picked the
# action up", as opposed to a page-level failure inside a live browser.
DESKTOP_OFFLINE_PREFIX = "desktop browser is offline"


async def await_result(action_id: str, timeout_seconds: float = 30.0) -> dict:
    """Block until the action resolves (or the timeout elapses).

    On timeout the error string distinguishes the two very different
    failure modes so the LLM can tell the user the right thing:
      • action still `pending`  → Electron never dequeued it: the desktop
        app isn't running (or its poller is down). NOT a page problem.
      • action was `in_flight`  → Electron took it but never answered:
        the page/executor stalled. The desktop IS up.
    """
    fut = _result_futures.get(action_id)
    if fut is None:
        return {"ok": False, "error": "no future registered for action"}
    try:
        return await asyncio.wait_for(fut, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        stalled_status = _mark_timed_out(action_id)
        _result_futures.pop(action_id, None)
        if stalled_status == ActionStatus.pending:
            return {
                "ok": False,
                "error": (
                    f"{DESKTOP_OFFLINE_PREFIX} — the ASTRA desktop app never "
                    f"picked this action up ({timeout_seconds:.0f}s). It is "
                    "probably not running. Tell the user to start the ASTRA "
                    "desktop app, then retry. This is not a page problem."
                ),
            }
        return {
            "ok": False,
            "error": (
                f"page did not respond — the desktop browser accepted the "
                f"action but returned no result within {timeout_seconds:.0f}s. "
                "The page may still be loading; retry once or check "
                "web_current_state."
            ),
        }


def _mark_timed_out(action_id: str) -> Optional[ActionStatus]:
    """Mark a stalled action timed_out. Returns the status it stalled in
    (pending / in_flight) so await_result can phrase the error, or None
    if the action is unknown or already resolved."""
    db = SessionLocal()
    try:
        action = db.query(WebAction).filter(WebAction.id == action_id).one_or_none()
        if action and action.status in (ActionStatus.pending, ActionStatus.in_flight):
            stalled_status = action.status
            action.status = ActionStatus.timed_out
            action.completed_at = _now()
            db.commit()
            return stalled_status
        return None
    finally:
        db.close()


def queue_depth(session_id: str) -> int:
    return len(_pending.get(session_id, []))
