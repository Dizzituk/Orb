# FILE: app/web_automation/action_queue.py
# Purpose: Per-session in-memory action queue with asyncio Futures for result delivery.
# Called-by: app.web_automation.bridge, app.web_automation.router
# Depends-on: app.db, app.web_automation.models
# Last-renovated: 2026-07-02
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
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, Optional, Tuple

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
        stalled_status, desktop_alive = _mark_timed_out(action_id)
        _result_futures.pop(action_id, None)
        if stalled_status == ActionStatus.pending and not desktop_alive:
            return {
                "ok": False,
                "error": (
                    f"{DESKTOP_OFFLINE_PREFIX} — the ASTRA desktop app never "
                    f"picked this action up ({timeout_seconds:.0f}s) and shows "
                    "no recent activity for this session. It is probably not "
                    "running. Tell the user to start the ASTRA desktop app, "
                    "then retry. This is not a page problem."
                ),
            }
        if stalled_status == ActionStatus.pending and desktop_alive:
            return {
                "ok": False,
                "error": (
                    f"browser is busy on a previous action — the desktop app "
                    f"IS running but is still executing earlier work for this "
                    f"session, so this action was not picked up within "
                    f"{timeout_seconds:.0f}s. Wait a few seconds and retry; "
                    "do NOT tell the user the desktop is offline."
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


# How recently Electron must have taken work for a session to count as
# alive. Electron long-polls every ~5s when idle, so anything inside this
# window means the app is up (a stuck-but-delivered action holds the
# session busy far longer — that case is covered by the in_flight check).
_ALIVE_WINDOW_SECONDS = 45


def _session_shows_life(db: Session, session_id: str, exclude_action_id: str) -> bool:
    """Evidence Electron is alive for this session despite an undelivered
    action: some OTHER action currently in_flight (taken, not yet
    answered — e.g. a wedged navigate holding the serial queue), or any
    delivery within the last _ALIVE_WINDOW_SECONDS."""
    in_flight = (
        db.query(WebAction.id)
        .filter(
            WebAction.session_id == session_id,
            WebAction.id != exclude_action_id,
            WebAction.status == ActionStatus.in_flight,
        )
        .first()
    )
    if in_flight:
        return True
    last_delivered = (
        db.query(WebAction.delivered_at)
        .filter(
            WebAction.session_id == session_id,
            WebAction.id != exclude_action_id,
            WebAction.delivered_at.isnot(None),
        )
        .order_by(WebAction.delivered_at.desc())
        .first()
    )
    if not last_delivered or not last_delivered[0]:
        return False
    ts = last_delivered[0]
    if ts.tzinfo is None:  # SQLite hands naive UTC back
        ts = ts.replace(tzinfo=timezone.utc)
    return (_now() - ts) <= timedelta(seconds=_ALIVE_WINDOW_SECONDS)


def _mark_timed_out(action_id: str) -> Tuple[Optional[ActionStatus], bool]:
    """Mark a stalled action timed_out. Returns (status it stalled in,
    desktop-alive evidence) so await_result can phrase the error:
    pending+dead → offline, pending+alive → busy, in_flight → page stall.
    (None, False) if the action is unknown or already resolved."""
    db = SessionLocal()
    try:
        action = db.query(WebAction).filter(WebAction.id == action_id).one_or_none()
        if action and action.status in (ActionStatus.pending, ActionStatus.in_flight):
            stalled_status = action.status
            action.status = ActionStatus.timed_out
            action.completed_at = _now()
            db.commit()
            desktop_alive = False
            if stalled_status == ActionStatus.pending:
                desktop_alive = _session_shows_life(db, action.session_id, action_id)
            return stalled_status, desktop_alive
        return None, False
    finally:
        db.close()


def queue_depth(session_id: str) -> int:
    return len(_pending.get(session_id, []))
