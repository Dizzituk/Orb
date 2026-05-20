# FILE: app/web_automation/router.py
"""
FastAPI router for the Web Automation module.

Two audiences:
  (a) Electron main process — polls for pending actions, posts results,
      reports live session state. These endpoints are NOT auth-wrapped:
      they run on localhost between the trusted backend and the trusted
      local shell. main.py's include_router is intentionally called
      without a global auth dependency for this reason.
  (b) The React frontend / debug UI / external callers — lists sessions,
      dispatches manual actions, inspects history. These ARE auth-wrapped
      per-endpoint.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth import require_auth
from app.db import get_db
from app.web_automation import action_queue, bridge, session_registry
from app.web_automation.models import SessionStatus, WebAction
from app.web_automation.schemas import (
    ActionOut,
    ActionRequest,
    ActionResultIn,
    PendingActionOut,
    SessionStatusIn,
    WebSessionCreate,
    WebSessionOut,
    WebSessionPatch,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/web_automation", tags=["Web Automation"])


# ══════════════════════════════════════════════════════════════════════
# Session CRUD (auth-gated — user/React-UI facing)
# ══════════════════════════════════════════════════════════════════════

@router.get("/sessions", response_model=List[WebSessionOut])
def list_sessions_endpoint(
    platform: Optional[str] = None,
    db: Session = Depends(get_db),
    _auth=Depends(require_auth),
):
    return session_registry.list_sessions(db, platform=platform)


@router.post("/sessions", response_model=WebSessionOut)
def create_session_endpoint(
    req: WebSessionCreate,
    db: Session = Depends(get_db),
    _auth=Depends(require_auth),
):
    return session_registry.create_session(
        db,
        platform=req.platform,
        label=req.label,
        partition=req.partition,
        landing_url=req.landing_url,
        purpose=req.purpose,
    )


@router.patch("/sessions/{session_id}", response_model=WebSessionOut)
def update_session_endpoint(
    session_id: str,
    req: WebSessionPatch,
    db: Session = Depends(get_db),
    _auth=Depends(require_auth),
):
    sess = session_registry.update_session(
        db, session_id,
        label=req.label,
        landing_url=req.landing_url,
        purpose=req.purpose,
    )
    if not sess:
        raise HTTPException(404, f"session {session_id} not found")
    return sess


@router.delete("/sessions/{session_id}")
def delete_session_endpoint(
    session_id: str,
    db: Session = Depends(get_db),
    _auth=Depends(require_auth),
):
    ok = session_registry.delete_session(db, session_id)
    if not ok:
        raise HTTPException(404, f"session {session_id} not found")
    return {"deleted": True}


# ══════════════════════════════════════════════════════════════════════
# Electron-facing endpoints (NO auth — hot path on localhost)
# ══════════════════════════════════════════════════════════════════════

@router.get("/electron-bootstrap")
def electron_bootstrap_endpoint(db: Session = Depends(get_db)):
    """
    Minimal session list for the Electron main process on startup.
    No auth — returns only partition + landing URL + labels so Electron
    can start poll loops. No PII, safe for local-trusted callers.
    Mirrors the trust model of /pending-action and /action-result.
    """
    sessions = session_registry.list_sessions(db)
    return [
        {
            "id": s.id,
            "platform": s.platform,
            "label": s.label,
            "partition": s.partition,
            "landing_url": s.landing_url,
        }
        for s in sessions
    ]


@router.get("/pending-action/{session_id}", response_model=PendingActionOut)
async def pending_action_endpoint(
    session_id: str,
    wait: float = 5.0,
    db: Session = Depends(get_db),
):
    """Electron long-polls this. Returns the next action (or empty)."""
    wait = max(0.0, min(30.0, wait))
    action_id = await action_queue.wait_for_next(session_id, timeout_seconds=wait)
    if not action_id:
        return PendingActionOut()

    action = action_queue.mark_in_flight(db, action_id)
    if not action:
        return PendingActionOut()

    return PendingActionOut(
        action_id=action.id,
        session_id=action.session_id,
        action_type=action.action_type,
        payload=action.payload,
    )


@router.post("/action-result/{action_id}")
def action_result_endpoint(
    action_id: str,
    result: ActionResultIn,
    db: Session = Depends(get_db),
):
    """Electron posts the outcome of a dispatched action here."""
    action = action_queue.resolve_action(
        db, action_id,
        ok=result.ok,
        result=result.result,
        error=result.error,
    )
    if not action:
        raise HTTPException(404, f"action {action_id} not found")

    if result.current_url or result.current_title:
        session_registry.set_status(
            db, action.session_id,
            current_url=result.current_url,
            current_title=result.current_title,
        )
    return {"resolved": True}


@router.post("/session-status/{session_id}")
def session_status_endpoint(
    session_id: str,
    body: SessionStatusIn,
    db: Session = Depends(get_db),
):
    """Electron reports live session transitions (opened/closed/crashed)."""
    try:
        status = SessionStatus(body.status)
    except ValueError:
        raise HTTPException(400, f"invalid status '{body.status}'")
    sess = session_registry.set_status(
        db, session_id,
        status=status,
        current_url=body.current_url,
        current_title=body.current_title,
        last_error=body.error,
    )
    if not sess:
        raise HTTPException(404, f"session {session_id} not found")
    return {"updated": True, "status": sess.status.value}


# ══════════════════════════════════════════════════════════════════════
# Manual execute (auth-gated — testing / React UI)
# ══════════════════════════════════════════════════════════════════════

@router.post("/execute")
async def execute_endpoint(
    req: ActionRequest,
    _auth=Depends(require_auth),
):
    return await bridge.execute_action(
        req.session_id,
        req.action_type,
        req.payload,
        timeout_seconds=req.timeout_seconds,
    )


# ══════════════════════════════════════════════════════════════════════
# Action history (auth-gated)
# ══════════════════════════════════════════════════════════════════════

@router.get("/sessions/{session_id}/actions", response_model=List[ActionOut])
def session_actions_endpoint(
    session_id: str,
    limit: int = 50,
    db: Session = Depends(get_db),
    _auth=Depends(require_auth),
):
    q = (
        db.query(WebAction)
        .filter(WebAction.session_id == session_id)
        .order_by(WebAction.created_at.desc())
        .limit(max(1, min(500, limit)))
    )
    return list(q.all())
