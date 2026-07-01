# FILE: app/reminders/router.py
# Purpose: Core Reminders API for the desktop app (create/list/ack/cancel).
# Called-by: main (include_router), orb-desktop reminderApi.ts
# Depends-on: app.auth, app.db, app.reminders.service, app.reminders.time_parse
# Last-renovated: 2026-07-01
"""
Follows the same pattern as app/lifestyle/router.py: prefix-scoped router,
auth on every endpoint via Depends(require_auth). Desktop is the only caller
today (chat covers phone-side create/cancel via the reminder_tools LLM
tools) — this REST surface exists so a future desktop panel doesn't need a
new backend endpoint.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import require_auth
from app.db import get_db
from app.reminders import service
from app.reminders.time_parse import parse_when

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/reminders",
    tags=["Reminders"],
    dependencies=[Depends(require_auth)],
)


class ReminderCreateRequest(BaseModel):
    text: str
    when: str
    source_device: Optional[str] = None
    model_due_at_iso: Optional[str] = None


class ReminderOut(BaseModel):
    id: int
    text: str
    due_at: str
    created_at: str
    fired_at: Optional[str] = None
    acked_at: Optional[str] = None
    source_device: Optional[str] = None


def _to_out(r) -> ReminderOut:
    return ReminderOut(
        id=r.id,
        text=r.text,
        due_at=r.due_at.isoformat() if r.due_at else "",
        created_at=r.created_at.isoformat() if r.created_at else "",
        fired_at=r.fired_at.isoformat() if r.fired_at else None,
        acked_at=r.acked_at.isoformat() if r.acked_at else None,
        source_device=r.source_device,
    )


@router.post("", response_model=ReminderOut)
def create_reminder(body: ReminderCreateRequest, db: Session = Depends(get_db)):
    due_at = parse_when(body.when, model_due_at_iso=body.model_due_at_iso)
    if due_at is None:
        raise HTTPException(status_code=400, detail=f"Could not parse a time from: {body.when!r}")
    reminder = service.create(db, text=body.text, due_at=due_at, source_device=body.source_device)
    return _to_out(reminder)


@router.get("", response_model=List[ReminderOut])
def list_reminders(within_hours: float = 24.0, db: Session = Depends(get_db)):
    """Upcoming (unfired) reminders plus currently due-and-unacked ones."""
    upcoming = service.list_upcoming(db, within_hours=within_hours)
    due = service.list_due_unacked(db)
    combined = {r.id: r for r in [*upcoming, *due]}
    return [_to_out(r) for r in sorted(combined.values(), key=lambda r: r.due_at)]


@router.post("/{reminder_id}/ack", response_model=ReminderOut)
def ack_reminder(reminder_id: int, db: Session = Depends(get_db)):
    reminder = service.ack(db, reminder_id)
    if reminder is None:
        raise HTTPException(status_code=404, detail="Reminder not found")
    return _to_out(reminder)


@router.delete("/{reminder_id}")
def cancel_reminder(reminder_id: int, db: Session = Depends(get_db)):
    ok = service.cancel(db, reminder_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Reminder not found")
    return {"ok": True}
