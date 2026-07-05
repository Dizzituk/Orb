# FILE: app/bridge/reminders_feed.py
# Purpose: Phone-facing Reminders feed — upcoming reminders for AstraBridge to arm as
#          exact alarms, plus a due-reminders catch-up feed and an ack endpoint.
# Called-by: main (include_router), Astra-Bridge ReminderSyncWorker
# Depends-on: app.bridge.schemas (require_bridge_auth), app.reminders.service, app.db
# Last-renovated: 2026-07-01
"""
Mirrors app/bridge/sentinel_alerts.py exactly. Two reads + one write:

  GET  /bridge/reminders/upcoming?within_hours=24  -> future, unfired reminders,
       so the phone can (re)schedule its local AlarmManager exact alarms.
  GET  /bridge/due-reminders                        -> fired-but-unacked reminders,
       for missed-reminder catch-up when the phone reconnects after being
       asleep/offline through the exact-alarm's fire time.
  POST /bridge/reminders/{id}/ack                    -> acknowledge from the phone
       (clears the desktop toast and the phone's own persistent notification).
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.reminders import service

from .schemas import require_bridge_auth

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/bridge", tags=["Bridge"])


class BridgeReminderOut(BaseModel):
    id: int
    text: str
    due_at: str


def _to_out(r) -> BridgeReminderOut:
    # UTC offset stamped back on (columns read back naive): AstraBridge's
    # ReminderSyncWorker parses due_at with OffsetDateTime.parse, which THROWS
    # on a naive string — so the phone's exact alarms silently never armed and
    # only the 15-min catch-up poll ever notified (2026-07-03 evening fix).
    due = r.due_at
    if due is not None and due.tzinfo is None:
        from datetime import timezone as _tz
        due = due.replace(tzinfo=_tz.utc)
    return BridgeReminderOut(id=r.id, text=r.text, due_at=due.isoformat() if due else "")


@router.get("/reminders/upcoming", response_model=List[BridgeReminderOut])
async def get_upcoming_reminders(
    within_hours: float = 24.0,
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_bridge_auth),
):
    rows = service.list_upcoming(db, within_hours=within_hours)
    return [_to_out(r) for r in rows]


@router.get("/due-reminders", response_model=List[BridgeReminderOut])
async def get_due_reminders(
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_bridge_auth),
):
    rows = service.list_due_unacked(db)
    return [_to_out(r) for r in rows]


class BridgeReminderAckResponse(BaseModel):
    ok: bool


@router.post("/reminders/{reminder_id}/ack", response_model=BridgeReminderAckResponse)
async def ack_reminder(
    reminder_id: int,
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_bridge_auth),
):
    reminder = service.ack(db, reminder_id)
    if reminder is None:
        raise HTTPException(status_code=404, detail="Reminder not found")
    return BridgeReminderAckResponse(ok=True)
