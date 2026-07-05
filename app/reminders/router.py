# FILE: app/reminders/router.py
# Purpose: Core Reminders API for the desktop app (create/list/range/ack/cancel/announce).
# Called-by: main (include_router), orb-desktop reminderApi.ts
# Depends-on: app.auth, app.db, app.reminders.service, app.reminders.time_parse, app.llm.routing.core
# Last-renovated: 2026-07-03
"""
Follows the same pattern as app/lifestyle/router.py: prefix-scoped router,
auth on every endpoint via Depends(require_auth). Desktop is the only caller
today (chat covers phone-side create/cancel via the reminder_tools LLM
tools).

2026-07-03: /range feeds the desktop Calendar tab's month grid;
/{id}/announce hands the desktop ReminderWatcher a casual LLM-phrased spoken
line for TTS and stamps delivered_at (first announce wins — a second caller
gets already_delivered=true and should stay silent).
"""
from __future__ import annotations

import logging
from datetime import datetime
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
    delivered_at: Optional[str] = None
    source_device: Optional[str] = None


def _utc_iso(dt) -> Optional[str]:
    """DB DateTime columns hold UTC wall time but read back NAIVE (the sqlite
    dialect drops the offset on write). Emitting naive ISO made the desktop
    parse UTC as local (times displayed an hour early in BST) and made the
    phone's OffsetDateTime.parse throw — so its exact alarms never armed
    (2026-07-03 evening fix). Stamp the UTC offset back on before emitting."""
    if not dt:
        return None
    if dt.tzinfo is None:
        from datetime import timezone as _tz
        dt = dt.replace(tzinfo=_tz.utc)
    return dt.isoformat()


def _to_out(r) -> ReminderOut:
    return ReminderOut(
        id=r.id,
        text=r.text,
        due_at=_utc_iso(r.due_at) or "",
        created_at=_utc_iso(r.created_at) or "",
        fired_at=_utc_iso(r.fired_at),
        acked_at=_utc_iso(r.acked_at),
        delivered_at=_utc_iso(r.delivered_at),
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


@router.get("/range", response_model=List[ReminderOut])
def list_reminders_range(start_iso: str, end_iso: str, db: Session = Depends(get_db)):
    """Every reminder (any state) due inside [start_iso, end_iso) — the Calendar
    tab's month-grid feed. States let the UI style upcoming / due / done."""
    try:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
    except ValueError:
        raise HTTPException(status_code=400, detail="start_iso/end_iso must be ISO 8601")
    return [_to_out(r) for r in service.list_in_range(db, start, end)]


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


# ── Announce (2026-07-03) ────────────────────────────────────────────

from app.reminders.announce import (
    FRESH_DELIVERY_WINDOW_S,
    announce_line,
    fallback_line,
    recall_line,
)


class AnnounceOut(BaseModel):
    ok: bool
    spoken_text: str
    speak: bool
    already_delivered: bool


@router.post("/{reminder_id}/announce", response_model=AnnounceOut)
async def announce_reminder(reminder_id: int, db: Session = Depends(get_db)):
    """Desktop watcher calls this when it surfaces a fired reminder.

    speak=True → chime + TTS the line. The scheduler's fire-time chat-turn
    delivery stamps delivered_at seconds before the watcher's next poll, so
    a FRESH delivery (≤ FRESH_DELIVERY_WINDOW_S old) keeps speak=True and
    returns the SAME cached line that landed in the chat — Astra's voice
    matches her message. Stale deliveries (desktop was off; phone alarm and
    chat history already told him) return speak=False: silent toast only."""
    reminder = service.get(db, reminder_id)
    if reminder is None:
        raise HTTPException(status_code=404, detail="Reminder not found")

    if reminder.delivered_at is None:
        spoken = await announce_line(reminder.text)
        service.mark_delivered(db, reminder_id)
        return AnnounceOut(ok=True, spoken_text=spoken, speak=True, already_delivered=False)

    age_s = (datetime.utcnow() - reminder.delivered_at).total_seconds()
    line = recall_line(reminder_id) or fallback_line(reminder.text)
    return AnnounceOut(
        ok=True,
        spoken_text=line,
        speak=age_s <= FRESH_DELIVERY_WINDOW_S,
        already_delivered=True,
    )
