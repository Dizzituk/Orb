# FILE: app/reminders/service.py
# Purpose: CRUD + query helpers for reminders — the DB is the source of truth for
#          every surface (router.py, bridge feed, scheduler all go through here).
# Called-by: app.reminders.router, app.bridge.reminders_feed, app.reminders.scheduler, app.tools.reminder_tools
# Depends-on: app.db, app.reminders.models
# Last-renovated: 2026-07-01
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from app.reminders.models import Reminder


def _now() -> datetime:
    return datetime.now(timezone.utc)


def create(
    db: Session,
    text: str,
    due_at: datetime,
    source_device: Optional[str] = None,
) -> Reminder:
    """due_at may be tz-naive-local or tz-aware; always stored as UTC."""
    if due_at.tzinfo is None:
        due_at = due_at.astimezone()
    reminder = Reminder(
        text=text.strip(),
        due_at=due_at.astimezone(timezone.utc),
        source_device=source_device,
    )
    db.add(reminder)
    db.commit()
    db.refresh(reminder)
    return reminder


def get(db: Session, reminder_id: int) -> Optional[Reminder]:
    return db.query(Reminder).filter(Reminder.id == reminder_id).first()


def list_upcoming(db: Session, within_hours: float = 24.0) -> List[Reminder]:
    """Future, unfired reminders due within the window — for phone alarm scheduling."""
    horizon = _now() + timedelta(hours=within_hours)
    return (
        db.query(Reminder)
        .filter(Reminder.fired_at.is_(None))
        .filter(Reminder.due_at <= horizon)
        .order_by(Reminder.due_at.asc())
        .all()
    )


def list_due_unacked(db: Session) -> List[Reminder]:
    """Fired but not yet acknowledged — the active nag set, polled by every surface."""
    return (
        db.query(Reminder)
        .filter(Reminder.fired_at.isnot(None))
        .filter(Reminder.acked_at.is_(None))
        .order_by(Reminder.due_at.asc())
        .all()
    )


def list_pending_due(db: Session) -> List[Reminder]:
    """Reminders whose due_at has passed but haven't fired yet — the scheduler's tick query."""
    return (
        db.query(Reminder)
        .filter(Reminder.fired_at.is_(None))
        .filter(Reminder.due_at <= _now())
        .all()
    )


def mark_fired(db: Session, reminder_id: int) -> Optional[Reminder]:
    reminder = get(db, reminder_id)
    if reminder and reminder.fired_at is None:
        reminder.fired_at = _now()
        db.commit()
        db.refresh(reminder)
    return reminder


def ack(db: Session, reminder_id: int) -> Optional[Reminder]:
    reminder = get(db, reminder_id)
    if reminder and reminder.acked_at is None:
        reminder.acked_at = _now()
        db.commit()
        db.refresh(reminder)
    return reminder


def cancel(db: Session, reminder_id: int) -> bool:
    reminder = get(db, reminder_id)
    if not reminder:
        return False
    db.delete(reminder)
    db.commit()
    return True
