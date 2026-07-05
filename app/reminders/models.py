# FILE: app/reminders/models.py
# Purpose: SQLAlchemy model for one-shot reminders (v1) — fires on desktop + phone alike.
# Called-by: app.reminders.service, app.reminders.feed, app.reminders.scheduler, app.reminders.router
# Depends-on: app.db
# Last-renovated: 2026-07-03
"""
Reminder — a single scheduled nudge created from chat on either surface.

v1 is one-shot only: `recurrence` is carried on the row (nullable, unused)
so Phase 2 daily/weekly recurrence doesn't need a schema migration later.

delivered_at (2026-07-03) records the first moment ANY surface actually
announced the fired reminder to Taz — the desktop TTS announce or a
chat-turn injection. It gates the chat-injection channel: only undelivered
reminders may be woven into a conversation, so a reminder announced this
morning can never resurface mid-chat hours later as if new. acked_at stays
the explicit user dismissal that clears the nagging surfaces (phone
notification, desktop toast).
due_at/created_at/fired_at/acked_at all follow the app-wide convention of a
plain (non-tz-typed) DateTime column holding a tz-aware UTC datetime.now(timezone.utc)
value — see app/lifestyle/models.py for the same pattern.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String

from app.db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Reminder(Base):
    __tablename__ = "reminders"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String(500), nullable=False)
    due_at = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=_now)
    fired_at = Column(DateTime, nullable=True)
    acked_at = Column(DateTime, nullable=True)
    delivered_at = Column(DateTime, nullable=True)  # first real announce to the user — see module docstring
    source_device = Column(String(20), nullable=True)  # "desktop" | "phone" | None
    recurrence = Column(String(40), nullable=True)  # unused in v1 — Phase 2 daily/weekly
