# FILE: app/tools/reminder_tools.py
# Purpose: Handlers for the reminder chat tools (create/list/cancel) — desktop+phone parity.
# Called-by: app.tools.reminder_tools_registration
# Depends-on: app.db, app.reminders.service, app.reminders.time_parse
# Last-renovated: 2026-07-03
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)


@contextmanager
def _db_session():
    """Yield a short-lived DB session, always closing on exit."""
    from app.db import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def create_reminder_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Parse `when` and create a one-shot reminder that fires on desktop + phone alike."""
    text = str(input_data.get("text") or "").strip()
    when = str(input_data.get("when") or "").strip()
    if not text:
        return {"ok": False, "reminder_id": None, "due_at": None, "error": "text is required"}
    if not when:
        return {"ok": False, "reminder_id": None, "due_at": None, "error": "when is required"}

    from app.reminders.time_parse import parse_when
    due_at = parse_when(when, model_due_at_iso=input_data.get("due_at_iso"))
    if due_at is None:
        # Self-heal path (2026-07-03): the model must NOT bounce this back to
        # the user ("say it as a clock time") — it can compute the timestamp
        # itself from the datetime in its prompt context and retry once.
        return {
            "ok": False, "reminder_id": None, "due_at": None,
            "error": (
                f"Could not work out a time from {when!r}. Retry create_reminder NOW "
                "with the same text and when, plus due_at_iso set to the exact "
                "ISO-8601 local datetime you compute from the current date/time in "
                "your context — do not ask the user to rephrase."
            ),
        }

    try:
        from app.reminders.service import create
        with _db_session() as db:
            reminder = create(db, text=text, due_at=due_at)
        return {
            "ok": True, "reminder_id": reminder.id,
            "due_at": reminder.due_at.isoformat(), "text": reminder.text,
        }
    except Exception as exc:
        logger.exception("[reminder_tools] create_reminder failed")
        return {"ok": False, "reminder_id": None, "due_at": None, "error": str(exc)}


async def list_reminders_handler(input_data: dict, context: Optional[dict]) -> dict:
    """List upcoming (unfired) and currently due-and-unacked reminders."""
    try:
        from app.reminders.service import list_upcoming, list_due_unacked
        with _db_session() as db:
            upcoming = list_upcoming(db, within_hours=float(input_data.get("within_hours") or 168.0))
            due = list_due_unacked(db)
            combined = {r.id: r for r in [*upcoming, *due]}
            items = [
                {
                    "id": r.id, "text": r.text,
                    "due_at": r.due_at.isoformat() if r.due_at else "",
                    "fired": r.fired_at is not None,
                }
                for r in sorted(combined.values(), key=lambda row: row.due_at)
            ]
        return {"ok": True, "reminders": items}
    except Exception as exc:
        logger.exception("[reminder_tools] list_reminders failed")
        return {"ok": False, "reminders": [], "error": str(exc)}


async def cancel_reminder_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Cancel a pending reminder by id."""
    try:
        reminder_id = int(input_data.get("reminder_id"))
    except (TypeError, ValueError):
        return {"ok": False, "error": "reminder_id must be an integer"}

    try:
        from app.reminders.service import cancel
        with _db_session() as db:
            ok = cancel(db, reminder_id)
        return {"ok": ok, "error": None if ok else "Reminder not found"}
    except Exception as exc:
        logger.exception("[reminder_tools] cancel_reminder failed")
        return {"ok": False, "error": str(exc)}
