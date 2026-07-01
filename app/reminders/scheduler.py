# FILE: app/reminders/scheduler.py
# Purpose: Reminders background scheduler — 30s poll for punctual desktop firing;
#          the phone gets its punctuality from a local exact alarm, not this loop.
# Called-by: main
# Depends-on: app.db, app.reminders.service, app.reminders.feed
# Last-renovated: 2026-07-01
"""
One asyncio loop, mirroring app/lifestyle/scheduler.py's pattern, but polling
every 30s (not 10 minutes) since a reminder must fire punctual-to-the-minute
on desktop. Each tick:
  1. Find reminders with due_at <= now and fired_at IS NULL -> mark fired.
  2. Refresh data/reminders/active.json with the full due+unacked set, so
     memory_injection's cooldown-gated pickup (app.reminders.feed) sees a
     fresh snapshot immediately after firing.

Env:
  ASTRA_REMINDER_SCHEDULER_ENABLED  (default true)
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_POLL_SECONDS = 30


def _enabled() -> bool:
    return os.getenv("ASTRA_REMINDER_SCHEDULER_ENABLED", "true").strip().lower() not in (
        "0", "false", "no", "off",
    )


class ReminderScheduler:
    def __init__(self) -> None:
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def _tick(self) -> None:
        from app.db import SessionLocal
        from app.reminders import service, feed

        db = SessionLocal()
        try:
            due = service.list_pending_due(db)
            for reminder in due:
                service.mark_fired(db, reminder.id)
                logger.info("[reminder_scheduler] fired reminder #%s: %s", reminder.id, reminder.text)

            unacked = service.list_due_unacked(db)
            feed.write_active_reminders([
                {
                    "id": r.id,
                    "text": r.text,
                    "due_at": r.due_at.isoformat() if r.due_at else "",
                }
                for r in unacked
            ])
        except Exception as exc:
            logger.error("[reminder_scheduler] tick failed: %s", exc)
        finally:
            db.close()

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._tick()
                await asyncio.sleep(_POLL_SECONDS)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("[reminder_scheduler] loop error: %s", exc)
                await asyncio.sleep(60)


_scheduler: Optional[ReminderScheduler] = None


def start_reminder_scheduler_background(loop: Optional[asyncio.AbstractEventLoop] = None) -> bool:
    """Start the scheduler on the given loop. Mirrors
    lifestyle.scheduler.start_lifestyle_scheduler_background so root main.py's
    sync startup hook can call it without awaiting."""
    global _scheduler
    if not _enabled():
        logger.info("[reminder_scheduler] disabled via env")
        return False
    if _scheduler is None:
        _scheduler = ReminderScheduler()
    if _scheduler._running:
        logger.warning("[reminder_scheduler] already running")
        return False
    if loop is None:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            logger.warning("[reminder_scheduler] no event loop")
            return False
    _scheduler._running = True
    _scheduler._task = loop.create_task(_scheduler._loop())
    logger.info("[reminder_scheduler] background task created (poll=%ss)", _POLL_SECONDS)
    return True
