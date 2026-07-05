# FILE: app/reminders/scheduler.py
# Purpose: Reminders background scheduler — 30s poll for punctual desktop firing;
#          the phone gets its punctuality from a local exact alarm, not this loop.
# Called-by: main
# Depends-on: app.db, app.reminders.service, app.reminders.feed
# Last-renovated: 2026-07-03
"""
One asyncio loop, mirroring app/lifestyle/scheduler.py's pattern, but polling
every 30s (not 10 minutes) since a reminder must fire punctual-to-the-minute
on desktop. Each tick:
  1. Find reminders with due_at <= now and fired_at IS NULL -> mark fired,
     then DELIVER AS A CHAT TURN (2026-07-03 evening): compose Astra's one
     casual line and insert it as a real assistant message into the most
     recently active conversation — the desktop's live-message poller shows
     it within ~2.5s, and the phone's missed-replies feed carries it (with
     lazy TTS) like any Astra reply. delivered_at is stamped ONLY on a
     successful insert, so the chat-injection channel below stays live as
     the safety net when this fails.
  2. Refresh data/reminders/active.json with the due+unacked+UNDELIVERED set,
     so memory_injection's pickup (app.reminders.feed) sees a fresh snapshot
     immediately after firing. Delivered reminders are pruned here: once any
     surface has announced one (desktop TTS announce or a chat injection),
     it must never resurface in a later conversation as if new (2026-07-03
     ghost fix). Desktop toasts and phone catch-up read the full due+unacked
     set straight from the DB over REST — they are unaffected.

Env:
  ASTRA_REMINDER_SCHEDULER_ENABLED  (default true)
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# 10s (was 30, 2026-07-03 night): a "remind me in one minute" landing up to
# ~50s late felt broken — fire lag is now ≤10s and the desktop watcher polls
# at 10s too. The tick is a tiny indexed query; active.json only rewrites on
# change (feed.write_active_reminders skips no-op writes).
_POLL_SECONDS = 10


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
                await _deliver_chat_turn(db, reminder)

            undelivered = service.list_due_unacked_undelivered(db)
            feed.write_active_reminders([
                {
                    "id": r.id,
                    "text": r.text,
                    "due_at": r.due_at.isoformat() if r.due_at else "",
                }
                for r in undelivered
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


async def _deliver_chat_turn(db, reminder) -> None:
    """Fire-time proactive delivery (2026-07-03 evening): Astra announces the
    reminder IN THE CHAT — one LLM-phrased line inserted as a real assistant
    message into the most recently active conversation. provider="reminder"
    keeps it out of memory harvesting (create_message early-returns before
    RAG/spine/Nat — the reminders ledger is the artifact, Taz's no-junk
    rule). The line is cached (announce.remember_line) so the desktop
    watcher's /announce call speaks EXACTLY what was written. Never raises;
    on failure delivered_at stays NULL so chat injection catches it."""
    try:
        from app.memory import models as memory_models
        from app.memory.schemas import MessageCreate
        from app.memory.service import create_message
        from app.reminders import announce
        from app.reminders import service as reminders_service

        latest = (
            db.query(memory_models.Message)
            .order_by(memory_models.Message.id.desc())
            .first()
        )
        if latest is None:
            logger.info(
                "[reminder_scheduler] no conversation yet — #%s left to the surfaces",
                reminder.id,
            )
            return

        line = await announce.announce_line(reminder.text)
        create_message(db, MessageCreate(
            project_id=latest.project_id,
            role="assistant",
            content=line,
            provider="reminder",
            model="reminder-announce",
        ))
        announce.remember_line(reminder.id, line)
        reminders_service.mark_delivered(db, reminder.id)
        logger.info(
            "[reminder_scheduler] delivered #%s as a chat turn into project %s",
            reminder.id, latest.project_id,
        )
    except Exception as exc:
        logger.warning(
            "[reminder_scheduler] chat-turn delivery failed for #%s "
            "(chat injection will catch it): %s", reminder.id, exc,
        )


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
