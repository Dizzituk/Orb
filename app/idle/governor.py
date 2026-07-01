# FILE: app/idle/governor.py
# Purpose: Activity-based idle detection + ledger drain loop (the idle governor).
# Called-by: main.py (startup), app.memory.service (record_user_activity hook)
# Depends-on: app.idle.ledger, app.idle.router, app.db
# Last-renovated: 2026-07-01
"""
The only activity-based scheduler in the codebase (everything else is
time-triggered). The message-creation chokepoint in app/memory/service.py
stamps user activity here; once no user message has arrived for IDLE_MINUTES
the governor drains the persistent task ledger, one task at a time.

A new user message never kills work mid-flight: handlers check
ctx.should_yield() between resumable units (the stamp makes it flip True),
persist progress, and yield. The next idle window resumes them first.

Env knobs (all read live, no restart needed):
    IDLE_MINUTES                  minutes of chat silence before draining (default 10)
    ASTRA_IDLE_GOVERNOR_ENABLED   kill-switch (default true)
    ASTRA_IDLE_POLL_SECONDS       loop poll interval (default 15)
    ASTRA_IDLE_CATCHUP_MINUTES    cadence re-check interval while up (default 10)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Callable, Optional

from sqlalchemy.orm import Session

from app.idle import ledger, router

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    return os.getenv("ASTRA_IDLE_GOVERNOR_ENABLED", "true").strip().lower() not in ("0", "false", "no", "off")


def _idle_minutes() -> float:
    try:
        return float(os.getenv("IDLE_MINUTES", "10"))
    except Exception:
        return 10.0


def _poll_seconds() -> float:
    try:
        return float(os.getenv("ASTRA_IDLE_POLL_SECONDS", "15"))
    except Exception:
        return 15.0


def _catchup_seconds() -> float:
    try:
        return float(os.getenv("ASTRA_IDLE_CATCHUP_MINUTES", "10")) * 60.0
    except Exception:
        return 600.0


def _default_session_factory() -> Session:
    from app.db import get_db_session

    return get_db_session()


class IdleGovernor:
    def __init__(self, session_factory: Optional[Callable[[], Session]] = None):
        self._session_factory = session_factory or _default_session_factory
        self._running = False
        self._stop_requested = False
        self._task: Optional[asyncio.Task] = None
        self._last_activity = time.monotonic()
        self._last_catchup = 0.0
        self._booted = False
        self._current_task_key: Optional[str] = None

    # ── activity signal (called from the message chokepoint) ──────────────

    def record_activity(self) -> None:
        self._last_activity = time.monotonic()

    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_activity

    def is_idle(self) -> bool:
        return _enabled() and self.idle_seconds() >= _idle_minutes() * 60.0

    def activity_after(self, mark: float) -> bool:
        return self._last_activity > mark

    def status(self) -> dict:
        return {
            "enabled": _enabled(),
            "idle_minutes_threshold": _idle_minutes(),
            "idle_seconds": round(self.idle_seconds(), 1),
            "draining": self.is_idle(),
            "current_task": self._current_task_key,
        }

    # ── boot + drain ───────────────────────────────────────────────────────

    def boot_catchup(self) -> None:
        """Recover crash-stranded rows and enqueue anything due since the
        last shutdown (e.g. today's watcher scrapes)."""
        router.ensure_builtin_tasks_registered()
        db = self._session_factory()
        try:
            ledger.recover_stale_running(db)
            router.catch_up(db)
            self._last_catchup = time.monotonic()
        finally:
            db.close()

    async def run_pending(self, max_tasks: Optional[int] = None) -> int:
        """Drain runnable tasks while idle. Returns the number processed
        (completed/paused/failed/skipped all count)."""
        processed = 0
        while self.is_idle() and not self._stop_requested:
            db = self._session_factory()
            try:
                if time.monotonic() - self._last_catchup >= _catchup_seconds():
                    router.catch_up(db)
                    self._last_catchup = time.monotonic()
                rec = ledger.next_runnable(db)
                if rec is None:
                    break
                await self._run_one(db, rec)
                processed += 1
            finally:
                db.close()
            if max_tasks is not None and processed >= max_tasks:
                break
        return processed

    async def _run_one(self, db: Session, rec) -> None:
        handler = router.get_handler(rec.task_type)
        if handler is None:
            ledger.mark_failed(db, rec, f"no handler registered for task_type={rec.task_type!r}")
            return

        try:
            params = json.loads(rec.params_json or "{}")
        except Exception:
            params = {}

        # Unchanged inputs mean skip — the no-repeated-work rule.
        fingerprint: Optional[str] = None
        fp_fn = router.get_fingerprint_fn(rec.task_type)
        if fp_fn is not None:
            try:
                fingerprint = fp_fn(params)
            except Exception as exc:
                logger.warning("[idle.governor] fingerprint failed for %s: %s (running anyway)", rec.task_key, exc)
            if fingerprint:
                last = ledger.last_completed_fingerprint(db, rec.task_key)
                if last is not None and last == fingerprint:
                    ledger.mark_skipped(db, rec, fingerprint, "inputs unchanged since last completed run")
                    return

        ledger.mark_running(db, rec)
        self._current_task_key = rec.task_key
        started = time.monotonic()
        ctx = router.TaskContext(
            db=db,
            record=rec,
            params=params,
            fingerprint=fingerprint,
            should_yield=lambda: self.activity_after(started) or self._stop_requested,
            session_factory=self._session_factory,
        )
        try:
            outcome = await handler(ctx)
        except Exception as exc:
            logger.exception("[idle.governor] task %s crashed", rec.task_key)
            ledger.mark_failed(db, rec, str(exc))
            return
        finally:
            self._current_task_key = None

        if outcome.status == "completed":
            ledger.mark_completed(db, rec, fingerprint=fingerprint, coverage=outcome.coverage, summary=outcome.summary)
        elif outcome.status == "paused":
            ledger.mark_paused(db, rec)
        else:
            ledger.mark_failed(db, rec, outcome.error or "handler reported failure")

    # ── background loop ────────────────────────────────────────────────────

    async def _loop(self) -> None:
        while self._running:
            try:
                if not self._booted:
                    self._booted = True
                    self.boot_catchup()
                if self.is_idle():
                    await self.run_pending()
                await asyncio.sleep(_poll_seconds())
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("[idle.governor] loop error: %s", exc)
                await asyncio.sleep(300)

    def start(self, loop=None) -> bool:
        if self._running:
            logger.warning("[idle.governor] already running")
            return False
        if loop is None:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                return False
        self._running = True
        self._stop_requested = False
        self._task = loop.create_task(self._loop())
        return True

    def stop(self) -> None:
        self._running = False
        self._stop_requested = True
        if self._task is not None:
            self._task.cancel()
            self._task = None


_governor: Optional[IdleGovernor] = None


def get_governor() -> IdleGovernor:
    global _governor
    if _governor is None:
        _governor = IdleGovernor()
    return _governor


def record_user_activity() -> None:
    """Stamp user chat activity. Safe before the governor is started."""
    get_governor().record_activity()


def start_idle_governor_background(loop=None) -> bool:
    if not _enabled():
        logger.info("[idle.governor] disabled via ASTRA_IDLE_GOVERNOR_ENABLED")
        return False
    return get_governor().start(loop=loop)
