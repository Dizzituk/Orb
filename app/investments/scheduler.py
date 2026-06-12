# FILE: app/investments/scheduler.py
# Purpose: APScheduler jobs for automated portfolio snapshots.
# Called-by: main
# Depends-on: app.db, app.investments.service
# Last-renovated: 2026-06-11
"""
APScheduler jobs for automated portfolio snapshots.

Runs:
- Morning snapshot at 08:00 UK time
- Evening snapshot at 18:00 UK time

Registers with FastAPI lifespan for clean start/stop.
"""
import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.db import SessionLocal

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


async def _run_scheduled_snapshot(label: str) -> None:
    """Execute a scheduled snapshot in a fresh DB session."""
    from app.investments.service import take_snapshot

    logger.info("[investments-scheduler] Starting %s snapshot", label)
    db = SessionLocal()
    try:
        result = await take_snapshot(db, snapshot_type="scheduled")
        logger.info(
            "[investments-scheduler] %s snapshot complete: %s (%d positions)",
            label, result.message, result.position_count,
        )
    except Exception as exc:
        logger.error(
            "[investments-scheduler] %s snapshot failed: %s",
            label, exc,
        )
    finally:
        db.close()


async def _morning_snapshot() -> None:
    await _run_scheduled_snapshot("morning")


async def _evening_snapshot() -> None:
    await _run_scheduled_snapshot("evening")


def start_scheduler() -> AsyncIOScheduler:
    """Create and start the snapshot scheduler."""
    global _scheduler

    if _scheduler is not None:
        return _scheduler

    _scheduler = AsyncIOScheduler()

    # UK timezone (handles BST/GMT automatically)
    _scheduler.add_job(
        _morning_snapshot,
        CronTrigger(hour=8, minute=0, timezone="Europe/London"),
        id="investments_morning_snapshot",
        name="Morning portfolio snapshot",
        replace_existing=True,
    )

    _scheduler.add_job(
        _evening_snapshot,
        CronTrigger(hour=18, minute=0, timezone="Europe/London"),
        id="investments_evening_snapshot",
        name="Evening portfolio snapshot",
        replace_existing=True,
    )

    _scheduler.start()
    logger.info("[investments-scheduler] Started (08:00 + 18:00 UK)")
    return _scheduler


def stop_scheduler() -> None:
    """Shutdown the scheduler gracefully."""
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        logger.info("[investments-scheduler] Stopped")
        _scheduler = None


@asynccontextmanager
async def investments_lifespan(app):
    """
    FastAPI lifespan hook for the investments scheduler.

    Usage in main.py:
        from app.investments.scheduler import investments_lifespan
        # Register in startup
    """
    start_scheduler()
    yield
    stop_scheduler()
