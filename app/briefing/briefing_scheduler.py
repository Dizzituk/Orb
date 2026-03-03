# FILE: app/briefing/briefing_scheduler.py
"""
Briefing Scheduler — Manages scheduled and on-demand briefing generation.

Provides:
- Background task that runs on a configurable schedule (daily/weekly)
- On-demand generation triggered via API
- Stores generated briefings for retrieval

Uses asyncio tasks for scheduling rather than external cron/celery
to keep the system self-contained.

v1.0 (2026-03): Initial implementation.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


# =========================================================================
# Briefing storage
# =========================================================================

BRIEFING_STORE_DIR = os.getenv(
    "BRIEFING_STORE_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "briefings"),
)


def _ensure_store_dir() -> Path:
    path = Path(BRIEFING_STORE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class BriefingRecord:
    """Stored briefing metadata."""
    id: str
    frequency: str              # "daily" or "weekly"
    generated_at: str
    title: str
    total_items: int
    text_digest_path: str       # Path to .md file
    audio_path: str = ""        # Path to .mp3 file (if generated)
    astra_alerts: List[str] = field(default_factory=list)


def _save_record(record: BriefingRecord) -> None:
    """Save briefing record to JSON index."""
    store = _ensure_store_dir()
    index_path = store / "briefing_index.json"

    records = []
    if index_path.exists():
        try:
            records = json.loads(index_path.read_text())
        except Exception:
            records = []

    records.append(asdict(record))

    # Keep last 30 briefings
    records = records[-30:]
    index_path.write_text(json.dumps(records, indent=2))


def get_recent_briefings(count: int = 10) -> list[dict]:
    """Get the most recent briefing records."""
    store = _ensure_store_dir()
    index_path = store / "briefing_index.json"
    if not index_path.exists():
        return []
    try:
        records = json.loads(index_path.read_text())
        return records[-count:]
    except Exception:
        return []


def get_latest_briefing() -> Optional[dict]:
    """Get the most recent briefing record."""
    recent = get_recent_briefings(1)
    return recent[0] if recent else None


# =========================================================================
# Briefing generation
# =========================================================================

async def generate_briefing(
    frequency: str = "daily",
    context: Optional[dict] = None,
) -> BriefingRecord:
    """
    Generate a complete briefing: collect → compile → audio.

    Args:
        frequency: "daily" or "weekly"
        context: Optional context dict for web searches

    Returns:
        BriefingRecord with paths to generated files.
    """
    from app.briefing.briefing_collector import collect_all_topics
    from app.briefing.briefing_compiler import compile_briefing
    from app.briefing.briefing_config import get_schedule

    logger.info("[briefing_scheduler] Starting %s briefing generation", frequency)

    # 1. Collect stories
    collection = await collect_all_topics(context=context)
    logger.info(
        "[briefing_scheduler] Collected: %d topics, %d stories",
        len(collection.topics), collection.total_stories,
    )

    # 2. Compile briefing
    briefing = compile_briefing(collection, frequency=frequency)

    # 3. Save text digest
    store = _ensure_store_dir()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    briefing_id = f"{frequency}_{timestamp}"

    digest_path = store / f"{briefing_id}.md"
    digest_path.write_text(briefing.text_digest, encoding="utf-8")
    logger.info("[briefing_scheduler] Text digest saved: %s", digest_path.name)

    # 4. Generate audio (if enabled)
    audio_path = ""
    schedule = get_schedule()
    if schedule.audio_enabled:
        try:
            from app.briefing.briefing_audio import generate_briefing_audio
            result = await generate_briefing_audio(briefing)
            if result:
                audio_path = result
                logger.info("[briefing_scheduler] Audio saved: %s", audio_path)
        except Exception as e:
            logger.warning("[briefing_scheduler] Audio generation failed: %s", e)

    # 5. Save record
    record = BriefingRecord(
        id=briefing_id,
        frequency=frequency,
        generated_at=briefing.generated_at,
        title=briefing.title,
        total_items=briefing.total_items,
        text_digest_path=str(digest_path),
        audio_path=audio_path,
        astra_alerts=briefing.astra_alerts,
    )
    _save_record(record)

    logger.info(
        "[briefing_scheduler] Briefing complete: id=%s, items=%d, audio=%s",
        briefing_id, briefing.total_items, "yes" if audio_path else "no",
    )

    return record


# =========================================================================
# Background scheduler
# =========================================================================

_scheduler_task: Optional[asyncio.Task] = None


async def _scheduler_loop():
    """Background loop that generates briefings on schedule."""
    from app.briefing.briefing_config import get_schedule

    logger.info("[briefing_scheduler] Background scheduler started")

    while True:
        try:
            schedule = get_schedule()
            if not schedule.auto_generate:
                await asyncio.sleep(60)
                continue

            now = datetime.now(timezone.utc)

            # Check if it's time for daily briefing
            if now.hour == schedule.daily_hour and now.minute == schedule.daily_minute:
                logger.info("[briefing_scheduler] Triggering daily briefing")
                try:
                    await generate_briefing(frequency="daily")
                except Exception as e:
                    logger.error("[briefing_scheduler] Daily briefing failed: %s", e)
                # Sleep past the trigger minute to avoid double-fire
                await asyncio.sleep(90)
                continue

            # Check if it's time for weekly briefing
            if (now.weekday() == schedule.weekly_day
                    and now.hour == schedule.weekly_hour
                    and now.minute == schedule.weekly_minute):
                logger.info("[briefing_scheduler] Triggering weekly briefing")
                try:
                    await generate_briefing(frequency="weekly")
                except Exception as e:
                    logger.error("[briefing_scheduler] Weekly briefing failed: %s", e)
                await asyncio.sleep(90)
                continue

            # Sleep for 30 seconds then check again
            await asyncio.sleep(30)

        except asyncio.CancelledError:
            logger.info("[briefing_scheduler] Scheduler cancelled")
            break
        except Exception as e:
            logger.error("[briefing_scheduler] Scheduler error: %s", e)
            await asyncio.sleep(60)


def start_scheduler():
    """Start the background briefing scheduler."""
    global _scheduler_task
    if _scheduler_task and not _scheduler_task.done():
        logger.warning("[briefing_scheduler] Scheduler already running")
        return

    try:
        loop = asyncio.get_event_loop()
        _scheduler_task = loop.create_task(_scheduler_loop())
        logger.info("[briefing_scheduler] Scheduler task created")
    except RuntimeError:
        logger.warning("[briefing_scheduler] No event loop — scheduler not started")


def start_scheduler_background(loop: Optional[asyncio.AbstractEventLoop] = None):
    """Start scheduler with an explicit event loop (called from main.py startup)."""
    global _scheduler_task
    if _scheduler_task and not _scheduler_task.done():
        logger.warning("[briefing_scheduler] Scheduler already running")
        return

    if loop is None:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            logger.warning("[briefing_scheduler] No event loop available")
            return

    _scheduler_task = loop.create_task(_scheduler_loop())
    logger.info("[briefing_scheduler] Background scheduler task created")


def stop_scheduler():
    """Stop the background briefing scheduler."""
    global _scheduler_task
    if _scheduler_task and not _scheduler_task.done():
        _scheduler_task.cancel()
        logger.info("[briefing_scheduler] Scheduler stopped")


__all__ = [
    "BriefingRecord",
    "generate_briefing",
    "get_recent_briefings",
    "get_latest_briefing",
    "start_scheduler",
    "start_scheduler_background",
    "stop_scheduler",
]
