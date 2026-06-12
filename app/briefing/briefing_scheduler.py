# Purpose: briefing scheduler
# Called-by: app.briefing.briefing_router, main
# Depends-on: app.briefing.briefing_audio, app.briefing.briefing_collector, app.briefing.briefing_compiler, app.briefing.briefing_config
# Last-renovated: 2026-06-11
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BRIEFING_STORE_DIR = os.getenv(
    "BRIEFING_STORE_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "briefings"),
)


@dataclass
class BriefingRecord:
    id: str
    frequency: str
    generated_at: str
    title: str
    total_items: int
    text_digest_path: str
    audio_path: str = ""
    astra_alerts: list[str] = field(default_factory=list)
    profile: str = "default"
    structure_path: str = ""


def _ensure_store_dir() -> Path:
    path = Path(BRIEFING_STORE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _index_path() -> Path:
    return _ensure_store_dir() / "briefing_index.json"


def _load_records() -> list[dict]:
    index_path = _index_path()
    if not index_path.exists():
        return []
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_records(records: list[dict]) -> None:
    _index_path().write_text(json.dumps(records[-30:], indent=2), encoding="utf-8")


def _save_record(record: BriefingRecord) -> None:
    records = _load_records()
    records.append(asdict(record))
    _save_records(records)


def get_recent_briefings(count: int = 10, profile: Optional[str] = None) -> list[dict]:
    records = _load_records()
    if profile:
        records = [record for record in records if record.get("profile", "default") == profile]
    return records[-count:]


def get_latest_briefing(profile: Optional[str] = None) -> Optional[dict]:
    recent = get_recent_briefings(1, profile=profile)
    return recent[0] if recent else None


def load_briefing_structure(record: dict) -> dict:
    structure_path = record.get("structure_path", "")
    if structure_path and Path(structure_path).exists():
        try:
            return json.loads(Path(structure_path).read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[briefing_scheduler] Failed to read structure %s: %s", structure_path, exc)
    return {
        "sections": [],
        "astra_alerts": record.get("astra_alerts", []),
        "profile": record.get("profile", "default"),
    }


async def generate_briefing(
    frequency: str = "daily",
    context: Optional[dict] = None,
    profile: str = "default",
) -> BriefingRecord:
    from app.briefing.briefing_collector import collect_all_topics
    from app.briefing.briefing_compiler import compile_briefing
    from app.briefing.briefing_config import get_schedule

    logger.info("[briefing_scheduler] Starting %s briefing generation for profile=%s", frequency, profile)

    collection = await collect_all_topics(context=context, profile=profile)
    briefing = compile_briefing(collection, frequency=frequency, profile=profile)

    store = _ensure_store_dir()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    briefing_id = f"{profile}_{frequency}_{timestamp}"

    digest_path = store / f"{briefing_id}.md"
    digest_path.write_text(briefing.text_digest, encoding="utf-8")

    structure_path = store / f"{briefing_id}.json"
    structure_payload = {
        "id": briefing_id,
        "profile": profile,
        "title": briefing.title,
        "generated_at": briefing.generated_at,
        "total_items": briefing.total_items,
        "astra_alerts": briefing.astra_alerts,
        "sections": [
            {
                "topic_name": section.topic_name,
                "topic_key": section.topic_key,
                "description": section.description,
                "items": [
                    {
                        "headline": item.headline,
                        "summary": item.summary,
                        "source_name": item.source_name,
                        "source_url": item.source_url,
                        "credibility": item.credibility,
                        "topic_key": item.topic_key,
                        "astra_flag": item.astra_flag,
                    }
                    for item in section.items
                ],
            }
            for section in briefing.sections
        ],
    }
    structure_path.write_text(json.dumps(structure_payload, indent=2), encoding="utf-8")

    audio_path = ""
    schedule = get_schedule()
    if schedule.audio_enabled:
        try:
            from app.briefing.briefing_audio import generate_briefing_audio
            result = await generate_briefing_audio(briefing)
            if result:
                audio_path = result
        except Exception as exc:
            logger.warning("[briefing_scheduler] Audio generation failed: %s", exc)

    record = BriefingRecord(
        id=briefing_id,
        frequency=frequency,
        generated_at=briefing.generated_at,
        title=briefing.title,
        total_items=briefing.total_items,
        text_digest_path=str(digest_path),
        audio_path=audio_path,
        astra_alerts=briefing.astra_alerts,
        profile=profile,
        structure_path=str(structure_path),
    )
    _save_record(record)
    return record


_scheduler_task: Optional[asyncio.Task] = None


async def _scheduler_loop():
    from app.briefing.briefing_config import get_schedule

    logger.info("[briefing_scheduler] Background scheduler started")
    while True:
        try:
            schedule = get_schedule()
            if not schedule.auto_generate:
                await asyncio.sleep(60)
                continue

            now = datetime.now(timezone.utc)
            if now.hour == schedule.daily_hour and now.minute == schedule.daily_minute:
                try:
                    await generate_briefing(frequency="daily")
                except Exception as exc:
                    logger.error("[briefing_scheduler] Daily briefing failed: %s", exc)
                await asyncio.sleep(90)
                continue

            if now.weekday() == schedule.weekly_day and now.hour == schedule.weekly_hour and now.minute == schedule.weekly_minute:
                try:
                    await generate_briefing(frequency="weekly")
                except Exception as exc:
                    logger.error("[briefing_scheduler] Weekly briefing failed: %s", exc)
                await asyncio.sleep(90)
                continue

            await asyncio.sleep(30)
        except asyncio.CancelledError:
            logger.info("[briefing_scheduler] Scheduler cancelled")
            break
        except Exception as exc:
            logger.error("[briefing_scheduler] Scheduler error: %s", exc)
            await asyncio.sleep(60)


def start_scheduler():
    global _scheduler_task
    if _scheduler_task and not _scheduler_task.done():
        logger.warning("[briefing_scheduler] Scheduler already running")
        return
    try:
        loop = asyncio.get_event_loop()
        _scheduler_task = loop.create_task(_scheduler_loop())
    except RuntimeError:
        logger.warning("[briefing_scheduler] No event loop — scheduler not started")


def start_scheduler_background(loop: Optional[asyncio.AbstractEventLoop] = None):
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


def stop_scheduler():
    global _scheduler_task
    if _scheduler_task and not _scheduler_task.done():
        _scheduler_task.cancel()


__all__ = [
    "BriefingRecord",
    "generate_briefing",
    "get_recent_briefings",
    "get_latest_briefing",
    "load_briefing_structure",
    "start_scheduler",
    "start_scheduler_background",
    "stop_scheduler",
]
