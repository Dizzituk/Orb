# FILE: app/memory/session_lifecycle.py
"""
Scheduled job for conversation session lifecycle management.

Runs periodically to transition sessions through their lifecycle:
  active  → idle     (2hr inactivity)
  idle    → archived (48hr inactivity)

On archive, triggers Layer 3 knowledge extraction (Phase 6).

Designed to integrate with the existing scheduler pattern used by
app/astra_memory/decay_job.py — register via start_lifecycle_scheduler().

Part of CONV-MEMORY-001: Phase 5 (Decay and Cleanup).
"""

import logging
from datetime import datetime, timedelta
from typing import List

from sqlalchemy.orm import Session as DbSession

from app.memory.conversation_models import ConversationSession
from app.memory.conversation_service import (
    IDLE_THRESHOLD,
    ARCHIVE_THRESHOLD,
    update_session_status,
)

logger = logging.getLogger(__name__)


async def process_stale_sessions(db: DbSession) -> dict:
    """
    Scan all non-archived sessions and transition stale ones.

    Returns a summary dict: {"idled": N, "archived": N, "extracted": N}
    """
    now = datetime.utcnow()
    results = {"idled": 0, "archived": 0, "extracted": 0}

    # 1. Active → Idle (past IDLE_THRESHOLD)
    stale_active = (
        db.query(ConversationSession)
        .filter(
            ConversationSession.status == "active",
            ConversationSession.last_activity_at < now - IDLE_THRESHOLD,
        )
        .all()
    )

    for session in stale_active:
        updated = update_session_status(db, session.id, "idle")
        if updated:
            results["idled"] += 1
            logger.info(
                "[lifecycle] Session %d → idle (last activity: %s)",
                session.id, session.last_activity_at,
            )

    # 2. Idle → Archived (past ARCHIVE_THRESHOLD)
    stale_idle = (
        db.query(ConversationSession)
        .filter(
            ConversationSession.status == "idle",
            ConversationSession.last_activity_at < now - ARCHIVE_THRESHOLD,
        )
        .all()
    )

    for session in stale_idle:
        updated = update_session_status(db, session.id, "archived")
        if updated:
            results["archived"] += 1
            logger.info(
                "[lifecycle] Session %d → archived (last activity: %s)",
                session.id, session.last_activity_at,
            )

            # Trigger Layer 3 extraction on archive
            extracted = await _trigger_knowledge_extraction_async(db, session)
            if extracted:
                results["extracted"] += 1

    if any(v > 0 for v in results.values()):
        logger.info("[lifecycle] Cycle complete: %s", results)
    else:
        logger.debug("[lifecycle] No stale sessions found")

    return results


async def _trigger_knowledge_extraction_async(
    db: DbSession,
    session: ConversationSession,
) -> bool:
    """
    Trigger Layer 3 knowledge extraction for an archived session.

    Async version — called from the lifecycle scheduler loop.
    Returns True if extraction ran successfully, False otherwise.
    Non-fatal — failures are logged and swallowed.
    """
    try:
        from app.memory.knowledge_extractor import extract_and_promote_async
        return await extract_and_promote_async(db, session)
    except ImportError:
        logger.debug(
            "[lifecycle] knowledge_extractor not available yet — "
            "skipping extraction for session %d",
            session.id,
        )
        return False
    except Exception as e:
        logger.warning(
            "[lifecycle] Knowledge extraction failed for session %d: %s",
            session.id, e,
        )
        return False


# =========================================================================
# Scheduler integration
# =========================================================================

_scheduler_running = False


def start_lifecycle_scheduler(interval_minutes: int = 15) -> None:
    """
    Start the background lifecycle job.

    Uses asyncio to run periodically. Call once at app startup.
    Safe to call multiple times — only starts one instance.
    """
    global _scheduler_running
    if _scheduler_running:
        logger.debug("[lifecycle] Scheduler already running")
        return

    import asyncio

    async def _lifecycle_loop():
        global _scheduler_running
        _scheduler_running = True
        logger.info(
            "[lifecycle] Scheduler started (interval=%dm)",
            interval_minutes,
        )

        while True:
            await asyncio.sleep(interval_minutes * 60)
            try:
                from app.db import get_db_session
                db = get_db_session()
                try:
                    await process_stale_sessions(db)
                finally:
                    db.close()
            except Exception as e:
                logger.warning("[lifecycle] Scheduler cycle failed: %s", e)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_lifecycle_loop())
    except RuntimeError:
        logger.debug("[lifecycle] No event loop — scheduler not started")
