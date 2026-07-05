# FILE: app/idle/tasks_visual.py
# Purpose: Idle task draining the visual-embed ingest queue (multimodal collection).
# Called-by: app.idle.router.ensure_builtin_tasks_registered (import side-effect)
# Depends-on: app.embeddings.visual_queue, app.embeddings.provider_router, app.gpu.orchestrator (lazy)
# Last-renovated: 2026-07-02 (LANE E)
"""
visual_embed_drain (LANE E, Task 2).

Visual items (video assets today; drive images / screenshots when those
producers land) queue in visual_embed_queue whenever the multimodal role
can't embed synchronously. This task drains them during idle:

  - provider=gemini: drain in any idle window (plain API calls).
  - provider=local: the model is LOAD-ON-DEMAND ONLY — ask the GPU
    orchestrator for BACKGROUND_INGEST (Chatterbox unloads, worker loads),
    wait briefly for residency, then drain. If residency doesn't arrive in
    time, pause and retry next window. On completion the orchestrator is
    told the ingest work is done so INTERACTIVE can be restored.

Checkpointing: the queue itself is the checkpoint (rows flip to done/failed
one batch at a time) — a crash mid-drain resumes exactly where it stopped.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from app.idle.router import RecurringSpec, TaskContext, TaskOutcome, register_task_handler

logger = logging.getLogger(__name__)

VISUAL_TASK = "visual_embed_drain"

_RESIDENCY_WAIT_SECONDS = 120.0
_RESIDENCY_POLL_SECONDS = 5.0


def _orchestrator():
    """Lazy, optional — the drain must work (gemini path) even if the GPU
    module is unavailable."""
    try:
        from app.gpu import orchestrator
        return orchestrator
    except Exception:
        return None


async def visual_handler(ctx: TaskContext) -> TaskOutcome:
    from app.embeddings import provider_router, visual_queue

    db = ctx.session_factory()
    try:
        pending = visual_queue.pending_count(db)
    finally:
        db.close()
    if pending == 0:
        return TaskOutcome.completed(summary="visual queue empty", coverage="0 items")

    orch = _orchestrator()
    local_mode = provider_router.multimodal_write_spec().provider == "local"

    # Local model residency: request BACKGROUND_INGEST and wait briefly.
    if local_mode and not provider_router.multimodal_ready():
        if orch is not None:
            try:
                orch.request_background_ingest(reason=f"{pending} visual item(s) queued")
            except Exception as exc:
                logger.warning("[visual_drain] ingest request failed: %s", exc)
        waited = 0.0
        while waited < _RESIDENCY_WAIT_SECONDS:
            if ctx.should_yield():
                return TaskOutcome.paused("user activity while waiting for model residency")
            if provider_router.multimodal_ready():
                break
            await asyncio.sleep(_RESIDENCY_POLL_SECONDS)
            waited += _RESIDENCY_POLL_SECONDS
        if not provider_router.multimodal_ready():
            return TaskOutcome.paused(
                f"multimodal model not resident after {int(waited)}s — retry next window"
            )

    embedded = 0
    failed = 0
    try:
        while True:
            if ctx.should_yield():
                ctx.save_progress({"embedded": embedded, "failed": failed})
                return TaskOutcome.paused(f"checkpointed after {embedded} item(s)")
            db = ctx.session_factory()
            try:
                stats = await asyncio.to_thread(visual_queue.drain_pending, db, 25)
            finally:
                db.close()
            if stats.get("skipped_unavailable"):
                return TaskOutcome.paused("multimodal became unavailable mid-drain")
            embedded += stats.get("embedded", 0)
            failed += stats.get("failed", 0)
            ctx.save_progress({"embedded": embedded, "failed": failed})
            db = ctx.session_factory()
            try:
                if visual_queue.pending_count(db) == 0:
                    break
            finally:
                db.close()
    finally:
        if local_mode and orch is not None:
            try:
                orch.notify_ingest_complete()
            except Exception:
                pass

    return TaskOutcome.completed(
        summary=f"visual drain: {embedded} embedded, {failed} failed",
        coverage=f"{embedded + failed} queued item(s)",
    )


def visual_fingerprint(params: dict) -> Optional[str]:
    """Skip when the queue is unchanged (usually: empty)."""
    try:
        from app.db import SessionLocal
        from app.embeddings import provider_router, visual_queue

        db = SessionLocal()
        try:
            pending = visual_queue.pending_count(db)
        finally:
            db.close()
        return f"pending:{pending}|{provider_router.multimodal_write_spec().label}"
    except Exception:
        return None


def _cadence_hours() -> float:
    try:
        return float(os.getenv("ASTRA_VISUAL_DRAIN_CADENCE_HOURS", "1.0"))
    except Exception:
        return 1.0


register_task_handler(
    VISUAL_TASK,
    visual_handler,
    fingerprint_fn=visual_fingerprint,
    recurring=RecurringSpec(task_type=VISUAL_TASK, params={}, cadence_hours=_cadence_hours()),
)
