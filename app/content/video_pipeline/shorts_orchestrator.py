# FILE: app/content/video_pipeline/shorts_orchestrator.py
# Purpose: Slim shorts pipeline — script -> HeyGen 9:16 -> word-synced captions -> deliver.
# Called-by: app.tools.social_posting_tools (create_short), tests.test_shorts_orchestrator
# Depends-on: .shorts_job, .shorts_script, .heygen_client, .caption_align, .shorts_delivery (injectable)
# Last-renovated: 2026-07-02
"""
Shorts orchestrator (jobspec Job 6).

Four stages, nothing more: script -> render -> captions -> deliver. The
amputation is the point — NO style cascade, NO asset resolution, NO clip
verification, NO director QA cycles. The longform orchestrator.py is
untouched and keeps all of that.

Every external effect (LLM, HeyGen, whisper, delivery, posting) is
injectable so the whole four-stage flow is unit-testable with fakes and
no network. create_short_job() returns immediately with a job id; the
tool layer runs run_short_job() in the background and delivery fires on
completion.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Optional

from app.content.video_pipeline.shorts_job import ShortsJob

logger = logging.getLogger(__name__)


def autopublish_enabled() -> bool:
    return os.getenv("ASTRA_SHORTS_AUTOPUBLISH", "false").strip().lower() in ("1", "true", "yes", "on")


def create_short_job(topic: str, notes: str = "") -> ShortsJob:
    """Make + persist a pending job. Returns instantly (tool returns the id)."""
    job = ShortsJob(topic=topic.strip(), notes=(notes or "").strip())
    job.record_event("created", "pending", f"short queued: {topic[:60]}")
    job.save()
    return job


async def run_short_job(
    job: ShortsJob,
    *,
    project_id: Optional[int] = None,
    autopublish: Optional[bool] = None,
    llm: Optional[Callable] = None,
    heygen_fn: Optional[Callable] = None,
    caption_fn: Optional[Callable] = None,
    deliver_fn: Optional[Callable] = None,
    post_fn: Optional[Callable] = None,
    event_cb: Optional[Callable] = None,
) -> ShortsJob:
    """Execute the four stages. Sets job.status and returns the job."""
    if autopublish is None:
        autopublish = autopublish_enabled()

    async def emit(stage: str, status: str, message: str = "", **data) -> None:
        ev = job.record_event(stage, status, message, **data)
        job.save()
        logger.info("[shorts:%s] %s: %s — %s", job.job_id, stage, status, message)
        if event_cb:
            try:
                await event_cb(ev)
            except Exception:
                pass

    job.status = "running"
    try:
        # 1) SCRIPT ─ hook-first, <=45s enforced in shorts_script.
        from app.content.video_pipeline import shorts_script
        await emit("script", "start", f"writing script: {job.topic[:60]}")
        parsed = await shorts_script.generate_script(job.topic, job.notes, llm=llm)
        job.script = parsed["script"]
        job.title = parsed["title"]
        job.caption = parsed["caption"]
        job.hashtags = parsed["hashtags"]
        await emit("script", "complete", f"{parsed['word_count']} words", title=job.title)

        # 2) RENDER ─ HeyGen 9:16 full-frame avatar, voice embedded, cache reused.
        heygen_fn = heygen_fn or _default_heygen
        await emit("render", "start", "rendering avatar (9:16)")
        rendered = await heygen_fn(text=job.script, segment_id=job.slug, aspect_ratio="9:16")
        job.mp4_path = rendered.get("file_path")
        job.duration_s = float(rendered.get("duration_s") or 0.0)
        job.cost_usd = float(rendered.get("cost_usd") or 0.0)
        if not job.mp4_path:
            raise RuntimeError("HeyGen returned no file_path")
        await emit("render", "complete", f"{job.duration_s:.1f}s", mp4=job.mp4_path)

        # 3) CAPTIONS ─ whisper word-timestamps -> styled ASS burn + SRT.
        caption_fn = caption_fn or _default_caption
        await emit("captions", "start", "aligning word-synced captions")
        cap = await _maybe_thread(
            caption_fn, job.mp4_path, str(job.out_dir),
            slug=job.slug, script_text=job.script, duration_s=job.duration_s,
        )
        job.captioned_path = cap.get("burned_path") or job.mp4_path
        job.srt_path = cap.get("srt_path") or None
        await emit("captions", "complete", f"{cap.get('caption_count', 0)} captions")

        # 4) DELIVER ─ pending ContentOutput + mp4 link to chat (or autopublish).
        deliver_fn = deliver_fn or _default_deliver
        await emit("deliver", "start", "delivering to chat")
        await deliver_fn(job, project_id=project_id, autopublish=autopublish, post_fn=post_fn)
        job.status = "complete"
        await emit("deliver", "complete", "short ready", output_id=job.output_id)
    except Exception as e:
        job.status = "error"
        job.error = str(e)
        logger.exception("[shorts:%s] failed at stage %s", job.job_id, job.stage)
        await emit(job.stage or "error", "error", str(e))
    job.save()
    return job


# ── default (real) stage implementations, lazily imported ────────────

async def _default_heygen(*, text: str, segment_id: str, aspect_ratio: str) -> dict:
    from app.content.video_pipeline import heygen_client
    return await heygen_client.generate_and_download(
        text=text, segment_id=segment_id, aspect_ratio=aspect_ratio
    )


def _default_caption(video_path, out_dir, *, slug, script_text, duration_s):
    from app.content.video_pipeline import caption_align
    return caption_align.align_and_burn(
        video_path, out_dir, slug=slug, script_text=script_text, duration_s=duration_s
    )


async def _default_deliver(job, *, project_id, autopublish, post_fn):
    from app.content.video_pipeline import shorts_delivery
    return await shorts_delivery.deliver_short(
        job, project_id=project_id, autopublish=autopublish, post_fn=post_fn
    )


async def _maybe_thread(fn, *args, **kwargs):
    """Run a sync stage fn in a thread; await it if it's a coroutine fn."""
    if asyncio.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))
