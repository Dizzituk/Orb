# FILE: app/tools/social_posting_tools.py
# Purpose: Chat tools for posting media + kicking off shorts (Jobs 4 + 8 tool surface).
# Called-by: app.tools.registry (_register_defaults)
# Depends-on: app.content.distribution.posting_drivers.meta_driver, app.content.video_pipeline.{shorts_orchestrator,shorts_delivery}
# Last-renovated: 2026-07-02
"""
Social posting + shorts chat tools.

Designed for a small local model: ONE tool call does the whole job, and
omitted arguments resolve to sensible defaults, so "post that to
Instagram / Facebook / social" after a short is delivered just works —
no ids to juggle.

  post_image_to_instagram  — post a still (defaults to newest AstraPictures image)
  post_reel_to_instagram   — post the most recent pending short (or a given mp4)
  create_short             — render a captioned 9:16 short from a topic (async)

The Business Suite composer posts to Facebook AND Instagram from one
login (IG->FB auto-share), so a single call covers both platforms.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")
_BG_TASKS: set = set()  # hold refs so background renders aren't GC'd


# ── schemas ──────────────────────────────────────────────────────────

TOOL_SCHEMAS = {
    "post_image_to_instagram": {
        "input": {
            "type": "object",
            "required": ["caption"],
            "properties": {
                "caption": {"type": "string", "description": "Caption text (with any hashtags)."},
                "image_path": {
                    "type": "string",
                    "description": "Absolute path to the image. Omit to use the most recent generated image.",
                },
            },
        },
        "output": {
            "type": "object",
            "required": ["ok"],
            "properties": {
                "ok": {"type": "boolean"},
                "platform": {"type": "string"},
                "permalink": {"type": "string"},
                "audit_dir": {"type": "string"},
                "failed_step": {"type": "string"},
                "error": {"type": "string"},
            },
        },
    },
    "post_reel_to_instagram": {
        "input": {
            "type": "object",
            "properties": {
                "video_path": {
                    "type": "string",
                    "description": "Absolute path to an mp4. Omit to post the most recent short awaiting review.",
                },
                "caption": {
                    "type": "string",
                    "description": "Caption. Omit to use the caption stored with the pending short.",
                },
            },
        },
        "output": {
            "type": "object",
            "required": ["ok"],
            "properties": {
                "ok": {"type": "boolean"},
                "platform": {"type": "string"},
                "permalink": {"type": "string"},
                "audit_dir": {"type": "string"},
                "failed_step": {"type": "string"},
                "error": {"type": "string"},
            },
        },
    },
    "create_short": {
        "input": {
            "type": "object",
            "required": ["topic"],
            "properties": {
                "topic": {"type": "string", "description": "What the short should be about."},
                "notes": {"type": "string", "description": "Optional angle/tone/constraints."},
            },
        },
        "output": {
            "type": "object",
            "required": ["ok"],
            "properties": {
                "ok": {"type": "boolean"},
                "job_id": {"type": "string"},
                "status": {"type": "string"},
                "message": {"type": "string"},
                "error": {"type": "string"},
            },
        },
    },
}


# ── helpers ──────────────────────────────────────────────────────────

def _newest_image() -> Optional[str]:
    from app.llm.image_output_dir import get_image_output_dir
    d = get_image_output_dir()
    try:
        files = [p for p in Path(d).iterdir() if p.suffix.lower() in _IMG_EXTS and p.is_file()]
    except (OSError, FileNotFoundError):
        return None
    if not files:
        return None
    return str(max(files, key=lambda p: p.stat().st_mtime))


# ── handlers ─────────────────────────────────────────────────────────

async def post_image_handler(input_data: dict, context: Optional[dict]) -> dict:
    from app.content.distribution.posting_drivers import meta_driver

    caption = str(input_data.get("caption") or "")
    image_path = input_data.get("image_path") or _newest_image()
    if not image_path:
        return {"ok": False, "error": "no image_path given and no images found in the output dir"}
    result = await meta_driver.post_image(image_path, caption)
    return result.to_dict()


async def post_reel_handler(input_data: dict, context: Optional[dict]) -> dict:
    from app.content.distribution.posting_drivers import meta_driver
    from app.content.video_pipeline.shorts_delivery import get_latest_pending_short, mark_short_published
    from app.db import SessionLocal

    video_path = input_data.get("video_path")
    caption = input_data.get("caption")
    output_id = None

    if not video_path:
        db = SessionLocal()
        try:
            pending = get_latest_pending_short(db)
            if not pending:
                return {"ok": False, "error": "no pending short to post — make one first with create_short"}
            video_path = pending.primary_asset_path
            output_id = pending.id
            if caption is None:
                caption = pending.caption_text or ""
        finally:
            db.close()

    if not video_path or not Path(video_path).exists():
        return {"ok": False, "error": f"reel file not found: {video_path}"}

    result = await meta_driver.post_reel(video_path, caption or "")

    # If this was a tracked pending short and it posted, flip it to published.
    if result.ok and output_id:
        db = SessionLocal()
        try:
            mark_short_published(db, output_id, result.permalink)
        finally:
            db.close()
    return result.to_dict()


async def create_short_handler(input_data: dict, context: Optional[dict]) -> dict:
    from app.content.video_pipeline import shorts_orchestrator

    topic = str(input_data.get("topic") or "").strip()
    if not topic:
        return {"ok": False, "error": "create_short requires a topic"}
    notes = str(input_data.get("notes") or "")

    job = shorts_orchestrator.create_short_job(topic, notes)
    project_id = (context or {}).get("project_id")
    task = asyncio.create_task(shorts_orchestrator.run_short_job(job, project_id=project_id))
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)

    return {
        "ok": True,
        "job_id": job.job_id,
        "status": "rendering",
        "message": (
            f"On it — rendering a short about “{topic[:60]}”. "
            "I'll drop the finished clip in chat when it's ready; then say "
            "“post that to Instagram” to publish."
        ),
    }


HANDLERS = {
    "post_image_to_instagram": post_image_handler,
    "post_reel_to_instagram": post_reel_handler,
    "create_short": create_short_handler,
}

TOOL_DESCRIPTIONS = {
    "post_image_to_instagram": (
        "Post an image to Instagram AND Facebook (one post via the Meta Business Suite "
        "composer — IG auto-shares to the FB page). Pass `caption`; omit `image_path` to "
        "use the most recently generated image. Use when the user says 'post this image / "
        "post that to Instagram / Facebook / social'. Returns a permalink or a clear failure "
        "with the audit dir."
    ),
    "post_reel_to_instagram": (
        "Post a reel/short to Instagram AND Facebook (one post via the Meta Business Suite "
        "composer). Omit BOTH args to publish the most recent short awaiting review with its "
        "stored caption — this is the 'post that to Instagram / social' path after a short is "
        "delivered. Pass `video_path` (any local mp4) and/or `caption` to override. Returns a "
        "permalink or a clear failure with the audit dir."
    ),
    "create_short": (
        "Make a captioned vertical (9:16) short about a topic: writes a hook-first <=45s "
        "script, renders a talking avatar with word-synced burned-in captions, and delivers "
        "the finished mp4 into chat for review. Returns immediately with a job id; the clip "
        "arrives when rendering finishes. Use for 'make/create a short/reel about X'. It does "
        "NOT auto-post — the user reviews it, then says 'post that to Instagram'."
    ),
}


def register_social_posting_tools() -> None:
    """Register posting + shorts tools with the global registry (chat + phone)."""
    from app.tools.registry import ToolDefinition, register_tool

    for name, handler in HANDLERS.items():
        register_tool(ToolDefinition(
            name=name,
            version="v1",
            description=TOOL_DESCRIPTIONS[name],
            input_schema=TOOL_SCHEMAS[name]["input"],
            output_schema=TOOL_SCHEMAS[name]["output"],
            handler=handler,
        ))
    logger.info("[social_posting_tools] registered %d tools", len(HANDLERS))
