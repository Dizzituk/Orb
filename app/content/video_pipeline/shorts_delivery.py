# FILE: app/content/video_pipeline/shorts_delivery.py
# Purpose: Deliver a rendered short to chat, hold for review, and mark published (Job 8).
# Called-by: app.content.video_pipeline.shorts_orchestrator, app.tools.social_posting_tools
# Depends-on: app.content.models, app.memory.service, app.bridge.artifacts, .posting_drivers (autopublish)
# Last-renovated: 2026-07-02
"""
Shorts delivery (jobspec Job 8).

On render complete: copy the captioned master into the flat video
artifact dir, create a ContentOutput in the PENDING state (published_at
NULL), and deliver a tappable mp4 link into chat via the SAME artifact
marker mechanism image generation uses ([ASTRA_ARTIFACT:video:...]).
Nothing auto-posts — the human watches it and says "post that", which
resolves the most-recent pending short (get_latest_pending_short) and
runs the driver. ASTRA_SHORTS_AUTOPUBLISH=true skips the hold and posts
on completion.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

REEL_FORMAT = "instagram_reel"


def _resolve_delivery_project(db, project_id: Optional[int]):
    """Where the 'short ready' message lands: the given project, else the
    most-recently-active one (almost always the chat the user asked from),
    else a created 'Shorts' project."""
    from app.memory import service
    from app.memory.models import Project
    from app.memory.schemas import ProjectCreate

    if project_id:
        p = service.get_project(db, project_id)
        if p:
            return p
    latest = db.query(Project).order_by(Project.updated_at.desc()).first()
    if latest:
        return latest
    try:
        return service.create_project(db, ProjectCreate(
            name="Shorts", description="Rendered shorts awaiting review", type="content",
        ))
    except Exception as e:
        logger.warning("[shorts_delivery] could not resolve/create project: %s", e)
        return None


def _copy_to_delivery(asset_path: str, job) -> Optional[str]:
    """Copy the master into the flat video artifact dir; return the flat filename."""
    try:
        from app.bridge.artifacts import _get_video_dir
        dest_dir = _get_video_dir()
        filename = f"short_{job.job_id}_{job.slug}.mp4"
        shutil.copy2(asset_path, dest_dir / filename)
        return filename
    except Exception as e:
        logger.warning("[shorts_delivery] delivery copy failed: %s", e)
        return None


def _create_records(db, job, asset_path: str):
    """Minimal ContentPiece + PENDING ContentOutput. Returns the output."""
    from app.content.models import ContentPiece, ContentOutput

    piece = ContentPiece(
        title=(job.title or job.topic or "Short")[:200],
        description=(job.caption or "")[:2000],
        content_category="short",
        status="review",
        recommended_formats=[REEL_FORMAT],
        draft_text=job.script or "",
    )
    db.add(piece)
    db.commit()
    db.refresh(piece)

    output = ContentOutput(
        piece_id=piece.id,
        output_format=REEL_FORMAT,
        platform="instagram",
        primary_asset_path=asset_path,          # captioned master the driver uploads
        caption_text=job.caption_with_tags(),
        platform_metadata={
            "title": job.title,
            "caption": job.caption,
            "hashtags": job.hashtags,
            "script": job.script,
            "job_id": job.job_id,
            "delivered_filename": job.delivered_filename,
            "srt_path": job.srt_path,
        },
        publish_device="phone",
    )
    db.add(output)
    db.commit()
    db.refresh(output)
    return output


def _post_message(db, project, text: str) -> None:
    if not project:
        return
    from app.memory import service
    from app.memory.schemas import MessageCreate
    try:
        service.create_message(db, MessageCreate(
            project_id=project.id, role="assistant", content=text,
            provider="shorts", model="shorts-pipeline",
        ))
    except Exception as e:
        logger.warning("[shorts_delivery] could not post delivery message: %s", e)


async def deliver_short(
    job,
    *,
    project_id: Optional[int] = None,
    autopublish: bool = False,
    post_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Create the pending record, deliver the link (or autopublish)."""
    from app.db import SessionLocal

    asset = job.captioned_path or job.mp4_path
    if not asset or not Path(asset).exists():
        return {"ok": False, "error": f"no rendered file to deliver: {asset}"}

    job.delivered_filename = _copy_to_delivery(asset, job)

    db = SessionLocal()
    try:
        output = _create_records(db, job, asset)
        job.output_id = output.id
        job.save()

        if autopublish:
            post_fn = post_fn or _default_post
            result = await post_fn(asset, job.caption_with_tags())
            ok = bool(getattr(result, "ok", False))
            if ok:
                job.permalink = getattr(result, "permalink", None)
                mark_short_published(db, output.id, job.permalink)
                _post_message(db, _resolve_delivery_project(db, project_id),
                              f"Posted your short: {job.title} — {job.permalink or 'published'}")
            else:
                err = getattr(result, "error", "unknown error")
                _post_message(db, _resolve_delivery_project(db, project_id),
                              f"Auto-publish of '{job.title}' failed: {err}. It's saved and ready to retry.")
            job.save()
            return {"ok": ok, "output_id": output.id, "autopublished": True, "permalink": job.permalink}

        # Review hold: deliver the tappable mp4 link. Nothing auto-posts.
        project = _resolve_delivery_project(db, project_id)
        if job.delivered_filename:
            msg = (f"Short ready: {job.title} — have a watch, then say "
                   f"“post that to Instagram”.\n[ASTRA_ARTIFACT:video:{job.delivered_filename}]")
        else:
            msg = f"Short ready: {job.title} (file at {asset}). Say “post that to Instagram” to publish."
        _post_message(db, project, msg)
        return {"ok": True, "output_id": output.id, "autopublished": False,
                "delivered_filename": job.delivered_filename}
    finally:
        db.close()


async def _default_post(file_path: str, caption: str):
    from app.content.distribution.posting_drivers import meta_driver
    return await meta_driver.post_reel(file_path, caption)


def get_latest_pending_short(db):
    """Most recent reel ContentOutput not yet published (for 'post that')."""
    from app.content.models import ContentOutput
    return (
        db.query(ContentOutput)
        .filter(ContentOutput.output_format == REEL_FORMAT)
        .filter(ContentOutput.published_at.is_(None))
        .order_by(ContentOutput.created_at.desc())
        .first()
    )


def mark_short_published(db, output_id: str, permalink: Optional[str]) -> bool:
    """Flip a ContentOutput to published with its permalink/post id."""
    from app.content.models import ContentOutput
    out = db.query(ContentOutput).filter(ContentOutput.id == output_id).one_or_none()
    if not out:
        return False
    out.published_at = datetime.now(timezone.utc)
    if permalink:
        out.platform_post_id = permalink
    db.commit()
    return True
