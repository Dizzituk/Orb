# FILE: app/content/production_router.py
# Purpose: Production Engine API endpoints.
# Called-by: main
# Depends-on: app.auth, app.auth.config, app.content.models, app.content.production.cutaway_gen (+9 more)
# Last-renovated: 2026-06-11
"""
Production Engine API endpoints.

Exposes cutaway generation, draft writing, carousel production,
multi-format output, and video assembly operations.
"""
import logging
import os
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/content/production",
    tags=["Content Production"],
    dependencies=[Depends(require_auth)],
)


# ─── REQUEST SCHEMAS ───

class CutawayApprovalRequest(BaseModel):
    cutaway_index: int
    decision: str  # approved | rejected | modified
    modifications: Optional[dict] = None


class DraftRequest(BaseModel):
    format_type: str = "blog_post"
    duration_minutes: int = 5


class DraftRefineRequest(BaseModel):
    feedback: str


class CarouselSlide(BaseModel):
    title: str = ""
    body: str = ""


class CarouselRequest(BaseModel):
    slides: List[CarouselSlide]


# ═══════════════════════════════════════════════════
# CUTAWAY GENERATION
# ═══════════════════════════════════════════════════

@router.post("/pieces/{piece_id}/cutaways/generate", response_model=dict)
async def generate_cutaways(
    piece_id: str,
    db: Session = Depends(get_db),
):
    """Generate cutaway concepts for a content piece (AI)."""
    from app.content.production.cutaway_gen import generate_cutaway_concepts

    try:
        concepts = await generate_cutaway_concepts(db, piece_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "piece_id": piece_id,
        "concepts_generated": len(concepts),
        "concepts": concepts,
    }


@router.post("/pieces/{piece_id}/cutaways/approve", response_model=dict)
def approve_cutaway_endpoint(
    piece_id: str,
    body: CutawayApprovalRequest,
    db: Session = Depends(get_db),
):
    """Approve, reject, or modify a cutaway concept."""
    from app.content.production.cutaway_gen import approve_cutaway

    try:
        result = approve_cutaway(
            db, piece_id, body.cutaway_index,
            decision=body.decision,
            modifications=body.modifications,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "piece_id": piece_id,
        "cutaway_index": body.cutaway_index,
        "decision": body.decision,
        "concept": result,
    }


@router.get("/cutaways/autonomous-status", response_model=dict)
def cutaway_autonomous_status(db: Session = Depends(get_db)):
    """Check if cutaway generation can go autonomous."""
    from app.content.production.cutaway_gen import (
        check_autonomous_cutaway_eligibility,
    )
    return check_autonomous_cutaway_eligibility(db)


# ═══════════════════════════════════════════════════
# DRAFT WRITING
# ═══════════════════════════════════════════════════

@router.post("/pieces/{piece_id}/draft", response_model=dict)
async def generate_draft_endpoint(
    piece_id: str,
    body: DraftRequest,
    db: Session = Depends(get_db),
):
    """Generate a text draft for a content piece (AI)."""
    from app.content.production.draft_writer import generate_draft

    try:
        draft = await generate_draft(
            db, piece_id, body.format_type, body.duration_minutes
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not draft:
        raise HTTPException(status_code=500, detail="Draft generation failed")

    return {
        "piece_id": piece_id,
        "format_type": body.format_type,
        "draft_length": len(draft),
        "draft": draft,
    }


@router.post("/pieces/{piece_id}/draft/refine", response_model=dict)
async def refine_draft_endpoint(
    piece_id: str,
    body: DraftRefineRequest,
    db: Session = Depends(get_db),
):
    """Refine an existing draft with feedback (AI)."""
    from app.content.production.draft_writer import refine_draft

    try:
        refined = await refine_draft(db, piece_id, body.feedback)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not refined:
        raise HTTPException(status_code=500, detail="Refinement failed")

    return {
        "piece_id": piece_id,
        "draft_length": len(refined),
        "draft": refined,
    }


# ═══════════════════════════════════════════════════
# STATIC CONTENT
# ═══════════════════════════════════════════════════

@router.post("/pieces/{piece_id}/carousel", response_model=dict)
def generate_carousel_endpoint(
    piece_id: str,
    body: Optional[CarouselRequest] = None,
    db: Session = Depends(get_db),
):
    """Generate Instagram carousel slides."""
    from app.content.production.static_gen import generate_carousel
    from app.content.models import ContentPiece

    piece = db.query(ContentPiece).get(piece_id)
    if not piece:
        raise HTTPException(status_code=404, detail="Piece not found")

    if body and body.slides:
        slides = [{"title": s.title, "body": s.body} for s in body.slides]
    else:
        # Auto-generate from piece data
        excerpts = piece.key_excerpts or []
        slides = [{"title": piece.title, "body": piece.description or ""}]
        for exc in excerpts[:6]:
            slides.append({"title": "", "body": exc})
        slides.append({
            "title": "What do you think?",
            "body": "Follow for more. Drop your thoughts below.",
        })

    paths = generate_carousel(piece_id, slides)

    return {
        "piece_id": piece_id,
        "slides_generated": len(paths),
        "paths": paths,
    }


# ═══════════════════════════════════════════════════
# MULTI-FORMAT PRODUCTION
# ═══════════════════════════════════════════════════

@router.post("/pieces/{piece_id}/produce-all", response_model=dict)
async def produce_all_formats_endpoint(
    piece_id: str,
    db: Session = Depends(get_db),
):
    """
    Produce all recommended output formats for a content piece.
    This is the main production trigger.
    """
    from app.content.production.format_converter import produce_all_formats

    try:
        outputs = await produce_all_formats(db, piece_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "piece_id": piece_id,
        "outputs_produced": len(outputs),
        "outputs": [
            {
                "id": o.id,
                "format": o.output_format,
                "platform": o.platform,
                "device": o.publish_device,
                "asset_path": o.primary_asset_path,
                "status": o.platform_metadata.get("status", "ready"),
            }
            for o in outputs
        ],
    }


# ═══════════════════════════════════════════════════
# SYSTEM STATUS
# ═══════════════════════════════════════════════════

@router.get("/status", response_model=dict)
def production_status():
    """Check production system capabilities."""
    from app.content.production.edit_engine import check_ffmpeg

    return {
        "ffmpeg_available": check_ffmpeg(),
        "pillow_available": True,  # Checked at import time
        "video_production": "ready" if check_ffmpeg() else "pending_ffmpeg",
        "carousel_production": "ready",
        "blog_production": "ready",
        "draft_writing": "ready",
        "cutaway_generation": "ready",
    }


# ═══════════════════════════════════════════════════
# SHORTS CREATION
# ═══════════════════════════════════════════════════

class CreateShortsRequest(BaseModel):
    """Request to create shorts from a source video."""
    source_item_id: str  # ContentItem ID from the upload
    prompt: str  # e.g. "3 shorts, key insights, 30-60 seconds"
    max_shorts: int = 3
    max_duration: int = 60  # seconds per short


@router.post("/shorts", response_model=dict)
async def create_shorts(
    body: CreateShortsRequest,
    db: Session = Depends(get_db),
):
    """Create short-form video clips from a source video.

    Uses Gemini to analyse the video and identify compelling
    moments, then FFmpeg to cut them. Results appear as
    ContentOutput records for review and YouTube queueing.
    """
    from app.content.project_models import ProjectContentItem
    from app.content.models import ContentPiece, ContentOutput
    from app.content.production.shorts_creator import (
        analyse_video_for_shorts,
        cut_shorts,
    )

    # Find the source item
    item = db.query(ProjectContentItem).get(body.source_item_id)
    if not item:
        raise HTTPException(404, "Source item not found")

    if not item.source_path or not os.path.exists(item.source_path):
        raise HTTPException(400, f"Source file not found: {item.source_path}")

    # Step 1: AI analysis
    segments = await analyse_video_for_shorts(
        video_path=item.source_path,
        prompt=body.prompt,
        max_shorts=body.max_shorts,
        max_duration=body.max_duration,
    )

    if not segments:
        return {
            "status": "no_segments",
            "message": "Could not identify suitable segments. Try a different prompt.",
        }

    # Step 2: Create a content piece for this batch
    source_title = item.title or "Untitled"
    piece = ContentPiece(
        title=f"Shorts from: {source_title}",
        description=f"AI-generated shorts. Prompt: {body.prompt}",
        content_category="educational",
        status="in_production",
        source_conversation_ids=[],
        source_tag_ids=[],
    )
    db.add(piece)
    db.flush()

    # Step 3: Cut the shorts with FFmpeg
    cut_results = cut_shorts(
        source_path=item.source_path,
        segments=segments,
        piece_id=piece.id,
        output_format="youtube_short",
    )

    if not cut_results:
        piece.status = "rejected"
        db.commit()
        return {
            "status": "cut_failed",
            "message": "FFmpeg failed to cut the identified segments.",
            "segments_found": len(segments),
        }

    # Step 4: Create ContentOutput for each short
    outputs = []
    for result in cut_results:
        output = ContentOutput(
            piece_id=piece.id,
            output_format="youtube_short",
            platform="youtube",
            primary_asset_path=result["path"],
            thumbnail_path=result.get("thumbnail"),
            caption_text=result["description"],
            platform_metadata={
                "title": result["title"],
                "description": result["description"],
                "hook": result["hook"],
                "source_item_id": body.source_item_id,
                "source_timestamps": {
                    "start": result["start"],
                    "end": result["end"],
                },
                "tags": [],
                "status": "draft",
            },
            publish_device="desktop",
        )
        db.add(output)
        outputs.append({
            "title": result["title"],
            "path": result["path"],
            "duration": result["duration"],
            "description": result["description"],
            "hook": result["hook"],
        })

    # Step 4b: Generate thumbnails for each short
    from app.content.production.thumbnail_gen import generate_thumbnail

    for i, result in enumerate(cut_results):
        thumb_path = result["path"].replace(".mp4", "_thumb.png")
        thumb = generate_thumbnail(
            video_path=result["path"],
            title=result["title"],
            output_path=thumb_path,
        )
        if thumb:
            result["thumbnail"] = thumb
            logger.info("[shorts] Thumbnail generated for short %d", i + 1)

    # Step 5: Quality check each short with Gemini 3.1 Pro
    from app.content.production.shorts_quality_check import (
        check_all_shorts,
    )
    from dataclasses import asdict

    quality_results = await check_all_shorts(
        shorts=cut_results,
        original_prompt=body.prompt,
    )

    # Apply quality verdicts to outputs
    approved = 0
    flagged = 0
    rejected = 0
    quality_data = []

    output_records = (
        db.query(ContentOutput)
        .filter(ContentOutput.piece_id == piece.id)
        .order_by(ContentOutput.created_at.asc())
        .all()
    )

    for i, qr in enumerate(quality_results):
        qd = asdict(qr)
        quality_data.append(qd)

        if i < len(output_records):
            rec = output_records[i]
            meta = rec.platform_metadata or {}
            meta["quality_score"] = qr.overall_score
            meta["quality_verdict"] = qr.verdict
            meta["quality_issues"] = qr.issues
            meta["quality_suggestions"] = qr.suggestions
            meta["quality_summary"] = qr.summary
            meta["quality_scores"] = {
                "hook_strength": qr.hook_strength,
                "pacing": qr.pacing,
                "audio_clarity": qr.audio_clarity,
                "visual_quality": qr.visual_quality,
                "standalone_coherence": qr.standalone_coherence,
            }

            if qr.verdict == "reject":
                meta["status"] = "rejected"
                rejected += 1
            elif qr.verdict == "flag":
                meta["status"] = "flagged"
                flagged += 1
            else:
                meta["status"] = "approved"
                approved += 1

            rec.platform_metadata = meta
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(rec, "platform_metadata")

    piece.status = "review"
    db.commit()

    return {
        "status": "created",
        "piece_id": piece.id,
        "shorts_created": len(outputs),
        "shorts": outputs,
        "quality_checked": len(quality_results) > 0,
        "quality_summary": {
            "approved": approved,
            "flagged": flagged,
            "rejected": rejected,
        },
        "quality_details": quality_data,
        "message": (
            f"Created {len(outputs)} shorts. "
            f"Quality: {approved} approved, {flagged} flagged, {rejected} rejected. "
            f"Review them in Content Output."
        ),
    }


# ═══════════════════════════════════════════════════
# FILE SERVING (for preview)
# ═══════════════════════════════════════════════════

@router.get("/file")
async def serve_content_file(
    path: str = Query(..., description="Relative path within data/content/"),
    token: Optional[str] = Query(None, description="Auth token (for video/img elements)"),
):
    """Serve a content file (video, image) for preview.

    Path is relative to the data/content/ directory.
    Accepts token via query param because HTML video/img
    elements cannot send Authorization headers.
    """
    from fastapi.responses import FileResponse
    from app.auth.config import validate_session

    # Auth via query param (video elements can't send headers)
    if not token or not validate_session(token):
        raise HTTPException(401, "Authentication required")

    base_dir = os.path.abspath(os.path.join("data", "content"))
    full_path = os.path.abspath(os.path.join(base_dir, path))

    # Security: ensure path is within data/content/
    if not full_path.startswith(base_dir):
        raise HTTPException(403, "Access denied")

    if not os.path.exists(full_path):
        raise HTTPException(404, "File not found")

    # Determine media type
    ext = full_path.rsplit(".", 1)[-1].lower() if "." in full_path else ""
    media_types = {
        "mp4": "video/mp4",
        "mov": "video/quicktime",
        "webm": "video/webm",
        "avi": "video/x-msvideo",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
    }
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(
        full_path,
        media_type=media_type,
        filename=os.path.basename(full_path),
    )


# ═══════════════════════════════════════════════════
# LIST OUTPUTS FOR A PIECE
# ═══════════════════════════════════════════════════

@router.get("/piece/{piece_id}/outputs", response_model=dict)
def list_piece_outputs(
    piece_id: str,
    db: Session = Depends(get_db),
):
    """List all content outputs for a specific piece."""
    from app.content.models import ContentPiece, ContentOutput

    piece = db.query(ContentPiece).get(piece_id)
    if not piece:
        raise HTTPException(404, "Piece not found")

    outputs = (
        db.query(ContentOutput)
        .filter(ContentOutput.piece_id == piece_id)
        .order_by(ContentOutput.created_at.asc())
        .all()
    )

    return {
        "piece_id": piece_id,
        "piece_title": piece.title,
        "piece_status": piece.status,
        "count": len(outputs),
        "outputs": [
            {
                "id": o.id,
                "output_format": o.output_format,
                "platform": o.platform,
                "primary_asset_path": o.primary_asset_path,
                "thumbnail_path": o.thumbnail_path,
                "caption_text": o.caption_text,
                "platform_metadata": o.platform_metadata or {},
                "scheduled_at": (
                    o.scheduled_at.isoformat() if o.scheduled_at else None
                ),
                "published_at": (
                    o.published_at.isoformat() if o.published_at else None
                ),
                "platform_post_id": o.platform_post_id,
                "preview_url": (
                    "/content/files/file?path="
                    + o.primary_asset_path.replace("\\", "/").replace("data/content/", "")
                    if o.primary_asset_path else None
                ),
            }
            for o in outputs
        ],
    }




# ═══════════════════════════════════════════════════
# DELETE OUTPUT
# ═══════════════════════════════════════════════════

@router.delete("/outputs/{output_id}", response_model=dict)
def delete_output(
    output_id: str,
    db: Session = Depends(get_db),
):
    """Delete a content output and its associated file."""
    from app.content.models import ContentOutput, ContentPiece

    output = db.query(ContentOutput).get(output_id)
    if not output:
        raise HTTPException(404, "Output not found")

    # Delete the file if it exists
    if output.primary_asset_path and os.path.exists(output.primary_asset_path):
        try:
            os.remove(output.primary_asset_path)
        except OSError as e:
            logger.warning("[production] Failed to delete file: %s", e)

    piece_id = output.piece_id

    db.delete(output)
    db.commit()

    # Update piece output count — if no outputs left, mark as rejected
    remaining = (
        db.query(ContentOutput)
        .filter(ContentOutput.piece_id == piece_id)
        .count()
    )
    if remaining == 0:
        piece = db.query(ContentPiece).get(piece_id)
        if piece:
            piece.status = "rejected"
            db.commit()

    return {
        "deleted": True,
        "output_id": output_id,
        "remaining_outputs": remaining,
    }








