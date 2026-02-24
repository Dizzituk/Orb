# FILE: app/content/production_router.py
"""
Production Engine API endpoints.

Exposes cutaway generation, draft writing, carousel production,
multi-format output, and video assembly operations.
"""
import logging
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
