# FILE: app/content/production/format_converter.py
# Purpose: Multi-Format Output Converter (Spec Section 9.2).
# Called-by: app.content.production_router
# Depends-on: app.content.models, app.content.production.draft_writer, app.content.production.edit_engine, app.content.production.static_gen (+1 more)
# Last-renovated: 2026-06-11
"""
Multi-Format Output Converter (Spec Section 9.2).

Takes a single content piece and produces platform-specific
outputs simultaneously. One conversation → many platforms.

Orchestrates the production subsystems:
- Edit Engine for video formats
- Caption system for all video
- Static Gen for carousels and blogs
- Draft Writer for text content
"""
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session

from app.content.models import ContentPiece, ContentOutput, ContentSeries
from app.content.service import ensure_default_style_profile

logger = logging.getLogger(__name__)

OUTPUT_BASE = Path("data/content/output")

# Platform → device routing (Spec Section 9.5)
DEVICE_ROUTING = {
    "instagram_reel": "phone",
    "instagram_carousel": "phone",
    "tiktok": "phone",
    "youtube_short": "desktop",
    "youtube_longform": "desktop",
    "facebook_video": "desktop",
    "blog_post": "desktop",
    "twitter_thread": "desktop",
}

# Format → platform mapping
FORMAT_PLATFORMS = {
    "instagram_reel": "instagram",
    "instagram_carousel": "instagram",
    "youtube_short": "youtube",
    "youtube_longform": "youtube",
    "tiktok": "tiktok",
    "facebook_video": "facebook",
    "blog_post": "blog",
    "twitter_thread": "twitter",
}


def create_output_record(
    db: Session,
    piece_id: str,
    output_format: str,
    primary_asset_path: Optional[str] = None,
    thumbnail_path: Optional[str] = None,
    caption_text: Optional[str] = None,
    platform_metadata: Optional[Dict] = None,
) -> ContentOutput:
    """Create a ContentOutput record for a produced format."""
    platform = FORMAT_PLATFORMS.get(output_format, "unknown")
    device = DEVICE_ROUTING.get(output_format, "desktop")

    output = ContentOutput(
        piece_id=piece_id,
        output_format=output_format,
        platform=platform,
        primary_asset_path=primary_asset_path,
        thumbnail_path=thumbnail_path,
        caption_text=caption_text,
        platform_metadata=platform_metadata or {},
        publish_device=device,
    )
    db.add(output)
    db.commit()
    db.refresh(output)

    logger.info(
        f"[format_converter] Created output: {output_format} for "
        f"piece {piece_id} (device: {device})"
    )
    return output


async def produce_all_formats(
    db: Session,
    piece_id: str,
) -> List[ContentOutput]:
    """
    Produce all recommended output formats for a content piece.
    Returns list of created ContentOutput records.

    This is the main orchestration function that calls into
    each production subsystem as needed.
    """
    piece = db.query(ContentPiece).get(piece_id)
    if not piece:
        raise ValueError(f"Content piece {piece_id} not found")

    formats = piece.recommended_formats or []
    if not formats:
        logger.warning(f"[format_converter] No formats recommended for {piece_id}")
        return []

    outputs = []
    piece_dir = OUTPUT_BASE / piece_id
    os.makedirs(piece_dir, exist_ok=True)

    for fmt in formats:
        try:
            output = await _produce_single_format(db, piece, fmt)
            if output:
                outputs.append(output)
        except Exception as e:
            logger.error(
                f"[format_converter] Failed to produce {fmt} for "
                f"{piece_id}: {e}"
            )

    # Update piece status
    if outputs:
        piece.status = "review"
        db.commit()

    logger.info(
        f"[format_converter] Produced {len(outputs)}/{len(formats)} "
        f"formats for '{piece.title}'"
    )
    return outputs


async def _produce_single_format(
    db: Session,
    piece: ContentPiece,
    output_format: str,
) -> Optional[ContentOutput]:
    """Produce a single output format for a piece."""

    if output_format in ("blog_post",):
        return await _produce_blog(db, piece)

    if output_format in ("instagram_carousel",):
        return await _produce_carousel(db, piece)

    if output_format in ("twitter_thread",):
        return await _produce_thread(db, piece)

    if output_format in (
        "instagram_reel", "youtube_short", "tiktok",
        "youtube_longform", "facebook_video",
    ):
        return await _produce_video(db, piece, output_format)

    logger.warning(f"[format_converter] Unknown format: {output_format}")
    return None


async def _produce_blog(
    db: Session,
    piece: ContentPiece,
) -> Optional[ContentOutput]:
    """Produce blog post output."""
    # Generate draft if not already done
    if not piece.draft_text:
        from app.content.production.draft_writer import generate_draft
        await generate_draft(db, piece.id, "blog_post")
        db.refresh(piece)

    if not piece.draft_text:
        return None

    # Render to HTML
    from app.content.production.static_gen import save_blog_html
    series_name = piece.series.name if piece.series else None
    path = save_blog_html(piece.id, piece.title, piece.draft_text, series_name)

    return create_output_record(
        db, piece.id, "blog_post",
        primary_asset_path=path,
        caption_text=piece.draft_text[:500],
    )


async def _produce_carousel(
    db: Session,
    piece: ContentPiece,
) -> Optional[ContentOutput]:
    """Produce Instagram carousel output."""
    # Build slides from key excerpts
    excerpts = piece.key_excerpts or []
    if not excerpts:
        return None

    # First slide: hook/title
    slides = [{"title": piece.title, "body": piece.description or ""}]

    # Middle slides: one point per slide
    for excerpt in excerpts[:6]:
        slides.append({"title": "", "body": excerpt})

    # Final slide: CTA
    slides.append({
        "title": "What do you think?",
        "body": "Follow for more on this topic. Drop your thoughts below.",
    })

    from app.content.production.static_gen import generate_carousel
    paths = generate_carousel(piece.id, slides)

    if not paths:
        return None

    return create_output_record(
        db, piece.id, "instagram_carousel",
        primary_asset_path=paths[0],  # First slide as primary
        platform_metadata={"slide_paths": paths, "slide_count": len(paths)},
    )


async def _produce_thread(
    db: Session,
    piece: ContentPiece,
) -> Optional[ContentOutput]:
    """Produce Twitter/X thread output."""
    if not piece.draft_text:
        from app.content.production.draft_writer import generate_draft
        await generate_draft(db, piece.id, "twitter_thread")
        db.refresh(piece)

    if not piece.draft_text:
        return None

    return create_output_record(
        db, piece.id, "twitter_thread",
        caption_text=piece.draft_text,
        platform_metadata={"format": "thread"},
    )


async def _produce_video(
    db: Session,
    piece: ContentPiece,
    output_format: str,
) -> Optional[ContentOutput]:
    """
    Produce video output. This is a placeholder that creates the
    output record — actual video assembly requires FFmpeg and source
    footage which may not be available yet.
    """
    from app.content.production.edit_engine import check_ffmpeg

    if not check_ffmpeg():
        logger.info(
            f"[format_converter] FFmpeg not available — creating "
            f"placeholder output for {output_format}"
        )
        return create_output_record(
            db, piece.id, output_format,
            platform_metadata={
                "status": "pending_ffmpeg",
                "note": "Video assembly requires FFmpeg installation",
            },
        )

    # When FFmpeg is available and source footage exists,
    # the edit engine will handle the actual production.
    # For now, create a record to track the intent.
    return create_output_record(
        db, piece.id, output_format,
        platform_metadata={
            "status": "pending_footage",
            "note": "Awaiting source video footage for assembly",
        },
    )
