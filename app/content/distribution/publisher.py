# FILE: app/content/distribution/publisher.py
# Purpose: Unified Publisher (Spec Section 9).
# Called-by: app.content.distribution.youtube_router, app.content.distribution_router
# Depends-on: app.content.distribution.facebook, app.content.distribution.instagram, app.content.distribution.scheduler, app.content.distribution.tiktok (+3 more)
# Last-renovated: 2026-06-11
"""
Unified Publisher (Spec Section 9).

Single interface for publishing content across all platforms.
Routes to the appropriate platform API based on output format.
Records platform post IDs for analytics tracking.
"""
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from sqlalchemy.orm import Session

from app.content.models import ContentOutput, ContentPiece
from app.content.service import transition_piece_status

logger = logging.getLogger(__name__)


async def publish_output(
    db: Session,
    output_id: str,
) -> Dict[str, Any]:
    """
    Publish a single output to its target platform.
    Routes to the appropriate platform API.
    Returns publishing result with platform post ID.
    """
    output = db.query(ContentOutput).get(output_id)
    if not output:
        raise ValueError(f"Output {output_id} not found")

    if output.published_at:
        return {
            "status": "already_published",
            "output_id": output_id,
            "published_at": output.published_at.isoformat(),
        }

    platform = output.platform
    result = None

    try:
        if platform == "youtube":
            result = await _publish_to_youtube(output)
        elif platform == "instagram":
            result = await _publish_to_instagram(output)
        elif platform == "tiktok":
            result = await _publish_to_tiktok(output)
        elif platform == "facebook":
            result = await _publish_to_facebook(output)
        else:
            result = {
                "status": "unsupported_platform",
                "platform": platform,
                "note": f"Platform '{platform}' publishing not yet implemented",
            }

    except Exception as e:
        logger.error(f"[publisher] Failed to publish to {platform}: {e}")
        result = {"status": "error", "error": str(e)}

    # Update output record
    if result and result.get("status") == "published":
        output.published_at = datetime.now(timezone.utc)
        output.platform_post_id = result.get("post_id")
        db.commit()

        # Check if all outputs for this piece are published
        _check_piece_fully_published(db, output.piece_id)

    return {
        "output_id": output_id,
        "platform": platform,
        "format": output.output_format,
        **result,
    }


async def _publish_to_youtube(output: ContentOutput) -> Dict[str, Any]:
    """Publish to YouTube via Data API."""
    from app.content.distribution.youtube import upload_video, is_configured

    if not is_configured():
        return {
            "status": "not_configured",
            "note": "YouTube API credentials not set in .env",
        }

    if not output.primary_asset_path:
        return {"status": "error", "error": "No video file to upload"}

    metadata = output.platform_metadata or {}
    caption = output.caption_text or ""

    # Append hashtags to description for Shorts discovery
    tags = metadata.get("tags", [])
    if tags and "#Shorts" not in caption:
        # Pick top 4 tags as hashtags + always include #Shorts
        top_tags = [t.replace(" ", "") for t in tags[:4]]
        hashtags = " ".join(f"#{t}" for t in top_tags) + " #Shorts"
        caption = caption.rstrip() + "\n\n" + hashtags

    result = await upload_video(
        video_path=output.primary_asset_path,
        title=metadata.get("title", "Untitled"),
        description=caption,
        tags=metadata.get("tags", []),
        category_id=metadata.get("category_id", "28"),
        privacy=metadata.get("privacy", "public"),
        scheduled_at=output.scheduled_at,
        thumbnail_path=output.thumbnail_path,
    )

    if result and result.get("video_id"):
        return {
            "status": "published",
            "post_id": result["video_id"],
            "url": result["url"],
        }

    return {"status": "error", "error": "Upload failed"}


async def _publish_to_instagram(output: ContentOutput) -> Dict[str, Any]:
    """Publish to Instagram via Graph API."""
    from app.content.distribution.instagram import (
        publish_reel, publish_carousel, is_configured,
    )

    if not is_configured():
        return {
            "status": "not_configured",
            "note": "Instagram API credentials not set in .env",
        }

    fmt = output.output_format
    caption = output.caption_text or ""
    metadata = output.platform_metadata or {}

    if fmt == "instagram_reel":
        video_url = metadata.get("public_video_url")
        if not video_url:
            return {
                "status": "error",
                "error": "Reel requires public_video_url in metadata",
            }
        result = await publish_reel(video_url, caption)

    elif fmt == "instagram_carousel":
        image_urls = metadata.get("public_image_urls", [])
        if len(image_urls) < 2:
            return {
                "status": "error",
                "error": "Carousel requires at least 2 public_image_urls",
            }
        result = await publish_carousel(image_urls, caption)

    else:
        return {"status": "unsupported_format", "format": fmt}

    if result and result.get("media_id"):
        return {
            "status": "published",
            "post_id": result["media_id"],
        }

    return {"status": "error", "error": "Publish failed"}


async def _publish_to_tiktok(output: ContentOutput) -> Dict[str, Any]:
    """Publish to TikTok via Content Posting API."""
    from app.content.distribution.tiktok import upload_video, is_configured

    if not is_configured():
        return {
            "status": "not_configured",
            "note": "TikTok API credentials not set in .env",
        }

    if not output.primary_asset_path:
        return {"status": "error", "error": "No video file to upload"}

    metadata = output.platform_metadata or {}
    caption = output.caption_text or ""

    # Append hashtags to description for Shorts discovery
    tags = metadata.get("tags", [])
    if tags and "#Shorts" not in caption:
        # Pick top 4 tags as hashtags + always include #Shorts
        top_tags = [t.replace(" ", "") for t in tags[:4]]
        hashtags = " ".join(f"#{t}" for t in top_tags) + " #Shorts"
        caption = caption.rstrip() + "\n\n" + hashtags

    result = await upload_video(
        video_path=output.primary_asset_path,
        caption=caption,
        privacy=metadata.get("privacy", "SELF_ONLY"),
        post_mode=metadata.get("post_mode", "DIRECT_POST"),
    )

    if result and result.get("publish_id"):
        return {
            "status": "published",
            "post_id": result["publish_id"],
        }

    return {"status": "error", "error": "Upload failed"}


async def _publish_to_facebook(output: ContentOutput) -> Dict[str, Any]:
    """Publish to Facebook via Graph API."""
    from app.content.distribution.facebook import (
        publish_video, publish_photo, publish_text_post, is_configured,
    )

    if not is_configured():
        return {
            "status": "not_configured",
            "note": "Facebook credentials not set in .env",
        }

    fmt = output.output_format
    caption = output.caption_text or ""
    metadata = output.platform_metadata or {}

    if fmt == "facebook_video":
        video_url = metadata.get("public_video_url")
        if not video_url:
            return {
                "status": "error",
                "error": "Video requires public_video_url in metadata",
            }
        result = await publish_video(
            video_url=video_url,
            description=caption,
            title=metadata.get("title"),
            as_reel=metadata.get("as_reel", True),
            scheduled_at=output.scheduled_at,
        )
    elif fmt in ("facebook_photo", "instagram_carousel"):
        image_url = metadata.get("public_image_url")
        if not image_url:
            return {"status": "error", "error": "Photo requires public_image_url"}
        result = await publish_photo(image_url, caption)
    else:
        # Default to text/link post
        result = await publish_text_post(
            message=caption,
            link=metadata.get("link"),
        )

    if result and result.get("post_id"):
        return {
            "status": "published",
            "post_id": result["post_id"],
        }

    return {"status": "error", "error": "Publish failed"}


def _check_piece_fully_published(db: Session, piece_id: str) -> None:
    """Check if all outputs for a piece are published and update status."""
    piece = db.query(ContentPiece).get(piece_id)
    if not piece:
        return

    outputs = piece.outputs or []
    if not outputs:
        return

    all_published = all(o.published_at is not None for o in outputs)
    if all_published and piece.status != "published":
        transition_piece_status(db, piece_id, "published")
        logger.info(f"[publisher] All outputs published for '{piece.title}'")


async def publish_due(db: Session) -> Dict[str, Any]:
    """
    Publish all outputs that are past their scheduled time.
    Called periodically (e.g., by a cron job or background task).
    """
    from app.content.distribution.scheduler import get_due_for_publishing

    due = get_due_for_publishing(db)
    published = 0
    failed = 0

    for output in due:
        try:
            result = await publish_output(db, output.id)
            if result.get("status") == "published":
                published += 1
            else:
                failed += 1
                logger.warning(
                    f"[publisher] Due publish failed for {output.id}: "
                    f"{result}"
                )
        except Exception as e:
            failed += 1
            logger.error(f"[publisher] Due publish error for {output.id}: {e}")

    return {
        "due_count": len(due),
        "published": published,
        "failed": failed,
    }


