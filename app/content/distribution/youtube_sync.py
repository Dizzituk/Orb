# FILE: app/content/distribution/youtube_sync.py
# Purpose: YouTube Channel Sync.
# Called-by: app.content.distribution.youtube_router
# Depends-on: app.content.distribution.youtube_channel, app.content.models
# Last-renovated: 2026-06-11
"""
YouTube Channel Sync.

Pulls existing videos from a YouTube channel into ASTRA's
content database so the dashboard can track them.

Creates ContentPiece + ContentOutput records for each video
found on the channel that isn't already tracked.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from sqlalchemy.orm import Session

from app.content.models import ContentPiece, ContentOutput

logger = logging.getLogger(__name__)


async def sync_channel_videos(
    db: Session,
    max_results: int = 50,
) -> Dict[str, Any]:
    """
    Pull all videos from the authenticated YouTube channel
    and create ASTRA content records for any not already tracked.

    Returns summary of what was synced.
    """
    from app.content.distribution.youtube_channel import (
        get_recent_videos,
        get_channel_info,
    )

    # Get channel info first
    info = await get_channel_info()
    if not info:
        return {"error": "Could not fetch channel info. Check OAuth."}

    channel_id = info["channel_id"]

    # Fetch videos from YouTube
    videos = await get_recent_videos(
        channel_id=channel_id,
        max_results=max_results,
    )

    if not videos:
        return {
            "channel_id": channel_id,
            "videos_found": 0,
            "synced": 0,
            "skipped": 0,
        }

    # Check which video IDs are already tracked
    existing_post_ids = set(
        row[0] for row in
        db.query(ContentOutput.platform_post_id)
        .filter(
            ContentOutput.platform == "youtube",
            ContentOutput.platform_post_id.isnot(None),
        )
        .all()
    )

    synced = 0
    skipped = 0

    for video in videos:
        video_id = video["video_id"]

        if video_id in existing_post_ids:
            skipped += 1
            continue

        try:
            _create_records_for_video(db, video, channel_id)
            synced += 1
        except Exception as e:
            logger.error(
                "[youtube-sync] Failed to create records for %s: %s",
                video_id, e,
            )

    db.commit()

    logger.info(
        "[youtube-sync] Synced %d videos from channel %s "
        "(%d skipped as already tracked)",
        synced, channel_id, skipped,
    )

    return {
        "channel_id": channel_id,
        "channel_name": info.get("title", ""),
        "videos_found": len(videos),
        "synced": synced,
        "skipped": skipped,
    }


def _create_records_for_video(
    db: Session,
    video: Dict[str, Any],
    channel_id: str,
) -> None:
    """Create a ContentPiece and ContentOutput for a YouTube video."""
    published_str = video.get("published_at")
    published_at = None
    if published_str:
        try:
            published_at = datetime.fromisoformat(
                published_str.replace("Z", "+00:00")
            )
        except (ValueError, TypeError):
            published_at = datetime.now(timezone.utc)

    # Create the content piece
    piece = ContentPiece(
        title=video.get("title", "Untitled"),
        description=video.get("description", ""),
        content_category="educational",
        status="published",
        source_conversation_ids=[],
        source_tag_ids=[],
        published_at=published_at,
    )
    db.add(piece)
    db.flush()  # Get the piece ID

    # Determine format from duration
    duration = video.get("duration", "")
    output_format = _classify_format(duration)

    # Create the content output
    output = ContentOutput(
        piece_id=piece.id,
        output_format=output_format,
        platform="youtube",
        caption_text=video.get("description", ""),
        platform_metadata={
            "title": video.get("title", ""),
            "channel_id": channel_id,
            "thumbnail": video.get("thumbnail"),
            "duration": duration,
            "synced_from_youtube": True,
        },
        scheduled_at=published_at,
        published_at=published_at,
        platform_post_id=video["video_id"],
        publish_device="desktop",
    )
    db.add(output)


def _classify_format(duration_iso: str) -> str:
    """Classify YouTube format based on ISO 8601 duration.

    PT1M30S → youtube_short (under 60s)
    PT5M0S  → youtube_longform
    """
    if not duration_iso:
        return "youtube_longform"

    import re
    match = re.match(
        r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration_iso
    )
    if not match:
        return "youtube_longform"

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)

    total_seconds = hours * 3600 + minutes * 60 + seconds

    if total_seconds <= 60:
        return "youtube_short"
    return "youtube_longform"
