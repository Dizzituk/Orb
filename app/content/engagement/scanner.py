# FILE: app/content/engagement/scanner.py
# Purpose: Comment Scanner.
# Called-by: app.content.engagement.router
# Depends-on: app.content.distribution.facebook, app.content.distribution.tiktok, app.content.distribution.youtube, app.content.engagement.classifier (+3 more)
# Last-renovated: 2026-06-11
"""
Comment Scanner.

Polls platform APIs for new comments on published content.
Normalises comments into the unified EngagementComment schema.
Triggers classification and auto-response preparation.

Designed to run periodically (e.g., every 30 minutes via
background task or cron job).
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.content.models import ContentOutput
from app.content.engagement.models import EngagementComment
from app.content.engagement.classifier import classify_comment
from app.content.engagement.responder import prepare_auto_response

logger = logging.getLogger(__name__)


async def scan_platform_comments(
    db: Session,
    platform: str,
    post_id: str,
    output_id: Optional[str] = None,
) -> List[EngagementComment]:
    """
    Fetch and process new comments for a specific post.
    Returns list of newly created EngagementComment records.
    """
    raw_comments = await _fetch_comments(platform, post_id)
    if not raw_comments:
        return []

    new_comments = []
    for raw in raw_comments:
        # Skip if we already have this comment
        existing = (
            db.query(EngagementComment)
            .filter(
                and_(
                    EngagementComment.platform == platform,
                    EngagementComment.platform_comment_id == raw["id"],
                )
            )
            .first()
        )
        if existing:
            continue

        # Create normalised comment
        comment = EngagementComment(
            platform=platform,
            platform_comment_id=raw["id"],
            platform_post_id=post_id,
            output_id=output_id,
            author_name=raw.get("author_name"),
            author_id=raw.get("author_id"),
            text=raw.get("text", ""),
            posted_at=raw.get("posted_at"),
            like_count=raw.get("like_count", 0),
            reply_count=raw.get("reply_count", 0),
        )

        # Classify
        sentiment, confidence, method = await classify_comment(comment.text)
        comment.sentiment = sentiment
        comment.confidence = confidence
        comment.classification_method = method
        comment.processed = True

        # Flag negative/toxic/question for review
        if sentiment in ("negative", "toxic", "question"):
            comment.flagged = True
            comment.flag_reason = f"Auto-flagged: {sentiment} ({confidence:.0%})"

        db.add(comment)
        db.flush()  # Get the ID before preparing response

        # Prepare auto-response for positive comments
        if sentiment == "positive" and confidence >= 0.6:
            prepare_auto_response(db, comment)

        new_comments.append(comment)

    db.commit()

    if new_comments:
        logger.info(
            f"[scanner] Processed {len(new_comments)} new comments "
            f"on {platform}/{post_id}"
        )

    return new_comments


async def _fetch_comments(
    platform: str,
    post_id: str,
) -> List[Dict[str, Any]]:
    """
    Fetch comments from a platform API and normalise them.
    Returns list of dicts with normalised keys.
    """
    if platform == "youtube":
        return await _fetch_youtube_comments(post_id)
    elif platform == "instagram":
        return await _fetch_instagram_comments(post_id)
    elif platform == "tiktok":
        return await _fetch_tiktok_comments(post_id)
    elif platform == "facebook":
        return await _fetch_facebook_comments(post_id)
    else:
        logger.warning(f"[scanner] Unsupported platform: {platform}")
        return []


async def _fetch_youtube_comments(video_id: str) -> List[Dict[str, Any]]:
    """Fetch and normalise YouTube comments."""
    import httpx
    import os

    headers = None
    try:
        from app.content.distribution.youtube import _get_auth_headers
        headers = await _get_auth_headers()
    except Exception:
        pass

    if not headers:
        # Fall back to API key for public read
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            return []

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://www.googleapis.com/youtube/v3/commentThreads",
                    params={
                        "part": "snippet",
                        "videoId": video_id,
                        "maxResults": 50,
                        "order": "time",
                        "key": api_key,
                    },
                )
                resp.raise_for_status()
                items = resp.json().get("items", [])
        except Exception as e:
            logger.error(f"[scanner] YouTube comment fetch failed: {e}")
            return []
    else:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    "https://www.googleapis.com/youtube/v3/commentThreads",
                    headers=headers,
                    params={
                        "part": "snippet",
                        "videoId": video_id,
                        "maxResults": 50,
                        "order": "time",
                    },
                )
                resp.raise_for_status()
                items = resp.json().get("items", [])
        except Exception as e:
            logger.error(f"[scanner] YouTube comment fetch failed: {e}")
            return []

    normalised = []
    for item in items:
        snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
        normalised.append({
            "id": item.get("id", ""),
            "author_name": snippet.get("authorDisplayName"),
            "author_id": snippet.get("authorChannelId", {}).get("value"),
            "text": snippet.get("textDisplay", ""),
            "posted_at": snippet.get("publishedAt"),
            "like_count": snippet.get("likeCount", 0),
            "reply_count": item.get("snippet", {}).get("totalReplyCount", 0),
        })

    return normalised


async def _fetch_instagram_comments(media_id: str) -> List[Dict[str, Any]]:
    """Fetch and normalise Instagram comments."""
    import httpx
    import os

    token = os.getenv("META_ACCESS_TOKEN") or os.getenv("INSTAGRAM_ACCESS_TOKEN")
    if not token:
        return []

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://graph.facebook.com/v19.0/{media_id}/comments",
                params={
                    "fields": "id,text,username,timestamp,like_count",
                    "limit": 50,
                    "access_token": token,
                },
            )
            resp.raise_for_status()
            items = resp.json().get("data", [])
    except Exception as e:
        logger.error(f"[scanner] Instagram comment fetch failed: {e}")
        return []

    return [
        {
            "id": item.get("id", ""),
            "author_name": item.get("username"),
            "author_id": item.get("username"),
            "text": item.get("text", ""),
            "posted_at": item.get("timestamp"),
            "like_count": item.get("like_count", 0),
            "reply_count": 0,
        }
        for item in items
    ]


async def _fetch_tiktok_comments(video_id: str) -> List[Dict[str, Any]]:
    """Fetch and normalise TikTok comments."""
    from app.content.distribution.tiktok import get_video_comments

    result = await get_video_comments(video_id)
    if not result:
        return []

    return [
        {
            "id": c.get("id", ""),
            "author_name": c.get("user", {}).get("display_name"),
            "author_id": c.get("user", {}).get("id"),
            "text": c.get("text", ""),
            "posted_at": c.get("create_time"),
            "like_count": c.get("like_count", 0),
            "reply_count": c.get("reply_count", 0),
        }
        for c in result.get("comments", [])
    ]


async def _fetch_facebook_comments(post_id: str) -> List[Dict[str, Any]]:
    """Fetch and normalise Facebook comments."""
    from app.content.distribution.facebook import get_post_comments

    result = await get_post_comments(post_id)
    if not result:
        return []

    return [
        {
            "id": c.get("id", ""),
            "author_name": c.get("from", {}).get("name"),
            "author_id": c.get("from", {}).get("id"),
            "text": c.get("message", ""),
            "posted_at": c.get("created_time"),
            "like_count": c.get("like_count", 0),
            "reply_count": c.get("comment_count", 0),
        }
        for c in result.get("comments", [])
    ]


# ═══════════════════════════════════════════════════
# BATCH SCAN (all recent posts)
# ═══════════════════════════════════════════════════

async def scan_all_recent(
    db: Session,
    max_age_days: int = 7,
) -> Dict[str, Any]:
    """
    Scan comments for all recently published outputs.
    Returns summary of scan results.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    outputs = (
        db.query(ContentOutput)
        .filter(
            and_(
                ContentOutput.published_at.isnot(None),
                ContentOutput.published_at >= cutoff,
                ContentOutput.platform_post_id.isnot(None),
            )
        )
        .all()
    )

    total_new = 0
    scanned = 0
    errors = 0

    for output in outputs:
        try:
            new_comments = await scan_platform_comments(
                db,
                platform=output.platform,
                post_id=output.platform_post_id,
                output_id=output.id,
            )
            total_new += len(new_comments)
            scanned += 1
        except Exception as e:
            logger.error(
                f"[scanner] Scan failed for {output.platform}/"
                f"{output.platform_post_id}: {e}"
            )
            errors += 1

    logger.info(
        f"[scanner] Scanned {scanned} posts, "
        f"found {total_new} new comments, {errors} errors"
    )

    return {
        "posts_scanned": scanned,
        "new_comments": total_new,
        "errors": errors,
    }
