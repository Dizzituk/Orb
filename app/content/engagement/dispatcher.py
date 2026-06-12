# FILE: app/content/engagement/dispatcher.py
# Purpose: Response Dispatcher.
# Called-by: app.content.engagement.router
# Depends-on: app.content.distribution.facebook, app.content.distribution.youtube, app.content.engagement.models, app.content.engagement.responder
# Last-renovated: 2026-06-11
"""
Response Dispatcher.

Sends pending auto-responses to platform APIs.
Runs periodically (e.g., every 15 minutes).
Handles platform-specific reply mechanisms.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, Any

from sqlalchemy.orm import Session

from app.content.engagement.models import EngagementComment, EngagementResponse
from app.content.engagement.responder import get_pending_responses

logger = logging.getLogger(__name__)


async def dispatch_pending(db: Session) -> Dict[str, Any]:
    """
    Send all pending auto-responses that are past their send time.
    Returns summary of dispatch results.
    """
    pending = get_pending_responses(db)

    sent = 0
    failed = 0

    for response in pending:
        comment = (
            db.query(EngagementComment)
            .filter(EngagementComment.id == response.comment_id)
            .first()
        )
        if not comment:
            response.send_status = "failed"
            failed += 1
            continue

        success = await _send_reply(
            platform=comment.platform,
            post_id=comment.platform_post_id,
            comment_id=comment.platform_comment_id,
            text=response.response_text,
        )

        if success:
            response.send_status = "sent"
            response.sent_at = datetime.now(timezone.utc)
            comment.responded_at = datetime.now(timezone.utc)
            sent += 1
        else:
            response.send_status = "failed"
            failed += 1

    db.commit()

    if sent or failed:
        logger.info(
            f"[dispatcher] Dispatched: {sent} sent, {failed} failed"
        )

    return {"sent": sent, "failed": failed, "pending": len(pending)}


async def _send_reply(
    platform: str,
    post_id: str,
    comment_id: str,
    text: str,
) -> bool:
    """Send a reply to a specific comment on a platform."""
    try:
        if platform == "youtube":
            return await _reply_youtube(comment_id, text)
        elif platform == "instagram":
            return await _reply_instagram(comment_id, text)
        elif platform == "facebook":
            return await _reply_facebook(comment_id, text)
        elif platform == "tiktok":
            # TikTok Content Posting API doesn't support comment replies
            # via API as of 2025 — log and skip
            logger.info(
                f"[dispatcher] TikTok reply not supported via API, "
                f"skipping comment {comment_id}"
            )
            return False
        else:
            logger.warning(f"[dispatcher] Unknown platform: {platform}")
            return False

    except Exception as e:
        logger.error(f"[dispatcher] Reply failed on {platform}: {e}")
        return False


async def _reply_youtube(comment_id: str, text: str) -> bool:
    """Reply to a YouTube comment."""
    import httpx
    from app.content.distribution.youtube import _get_auth_headers

    headers = await _get_auth_headers()
    if not headers:
        return False

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://www.googleapis.com/youtube/v3/comments",
                headers={**headers, "Content-Type": "application/json"},
                params={"part": "snippet"},
                json={
                    "snippet": {
                        "parentId": comment_id,
                        "textOriginal": text,
                    }
                },
            )
            resp.raise_for_status()
            return True

    except Exception as e:
        logger.error(f"[dispatcher] YouTube reply failed: {e}")
        return False


async def _reply_instagram(comment_id: str, text: str) -> bool:
    """Reply to an Instagram comment."""
    import httpx
    import os

    token = os.getenv("META_ACCESS_TOKEN") or os.getenv("INSTAGRAM_ACCESS_TOKEN")
    if not token:
        return False

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"https://graph.facebook.com/v19.0/{comment_id}/replies",
                data={
                    "message": text,
                    "access_token": token,
                },
            )
            resp.raise_for_status()
            return True

    except Exception as e:
        logger.error(f"[dispatcher] Instagram reply failed: {e}")
        return False


async def _reply_facebook(comment_id: str, text: str) -> bool:
    """Reply to a Facebook comment."""
    from app.content.distribution.facebook import reply_to_comment

    result = await reply_to_comment(comment_id, text)
    return result is not None and result.get("status") == "replied"
