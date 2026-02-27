# FILE: app/content/distribution/facebook.py
"""
Facebook Graph API Integration (Spec Section 9.1).

Handles:
- Video publishing (Reels and regular video posts)
- Photo/carousel publishing
- Text post publishing
- Page insights retrieval
- Comment fetching for engagement system

Uses unified Meta credentials: META_ACCESS_TOKEN, META_APP_ID.
Requires a Facebook Page linked to the Meta Developer App.

The Facebook Graph API shares the same auth infrastructure as
Instagram (both use Meta Graph API), but targets different
endpoints for Page publishing vs Instagram container publishing.
"""
import os
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

GRAPH_API_URL = "https://graph.facebook.com/v19.0"


def _get_config() -> Optional[Dict[str, str]]:
    """
    Get Facebook API config from environment.
    Uses unified Meta credentials.
    """
    token = os.getenv("META_ACCESS_TOKEN")
    page_id = os.getenv("FACEBOOK_PAGE_ID")

    if not token:
        logger.warning("[facebook] Missing META_ACCESS_TOKEN")
        return None

    if not page_id:
        logger.warning(
            "[facebook] Missing FACEBOOK_PAGE_ID. "
            "Set this to your Facebook Page ID after creating a page."
        )
        return None

    return {"access_token": token, "page_id": page_id}


# ═══════════════════════════════════════════════════
# PAGE TOKEN EXCHANGE
# ═══════════════════════════════════════════════════

async def _get_page_access_token() -> Optional[Dict[str, str]]:
    """
    Exchange user access token for a Page access token.
    Page tokens are required for publishing on behalf of a Page.
    Returns config dict with page_token and page_id.
    """
    import httpx

    config = _get_config()
    if not config:
        return None

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{GRAPH_API_URL}/{config['page_id']}",
                params={
                    "fields": "access_token,name",
                    "access_token": config["access_token"],
                },
            )
            resp.raise_for_status()
            data = resp.json()

            page_token = data.get("access_token")
            if not page_token:
                logger.error("[facebook] No page token returned")
                return None

            return {
                "page_token": page_token,
                "page_id": config["page_id"],
                "page_name": data.get("name", "Unknown"),
            }

    except Exception as e:
        logger.error(f"[facebook] Page token exchange failed: {e}")
        return None


# ═══════════════════════════════════════════════════
# VIDEO PUBLISHING
# ═══════════════════════════════════════════════════

async def publish_video(
    video_url: str,
    description: str,
    title: Optional[str] = None,
    as_reel: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Publish a video to Facebook Page.

    Args:
        video_url: Publicly accessible HTTPS URL of the video.
        description: Post description text.
        title: Optional video title.
        as_reel: If True, publish as a Reel. If False, regular video post.
    """
    import httpx

    page_config = await _get_page_access_token()
    if not page_config:
        return None

    page_id = page_config["page_id"]
    page_token = page_config["page_token"]

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            post_data = {
                "file_url": video_url,
                "description": description,
                "access_token": page_token,
            }

            if title:
                post_data["title"] = title

            # Reels use a different endpoint
            if as_reel:
                endpoint = f"{GRAPH_API_URL}/{page_id}/video_reels"
            else:
                endpoint = f"{GRAPH_API_URL}/{page_id}/videos"

            resp = await client.post(endpoint, data=post_data)
            resp.raise_for_status()
            result = resp.json()

            video_id = result.get("id")
            logger.info(
                f"[facebook] Published {'reel' if as_reel else 'video'}: "
                f"{video_id}"
            )

            return {
                "post_id": video_id,
                "status": "published",
                "type": "reel" if as_reel else "video",
            }

    except Exception as e:
        logger.error(f"[facebook] Video publish failed: {e}")
        return None


# ═══════════════════════════════════════════════════
# PHOTO / LINK PUBLISHING
# ═══════════════════════════════════════════════════

async def publish_photo(
    image_url: str,
    caption: str,
) -> Optional[Dict[str, Any]]:
    """Publish a photo post to Facebook Page."""
    import httpx

    page_config = await _get_page_access_token()
    if not page_config:
        return None

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{GRAPH_API_URL}/{page_config['page_id']}/photos",
                data={
                    "url": image_url,
                    "caption": caption,
                    "access_token": page_config["page_token"],
                },
            )
            resp.raise_for_status()
            result = resp.json()

            return {
                "post_id": result.get("post_id") or result.get("id"),
                "status": "published",
                "type": "photo",
            }

    except Exception as e:
        logger.error(f"[facebook] Photo publish failed: {e}")
        return None


async def publish_text_post(
    message: str,
    link: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Publish a text (or link) post to Facebook Page."""
    import httpx

    page_config = await _get_page_access_token()
    if not page_config:
        return None

    try:
        async with httpx.AsyncClient() as client:
            post_data: Dict[str, Any] = {
                "message": message,
                "access_token": page_config["page_token"],
            }
            if link:
                post_data["link"] = link

            resp = await client.post(
                f"{GRAPH_API_URL}/{page_config['page_id']}/feed",
                data=post_data,
            )
            resp.raise_for_status()
            result = resp.json()

            return {
                "post_id": result.get("id"),
                "status": "published",
                "type": "link" if link else "text",
            }

    except Exception as e:
        logger.error(f"[facebook] Text post failed: {e}")
        return None


# ═══════════════════════════════════════════════════
# INSIGHTS / ANALYTICS
# ═══════════════════════════════════════════════════

async def get_post_insights(
    post_id: str,
) -> Optional[Dict[str, Any]]:
    """Pull engagement insights for a published post."""
    import httpx

    config = _get_config()
    if not config:
        return None

    metrics = (
        "post_impressions,post_impressions_unique,"
        "post_engaged_users,post_clicks,"
        "post_reactions_like_total"
    )

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{GRAPH_API_URL}/{post_id}/insights",
                params={
                    "metric": metrics,
                    "access_token": config["access_token"],
                },
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])

            insights = {}
            for metric in data:
                name = metric.get("name")
                values = metric.get("values", [{}])
                insights[name] = values[0].get("value", 0)

            # Also fetch basic engagement counts
            engagement_resp = await client.get(
                f"{GRAPH_API_URL}/{post_id}",
                params={
                    "fields": "likes.summary(true),comments.summary(true),"
                              "shares",
                    "access_token": config["access_token"],
                },
            )
            engagement_resp.raise_for_status()
            eng_data = engagement_resp.json()

            likes_count = (
                eng_data.get("likes", {})
                .get("summary", {})
                .get("total_count", 0)
            )
            comments_count = (
                eng_data.get("comments", {})
                .get("summary", {})
                .get("total_count", 0)
            )
            shares_count = eng_data.get("shares", {}).get("count", 0)

            return {
                "impressions": insights.get("post_impressions", 0),
                "reach": insights.get("post_impressions_unique", 0),
                "engaged_users": insights.get("post_engaged_users", 0),
                "clicks": insights.get("post_clicks", 0),
                "likes": likes_count,
                "comments": comments_count,
                "shares": shares_count,
            }

    except Exception as e:
        logger.error(f"[facebook] Insights fetch failed: {e}")
        return None


# ═══════════════════════════════════════════════════
# COMMENTS (for engagement system)
# ═══════════════════════════════════════════════════

async def get_post_comments(
    post_id: str,
    limit: int = 50,
    after_cursor: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fetch comments on a post for the engagement system.
    Returns comments with author info and sentiment data.
    """
    import httpx

    config = _get_config()
    if not config:
        return None

    params: Dict[str, Any] = {
        "fields": "id,message,from,created_time,like_count,comment_count",
        "limit": limit,
        "access_token": config["access_token"],
    }
    if after_cursor:
        params["after"] = after_cursor

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{GRAPH_API_URL}/{post_id}/comments",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

            paging = data.get("paging", {})
            cursors = paging.get("cursors", {})

            return {
                "comments": data.get("data", []),
                "after_cursor": cursors.get("after"),
                "has_more": "next" in paging,
            }

    except Exception as e:
        logger.error(f"[facebook] Comment fetch failed: {e}")
        return None


async def reply_to_comment(
    comment_id: str,
    message: str,
) -> Optional[Dict[str, Any]]:
    """Reply to a comment on a Facebook post."""
    import httpx

    page_config = await _get_page_access_token()
    if not page_config:
        return None

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{GRAPH_API_URL}/{comment_id}/comments",
                data={
                    "message": message,
                    "access_token": page_config["page_token"],
                },
            )
            resp.raise_for_status()
            result = resp.json()

            return {
                "reply_id": result.get("id"),
                "status": "replied",
            }

    except Exception as e:
        logger.error(f"[facebook] Reply failed: {e}")
        return None


def is_configured() -> bool:
    """Check if Facebook API credentials are available."""
    return bool(
        os.getenv("META_ACCESS_TOKEN")
        and os.getenv("FACEBOOK_PAGE_ID")
    )
