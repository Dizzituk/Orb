# FILE: app/content/distribution/instagram.py
"""
Instagram Graph API Integration (Spec Section 9.1).

Handles:
- Reel publishing (video upload → container → publish)
- Carousel publishing (multi-image container)
- Caption and hashtag management
- Insights retrieval (engagement metrics)

Requires: Instagram Business/Creator account linked to
a Facebook Page. Uses META_ACCESS_TOKEN and META_APP_ID from
the unified Meta credentials in settings.

Falls back to legacy INSTAGRAM_ACCESS_TOKEN / INSTAGRAM_ACCOUNT_ID
if Meta credentials are not yet configured.

Note: Instagram Graph API requires app review for publishing
permissions. Development mode works for testing with your own account.
"""
import os
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

GRAPH_API_URL = "https://graph.facebook.com/v19.0"


def _get_config() -> Optional[Dict[str, str]]:
    """
    Get Instagram API config from environment.
    Prefers unified Meta credentials, falls back to legacy Instagram vars.
    """
    # Prefer new unified Meta credentials
    token = os.getenv("META_ACCESS_TOKEN")
    account_id = os.getenv("META_APP_ID")

    # Fall back to legacy env vars
    if not token:
        token = os.getenv("INSTAGRAM_ACCESS_TOKEN")
    if not account_id:
        account_id = os.getenv("INSTAGRAM_ACCOUNT_ID")

    if not token or not account_id:
        logger.warning(
            "[instagram] Missing META_ACCESS_TOKEN/META_APP_ID "
            "(or legacy INSTAGRAM_ACCESS_TOKEN/INSTAGRAM_ACCOUNT_ID)"
        )
        return None

    return {"access_token": token, "account_id": account_id}


# ═══════════════════════════════════════════════════
# REEL PUBLISHING
# ═══════════════════════════════════════════════════

async def publish_reel(
    video_url: str,
    caption: str,
    thumbnail_url: Optional[str] = None,
    share_to_feed: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Publish a Reel to Instagram.

    Note: video_url must be a publicly accessible HTTPS URL.
    The video file needs to be hosted somewhere Instagram can
    fetch it (e.g., Cloudflare R2, S3, or via Cloudflare Tunnel).

    Process:
    1. Create media container with video URL
    2. Wait for processing
    3. Publish the container
    """
    import httpx

    config = _get_config()
    if not config:
        return None

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            # Step 1: Create media container
            container_data = {
                "media_type": "REELS",
                "video_url": video_url,
                "caption": caption,
                "share_to_feed": str(share_to_feed).lower(),
                "access_token": config["access_token"],
            }
            if thumbnail_url:
                container_data["thumb_offset"] = "0"

            resp = await client.post(
                f"{GRAPH_API_URL}/{config['account_id']}/media",
                data=container_data,
            )
            resp.raise_for_status()
            container_id = resp.json().get("id")

            if not container_id:
                logger.error("[instagram] No container ID returned")
                return None

            # Step 2: Check container status (poll until ready)
            import asyncio
            for _ in range(30):  # Max 5 minutes of polling
                status_resp = await client.get(
                    f"{GRAPH_API_URL}/{container_id}",
                    params={
                        "fields": "status_code",
                        "access_token": config["access_token"],
                    },
                )
                status = status_resp.json().get("status_code")
                if status == "FINISHED":
                    break
                if status == "ERROR":
                    logger.error("[instagram] Container processing failed")
                    return None
                await asyncio.sleep(10)

            # Step 3: Publish
            pub_resp = await client.post(
                f"{GRAPH_API_URL}/{config['account_id']}/media_publish",
                data={
                    "creation_id": container_id,
                    "access_token": config["access_token"],
                },
            )
            pub_resp.raise_for_status()
            media_id = pub_resp.json().get("id")

            logger.info(f"[instagram] Published reel: {media_id}")
            return {"media_id": media_id, "status": "published"}

    except Exception as e:
        logger.error(f"[instagram] Reel publish failed: {e}")
        return None


# ═══════════════════════════════════════════════════
# CAROUSEL PUBLISHING
# ═══════════════════════════════════════════════════

async def publish_carousel(
    image_urls: List[str],
    caption: str,
) -> Optional[Dict[str, Any]]:
    """
    Publish a carousel (multi-image) post to Instagram.

    image_urls: List of publicly accessible HTTPS image URLs.
    Each image needs to be hosted where Instagram can fetch it.

    Process:
    1. Create individual item containers for each image
    2. Create carousel container referencing all items
    3. Publish the carousel container
    """
    import httpx

    config = _get_config()
    if not config:
        return None

    if len(image_urls) < 2:
        logger.error("[instagram] Carousel requires at least 2 images")
        return None

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            # Step 1: Create item containers
            item_ids = []
            for url in image_urls[:10]:  # Max 10 items
                resp = await client.post(
                    f"{GRAPH_API_URL}/{config['account_id']}/media",
                    data={
                        "image_url": url,
                        "is_carousel_item": "true",
                        "access_token": config["access_token"],
                    },
                )
                resp.raise_for_status()
                item_id = resp.json().get("id")
                if item_id:
                    item_ids.append(item_id)

            if len(item_ids) < 2:
                logger.error("[instagram] Not enough carousel items created")
                return None

            # Step 2: Create carousel container
            resp = await client.post(
                f"{GRAPH_API_URL}/{config['account_id']}/media",
                data={
                    "media_type": "CAROUSEL",
                    "children": ",".join(item_ids),
                    "caption": caption,
                    "access_token": config["access_token"],
                },
            )
            resp.raise_for_status()
            container_id = resp.json().get("id")

            # Step 3: Publish
            pub_resp = await client.post(
                f"{GRAPH_API_URL}/{config['account_id']}/media_publish",
                data={
                    "creation_id": container_id,
                    "access_token": config["access_token"],
                },
            )
            pub_resp.raise_for_status()
            media_id = pub_resp.json().get("id")

            logger.info(f"[instagram] Published carousel: {media_id}")
            return {
                "media_id": media_id,
                "item_count": len(item_ids),
                "status": "published",
            }

    except Exception as e:
        logger.error(f"[instagram] Carousel publish failed: {e}")
        return None


# ═══════════════════════════════════════════════════
# INSIGHTS / ANALYTICS
# ═══════════════════════════════════════════════════

async def get_media_insights(
    media_id: str,
) -> Optional[Dict[str, Any]]:
    """Pull engagement insights for a published media item."""
    import httpx

    config = _get_config()
    if not config:
        return None

    metrics = "impressions,reach,engagement,saved,shares,likes,comments"

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{GRAPH_API_URL}/{media_id}/insights",
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

            return insights

    except Exception as e:
        logger.error(f"[instagram] Insights fetch failed: {e}")
        return None


def is_configured() -> bool:
    """Check if Instagram API credentials are available."""
    has_meta = bool(
        os.getenv("META_ACCESS_TOKEN")
        and os.getenv("META_APP_ID")
    )
    has_legacy = bool(
        os.getenv("INSTAGRAM_ACCESS_TOKEN")
        and os.getenv("INSTAGRAM_ACCOUNT_ID")
    )
    return has_meta or has_legacy
