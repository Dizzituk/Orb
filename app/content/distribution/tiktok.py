# FILE: app/content/distribution/tiktok.py
"""
TikTok Content Posting API Integration (Spec Section 9.1).

Handles:
- Video uploads via Content Posting API v2
- Direct post and inbox (draft) publishing
- Caption and privacy management
- Video insights retrieval
- Comment fetching for engagement system

Requires: TIKTOK_CLIENT_KEY, TIKTOK_CLIENT_SECRET in .env
OAuth2 credentials from TikTok Developer Portal.

TikTok Content Posting API flow:
1. Init upload → get upload_url
2. PUT video bytes to upload_url
3. POST publish with publish_id
4. Poll status until complete
"""
import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# TikTok API endpoints
TIKTOK_API_URL = "https://open.tiktokapis.com/v2"
TIKTOK_AUTH_URL = "https://open.tiktokapis.com/v2/oauth/token/"

SCOPES = [
    "video.upload",
    "video.publish",
    "video.list",
    "comment.list",
    "comment.list.manage",
    "user.info.basic",
]

TOKEN_PATH = Path("data/content/tiktok_token.json")


# ═══════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════

def _get_credentials() -> Optional[Dict[str, str]]:
    """Load TikTok OAuth credentials from stored token."""
    if not TOKEN_PATH.exists():
        logger.warning("[tiktok] No OAuth token found. Run auth flow first.")
        return None

    try:
        with open(TOKEN_PATH) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"[tiktok] Failed to load credentials: {e}")
        return None


def _save_credentials(creds: Dict[str, str]) -> None:
    """Save OAuth credentials to disk."""
    os.makedirs(TOKEN_PATH.parent, exist_ok=True)
    with open(TOKEN_PATH, "w") as f:
        json.dump(creds, f, indent=2)


async def _refresh_token(creds: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Refresh expired OAuth access token."""
    import httpx

    client_key = os.getenv("TIKTOK_CLIENT_KEY")
    client_secret = os.getenv("TIKTOK_CLIENT_SECRET")
    refresh_token = creds.get("refresh_token")

    if not all([client_key, client_secret, refresh_token]):
        logger.error("[tiktok] Missing OAuth credentials for refresh")
        return None

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                TIKTOK_AUTH_URL,
                data={
                    "client_key": client_key,
                    "client_secret": client_secret,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            resp.raise_for_status()
            data = resp.json()

            creds["access_token"] = data["access_token"]
            if "refresh_token" in data:
                creds["refresh_token"] = data["refresh_token"]
            creds["expires_at"] = (
                datetime.now(timezone.utc).timestamp()
                + data.get("expires_in", 86400)
            )
            creds["open_id"] = data.get("open_id", creds.get("open_id"))
            _save_credentials(creds)
            return creds

    except Exception as e:
        logger.error(f"[tiktok] Token refresh failed: {e}")
        return None


async def _get_auth_headers() -> Optional[Dict[str, str]]:
    """Get authorization headers, refreshing token if needed."""
    creds = _get_credentials()
    if not creds:
        return None

    expires_at = creds.get("expires_at", 0)
    if datetime.now(timezone.utc).timestamp() >= expires_at - 60:
        creds = await _refresh_token(creds)
        if not creds:
            return None

    return {
        "Authorization": f"Bearer {creds['access_token']}",
        "Content-Type": "application/json; charset=UTF-8",
    }


def _get_open_id() -> Optional[str]:
    """Get the TikTok open_id from stored credentials."""
    creds = _get_credentials()
    return creds.get("open_id") if creds else None


# ═══════════════════════════════════════════════════
# VIDEO UPLOAD (Content Posting API v2)
# ═══════════════════════════════════════════════════

async def upload_video(
    video_path: str,
    caption: str,
    privacy: str = "SELF_ONLY",
    disable_duet: bool = False,
    disable_stitch: bool = False,
    disable_comment: bool = False,
    post_mode: str = "DIRECT_POST",
) -> Optional[Dict[str, Any]]:
    """
    Upload a video to TikTok.

    Args:
        video_path: Local path to video file.
        caption: Video description (max 2200 chars).
        privacy: SELF_ONLY | MUTUAL_FOLLOW_FRIENDS | FOLLOWER_OF_CREATOR | PUBLIC_TO_EVERYONE
        post_mode: DIRECT_POST | MEDIA_UPLOAD (inbox/draft)
    """
    import httpx

    headers = await _get_auth_headers()
    if not headers:
        return None

    if not os.path.exists(video_path):
        logger.error(f"[tiktok] Video file not found: {video_path}")
        return None

    file_size = os.path.getsize(video_path)

    try:
        async with httpx.AsyncClient(timeout=600) as client:
            # Step 1: Init upload
            init_body = {
                "post_info": {
                    "title": caption[:2200],
                    "privacy_level": privacy,
                    "disable_duet": disable_duet,
                    "disable_stitch": disable_stitch,
                    "disable_comment": disable_comment,
                },
                "source_info": {
                    "source": "FILE_UPLOAD",
                    "video_size": file_size,
                    "chunk_size": file_size,
                    "total_chunk_count": 1,
                },
            }

            init_resp = await client.post(
                f"{TIKTOK_API_URL}/post/publish/video/init/",
                headers=headers,
                json=init_body,
            )
            init_resp.raise_for_status()
            init_data = init_resp.json().get("data", {})

            publish_id = init_data.get("publish_id")
            upload_url = init_data.get("upload_url")

            if not publish_id or not upload_url:
                logger.error(f"[tiktok] Init upload failed: {init_data}")
                return None

            # Step 2: Upload video bytes
            with open(video_path, "rb") as vf:
                video_data = vf.read()

            upload_headers = {
                "Content-Type": "video/mp4",
                "Content-Range": f"bytes 0-{file_size - 1}/{file_size}",
            }
            upload_resp = await client.put(
                upload_url,
                headers=upload_headers,
                content=video_data,
            )
            upload_resp.raise_for_status()

            # Step 3: Poll publish status
            import asyncio
            for _ in range(30):  # Max ~5 min polling
                status = await _check_publish_status(client, headers, publish_id)
                if status == "PUBLISH_COMPLETE":
                    break
                if status in ("FAILED", "PUBLISH_FAILED"):
                    logger.error(f"[tiktok] Publish failed for {publish_id}")
                    return None
                await asyncio.sleep(10)

            logger.info(f"[tiktok] Published video: {publish_id}")
            return {
                "publish_id": publish_id,
                "status": "published",
            }

    except Exception as e:
        logger.error(f"[tiktok] Upload failed: {e}")
        return None


async def _check_publish_status(
    client, headers: Dict, publish_id: str,
) -> str:
    """Poll the publish status of an uploaded video."""
    try:
        resp = await client.post(
            f"{TIKTOK_API_URL}/post/publish/status/fetch/",
            headers=headers,
            json={"publish_id": publish_id},
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})
        return data.get("status", "UNKNOWN")
    except Exception as e:
        logger.warning(f"[tiktok] Status check failed: {e}")
        return "UNKNOWN"


# ═══════════════════════════════════════════════════
# VIDEO LIST & INSIGHTS
# ═══════════════════════════════════════════════════

async def get_video_list(
    max_count: int = 20,
) -> Optional[List[Dict[str, Any]]]:
    """Get list of user's recent TikTok videos."""
    import httpx

    headers = await _get_auth_headers()
    if not headers:
        return None

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{TIKTOK_API_URL}/video/list/",
                headers=headers,
                json={"max_count": max_count},
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            return data.get("videos", [])

    except Exception as e:
        logger.error(f"[tiktok] Video list failed: {e}")
        return None


async def get_video_insights(
    video_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Pull insights for a specific TikTok video.
    Returns views, likes, comments, shares.
    """
    import httpx

    headers = await _get_auth_headers()
    if not headers:
        return None

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{TIKTOK_API_URL}/video/query/",
                headers=headers,
                json={
                    "filters": {"video_ids": [video_id]},
                    "fields": [
                        "id", "title", "view_count", "like_count",
                        "comment_count", "share_count", "create_time",
                    ],
                },
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            videos = data.get("videos", [])

            if not videos:
                return None

            v = videos[0]
            return {
                "video_id": v.get("id"),
                "views": v.get("view_count", 0),
                "likes": v.get("like_count", 0),
                "comments": v.get("comment_count", 0),
                "shares": v.get("share_count", 0),
            }

    except Exception as e:
        logger.error(f"[tiktok] Video insights failed: {e}")
        return None


# ═══════════════════════════════════════════════════
# COMMENTS (for engagement system)
# ═══════════════════════════════════════════════════

async def get_video_comments(
    video_id: str,
    max_count: int = 50,
    cursor: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Fetch comments on a video for the engagement system.
    Returns list of comments with author info.
    """
    import httpx

    headers = await _get_auth_headers()
    if not headers:
        return None

    body: Dict[str, Any] = {
        "video_id": video_id,
        "max_count": max_count,
    }
    if cursor is not None:
        body["cursor"] = cursor

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{TIKTOK_API_URL}/comment/list/",
                headers=headers,
                json=body,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})

            return {
                "comments": data.get("comments", []),
                "cursor": data.get("cursor"),
                "has_more": data.get("has_more", False),
            }

    except Exception as e:
        logger.error(f"[tiktok] Comment fetch failed: {e}")
        return None


def is_configured() -> bool:
    """Check if TikTok API credentials are available."""
    return bool(
        os.getenv("TIKTOK_CLIENT_KEY")
        and os.getenv("TIKTOK_CLIENT_SECRET")
    )
