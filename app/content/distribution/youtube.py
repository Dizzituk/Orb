# FILE: app/content/distribution/youtube.py
"""
YouTube Data API v3 Integration (Spec Section 9.1).

Handles:
- Video uploads (resumable, chunked for large files)
- Metadata setting (title, description, tags, category)
- Thumbnail upload
- Scheduled publishing
- Analytics retrieval (YouTube Analytics API)
- Chapter marker generation

Requires: YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET in .env
Google OAuth2 credentials with YouTube Data API v3 scope.
"""
import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# YouTube API constants
YOUTUBE_UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3"
YOUTUBE_ANALYTICS_URL = "https://youtubeanalytics.googleapis.com/v2"

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]

# Video category IDs
CATEGORIES = {
    "science_tech": "28",
    "education": "27",
    "entertainment": "24",
    "people_blogs": "22",
    "news_politics": "25",
    "howto_style": "26",
}

TOKEN_PATH = Path("data/content/youtube_token.json")


def _get_credentials() -> Optional[Dict[str, str]]:
    """Load YouTube OAuth credentials from stored token."""
    if not TOKEN_PATH.exists():
        logger.warning("[youtube] No OAuth token found. Run auth flow first.")
        return None

    try:
        with open(TOKEN_PATH) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"[youtube] Failed to load credentials: {e}")
        return None


def _save_credentials(creds: Dict[str, str]) -> None:
    """Save OAuth credentials to disk."""
    os.makedirs(TOKEN_PATH.parent, exist_ok=True)
    with open(TOKEN_PATH, "w") as f:
        json.dump(creds, f, indent=2)


async def _refresh_token(creds: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Refresh expired OAuth access token."""
    import httpx

    client_id = os.getenv("YOUTUBE_CLIENT_ID")
    client_secret = os.getenv("YOUTUBE_CLIENT_SECRET")
    refresh_token = creds.get("refresh_token")

    if not all([client_id, client_secret, refresh_token]):
        logger.error("[youtube] Missing OAuth credentials for refresh")
        return None

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            resp.raise_for_status()
            data = resp.json()

            creds["access_token"] = data["access_token"]
            if "refresh_token" in data:
                creds["refresh_token"] = data["refresh_token"]
            creds["expires_at"] = (
                datetime.now(timezone.utc).timestamp() + data.get("expires_in", 3600)
            )
            _save_credentials(creds)
            return creds

    except Exception as e:
        logger.error(f"[youtube] Token refresh failed: {e}")
        return None


async def _get_auth_headers() -> Optional[Dict[str, str]]:
    """Get authorization headers via the youtube_auth module.

    Single source of truth for YouTube OAuth credentials.
    Handles refresh automatically via google-auth library.
    """
    from app.content.distribution.youtube_auth import get_youtube_credentials

    creds = get_youtube_credentials()
    if not creds:
        return None

    return {"Authorization": f"Bearer {creds.token}"}


# ═══════════════════════════════════════════════════
# VIDEO UPLOAD
# ═══════════════════════════════════════════════════

async def upload_video(
    video_path: str,
    title: str,
    description: str,
    tags: List[str] = None,
    category_id: str = "28",
    privacy: str = "private",
    scheduled_at: Optional[datetime] = None,
    thumbnail_path: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Upload a video to YouTube.

    Args:
        privacy: "private", "unlisted", or "public"
        scheduled_at: If set, uploads as private then schedules publish
    """
    import httpx

    headers = await _get_auth_headers()
    if not headers:
        return None

    if not os.path.exists(video_path):
        logger.error(f"[youtube] Video file not found: {video_path}")
        return None

    # Build metadata
    publish_at = None
    if scheduled_at and privacy == "private":
        publish_at = scheduled_at.strftime("%Y-%m-%dT%H:%M:%S.0Z")

    metadata = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags or [],
            "categoryId": category_id,
        },
        "status": {
            "privacyStatus": privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    if publish_at:
        metadata["status"]["publishAt"] = publish_at
        metadata["status"]["privacyStatus"] = "private"

    try:
        async with httpx.AsyncClient(timeout=600) as client:
            # Step 1: Initiate resumable upload
            init_resp = await client.post(
                f"{YOUTUBE_UPLOAD_URL}?uploadType=resumable&part=snippet,status",
                headers={
                    **headers,
                    "Content-Type": "application/json; charset=UTF-8",
                    "X-Upload-Content-Type": "video/mp4",
                },
                json=metadata,
            )
            init_resp.raise_for_status()
            upload_url = init_resp.headers.get("Location")

            if not upload_url:
                logger.error("[youtube] No upload URL returned")
                return None

            # Step 2: Upload video data
            with open(video_path, "rb") as vf:
                video_data = vf.read()

            upload_resp = await client.put(
                upload_url,
                headers={"Content-Type": "video/mp4"},
                content=video_data,
            )
            upload_resp.raise_for_status()
            result = upload_resp.json()

            video_id = result.get("id")
            logger.info(f"[youtube] Uploaded video: {video_id}")

            # Step 3: Upload thumbnail if provided
            if thumbnail_path and video_id:
                await _upload_thumbnail(client, headers, video_id, thumbnail_path)

            return {
                "video_id": video_id,
                "url": f"https://youtube.com/watch?v={video_id}",
                "status": result.get("status", {}).get("privacyStatus"),
                "publish_at": publish_at,
            }

    except Exception as e:
        logger.error(f"[youtube] Upload failed: {e}")
        return None


async def _upload_thumbnail(
    client, headers: Dict, video_id: str, thumbnail_path: str,
) -> bool:
    """Upload a custom thumbnail for a video."""
    try:
        with open(thumbnail_path, "rb") as f:
            thumb_data = f.read()

        resp = await client.post(
            f"{YOUTUBE_API_URL}/thumbnails/set?videoId={video_id}",
            headers={**headers, "Content-Type": "image/png"},
            content=thumb_data,
        )
        resp.raise_for_status()
        logger.info(f"[youtube] Thumbnail uploaded for {video_id}")
        return True
    except Exception as e:
        logger.warning(f"[youtube] Thumbnail upload failed: {e}")
        return False


# ═══════════════════════════════════════════════════
# ANALYTICS
# ═══════════════════════════════════════════════════

async def get_video_analytics(
    video_id: str,
) -> Optional[Dict[str, Any]]:
    """Pull analytics for a specific video."""
    import httpx

    headers = await _get_auth_headers()
    if not headers:
        return None

    try:
        async with httpx.AsyncClient() as client:
            # Basic video stats
            resp = await client.get(
                f"{YOUTUBE_API_URL}/videos",
                headers=headers,
                params={
                    "part": "statistics,contentDetails",
                    "id": video_id,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            items = data.get("items", [])
            if not items:
                return None

            stats = items[0].get("statistics", {})

            return {
                "video_id": video_id,
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "comments": int(stats.get("commentCount", 0)),
                "favorites": int(stats.get("favoriteCount", 0)),
            }

    except Exception as e:
        logger.error(f"[youtube] Analytics fetch failed: {e}")
        return None


def generate_chapter_markers(
    chapters: List[Dict[str, Any]],
) -> str:
    """
    Generate YouTube chapter markers for video description.
    Input: [{"time": "0:00", "title": "Introduction"}, ...]
    """
    lines = []
    for ch in chapters:
        lines.append(f"{ch['time']} {ch['title']}")
    return "\n".join(lines)


def is_configured() -> bool:
    """Check if YouTube API credentials are available (any of the three)."""
    return bool(
        (os.getenv("YOUTUBE_CLIENT_ID") and os.getenv("YOUTUBE_CLIENT_SECRET"))
        or os.getenv("YOUTUBE_API_KEY")
    )


def is_authenticated() -> bool:
    """Check if YouTube OAuth is fully authenticated (has valid token)."""
    from app.content.distribution.youtube_auth import get_youtube_credentials
    return get_youtube_credentials() is not None

