# FILE: app/social/meta_client.py
"""
Meta (Facebook) Graph API client.

Low-level HTTP client for posting to Facebook Pages via the Graph API.
Handles credential resolution from settings, multipart photo upload,
scheduled posts, and cross-channel verification (post-exists check
after publish).

Public surface:
  - MetaApiError                          : structured exception
  - get_token() / get_page_id()           : credential resolution from settings
  - post_photo_to_page(...)               : multipart upload + publish/schedule
  - verify_post_exists(post_id)           : cross-channel GET verification

Why direct HTTP instead of facebook-sdk: facebook-sdk is unmaintained
(last release 2017). Direct httpx keeps the dependency surface small and
lets us target any v19+ Graph endpoint without waiting on third-party
maintenance.

Why kwargs only: making token / page_id explicit kwargs (with settings
fallback) keeps this client unit-testable without monkeypatching env vars.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

GRAPH_API_VERSION = "v21.0"
GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_API_VERSION}"

# Meta scheduling window: must be >10 min and <6 months in the future.
# Pad the lower bound by 30s for clock skew safety.
MIN_SCHEDULE_LEAD_SECONDS = 10 * 60 + 30
MAX_SCHEDULE_LEAD_SECONDS = 180 * 24 * 3600


class MetaApiError(Exception):
    """Graph API returned a non-2xx response or unexpected payload."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        graph_code: Optional[int] = None,
        graph_subcode: Optional[int] = None,
        graph_type: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.graph_code = graph_code
        self.graph_subcode = graph_subcode
        self.graph_type = graph_type


# =============================================================================
# CREDENTIAL RESOLUTION
# =============================================================================

def get_token() -> Optional[str]:
    """
    Resolve Meta access token from environment.

    Settings service syncs DB-stored keys into os.environ at startup, so
    reading the env var picks up either source transparently. Returns
    None when nothing is configured — callers should report a clean
    config error rather than blow up.
    """
    return os.getenv("META_ACCESS_TOKEN") or None


def get_page_id() -> Optional[str]:
    """Resolve Facebook Page ID from settings/env."""
    return os.getenv("FACEBOOK_PAGE_ID") or None


# =============================================================================
# REQUEST HELPERS
# =============================================================================

def _parse_graph_error(resp: httpx.Response) -> MetaApiError:
    """Convert a non-2xx Graph API response into a structured error."""
    try:
        body = resp.json()
        err = body.get("error", {}) if isinstance(body, dict) else {}
        message = err.get("message") or f"HTTP {resp.status_code}"
        return MetaApiError(
            message,
            status_code=resp.status_code,
            graph_code=err.get("code"),
            graph_subcode=err.get("error_subcode"),
            graph_type=err.get("type"),
        )
    except Exception:
        return MetaApiError(
            f"HTTP {resp.status_code}: {resp.text[:200]}",
            status_code=resp.status_code,
        )


_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _guess_mime(path: Path) -> str:
    return _MIME_BY_EXT.get(path.suffix.lower(), "application/octet-stream")


# =============================================================================
# POST PHOTO TO PAGE
# =============================================================================

async def post_photo_to_page(
    image_path: str,
    caption: str,
    *,
    scheduled_at: Optional[int] = None,
    page_id: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Upload an image to a Facebook Page via multipart and publish (or schedule).

    Args:
        image_path   : Absolute path to image on disk.
        caption      : Post caption text. Empty string allowed.
        scheduled_at : Optional Unix timestamp for scheduled publish.
                       Must be 10 minutes to 180 days in the future.
                       None = publish immediately.
        page_id      : Override Page ID. Defaults to settings.
        token        : Override access token. Defaults to settings.

    Returns dict with keys:
        id              : photo object ID
        post_id         : Page post ID (used for permalink lookups)
        permalink_url   : Public URL of the post (best-effort; may be None
                          for scheduled posts which aren't queryable yet)
        scheduled       : True if scheduled, False if published immediately
        scheduled_at    : Unix timestamp if scheduled, else None

    Raises:
        MetaApiError      on Graph API failure.
        FileNotFoundError if image_path doesn't exist.
        ValueError        on invalid scheduled_at or missing credentials.
    """
    page_id = page_id or get_page_id()
    token = token or get_token()
    if not page_id:
        raise ValueError(
            "Facebook Page ID not configured. Set 'facebook_page_id' in "
            "Settings or FACEBOOK_PAGE_ID env var."
        )
    if not token:
        raise ValueError(
            "Meta access token not configured. Set 'meta_access_token' in "
            "Settings or META_ACCESS_TOKEN env var."
        )

    img = Path(image_path)
    if not img.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Validate scheduling window
    published = True
    if scheduled_at is not None:
        now = int(time.time())
        lead = scheduled_at - now
        if lead < MIN_SCHEDULE_LEAD_SECONDS:
            raise ValueError(
                f"Scheduled time too soon: must be at least 11 minutes in the "
                f"future (got {lead}s lead)."
            )
        if lead > MAX_SCHEDULE_LEAD_SECONDS:
            raise ValueError(
                "Scheduled time too far: max 180 days in the future."
            )
        published = False

    url = f"{GRAPH_BASE}/{page_id}/photos"
    data: Dict[str, Any] = {
        "caption": caption,
        "access_token": token,
        "published": "true" if published else "false",
    }
    if not published:
        data["scheduled_publish_time"] = str(scheduled_at)

    logger.info(
        "[meta_client] Posting photo to page %s (scheduled=%s, image=%s)",
        page_id, not published, img.name,
    )

    async with httpx.AsyncClient(timeout=60) as client:
        with img.open("rb") as fh:
            files = {"source": (img.name, fh, _guess_mime(img))}
            resp = await client.post(url, data=data, files=files)

    if resp.status_code >= 400:
        raise _parse_graph_error(resp)

    body = resp.json()
    photo_id = body.get("id")
    post_id = body.get("post_id") or photo_id

    # Best-effort permalink lookup (only meaningful for immediately-published
    # posts; scheduled posts don't have one until they go live).
    permalink: Optional[str] = None
    if published and post_id:
        try:
            permalink = await _fetch_permalink(post_id, token)
        except MetaApiError as exc:
            logger.warning(
                "[meta_client] Permalink lookup failed for %s: %s",
                post_id, exc,
            )

    return {
        "id": photo_id,
        "post_id": post_id,
        "permalink_url": permalink,
        "scheduled": not published,
        "scheduled_at": scheduled_at,
    }


# =============================================================================
# CROSS-CHANNEL VERIFICATION
# =============================================================================

async def _fetch_permalink(post_id: str, token: str) -> Optional[str]:
    """Fetch the permalink_url for a published post."""
    url = f"{GRAPH_BASE}/{post_id}"
    params = {"fields": "permalink_url", "access_token": token}
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params)
    if resp.status_code >= 400:
        raise _parse_graph_error(resp)
    return resp.json().get("permalink_url")


async def verify_post_exists(
    post_id: str,
    *,
    token: Optional[str] = None,
) -> bool:
    """
    Cross-channel verify: confirm the post is live in Meta's graph.

    Uses GET on the post object — different HTTP verb and endpoint from
    the upload, so a successful read here means the post genuinely exists
    in Meta's graph, not just that the upload appeared to succeed.
    Returns False on any error so callers can treat it as a soft signal.
    """
    token = token or get_token()
    if not token:
        return False
    try:
        await _fetch_permalink(post_id, token)
        return True
    except MetaApiError:
        return False
