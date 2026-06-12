# FILE: app/content/video_pipeline/pixabay_client.py
# Purpose: Pixabay API client for free stock video search.
# Called-by: app.content.video_pipeline.asset_resolver
# Depends-on: app.db, app.settings.service
# Last-renovated: 2026-06-11
"""
Pixabay API client for free stock video search.

Thin wrapper around pixabay.com/api/videos.
API key loaded from encrypted settings (PIXABAY_API_KEY).

Pixabay has a different library from Pexels — different contributors,
different aesthetic. Used as a second free tier in the asset cascade
to increase visual variety before escalating to paid AI generation.

Rate limit: 100 requests per minute.
License: Free for commercial use, Pixabay mention required.
"""
import os
import logging
import httpx
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

PIXABAY_BASE_URL = "https://pixabay.com/api/videos/"
DOWNLOAD_DIR = Path("data/content/video_pipeline/downloads/pixabay")


def _get_api_key() -> str:
    """Get Pixabay API key from environment."""
    key = os.getenv("PIXABAY_API_KEY", "")
    if not key:
        # Try DB fallback
        try:
            from app.db import SessionLocal
            from app.settings.service import get_key_value
            with SessionLocal() as _db:
                key = get_key_value(_db, "pixabay_api_key") or ""
        except Exception:
            pass
    if not key:
        raise ValueError(
            "PIXABAY_API_KEY not set. Add it in Settings > API Keys."
        )
    return key


async def search_videos(
    query: str,
    orientation: str = "horizontal",
    per_page: int = 10,
    page: int = 1,
    min_width: int = 1280,
    min_height: int = 720,
) -> List[Dict[str, Any]]:
    """
    Search Pixabay for videos matching query.

    Pixabay orientation values: "all", "horizontal", "vertical"
    (different from Pexels which uses "landscape"/"portrait").

    Returns list of video results with:
      id, tags, duration, videos{large, medium, small, tiny}
    """
    key = _get_api_key()
    params = {
        "key": key,
        "q": query,
        "orientation": orientation,
        "per_page": min(per_page, 200),
        "page": page,
        "min_width": min_width,
        "min_height": min_height,
        "safesearch": "true",
    }

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            PIXABAY_BASE_URL,
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()

    hits = data.get("hits", [])
    logger.info(
        f"[pixabay] Search '{query}': {len(hits)} results "
        f"(page {page}, total {data.get('totalHits', 0)})"
    )
    return hits


def pick_best_file(
    video: Dict[str, Any],
    target_height: int = 1080,
) -> Optional[Dict]:
    """
    From a Pixabay video result, pick the best quality file.

    Pixabay nests video files under videos.large, videos.medium, etc.
    Each has: url, width, height, size, thumbnail.
    Prefers 'large' (1920x1080) then 'medium' (1280x720).
    """
    videos = video.get("videos", {})
    if not videos:
        return None

    # Priority order: large (1080p) → medium (720p) → small → tiny
    for quality in ["large", "medium", "small", "tiny"]:
        entry = videos.get(quality, {})
        url = entry.get("url", "")
        height = entry.get("height", 0)
        if url and height > 0:
            if height <= target_height:
                return {
                    "url": url,
                    "width": entry.get("width", 0),
                    "height": height,
                    "size": entry.get("size", 0),
                    "quality": quality,
                }

    # Fallback: return whatever is available
    for quality in ["large", "medium", "small", "tiny"]:
        entry = videos.get(quality, {})
        if entry.get("url"):
            return {
                "url": entry["url"],
                "width": entry.get("width", 0),
                "height": entry.get("height", 0),
                "size": entry.get("size", 0),
                "quality": quality,
            }

    return None


async def download_video(
    video: Dict[str, Any],
    target_height: int = 1080,
) -> Optional[str]:
    """
    Download a Pixabay video to local cache.
    Returns the local file path or None on failure.
    """
    best = pick_best_file(video, target_height)
    if not best:
        logger.warning(
            f"[pixabay] No suitable file for video {video.get('id')}"
        )
        return None

    download_url = best.get("url", "")
    if not download_url:
        return None

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"pixabay_{video['id']}_{best.get('height', 0)}p.mp4"
    filepath = DOWNLOAD_DIR / filename

    # Skip if already downloaded
    if filepath.exists():
        logger.info(f"[pixabay] Already cached: {filepath}")
        return str(filepath)

    async with httpx.AsyncClient(
        timeout=60, follow_redirects=True,
    ) as client:
        resp = await client.get(download_url)
        resp.raise_for_status()
        filepath.write_bytes(resp.content)

    logger.info(
        f"[pixabay] Downloaded: {filepath} "
        f"({best.get('width')}x{best.get('height')}, "
        f"{len(resp.content) / 1024 / 1024:.1f} MB)"
    )
    return str(filepath)


async def search_and_download(
    query: str,
    orientation: str = "horizontal",
    max_results: int = 3,
) -> List[Dict[str, Any]]:
    """
    Search and download top results. Returns list of dicts with
    video metadata + local 'file_path' field.

    Same interface as pexels_client.search_and_download so the
    asset resolver can treat them interchangeably.
    """
    videos = await search_videos(
        query=query,
        orientation=orientation,
        per_page=max_results,
    )

    results = []
    for video in videos[:max_results]:
        path = await download_video(video)
        results.append({
            "id": video.get("id"),
            "duration": video.get("duration", 0),
            "width": video.get("videos", {}).get("large", {}).get("width", 0),
            "height": video.get("videos", {}).get("large", {}).get("height", 0),
            "url": video.get("pageURL", ""),
            "image": video.get("videos", {}).get("large", {}).get("thumbnail", ""),
            "tags": video.get("tags", ""),
            "file_path": path,
        })

    return results
