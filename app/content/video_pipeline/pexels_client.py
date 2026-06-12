# FILE: app/content/video_pipeline/pexels_client.py
# Purpose: Pexels API client for free stock video search.
# Called-by: app.content.video_pipeline.asset_resolver
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Pexels API client for free stock video search.

Thin wrapper around api.pexels.com/videos/search.
API key loaded from encrypted settings (PEXELS_API_KEY).
"""
import os
import logging
import httpx
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

PEXELS_BASE_URL = "https://api.pexels.com"
DOWNLOAD_DIR = Path("data/content/video_pipeline/downloads/pexels")


def _get_api_key() -> str:
    """Get Pexels API key from environment (synced from encrypted settings)."""
    key = os.getenv("PEXELS_API_KEY", "")
    if not key:
        raise ValueError(
            "PEXELS_API_KEY not set. Add it in Settings > API Keys."
        )
    return key


async def search_videos(
    query: str,
    orientation: str = "landscape",
    min_size: str = "medium",
    per_page: int = 10,
    page: int = 1,
    min_duration: Optional[int] = None,
    max_duration: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Search Pexels for videos matching query.

    Returns list of video results with:
      id, url, duration, width, height, video_files[], image (thumbnail)
    """
    key = _get_api_key()
    params = {
        "query": query,
        "orientation": orientation,
        "size": min_size,
        "per_page": per_page,
        "page": page,
    }
    if min_duration:
        params["min_duration"] = min_duration
    if max_duration:
        params["max_duration"] = max_duration

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            f"{PEXELS_BASE_URL}/videos/search",
            headers={"Authorization": key},
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()

    videos = data.get("videos", [])
    logger.info(
        f"[pexels] Search '{query}': {len(videos)} results "
        f"(page {page}, total {data.get('total_results', 0)})"
    )
    return videos


def pick_best_file(video: Dict[str, Any], target_height: int = 1080) -> Optional[Dict]:
    """
    From a Pexels video result, pick the best quality video file.
    Prefers the file closest to target_height without exceeding it.
    """
    files = video.get("video_files", [])
    if not files:
        return None

    # Sort by height descending, pick closest to target
    suitable = [f for f in files if f.get("height", 0) <= target_height]
    if not suitable:
        suitable = files

    suitable.sort(key=lambda f: f.get("height", 0), reverse=True)
    return suitable[0]


async def download_video(
    video: Dict[str, Any],
    target_height: int = 1080,
) -> Optional[str]:
    """
    Download a Pexels video to local cache.
    Returns the local file path or None on failure.
    """
    best = pick_best_file(video, target_height)
    if not best:
        logger.warning(f"[pexels] No suitable file for video {video.get('id')}")
        return None

    download_url = best.get("link", "")
    if not download_url:
        return None

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ext = best.get("file_type", "video/mp4").split("/")[-1]
    filename = f"pexels_{video['id']}_{best.get('height', 0)}p.{ext}"
    filepath = DOWNLOAD_DIR / filename

    # Skip if already downloaded
    if filepath.exists():
        logger.info(f"[pexels] Already cached: {filepath}")
        return str(filepath)

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        resp = await client.get(download_url)
        resp.raise_for_status()
        filepath.write_bytes(resp.content)

    logger.info(
        f"[pexels] Downloaded: {filepath} "
        f"({best.get('width')}x{best.get('height')}, "
        f"{len(resp.content) / 1024 / 1024:.1f} MB)"
    )
    return str(filepath)


async def search_and_download(
    query: str,
    orientation: str = "landscape",
    max_results: int = 3,
) -> List[Dict[str, Any]]:
    """
    Search and download top results. Returns list of dicts with
    video metadata + local 'file_path' field.
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
            "width": video.get("width", 0),
            "height": video.get("height", 0),
            "url": video.get("url", ""),
            "image": video.get("image", ""),
            "file_path": path,
        })

    return results
