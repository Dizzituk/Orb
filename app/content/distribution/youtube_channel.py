# FILE: app/content/distribution/youtube_channel.py
"""
YouTube Channel Analytics.

Provides channel-level data for the Social Media dashboard:
- Channel info (name, subscriber count, total views)
- Recent video list with stats
- Channel-wide analytics via YouTube Analytics API

Uses two paths:
- API key for public reads (channel stats, video lists)
- OAuth for private data (YouTube Analytics API, watch time)
"""
import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

import httpx

logger = logging.getLogger(__name__)

YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3"
YOUTUBE_ANALYTICS_URL = "https://youtubeanalytics.googleapis.com/v2"


def _get_api_key() -> Optional[str]:
    """Get the YouTube API key for public reads."""
    return os.getenv("YOUTUBE_API_KEY")


async def _get_oauth_headers() -> Optional[Dict[str, str]]:
    """Get OAuth headers from the auth module."""
    from app.content.distribution.youtube_auth import get_youtube_credentials

    creds = get_youtube_credentials()
    if not creds:
        return None

    return {"Authorization": f"Bearer {creds.token}"}


# ═══════════════════════════════════════════════════
# CHANNEL INFO (API Key — public data)
# ═══════════════════════════════════════════════════

async def get_channel_info(
    channel_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get channel details: name, description, subscriber count, etc.

    If channel_id is None and OAuth is available, fetches
    the authenticated user's channel. Otherwise uses API key.
    """
    headers = {}
    params: Dict[str, Any] = {
        "part": "snippet,statistics,brandingSettings",
    }

    # Try OAuth first (can get 'mine=true')
    if not channel_id:
        oauth_headers = await _get_oauth_headers()
        if oauth_headers:
            headers = oauth_headers
            params["mine"] = "true"
        else:
            logger.warning("[youtube-channel] No channel_id and no OAuth")
            return None
    else:
        api_key = _get_api_key()
        if api_key:
            params["key"] = api_key
            params["id"] = channel_id
        else:
            # Fall back to OAuth
            oauth_headers = await _get_oauth_headers()
            if not oauth_headers:
                return None
            headers = oauth_headers
            params["id"] = channel_id

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{YOUTUBE_API_URL}/channels",
                headers=headers,
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

            items = data.get("items", [])
            if not items:
                return None

            item = items[0]
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})

            return {
                "channel_id": item["id"],
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "custom_url": snippet.get("customUrl", ""),
                "published_at": snippet.get("publishedAt"),
                "thumbnail": (
                    snippet.get("thumbnails", {})
                    .get("medium", {})
                    .get("url")
                ),
                "subscriber_count": int(
                    stats.get("subscriberCount", 0)
                ),
                "video_count": int(stats.get("videoCount", 0)),
                "view_count": int(stats.get("viewCount", 0)),
                "hidden_subscriber_count": stats.get(
                    "hiddenSubscriberCount", False
                ),
            }

    except Exception as e:
        logger.error("[youtube-channel] get_channel_info failed: %s", e)
        return None


# ═══════════════════════════════════════════════════
# RECENT VIDEOS (API Key — public data)
# ═══════════════════════════════════════════════════

async def get_recent_videos(
    channel_id: Optional[str] = None,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    Get recent videos for a channel with stats.

    Two-step: search for video IDs, then fetch statistics.
    """
    # Step 1: Resolve channel ID if not provided
    if not channel_id:
        info = await get_channel_info()
        if not info:
            return []
        channel_id = info["channel_id"]

    api_key = _get_api_key()
    headers = {}
    base_params: Dict[str, Any] = {}

    if api_key:
        base_params["key"] = api_key
    else:
        oauth_headers = await _get_oauth_headers()
        if not oauth_headers:
            return []
        headers = oauth_headers

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            # Search for recent uploads
            search_resp = await client.get(
                f"{YOUTUBE_API_URL}/search",
                headers=headers,
                params={
                    **base_params,
                    "channelId": channel_id,
                    "part": "snippet",
                    "order": "date",
                    "maxResults": max_results,
                    "type": "video",
                },
            )
            search_resp.raise_for_status()
            search_data = search_resp.json()

            video_ids = [
                item["id"]["videoId"]
                for item in search_data.get("items", [])
                if item.get("id", {}).get("videoId")
            ]

            if not video_ids:
                return []

            # Fetch stats for all videos in one call
            stats_resp = await client.get(
                f"{YOUTUBE_API_URL}/videos",
                headers=headers,
                params={
                    **base_params,
                    "part": "snippet,statistics,contentDetails",
                    "id": ",".join(video_ids),
                },
            )
            stats_resp.raise_for_status()
            stats_data = stats_resp.json()

            results = []
            for item in stats_data.get("items", []):
                snippet = item.get("snippet", {})
                stats = item.get("statistics", {})
                content = item.get("contentDetails", {})

                results.append({
                    "video_id": item["id"],
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", "")[:200],
                    "published_at": snippet.get("publishedAt"),
                    "thumbnail": (
                        snippet.get("thumbnails", {})
                        .get("medium", {})
                        .get("url")
                    ),
                    "duration": content.get("duration", ""),
                    "views": int(stats.get("viewCount", 0)),
                    "likes": int(stats.get("likeCount", 0)),
                    "comments": int(stats.get("commentCount", 0)),
                })

            return results

    except Exception as e:
        logger.error("[youtube-channel] get_recent_videos failed: %s", e)
        return []


# ═══════════════════════════════════════════════════
# CHANNEL ANALYTICS (OAuth — private data)
# ═══════════════════════════════════════════════════

async def get_channel_analytics(
    days: int = 30,
) -> Optional[Dict[str, Any]]:
    """
    Pull channel-wide analytics from the YouTube Analytics API.

    Requires OAuth with yt-analytics.readonly scope.
    Returns daily views, watch time, subscribers gained/lost.
    """
    headers = await _get_oauth_headers()
    if not headers:
        logger.warning("[youtube-channel] No OAuth for analytics")
        return None

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days)

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{YOUTUBE_ANALYTICS_URL}/reports",
                headers=headers,
                params={
                    "ids": "channel==MINE",
                    "startDate": start_date.isoformat(),
                    "endDate": end_date.isoformat(),
                    "metrics": (
                        "views,estimatedMinutesWatched,"
                        "averageViewDuration,subscribersGained,"
                        "subscribersLost,likes,dislikes,comments"
                    ),
                    "dimensions": "day",
                    "sort": "day",
                },
            )
            resp.raise_for_status()
            data = resp.json()

            rows = data.get("rows", [])
            headers_list = [
                h["name"] for h in data.get("columnHeaders", [])
            ]

            daily = []
            totals = {
                "views": 0,
                "watch_time_minutes": 0,
                "subscribers_gained": 0,
                "subscribers_lost": 0,
                "likes": 0,
                "comments": 0,
            }

            for row in rows:
                entry = dict(zip(headers_list, row))
                daily.append(entry)

                totals["views"] += entry.get("views", 0)
                totals["watch_time_minutes"] += entry.get(
                    "estimatedMinutesWatched", 0
                )
                totals["subscribers_gained"] += entry.get(
                    "subscribersGained", 0
                )
                totals["subscribers_lost"] += entry.get(
                    "subscribersLost", 0
                )
                totals["likes"] += entry.get("likes", 0)
                totals["comments"] += entry.get("comments", 0)

            totals["net_subscribers"] = (
                totals["subscribers_gained"]
                - totals["subscribers_lost"]
            )

            return {
                "period_days": days,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "totals": totals,
                "daily": daily,
            }

    except Exception as e:
        logger.error(
            "[youtube-channel] get_channel_analytics failed: %s", e
        )
        return None


async def get_top_videos_analytics(
    days: int = 30,
    max_results: int = 10,
) -> Optional[List[Dict[str, Any]]]:
    """
    Get top performing videos by views over a period.

    Uses YouTube Analytics API (OAuth required).
    """
    headers = await _get_oauth_headers()
    if not headers:
        return None

    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days)

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{YOUTUBE_ANALYTICS_URL}/reports",
                headers=headers,
                params={
                    "ids": "channel==MINE",
                    "startDate": start_date.isoformat(),
                    "endDate": end_date.isoformat(),
                    "metrics": (
                        "views,estimatedMinutesWatched,"
                        "averageViewDuration,likes,comments"
                    ),
                    "dimensions": "video",
                    "sort": "-views",
                    "maxResults": max_results,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            rows = data.get("rows", [])
            header_names = [
                h["name"] for h in data.get("columnHeaders", [])
            ]

            results = []
            for row in rows:
                entry = dict(zip(header_names, row))
                results.append(entry)

            return results

    except Exception as e:
        logger.error(
            "[youtube-channel] top_videos_analytics failed: %s", e
        )
        return None
