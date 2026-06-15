# FILE: app/media/youtube_search.py
# Purpose: YouTube voice search (Data API v3 search.list + videos.list for
#          durations) and watch-page dispatch to a named display.
# Called-by: app.tools.stream_tools
# Depends-on: app.content.distribution.youtube_auth (existing OAuth), app.media.display_client
# Last-renovated: 2026-06-12
"""
YouTube search for "find me a video about <x>".

Reuses the content pipeline's existing YouTube OAuth (youtube_auth.py,
token at data/content/youtube_token.json) — no new credentials, shared
quota (search.list costs 100 units of the 10k/day pool; fine for voice use).

play_video opens the watch URL on a display via the display manager.
Electron windows allow autoplay (no user-gesture policy), so playback
starts by itself — we deliberately do NOT press 'k' afterwards, which
would pause it again. toggle/skip live in stream_sites ("youtube_watch").
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

WATCH_URL = "https://www.youtube.com/watch?v={video_id}"

# Last search results so "play the second one" resolves.
_recent_videos: List[dict] = []


class YouTubeNotAuthenticated(Exception):
    """No OAuth token — the existing /content/youtube auth flow fixes it."""


def _service():
    from app.content.distribution.youtube_auth import get_youtube_credentials
    creds = get_youtube_credentials()
    if not creds:
        raise YouTubeNotAuthenticated(
            "YouTube isn't connected — connect it in the Content Hub "
            "(existing YouTube auth flow) and try again."
        )
    from googleapiclient.discovery import build
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def parse_iso8601_duration(value: str) -> Optional[int]:
    """PT1H2M3S -> seconds (None when unparseable)."""
    m = re.fullmatch(
        r"P(?:(?P<d>\d+)D)?T?(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+)S)?",
        str(value or ""),
    )
    if not m or not any(m.groupdict().values()):
        return None
    days = int(m.group("d") or 0)
    return ((days * 24 + int(m.group("h") or 0)) * 60 + int(m.group("m") or 0)) * 60 \
        + int(m.group("s") or 0)


def _search_blocking(query: str, max_results: int) -> List[dict]:
    service = _service()
    found = service.search().list(
        part="snippet", q=query, type="video", maxResults=max_results,
        safeSearch="none",
    ).execute()
    ids, rows = [], []
    for item in found.get("items") or []:
        video_id = (item.get("id") or {}).get("videoId")
        snippet = item.get("snippet") or {}
        if not video_id:
            continue
        ids.append(video_id)
        rows.append({
            "video_id": video_id,
            "title": snippet.get("title", ""),
            "channel": snippet.get("channelTitle", ""),
            "published": snippet.get("publishedAt", ""),
            "duration_seconds": None,
        })
    if ids:
        details = service.videos().list(
            part="contentDetails", id=",".join(ids),
        ).execute()
        durations = {
            item.get("id"): parse_iso8601_duration(
                (item.get("contentDetails") or {}).get("duration"))
            for item in details.get("items") or []
        }
        for row in rows:
            row["duration_seconds"] = durations.get(row["video_id"])
    return rows


async def search(query: str, max_results: int = 5) -> List[dict]:
    """Titles, channels, durations, video ids — newest cached for follow-ups."""
    global _recent_videos
    rows = await asyncio.to_thread(_search_blocking, query, max_results)
    _recent_videos = [{**row, "index": i + 1} for i, row in enumerate(rows)]
    return _recent_videos


def resolve_video(selection) -> Optional[dict]:
    """1-based index, video id, URL, or title fragment against the last search."""
    if selection is None or selection == "":
        return _recent_videos[0] if _recent_videos else None
    if isinstance(selection, int) or (isinstance(selection, str) and str(selection).strip().isdigit()):
        idx = int(selection) - 1
        return _recent_videos[idx] if 0 <= idx < len(_recent_videos) else None
    text = str(selection).strip()
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{6,})", text)
    if m:
        return {"video_id": m.group(1), "title": text}
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", text):
        return {"video_id": text, "title": text}
    wanted = text.lower()
    for row in _recent_videos:
        if wanted in row["title"].lower():
            return row
    return None


async def play_video(selection, display: str = "main") -> dict:
    """Open the watch page fullscreen on a display; autoplay does the rest."""
    video = resolve_video(selection)
    if not video:
        return {"ok": False, "error": "I don't have a video in mind — search first "
                                      "or give me a YouTube link"}
    from app.media import display_client, stream_sites
    opened = await display_client.open_on_display(
        WATCH_URL.format(video_id=video["video_id"]),
        display, cdp=True, session="youtube_watch",
    )
    if not opened.get("ok"):
        return {"ok": False, "error": f"couldn't open the window: "
                                      f"{opened.get('error', 'desktop unreachable')}"}
    window = opened.get("result") or {}
    stream_sites.note_video_opened(window.get("window_id"))
    # Best-effort consent dismiss on a fresh partition; autoplay then runs.
    try:
        await stream_sites._dismiss_cookies("youtube_watch")
    except Exception:
        pass
    return {"ok": True, "playing": video.get("title"), "video_id": video["video_id"],
            "window": window, "note": window.get("note")}
