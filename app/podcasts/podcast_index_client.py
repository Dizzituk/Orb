# FILE: app/podcasts/podcast_index_client.py
# Purpose: PodcastIndex API client (search/trending/episodes) with sha1 auth
#          and a keyless iTunes Search fallback for search.
# Called-by: app.tools.podcast_tools, app.podcasts.subscriptions (dataclasses)
# Depends-on: httpx; env PODCASTINDEX_API_KEY / PODCASTINDEX_API_SECRET
# Last-renovated: 2026-06-12
"""
PodcastIndex client.

Auth (exactly per https://podcastindex-org.github.io/docs-api/):
  X-Auth-Date:   unix epoch seconds, as a string
  X-Auth-Key:    the api key
  Authorization: sha1hex(key + secret + epoch)
  User-Agent:    required, identifies the app

Every public function returns slim dataclasses — raw PodcastIndex JSON never
leaks upward. Descriptions are truncated to 200 chars for TTS at the edge.
If PodcastIndex is down or unkeyed, search() falls back to the iTunes Search
API (no key needed; search only — trending/episodes need PodcastIndex).
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)

API_BASE = "https://api.podcastindex.org/api/1.0"
ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
USER_AGENT = "ASTRA-Orb/1.0 (personal assistant; podcasts module)"
DESCRIPTION_TTS_LIMIT = 200
REQUEST_TIMEOUT = 12.0


class PodcastIndexError(Exception):
    """PodcastIndex unreachable / bad reply / missing keys."""


@dataclass
class PodcastShow:
    title: str
    author: str
    description: str           # already truncated for TTS
    feed_url: str
    feed_id: Optional[int] = None   # PodcastIndex id; None when found via iTunes
    episode_count: Optional[int] = None
    source: str = "podcastindex"    # or "itunes"


@dataclass
class Episode:
    title: str
    description: str           # already truncated for TTS
    enclosure_url: str
    duration_seconds: Optional[int] = None
    published_ts: Optional[int] = None
    feed_id: Optional[int] = None
    feed_title: str = ""


def truncate_for_tts(text: Optional[str], limit: int = DESCRIPTION_TTS_LIMIT) -> str:
    """Hour-long show notes become one spoken-friendly sentence-ish chunk."""
    clean = " ".join(str(text or "").split())
    if len(clean) <= limit:
        return clean
    cut = clean[:limit]
    # break on the last word, not mid-word
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut + "…"


def _credentials() -> Optional[tuple[str, str]]:
    key = (os.getenv("PODCASTINDEX_API_KEY") or "").strip()
    secret = (os.getenv("PODCASTINDEX_API_SECRET") or "").strip()
    return (key, secret) if key and secret else None


def auth_headers(key: str, secret: str, epoch: Optional[int] = None) -> dict:
    """The documented key + secret + unix-time sha1 header set."""
    ts = str(int(epoch if epoch is not None else time.time()))
    digest = hashlib.sha1((key + secret + ts).encode("utf-8")).hexdigest()
    return {
        "User-Agent": USER_AGENT,
        "X-Auth-Key": key,
        "X-Auth-Date": ts,
        "Authorization": digest,
    }


async def _get(path: str, params: dict) -> dict:
    creds = _credentials()
    if not creds:
        raise PodcastIndexError(
            "PodcastIndex keys missing — set PODCASTINDEX_API_KEY and "
            "PODCASTINDEX_API_SECRET (free key from api.podcastindex.org)"
        )
    headers = auth_headers(*creds)
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.get(f"{API_BASE}{path}", params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()
    except PodcastIndexError:
        raise
    except Exception as exc:  # network, HTTP status, JSON — all one failure class
        raise PodcastIndexError(f"PodcastIndex request failed: {exc}") from exc


def _show_from_feed(feed: dict) -> PodcastShow:
    return PodcastShow(
        title=str(feed.get("title") or "Untitled"),
        author=str(feed.get("author") or "Unknown"),
        description=truncate_for_tts(feed.get("description")),
        feed_url=str(feed.get("url") or ""),
        feed_id=feed.get("id"),
        episode_count=feed.get("episodeCount"),
        source="podcastindex",
    )


async def search(term: str, max_results: int = 5) -> List[PodcastShow]:
    """Search shows by term. Falls back to iTunes when PodcastIndex fails."""
    try:
        data = await _get("/search/byterm", {"q": term, "max": max_results})
        return [_show_from_feed(f) for f in (data.get("feeds") or [])[:max_results]]
    except PodcastIndexError as exc:
        logger.warning("[podcasts] PodcastIndex search failed (%s) — iTunes fallback", exc)
        return await itunes_search(term, max_results)


async def trending(category: Optional[str] = None, lang: str = "en",
                   max_results: int = 5) -> List[PodcastShow]:
    """Trending shows (PodcastIndex only — no keyless fallback exists)."""
    params: dict = {"max": max_results, "lang": lang}
    if category:
        params["cat"] = category
    data = await _get("/podcasts/trending", params)
    return [_show_from_feed(f) for f in (data.get("feeds") or [])[:max_results]]


async def episodes_by_feed(feed_id: int, max_results: int = 10) -> List[Episode]:
    """Newest-first episodes for a PodcastIndex feed id."""
    data = await _get("/episodes/byfeedid", {"id": feed_id, "max": max_results})
    out: List[Episode] = []
    for item in (data.get("items") or [])[:max_results]:
        enclosure = str(item.get("enclosureUrl") or "")
        if not enclosure:
            continue
        out.append(Episode(
            title=str(item.get("title") or "Untitled episode"),
            description=truncate_for_tts(item.get("description")),
            enclosure_url=enclosure,
            duration_seconds=item.get("duration"),
            published_ts=item.get("datePublished"),
            feed_id=item.get("feedId"),
            feed_title=str(item.get("feedTitle") or ""),
        ))
    return out


async def itunes_search(term: str, max_results: int = 5) -> List[PodcastShow]:
    """Keyless iTunes Search API — search only, no descriptions or episodes."""
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.get(
                ITUNES_SEARCH_URL,
                params={"media": "podcast", "term": term, "limit": max_results},
                headers={"User-Agent": USER_AGENT},
            )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise PodcastIndexError(f"iTunes fallback also failed: {exc}") from exc

    out: List[PodcastShow] = []
    for row in (data.get("results") or [])[:max_results]:
        feed_url = str(row.get("feedUrl") or "")
        if not feed_url:
            continue
        out.append(PodcastShow(
            title=str(row.get("collectionName") or "Untitled"),
            author=str(row.get("artistName") or "Unknown"),
            description="",  # iTunes search carries no show description
            feed_url=feed_url,
            feed_id=None,
            episode_count=row.get("trackCount"),
            source="itunes",
        ))
    return out
