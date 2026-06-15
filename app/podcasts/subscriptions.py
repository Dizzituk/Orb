# FILE: app/podcasts/subscriptions.py
# Purpose: JSON-backed podcast subscriptions + RSS feed reading (feedparser) —
#          subscribe/unsubscribe/list and the "any new episodes?" check.
# Called-by: app.tools.podcast_tools
# Depends-on: feedparser, httpx, app.podcasts.podcast_index_client (dataclasses)
# Last-renovated: 2026-06-12
"""
Podcast subscriptions.

Store: data/podcasts/subscriptions.json (sentinel geo-cache pattern —
module-level cache, lazy load, best-effort save). One record per feed:
{feed_url, title, author, added_at, last_seen_published_ts}.

Feeds are fetched with httpx and parsed with feedparser in a worker thread
(feedparser is blocking). new_episodes() returns episodes newer than each
feed's last_seen mark and advances the mark, so asking twice in a row is
quiet the second time — matching how the question is used in conversation.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import feedparser
import httpx

from app.podcasts.podcast_index_client import (
    Episode,
    USER_AGENT,
    truncate_for_tts,
)

logger = logging.getLogger(__name__)

_STORE_FILE = Path(__file__).resolve().parents[2] / "data" / "podcasts" / "subscriptions.json"
_FETCH_TIMEOUT = 15.0

_store: Optional[Dict[str, dict]] = None  # feed_url -> record


def _load() -> Dict[str, dict]:
    global _store
    if _store is None:
        try:
            _store = json.loads(_STORE_FILE.read_text(encoding="utf-8")) if _STORE_FILE.exists() else {}
        except Exception:
            logger.warning("[podcasts] subscriptions store unreadable — starting fresh")
            _store = {}
    return _store


def _save() -> None:
    try:
        _STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STORE_FILE.write_text(json.dumps(_load(), indent=1), encoding="utf-8")
    except Exception as exc:
        logger.warning("[podcasts] subscriptions save failed: %s", exc)


async def fetch_feed(feed_url: str) -> feedparser.FeedParserDict:
    """GET the RSS with httpx, parse with feedparser off the event loop."""
    async with httpx.AsyncClient(
        timeout=_FETCH_TIMEOUT, follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    ) as client:
        resp = await client.get(feed_url)
    resp.raise_for_status()
    parsed = await asyncio.to_thread(feedparser.parse, resp.content)
    if parsed.get("bozo") and not parsed.get("entries"):
        raise ValueError(f"feed unparseable: {parsed.get('bozo_exception')}")
    return parsed


def _entry_published_ts(entry: dict) -> Optional[int]:
    for key in ("published_parsed", "updated_parsed"):
        parsed_time = entry.get(key)
        if parsed_time:
            try:
                return int(time.mktime(parsed_time))
            except Exception:
                continue
    return None


def _entry_enclosure(entry: dict) -> Optional[str]:
    for enc in entry.get("enclosures") or []:
        href = enc.get("href") or enc.get("url")
        if href:
            return str(href)
    for link in entry.get("links") or []:
        if str(link.get("rel")) == "enclosure" and link.get("href"):
            return str(link["href"])
    return None


def episodes_from_parsed(parsed: feedparser.FeedParserDict,
                         max_results: int = 10) -> List[Episode]:
    """feedparser entries -> slim Episode dataclasses (enclosure required)."""
    feed_title = str((parsed.get("feed") or {}).get("title") or "")
    out: List[Episode] = []
    for entry in (parsed.get("entries") or [])[:max_results * 2]:
        enclosure = _entry_enclosure(entry)
        if not enclosure:
            continue
        out.append(Episode(
            title=str(entry.get("title") or "Untitled episode"),
            description=truncate_for_tts(entry.get("summary")),
            enclosure_url=enclosure,
            duration_seconds=None,
            published_ts=_entry_published_ts(entry),
            feed_id=None,
            feed_title=feed_title,
        ))
        if len(out) >= max_results:
            break
    return out


async def subscribe(feed_url: str, title: Optional[str] = None) -> dict:
    """Validate the feed by fetching it, then store the subscription."""
    store = _load()
    if feed_url in store:
        return {"ok": True, "already": True, "title": store[feed_url]["title"]}
    try:
        parsed = await fetch_feed(feed_url)
    except Exception as exc:
        return {"ok": False, "error": f"couldn't read that feed: {exc}"}
    feed_meta = parsed.get("feed") or {}
    episodes = episodes_from_parsed(parsed, max_results=1)
    record = {
        "feed_url": feed_url,
        "title": title or str(feed_meta.get("title") or feed_url),
        "author": str(feed_meta.get("author") or ""),
        "added_at": int(time.time()),
        # start the new-episode mark at the newest current episode, so
        # "any new episodes?" right after subscribing is quiet, not a dump
        "last_seen_published_ts": episodes[0].published_ts if episodes else 0,
    }
    store[feed_url] = record
    _save()
    return {"ok": True, "already": False, "title": record["title"]}


def unsubscribe(ref: str) -> dict:
    """Remove by feed URL or (case-insensitive) title."""
    store = _load()
    key = None
    if ref in store:
        key = ref
    else:
        wanted = ref.strip().lower()
        for url, rec in store.items():
            if rec.get("title", "").strip().lower() == wanted:
                key = url
                break
    if key is None:
        return {"ok": False, "error": f"no subscription matches '{ref}'"}
    removed = store.pop(key)
    _save()
    return {"ok": True, "title": removed.get("title")}


def list_subscriptions() -> List[dict]:
    return [
        {"title": rec.get("title"), "author": rec.get("author"), "feed_url": url}
        for url, rec in _load().items()
    ]


def find_subscription(ref: str) -> Optional[dict]:
    """Feed URL or fuzzy title -> stored record."""
    store = _load()
    if ref in store:
        return store[ref]
    wanted = ref.strip().lower()
    for rec in store.values():
        title = rec.get("title", "").strip().lower()
        if title == wanted or (wanted and wanted in title):
            return rec
    return None


async def new_episodes(max_per_feed: int = 3) -> dict:
    """
    Episodes newer than each feed's last_seen mark, then advance the marks.
    Returns {ok, feeds: [{title, episodes: [...]}], checked, failures}.
    """
    store = _load()
    results: List[dict] = []
    failures: List[str] = []
    for url, rec in list(store.items()):
        try:
            parsed = await fetch_feed(url)
        except Exception as exc:
            failures.append(rec.get("title") or url)
            logger.warning("[podcasts] feed refresh failed for %s: %s", url, exc)
            continue
        episodes = episodes_from_parsed(parsed, max_results=10)
        mark = int(rec.get("last_seen_published_ts") or 0)
        fresh = [e for e in episodes if (e.published_ts or 0) > mark]
        if fresh:
            newest = max(e.published_ts or 0 for e in fresh)
            rec["last_seen_published_ts"] = max(mark, newest)
            results.append({
                "title": rec.get("title"),
                "episodes": [
                    {
                        "title": e.title,
                        "description": e.description,
                        "enclosure_url": e.enclosure_url,
                        "published_ts": e.published_ts,
                    }
                    for e in fresh[:max_per_feed]
                ],
            })
    if results:
        _save()
    return {"ok": True, "feeds": results, "checked": len(store), "failures": failures}
