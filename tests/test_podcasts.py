# FILE: tests/test_podcasts.py
# Purpose: Unit tests for app/podcasts (client auth/fallback, subscriptions
#          round-trip on fixture RSS) + app/tools/podcast_tools resolution.
# Called-by: pytest
# Depends-on: app.podcasts, app.tools.podcast_tools
# Last-renovated: 2026-06-12
from __future__ import annotations

import hashlib
import json
import time

import pytest

from app.podcasts import podcast_index_client as pic
from app.podcasts import subscriptions as subs
from app.tools import podcast_tools


# ── auth + truncation ──────────────────────────────────────────────────────

def test_auth_headers_match_documented_sha1():
    headers = pic.auth_headers("KEY", "SECRET", epoch=1700000000)
    assert headers["X-Auth-Key"] == "KEY"
    assert headers["X-Auth-Date"] == "1700000000"
    expected = hashlib.sha1(b"KEYSECRET1700000000").hexdigest()
    assert headers["Authorization"] == expected
    assert headers["User-Agent"]  # PodcastIndex rejects empty UAs


def test_truncate_for_tts_word_boundary():
    long = "word " * 100
    out = pic.truncate_for_tts(long)
    assert len(out) <= pic.DESCRIPTION_TTS_LIMIT + 1
    assert out.endswith("…")
    assert not out[:-1].endswith(" wor")  # no mid-word cut
    assert pic.truncate_for_tts("short") == "short"
    assert pic.truncate_for_tts(None) == ""


@pytest.mark.asyncio
async def test_search_falls_back_to_itunes(monkeypatch):
    async def broken_get(path, params):
        raise pic.PodcastIndexError("keys missing")

    async def fake_itunes(term, max_results=5):
        return [pic.PodcastShow(title="Fallback Show", author="A", description="",
                                feed_url="https://x/feed.xml", source="itunes")]

    monkeypatch.setattr(pic, "_get", broken_get)
    monkeypatch.setattr(pic, "itunes_search", fake_itunes)
    out = await pic.search("anything")
    assert out[0].source == "itunes"


# ── subscriptions on fixture RSS (no network) ──────────────────────────────

RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>Test Feed</title>
  <item>
    <title>Episode Two</title>
    <description>{desc}</description>
    <pubDate>Wed, 10 Jun 2026 10:00:00 GMT</pubDate>
    <enclosure url="https://cdn.example/ep2.mp3" type="audio/mpeg"/>
  </item>
  <item>
    <title>Episode One</title>
    <pubDate>Mon, 01 Jun 2026 10:00:00 GMT</pubDate>
    <enclosure url="https://cdn.example/ep1.mp3" type="audio/mpeg"/>
  </item>
</channel></rss>"""


@pytest.fixture()
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(subs, "_STORE_FILE", tmp_path / "subscriptions.json")
    monkeypatch.setattr(subs, "_store", None)
    yield


def _patch_feed(monkeypatch, xml: str):
    import feedparser

    async def fake_fetch(feed_url):
        return feedparser.parse(xml.encode("utf-8"))

    monkeypatch.setattr(subs, "fetch_feed", fake_fetch)


@pytest.mark.asyncio
async def test_subscribe_lists_and_unsubscribes(isolated_store, monkeypatch):
    _patch_feed(monkeypatch, RSS.format(desc="x"))
    out = await subs.subscribe("https://x/feed.xml")
    assert out["ok"] and out["title"] == "Test Feed"
    assert subs.list_subscriptions()[0]["title"] == "Test Feed"
    # duplicate subscribe is a friendly no-op
    again = await subs.subscribe("https://x/feed.xml")
    assert again["already"] is True
    gone = subs.unsubscribe("test feed")  # case-insensitive title
    assert gone["ok"]
    assert subs.list_subscriptions() == []


@pytest.mark.asyncio
async def test_new_episodes_quiet_then_loud(isolated_store, monkeypatch):
    _patch_feed(monkeypatch, RSS.format(desc="x"))
    await subs.subscribe("https://x/feed.xml")
    # mark starts at newest episode -> immediate check is quiet
    first = await subs.new_episodes()
    assert first["ok"] and first["feeds"] == []
    # a fresh episode appears
    newer = RSS.format(desc="x").replace(
        "Wed, 10 Jun 2026 10:00:00 GMT", "Thu, 11 Jun 2026 10:00:00 GMT"
    ).replace("Episode Two", "Episode Three")
    _patch_feed(monkeypatch, newer)
    second = await subs.new_episodes()
    assert len(second["feeds"]) == 1
    titles = [e["title"] for e in second["feeds"][0]["episodes"]]
    assert "Episode Three" in titles
    # and asking again right away is quiet (mark advanced)
    third = await subs.new_episodes()
    assert third["feeds"] == []


@pytest.mark.asyncio
async def test_long_descriptions_truncated_in_episodes(isolated_store, monkeypatch):
    _patch_feed(monkeypatch, RSS.format(desc="blah " * 200))
    parsed = await subs.fetch_feed("ignored")
    eps = subs.episodes_from_parsed(parsed)
    assert len(eps[0].description) <= pic.DESCRIPTION_TTS_LIMIT + 1


# ── tool layer: selection resolution + playback dispatch ──────────────────

@pytest.mark.asyncio
async def test_find_then_play_first_one(monkeypatch):
    async def fake_search(topic, max_results=5):
        return [
            pic.PodcastShow(title="AI Today", author="A", description="d",
                            feed_url="https://a/feed.xml", feed_id=11),
            pic.PodcastShow(title="Deep Nets", author="B", description="d",
                            feed_url="https://b/feed.xml", feed_id=22),
        ]

    monkeypatch.setattr(pic, "search", fake_search)
    found = await podcast_tools.find_podcasts_handler({"topic": "ai"}, None)
    assert found["ok"] and found["shows"][0]["index"] == 1

    async def fake_episodes(feed_id, max_results=10):
        assert feed_id == 11  # "the first one"
        return [pic.Episode(title="Newest", description="", enclosure_url="https://cdn/n.mp3")]

    opened = {}

    async def fake_open(url, display="main", **kwargs):
        opened["url"] = url
        opened["display"] = display
        return {"ok": True, "result": {"window_id": 5}}

    monkeypatch.setattr(pic, "episodes_by_feed", fake_episodes)
    from app.media import display_client
    monkeypatch.setattr(display_client, "open_on_display", fake_open)

    played = await podcast_tools.play_podcast_handler({"selection": 1}, None)
    assert played["ok"] and played["playing"] == "Newest"
    assert opened["url"] == "https://cdn/n.mp3"
    assert opened["display"] == "main"


@pytest.mark.asyncio
async def test_play_with_no_context_is_honest():
    podcast_tools._recent_shows.clear()
    podcast_tools._last_played = None
    out = await podcast_tools.play_podcast_handler({}, None)
    assert out["ok"] is False and "search first" in out["error"]


@pytest.mark.asyncio
async def test_play_on_phone_reports_followup():
    out = await podcast_tools.play_podcast_handler({"target": "phone"}, None)
    assert out["ok"] is False and "Bridge" in out["error"]


def test_podcast_tools_register_cleanly():
    from app.tools.registry import list_tools
    podcast_tools.register_podcast_tools()
    names = {t["name"] for t in list_tools()}
    assert {"find_podcasts", "get_trending_podcasts", "play_podcast",
            "subscribe_to_podcast", "unsubscribe_podcast",
            "list_podcast_subscriptions", "check_new_episodes"} <= names
