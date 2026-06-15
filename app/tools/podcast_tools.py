# FILE: app/tools/podcast_tools.py
# Purpose: Podcast chat tools — find/trending/play/subscribe/new-episodes by
#          voice, with "play the first one" follow-up state.
# Called-by: app.tools.registry (_register_defaults)
# Depends-on: app.podcasts (podcast_index_client, subscriptions), app.media.display_client
# Last-renovated: 2026-06-12
"""
Podcast chat tools (v1, 2026-06-12).

  find_podcasts               "find me a podcast about decentralised AI"
  get_trending_podcasts       "what podcasts are people listening to"
  play_podcast                "play that" / "play the first one" / "play <title>"
  subscribe_to_podcast        "subscribe to that"
  unsubscribe_podcast         "unsubscribe from <title>"
  list_podcast_subscriptions  "what am I subscribed to"
  check_new_episodes          "any new episodes?"

Conversational state: the last search/trending results are remembered in
memory so "play the first one" / "subscribe to that" resolve without the
model repeating URLs. Playback target v1 is the desktop: the episode MP3
opens in a small display-manager window (the audio pane replaces this
later). Phone playback = follow-up Bridge task, reported honestly.

Handler style mirrors vehicle_tools: errors return {ok: false, error};
schemas inline; self-contained.
"""
from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Last search/trending results, newest first; "play the first one" indexes this.
_recent_shows: List[dict] = []
_last_played: Optional[dict] = None


def _remember_shows(shows) -> List[dict]:
    global _recent_shows
    _recent_shows = [
        {
            "index": i + 1,
            "title": s.title,
            "author": s.author,
            "description": s.description,
            "feed_url": s.feed_url,
            "feed_id": s.feed_id,
            "source": s.source,
        }
        for i, s in enumerate(shows)
    ]
    return _recent_shows


def _resolve_show(selection) -> Optional[dict]:
    """1-based index, title fragment, feed URL, or None = first recent/last played."""
    from app.podcasts import subscriptions

    if selection is None or selection == "":
        if _recent_shows:
            return _recent_shows[0]
        return _last_played
    if isinstance(selection, int) or (isinstance(selection, str) and selection.strip().isdigit()):
        idx = int(selection) - 1
        return _recent_shows[idx] if 0 <= idx < len(_recent_shows) else None
    text = str(selection).strip()
    if text.startswith("http"):
        return {"title": text, "feed_url": text, "feed_id": None}
    wanted = text.lower()
    for show in _recent_shows:
        if wanted in show["title"].lower():
            return show
    sub = subscriptions.find_subscription(text)
    if sub:
        return {"title": sub.get("title"), "feed_url": sub.get("feed_url"), "feed_id": None}
    return None


async def _episodes_for_show(show: dict, max_results: int = 5) -> list:
    """Newest-first episodes — PodcastIndex by feed id, else direct RSS."""
    from app.podcasts import podcast_index_client, subscriptions

    if show.get("feed_id"):
        try:
            return await podcast_index_client.episodes_by_feed(show["feed_id"], max_results)
        except podcast_index_client.PodcastIndexError as exc:
            logger.warning("[podcast_tools] episodes via index failed (%s) — RSS direct", exc)
    parsed = await subscriptions.fetch_feed(show["feed_url"])
    return subscriptions.episodes_from_parsed(parsed, max_results=max_results)


# ────────────────────────────────────────────────────────────────────────
# Handlers
# ────────────────────────────────────────────────────────────────────────

async def find_podcasts_handler(input_data: dict, context: Optional[dict]) -> dict:
    topic = (input_data.get("topic") or "").strip()
    if not topic:
        return {"ok": False, "error": "topic is required"}
    try:
        from app.podcasts import podcast_index_client
        shows = await podcast_index_client.search(
            topic, max_results=int(input_data.get("max", 5))
        )
        if not shows:
            return {"ok": True, "shows": [], "note": "nothing found for that topic"}
        return {"ok": True, "shows": _remember_shows(shows)}
    except Exception as exc:
        logger.exception("[podcast_tools] find_podcasts failed")
        return {"ok": False, "error": str(exc)}


async def get_trending_podcasts_handler(input_data: dict, context: Optional[dict]) -> dict:
    try:
        from app.podcasts import podcast_index_client
        shows = await podcast_index_client.trending(
            category=input_data.get("category"),
            max_results=int(input_data.get("max", 5)),
        )
        return {"ok": True, "shows": _remember_shows(shows)}
    except Exception as exc:
        logger.exception("[podcast_tools] get_trending_podcasts failed")
        return {"ok": False, "error": str(exc)}


async def play_podcast_handler(input_data: dict, context: Optional[dict]) -> dict:
    global _last_played
    target = (input_data.get("target") or "desktop").lower()
    if target == "phone":
        return {
            "ok": False,
            "error": "Phone podcast playback isn't wired yet (follow-up Bridge task) "
                     "— I can play it on the desktop instead.",
        }
    show = _resolve_show(input_data.get("selection"))
    if not show:
        return {"ok": False, "error": "I don't have a podcast in mind — search first "
                                      "or give me a show name/feed URL"}
    try:
        episodes = await _episodes_for_show(show, max_results=5)
        if not episodes:
            return {"ok": False, "error": f"no playable episodes found for {show['title']}"}
        ep_index = max(1, int(input_data.get("episode_index", 1)))
        episode = episodes[min(ep_index, len(episodes)) - 1]

        from app.media import display_client
        opened = await display_client.open_on_display(
            episode.enclosure_url,
            display=input_data.get("display") or "main",
            fullscreen=False,  # minimal audio window until the audio pane exists
        )
        if not opened.get("ok"):
            return {"ok": False, "error": f"couldn't open the player window: "
                                          f"{opened.get('error', 'desktop unreachable')}"}
        _last_played = show
        return {
            "ok": True,
            "playing": episode.title,
            "show": show["title"],
            "duration_seconds": episode.duration_seconds,
            "window": opened.get("result"),
            "more_episodes": [e.title for e in episodes[1:4]],
        }
    except Exception as exc:
        logger.exception("[podcast_tools] play_podcast failed")
        return {"ok": False, "error": str(exc)}


async def subscribe_to_podcast_handler(input_data: dict, context: Optional[dict]) -> dict:
    show = _resolve_show(input_data.get("selection"))
    if not show:
        return {"ok": False, "error": "I don't know which show to subscribe to — "
                                      "search first or give me a feed URL"}
    try:
        from app.podcasts import subscriptions
        return await subscriptions.subscribe(show["feed_url"], title=show.get("title"))
    except Exception as exc:
        logger.exception("[podcast_tools] subscribe failed")
        return {"ok": False, "error": str(exc)}


async def unsubscribe_podcast_handler(input_data: dict, context: Optional[dict]) -> dict:
    ref = (input_data.get("ref") or "").strip()
    if not ref:
        return {"ok": False, "error": "ref (title or feed URL) is required"}
    try:
        from app.podcasts import subscriptions
        return subscriptions.unsubscribe(ref)
    except Exception as exc:
        logger.exception("[podcast_tools] unsubscribe failed")
        return {"ok": False, "error": str(exc)}


async def list_podcast_subscriptions_handler(input_data: dict, context: Optional[dict]) -> dict:
    try:
        from app.podcasts import subscriptions
        return {"ok": True, "subscriptions": subscriptions.list_subscriptions()}
    except Exception as exc:
        logger.exception("[podcast_tools] list subscriptions failed")
        return {"ok": False, "error": str(exc)}


async def check_new_episodes_handler(input_data: dict, context: Optional[dict]) -> dict:
    try:
        from app.podcasts import subscriptions
        return await subscriptions.new_episodes()
    except Exception as exc:
        logger.exception("[podcast_tools] check_new_episodes failed")
        return {"ok": False, "error": str(exc)}


# ────────────────────────────────────────────────────────────────────────
# Schemas (inline — self-contained file)
# ────────────────────────────────────────────────────────────────────────

_OK_ERR = {"ok": {"type": "boolean"}, "error": {"type": "string"}}

_SHOWS_OUTPUT = {
    "type": "object",
    "required": ["ok"],
    "properties": {
        **_OK_ERR,
        "shows": {"type": ["array", "null"], "items": {"type": "object"}},
        "note": {"type": ["string", "null"]},
    },
}

FIND_PODCASTS_INPUT = {
    "type": "object",
    "required": ["topic"],
    "properties": {
        "topic": {"type": "string", "description": "What the podcast should be about"},
        "max": {"type": "integer", "description": "Max results (default 5)"},
    },
}

TRENDING_INPUT = {
    "type": "object",
    "properties": {
        "category": {"type": ["string", "null"],
                     "description": "Optional PodcastIndex category, e.g. 'Technology'"},
        "max": {"type": "integer"},
    },
}

PLAY_PODCAST_INPUT = {
    "type": "object",
    "properties": {
        "selection": {"type": ["string", "integer", "null"],
                      "description": "1-based index into the last results ('the first "
                                     "one' = 1), a show title, or a feed URL. Omit for "
                                     "the top result of the last search."},
        "episode_index": {"type": "integer",
                          "description": "1 = latest episode (default), 2 = one before…"},
        "display": {"type": ["string", "null"],
                    "description": "Display alias for the player window (default main)"},
        "target": {"type": "string", "enum": ["desktop", "phone"],
                   "description": "Where to play (phone not wired yet)"},
    },
}
PLAY_PODCAST_OUTPUT = {
    "type": "object",
    "required": ["ok"],
    "properties": {
        **_OK_ERR,
        "playing": {"type": ["string", "null"]},
        "show": {"type": ["string", "null"]},
        "duration_seconds": {"type": ["integer", "null"]},
        "window": {"type": ["object", "null"]},
        "more_episodes": {"type": ["array", "null"], "items": {"type": "string"}},
    },
}

SUBSCRIBE_INPUT = {
    "type": "object",
    "properties": {
        "selection": {"type": ["string", "integer", "null"],
                      "description": "Same resolution as play_podcast: index, title, "
                                     "feed URL, or omit for the last show discussed"},
    },
}

UNSUBSCRIBE_INPUT = {
    "type": "object",
    "required": ["ref"],
    "properties": {"ref": {"type": "string", "description": "Show title or feed URL"}},
}

EMPTY_INPUT = {"type": "object", "properties": {}}

NEW_EPISODES_OUTPUT = {
    "type": "object",
    "required": ["ok"],
    "properties": {
        **_OK_ERR,
        "feeds": {"type": ["array", "null"], "items": {"type": "object"}},
        "checked": {"type": ["integer", "null"]},
        "failures": {"type": ["array", "null"], "items": {"type": "string"}},
    },
}

_GENERIC_OUTPUT = {
    "type": "object",
    "required": ["ok"],
    "properties": {
        **_OK_ERR,
        "title": {"type": ["string", "null"]},
        "already": {"type": ["boolean", "null"]},
        "subscriptions": {"type": ["array", "null"], "items": {"type": "object"}},
    },
}


# ────────────────────────────────────────────────────────────────────────
# Registration
# ────────────────────────────────────────────────────────────────────────

def register_podcast_tools() -> None:
    """Register the podcast tools with the global tool registry.
    Called once from registry._register_defaults at module load."""
    from app.tools.registry import ToolDefinition, register_tool

    register_tool(ToolDefinition(
        name="find_podcasts",
        version="v1",
        description=(
            "Search podcasts by topic ('find me a podcast about decentralised "
            "AI'). Returns up to 5 shows with index numbers. When speaking the "
            "results, give the top 3 as a short conversational sentence or two "
            "of prose — never a numbered list read-out. The user can then say "
            "'play the first one' or 'subscribe to that'."
        ),
        input_schema=FIND_PODCASTS_INPUT,
        output_schema=_SHOWS_OUTPUT,
        handler=find_podcasts_handler,
    ))

    register_tool(ToolDefinition(
        name="get_trending_podcasts",
        version="v1",
        description=(
            "What's trending on PodcastIndex right now, optionally by "
            "category. Same follow-up behaviour as find_podcasts."
        ),
        input_schema=TRENDING_INPUT,
        output_schema=_SHOWS_OUTPUT,
        handler=get_trending_podcasts_handler,
    ))

    register_tool(ToolDefinition(
        name="play_podcast",
        version="v1",
        description=(
            "Play a podcast episode on the desktop ('play that', 'play the "
            "first one', 'play the latest Lex Fridman'). selection resolves "
            "against the last search results, subscriptions, titles or a feed "
            "URL; episode_index picks older episodes. Opens the episode audio "
            "in a small window on the chosen display. Phone playback is not "
            "wired yet — say so honestly if asked."
        ),
        input_schema=PLAY_PODCAST_INPUT,
        output_schema=PLAY_PODCAST_OUTPUT,
        handler=play_podcast_handler,
    ))

    register_tool(ToolDefinition(
        name="subscribe_to_podcast",
        version="v1",
        description=(
            "Subscribe to a podcast feed ('subscribe to that'). selection "
            "resolves like play_podcast. New episodes then surface via "
            "check_new_episodes."
        ),
        input_schema=SUBSCRIBE_INPUT,
        output_schema=_GENERIC_OUTPUT,
        handler=subscribe_to_podcast_handler,
    ))

    register_tool(ToolDefinition(
        name="unsubscribe_podcast",
        version="v1",
        description="Remove a podcast subscription by title or feed URL.",
        input_schema=UNSUBSCRIBE_INPUT,
        output_schema=_GENERIC_OUTPUT,
        handler=unsubscribe_podcast_handler,
    ))

    register_tool(ToolDefinition(
        name="list_podcast_subscriptions",
        version="v1",
        description="List the podcasts currently subscribed to.",
        input_schema=EMPTY_INPUT,
        output_schema=_GENERIC_OUTPUT,
        handler=list_podcast_subscriptions_handler,
    ))

    register_tool(ToolDefinition(
        name="check_new_episodes",
        version="v1",
        description=(
            "Check every subscribed feed for episodes newer than the last "
            "check ('any new episodes?'). Summarise conversationally and "
            "briefly; descriptions are already TTS-truncated."
        ),
        input_schema=EMPTY_INPUT,
        output_schema=NEW_EPISODES_OUTPUT,
        handler=check_new_episodes_handler,
    ))
