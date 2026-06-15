# FILE: app/tools/stream_tools.py
# Purpose: Desktop media chat tools — "play my SoundCloud on screen two",
#          pause/skip the open window, YouTube voice search + play.
# Called-by: app.tools.registry (_register_defaults)
# Depends-on: app.media.stream_sites, app.media.youtube_search
# Last-renovated: 2026-06-12
"""
Stream/media chat tools (v1, 2026-06-12).

  play_soundcloud_desktop  "play my SoundCloud (on the bed screen)"
  play_mixcloud_desktop    same for Mixcloud
  toggle_desktop_playback  "pause" / "resume" the open media window
  skip_desktop_media       "skip" / "go back" on the open media window
  find_videos              "find me a video about agrivoltaics" (Data API)
  play_video               "play the second one" -> watch page on a display

These drive the LOGGED-IN browser windows (persist:media partition) via
the display manager + automation executors — the van's phone ritual has
MediaSessionController on the Bridge; this is the desktop equivalent.
First run per site needs one manual login on that window, then never again.

Handler style mirrors vehicle_tools: {ok: false, error} on failure;
schemas inline; self-contained.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
# Handlers
# ────────────────────────────────────────────────────────────────────────

async def play_soundcloud_handler(input_data: dict, context: Optional[dict]) -> dict:
    try:
        from app.media import stream_sites
        return await stream_sites.play_soundcloud(input_data.get("display") or "main")
    except Exception as exc:
        logger.exception("[stream_tools] play_soundcloud failed")
        return {"ok": False, "error": str(exc)}


async def play_mixcloud_handler(input_data: dict, context: Optional[dict]) -> dict:
    try:
        from app.media import stream_sites
        return await stream_sites.play_mixcloud(input_data.get("display") or "main")
    except Exception as exc:
        logger.exception("[stream_tools] play_mixcloud failed")
        return {"ok": False, "error": str(exc)}


async def toggle_playback_handler(input_data: dict, context: Optional[dict]) -> dict:
    try:
        from app.media import stream_sites
        return await stream_sites.toggle_playback(input_data.get("platform"))
    except Exception as exc:
        logger.exception("[stream_tools] toggle_playback failed")
        return {"ok": False, "error": str(exc)}


async def skip_media_handler(input_data: dict, context: Optional[dict]) -> dict:
    try:
        from app.media import stream_sites
        return await stream_sites.skip(
            input_data.get("direction") or "next", input_data.get("platform")
        )
    except Exception as exc:
        logger.exception("[stream_tools] skip failed")
        return {"ok": False, "error": str(exc)}


async def find_videos_handler(input_data: dict, context: Optional[dict]) -> dict:
    query = (input_data.get("query") or "").strip()
    if not query:
        return {"ok": False, "error": "query is required"}
    try:
        from app.media import youtube_search
        videos = await youtube_search.search(query, int(input_data.get("max", 5)))
        return {"ok": True, "videos": videos}
    except Exception as exc:
        logger.exception("[stream_tools] find_videos failed")
        return {"ok": False, "error": str(exc)}


async def play_video_handler(input_data: dict, context: Optional[dict]) -> dict:
    try:
        from app.media import youtube_search
        return await youtube_search.play_video(
            input_data.get("selection"), input_data.get("display") or "main"
        )
    except Exception as exc:
        logger.exception("[stream_tools] play_video failed")
        return {"ok": False, "error": str(exc)}


# ────────────────────────────────────────────────────────────────────────
# Schemas (inline — self-contained file)
# ────────────────────────────────────────────────────────────────────────

_OK_ERR = {"ok": {"type": "boolean"}, "error": {"type": "string"}}

_PLAY_SITE_INPUT = {
    "type": "object",
    "properties": {
        "display": {"type": ["string", "null"],
                    "description": "Display alias ('main', 'bed screen', 'screen 2'). "
                                   "Default main."},
    },
}
_PLAY_SITE_OUTPUT = {
    "type": "object",
    "required": ["ok"],
    "properties": {
        **_OK_ERR,
        "platform": {"type": ["string", "null"]},
        "window": {"type": ["object", "null"]},
        "ladder": {"type": ["array", "null"], "items": {"type": "string"}},
        "note": {"type": ["string", "null"]},
    },
}

_TOGGLE_INPUT = {
    "type": "object",
    "properties": {
        "platform": {"type": ["string", "null"],
                     "enum": ["soundcloud", "mixcloud", "youtube_watch", None],
                     "description": "Defaults to whatever was opened last"},
    },
}

_SKIP_INPUT = {
    "type": "object",
    "properties": {
        "direction": {"type": "string", "enum": ["next", "previous"],
                      "description": "Default next"},
        "platform": {"type": ["string", "null"],
                     "description": "Defaults to whatever was opened last"},
    },
}

_VIA_OUTPUT = {
    "type": "object",
    "required": ["ok"],
    "properties": {
        **_OK_ERR,
        "platform": {"type": ["string", "null"]},
        "via": {"type": ["string", "null"]},
    },
}

FIND_VIDEOS_INPUT = {
    "type": "object",
    "required": ["query"],
    "properties": {
        "query": {"type": "string"},
        "max": {"type": "integer", "description": "Default 5"},
    },
}
FIND_VIDEOS_OUTPUT = {
    "type": "object",
    "required": ["ok"],
    "properties": {
        **_OK_ERR,
        "videos": {"type": ["array", "null"], "items": {"type": "object"}},
    },
}

PLAY_VIDEO_INPUT = {
    "type": "object",
    "properties": {
        "selection": {"type": ["string", "integer", "null"],
                      "description": "1-based index into the last find_videos "
                                     "results, a video id, a YouTube URL, or a "
                                     "title fragment. Omit for the top result."},
        "display": {"type": ["string", "null"],
                    "description": "Display alias (default main)"},
    },
}
PLAY_VIDEO_OUTPUT = {
    "type": "object",
    "required": ["ok"],
    "properties": {
        **_OK_ERR,
        "playing": {"type": ["string", "null"]},
        "video_id": {"type": ["string", "null"]},
        "window": {"type": ["object", "null"]},
        "note": {"type": ["string", "null"]},
    },
}


# ────────────────────────────────────────────────────────────────────────
# Registration
# ────────────────────────────────────────────────────────────────────────

def register_stream_tools() -> None:
    """Register the stream/media tools with the global tool registry.
    Called once from registry._register_defaults at module load."""
    from app.tools.registry import ToolDefinition, register_tool

    register_tool(ToolDefinition(
        name="play_soundcloud_desktop",
        version="v1",
        description=(
            "Open the logged-in SoundCloud feed fullscreen on a named display "
            "and press play ('play my SoundCloud on screen two'). The result's "
            "ladder shows what worked; an error mentioning login means the "
            "user must log in once on that window — say so. Speak the note "
            "field if present (single-screen fallback)."
        ),
        input_schema=_PLAY_SITE_INPUT,
        output_schema=_PLAY_SITE_OUTPUT,
        handler=play_soundcloud_handler,
    ))

    register_tool(ToolDefinition(
        name="play_mixcloud_desktop",
        version="v1",
        description="Same as play_soundcloud_desktop but for Mixcloud.",
        input_schema=_PLAY_SITE_INPUT,
        output_schema=_PLAY_SITE_OUTPUT,
        handler=play_mixcloud_handler,
    ))

    register_tool(ToolDefinition(
        name="toggle_desktop_playback",
        version="v1",
        description=(
            "Pause OR resume the open desktop media window (SoundCloud/"
            "Mixcloud/YouTube) — it presses the site's play/pause control, "
            "which toggles. Use for 'pause', 'pause the music', 'resume', "
            "'play' when desktop media is the context (phone media has its "
            "own on-device control)."
        ),
        input_schema=_TOGGLE_INPUT,
        output_schema=_VIA_OUTPUT,
        handler=toggle_playback_handler,
    ))

    register_tool(ToolDefinition(
        name="skip_desktop_media",
        version="v1",
        description=(
            "Skip to the next (or previous) track/video in the open desktop "
            "media window. Use for 'skip', 'next', 'go back a track' when "
            "desktop media is playing."
        ),
        input_schema=_SKIP_INPUT,
        output_schema=_VIA_OUTPUT,
        handler=skip_media_handler,
    ))

    register_tool(ToolDefinition(
        name="find_videos",
        version="v1",
        description=(
            "YouTube search ('find me a video about agrivoltaics'). Returns "
            "up to 5 with titles, channels and durations — speak the top 3 "
            "as short prose with rough lengths ('twelve minutes'), then the "
            "user can say 'play the second one'. Errors about YouTube not "
            "connected mean the Content Hub YouTube auth needs running once."
        ),
        input_schema=FIND_VIDEOS_INPUT,
        output_schema=FIND_VIDEOS_OUTPUT,
        handler=find_videos_handler,
    ))

    register_tool(ToolDefinition(
        name="play_video",
        version="v1",
        description=(
            "Open a YouTube video fullscreen on a named display and let it "
            "autoplay ('play the second one on the bed screen'). selection "
            "resolves against the last find_videos results, ids, URLs or "
            "title fragments."
        ),
        input_schema=PLAY_VIDEO_INPUT,
        output_schema=PLAY_VIDEO_OUTPUT,
        handler=play_video_handler,
    ))
