# FILE: app/tools/display_tools.py
# Purpose: Display chat tools — "open YouTube on the bed screen", "close that",
#          list screens, move windows, save spoken display aliases.
# Called-by: app.tools.registry (_register_defaults)
# Depends-on: app.media.display_client
# Last-renovated: 2026-06-14
"""
Display-manager chat tools (v1, 2026-06-12).

Six tools that let any ASTRA chat put things on named screens:

  list_displays          what screens exist + their aliases
  open_on_display        URL -> fullscreen window on a named display
  close_display_window   "close that" / "close the bed screen"
  move_display_window    move an open window to another screen
  list_display_windows   what's open where
  save_display_alias     persist a new spoken name for a screen

Handler style mirrors vehicle_tools: thin, errors return {ok: false,
error} so the model can react. An UNRESOLVED_ALIAS error carries
known_aliases — the model should ask which screen was meant, then offer
to remember it via save_display_alias. Schemas inline; self-contained.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
# Handlers
# ────────────────────────────────────────────────────────────────────────

async def list_displays_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Connected displays with bounds, primary flag and aliases."""
    try:
        from app.media import display_client
        return await display_client.list_displays()
    except Exception as exc:
        logger.exception("[display_tools] list_displays failed")
        return {"ok": False, "error": str(exc)}


async def open_on_display_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Open a URL fullscreen on a named display."""
    url = input_data.get("url")
    if not url:
        return {"ok": False, "error": "url is required"}
    try:
        from app.media import display_client
        return await display_client.open_on_display(
            url,
            display=input_data.get("display") or "main",
            fullscreen=input_data.get("fullscreen", True),
            cdp=bool(input_data.get("cdp", False)),
            session=input_data.get("session"),
        )
    except Exception as exc:
        logger.exception("[display_tools] open_on_display failed")
        return {"ok": False, "error": str(exc)}


async def close_display_window_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Close by window id, by display name, or the most recent window."""
    try:
        from app.media import display_client
        return await display_client.close_window(
            window_id=input_data.get("window_id"),
            display=input_data.get("display"),
        )
    except Exception as exc:
        logger.exception("[display_tools] close_display_window failed")
        return {"ok": False, "error": str(exc)}


async def move_display_window_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Move an open window to another named display."""
    window_id = input_data.get("window_id")
    display = input_data.get("display")
    if window_id is None or not display:
        return {"ok": False, "error": "window_id and display are required"}
    try:
        from app.media import display_client
        return await display_client.move_window(window_id, display)
    except Exception as exc:
        logger.exception("[display_tools] move_display_window failed")
        return {"ok": False, "error": str(exc)}


async def list_display_windows_handler(input_data: dict, context: Optional[dict]) -> dict:
    """What display-manager windows are open, and where."""
    try:
        from app.media import display_client
        return await display_client.list_open_windows()
    except Exception as exc:
        logger.exception("[display_tools] list_display_windows failed")
        return {"ok": False, "error": str(exc)}


async def save_display_alias_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Persist a spoken alias for a display (after the user confirms)."""
    alias = input_data.get("alias")
    display_id = input_data.get("display_id")
    display_index = input_data.get("display_index")
    if not alias or (display_id is None and display_index is None):
        return {"ok": False,
                "error": "alias and (display_id or display_index) are required"}
    try:
        from app.media import display_client
        return await display_client.save_alias(
            alias,
            None if display_index is None else int(display_index),
            display_id=None if display_id is None else int(display_id),
        )
    except Exception as exc:
        logger.exception("[display_tools] save_display_alias failed")
        return {"ok": False, "error": str(exc)}


async def calibrate_displays_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Flash a labelled marker on every screen; returns the marker->display map."""
    try:
        from app.media import display_client
        return await display_client.calibrate(duration_ms=input_data.get("duration_ms"))
    except Exception as exc:
        logger.exception("[display_tools] calibrate_displays failed")
        return {"ok": False, "error": str(exc)}


async def end_display_calibration_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Close the calibration markers once each screen has been identified."""
    try:
        from app.media import display_client
        return await display_client.close_calibration()
    except Exception as exc:
        logger.exception("[display_tools] end_display_calibration failed")
        return {"ok": False, "error": str(exc)}


# ────────────────────────────────────────────────────────────────────────
# Schemas (inline — self-contained file)
# ────────────────────────────────────────────────────────────────────────

_OK_ERR = {
    "ok": {"type": "boolean"},
    # Nullable: successful results carry error=None, which tripped
    # "output schema mismatch: error.expected string, got NoneType" warnings
    # for every display tool. Allow null as well as a string message.
    "error": {"type": ["string", "null"]},
}

_RESULT_OUTPUT = {
    "type": "object",
    "required": ["ok"],
    "properties": {
        **_OK_ERR,
        "result": {"type": ["object", "null"]},
        "error_code": {"type": ["string", "null"]},
        "known_aliases": {"type": ["array", "null"], "items": {"type": "string"}},
    },
}

LIST_DISPLAYS_INPUT = {"type": "object", "properties": {}}

OPEN_ON_DISPLAY_INPUT = {
    "type": "object",
    "required": ["url"],
    "properties": {
        "url": {"type": "string", "description": "Full URL to open"},
        "display": {"type": "string",
                    "description": "Display alias ('main', 'bed screen', 'screen 2'). "
                                   "Default 'main'."},
        "fullscreen": {"type": "boolean", "description": "Default true"},
        "cdp": {"type": "boolean",
                "description": "Register the window for browser automation "
                               "(needed when ASTRA must click/type in it)"},
        "session": {"type": ["string", "null"],
                    "description": "web_automation platform key to register under "
                                   "(e.g. 'soundcloud', 'streaming') when cdp=true"},
    },
}

CLOSE_DISPLAY_WINDOW_INPUT = {
    "type": "object",
    "properties": {
        "window_id": {"type": ["integer", "null"],
                      "description": "Specific window id from a previous open/list"},
        "display": {"type": ["string", "null"],
                    "description": "Close every media window on this display"},
    },
}

MOVE_DISPLAY_WINDOW_INPUT = {
    "type": "object",
    "required": ["window_id", "display"],
    "properties": {
        "window_id": {"type": "integer"},
        "display": {"type": "string", "description": "Target display alias"},
    },
}

LIST_DISPLAY_WINDOWS_INPUT = {"type": "object", "properties": {}}

SAVE_DISPLAY_ALIAS_INPUT = {
    "type": "object",
    "required": ["alias"],
    "properties": {
        "alias": {"type": "string", "description": "Spoken name, e.g. 'bed screen'"},
        "display_id": {"type": ["integer", "null"],
                       "description": "PREFERRED. Stable id of the physical panel "
                                      "(from list_displays or calibrate_displays). "
                                      "Survives rearranging/unplugging monitors."},
        "display_index": {"type": ["integer", "null"],
                          "description": "Legacy fallback: 0-based left-to-right "
                                         "index. Fragile (shifts if monitors move) — "
                                         "prefer display_id when you have it."},
    },
}

CALIBRATE_DISPLAYS_INPUT = {
    "type": "object",
    "properties": {
        "duration_ms": {"type": ["integer", "null"],
                        "description": "Auto-close the markers after this many ms "
                                       "(default 45000)."},
    },
}

END_DISPLAY_CALIBRATION_INPUT = {"type": "object", "properties": {}}


# ────────────────────────────────────────────────────────────────────────
# Registration
# ────────────────────────────────────────────────────────────────────────

def register_display_tools() -> None:
    """Register the display tools with the global tool registry.
    Called once from registry._register_defaults at module load."""
    from app.tools.registry import ToolDefinition, register_tool

    register_tool(ToolDefinition(
        name="list_displays",
        version="v1",
        description=(
            "List the PC's connected displays. Each has a STABLE id (pass it "
            "to save_display_alias), an orientation (portrait/landscape), a "
            "rough position (far-left / centre / upper far-right), the primary "
            "flag, and the spoken aliases pointing at it ('main', 'bed "
            "screen'). Use before opening something on a screen the user "
            "hasn't named before, or when asked what screens exist."
        ),
        input_schema=LIST_DISPLAYS_INPUT,
        output_schema=_RESULT_OUTPUT,
        handler=list_displays_handler,
    ))

    register_tool(ToolDefinition(
        name="open_on_display",
        version="v1",
        description=(
            "Open a URL as a fullscreen window on a named display — 'put "
            "YouTube on the bed screen'. display accepts saved aliases, "
            "'screen 2' style names, or 'main'. If the result has "
            "error_code UNRESOLVED_ALIAS, ask the user which screen they "
            "meant (known_aliases lists what exists), then offer to "
            "remember the name with save_display_alias. With one screen "
            "connected it opens there and the result carries a note to "
            "speak. Set cdp=true with a session platform key only when "
            "ASTRA must drive the page afterwards."
        ),
        input_schema=OPEN_ON_DISPLAY_INPUT,
        output_schema=_RESULT_OUTPUT,
        handler=open_on_display_handler,
    ))

    register_tool(ToolDefinition(
        name="close_display_window",
        version="v1",
        description=(
            "Close media windows opened on displays. 'Close that' = no "
            "arguments (closes the most recent). 'Close the bed screen' = "
            "display='bed screen'. A specific window_id closes exactly one."
        ),
        input_schema=CLOSE_DISPLAY_WINDOW_INPUT,
        output_schema=_RESULT_OUTPUT,
        handler=close_display_window_handler,
    ))

    register_tool(ToolDefinition(
        name="move_display_window",
        version="v1",
        description=(
            "Move an open media window to another named display ('put it "
            "on the main screen instead'). window_id comes from "
            "list_display_windows or the original open result."
        ),
        input_schema=MOVE_DISPLAY_WINDOW_INPUT,
        output_schema=_RESULT_OUTPUT,
        handler=move_display_window_handler,
    ))

    register_tool(ToolDefinition(
        name="list_display_windows",
        version="v1",
        description=(
            "List the media windows currently open on displays (url, title, "
            "which screen, window_id). Use to answer 'what's on the bed "
            "screen' or to find a window_id to close/move."
        ),
        input_schema=LIST_DISPLAY_WINDOWS_INPUT,
        output_schema=_RESULT_OUTPUT,
        handler=list_display_windows_handler,
    ))

    register_tool(ToolDefinition(
        name="save_display_alias",
        version="v1",
        description=(
            "Remember a spoken name for a display ('call it the bed "
            "screen'), persisted to the desktop's displays.json. Only call "
            "after the user has confirmed which physical screen they mean. "
            "PREFER display_id (the stable panel id from list_displays or "
            "calibrate_displays) so the name survives rearranging monitors; "
            "display_index (0-based left-to-right) is a fragile fallback."
        ),
        input_schema=SAVE_DISPLAY_ALIAS_INPUT,
        output_schema=_RESULT_OUTPUT,
        handler=save_display_alias_handler,
    ))

    register_tool(ToolDefinition(
        name="calibrate_displays",
        version="v1",
        description=(
            "Flash a big letter (A, B, C...) on every screen so the user can "
            "say which physical monitor is which. Returns a marker->display "
            "map (each marker's display_id, orientation, position, primary "
            "flag). Use when a screen name opened on the wrong monitor, or to "
            "(re)bind names after moving monitors: call this, ask the user "
            "'which letter is your <name>?', then save_display_alias with that "
            "marker's display_id. Call end_display_calibration when finished."
        ),
        input_schema=CALIBRATE_DISPLAYS_INPUT,
        output_schema=_RESULT_OUTPUT,
        handler=calibrate_displays_handler,
    ))

    register_tool(ToolDefinition(
        name="end_display_calibration",
        version="v1",
        description=(
            "Close the calibration markers opened by calibrate_displays, once "
            "every named screen has been bound. They also auto-close on a timer."
        ),
        input_schema=END_DISPLAY_CALIBRATION_INPUT,
        output_schema=_RESULT_OUTPUT,
        handler=end_display_calibration_handler,
    ))
