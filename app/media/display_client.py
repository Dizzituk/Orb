# FILE: app/media/display_client.py
# Purpose: Thin async client for the desktop display manager — open/close/move
#          windows on named displays ("the bed screen") from intent handlers.
# Called-by: app.tools.display_tools, app.media flows (stream_sites, movie_night, podcasts)
# Depends-on: app.web_automation.bridge, app.web_automation.session_registry
# Last-renovated: 2026-06-14
"""
Display-manager client.

All calls ride the existing web_automation action channel: the desktop's
poll loop for the "displays" control session executes displays_* actions
in the Electron main process. Results come back through the normal
action-result path, so every function here returns the executor's dict:
{"ok": True, "result": {...}} or {"ok": False, "error": "..."} — never raises.

A window opened with cdp=True and a session platform (e.g. "soundcloud")
is registered with Electron's web-automation session manager, after which
the whole existing executor set (navigate/click/type/press_key/screenshot)
drives that window via bridge.execute_by_platform(platform, ...).
"""
from __future__ import annotations

from typing import Optional

from app.web_automation import bridge

DISPLAYS_PLATFORM = "displays"


def _resolve_session_id(session: str) -> Optional[str]:
    """Platform key or session id -> session id (None when unknown)."""
    from app.db import SessionLocal
    from app.web_automation import session_registry

    db = SessionLocal()
    try:
        sess = session_registry.get_session(db, session) \
            or session_registry.get_session_by_platform(db, session)
        return sess.id if sess else None
    finally:
        db.close()


async def list_displays() -> dict:
    """All connected displays with bounds, primary flag and known aliases."""
    return await bridge.execute_by_platform(DISPLAYS_PLATFORM, "displays_list", {})


async def open_on_display(
    url: str,
    display: str = "main",
    *,
    fullscreen: bool = True,
    cdp: bool = False,
    session: Optional[str] = None,
    partition: Optional[str] = None,
    timeout_seconds: float = 20.0,
) -> dict:
    """
    Open a URL on a named display. session (a web_automation platform key
    like "soundcloud") + cdp=True makes the window automation-addressable.
    """
    payload: dict = {
        "url": url,
        "display": display,
        "fullscreen": fullscreen,
        "cdp": cdp,
    }
    if partition:
        payload["partition"] = partition
    if cdp and session:
        session_id = _resolve_session_id(session)
        if session_id:
            payload["session_id"] = session_id
    return await bridge.execute_by_platform(
        DISPLAYS_PLATFORM, "displays_open", payload, timeout_seconds=timeout_seconds
    )


async def close_window(window_id: Optional[int] = None, display: Optional[str] = None) -> dict:
    """Close a window by id, every media window on a display, or the latest."""
    payload: dict = {}
    if window_id is not None:
        payload["window_id"] = window_id
    if display:
        payload["display"] = display
    return await bridge.execute_by_platform(DISPLAYS_PLATFORM, "displays_close", payload)


async def move_window(window_id: int, display: str) -> dict:
    """Move an open media window to another named display."""
    return await bridge.execute_by_platform(
        DISPLAYS_PLATFORM, "displays_move", {"window_id": window_id, "display": display}
    )


async def list_open_windows() -> dict:
    """Currently open display-manager windows (stale-safe on the desktop side)."""
    return await bridge.execute_by_platform(DISPLAYS_PLATFORM, "displays_windows", {})


async def save_alias(
    alias: str,
    display_index: Optional[int] = None,
    *,
    display_id: Optional[int] = None,
) -> dict:
    """Persist a display alias bound to a PHYSICAL panel. Prefer display_id
    (stable across rearrange/reboot — from list_displays or calibrate); the
    legacy display_index is a 0-based left-to-right position (fragile)."""
    payload: dict = {"alias": alias}
    if display_id is not None:
        payload["display_id"] = display_id
    if display_index is not None:
        payload["display_index"] = display_index
    return await bridge.execute_by_platform(
        DISPLAYS_PLATFORM, "displays_save_alias", payload
    )


async def calibrate(duration_ms: Optional[int] = None) -> dict:
    """Flash a big labelled marker (A/B/C...) on every screen and return the
    marker->display map (marker, display id, orientation, position) so the
    caller can ask the user which letter is which named screen, then bind by
    display_id. Markers auto-close after duration_ms (desktop default 45s)."""
    payload: dict = {}
    if duration_ms is not None:
        payload["duration_ms"] = duration_ms
    return await bridge.execute_by_platform(
        DISPLAYS_PLATFORM, "displays_calibrate", payload
    )


async def close_calibration() -> dict:
    """Tear down the calibration markers (also auto-closed by a safety timer)."""
    return await bridge.execute_by_platform(
        DISPLAYS_PLATFORM, "displays_close_calibration", {}
    )
