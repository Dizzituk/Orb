# FILE: app/media/stream_sites.py
# Purpose: SoundCloud / Mixcloud desktop playback — open the logged-in site on
#          a named display and drive play/pause/skip through the automation
#          executors (browser control, zero platform APIs).
# Called-by: app.tools.stream_tools
# Depends-on: app.media.display_client, app.web_automation.bridge
# Last-renovated: 2026-06-12
"""
Stream-site flows.

The value is the logged-in personalised feed — that lives in the account
session (persist:media partition), not any public API. Windows open via the
display manager registered under their platform session ("soundcloud" /
"mixcloud" / "youtube_watch"), after which the existing executor set drives
them: tolerant cookie dismiss, then a defensive play ladder (player-bar
selector -> first feed item -> Space key). The click executor self-verifies
(element state before/after), so results carry real evidence.

ALL selectors live in SELECTORS below — breakage is a one-line fix.
First run per site needs one manual login on that window; the persistent
partition remembers it forever after.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from app.media import display_client
from app.web_automation import bridge

logger = logging.getLogger(__name__)

# ── THE constants block: every selector/URL/key in one place ──────────────
SELECTORS = {
    "soundcloud": {
        "url": "https://soundcloud.com/feed",
        "cookie_accept": ["#onetrust-accept-btn-handler"],
        "play_toggle": [".playControl", ".playControls__play"],
        "first_item_play": [
            ".soundList__item .sc-button-play",
            ".sound__soundActions .sc-button-play",
            ".playButton",
        ],
        "next": [".skipControl__next", ".playControls__next"],
        "previous": [".skipControl__previous", ".playControls__previous"],
        "toggle_key": "Space",
        "login_hint": "the SoundCloud window is open, but I couldn't reach the "
                      "player — log in once on that screen and it'll stick.",
    },
    "mixcloud": {
        "url": "https://www.mixcloud.com/",
        "cookie_accept": ["#onetrust-accept-btn-handler",
                          "button[data-testid='cookie-accept']"],
        "play_toggle": ["[data-testid='player-play-button']",
                        "button[aria-label='Play']",
                        "button[aria-label='Pause']"],
        "first_item_play": ["[data-testid='play-button']",
                            "button[aria-label^='Play']"],
        "next": ["[data-testid='player-next-button']",
                 "button[aria-label='Next']"],
        "previous": ["[data-testid='player-previous-button']",
                     "button[aria-label='Previous']"],
        "toggle_key": "Space",
        "login_hint": "the Mixcloud window is open, but I couldn't reach the "
                      "player — log in once on that screen and it'll stick.",
    },
    "youtube_watch": {
        "url": "https://www.youtube.com/",
        "cookie_accept": ["button[aria-label*='Accept all']",
                          "button[aria-label*='Accept the use']"],
        "play_toggle": [".ytp-play-button"],
        "first_item_play": [],
        "next": [".ytp-next-button"],
        "previous": [],
        "toggle_key": "k",
        "login_hint": "the YouTube window is open but the player didn't respond.",
    },
}

PAGE_SETTLE_MS = 3500

# What was opened last, so bare "pause" / "skip" routes correctly.
_active: dict = {"platform": None, "window_id": None}


def active_platform() -> Optional[str]:
    return _active["platform"]


async def _act(platform: str, action: str, payload: dict,
               timeout: float = 20.0) -> dict:
    return await bridge.execute_by_platform(platform, action, payload,
                                            timeout_seconds=timeout)


async def _click_first(platform: str, selectors: List[str]) -> dict:
    """Try selectors in order; first ok click wins. Returns the last result."""
    last: dict = {"ok": False, "error": "no selectors defined"}
    for sel in selectors:
        last = await _act(platform, "click", {"selector": sel})
        if last.get("ok"):
            last["selector_used"] = sel
            return last
    return last


async def _dismiss_cookies(platform: str) -> None:
    """Tolerant try-pattern — consent banners exist on first load only."""
    for sel in SELECTORS[platform]["cookie_accept"]:
        try:
            result = await _act(platform, "click", {"selector": sel}, timeout=8.0)
            if result.get("ok"):
                logger.info("[stream_sites] dismissed consent on %s via %s", platform, sel)
                return
        except Exception:
            pass


async def open_and_play(platform: str, display: str = "main") -> dict:
    """Open the site fullscreen on a display and walk the play ladder."""
    spec = SELECTORS[platform]
    opened = await display_client.open_on_display(
        spec["url"], display, cdp=True, session=platform,
    )
    if not opened.get("ok"):
        return {"ok": False, "error": f"couldn't open the window: "
                                      f"{opened.get('error', 'desktop unreachable')}"}
    _active.update(platform=platform,
                   window_id=(opened.get("result") or {}).get("window_id"))

    await _act(platform, "wait", {"ms": PAGE_SETTLE_MS})
    await _dismiss_cookies(platform)

    # Play ladder: global player control -> first feed item -> keyboard.
    ladder: List[str] = []
    clicked = await _click_first(platform, spec["play_toggle"])
    ladder.append(f"play_toggle:{'ok' if clicked.get('ok') else 'miss'}")
    if not clicked.get("ok") and spec["first_item_play"]:
        clicked = await _click_first(platform, spec["first_item_play"])
        ladder.append(f"first_item:{'ok' if clicked.get('ok') else 'miss'}")
    if not clicked.get("ok"):
        keyed = await _act(platform, "press_key", {"key": spec["toggle_key"]})
        ladder.append(f"key:{'ok' if keyed.get('ok') else 'miss'}")
        if not keyed.get("ok"):
            return {"ok": False, "window": opened.get("result"),
                    "ladder": ladder, "error": spec["login_hint"]}
    return {"ok": True, "platform": platform, "window": opened.get("result"),
            "ladder": ladder, "note": (opened.get("result") or {}).get("note")}


async def play_soundcloud(display: str = "main") -> dict:
    return await open_and_play("soundcloud", display)


async def play_mixcloud(display: str = "main") -> dict:
    return await open_and_play("mixcloud", display)


async def toggle_playback(platform: Optional[str] = None) -> dict:
    """Pause/resume the active stream window (player toggle, key fallback)."""
    platform = platform or _active["platform"]
    if not platform:
        return {"ok": False, "error": "nothing has been opened to pause"}
    spec = SELECTORS[platform]
    clicked = await _click_first(platform, spec["play_toggle"])
    if clicked.get("ok"):
        return {"ok": True, "platform": platform, "via": clicked.get("selector_used")}
    keyed = await _act(platform, "press_key", {"key": spec["toggle_key"]})
    if keyed.get("ok"):
        return {"ok": True, "platform": platform, "via": f"key:{spec['toggle_key']}"}
    return {"ok": False, "platform": platform, "error": spec["login_hint"]}


async def skip(direction: str = "next", platform: Optional[str] = None) -> dict:
    """Next/previous track on the active stream window."""
    platform = platform or _active["platform"]
    if not platform:
        return {"ok": False, "error": "nothing has been opened to skip"}
    spec = SELECTORS[platform]
    selectors = spec["next"] if direction != "previous" else spec["previous"]
    if not selectors:
        return {"ok": False, "error": f"{platform} has no {direction} control"}
    clicked = await _click_first(platform, selectors)
    if clicked.get("ok"):
        return {"ok": True, "platform": platform, "via": clicked.get("selector_used")}
    return {"ok": False, "platform": platform,
            "error": f"couldn't find the {direction} control — {spec['login_hint']}"}


def note_video_opened(window_id=None) -> None:
    """youtube_search tells us a watch window opened so pause/skip route to it."""
    _active.update(platform="youtube_watch", window_id=window_id)
