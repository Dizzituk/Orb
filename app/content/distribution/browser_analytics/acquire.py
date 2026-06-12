# FILE: app/content/distribution/browser_analytics/acquire.py
# Purpose: Low-level snapshot acquisition for a single URL attempt.
# Called-by: app.content.distribution.browser_analytics.scrape
# Depends-on: app.content.distribution.browser_analytics.popup_dismiss, app.web_automation, app.web_automation.bridge
# Last-renovated: 2026-06-11
"""
Low-level snapshot acquisition for a single URL attempt.

Separated from scrape.py so the strategy loop stays readable, and so
this file can be tested with a mocked bridge independently.

One call =
    1. ensure session's view is open
    2. navigate to url
    3. initial paint wait
    4. optional: scroll down twice (trigger lazy-loaded charts), scroll
       back to top
    5. dismiss common popups, wait briefly
    6. grab current_state (landed URL + title)
    7. grab DOM accessibility snapshot
"""
from __future__ import annotations

import logging
from typing import Any, TypedDict

from app.web_automation import bridge

logger = logging.getLogger(__name__)

DEFAULT_WAIT_MS = 8_000
SCROLL_SETTLE_MS = 2_000
POPUP_DISMISS_SETTLE_MS = 2_500


class SnapshotResult(TypedDict):
    ok: bool
    snapshot: dict[str, Any]
    landed_url: str
    page_title: str
    dismissed_popups: list[str]
    error: str


async def acquire_snapshot(
    session_id: str,
    url: str,
    scroll: bool = False,
    wait_ms: int = DEFAULT_WAIT_MS,
) -> SnapshotResult:
    """
    Navigate + settle + (optionally scroll to force lazy-loads) + snapshot.

    Never raises — returns ok=False on any failure so the strategy loop
    can move to the next attempt.
    """
    # Local import to avoid cycles at module load.
    from app.content.distribution.browser_analytics.popup_dismiss import (
        dismiss_common_popups,
    )

    empty: SnapshotResult = {
        "ok": False,
        "snapshot": {},
        "landed_url": "",
        "page_title": "",
        "dismissed_popups": [],
        "error": "",
    }

    try:
        await bridge.ensure_session_open(session_id, timeout_seconds=20.0)

        await bridge.execute_action(
            session_id, "navigate", {"url": url}, timeout_seconds=30.0,
        )
        await bridge.execute_action(
            session_id, "wait", {"ms": wait_ms}, timeout_seconds=wait_ms / 1000 + 4,
        )

        # Optional scroll dance. Two scrolls down to get past any
        # above-the-fold content and trigger IntersectionObserver-based
        # lazy-loads, then back to top so the "hero" metrics (which
        # usually live at the top) are in-view when we snapshot.
        if scroll:
            for _ in range(2):
                await bridge.execute_action(
                    session_id, "scroll",
                    {"direction": "down", "amount": 800},
                    timeout_seconds=6.0,
                )
                await bridge.execute_action(
                    session_id, "wait", {"ms": SCROLL_SETTLE_MS},
                    timeout_seconds=5.0,
                )
            await bridge.execute_action(
                session_id, "scroll", {"direction": "top", "amount": 0},
                timeout_seconds=6.0,
            )
            await bridge.execute_action(
                session_id, "wait", {"ms": SCROLL_SETTLE_MS},
                timeout_seconds=5.0,
            )

        # Popups can appear at any time during the scroll dance above.
        dismissed = await dismiss_common_popups(session_id)
        if dismissed:
            await bridge.execute_action(
                session_id, "wait", {"ms": POPUP_DISMISS_SETTLE_MS},
                timeout_seconds=6.0,
            )

        state = await bridge.execute_action(
            session_id, "current_state", {}, timeout_seconds=10.0,
        )
        landed_url = ""
        page_title = ""
        if state.get("ok"):
            r = state.get("result", {}) or {}
            landed_url = r.get("url", "")
            page_title = r.get("title", "")

        snap_result = await bridge.execute_action(
            session_id, "dom_snapshot", {}, timeout_seconds=20.0,
        )
        if not snap_result.get("ok"):
            return {
                **empty,
                "landed_url": landed_url,
                "page_title": page_title,
                "dismissed_popups": dismissed,
                "error": f"dom_snapshot failed: {snap_result.get('error')}",
            }

        return {
            "ok": True,
            "snapshot": snap_result.get("result", {}) or {},
            "landed_url": landed_url,
            "page_title": page_title,
            "dismissed_popups": dismissed,
            "error": "",
        }

    except Exception as e:
        logger.exception(f"[acquire] failed for {url}")
        return {**empty, "error": f"{type(e).__name__}: {e}"}
