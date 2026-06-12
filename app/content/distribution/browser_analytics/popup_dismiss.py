# FILE: app/content/distribution/browser_analytics/popup_dismiss.py
# Purpose: Best-effort popup dismissal. Each platform puts a cookie banner,
# Called-by: app.content.distribution.browser_analytics.acquire, app.content.distribution.browser_analytics.recon
# Depends-on: app.web_automation, app.web_automation.bridge
# Last-renovated: 2026-06-11
"""
Best-effort popup dismissal. Each platform puts a cookie banner,
"Customize your experience" dialog, or tutorial overlay on first load.
We try a list of common close patterns and move on.

This is fire-and-forget: if none of the selectors match, the scrape
continues anyway. Most of these dialogs have semi-transparent backdrops
that don't actually block selector-based interaction — but they DO
block visual text extraction and can intercept clicks, so we try.
"""
from __future__ import annotations

import logging
from typing import List

from app.web_automation import bridge

logger = logging.getLogger(__name__)


# Ordered by specificity — most-targeted first, broadest last.
# Each platform has its own quirks; we try them all regardless of
# which platform we're on, since they're cheap no-ops if absent.
_COMMON_DISMISS_SELECTORS: List[str] = [
    # Generic close buttons with explicit aria labels
    'button[aria-label="Close"]',
    'button[aria-label="Dismiss"]',
    'button[aria-label="Not now"]',
    'button[aria-label="No thanks"]',
    'button[aria-label="Got it"]',

    # TikTok-specific overlays
    'button[data-e2e="modal-close-inner-button"]',
    'div[data-e2e="notification-close-icon"]',

    # Meta/Facebook-specific
    'div[role="button"][aria-label="Close"]',
    'div[aria-label="Close"][role="button"]',

    # YouTube-specific
    'button[aria-label="Dismiss"]',
    'tp-yt-paper-button[aria-label="Got it"]',
    'ytcp-button[label="Got it"]',
    'button.ytcp-dismiss-dialog-button',

    # Cookie banners
    'button[data-cookiebanner="accept_button"]',
    'button[data-testid="cookie-policy-manage-dialog-accept-button"]',
]


async def dismiss_common_popups(session_id: str, *, max_attempts: int = 3) -> List[str]:
    """
    Try each common dismissal selector. Returns the list of selectors
    that actually succeeded. Non-fatal if nothing matches.

    max_attempts: sometimes dismissing one popup reveals another. We
    loop a few times to let secondary dialogs surface.
    """
    successes: List[str] = []

    for attempt in range(max_attempts):
        attempt_hits = 0
        for sel in _COMMON_DISMISS_SELECTORS:
            try:
                result = await bridge.execute_action(
                    session_id,
                    "click",
                    {"selector": sel},
                    timeout_seconds=3.0,
                )
                if result.get("ok") and result.get("result", {}).get("clicked"):
                    successes.append(sel)
                    attempt_hits += 1
                    logger.debug(f"[popup_dismiss] clicked {sel}")
            except Exception:
                # A click failure here is normal — most selectors won't match.
                pass

        if attempt_hits == 0:
            break  # Nothing got clicked this pass, no point trying again

    return successes
