# FILE: app/content/distribution/browser_analytics/recon.py
# Purpose: Reconnaissance — Phase 1 of the browser-analytics build.
# Called-by: app.content.distribution.browser_analytics.router
# Depends-on: app.content.distribution.browser_analytics.popup_dismiss, app.content.distribution.browser_analytics.urls, app.web_automation, app.web_automation.bridge (+1 more)
# Last-renovated: 2026-06-11
"""
Reconnaissance — Phase 1 of the browser-analytics build.

Drives a session to its insights URL, waits for content to settle,
dismisses common popups, and dumps everything that's there to a
timestamped file under D:\\Orb\\data\\browser_recon\\.

No parsing, no DB writes. The output files exist so we (Claude) can
read them and write parsers against the REAL text the platform shows,
not guessed selectors from memory.

Usage (after backend restart):

    POST /content/distribution/browser_analytics/recon/meta_business

    -> saves D:\\Orb\\data\\browser_recon\\meta_business-YYYYMMDD-HHMMSS.txt
    -> returns {"path": "...", "bytes": ..., "current_url": "...", "title": "..."}
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.web_automation import bridge, session_registry
from app.content.distribution.browser_analytics.urls import get_insights_url
from app.content.distribution.browser_analytics.popup_dismiss import (
    dismiss_common_popups,
)

logger = logging.getLogger(__name__)


RECON_DIR = Path("D:/Orb/data/browser_recon")


async def run_recon(db: Session, platform: str) -> dict[str, Any]:
    """
    Navigate the platform's session to its insights page, dismiss
    popups, extract text + current state, save to disk.

    Returns a summary dict with the file path and basic state info.
    """
    # 1. Find the session
    sess = session_registry.get_session_by_platform(db, platform)
    if sess is None:
        return {"ok": False, "error": f"no session registered for platform '{platform}'"}
    session_id = sess.id

    # 2. Get the insights URL
    url = get_insights_url(platform)
    if url is None:
        return {"ok": False, "error": f"no insights URL configured for '{platform}'"}

    logger.info(f"[recon] starting {platform} -> {url}")

    # 3. Ensure the view is open (no-op if already live)
    await bridge.ensure_session_open(session_id, timeout_seconds=20.0)

    # 4. Navigate to the insights URL
    nav_result = await bridge.execute_action(
        session_id, "navigate", {"url": url}, timeout_seconds=30.0,
    )
    if not nav_result.get("ok"):
        logger.warning(f"[recon] navigate failed: {nav_result}")
        # Continue anyway — some nav failures still leave the page loaded

    # 5. Wait for JS-rendered content to paint
    await bridge.execute_action(session_id, "wait", {"ms": 5000}, timeout_seconds=10.0)

    # 6. Try to dismiss any popups that appeared
    dismissed = await dismiss_common_popups(session_id)

    # 7. Another short wait in case popup-dismissal triggered re-renders
    await bridge.execute_action(session_id, "wait", {"ms": 2000}, timeout_seconds=10.0)

    # 8. Grab current state (URL + title) so we know where the nav actually ended up
    state_result = await bridge.execute_action(
        session_id, "current_state", {}, timeout_seconds=10.0,
    )
    current_url = ""
    title = ""
    if state_result.get("ok"):
        state = state_result.get("result") or {}
        current_url = state.get("url", "")
        title = state.get("title", "")

    # 9. Extract all visible text from the main content area.
    #    Fallback to 'body' if 'main' doesn't match — TikTok Studio
    #    and Meta Business don't always use <main>.
    body_text = await _extract_best_effort(session_id)

    # 10. Also grab the AX-tree DOM snapshot — gives us reliable
    #     structural info (roles, labels) that survives CSS changes.
    dom_result = await bridge.execute_action(
        session_id, "dom_snapshot", {}, timeout_seconds=15.0,
    )
    dom_tree = ""
    if dom_result.get("ok"):
        dom_tree = str(dom_result.get("result", ""))[:50000]  # cap at 50KB

    # 11. Write to disk
    RECON_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = RECON_DIR / f"{platform}-{stamp}.txt"

    report = _format_report(
        platform=platform,
        requested_url=url,
        current_url=current_url,
        title=title,
        dismissed=dismissed,
        body_text=body_text,
        dom_tree=dom_tree,
    )
    out_path.write_text(report, encoding="utf-8")
    size = out_path.stat().st_size

    logger.info(f"[recon] {platform} complete: {size} bytes -> {out_path}")

    return {
        "ok": True,
        "platform": platform,
        "path": str(out_path),
        "bytes": size,
        "current_url": current_url,
        "title": title,
        "popups_dismissed": dismissed,
    }


async def _extract_best_effort(session_id: str) -> str:
    """Try several selectors to get the insights text. Return the largest."""
    candidates: list[tuple[str, str]] = []  # (selector, text)

    for selector in ["main", "[role='main']", "body"]:
        try:
            r = await bridge.execute_action(
                session_id,
                "extract_text",
                {"selector": selector, "limit": 500},
                timeout_seconds=15.0,
            )
            if r.get("ok"):
                items = r.get("result", {}).get("items", []) or []
                joined = "\n".join((i.get("text") or "") for i in items)
                candidates.append((selector, joined))
        except Exception as e:
            logger.debug(f"[recon] extract_text({selector}) failed: {e}")

    if not candidates:
        return ""
    # Return the largest haul
    candidates.sort(key=lambda x: len(x[1]), reverse=True)
    selector, text = candidates[0]
    return f"[extracted via selector: {selector}]\n\n{text}"


def _format_report(
    *, platform: str, requested_url: str, current_url: str, title: str,
    dismissed: list[str], body_text: str, dom_tree: str,
) -> str:
    parts: list[str] = []
    parts.append(f"# RECON REPORT — {platform}")
    parts.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    parts.append(f"Requested URL: {requested_url}")
    parts.append(f"Actual URL after nav: {current_url}")
    parts.append(f"Page title: {title}")
    parts.append(f"Popups dismissed ({len(dismissed)}): {dismissed}")
    parts.append("")
    parts.append("=" * 80)
    parts.append("## VISIBLE TEXT")
    parts.append("=" * 80)
    parts.append(body_text or "(empty)")
    parts.append("")
    parts.append("=" * 80)
    parts.append("## DOM / ACCESSIBILITY TREE (first 50KB)")
    parts.append("=" * 80)
    parts.append(dom_tree or "(empty)")
    return "\n".join(parts)
