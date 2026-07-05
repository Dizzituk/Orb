# FILE: app/content/distribution/posting_drivers/coursera_recon.py
# Purpose: Non-destructive recon of the Coursera learner surfaces (Amendment A Job 10).
# Called-by: manual/supervised run + tests.test_coursera_driver (import only)
# Depends-on: .driver_runner, .coursera_driver, app.web_automation.coursera_reader, app.web_automation.bridge
# Last-renovated: 2026-07-02
"""
Coursera recon — capture accessibility tree + screenshot for each
study-loop surface so selector_maps/coursera.json can be confirmed
against the real DOM: My Learning dashboard, course home, lesson page
(video, with the transcript panel open), a reading item, and a quiz
LANDING page.

SAFETY: non-destructive. It never starts, answers or submits a quiz —
the quiz state is captured from its landing page only, then the view
navigates back to the lesson it came from. Requires the desktop app up
and Coursera logged in (the login health check runs first and aborts
with the standard actionable message otherwise) — the same supervised
gate the meta_business composer recon had.

Run (backend up, Coursera logged in) via an async shim:
    from app.content.distribution.posting_drivers.coursera_recon import run_coursera_recon
    await run_coursera_recon()
Each state writes coursera-<state>-<stamp>.json/.png to data/browser_recon/.
"""
from __future__ import annotations

import base64
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import urljoin

from app.content.distribution.posting_drivers.coursera_driver import (
    PLATFORM,
    _is_item,
    item_type_from_url,
)
from app.content.distribution.posting_drivers.driver_runner import (
    find_in_snapshot,
    load_selector_map,
)

logger = logging.getLogger(__name__)

RECON_DIR = Path("D:/Orb/data/browser_recon")


def _default_bridge():
    from app.web_automation import bridge
    return bridge


def _resolve_session_id() -> Optional[str]:
    from app.db import SessionLocal
    from app.web_automation import session_registry
    db = SessionLocal()
    try:
        s = session_registry.get_session_by_platform(db, PLATFORM)
        return s.id if s else None
    finally:
        db.close()


async def _capture(bridge, session_id: str, state: str) -> dict:
    """Snapshot + screenshot one state to disk. Returns a small summary."""
    RECON_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = RECON_DIR / f"{PLATFORM}-{state}-{stamp}"

    dom = {}
    r = await bridge.execute_action(session_id, "dom_snapshot", {}, timeout_seconds=20.0)
    if r.get("ok"):
        dom = r.get("result") or {}
    st = await bridge.execute_action(session_id, "current_state", {}, timeout_seconds=8.0)
    cur = (st.get("result") or {}) if st.get("ok") else {}
    payload = {"state": state, "url": cur.get("url", ""), "title": cur.get("title", ""), "dom": dom}
    base.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    shot = await bridge.execute_action(
        session_id, "screenshot", {"full_page": False}, timeout_seconds=20.0
    )
    b64 = (shot.get("result") or {}).get("image_base64") or ""
    if b64:
        try:
            base.with_suffix(".png").write_bytes(base64.b64decode(b64))
        except Exception:
            pass
    n = len((dom or {}).get("elements") or [])
    logger.info("[coursera_recon] captured %s: %d elements -> %s.json", state, n, base)
    return {"state": state, "elements": n, "path": str(base.with_suffix(".json"))}


async def _elements(bridge, sid) -> List[dict]:
    r = await bridge.execute_action(sid, "dom_snapshot", {}, timeout_seconds=20.0)
    return ((r.get("result") or {}).get("elements")) or []


async def _click_text(bridge, sid, candidates: List[dict], wait_ms: int = 2500) -> bool:
    """Click the first text candidate found in a FRESH snapshot (coords
    from the snapshot taken immediately before — the sanctioned pattern).
    Recon-only helper; the driver itself clicks via the StepRunner."""
    els = await _elements(bridge, sid)
    for cand in candidates:
        coords = None
        if cand.get("css"):
            g = await bridge.execute_action(
                sid, "wait_for",
                {"selector": cand["css"], "state": "visible", "timeout_ms": 1500},
                timeout_seconds=10.0,
            )
            if g.get("ok") and (g.get("result") or {}).get("matched"):
                c = await bridge.execute_action(
                    sid, "click", {"selector": cand["css"]}, timeout_seconds=10.0
                )
                if c.get("ok"):
                    await bridge.execute_action(sid, "wait", {"ms": wait_ms}, timeout_seconds=8.0)
                    return True
            continue
        coords = find_in_snapshot(els, cand.get("text", ""), cand.get("role", ""))
        if coords:
            c = await bridge.execute_action(sid, "click", coords, timeout_seconds=10.0)
            if c.get("ok"):
                await bridge.execute_action(sid, "wait", {"ms": wait_ms}, timeout_seconds=8.0)
                return True
    return False


async def _current_url(bridge, sid) -> str:
    st = await bridge.execute_action(sid, "current_state", {}, timeout_seconds=8.0)
    return ((st.get("result") or {}).get("url") or "")


async def run_coursera_recon(
    *,
    bridge: Any = None,
    session_id: Optional[str] = None,
) -> dict:
    """Capture the study-loop states. Never touches quiz content.
    Returns {ok, states:[...]} or the health verdict when not logged in."""
    bridge = bridge or _default_bridge()
    sid = session_id or _resolve_session_id()
    if not sid:
        return {"ok": False, "error": f"no '{PLATFORM}' session registered"}

    # Login gate first — dumping the public site would only produce
    # misleading selector evidence.
    from app.web_automation.coursera_reader import _login_state
    health = await _login_state(PLATFORM)
    if health.get("state") != "ok":
        return {
            "ok": False,
            "state": health.get("state"),
            "error": health.get("error", ""),
            "message": health.get("message", "Coursera session is not ready for recon."),
        }

    smap = load_selector_map("coursera")
    steps = smap.get("steps", {})
    captured: List[dict] = []

    captured.append(await _capture(bridge, sid, "my_learning"))

    # Dashboard -> course (course home or straight to the last item).
    await _click_text(bridge, sid, steps.get("resume_entry", []), wait_ms=4000)
    url = await _current_url(bridge, sid)
    if item_type_from_url(url) == "course_home":
        captured.append(await _capture(bridge, sid, "course_home"))
        await _click_text(bridge, sid, steps.get("course_home_resume", []), wait_ms=4000)
        url = await _current_url(bridge, sid)

    lesson_url = url if _is_item(url) else ""
    if lesson_url:
        itype = item_type_from_url(lesson_url)
        captured.append(await _capture(bridge, sid, f"lesson_{itype}"))
        if itype == "video":
            if await _click_text(bridge, sid, steps.get("transcript_toggle", []), wait_ms=2000):
                captured.append(await _capture(bridge, sid, "lesson_video_transcript"))

        # Reading + quiz landing via hrefs in the lesson page's course nav.
        els = await _elements(bridge, sid)
        reading_href, quiz_href = "", ""
        for el in els:
            href = el.get("href") or ""
            if not reading_href and "/supplement/" in href:
                reading_href = href
            if not quiz_href and ("/quiz/" in href or "/exam/" in href):
                quiz_href = href
        for tag, href in (("reading", reading_href), ("quiz_landing", quiz_href)):
            if not href:
                continue
            await bridge.execute_action(
                sid, "navigate", {"url": urljoin(lesson_url, href)}, timeout_seconds=25.0
            )
            await bridge.execute_action(sid, "wait", {"ms": 3000}, timeout_seconds=8.0)
            captured.append(await _capture(bridge, sid, tag))
            # LANDING ONLY for quizzes — no clicks on that page, ever.

        # Put the view back where the user's study loop left it.
        await bridge.execute_action(
            sid, "navigate", {"url": lesson_url}, timeout_seconds=25.0
        )

    logger.info("[coursera_recon] done: %d states (quiz landing only, nothing attempted)", len(captured))
    return {
        "ok": True,
        "states": captured,
        "note": "quiz captured from its landing page only; nothing started or answered",
    }
