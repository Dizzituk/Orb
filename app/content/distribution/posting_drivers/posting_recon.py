# FILE: app/content/distribution/posting_drivers/posting_recon.py
# Purpose: Non-destructive recon of the Meta Business Suite composer states (Job 2).
# Called-by: manual/supervised run + tests.test_posting_recon (import only)
# Depends-on: .driver_runner, app.web_automation.bridge, browser_analytics.popup_dismiss
# Last-renovated: 2026-07-02
"""
Composer recon (jobspec Job 2) — capture the accessibility tree + a
screenshot for each REACHABLE composer state so selector_maps can be
confirmed/refined against the real DOM.

SAFETY: this is non-destructive. It walks home -> composer open ->
(optional) media attached -> caption focused, and STOPS. It never
clicks Publish/Share, so nothing is posted. The share/success states
are captured on the first supervised real post (they can't be reached
without publishing to the live accounts).

Run it (backend up, meta_business logged in) via a small async shim or
the tools layer; each state writes to data/browser_recon/.
"""
from __future__ import annotations

import base64
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from app.content.distribution.posting_drivers.driver_runner import load_selector_map

logger = logging.getLogger(__name__)

RECON_DIR = Path("D:/Orb/data/browser_recon")
PLATFORM = "meta_business"


def _default_bridge():
    from app.web_automation import bridge
    return bridge


def _resolve_session_id(platform: str) -> Optional[str]:
    from app.db import SessionLocal
    from app.web_automation import session_registry
    db = SessionLocal()
    try:
        s = session_registry.get_session_by_platform(db, platform)
        return s.id if s else None
    finally:
        db.close()


async def _capture(bridge, session_id: str, state: str) -> dict:
    """Snapshot + screenshot one state to disk. Returns a small summary."""
    RECON_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base = RECON_DIR / f"{PLATFORM}-composer-{state}-{stamp}"

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
    logger.info("[posting_recon] captured %s: %d elements -> %s.json", state, n, base)
    return {"state": state, "elements": n, "path": str(base.with_suffix(".json"))}


async def _click_first(bridge, session_id: str, candidates: List[dict], timeout_ms: int) -> bool:
    """Gate + click the first candidate that appears. Selector-only (recon safe)."""
    for cand in candidates:
        if not cand.get("css"):
            continue
        g = await bridge.execute_action(
            session_id, "wait_for",
            {"selector": cand["css"], "state": "visible", "timeout_ms": timeout_ms},
            timeout_seconds=timeout_ms / 1000.0 + 8.0,
        )
        if bool(g.get("ok")) and (g.get("result") or {}).get("matched"):
            c = await bridge.execute_action(
                session_id, "click", {"selector": cand["css"]}, timeout_seconds=10.0
            )
            if c.get("ok"):
                return True
    return False


async def run_posting_recon(
    kind: str = "post",
    *,
    bridge: Any = None,
    session_id: Optional[str] = None,
    attach_file: Optional[str] = None,
) -> dict:
    """Capture composer states. Never publishes. Returns {ok, states:[...]}."""
    bridge = bridge or _default_bridge()
    sid = session_id or _resolve_session_id(PLATFORM)
    if not sid:
        return {"ok": False, "error": f"no '{PLATFORM}' session registered"}

    smap = load_selector_map("meta_business")
    steps = smap.get("steps", {})
    landing = smap.get("landing_url", "https://business.facebook.com/latest/home")
    captured: List[dict] = []

    await bridge.ensure_session_open(sid, timeout_seconds=20.0)
    await bridge.execute_action(sid, "navigate", {"url": landing}, timeout_seconds=30.0)
    await bridge.execute_action(sid, "wait", {"ms": 4000}, timeout_seconds=8.0)
    try:
        from app.content.distribution.browser_analytics.popup_dismiss import dismiss_common_popups
        await dismiss_common_popups(sid)
    except Exception:
        pass
    captured.append(await _capture(bridge, sid, "home"))

    open_step = "reel_composer_open" if kind == "reel" else "composer_open"
    if await _click_first(bridge, sid, steps.get(open_step, []), 12000):
        await bridge.execute_action(sid, "wait", {"ms": 2500}, timeout_seconds=6.0)
        captured.append(await _capture(bridge, sid, "composer_open"))

        if attach_file and Path(attach_file).exists():
            for cand in steps.get("media_input", []):
                if not cand.get("css"):
                    continue
                u = await bridge.execute_action(
                    sid, "upload_file", {"selector": cand["css"], "file_path": attach_file},
                    timeout_seconds=30.0,
                )
                if u.get("ok"):
                    await bridge.execute_action(sid, "wait", {"ms": 5000}, timeout_seconds=8.0)
                    captured.append(await _capture(bridge, sid, "media_attached"))
                    break

        if await _click_first(bridge, sid, steps.get("caption_field", []), 6000):
            await bridge.execute_action(sid, "wait", {"ms": 1200}, timeout_seconds=4.0)
            captured.append(await _capture(bridge, sid, "caption_focused"))

    logger.info("[posting_recon] done: %d states (STOPPED before publish)", len(captured))
    return {"ok": True, "kind": kind, "states": captured, "note": "share/success not captured — reached only by publishing"}
