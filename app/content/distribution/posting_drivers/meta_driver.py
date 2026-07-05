# FILE: app/content/distribution/posting_drivers/meta_driver.py
# Purpose: Composite posting driver for the Meta Business Suite composer (FB+IG).
# Called-by: app.tools.social_posting_tools, tests.test_posting_meta_driver
# Depends-on: .driver_runner, .self_heal, .results, app.web_automation.bridge (+session_registry, popup_dismiss)
# Last-renovated: 2026-07-02
"""
Meta Business Suite posting driver.

post_image() and post_reel() drive the unified FB+IG composer through a
deterministic, wait-gated step sequence (see selector_maps/meta_business.json)
and return a PostResult. IG->FB auto-share on the linked accounts means one
post covers both platforms, so this single driver is the whole posting layer.

Everything the network/model touches is injectable (bridge, vision_fn,
session_id) so the flow is unit-testable end-to-end with a fake bridge and
no live browser and no model.
"""
from __future__ import annotations

import asyncio
import base64
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.content.distribution.posting_drivers import self_heal
from app.content.distribution.posting_drivers.driver_runner import StepRunner
from app.content.distribution.posting_drivers.results import PostResult, failure

logger = logging.getLogger(__name__)

PLATFORM = "meta_business"
AUDIT_ROOT = Path("D:/Orb/data/posting_audit")

# Human-readable goal per step, handed to self-heal's vision model.
_GOALS = {
    "composer_open": "the button that opens the post composer on the Business Suite home",
    "reel_composer_open": "the button that opens the reel composer on the Business Suite home",
    "composer_dialog": "the open composer dialog / modal",
    "media_input": "the file input for adding a photo or video in the composer",
    "add_media_button": "the Add photo/video button in the composer",
    "media_preview": "the attached media thumbnail or video preview inside the composer",
    "caption_field": "the caption / description text box in the composer",
    "next_button": "the Next button that advances the composer",
    "publish_button": "the Publish / Share button in the composer",
    "success_signal": "the confirmation that the post has been published",
}


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


async def _save_audit_shot(bridge, session_id: str, audit_dir: Path, tag: str) -> None:
    try:
        s = await bridge.execute_action(
            session_id, "screenshot", {"full_page": False}, timeout_seconds=20.0
        )
        b64 = (s.get("result") or {}).get("image_base64") or ""
        if b64:
            audit_dir.mkdir(parents=True, exist_ok=True)
            (audit_dir / f"{tag}.png").write_bytes(base64.b64decode(b64))
    except Exception as e:
        logger.debug("[meta_driver] audit screenshot (%s) failed: %s", tag, e)


async def _read_permalink(bridge, session_id: str) -> Optional[str]:
    """Best-effort: after publish, look for a 'View post' style link. Tolerates absence."""
    try:
        r = await bridge.execute_action(session_id, "dom_snapshot", {}, timeout_seconds=15.0)
        for el in ((r.get("result") or {}).get("elements") or []):
            href = el.get("href") or ""
            txt = (el.get("text") or "").lower()
            if href and ("view" in txt or "/posts/" in href or "/reel" in href):
                return href
    except Exception:
        pass
    return None


async def _run(
    kind: str,
    file_path: str,
    caption: str,
    *,
    bridge: Any = None,
    session_id: Optional[str] = None,
    vision_fn=None,
    pace_range: tuple = (0.8, 2.5),
    persist_heal: bool = True,
    selector_map: Optional[dict] = None,
) -> PostResult:
    bridge = bridge or _default_bridge()
    if not file_path or not Path(file_path).exists():
        return failure(PLATFORM, failed_step="preflight", error=f"file not found: {file_path}")

    sid = session_id or _resolve_session_id(PLATFORM)
    if not sid:
        return failure(
            PLATFORM, failed_step="session",
            error=f"no '{PLATFORM}' web session registered — cannot reach the composer",
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    audit_dir = AUDIT_ROOT / f"{stamp}_{kind}"

    async def heal(session_id_, step, goal, els):
        return await self_heal.relocate(
            session_id_, step, goal, els, bridge=bridge, vision_fn=vision_fn
        )

    runner = StepRunner(
        sid, PLATFORM, "meta_business",
        bridge=bridge, heal=heal, pace_range=pace_range,
        persist_heal=persist_heal, selector_map=selector_map,
    )

    async def fail(step: str, err: str) -> PostResult:
        await _save_audit_shot(bridge, sid, audit_dir, f"fail-{step}")
        await self_heal.dump_recon(sid, PLATFORM, step, bridge=bridge, audit_dir=str(audit_dir))
        return failure(
            PLATFORM, failed_step=step, error=err,
            audit_dir=str(audit_dir), steps=runner.trace,
        )

    # 1. Make sure the session view is live and on the home surface.
    try:
        await bridge.ensure_session_open(sid, timeout_seconds=20.0)
    except Exception as e:
        return await fail("session", f"could not open session view: {e}")

    landing = (runner.map.get("landing_url")) or "https://business.facebook.com/latest/home"
    await bridge.execute_action(sid, "navigate", {"url": landing}, timeout_seconds=30.0)
    if not await runner.gate("home_ready", timeout_ms=20000):
        return await fail("home_ready", "Business Suite home did not load (session may be logged out)")

    # 2. Clear cookie/first-run popups (fire-and-forget, shared helper).
    try:
        from app.content.distribution.browser_analytics.popup_dismiss import dismiss_common_popups
        await dismiss_common_popups(sid)
    except Exception:
        pass

    # 3. Open the composer.
    open_step = "reel_composer_open" if kind == "reel" else "composer_open"
    await runner.pace()
    r = await runner.act(open_step, "click", goal=_GOALS[open_step], timeout_ms=15000)
    if not r["ok"]:
        return await fail(open_step, r.get("error", "could not open composer"))

    if not await runner.gate("composer_dialog", timeout_ms=15000):
        return await fail("composer_dialog", "composer dialog did not appear")

    # 4. Attach the media — selector file-input first (no click), button fallback.
    await runner.pace()
    up = await runner.act(
        "media_input", "upload", value=file_path,
        gate_state="attached", timeout_ms=12000, goal=_GOALS["media_input"],
    )
    if not up["ok"]:
        up = await runner.act(
            "add_media_button", "upload", value=file_path,
            timeout_ms=12000, goal=_GOALS["add_media_button"],
        )
    if not up["ok"]:
        return await fail("media_input", up.get("error", "could not attach media"))

    # 5. Wait for the preview to render (upload + transcode can be slow).
    if not await runner.gate("media_preview", timeout_ms=45000):
        return await fail("media_preview", "media preview never rendered after upload")

    # 6. Caption.
    await runner.pace()
    cap = await runner.act("caption_field", "type", value=caption, goal=_GOALS["caption_field"])
    if not cap["ok"]:
        return await fail("caption_field", cap.get("error", "could not enter caption"))

    # 7. Reels walk through one or more Next steps before publish; posts don't.
    if kind == "reel":
        for _ in range(3):
            if not await runner.gate("next_button", timeout_ms=4000):
                break
            await runner.pace()
            nxt = await runner.act("next_button", "click", timeout_ms=6000, goal=_GOALS["next_button"])
            if not nxt["ok"]:
                break

    # 8. Publish.
    await runner.pace()
    pub = await runner.act("publish_button", "click", timeout_ms=15000, goal=_GOALS["publish_button"])
    if not pub["ok"]:
        return await fail("publish_button", pub.get("error", "could not click publish"))

    # 9. Confirm — DOM success signal is the source of truth.
    if not await runner.gate("success_signal", timeout_ms=45000):
        return await fail("success_signal", "no publish confirmation seen; post status unknown")

    permalink = await _read_permalink(bridge, sid)
    await _save_audit_shot(bridge, sid, audit_dir, "success")
    logger.info("[meta_driver] %s published (%s) permalink=%s", kind, PLATFORM, permalink)
    return PostResult(
        ok=True, platform=PLATFORM, permalink=permalink,
        audit_dir=str(audit_dir), steps=runner.trace,
    )


async def post_image(file_path: str, caption: str, **kwargs) -> PostResult:
    """Publish a still image to the Business Suite composer (FB + IG)."""
    return await _run("image", file_path, caption, **kwargs)


async def post_reel(file_path: str, caption: str = "", **kwargs) -> PostResult:
    """Publish a reel/short (any local mp4) to the Business Suite composer (FB + IG)."""
    return await _run("reel", file_path, caption, **kwargs)
