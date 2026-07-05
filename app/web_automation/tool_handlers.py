# FILE: app/web_automation/tool_handlers.py
# Purpose: Tool handlers for ASTRA's LLM to invoke web-automation primitives.
# Called-by: app.debug.executors.web_automation, app.web_automation.coursera_reader, app.web_automation.register
# Depends-on: app.db, app.llm.gemini_vision, app.web_automation, app.web_automation.bridge (+2 more)
# Last-renovated: 2026-07-01
"""
Tool handlers for ASTRA's LLM to invoke web-automation primitives.

Each handler:
  1. Resolves a session reference (id or platform key) to a WebSession row.
  2. Dispatches the primitive action via the bridge.
  3. Normalises the result shape so the LLM always sees a predictable schema.

The LLM sees these via app.tools.registry. The tool descriptions below
are what it reads to decide which to call.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.db import SessionLocal
from app.web_automation import bridge, session_registry
from app.web_automation.models import WebSession

logger = logging.getLogger(__name__)


def _resolve_session(ref: str) -> Optional[WebSession]:
    """Accept either a UUID session id or a platform key."""
    db = SessionLocal()
    try:
        sess = session_registry.get_session(db, ref)
        if sess:
            return sess
        return session_registry.get_session_by_platform(db, ref)
    finally:
        db.close()


async def _dispatch(ref: str, action_type: str, payload: dict, timeout_seconds: float = 30.0) -> dict:
    sess = _resolve_session(ref)
    if not sess:
        return {"ok": False, "error": f"no session matches '{ref}'"}
    return await bridge.execute_action(
        sess.id, action_type, payload, timeout_seconds=timeout_seconds
    )


# ── Handlers (one per tool) ──────────────────────────────────────────

async def list_sessions_handler(input_data: dict, context: Optional[dict]) -> dict:
    db = SessionLocal()
    try:
        sessions = session_registry.list_sessions(db)
        return {
            "sessions": [
                {
                    "id": s.id,
                    "platform": s.platform,
                    "label": s.label,
                    "purpose": s.purpose or "",
                    "status": s.status.value if hasattr(s.status, "value") else str(s.status),
                    "current_url": s.current_url or "",
                }
                for s in sessions
            ],
        }
    finally:
        db.close()


async def open_session_handler(input_data: dict, context: Optional[dict]) -> dict:
    ref = str(input_data.get("session") or "")
    sess = _resolve_session(ref)
    if not sess:
        return {"ok": False, "error": f"no session matches '{ref}'"}
    result = await bridge.ensure_session_open(sess.id)
    sess2 = _resolve_session(sess.id)
    return {
        "ok": bool(result.get("ok")),
        "session_id": sess.id,
        "current_url": (sess2.current_url if sess2 else None) or "",
        "error": result.get("error") or "",
    }


async def navigate_handler(input_data: dict, context: Optional[dict]) -> dict:
    ref = str(input_data.get("session") or "")
    url = str(input_data.get("url") or "")
    result = await _dispatch(ref, "navigate", {"url": url})
    r = result.get("result") or {}
    return {
        "ok": bool(result.get("ok")),
        "current_url": r.get("current_url", ""),
        "current_title": r.get("current_title", ""),
        "error": result.get("error") or "",
    }


async def click_handler(input_data: dict, context: Optional[dict]) -> dict:
    ref = str(input_data.get("session") or "")
    payload: dict = {}
    if "selector" in input_data and input_data["selector"]:
        payload["selector"] = input_data["selector"]
    # role+name targeting: resolved INSIDE the page at click time (the
    # executor matches role/aria-label/visible text, the same fields
    # dom_snapshot reports) so it survives SPA re-renders.
    if input_data.get("name"):
        payload["name"] = str(input_data["name"])
        if input_data.get("role"):
            payload["role"] = str(input_data["role"])
    if "x" in input_data and "y" in input_data:
        payload["x"] = int(input_data["x"])
        payload["y"] = int(input_data["y"])
    if input_data.get("snap_to_button"):
        payload["snap_to_button"] = True
    if not payload:
        return {"ok": False, "error": "click requires a selector, a role+name pair, or (x, y)"}
    # Bigger timeout: click.js now captures state, waits 800ms to
    # settle, and captures again. Budget ~2s; give bridge 10s headroom.
    result = await _dispatch(ref, "click", payload, timeout_seconds=10.0)
    r = result.get("result") or {}
    return {
        "ok": bool(result.get("ok")),
        "clicked_at": r.get("clicked_at"),
        "resolved": r.get("resolved"),
        "snapped": r.get("snapped"),
        "before": r.get("before"),
        "after": r.get("after"),
        "changed": bool(r.get("changed", False)),
        "url_changed": bool(r.get("url_changed", False)),
        "title_changed": bool(r.get("title_changed", False)),
        "heading_changed": bool(r.get("heading_changed", False)),
        "element_changed": bool(r.get("element_changed", False)),
        "error": result.get("error") or "",
    }


async def type_handler(input_data: dict, context: Optional[dict]) -> dict:
    ref = str(input_data.get("session") or "")
    payload = {
        "selector": input_data.get("selector"),
        "text": input_data.get("text"),
        "clear_first": bool(input_data.get("clear_first", True)),
    }
    result = await _dispatch(ref, "type", payload)
    return {"ok": bool(result.get("ok")), "error": result.get("error") or ""}


async def screenshot_handler(input_data: dict, context: Optional[dict]) -> dict:
    ref = str(input_data.get("session") or "")
    full = bool(input_data.get("full_page", False))
    result = await _dispatch(ref, "screenshot", {"full_page": full}, timeout_seconds=20.0)
    r = result.get("result") or {}
    return {
        "ok": bool(result.get("ok")),
        "image_base64": r.get("image_base64", ""),
        "image_path": r.get("image_path", ""),
        "error": result.get("error") or "",
    }


async def extract_text_handler(input_data: dict, context: Optional[dict]) -> dict:
    ref = str(input_data.get("session") or "")
    payload = {
        "selector": input_data.get("selector"),
        "limit": int(input_data.get("limit", 20)),
    }
    result = await _dispatch(ref, "extract_text", payload)
    r = result.get("result") or {}
    return {
        "ok": bool(result.get("ok")),
        "matches": r.get("matches", []),
        "error": result.get("error") or "",
    }


async def current_state_handler(input_data: dict, context: Optional[dict]) -> dict:
    ref = str(input_data.get("session") or "")
    result = await _dispatch(ref, "current_state", {}, timeout_seconds=5.0)
    r = result.get("result") or {}
    return {
        "ok": bool(result.get("ok")),
        "url": r.get("url", ""),
        "title": r.get("title", ""),
        "error": result.get("error") or "",
    }


async def scroll_handler(input_data: dict, context: Optional[dict]) -> dict:
    ref = str(input_data.get("session") or "")
    payload = {
        "direction": input_data.get("direction", "down"),
        "amount": int(input_data.get("amount", 500)),
    }
    result = await _dispatch(ref, "scroll", payload)
    return {"ok": bool(result.get("ok")), "error": result.get("error") or ""}


async def wait_for_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Block until a page condition holds (element/text/URL) or timeout.

    Timeout is NOT an error: the executor answers ok=true, matched=false,
    timeout=true and the caller decides what that means for its step.
    """
    ref = str(input_data.get("session") or "")
    payload: dict = {}
    for key in ("selector", "text", "url_pattern"):
        if input_data.get(key):
            payload[key] = str(input_data[key])
    if not payload:
        return {"ok": False, "error": "wait_for requires at least one of selector, text, url_pattern"}
    if input_data.get("state"):
        payload["state"] = str(input_data["state"])
    try:
        timeout_ms = int(input_data.get("timeout_ms", 15000))
        if "poll_ms" in input_data:
            payload["poll_ms"] = int(input_data["poll_ms"])
    except (TypeError, ValueError):
        return {"ok": False, "error": "timeout_ms and poll_ms must be integers"}
    payload["timeout_ms"] = timeout_ms

    # Bridge timeout must outlive the executor's own poll deadline
    # (executor clamps to 60s), plus dispatch overhead.
    result = await _dispatch(
        ref, "wait_for", payload,
        timeout_seconds=min(max(timeout_ms, 0) / 1000.0, 60.0) + 10.0,
    )
    r = result.get("result") or {}
    return {
        "ok": bool(result.get("ok")),
        "matched": bool(r.get("matched", False)),
        "timeout": bool(r.get("timeout", False)),
        "waited_ms": int(r.get("waited_ms", 0)),
        "error": result.get("error") or "",
    }


async def dom_snapshot_handler(input_data: dict, context: Optional[dict]) -> dict:
    ref = str(input_data.get("session") or "")
    result = await _dispatch(ref, "dom_snapshot", {}, timeout_seconds=20.0)
    r = result.get("result") or {}
    return {
        "ok": bool(result.get("ok")),
        "elements": r.get("elements", []),
        "error": result.get("error") or "",
    }


async def vision_check_handler(input_data: dict, context: Optional[dict]) -> dict:
    """
    Screenshot the session's current page and ask a vision model a
    natural-language question about what's visible.

    Used for high-stakes verification that DOM/element-state checks
    can't answer confidently:
      • Pre-submit read-back ("does the compose box contain X?")
      • Post-submit confirmation ("is the post now live on the wall?")
      • Error detection ("is there an error banner visible?")
      • Layout/visual cross-check ("does this look right?")

    Kept intentionally expensive-by-default so it isn't over-used. The
    click and type primitives already have cheap DOM-level verify; this
    tool is for the irreversible-action tier.
    """
    import asyncio
    import base64 as _b64

    ref = str(input_data.get("session") or "")
    question = str(input_data.get("question") or "").strip()
    if not question:
        return {"ok": False, "error": "vision_check requires a 'question' parameter"}

    # 1. Capture a fresh screenshot. Not reusing an older one — the
    #    point of vision check is to see the page AS IT IS NOW, after
    #    whatever we just did. Bigger timeout because screenshot itself
    #    can take a few seconds on busy pages.
    shot = await _dispatch(
        ref, "screenshot", {"full_page": False}, timeout_seconds=20.0
    )
    if not shot.get("ok"):
        return {
            "ok": False,
            "error": f"screenshot failed: {shot.get('error') or 'unknown'}",
        }
    shot_r = shot.get("result") or {}
    b64 = shot_r.get("image_base64") or ""
    if not b64:
        return {"ok": False, "error": "screenshot returned no image data"}

    # 2. Decode and feed to ask_about_image. That function is sync
    #    (Gemini SDK call) so run it in a thread to keep the async
    #    event loop responsive.
    try:
        image_bytes = _b64.b64decode(b64)
    except Exception as e:
        return {"ok": False, "error": f"could not decode screenshot: {e}"}

    from app.llm.gemini_vision import ask_about_image

    loop = asyncio.get_event_loop()
    try:
        vision_result = await loop.run_in_executor(
            None,
            lambda: ask_about_image(
                image_source=image_bytes,
                user_question=question,
                mime_type="image/png",
                # Flash tier: fast + cheap, sufficient for verify-style
                # yes/no/read-back questions. Bump to "complex" (Gemini
                # 2.5 Pro) later if accuracy becomes an issue.
                tier="fast",
            ),
        )
    except Exception as e:
        logger.exception("[vision_check] Vision model call raised")
        return {"ok": False, "error": f"vision model error: {e}"}

    if vision_result.get("error"):
        return {
            "ok": False,
            "error": f"vision model: {vision_result['error']}",
            "answer": vision_result.get("answer", ""),
        }

    return {
        "ok": True,
        "answer": vision_result.get("answer", ""),
        "provider": vision_result.get("provider", ""),
        "model": vision_result.get("model", ""),
        "screenshot_path": shot_r.get("image_path", ""),
    }


async def upload_file_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Attach a local file to a web upload control.

    Mode is auto-selected from the params: a `selector` triggers the legacy
    DOM.setFileInputFiles path; otherwise `(x, y)` triggers the modern
    Page.setInterceptFileChooserDialog path. Modern path is what works on
    Meta / IG / TikTok / YouTube / WordPress.
    """
    ref = str(input_data.get("session") or "")
    file_path = str(input_data.get("file_path") or "")
    if not file_path:
        return {"ok": False, "error": "upload_file requires file_path"}

    payload: dict = {"file_path": file_path}
    if input_data.get("selector"):
        payload["selector"] = input_data["selector"]
    elif "x" in input_data and "y" in input_data:
        try:
            payload["x"] = int(input_data["x"])
            payload["y"] = int(input_data["y"])
        except (TypeError, ValueError):
            return {"ok": False, "error": "x and y must be integers"}
    else:
        return {
            "ok": False,
            "error": "upload_file requires either `selector` or both `x` and `y`",
        }

    # File picker can take a moment on busy SPAs; 30s headroom is plenty.
    result = await _dispatch(ref, "upload_file", payload, timeout_seconds=30.0)
    r = result.get("result") or {}
    return {
        "ok": bool(result.get("ok")),
        "mode": r.get("mode", ""),
        "attached": r.get("attached", ""),
        "error": result.get("error") or "",
    }


async def system_keys_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Send keystrokes to whatever native OS window currently has focus.

    Used after a click that opened a native file Open dialog (Meta IG
    TikTok WordPress — "Upload from desktop" routes through the OS
    chooser when CDP intercept misses). Type the absolute path and
    press Enter; Windows accepts paths in the filename field, no
    folder navigation needed.
    """
    ref = str(input_data.get("session") or "")
    text = str(input_data.get("text") or "")
    if not text:
        return {"ok": False, "error": "system_keys requires `text`"}

    payload: dict = {"text": text}
    if "press_enter_after" in input_data:
        payload["press_enter_after"] = bool(input_data["press_enter_after"])
    if "pre_delay_ms" in input_data:
        try:
            payload["pre_delay_ms"] = int(input_data["pre_delay_ms"])
        except (TypeError, ValueError):
            return {"ok": False, "error": "pre_delay_ms must be an integer"}

    # 15s gives the executor's own 10s PowerShell timeout headroom
    # plus dispatch overhead. Real run is sub-second.
    result = await _dispatch(ref, "system_keys", payload, timeout_seconds=15.0)
    r = result.get("result") or {}
    return {
        "ok":            bool(result.get("ok")),
        "sent":          r.get("sent", ""),
        "pre_delay_ms":  r.get("pre_delay_ms", 0),
        "pressed_enter": bool(r.get("pressed_enter", False)),
        "error":         result.get("error") or "",
    }


# ── Registry ─────────────────────────────────────────────────────────

HANDLERS = {
    "web_list_sessions":  list_sessions_handler,
    "web_open_session":   open_session_handler,
    "web_navigate":       navigate_handler,
    "web_click":          click_handler,
    "web_type":           type_handler,
    "web_screenshot":     screenshot_handler,
    "web_extract_text":   extract_text_handler,
    "web_current_state":  current_state_handler,
    "web_scroll":         scroll_handler,
    "web_wait_for":       wait_for_handler,
    "web_dom_snapshot":   dom_snapshot_handler,
    "web_vision_check":   vision_check_handler,
    "web_upload_file":    upload_file_handler,
    "system_keys":        system_keys_handler,
}


TOOL_DESCRIPTIONS = {
    "web_list_sessions":
        "List all registered web-automation sessions (Facebook, Instagram, TikTok, YouTube Studio, "
        "Coursera, WordPress, etc.) with their platform key, label, purpose, and live status. "
        "Call this first if you don't know which session you need.",
    "web_open_session":
        "Ensure a web session is live (Electron spawns the browser view if not). "
        "Pass either the platform key (e.g. 'meta_business') or the session UUID as `session`. "
        "If this (or any web tool) errors with 'desktop browser is offline', the ASTRA "
        "desktop app is not running — tell the user to start it; that is NOT a page problem. "
        "'browser is busy on a previous action' means the desktop IS up but mid-action: "
        "wait a few seconds and retry instead.",
    "web_navigate":
        "Navigate the named session's browser view to a URL. Returns after the page's "
        "load event; SPA content can still render late — follow with web_wait_for on a "
        "known element before the next read or action.",
    "web_click":
        "Click an element. Targeting, in order of preference: (1) `role` + `name` from "
        "the latest web_dom_snapshot — resolved INSIDE the live page at click time, so "
        "it survives re-renders; (2) a stable CSS `selector` (aria-label attributes, "
        "IDs — never obfuscated class names); (3) `(x, y)` ONLY from a snapshot taken "
        "immediately before this click with nothing in between — SPA re-renders, "
        "lazy-loading and scrolling all move elements, and a click at stale coordinates "
        "hits whatever sits there NOW. The result includes `resolved` (what role/name "
        "matched), `before`/`after` states and a `changed` boolean. If `changed` is "
        "false, do NOT tell the user 'done'; re-run web_dom_snapshot and re-orient. "
        "When multiple elements share the same text, prefer the larger `w`/`h`.",
    "web_type":
        "Type `text` into the element matching `selector`. Field is cleared first by default.",
    "web_screenshot":
        "Capture a PNG screenshot of the session's current page. Returns base64 + a saved file path.",
    "web_extract_text":
        "Return text content of all elements matching a CSS `selector` (up to `limit`). "
        "Useful for reading comment threads, insights values, article bodies, etc. "
        "TEXT ONLY: completion/progress state (Coursera module ticks, progress bars) "
        "is rendered visually and does NOT come back — use web_coursera_progress or "
        "web_vision_check for progress questions, never this.",
    "web_current_state":
        "Report the current URL and page title for a session. Cheap — use before deciding next action.",
    "web_scroll":
        "Scroll the page by `amount` pixels in `direction` (up/down) or jump to 'top'/'bottom'.",
    "web_wait_for":
        "Wait until the page is actually ready instead of sleeping: pass a CSS `selector` "
        "(with `state` visible/attached/gone), a `text` snippet, and/or a `url_pattern` regex. "
        "Polls until matched or `timeout_ms` (default 15000). A timeout comes back ok=true "
        "with timeout=true — decide yourself whether that's fatal. Use this after EVERY "
        "navigate, after any click that changes page state (opens a composer/dialog/route), "
        "and after upload/attach steps, BEFORE the next read or action; use state='gone' to "
        "wait for spinners/modals to clear. On timeout, do NOT blind-retry the last action — "
        "take a fresh web_dom_snapshot and re-orient first.",
    "web_dom_snapshot":
        "Return the page's accessibility tree: every interactive element (links, buttons, "
        "headings) with role, name, text, href, and pixel coordinates. The PRIMARY way to "
        "understand a page. Act on what you find via web_click with role+name (copy them "
        "from the element), or a stable CSS selector; use coordinates only immediately "
        "after this snapshot with nothing in between — they go stale on re-render/"
        "lazy-load/scroll. Works reliably on React SPAs with obfuscated class names. "
        "CAVEAT: the tree carries no visual done-state — course completion ticks/progress "
        "bars (Coursera) look identical for done and not-done items here; read progress "
        "via web_coursera_progress or web_vision_check instead.",
    "web_vision_check":
        "Screenshot the session's live page and ask a vision model a natural-language "
        "question about what's visible. Returns the model's text answer. Use for "
        "high-stakes or visual-only checks: pre-submit read-back, post-submit "
        "confirmation, error-banner detection, and reading VISUAL state that text "
        "extraction can't see — e.g. Coursera module completion ticks and progress "
        "bars (for those, prefer the web_coursera_progress composite). Costs a real "
        "vision-model call; don't use it where the DOM verify already answers.",
    "web_upload_file":
        "Attach a local file to a web upload control. Two modes: pass `selector` for plain HTML "
        "<input type=file>, or pass (x, y) of the upload BUTTON for modern SPAs (Meta, IG, TikTok, "
        "YouTube) where the input is hidden behind a styled button. The (x, y) path uses Chrome "
        "DevTools file-chooser interception — it suppresses the native OS dialog and injects the "
        "file directly. Use web_dom_snapshot first to get the upload button's coordinates. Never "
        "use read_file to get image bytes for upload — that does NOT attach the file to the form.",
    "system_keys":
        "Send keystrokes to whatever native OS window currently has focus. Use this when a click "
        "in the embedded webview opened a NATIVE file Open dialog (Windows file picker from Meta's "
        "'Upload from desktop', etc.) and CDP intercept didn't catch it. Pass the absolute file "
        "path as `text` and `press_enter_after=true`. Windows accepts paths directly in the "
        "filename field, so no folder navigation is needed. Currently Windows-only.",
}
