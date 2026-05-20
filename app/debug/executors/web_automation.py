# FILE: app/debug/executors/web_automation.py
"""
Chat-callable executors for web browsing.

Each function wraps a handler in app/web_automation/tool_handlers.py
and adapts its return shape to what the chat tool loop expects:
  - Chat executors take  `params: Dict`          and return a `str`.
  - web_automation handlers take (input_data, context) and return a dict.

The str returned here is JSON-formatted so the LLM can parse it back
into structured data during its tool-result reading step. Keep return
strings compact: the chat_tool_loop caps each tool result at 30,000
chars, and dom_snapshots on busy pages can exceed that easily.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

from app.web_automation.tool_handlers import (
    list_sessions_handler,
    open_session_handler,
    navigate_handler,
    current_state_handler,
    dom_snapshot_handler,
    click_handler,
    type_handler,
    scroll_handler,
    extract_text_handler,
    screenshot_handler,
    vision_check_handler,
    upload_file_handler,
    system_keys_handler,
)

logger = logging.getLogger(__name__)


# ─── Formatting helpers ──────────────────────────────────────────────

def _fmt(data: Dict[str, Any]) -> str:
    """JSON-pretty a result dict. Falls back to repr on encode error."""
    try:
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)
    except Exception:
        return repr(data)


def _err(tool: str, msg: str) -> str:
    return _fmt({"ok": False, "tool": tool, "error": msg})


# ─── 1 · list sessions ──────────────────────────────────────────────

async def execute_web_list_sessions(params: Dict[str, Any]) -> str:
    """
    Return all registered browser sessions. Shape mirrors the LLM-facing
    schema: one row per session with id, platform, label, purpose,
    status, current_url.
    """
    try:
        result = await list_sessions_handler({}, None)
        return _fmt(result)
    except Exception as e:
        logger.exception("[web_exec] list_sessions failed")
        return _err("web_list_sessions", str(e))


# ─── 2 · open session ───────────────────────────────────────────────

async def execute_web_open_session(params: Dict[str, Any]) -> str:
    try:
        result = await open_session_handler(
            {"session": params.get("session", "")}, None,
        )
        return _fmt(result)
    except Exception as e:
        logger.exception("[web_exec] open_session failed")
        return _err("web_open_session", str(e))


# ─── 3 · current state ──────────────────────────────────────────────

async def execute_web_current_state(params: Dict[str, Any]) -> str:
    try:
        result = await current_state_handler(
            {"session": params.get("session", "")}, None,
        )
        return _fmt(result)
    except Exception as e:
        logger.exception("[web_exec] current_state failed")
        return _err("web_current_state", str(e))


# ─── 4 · dom snapshot ───────────────────────────────────────────────

# Guard against huge pages blowing the LLM context. chat_tool_loop
# caps at 30k chars; we pre-filter elements down to the most useful
# subset (visible interactive elements + headings + links with text).
_DOM_MAX_ELEMENTS = 200

_USEFUL_ROLES = {
    "link", "button", "heading", "menuitem", "tab",
    "textbox", "searchbox", "combobox", "listitem",
}
_USEFUL_TAGS = {"a", "button", "h1", "h2", "h3", "h4", "input", "textarea"}


def _filter_dom(elements: list[dict]) -> list[dict]:
    """
    Return a subset of elements that are likely relevant for navigation:
    interactive roles (links, buttons) and text-bearing structural ones
    (headings). Drops purely decorative or empty entries. Cap length
    at _DOM_MAX_ELEMENTS so we don't blow the LLM context.
    """
    filtered: list[dict] = []
    for el in elements:
        role = (el.get("role") or "").lower()
        tag = (el.get("tag") or "").lower()
        text = (el.get("text") or "").strip()
        href = el.get("href") or ""

        is_useful = (
            role in _USEFUL_ROLES
            or tag in _USEFUL_TAGS
            or (text and len(text) < 500)  # any text element, short enough
        )
        if not is_useful:
            continue

        # Keep the fields the LLM uses for navigation + disambiguation.
        # w/h matter when multiple elements share the same text — the
        # LLM should prefer the larger one (CTA button) over a smaller
        # duplicate (menu item). Cost: ~8 tokens per element.
        trimmed = {
            "tag": tag or None,
            "role": role or None,
            "text": text[:300] if text else None,
            "x": el.get("x"),
            "y": el.get("y"),
            "w": el.get("w"),
            "h": el.get("h"),
        }
        if href:
            trimmed["href"] = href[:500]
        # Strip None fields
        trimmed = {k: v for k, v in trimmed.items() if v is not None}

        filtered.append(trimmed)
        if len(filtered) >= _DOM_MAX_ELEMENTS:
            break

    return filtered


async def execute_web_dom_snapshot(params: Dict[str, Any]) -> str:
    try:
        result = await dom_snapshot_handler(
            {"session": params.get("session", "")}, None,
        )
        if not result.get("ok"):
            return _fmt(result)

        raw = result.get("elements", []) or []
        filtered = _filter_dom(raw)
        return _fmt({
            "ok": True,
            "total_elements_raw": len(raw),
            "returned": len(filtered),
            "truncated": len(raw) > _DOM_MAX_ELEMENTS,
            "elements": filtered,
        })
    except Exception as e:
        logger.exception("[web_exec] dom_snapshot failed")
        return _err("web_dom_snapshot", str(e))


# ─── 5 · navigate ───────────────────────────────────────────────────

async def execute_web_navigate(params: Dict[str, Any]) -> str:
    try:
        result = await navigate_handler(
            {
                "session": params.get("session", ""),
                "url": params.get("url", ""),
            },
            None,
        )
        return _fmt(result)
    except Exception as e:
        logger.exception("[web_exec] navigate failed")
        return _err("web_navigate", str(e))


# ─── 6 · click ──────────────────────────────────────────────────────

async def execute_web_click(params: Dict[str, Any]) -> str:
    try:
        payload: Dict[str, Any] = {"session": params.get("session", "")}
        # Pass through whichever targeting method the caller provided.
        if "selector" in params and params["selector"]:
            payload["selector"] = params["selector"]
        if "x" in params and "y" in params:
            payload["x"] = params["x"]
            payload["y"] = params["y"]
        if params.get("snap_to_button"):
            payload["snap_to_button"] = True
        result = await click_handler(payload, None)

        # Self-verify hint. click.js captures url+title+heading AND the
        # element at the click coordinates (checked/aria-checked/data-state)
        # before and after, with an 800ms settle. If none of the four
        # signals changed, the click almost certainly missed — surface
        # this as an explicit natural-language nudge so the LLM doesn't
        # report "done" on a phantom click.
        if result.get("ok") and not result.get("changed"):
            result["note"] = (
                "The click executed but nothing changed — url, title, "
                "heading and the element at the click coordinates are all "
                "identical before and after. The click target was probably "
                "not clickable, the coordinates missed, or the element was "
                "obscured by an overlay. Do NOT tell the user the action "
                "succeeded. Instead, re-run web_dom_snapshot and pick a "
                "different target \u2014 when multiple elements share the "
                "same text, prefer the one with the larger w\u00d7h (CTA "
                "buttons are typically >300\u00d750px; menu items are "
                "smaller). For radio buttons / checkboxes, make sure you "
                "are clicking the visible indicator (○ / □), not the "
                "label text next to it."
            )
        return _fmt(result)
    except Exception as e:
        logger.exception("[web_exec] click failed")
        return _err("web_click", str(e))


# ─── 7 · type ───────────────────────────────────────────────────────

async def execute_web_type(params: Dict[str, Any]) -> str:
    try:
        result = await type_handler(
            {
                "session": params.get("session", ""),
                "selector": params.get("selector", ""),
                "text": params.get("text", ""),
                "clear_first": params.get("clear_first", True),
            },
            None,
        )
        return _fmt(result)
    except Exception as e:
        logger.exception("[web_exec] type failed")
        return _err("web_type", str(e))


# ─── 8 · scroll ─────────────────────────────────────────────────────

async def execute_web_scroll(params: Dict[str, Any]) -> str:
    try:
        result = await scroll_handler(
            {
                "session": params.get("session", ""),
                "direction": params.get("direction", "down"),
                "amount": params.get("amount", 500),
            },
            None,
        )
        return _fmt(result)
    except Exception as e:
        logger.exception("[web_exec] scroll failed")
        return _err("web_scroll", str(e))


# ─── 9 · extract text ───────────────────────────────────────────────

async def execute_web_extract_text(params: Dict[str, Any]) -> str:
    try:
        result = await extract_text_handler(
            {
                "session": params.get("session", ""),
                "selector": params.get("selector", ""),
                "limit": params.get("limit", 20),
            },
            None,
        )
        return _fmt(result)
    except Exception as e:
        logger.exception("[web_exec] extract_text failed")
        return _err("web_extract_text", str(e))


# ─── 10 · screenshot ────────────────────────────────────────────────

async def execute_web_screenshot(params: Dict[str, Any]) -> str:
    try:
        result = await screenshot_handler(
            {
                "session": params.get("session", ""),
                "full_page": params.get("full_page", False),
            },
            None,
        )
        # Drop the base64 payload from the string we return to the LLM.
        # The saved file path is still there if it needs to reference
        # the image; including the base64 would blow the context.
        if result.get("image_base64"):
            result["image_base64"] = f"<{len(result['image_base64'])} chars, omitted>"
        return _fmt(result)
    except Exception as e:
        logger.exception("[web_exec] screenshot failed")
        return _err("web_screenshot", str(e))


# ─── 11 · vision check ───────────────────────────────

async def execute_web_vision_check(params: Dict[str, Any]) -> str:
    """
    Screenshot the session's current page and ask a vision model a
    natural-language question about what's visible. Returns the model's
    answer as a string — no image data goes back to the chat LLM.

    The chat LLM can't act on a raw image and screenshots blow context;
    the point of this tool is to get a text answer the LLM can reason
    with ("yes, the post text reads: 'Hey, I'm Astra...'" or "the
    compose box is empty, nothing was typed").
    """
    try:
        result = await vision_check_handler(
            {
                "session":  params.get("session", ""),
                "question": params.get("question", ""),
            },
            None,
        )
        return _fmt(result)
    except Exception as e:
        logger.exception("[web_exec] vision_check failed")
        return _err("web_vision_check", str(e))


# ─── 12 · upload file ───────────────────────────────

async def execute_web_upload_file(params: Dict[str, Any]) -> str:
    """
    Attach a local file to a web upload control. Bridges chat tool calls
    to upload_file_handler with the right shape.

    The chat LLM passes one of two shapes:
      * `{ session, file_path, selector }`   — plain HTML form
      * `{ session, file_path, x, y }`       — SPA upload button (Meta, IG…)

    The (x, y) path uses Page.setInterceptFileChooserDialog under the hood
    so the native OS file dialog is suppressed and the file is injected
    directly. This is the only reliable way to upload on modern social
    platforms where the underlying <input> is hidden behind a styled button.
    """
    try:
        forwarded: Dict[str, Any] = {
            "session":   params.get("session", ""),
            "file_path": params.get("file_path", ""),
        }
        if params.get("selector"):
            forwarded["selector"] = params["selector"]
        if "x" in params:
            forwarded["x"] = params["x"]
        if "y" in params:
            forwarded["y"] = params["y"]
        result = await upload_file_handler(forwarded, None)
        return _fmt(result)
    except Exception as e:
        logger.exception("[web_exec] upload_file failed")
        return _err("web_upload_file", str(e))


# ─── 13 · system keys (drive native dialogs) ──────────────────────

async def execute_system_keys(params: Dict[str, Any]) -> str:
    """
    Send keystrokes to whatever native OS window has focus.

    Use case: a web_click in the embedded webview opened a NATIVE file
    Open dialog (most often Meta / Instagram / TikTok / WordPress
    "Upload from desktop" when CDP intercept misses). The dialog
    auto-focuses, so SendKeys lands on it. Pass an absolute path as
    `text` and `press_enter_after=true` to type the path and submit.
    Windows accepts absolute paths in the filename field, bypassing
    folder navigation entirely.
    """
    try:
        forwarded: Dict[str, Any] = {
            "session": params.get("session", ""),
            "text":    params.get("text", ""),
        }
        if "press_enter_after" in params:
            forwarded["press_enter_after"] = params["press_enter_after"]
        if "pre_delay_ms" in params:
            forwarded["pre_delay_ms"] = params["pre_delay_ms"]
        result = await system_keys_handler(forwarded, None)
        return _fmt(result)
    except Exception as e:
        logger.exception("[web_exec] system_keys failed")
        return _err("system_keys", str(e))
