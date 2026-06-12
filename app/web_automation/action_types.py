# FILE: app/web_automation/action_types.py
# Purpose: Primitive action type constants + tiny typed constructors.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Primitive action type constants + tiny typed constructors.

Kept as a module of plain functions (not Pydantic/dataclass) because the
payload gets serialised to JSON and sent to Electron — one shape, no
ceremony. Every constructor returns (action_type, payload_dict).

These are PRIMITIVES. Composite flows (e.g. post_to_facebook_page)
live elsewhere and orchestrate these.
"""
from __future__ import annotations

from typing import Literal, Optional


# Action type string constants. Electron executor implements each.
NAVIGATE      = "navigate"
CLICK         = "click"
TYPE_TEXT     = "type"
PRESS_KEY     = "press_key"
SCROLL        = "scroll"
WAIT          = "wait"
SCREENSHOT    = "screenshot"
DOM_SNAPSHOT  = "dom_snapshot"      # Accessibility tree
EXTRACT_TEXT  = "extract_text"      # CSS-selector-based text extraction
UPLOAD_FILE   = "upload_file"
CURRENT_STATE = "current_state"     # URL + title
RELOAD        = "reload"
GO_BACK       = "go_back"
CLOSE         = "close"             # Close the session's view
SYSTEM_KEYS   = "system_keys"       # Send keystrokes to focused OS window


ALL_ACTIONS = {
    NAVIGATE, CLICK, TYPE_TEXT, PRESS_KEY, SCROLL, WAIT,
    SCREENSHOT, DOM_SNAPSHOT, EXTRACT_TEXT, UPLOAD_FILE,
    CURRENT_STATE, RELOAD, GO_BACK, CLOSE, SYSTEM_KEYS,
}


def navigate(url: Optional[str] = None) -> tuple[str, dict]:
    """url=None means 'use the session's configured landing_url'."""
    return NAVIGATE, {"url": url}


def click(selector: Optional[str] = None,
          x: Optional[int] = None,
          y: Optional[int] = None) -> tuple[str, dict]:
    if selector:
        return CLICK, {"selector": selector}
    if x is not None and y is not None:
        return CLICK, {"x": x, "y": y}
    raise ValueError("click requires either selector or (x, y)")


def type_text(selector: str, text: str, clear_first: bool = True) -> tuple[str, dict]:
    return TYPE_TEXT, {"selector": selector, "text": text, "clear_first": clear_first}


def press_key(key: str) -> tuple[str, dict]:
    """Key names match Electron webContents.sendInputEvent (e.g. 'Enter', 'Tab')."""
    return PRESS_KEY, {"key": key}


def scroll(direction: Literal["up", "down", "top", "bottom"] = "down",
           amount: int = 500) -> tuple[str, dict]:
    return SCROLL, {"direction": direction, "amount": amount}


def wait(ms: int) -> tuple[str, dict]:
    return WAIT, {"ms": ms}


def screenshot(full_page: bool = False) -> tuple[str, dict]:
    return SCREENSHOT, {"full_page": full_page}


def dom_snapshot() -> tuple[str, dict]:
    return DOM_SNAPSHOT, {}


def extract_text(selector: str, limit: int = 20) -> tuple[str, dict]:
    return EXTRACT_TEXT, {"selector": selector, "limit": limit}


def upload_file(
    selector: Optional[str] = None,
    file_path: Optional[str] = None,
    *,
    x: Optional[int] = None,
    y: Optional[int] = None,
) -> tuple[str, dict]:
    """Attach a local file to a web upload control.

    Two modes — exactly one is required:
      * `selector` — CSS selector for an addressable <input type="file">.
        Used by plain HTML forms.
      * `(x, y)`  — keyword-only. Pixel coordinates of the upload BUTTON.
        Used by modern SPAs (Meta, IG, TikTok, YouTube…) where the input
        is hidden. Electron drives a CDP file-chooser interception path:
        suppress the OS dialog, click at (x, y), respond to the resulting
        Page.fileChooserOpened event with the file. This is the only
        reliable way to upload on those platforms.

    Backward-compat note: legacy positional form `upload_file(selector, file_path)`
    still works. New callers should prefer the keyword form
    `upload_file(file_path=..., x=..., y=...)` for click mode.
    """
    if not file_path:
        raise ValueError("upload_file requires file_path")
    payload: dict = {"file_path": file_path}
    if selector:
        payload["selector"] = selector
    elif x is not None and y is not None:
        payload["x"] = int(x)
        payload["y"] = int(y)
    else:
        raise ValueError("upload_file requires either `selector` or both `x` and `y`")
    return UPLOAD_FILE, payload


def current_state() -> tuple[str, dict]:
    return CURRENT_STATE, {}


def reload() -> tuple[str, dict]:
    return RELOAD, {}


def go_back() -> tuple[str, dict]:
    return GO_BACK, {}


def close() -> tuple[str, dict]:
    return CLOSE, {}


def system_keys(
    text: str,
    *,
    press_enter_after: bool = False,
    pre_delay_ms: int = 600,
) -> tuple[str, dict]:
    """Send keystrokes to whatever native OS window currently has focus.

    Use case: a click in the embedded webview opened a NATIVE OS dialog
    (Windows file Open dialog from Meta's "Upload from desktop", etc.).
    The CDP path can't see native dialogs, but the dialog auto-focuses
    when it opens, so SendKeys lands on it. Type the file path + Enter
    and you're done — no folder navigation required because Windows
    accepts an absolute path in the filename field.

    Args:
      text: literal text to type. Path strings work as-is.
      press_enter_after: append a press of the Enter key after the text
        is typed. True for file dialog submission.
      pre_delay_ms: how long to wait before the first keystroke. Lets
        the native dialog finish appearing and grab focus. 600ms covers
        almost all cases; bump for slow systems.

    Currently Windows-only (relies on PowerShell + WScript.Shell).
    """
    if not text:
        raise ValueError("system_keys requires non-empty text")
    return SYSTEM_KEYS, {
        "text": text,
        "press_enter_after": bool(press_enter_after),
        "pre_delay_ms": int(pre_delay_ms),
    }
