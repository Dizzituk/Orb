# FILE: app/web_automation/tool_schemas.py
# Purpose: JSONSchema for ASTRA LLM-callable web-automation tools.
# Called-by: app.web_automation.register
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
JSONSchema for ASTRA LLM-callable web-automation tools.

These schemas are passed to the LLM so it knows what tools exist and
how to call them. Shape matches the existing app.tools.schemas module.
"""
from __future__ import annotations


# `session` accepts either a UUID session id or a platform key.
# We intentionally keep it as a plain string in the schema (for strict
# LLM tool-call validators) and resolve at the handler layer.
_SESSION_FIELD = {
    "type": "string",
    "description": "Either the session UUID or a platform key like 'meta_business', 'coursera', 'tiktok_astraukai'.",
    "minLength": 1,
    "maxLength": 128,
}


# ── web_list_sessions ────────────────────────────────────────────────

WEB_LIST_SESSIONS = {
    "input": {
        "type": "object",
        "properties": {},
    },
    "output": {
        "type": "object",
        "required": ["sessions"],
        "properties": {
            "sessions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "platform", "label", "status"],
                    "properties": {
                        "id": {"type": "string"},
                        "platform": {"type": "string"},
                        "label": {"type": "string"},
                        "purpose": {"type": "string"},
                        "status": {"type": "string"},
                        "current_url": {"type": "string"},
                    },
                },
            },
        },
    },
}


# ── web_open_session ─────────────────────────────────────────────────

WEB_OPEN_SESSION = {
    "input": {
        "type": "object",
        "required": ["session"],
        "properties": {"session": _SESSION_FIELD},
    },
    "output": {
        "type": "object",
        "required": ["ok"],
        "properties": {
            "ok": {"type": "boolean"},
            "session_id": {"type": "string"},
            "current_url": {"type": "string"},
            "error": {"type": "string"},
        },
    },
}


# ── web_navigate ─────────────────────────────────────────────────────

WEB_NAVIGATE = {
    "input": {
        "type": "object",
        "required": ["session", "url"],
        "properties": {
            "session": _SESSION_FIELD,
            "url": {"type": "string", "minLength": 1, "maxLength": 2048},
        },
    },
    "output": {
        "type": "object",
        "required": ["ok"],
        "properties": {
            "ok": {"type": "boolean"},
            "current_url": {"type": "string"},
            "current_title": {"type": "string"},
            "error": {"type": "string"},
        },
    },
}


# ── web_click ────────────────────────────────────────────────────────

WEB_CLICK = {
    "input": {
        "type": "object",
        "required": ["session"],
        "properties": {
            "session": _SESSION_FIELD,
            "role": {
                "type": "string",
                "description": "ARIA role of the target (e.g. 'button', 'link', 'tab') — use with `name`. Resolved inside the page at click time.",
            },
            "name": {
                "type": "string",
                "description": "Accessible name of the target: its aria-label or visible text exactly as web_dom_snapshot reported it. Preferred targeting mode (with `role`).",
            },
            "selector": {"type": "string", "description": "CSS selector of the element to click."},
            "x": {"type": "integer", "description": "X coordinate (px) — only from a snapshot taken immediately before, with nothing in between."},
            "y": {"type": "integer", "description": "Y coordinate (px) — only from a snapshot taken immediately before, with nothing in between."},
        },
    },
    "output": {
        "type": "object",
        "required": ["ok"],
        "properties": {
            "ok": {"type": "boolean"},
            "error": {"type": "string"},
        },
    },
}


# ── web_type ─────────────────────────────────────────────────────────

WEB_TYPE = {
    "input": {
        "type": "object",
        "required": ["session", "selector", "text"],
        "properties": {
            "session": _SESSION_FIELD,
            "selector": {"type": "string"},
            "text": {"type": "string"},
            "clear_first": {"type": "boolean", "description": "Clear the field before typing (default true)."},
        },
    },
    "output": {
        "type": "object",
        "required": ["ok"],
        "properties": {
            "ok": {"type": "boolean"},
            "error": {"type": "string"},
        },
    },
}


# ── web_screenshot ───────────────────────────────────────────────────

WEB_SCREENSHOT = {
    "input": {
        "type": "object",
        "required": ["session"],
        "properties": {
            "session": _SESSION_FIELD,
            "full_page": {"type": "boolean", "description": "Capture the full scrollable page (default false)."},
        },
    },
    "output": {
        "type": "object",
        "required": ["ok"],
        "properties": {
            "ok": {"type": "boolean"},
            "image_base64": {"type": "string", "description": "Base64-encoded PNG."},
            "image_path": {"type": "string", "description": "Filesystem path to the saved PNG (if Electron saved it)."},
            "error": {"type": "string"},
        },
    },
}


# ── web_extract_text ─────────────────────────────────────────────────

WEB_EXTRACT_TEXT = {
    "input": {
        "type": "object",
        "required": ["session", "selector"],
        "properties": {
            "session": _SESSION_FIELD,
            "selector": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 500},
        },
    },
    "output": {
        "type": "object",
        "required": ["ok"],
        "properties": {
            "ok": {"type": "boolean"},
            "matches": {"type": "array", "items": {"type": "string"}},
            "error": {"type": "string"},
        },
    },
}


# ── web_current_state ────────────────────────────────────────────────

WEB_CURRENT_STATE = {
    "input": {
        "type": "object",
        "required": ["session"],
        "properties": {"session": _SESSION_FIELD},
    },
    "output": {
        "type": "object",
        "required": ["ok"],
        "properties": {
            "ok": {"type": "boolean"},
            "url": {"type": "string"},
            "title": {"type": "string"},
            "error": {"type": "string"},
        },
    },
}


# ── web_scroll ───────────────────────────────────────────────────────

WEB_SCROLL = {
    "input": {
        "type": "object",
        "required": ["session", "direction"],
        "properties": {
            "session": _SESSION_FIELD,
            "direction": {"type": "string", "enum": ["up", "down", "top", "bottom"]},
            "amount": {"type": "integer", "minimum": 1, "maximum": 20000},
        },
    },
    "output": {
        "type": "object",
        "required": ["ok"],
        "properties": {
            "ok": {"type": "boolean"},
            "error": {"type": "string"},
        },
    },
}


# ── web_wait_for ─────────────────────────────────────────────────────

# Condition wait — poll for an element / text / URL instead of sleeping
# a fixed time. A timeout comes back ok=true + timeout=true (NOT an
# error): the caller decides whether the miss is fatal.
WEB_WAIT_FOR = {
    "input": {
        "type": "object",
        "required": ["session"],
        "properties": {
            "session": _SESSION_FIELD,
            "selector": {
                "type": "string",
                "description": "CSS selector to wait for.",
            },
            "text": {
                "type": "string",
                "description": "Wait for this text on the page (or inside `selector` matches if both given).",
            },
            "url_pattern": {
                "type": "string",
                "description": "Regex tested against the current URL.",
            },
            "state": {
                "type": "string",
                "enum": ["visible", "attached", "gone"],
                "description": "visible (default) = exists with non-zero size; attached = present in DOM; gone = no longer present (spinners, modals).",
            },
            "timeout_ms": {"type": "integer", "minimum": 0, "maximum": 60000},
            "poll_ms": {"type": "integer", "minimum": 50, "maximum": 2000},
        },
    },
    "output": {
        "type": "object",
        "required": ["ok"],
        "properties": {
            "ok": {"type": "boolean"},
            "matched": {"type": "boolean"},
            "timeout": {"type": "boolean"},
            "waited_ms": {"type": "integer"},
            "error": {"type": "string"},
        },
    },
}


# ── web_dom_snapshot ─────────────────────────────────────────────

# Returns the page's accessibility tree as structured JSON — every
# interactive element (links, buttons, headings) with its text, role,
# href, and pixel coordinates. Strongly preferred over CSS selectors
# for modern React apps (Coursera, Meta, TikTok) where class names are
# obfuscated and change between deploys.
WEB_DOM_SNAPSHOT = {
    "input": {
        "type": "object",
        "required": ["session"],
        "properties": {"session": _SESSION_FIELD},
    },
    "output": {
        "type": "object",
        "required": ["ok"],
        "properties": {
            "ok":       {"type": "boolean"},
            "elements": {"type": "array"},
            "error":    {"type": "string"},
        },
    },
}


# ── web_vision_check ───────────────────────────────

# Screenshot + vision-model Q&A. Reserved for high-stakes visual
# checks where cheap DOM/element verify isn't enough (pre-submit
# read-back, post-submit confirmation, error banner detection).
WEB_VISION_CHECK = {
    "input": {
        "type": "object",
        "required": ["session", "question"],
        "properties": {
            "session":  _SESSION_FIELD,
            "question": {
                "type": "string",
                "minLength": 1,
                "maxLength": 2000,
                "description": "Natural-language question to ask the vision model about the current page.",
            },
        },
    },
    "output": {
        "type": "object",
        "required": ["ok"],
        "properties": {
            "ok":              {"type": "boolean"},
            "answer":          {"type": "string"},
            "provider":        {"type": "string"},
            "model":           {"type": "string"},
            "screenshot_path": {"type": "string"},
            "error":           {"type": "string"},
        },
    },
}


# ── web_upload_file ──────────────────────────────

# Two modes — selector for plain HTML, (x, y) for SPAs (Meta, IG, etc).
# See upload_file() in app/web_automation/action_types.py for full notes.
WEB_UPLOAD_FILE = {
    "input": {
        "type": "object",
        "required": ["session", "file_path"],
        "properties": {
            "session":   _SESSION_FIELD,
            "file_path": {
                "type": "string",
                "description": "Absolute filesystem path to the file to attach.",
            },
            "selector": {
                "type": "string",
                "description": "CSS selector of an <input type='file'>. Plain HTML forms only.",
            },
            "x": {
                "type": "integer",
                "description": "X coordinate (px) of the upload BUTTON — use with `y` for SPAs.",
            },
            "y": {
                "type": "integer",
                "description": "Y coordinate (px) of the upload BUTTON — use with `x` for SPAs.",
            },
        },
    },
    "output": {
        "type": "object",
        "required": ["ok"],
        "properties": {
            "ok":       {"type": "boolean"},
            "mode":     {"type": "string", "enum": ["selector", "click"]},
            "attached": {"type": "string"},
            "error":    {"type": "string"},
        },
    },
}


# ── system_keys ─────────────────────────────────

# Sends keystrokes to whatever native OS window currently has focus.
# Used to drive native file Open dialogs that pop up from Meta /
# Instagram / TikTok / etc. when CDP file-chooser interception fails.
SYSTEM_KEYS_SCHEMA = {
    "input": {
        "type": "object",
        "required": ["session", "text"],
        "properties": {
            "session":  _SESSION_FIELD,
            "text": {
                "type": "string",
                "description": "Literal text to type, e.g. an absolute file path.",
            },
            "press_enter_after": {
                "type": "boolean",
                "description": "Press Enter after typing the text. True for file-dialog submission.",
            },
            "pre_delay_ms": {
                "type": "integer",
                "description": "Wait this many ms before sending keys (default 600).",
            },
        },
    },
    "output": {
        "type": "object",
        "required": ["ok"],
        "properties": {
            "ok":            {"type": "boolean"},
            "sent":          {"type": "string"},
            "pre_delay_ms":  {"type": "integer"},
            "pressed_enter": {"type": "boolean"},
            "error":         {"type": "string"},
        },
    },
}


TOOL_SCHEMAS = {
    "web_list_sessions":  WEB_LIST_SESSIONS,
    "web_open_session":   WEB_OPEN_SESSION,
    "web_navigate":       WEB_NAVIGATE,
    "web_click":          WEB_CLICK,
    "web_type":           WEB_TYPE,
    "web_screenshot":     WEB_SCREENSHOT,
    "web_extract_text":   WEB_EXTRACT_TEXT,
    "web_current_state":  WEB_CURRENT_STATE,
    "web_scroll":         WEB_SCROLL,
    "web_wait_for":       WEB_WAIT_FOR,
    "web_dom_snapshot":   WEB_DOM_SNAPSHOT,
    "web_vision_check":   WEB_VISION_CHECK,
    "web_upload_file":    WEB_UPLOAD_FILE,
    "system_keys":        SYSTEM_KEYS_SCHEMA,
}
