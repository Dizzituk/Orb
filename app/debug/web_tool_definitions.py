# FILE: app/debug/web_tool_definitions.py
# Purpose: Chat-facing tool definitions for web browsing.
# Called-by: app.debug.tool_definitions
# Depends-on: app.debug.web_tool_playbooks, app.debug.web_upload_tool_definitions, app.web_automation.coursera_reader, app.web_automation.coursera_study_tools
# Last-renovated: 2026-07-02
"""
Chat-facing tool definitions for web browsing.

Gives the chat LLM the ability to drive logged-in browser sessions
(Coursera, Meta Business Suite, TikTok Studio, YouTube Studio,
WordPress) through natural language. Each tool definition maps 1:1 to
a handler in app/debug/executors/web_automation.py, which in turn
wraps the underlying app/web_automation primitives.

Why a parallel set of definitions? The chat loop pulls tool defs
from app/debug/tool_definitions.py in a specific "parameters" shape,
while app/web_automation/tool_schemas.py uses "input_schema" for its
API consumers. Keeping the two shapes separate and translating at the
boundary is simpler than trying to unify them.

Reliability policy (2026-07-02, Amendment A Job 9): DOM-first with
resolve-at-click-time targeting. Preferred action order is role+name
from the snapshot, then stable CSS, then fresh-snapshot coordinates,
then vision+snap as last resort; web_wait_for gates every navigation
and state-changing click. Upload + native-dialog defs live in
web_upload_tool_definitions.py (size cap).
"""
from __future__ import annotations

from typing import List

from app.debug.web_tool_playbooks import (
    CLICK_RESULT_INTERPRETATION,
    POST_SUBMIT_MODAL_RECOVERY,
    TEXT_INPUT_TARGETING,
)
# Long-prose upload defs live in their own module (file-size cap).
from app.debug.web_upload_tool_definitions import (
    SYSTEM_KEYS_TOOL,
    WEB_UPLOAD_FILE_TOOL,
)
# Coursera composites keep their full chat tool defs next to their logic
# in coursera_reader.py / coursera_study_tools.py; import rather than
# grow this file (near the cap).
from app.web_automation.coursera_reader import (
    WEB_COURSERA_HEALTH_TOOL,
    WEB_COURSERA_PROGRESS_TOOL,
)
from app.web_automation.coursera_study_tools import (
    COURSERA_NEXT_ITEM_TOOL,
    COURSERA_READ_LESSON_TOOL,
    COURSERA_RESUME_TOOL,
)


_SESSION_DESC = (
    "Platform key of the target session. Common values: 'coursera', "
    "'meta_business', 'tiktok_astraukai', 'youtube_studio', "
    "'wordpress_admin'. Call web_list_sessions if unsure."
)


WEB_LIST_SESSIONS_TOOL = {
    "name": "web_list_sessions",
    "description": (
        "List all logged-in browser sessions the user has registered in ASTRA. "
        "Returns each session's platform key, label, purpose, current URL, and "
        "live status. Call this FIRST when the user asks about their browser "
        "sessions ('what tabs do I have', 'is Coursera open') or when you need "
        "to know the right platform key before calling other web_* tools."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

WEB_OPEN_SESSION_TOOL = {
    "name": "web_open_session",
    "description": (
        "Make sure a session's browser view is live (Electron spawns it if "
        "not already). Call this before navigate/click/etc on a session "
        "that may not be open yet. The user will SEE the browser appear "
        "in the Browser tab. Quick (1-2s). ERROR MEANINGS: any web tool "
        "failing with 'desktop browser is offline' means the ASTRA desktop "
        "app is NOT RUNNING — tell the user to start the desktop app; do "
        "not retry or blame the page. 'page did not respond' means the "
        "desktop IS up but the page stalled — that one is worth a retry. "
        "'browser is busy on a previous action' means the desktop is up and "
        "mid-action — wait a few seconds and retry; never report it as "
        "offline."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session": {"type": "string", "description": _SESSION_DESC},
        },
        "required": ["session"],
    },
}

WEB_NAVIGATE_TOOL = {
    "name": "web_navigate",
    "description": (
        "Navigate a browser session to a specific URL. Use when you know "
        "exactly where to go, e.g. 'https://www.coursera.org/learn/<course>'. "
        "Returns after the page's load event fires; SPA content often "
        "renders AFTER that — follow with web_wait_for on a known element "
        "(or url_pattern) before the next read or action, then "
        "web_dom_snapshot to see where you landed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session": {"type": "string", "description": _SESSION_DESC},
            "url": {"type": "string", "description": "Full URL including https://"},
        },
        "required": ["session", "url"],
    },
}

WEB_CURRENT_STATE_TOOL = {
    "name": "web_current_state",
    "description": (
        "Cheap (<1s) check of a session's current URL and page title. Use "
        "to confirm a navigation landed correctly, or to see where a redirect "
        "went. Does not return page content — use web_dom_snapshot for that."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session": {"type": "string", "description": _SESSION_DESC},
        },
        "required": ["session"],
    },
}

WEB_DOM_SNAPSHOT_TOOL = {
    "name": "web_dom_snapshot",
    "description": (
        "Return the page's accessibility tree: every interactive element "
        "(links, buttons, headings, form fields) with role, name, text "
        "content, href target, and pixel coordinates + size (x, y, w, h). "
        "This is the PRIMARY way to understand what's on a page. Act on "
        "what you find with web_click: pass role + name copied straight "
        "from the element (BEST — re-resolved inside the live page at "
        "click time), or a stable CSS selector when one exists. "
        "Coordinates are valid only immediately after this snapshot with "
        "NOTHING in between — SPA re-renders, lazy-loads and scrolls move "
        "elements, and stale coords click whatever sits there now. Works "
        "reliably on React apps like Coursera where class names are "
        "obfuscated — but it carries NO visual done-state: Coursera "
        "completion ticks / progress bars look identical for finished and "
        "unfinished modules in the tree, so for course progress questions "
        "use web_coursera_progress instead. When multiple elements share "
        "the same text (common for 'Get started', 'Sign in', 'Post' etc.), "
        "prefer the one with the LARGEST w×h — CTA buttons are typically "
        "300-500px wide × 48-64px tall, while menu items are smaller. "
        "Large pages return 50-200+ elements; filter mentally for what "
        "you need rather than dumping everything to the user."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session": {"type": "string", "description": _SESSION_DESC},
        },
        "required": ["session"],
    },
}

WEB_EXTRACT_TEXT_TOOL = {
    "name": "web_extract_text",
    "description": (
        "Return text content of all elements matching a CSS selector. Use "
        "this only when you already know the exact selector (from an earlier "
        "web_dom_snapshot, or well-known markup like 'h1', 'article p'). "
        "For most cases prefer web_dom_snapshot which doesn't require guessing "
        "selectors. Useful for: reading a full article body once you know "
        "its wrapper, dumping all list items of a known container. "
        "TEXT ONLY — visual state (Coursera completion ticks, progress bars, "
        "toggle positions) does NOT come back: module titles read identically "
        "whether done or not. For course progress/position questions use "
        "web_coursera_progress (or web_vision_check), never this."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session": {"type": "string", "description": _SESSION_DESC},
            "selector": {"type": "string", "description": "CSS selector to match."},
            "limit": {"type": "integer", "description": "Max results (default 20)."},
        },
        "required": ["session", "selector"],
    },
}

WEB_CLICK_TOOL = {
    "name": "web_click",
    "description": (
        "Click an element. TARGETING — in order of preference:"
        "\n\n"
        "1. role + name (BEST): copy `role` and `name` (aria-label or "
        "visible text) straight from an element in the latest "
        "web_dom_snapshot. The click re-resolves the element INSIDE the "
        "live page at click time — by role, aria-label and visible text, "
        "the same fields the snapshot reports — so it still lands after "
        "SPA re-renders, lazy-loads and scrolls. Matching is "
        "case-insensitive; exact names beat substrings; on ties the "
        "larger element wins. The result's `resolved` field shows exactly "
        "which element was hit."
        "\n\n"
        "2. CSS `selector` — when a stable selector exists (aria-label "
        "attributes, IDs, input[type=file]). Never guess obfuscated / "
        "hashed class names."
        "\n\n"
        "3. (x, y) coordinates — ONLY from a web_dom_snapshot taken "
        "immediately before this click with NOTHING in between. "
        "Coordinates go stale fast: SPA re-renders, lazy-loading and "
        "scrolling all move elements, and a click at old coordinates hits "
        "whatever sits there NOW. If anything happened since the snapshot "
        "(a click, a wait, a scroll, late rendering), snapshot again "
        "first."
        "\n\n"
        "4. Vision-derived coords + snap_to_button=true — LAST resort, for "
        "targets the DOM tree cannot disambiguate. Vision is approximate "
        "(±50 px); snap walks from the vision pixel to the nearest "
        "interactive element (expanding rings up to 150 px) and clicks ITS "
        "centre. The `snapped` result field shows what was actually hit "
        "(tag, text, w, h, search_distance). Do NOT pass snap_to_button "
        "for role/name, selector, or dom_snapshot-coordinate clicks."
        "\n\n"
        + TEXT_INPUT_TARGETING
        + "\n\n"
        + CLICK_RESULT_INTERPRETATION
        + "\n\n"
        + POST_SUBMIT_MODAL_RECOVERY
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session":  {"type": "string", "description": _SESSION_DESC},
            "role":     {"type": "string", "description": "ARIA role from the snapshot ('button', 'link', 'tab', 'menuitem', ...). Use with `name`."},
            "name":     {"type": "string", "description": "Accessible name: the element's aria-label or visible text as shown in the snapshot. Preferred targeting (with `role`)."},
            "selector": {"type": "string", "description": "CSS selector of element to click."},
            "x":        {"type": "integer", "description": "X coordinate in pixels — only from a snapshot taken immediately before this click, nothing in between."},
            "y":        {"type": "integer", "description": "Y coordinate in pixels — only from a snapshot taken immediately before this click, nothing in between."},
            "snap_to_button": {
                "type": "boolean",
                "description": (
                    "Snap (x, y) to the nearest interactive element's centre "
                    "before clicking. Set TRUE only when coords came from "
                    "web_vision_check (vision is ~50px imprecise). Leave "
                    "FALSE / unset for every other targeting mode."
                ),
            },
        },
        "required": ["session"],
    },
}

WEB_WAIT_FOR_TOOL = {
    "name": "web_wait_for",
    "description": (
        "Wait until the page is actually ready instead of guessing. Polls "
        "for a CSS `selector` (with `state`: visible / attached / gone), a "
        "`text` snippet, and/or a `url_pattern` regex; returns as soon as "
        "the condition holds. A timeout is NOT an error: you get ok=true, "
        "matched=false, timeout=true — decide yourself whether that's "
        "fatal. WHEN TO USE (mandatory habit): after EVERY web_navigate; "
        "after any click that changes page state (opens a composer, modal, "
        "menu or new route); and after upload/attach steps — always BEFORE "
        "the next read or action. Use state='gone' to wait for spinners / "
        "overlays to clear. ON TIMEOUT: do NOT blindly retry the previous "
        "click — take a fresh web_dom_snapshot, re-orient, then act. Cheap "
        "(a DOM poll every 250ms), so prefer it over any fixed pause."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session":     {"type": "string", "description": _SESSION_DESC},
            "selector":    {"type": "string", "description": "CSS selector to wait for."},
            "text":        {"type": "string", "description": "Wait for this text on the page (or inside selector matches if both given)."},
            "url_pattern": {"type": "string", "description": "Regex tested against the current URL."},
            "state": {
                "type": "string",
                "enum": ["visible", "attached", "gone"],
                "description": (
                    "visible (default) = exists with non-zero size; attached "
                    "= present in DOM; gone = no longer present (spinners, "
                    "modals)."
                ),
            },
            "timeout_ms": {"type": "integer", "description": "Max wait in ms (default 15000, cap 60000)."},
        },
        "required": ["session"],
    },
}

WEB_TYPE_TOOL = {
    "name": "web_type",
    "description": (
        "Type text into an input/textarea matching the CSS selector. Field "
        "is cleared first by default. Use for search boxes, form inputs, "
        "composing posts. For sensitive fields (passwords), prefer the user "
        "types manually in the browser view."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session":     {"type": "string", "description": _SESSION_DESC},
            "selector":    {"type": "string", "description": "CSS selector of input field."},
            "text":        {"type": "string", "description": "Text to type."},
            "clear_first": {"type": "boolean", "description": "Clear field first (default true)."},
        },
        "required": ["session", "selector", "text"],
    },
}

WEB_SCROLL_TOOL = {
    "name": "web_scroll",
    "description": (
        "Scroll the page. Directions: 'up' / 'down' scroll by `amount` "
        "pixels (default 500); 'top' / 'bottom' jump to that edge. Useful "
        "when the content you need is below the fold, or to trigger "
        "lazy-loaded elements (infinite-scroll feeds, chart dashboards). "
        "Scrolling moves elements: any coordinates from an earlier "
        "snapshot are stale afterwards — snapshot again, or click by "
        "role+name which re-resolves at click time."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session":   {"type": "string", "description": _SESSION_DESC},
            "direction": {"type": "string", "enum": ["up", "down", "top", "bottom"]},
            "amount":    {"type": "integer", "description": "Scroll amount in pixels (default 500)."},
        },
        "required": ["session", "direction"],
    },
}

WEB_SCREENSHOT_TOOL = {
    "name": "web_screenshot",
    "description": (
        "Capture a PNG screenshot of a session's current page. Returns a "
        "file path the user can view. Use when the user asks to see what's "
        "on screen, or when you need to show them what you found. Do NOT "
        "use as a substitute for web_dom_snapshot when reading content — "
        "screenshots are bigger and slower to generate."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session":   {"type": "string", "description": _SESSION_DESC},
            "full_page": {"type": "boolean", "description": "Capture the full scrollable page (default false)."},
        },
        "required": ["session"],
    },
}

WEB_VISION_CHECK_TOOL = {
    "name": "web_vision_check",
    "description": (
        "Screenshot the live page and ask a vision model a natural-language "
        "question about what's visible. Returns the vision model's answer "
        "as text (no image data goes back to you)."
        "\n\n"
        "WHEN TO USE IT: the DOM checks in web_click / web_dom_snapshot "
        "catch almost all failures cheaply, so vision_check is reserved "
        "for high-stakes, irreversible, or visual-only verification:"
        "\n"
        "  • Before submitting a post / publishing / sending: "
        "'does the compose box contain exactly the text I typed?'"
        "\n"
        "  • After submitting: 'is the post now visible on the wall / feed?'"
        "\n"
        "  • Error detection: 'is there a red error banner anywhere on the page?'"
        "\n"
        "  • Layout/visual cross-check the DOM can't express: "
        "'which radio option in question 3 is currently selected?'"
        "\n"
        "  • READING VISUAL COMPLETION STATE: course progress ticks and "
        "progress bars (Coursera module done-state) exist only as pixels — "
        "extract_text/dom_snapshot cannot see them. For Coursera progress "
        "specifically, prefer the web_coursera_progress composite, which "
        "also verifies the session is actually logged in first."
        "\n"
        "  • FINDING THE FILE-UPLOAD BUTTON on a social-media composer "
        "(Meta / Instagram / TikTok / YouTube / WordPress). The DOM "
        "accessibility tree returns dozens of generic 'button' elements "
        "in the photo-picker modal — Recents thumbnails, tab switchers, "
        "close buttons, the actual upload trigger — and they look "
        "identical from the tree. Vision can disambiguate. ALWAYS "
        "vision_check before web_upload_file's (x, y) on these "
        "platforms. See the worked example in web_upload_file's docs."
        "\n\n"
        "HOW TO ASK: be specific and target the thing you actually care "
        "about, not 'describe the page'. Good: 'what is the exact text "
        "currently in the Facebook compose box?'. Bad: 'is Facebook open?'"
        "\n"
        "For finding upload buttons specifically: 'Return the pixel "
        "coordinates (x, y) of the centre of the button that opens a "
        "file dialog to upload a NEW photo from my computer. Ignore "
        "thumbnails of existing photos in any Recents grid. Format your "
        "answer as: x=NUMBER, y=NUMBER followed by a one-sentence "
        "description of which button you picked.'"
        "\n\n"
        "COST: ~2-4 seconds and real API tokens per call. Don't use after "
        "every click — the cheap DOM verify handles those. Use when the "
        "cost of a silent failure would be real (a post going out wrong, "
        "a quiz answer submitted without confirmation, etc.)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session":  {"type": "string", "description": _SESSION_DESC},
            "question": {
                "type": "string",
                "description": (
                    "The natural-language question to ask the vision model "
                    "about the current page. Be specific about what to look "
                    "for."
                ),
            },
        },
        "required": ["session", "question"],
    },
}


def get_web_tools() -> List[dict]:
    """All chat-facing browser tools. Included in get_phase1_tools()."""
    return [
        WEB_LIST_SESSIONS_TOOL,
        WEB_OPEN_SESSION_TOOL,
        WEB_CURRENT_STATE_TOOL,
        WEB_DOM_SNAPSHOT_TOOL,
        WEB_NAVIGATE_TOOL,
        WEB_WAIT_FOR_TOOL,
        WEB_CLICK_TOOL,
        WEB_TYPE_TOOL,
        WEB_SCROLL_TOOL,
        WEB_EXTRACT_TEXT_TOOL,
        WEB_SCREENSHOT_TOOL,
        WEB_VISION_CHECK_TOOL,
        WEB_COURSERA_HEALTH_TOOL,
        WEB_COURSERA_PROGRESS_TOOL,
        COURSERA_RESUME_TOOL,
        COURSERA_READ_LESSON_TOOL,
        COURSERA_NEXT_ITEM_TOOL,
        WEB_UPLOAD_FILE_TOOL,
        SYSTEM_KEYS_TOOL,
    ]
