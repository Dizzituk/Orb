# FILE: app/debug/web_tool_definitions.py
# Purpose: Chat-facing tool definitions for web browsing.
# Called-by: app.debug.tool_definitions
# Depends-on: app.debug
# Last-renovated: 2026-06-11
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
"""
from __future__ import annotations

from typing import List

from app.debug.web_tool_playbooks import (
    CLICK_RESULT_INTERPRETATION,
    META_UPLOAD_PLAYBOOK,
    RETRY_POLICY,
    TEXT_INPUT_TARGETING,
    WRONG_PAGE_RECOVERY,
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
        "in the Browser tab. Quick (1-2s)."
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
        "Page load typically takes 3-8 seconds. After navigating, call "
        "web_dom_snapshot or web_current_state to see where you landed."
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
        "(links, buttons, headings, form fields) with role, text content, "
        "href target, and pixel coordinates + size (x, y, w, h). This is "
        "the PRIMARY way to understand what's on a page when you don't "
        "already know the CSS selectors. Works reliably on React apps like "
        "Coursera where class names are obfuscated. After reading the "
        "snapshot, click by coordinates (web_click with x/y) or navigate "
        "to any href you found. When multiple elements share the same "
        "text (common for 'Get started', 'Sign in', 'Post' etc.), prefer "
        "the one with the LARGEST w×h — CTA buttons are typically "
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
        "its wrapper, dumping all list items of a known container."
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
        "Click an element. Pass EITHER a CSS selector OR (x, y) pixel "
        "coordinates from web_dom_snapshot — not both. Coordinates are "
        "more reliable than selectors on React apps. "
        "\n\n"
        "SNAP-TO-BUTTON for vision-driven clicks: when the (x, y) coords "
        "come from web_vision_check rather than web_dom_snapshot, ALSO "
        "pass `snap_to_button: true`. Vision is approximate (±50 px "
        "wobble); snap mode walks up the DOM from the vision pixel to "
        "the nearest interactive element (button, link, role=button) "
        "and clicks ITS centre instead of the raw vision coord. If the "
        "vision pixel falls in margin / padding (no element there), "
        "snap then searches outward in expanding rings up to 150 px and "
        "snaps to the nearest button-sized interactive element. The "
        "result includes a `snapped` field showing what element you "
        "actually hit (tag, text, w, h, search_distance)."
        "\n\n"
        "Do NOT pass snap_to_button when coords come from dom_snapshot "
        "— those coords are already centre-of-element."
        "\n\n"
        + TEXT_INPUT_TARGETING
        + "\n\n"
        + CLICK_RESULT_INTERPRETATION
        + "\n\n"
        "AFTER PUBLISHING / POSTING / SUBMITTING: most social platforms "
        "throw up a follow-up modal once the action succeeds — 'Schedule "
        "another post?', 'Boost this post?', 'Try Premium', etc. The post "
        "itself went through, but the modal blocks the next interaction "
        "with the page and the user is left staring at it. The task is "
        "NOT complete until the page is back to a usable state. After a "
        "successful submit, run web_dom_snapshot — if you see a modal "
        "with a close 'X', a 'Maybe later', 'Not now', 'Skip', 'No thanks', "
        "'Dismiss', or similar dismiss button, click it to close the modal "
        "before declaring the task done. Do NOT pick the affirmative option "
        "('Schedule another post', 'Boost', 'Upgrade') unless the user "
        "explicitly asked for it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session":  {"type": "string", "description": _SESSION_DESC},
            "selector": {"type": "string", "description": "CSS selector of element to click."},
            "x":        {"type": "integer", "description": "X coordinate in pixels (from web_dom_snapshot or web_vision_check)."},
            "y":        {"type": "integer", "description": "Y coordinate in pixels (from web_dom_snapshot or web_vision_check)."},
            "snap_to_button": {
                "type": "boolean",
                "description": (
                    "Snap (x, y) to the nearest interactive element's centre "
                    "before clicking. Set TRUE when coords came from "
                    "web_vision_check (vision is ~50px imprecise). Leave "
                    "FALSE / unset when coords came from web_dom_snapshot."
                ),
            },
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
        "lazy-loaded elements (infinite-scroll feeds, chart dashboards)."
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


WEB_UPLOAD_FILE_TOOL = {
    "name": "web_upload_file",
    "description": (
        "Attach a local file (image, video, document) to a web upload control. "
        "Do NOT use read_file or any other filesystem tool to attach images to "
        "a web form — reading the file just dumps bytes into the chat, it does "
        "not attach anything."
        "\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "WHEN TO USE THIS TOOL vs system_keys\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        "\n\n"
        "PRIMARY (recommended for Meta / Instagram / TikTok / YouTube / "
        "WordPress and other modern SPAs): use web_click to click the upload "
        "button, let the native OS file dialog open, then use system_keys to "
        "type the absolute path and press Enter. Production-tested. Works "
        "reliably because it bypasses the in-page chooser-intercept layer "
        "entirely — the OS dialog auto-focuses and SendKeys lands on it. "
        "See the worked example below."
        "\n\n"
        "FALLBACK (use this tool, web_upload_file): pass `(x, y)` of the "
        "upload button. Electron enables Chrome DevTools file-chooser "
        "interception, which suppresses the OS dialog and injects the file "
        "directly. Cleaner when it works — no focus dependency, no SendKeys "
        "— BUT Meta and several other platforms wrap their upload control "
        "in a way that the intercept misses, in which case this tool times "
        "out after 5 seconds with a 'no file chooser opened' error. If you "
        "hit that error, switch to the system_keys flow on the next attempt."
        "\n\n"
        "SELECTOR MODE: pass `selector` (CSS selector for an actual "
        "<input type=file>) for plain HTML forms. Modern social platforms "
        "hide their inputs, so this rarely applies to them."
        "\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        + META_UPLOAD_PLAYBOOK
        + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        + RETRY_POLICY
        + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        + WRONG_PAGE_RECOVERY
        + "\n"
        "DO NOT pick coordinates from web_dom_snapshot alone for the upload "
        "trigger inside a modal-style picker. The accessibility tree returns "
        "dozens of generic 'button' elements that look identical — Recents "
        "thumbnails, tab switchers, close buttons, the actual upload "
        "trigger. Vision can disambiguate; the DOM tree alone cannot. "
        "For dropdown-style menus the DOM IS enough — the sub-options have "
        "distinctive text labels."
        "\n\n"
        "FILE PATH: Always absolute. ASTRA's generated images live in the "
        "directory listed in the [ASTRA FILESYSTEM] block of your context; "
        "join that directory with the filename you saw in chat history. "
        "No filesystem search needed."
        "\n\n"
        "AFTER UPLOAD: the file is attached but the post is NOT yet "
        "submitted. Still need to click the platform's Post / Publish / "
        "Schedule button."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session":   {"type": "string", "description": _SESSION_DESC},
            "file_path": {
                "type": "string",
                "description": (
                    "Absolute filesystem path to the file. For ASTRA-generated "
                    "images, build it from the [ASTRA FILESYSTEM] directory + "
                    "filename in chat history. Forward or back slashes both fine."
                ),
            },
            "x": {
                "type": "integer",
                "description": (
                    "X pixel coordinate of the upload BUTTON (click mode). "
                    "From web_dom_snapshot. Pair with y."
                ),
            },
            "y": {
                "type": "integer",
                "description": (
                    "Y pixel coordinate of the upload BUTTON (click mode). "
                    "From web_dom_snapshot. Pair with x."
                ),
            },
            "selector": {
                "type": "string",
                "description": (
                    "CSS selector for an <input type='file'> element "
                    "(selector mode). Plain HTML forms only — modern SPAs "
                    "hide this input, so use (x, y) instead."
                ),
            },
        },
        "required": ["session", "file_path"],
    },
}


SYSTEM_KEYS_TOOL = {
    "name": "system_keys",
    "description": (
        "Send keystrokes to whatever native OS window currently has focus."
        "\n\n"
        "PRIMARY USE CASE: driving the native Windows file Open dialog that "
        "appears when you click 'Upload from desktop' on Meta / Instagram / "
        "TikTok / YouTube / WordPress. The dialog auto-focuses the moment "
        "it appears, so SendKeys lands on it. Pass an absolute file path as "
        "`text` and `press_enter_after=true` — Windows accepts paths "
        "directly in the filename field, so no folder navigation is needed. "
        "The dialog typing finishes in ~1 second."
        "\n\n"
        "FLOW (combined with web_click and web_vision_check) — use the"
        " verify-before-commit pattern, NOT atomic type+Enter:"
        "\n\n"
        "  1. web_vision_check returns the upload button's (x, y)."
        "\n"
        "  2. web_click(x, y) — native file dialog opens."
        "\n"
        "  3. web_vision_check: 'Is a Windows file Open dialog visible "
        "with a File name field?' — confirm dialog is up before typing."
        "\n"
        "  4. system_keys(text='C:\\\\path\\\\to\\\\file.png', "
        "press_enter_after=FALSE, pre_delay_ms=1500) — path typed, "
        "NOT yet committed. Empty / partial result here is recoverable; "
        "a committed wrong path isn't."
        "\n"
        "  5. web_vision_check: 'What text is in the File name input?' — "
        "verify the typed path appears correctly. If empty or partial, "
        "focus was stolen — re-click the filename field and retry step 4 "
        "with pre_delay_ms=2000."
        "\n"
        "  6. system_keys(text='', press_enter_after=true, pre_delay_ms=300) "
        "— commits the verified path with Enter alone, no retyping. "
        "(Empty-text + Enter is supported specifically for this commit "
        "step; ALL OTHER calls require non-empty text.)"
        "\n"
        "  7. web_dom_snapshot — confirm the thumbnail appeared in the "
        "composer."
        "\n\n"
        "WHY SPLIT TYPE FROM COMMIT: atomic press_enter_after=true with a "
        "long path is a focus-failure trap. If the dialog isn't actually "
        "focused (window-stack race, app stole focus mid-keypress, etc.), "
        "the path leaks into the browser view and Enter is consumed by "
        "the page. The composer ends up empty and you only notice when "
        "the dom_snapshot at step 7 shows no thumbnail. Verifying at "
        "step 5 catches focus failures BEFORE commit, when they're still "
        "recoverable."
        "\n\n"
        "WHY THIS EXISTS: this is the FALLBACK for the rare cases "
        "where web_upload_file's CDP intercept misses the file-chooser "
        "request (closed Shadow DOM, custom picker widgets, A/B test "
        "variants). The PRIMARY upload path is web_upload_file with "
        "(x, y) of the upload button — that path uses Chrome DevTools "
        "Protocol to intercept the chooser BEFORE the native dialog "
        "opens, so no system_keys typing is needed and there is no "
        "focus-race risk. Only fall through to system_keys when "
        "web_upload_file returns mode='os_dialog' or explicitly reports "
        "it didn't catch the chooser. Do not use system_keys as your "
        "first attempt for an upload — even though it works, going "
        "straight to it skips the deterministic CDP path and forces "
        "every upload through the brittle native-dialog flow."
        "\n\n"
        "FOCUS CAVEAT: SendKeys goes to whatever window has focus. Native "
        "file dialogs auto-focus when they open, so this is normally "
        "reliable. If you suspect the dialog isn't focused (rare), bump "
        "`pre_delay_ms` from the default 600 to 1500 to give Windows more "
        "time to settle."
        "\n\n"
        "PATH ESCAPING: handled internally. Backslashes, colons, spaces "
        "and parentheses in paths are all fine."
        "\n\n"
        "PLATFORM: currently Windows-only (PowerShell + WScript.Shell). On "
        "non-Windows hosts the call returns an error."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session": {"type": "string", "description": _SESSION_DESC},
            "text": {
                "type": "string",
                "description": (
                    "Literal text to type into the focused window. For file "
                    "dialogs, this is the absolute path of the file to upload."
                ),
            },
            "press_enter_after": {
                "type": "boolean",
                "description": (
                    "Append a press of the Enter key after typing the text. "
                    "Set to true to submit a file dialog. Default false."
                ),
            },
            "pre_delay_ms": {
                "type": "integer",
                "description": (
                    "Wait this many ms before sending the first keystroke. "
                    "Default 600 — enough for native dialogs to appear and "
                    "focus. Bump to 1500 if you suspect timing issues."
                ),
            },
        },
        "required": ["session", "text"],
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
        WEB_CLICK_TOOL,
        WEB_TYPE_TOOL,
        WEB_SCROLL_TOOL,
        WEB_EXTRACT_TEXT_TOOL,
        WEB_SCREENSHOT_TOOL,
        WEB_VISION_CHECK_TOOL,
        WEB_UPLOAD_FILE_TOOL,
        SYSTEM_KEYS_TOOL,
    ]
