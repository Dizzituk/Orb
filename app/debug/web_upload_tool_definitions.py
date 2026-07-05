# FILE: app/debug/web_upload_tool_definitions.py
# Purpose: Chat-facing tool definitions for web file upload + native-dialog keys.
# Called-by: app.debug.web_tool_definitions (aggregated into get_web_tools)
# Depends-on: app.debug.web_tool_playbooks
# Last-renovated: 2026-07-02
"""
Chat-facing tool definitions for file upload into web pages.

Split out of web_tool_definitions.py (2026-07-02, Amendment A Job 9c)
purely for the file-size cap — these two definitions carry the longest
playbook prose. Definitions are unchanged; the standard-posting
redirect lives in META_UPLOAD_PLAYBOOK (web_tool_playbooks).
"""
from __future__ import annotations

from app.debug.web_tool_playbooks import (
    META_UPLOAD_PLAYBOOK,
    RETRY_POLICY,
    WRONG_PAGE_RECOVERY,
)


_SESSION_DESC = (
    "Platform key of the target session. Common values: 'coursera', "
    "'meta_business', 'tiktok_astraukai', 'youtube_studio', "
    "'wordpress_admin'. Call web_list_sessions if unsure."
)


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
