# FILE: app/debug/tool_definitions.py
"""
Tool schemas for the Debug Assistant LLM.

These are the tool definitions sent to the LLM so it can request
file reads, writes, command execution, etc.

Phase 1: Read + user-file-write tools.
Phase 2: Full write tools (sandbox, emulator, commands).
"""

from __future__ import annotations

from typing import List


# =============================================================================
# USER FILE TOOLS (v0.15.0) — search, read, and write personal files
# =============================================================================

SEARCH_MY_FILES_TOOL = {
    "name": "search_my_files",
    "description": (
        "Search the user's personal files (Documents, Pictures, Music, Videos, "
        "Desktop, Screenshots, ASTRA Output, Android Project) by filename, "
        "extension, or category. Returns matching file paths, sizes, and types. "
        "Use this when the user asks to find, locate, open, or list files from "
        "their computer. Example: query='learning roadmap' finds files with that "
        "name. You can also filter by category (documents, pictures, music, "
        "videos, desktop, screenshots) or extension (pdf, docx, mp3, etc). "
        "NOTE: If the user has already pasted or shared content in the current "
        "conversation, check whether they are referring to that content before "
        "searching the filesystem. When in doubt, ask which they mean."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search term to match against filenames (case-insensitive partial match).",
            },
            "category": {
                "type": "string",
                "description": "Optional: filter by category (documents, pictures, music, videos, desktop, screenshots, astra_output, android_project).",
            },
            "extension": {
                "type": "string",
                "description": "Optional: filter by file extension without dot (e.g. pdf, docx, mp3, jpg).",
            },
        },
        "required": ["query"],
    },
}

READ_USER_FILE_TOOL = {
    "name": "read_user_file",
    "description": (
        "Read the text content of one of the user's personal files. "
        "Use the path returned by search_my_files. Works with text files, "
        "documents (docx, pdf, xlsx, pptx), code files, and other text-readable "
        "formats. Returns extracted text content. For images/audio/video, "
        "returns metadata only. "
        "NOTE: If the user has already shared content in the conversation, check "
        "whether they are referring to that content before reading from disk."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Full file path (from search_my_files results).",
            },
        },
        "required": ["path"],
    },
}

WRITE_USER_FILE_TOOL = {
    "name": "write_user_file",
    "description": (
        "Create or overwrite a file in the user's personal folders "
        "(Documents, Pictures, Music, Videos, Desktop, Screenshots, ASTRA Output). "
        "Use this when the user asks you to save, create, or write a file in their "
        "personal areas. Call get_user_folders first to get the correct base path, "
        "then provide the full absolute path. "
        "Example: get_user_folders -> documents is 'C:/Users/.../Documents' -> "
        "write to 'C:/Users/.../Documents/my_poem.txt'. "
        "IMPORTANT: Only works within allowed user folders. Cannot write to "
        "ASTRA codebase or system directories."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Absolute file path within a user folder. "
                    "Use paths from get_user_folders as the base."
                ),
            },
            "content": {
                "type": "string",
                "description": "Full file content to write.",
            },
        },
        "required": ["path", "content"],
    },
}

GET_USER_FOLDERS_TOOL = {
    "name": "get_user_folders",
    "description": (
        "Get the resolved absolute paths for all user personal folders "
        "(Documents, Pictures, Music, Videos, Desktop, Screenshots, ASTRA Output). "
        "Call this BEFORE writing files to know the correct paths. "
        "These paths are the real filesystem locations (may include OneDrive paths). "
        "Use the returned paths as the base when constructing paths for "
        "write_user_file or when telling the user where their files are."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

# =============================================================================
# CODEBASE / SANDBOX TOOLS (read-only in Phase 1)
# =============================================================================

READ_FILE_TOOL = {
    "name": "read_file",
    "description": (
        "Read the contents of a file. Works in the sandbox project directory "
        "and for host scan output files (architecture scan, file health scan). "
        "Use absolute paths for sandbox files (e.g., D:/Orb/app/debug/model_router.py)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute file path to read.",
            },
            "head": {
                "type": "integer",
                "description": "If set, only return the first N lines.",
            },
            "tail": {
                "type": "integer",
                "description": "If set, only return the last N lines.",
            },
        },
        "required": ["path"],
    },
}

LIST_FILES_TOOL = {
    "name": "list_files",
    "description": (
        "List files and directories at a given path. Returns names with [FILE] or [DIR] prefix. "
        "Works in the sandbox and for host scan directories."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute directory path to list.",
            },
        },
        "required": ["path"],
    },
}

READ_PIPELINE_STATE_TOOL = {
    "name": "read_pipeline_state",
    "description": (
        "Get the current ASTRA pipeline state including active flow, recent stage traces, "
        "and any active spec. Use this to understand where the pipeline is at."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

READ_LOGS_TOOL = {
    "name": "read_logs",
    "description": (
        "Read recent log entries from ASTRA. Can filter by level (ERROR, WARNING, INFO) "
        "and limit the number of entries returned."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "level": {
                "type": "string",
                "enum": ["ERROR", "WARNING", "INFO", "ALL"],
                "description": "Log level filter. Default: ALL.",
            },
            "limit": {
                "type": "integer",
                "description": "Max number of log lines to return. Default: 50.",
            },
        },
        "required": [],
    },
}

SEARCH_FILES_TOOL = {
    "name": "search_files",
    "description": (
        "Search for files matching a glob pattern within the sandbox project. "
        "Returns matching file paths. Example: '**/*.py' finds all Python files."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Root directory to search from.",
            },
            "pattern": {
                "type": "string",
                "description": "Glob pattern to match (e.g., '**/*.py', '**/test_*.py').",
            },
        },
        "required": ["path", "pattern"],
    },
}


# =============================================================================
# PHASE 2: WRITE TOOLS (sandbox/codebase writes, commands, emulator)
# =============================================================================

WRITE_FILE_TOOL = {
    "name": "write_file",
    "description": (
        "Create or overwrite a file on the filesystem. Use absolute paths. "
        "Use for implementing fixes, creating new files, saving documents, "
        "architecture maps, or any file the user requests."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute file path in the sandbox.",
            },
            "content": {
                "type": "string",
                "description": "Full file content to write.",
            },
        },
        "required": ["path", "content"],
    },
}

EDIT_FILE_TOOL = {
    "name": "edit_file",
    "description": (
        "Apply targeted edits to a file. Uses exact string matching "
        "to find and replace content. Use absolute paths."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute file path in the sandbox.",
            },
            "old_text": {
                "type": "string",
                "description": "Exact text to find (must be unique in the file).",
            },
            "new_text": {
                "type": "string",
                "description": "Text to replace it with.",
            },
        },
        "required": ["path", "old_text", "new_text"],
    },
}

RUN_COMMAND_TOOL = {
    "name": "run_command",
    "description": (
        "Execute a PowerShell command in the sandbox. Returns stdout and stderr. "
        "Use for running tests, checking process status, installing packages, etc. "
        "SANDBOX ONLY — commands cannot affect the host."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "PowerShell command to execute.",
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for the command. Default: D:/Orb",
            },
            "timeout_sec": {
                "type": "integer",
                "description": "Timeout in seconds. Default: 30.",
            },
        },
        "required": ["command"],
    },
}


# =============================================================================
# TOOL SETS
# =============================================================================


# ═══════════════════════════════════════════════════════════════
# ADB EMULATOR TOOLS (for Android app testing)
# ═══════════════════════════════════════════════════════════════

SCREENSHOT_TOOL = {
    "name": "emulator_screenshot",
    "description": (
        "Take a screenshot of the Android emulator. Returns the file path "
        "and size. Use this to visually verify UI layout, check if buttons "
        "are visible, verify text fields are present, etc. The screenshot "
        "is saved locally for inspection."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

UI_DUMP_TOOL = {
    "name": "emulator_ui_dump",
    "description": (
        "Dump the UI view hierarchy from the Android emulator as XML. "
        "Returns all visible UI elements with their class names, text, "
        "content descriptions, resource IDs, and screen bounds (coordinates). "
        "Use this to find clickable buttons, text fields, and verify layout structure."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

EMULATOR_TAP_TOOL = {
    "name": "emulator_tap",
    "description": (
        "Tap at screen coordinates on the Android emulator. "
        "Get coordinates from emulator_ui_dump (bounds attribute). "
        "Use to click buttons, focus text fields, toggle switches, etc."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "X coordinate"},
            "y": {"type": "integer", "description": "Y coordinate"},
        },
        "required": ["x", "y"],
    },
}

EMULATOR_TYPE_TOOL = {
    "name": "emulator_type",
    "description": (
        "Type text into the currently focused field on the Android emulator. "
        "First tap a text field using emulator_tap, then use this to enter text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to type"},
        },
        "required": ["text"],
    },
}

EMULATOR_KEY_TOOL = {
    "name": "emulator_key",
    "description": (
        "Press a key on the Android emulator. Common keycodes: "
        "KEYCODE_ENTER (send/confirm), KEYCODE_BACK (navigate back), "
        "KEYCODE_HOME (home screen), KEYCODE_DEL (backspace)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "keycode": {"type": "string", "description": "Android keycode name"},
        },
        "required": ["keycode"],
    },
}

GRADLE_BUILD_TOOL = {
    "name": "gradle_build",
    "description": (
        "Run Gradle assembleDebug for AstraBridge. Compiles the app and "
        "reports success or failure with error details. Use after making "
        "code changes to verify they compile."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

GRADLE_INSTALL_TOOL = {
    "name": "gradle_install",
    "description": (
        "Build and install an Android debug APK on the emulator. "
        "Provide package/activity/project details for the app under test when available. "
        "Compiles, packages, and deploys in one step. Use after fixing code to get "
        "the new version running on the emulator for testing."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "project_root": {
                "type": "string",
                "description": "Absolute path to the Android project root. Defaults to the AstraBridge project for backward compatibility.",
            },
            "apk_path": {
                "type": "string",
                "description": "Optional absolute path to a built APK to install instead of building from project_root.",
            },
            "package_name": {
                "type": "string",
                "description": "Android package name for the app under test.",
            },
            "activity_name": {
                "type": "string",
                "description": "Fully qualified or relative launcher activity name for the app under test.",
            },
        },
        "required": [],
    },
}

APP_RESTART_TOOL = {
    "name": "app_restart",
    "description": (
        "Force stop and relaunch AstraBridge on the emulator. "
        "Use after installing a new build or when the app is in a bad state."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

CRASH_LOG_TOOL = {
    "name": "get_crash_log",
    "description": (
        "Get the most recent crash log for AstraBridge from logcat. "
        "Shows FATAL EXCEPTION stack traces. Use when the app crashes "
        "after a change to diagnose the error."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

# =============================================================================
# UNIVERSAL TOOLS (available to ALL models, regardless of tool eligibility)
# =============================================================================

WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the public web for current information. Use this when the user "
        "asks you to research, look up, find out about, or get current data on "
        "any topic. Returns search results with titles, URLs, and snippets. "
        "IMPORTANT: Actually CALL this tool when the user asks for research or "
        "current information. Do not just say you will search — call the tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query (1-512 characters). Be specific.",
            },
            "max_results": {
                "type": "integer",
                "description": "Number of results to return (1-10, default 5).",
            },
        },
        "required": ["query"],
    },
}




# =============================================================================
# CLOUD STORAGE TOOLS (Google Drive via rclone)
# =============================================================================

CLOUD_UPLOAD_TOOL = {
    "name": "cloud_upload",
    "description": (
        "Upload a file from the local filesystem to Google Drive. "
        "Use this when the user asks you to put a file on their Drive, "
        "share a document via Drive, or make a file accessible on their phone. "
        "Provide the local file path (absolute) and a cloud destination path. "
        "The cloud path is relative to the Drive root, e.g. 'Documents/report.pdf'. "
        "Use search_my_files or get_user_folders to find the local file first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "local_path": {
                "type": "string",
                "description": "Absolute path to the local file to upload.",
            },
            "cloud_path": {
                "type": "string",
                "description": "Destination path on Google Drive (e.g. 'Documents/report.pdf', 'ASTRA/output.txt'). Folders are created automatically.",
            },
        },
        "required": ["local_path", "cloud_path"],
    },
}

CLOUD_LIST_TOOL = {
    "name": "cloud_list",
    "description": (
        "List files and folders on Google Drive at a given path. "
        "Returns names, sizes, and whether each item is a file or directory. "
        "Use to check what exists on Drive before uploading or to help the user "
        "find files. Pass an empty string or '/' to list the root."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Cloud path to list (e.g. 'Documents', 'APKs'). Empty for root.",
            },
        },
        "required": [],
    },
}

# Desktop Computer Use Tools
DESKTOP_SCREENSHOT_TOOL = {"name": "desktop_screenshot", "description": "Take a screenshot of the ASTRA desktop application or full screen. Optionally specify a window title to capture just that window.", "parameters": {"type": "object", "properties": {"window_title": {"type": "string", "description": "Optional window title to capture (e.g., 'Astra')."}}, "required": []}}
DESKTOP_CLICK_TOOL = {"name": "desktop_click", "description": "Click at screen coordinates. Use desktop_screenshot first to see current state. Only works within approved windows (ASTRA, Windows Sandbox, Android Studio).", "parameters": {"type": "object", "properties": {"x": {"type": "integer", "description": "X coordinate"}, "y": {"type": "integer", "description": "Y coordinate"}, "button": {"type": "string", "description": "'left', 'right', or 'middle'. Default: 'left'"}, "clicks": {"type": "integer", "description": "1=single, 2=double. Default: 1"}}, "required": ["x", "y"]}}
DESKTOP_TYPE_TOOL = {"name": "desktop_type", "description": "Type text at current cursor position. Click a text field first. Only works when an approved window is focused.", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "Text to type"}}, "required": ["text"]}}
DESKTOP_KEY_TOOL = {"name": "desktop_key", "description": "Press a key or combo (e.g., 'enter', 'tab', 'ctrl+a', 'alt+f4'). Only works when an approved window is focused.", "parameters": {"type": "object", "properties": {"key": {"type": "string", "description": "Key or combo"}}, "required": ["key"]}}
DESKTOP_SCROLL_TOOL = {"name": "desktop_scroll", "description": "Scroll at a position. Positive=up, negative=down. Only works within approved windows.", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "clicks": {"type": "integer", "description": "Scroll amount. Positive=up, negative=down."}}, "required": ["x", "y"]}}
DESKTOP_FIND_WINDOW_TOOL = {"name": "desktop_find_window", "description": "Find a window by title and get its position and size.", "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "Window title to search for (partial match)"}}, "required": ["title"]}}
DESKTOP_READ_SCREEN_TOOL = {"name": "desktop_read_screen", "description": "OCR the screen to extract visible text. Optionally specify a region.", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "width": {"type": "integer"}, "height": {"type": "integer"}}, "required": []}}

# =============================================================================
# SOCIAL MEDIA API TOOLS
# First-party HTTP integrations. Preferred over browser automation whenever
# the platform has an API equivalent (deterministic, single round-trip,
# structured errors). Browser flow stays for engagement / discovery work.
# =============================================================================

META_POST_TOOL = {
    "name": "meta_post",
    "description": (
        "Publish (or schedule) an image post to Facebook via the Meta Graph "
        "API. Use this whenever the user asks you to post, publish, share, "
        "or schedule an image to Facebook. PREFER this over the browser "
        "upload flow (web_open_session 'meta_business' + system_keys + etc.) "
        "because it is deterministic: one HTTP call, structured success or "
        "structured error, no focus races, no native-dialog vision blind "
        "spots. The browser path remains for engagement reading, comment "
        "drafting, and tasks without an API equivalent.\n\n"
        "Required: image_path (absolute), caption (text; empty string "
        "allowed). Optional: scheduled_at (Unix timestamp; must be 11 min "
        "to 180 days ahead). Default target is 'facebook'. Instagram is "
        "not yet supported through this tool (needs a public image URL; "
        "hosting decision pending).\n\n"
        "Configuration: Settings -> API Keys must contain 'meta_access_token' "
        "(long-lived User or Page Access Token with pages_manage_posts "
        "scope) and 'facebook_page_id'. If either is missing the tool "
        "returns a config error — surface that to the user verbatim so they "
        "know exactly what to add."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": (
                    "Absolute path to image file on disk "
                    "(PNG, JPG, JPEG, WebP, GIF)."
                ),
            },
            "caption": {
                "type": "string",
                "description": "Caption text for the post. Empty string allowed.",
            },
            "target": {
                "type": "string",
                "description": (
                    "Target platform. Currently only 'facebook' (default). "
                    "Instagram pending public image hosting decision."
                ),
            },
            "scheduled_at": {
                "type": "integer",
                "description": (
                    "Unix timestamp for scheduled publish. Must be at least "
                    "11 minutes and at most 180 days in the future. Omit "
                    "to publish immediately."
                ),
            },
            "verify": {
                "type": "boolean",
                "description": (
                    "If true (default), performs a cross-channel "
                    "verification after upload by reading the post back via "
                    "a separate GET request. Skipped automatically for "
                    "scheduled posts (object not yet queryable until "
                    "publish time)."
                ),
            },
        },
        "required": ["image_path", "caption"],
    },
}

# =============================================================================
# FLOW MEMORY TOOLS
# Cached, verified multi-step interaction patterns. Each step has a
# precondition, an action, and a postcondition; the runner halts on the
# first verification failure with a structured diagnostic that names the
# failed step and lists every step that confirmed working before it.
# This is the failure-isolation guarantee: when something breaks, the
# system knows exactly which stage is the problem and the rest is preserved.
# =============================================================================

FLOW_RUN_TOOL = {
    "name": "flow_run",
    "description": (
        "Execute a previously-saved interaction flow on a platform "
        "(Meta, TikTok, WordPress, Coursera, etc.). Each step's "
        "postcondition is verified before moving on, so successful "
        "runs end with high confidence the task actually completed; "
        "failed runs halt at the exact step that broke and tell you "
        "which steps confirmed working before it. Prefer this over "
        "manually re-driving the same task with web_click + "
        "web_dom_snapshot loops once a flow has been recorded.\n\n"
        "Use flow_inspect first if you need to see what flows exist "
        "or read a flow's step definitions. Use flow_save to record a "
        "new flow after completing a task manually."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "platform": {
                "type": "string",
                "description": (
                    "Platform key, e.g. 'meta_business', 'tiktok_studio', "
                    "'wordpress', 'coursera'."
                ),
            },
            "task": {
                "type": "string",
                "description": (
                    "Task key, e.g. 'reply_top_comment', 'schedule_video', "
                    "'publish_draft', 'mark_lesson_complete'."
                ),
            },
            "default_session": {
                "type": "string",
                "description": (
                    "Optional web session id used for any step that "
                    "doesn't specify its own session. Usually the same "
                    "key as platform."
                ),
            },
        },
        "required": ["platform", "task"],
    },
}

FLOW_SAVE_TOOL = {
    "name": "flow_save",
    "description": (
        "Save (create or update) a flow definition. Use this AFTER you "
        "have just successfully completed a multi-step task on a "
        "platform, to record the exact sequence of actions and the "
        "verifications that confirmed each one worked. The next time "
        "the same task is needed, flow_run replays the cached pattern "
        "in a fraction of the time with built-in failure isolation.\n\n"
        "Each step is a dict with these fields:\n"
        "  step_id      : short stable identifier ('open_composer')\n"
        "  description  : one-line human-readable summary\n"
        "  session      : web session id (for browser steps)\n"
        "  precondition : optional Check that must hold before the action\n"
        "  action       : {kind: <tool_name>, params: {...}}\n"
        "  postcondition: Check that confirms the action worked\n\n"
        "A Check is {kind: dom_includes|dom_excludes|url_includes|"
        "text_includes|always_pass, expected: [...substrings...], "
        "timeout_ms: int}. Substrings match against the result of "
        "web_dom_snapshot, web_current_state, or web_extract_text "
        "depending on kind. Choose substrings that are stable across "
        "sessions (aria-labels, button text) and unique enough to not "
        "match unrelated pages."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "platform": {"type": "string", "description": "Platform key."},
            "task": {"type": "string", "description": "Task key."},
            "description": {
                "type": "string",
                "description": "Optional human-readable description of the flow.",
            },
            "steps": {
                "type": "array",
                "description": (
                    "Ordered list of step dicts. See description above "
                    "for the schema of each step."
                ),
                "items": {"type": "object"},
            },
        },
        "required": ["platform", "task", "steps"],
    },
}

FLOW_INSPECT_TOOL = {
    "name": "flow_inspect",
    "description": (
        "List saved flows or read one in full. Call with no params to "
        "list every saved flow across all platforms. Call with platform "
        "alone to filter by platform. Call with both platform and task "
        "to read a single flow's full JSON definition (useful before "
        "editing it via flow_save, or after a flow_run failure to find "
        "the failing step's current expectations)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "platform": {
                "type": "string",
                "description": "Optional platform filter.",
            },
            "task": {
                "type": "string",
                "description": (
                    "Optional task key; combined with platform, returns "
                    "the full flow definition."
                ),
            },
        },
        "required": [],
    },
}

def get_universal_tools() -> List[dict]:
    """Tools available to ALL models - web search + memory access.

    These are injected regardless of model trust level. Memory tools
    are universal because the chat LLM needs to be able to write to
    memory in every turn, on every model.
    """
    from app.debug.memory_tool_definitions import get_memory_tools
    return [WEB_SEARCH_TOOL] + get_memory_tools()


def get_phase1_tools() -> List[dict]:
    """Core tools: read + user file write access + web browsing.

    v0.15.0: Added write_user_file and get_user_folders so ALL models
    can create/save files in the user's personal folders (Documents,
    Pictures, Desktop, etc.) while keeping codebase writes sandbox-only.

    v0.18.0: Added web browsing tools (web_list_sessions, web_navigate,
    web_dom_snapshot, etc.) so chat can drive the user's logged-in
    browser sessions - Coursera, Meta Business Suite, TikTok Studio,
    YouTube Studio, WordPress. Same read+user-write tier because
    browsing is read-equivalent for the filesystem; any writes happen
    to third-party services the user has authed into.
    """
    from app.debug.web_tool_definitions import get_web_tools

    return [
        SEARCH_MY_FILES_TOOL,
        READ_USER_FILE_TOOL,
        WRITE_USER_FILE_TOOL,
        GET_USER_FOLDERS_TOOL,
        RESCAN_MANIFEST_TOOL,
        REINDEX_FILE_TOOL,
        SEARCH_DISK_LIVE_TOOL,
        CREATE_FOLDER_TOOL,
        MOVE_FILE_TOOL,
        MOVE_FILES_BATCH_TOOL,
        CREATE_DOCX_TOOL,
        CREATE_PDF_TOOL,
        CREATE_XLSX_TOOL,
        CREATE_HTML_REPORT_TOOL,
        READ_IMAGE_TOOL,
        READ_FILE_TOOL,
        LIST_FILES_TOOL,
        READ_PIPELINE_STATE_TOOL,
        READ_LOGS_TOOL,
        SEARCH_FILES_TOOL,
        CLOUD_UPLOAD_TOOL,
        CLOUD_LIST_TOOL,
        META_POST_TOOL,
        FLOW_RUN_TOOL,
        FLOW_SAVE_TOOL,
        FLOW_INSPECT_TOOL,
    ] + get_web_tools()


def get_phase2_tools() -> List[dict]:
    """Full tool set including sandbox write access for Phase 2."""
    return get_phase1_tools() + [
        WRITE_FILE_TOOL,
        EDIT_FILE_TOOL,
        RUN_COMMAND_TOOL,
        SCREENSHOT_TOOL,
        UI_DUMP_TOOL,
        EMULATOR_TAP_TOOL,
        EMULATOR_TYPE_TOOL,
        EMULATOR_KEY_TOOL,
        GRADLE_BUILD_TOOL,
        GRADLE_INSTALL_TOOL,
        APP_RESTART_TOOL,
        CRASH_LOG_TOOL,
        DESKTOP_SCREENSHOT_TOOL, DESKTOP_CLICK_TOOL, DESKTOP_TYPE_TOOL,
        DESKTOP_KEY_TOOL, DESKTOP_SCROLL_TOOL, DESKTOP_FIND_WINDOW_TOOL, DESKTOP_READ_SCREEN_TOOL,
    ]


def get_tools_for_tier(tier: str) -> List[dict]:
    """Get appropriate tools based on routing tier."""
    if tier == "agentic":
        return get_phase2_tools()
    return get_phase1_tools()


# =============================================================================
# MANIFEST RESCAN TOOLS (v0.16.0) — fallbacks for stale manifest
# =============================================================================

RESCAN_MANIFEST_TOOL = {
    "name": "rescan_manifest",
    "description": (
        "Force a full rescan of the user's personal folders and refresh the "
        "file manifest. Use this ONLY as a fallback when search_my_files "
        "returns no results for a file the user insists exists. The live "
        "file watcher normally keeps the manifest current, so this should "
        "rarely be needed. Takes no parameters."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

REINDEX_FILE_TOOL = {
    "name": "reindex_file",
    "description": (
        "Refresh a single file's entry in the manifest. Faster than a full "
        "rescan when you know the exact path of a file that is missing from "
        "or stale in search_my_files results."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute file path to reindex.",
            },
        },
        "required": ["path"],
    },
}

SEARCH_DISK_LIVE_TOOL = {
    "name": "search_disk_live",
    "description": (
        "Search the actual filesystem for a file by name, bypassing the "
        "manifest cache. Use this as a FALLBACK when search_my_files "
        "returns no results for a file the user insists exists. Slower "
        "(100-500ms) but authoritative — if the file is on disk, this "
        "finds it. Automatically heals the manifest cache for any matches. "
        "Same query semantics as search_my_files (case-insensitive substring)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search term to match against filenames (case-insensitive substring).",
            },
            "category": {
                "type": "string",
                "description": "Optional: limit walk to one category (documents, pictures, music, videos, desktop, screenshots).",
            },
            "extension": {
                "type": "string",
                "description": "Optional: filter by file extension without dot (e.g. pdf, docx, mp3).",
            },
        },
        "required": ["query"],
    },
}

READ_IMAGE_TOOL = {
    "name": "read_image",
    "description": (
        "Read the visual content of an image file (PNG, JPEG, WebP, GIF, etc.) "
        "using Gemini Vision. Returns a description of what is shown in the image, "
        "including any visible text, UI elements, timestamps, or notable details. "
        "Use this when the user asks what is shown in a screenshot or photo, or when "
        "you need to extract information from an image rather than just list its filename. "
        "Provide a specific question for targeted answers; otherwise omit it for a "
        "general description."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to the image file on disk.",
            },
            "question": {
                "type": "string",
                "description": (
                    "Optional question to ask about the image. If omitted, returns a "
                    "general description of contents, visible text, and notable elements."
                ),
            },
        },
        "required": ["path"],
    },
}
# =============================================================================
# FILE OPS TOOLS (v0.17.0) - move/create/batch-move for user folder organisation
# =============================================================================

CREATE_FOLDER_TOOL = {
    "name": "create_folder",
    "description": (
        "Create a new directory in the user's personal folders (Documents, "
        "Pictures, Desktop, Downloads, Music, Videos, OneDrive, etc.). "
        "Creates parent directories as needed. Cannot create folders inside "
        "ASTRA's protected codebase or Windows system paths. Returns "
        "confirmation or 'already exists' if the folder is already there."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path of the folder to create.",
            },
        },
        "required": ["path"],
    },
}

MOVE_FILE_TOOL = {
    "name": "move_file",
    "description": (
        "Move or rename a single file. Both source and destination must be "
        "inside allowed user folders. Refuses to overwrite an existing "
        "destination unless overwrite=true is passed explicitly. Use this for "
        "renaming or relocating files; for many files at once, use "
        "move_files_batch instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source": {"type": "string", "description": "Absolute path of the file to move."},
            "destination": {"type": "string", "description": "Absolute target path (full filename, not just folder)."},
            "overwrite": {"type": "boolean", "description": "If true, replace destination when it exists. Default false."},
        },
        "required": ["source", "destination"],
    },
}

MOVE_FILES_BATCH_TOOL = {
    "name": "move_files_batch",
    "description": (
        "Move many files in one call. Use this when sorting or reorganising "
        "more than two or three files - one call is far cheaper than many "
        "sequential move_file calls. Skip-and-continue: a failure on one "
        "file does NOT abort the batch. Returns a summary with succeeded "
        "count and a list of failures."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "moves": {
                "type": "array",
                "description": "List of {source, destination} objects.",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "destination": {"type": "string"},
                    },
                    "required": ["source", "destination"],
                },
            },
            "overwrite": {"type": "boolean", "description": "If true, replace destinations that exist. Default false."},
        },
        "required": ["moves"],
    },
}


# =============================================================================
# STYLED FILE CREATION TOOLS (v0.17.0) - presentation-quality docs
# =============================================================================

# Shared content schema description for all four styled creators
_CONTENT_SCHEMA_DESC = (
    "List of block objects. Each block has a 'type' field. Supported types:\n"
    "  - {type:'heading', level:1-4, text:'...'}\n"
    "  - {type:'paragraph', text:'...'}\n"
    "  - {type:'list', items:['...','...'], ordered:false}\n"
    "  - {type:'table', headers:['col1','col2'], rows:[['a','b'],['c','d']]}\n"
    "  - {type:'rule'}    (horizontal divider)\n"
    "  - {type:'spacer'}  (blank line)\n"
    "  - {type:'code', text:'...', language:'python'}\n"
    "Order in the list determines render order."
)

_THEME_DESC = (
    "Visual theme. 'auto' (default) inspects the filename for keywords like "
    "'legal', 'evidence', 'letter' and picks 'astra_minimal' (plain, formal); "
    "everything else gets 'astra_default' (modern, branded). Pass 'minimal' "
    "or 'default' to force a choice."
)

_SKILL_DESC = (
    "Optional skill playbook ID that guides document structure and tone. "
    "Available skills: 'formal_document' (legal, letters to MPs, formal "
    "reports - serif, restrained, no tables in body), 'casual_document' "
    "(personal writing, friendly updates - sans-serif, natural voice), "
    "'data_spreadsheet' (for xlsx: proper header row, evidence ref column, "
    "no inline totals). If omitted, the skill is auto-detected from the "
    "filename and title keywords. Pass an explicit value to override."
)

_BRIEF_DESC = (
    "Natural-language description of what the document should contain. When "
    "provided INSTEAD of pre-structured 'content', ASTRA runs a reasoning-"
    "tier structuring pass that follows the chosen skill\u2019s playbook to "
    "produce the final document. Use this when you do not want to hand-build "
    "content blocks yourself. You can combine brief with 'source_material' "
    "to provide facts, data, or prior text that should be incorporated."
)

_SOURCE_MATERIAL_DESC = (
    "Optional supporting data for the structuring pass. Can be a plain string "
    "(prior text to incorporate), a JSON object/array (structured facts, a "
    "dataset of records), or a JSON-encoded string. Ignored when 'content' "
    "or 'sheets' is provided directly."
)

CREATE_DOCX_TOOL = {
    "name": "create_docx",
    "description": (
        "Create a styled Microsoft Word document. Use this when the user "
        "wants a presentation-quality .docx for a report, summary, proposal, "
        "letter, or evidence pack. "
        "Two modes: (1) pass pre-structured 'content' blocks for exact "
        "control, or (2) pass a natural-language 'brief' and let ASTRA "
        "structure the document via a skill playbook using a reasoning-tier "
        "model. Mode 2 is preferred for most cases - you describe what the "
        "doc should say and ASTRA produces proper structure automatically. "
        "Theme auto-selects from filename keywords (legal/evidence/letter "
        "-> minimal plain style; everything else -> branded styled). Cover "
        "page is added automatically for the styled theme."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute output path. .docx extension added if missing."},
            "title": {"type": "string", "description": "Document title (used on cover page and Word metadata)."},
            "subtitle": {"type": "string", "description": "Optional subtitle."},
            "author": {"type": "string", "description": "Optional author name."},
            "theme": {"type": "string", "description": _THEME_DESC},
            "skill": {"type": "string", "description": _SKILL_DESC},
            "brief": {"type": "string", "description": _BRIEF_DESC},
            "source_material": {"description": _SOURCE_MATERIAL_DESC},
            "content": {"type": "array", "description": _CONTENT_SCHEMA_DESC, "items": {"type": "object"}},
        },
        "required": ["path", "title"],
    },
}

CREATE_PDF_TOOL = {
    "name": "create_pdf",
    "description": (
        "Create a styled PDF document. Same schema, skill, and brief modes "
        "as create_docx. Output is a clean A4 PDF with page numbers and "
        "generated-date footer. Good for documents that will be shared, "
        "printed, or attached to correspondence. For editable outputs prefer "
        "create_docx (the user can then export to PDF themselves)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute output path. .pdf extension added if missing."},
            "title": {"type": "string", "description": "Document title."},
            "subtitle": {"type": "string", "description": "Optional subtitle."},
            "author": {"type": "string", "description": "Optional author name."},
            "theme": {"type": "string", "description": _THEME_DESC},
            "skill": {"type": "string", "description": _SKILL_DESC},
            "brief": {"type": "string", "description": _BRIEF_DESC},
            "source_material": {"description": _SOURCE_MATERIAL_DESC},
            "content": {"type": "array", "description": _CONTENT_SCHEMA_DESC, "items": {"type": "object"}},
        },
        "required": ["path", "title"],
    },
}

CREATE_XLSX_TOOL = {
    "name": "create_xlsx",
    "description": (
        "Create a styled Excel workbook. Two modes: (1) pass pre-built "
        "'sheets' objects for exact control, or (2) pass a natural-language "
        "'brief' and optional 'source_material' and let ASTRA structure the "
        "workbook via the data_spreadsheet skill using a reasoning-tier "
        "model. Mode 2 is preferred when you have raw data that needs "
        "turning into a proper workbook. Header row gets theme fill and bold "
        "text, column widths auto-fit, freeze pane and auto-filter applied."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute output path. .xlsx extension added if missing."},
            "title": {"type": "string", "description": "Workbook title (used as Excel metadata)."},
            "theme": {"type": "string", "description": _THEME_DESC},
            "skill": {"type": "string", "description": _SKILL_DESC},
            "brief": {"type": "string", "description": _BRIEF_DESC},
            "source_material": {"description": _SOURCE_MATERIAL_DESC},
            "sheets": {
                "type": "array",
                "description": (
                    "List of sheet objects (mode 1): "
                    "{name:'Tab name', headers:['col1','col2'], rows:[[...],[...]], "
                    "freeze_header:true, auto_filter:true, column_widths:[12,30]}. "
                    "Only 'name' is mandatory; everything else has sensible defaults."
                ),
                "items": {"type": "object"}
            },
        },
        "required": ["path"],
    },
}

CREATE_HTML_REPORT_TOOL = {
    "name": "create_html_report",
    "description": (
        "Create a single-file HTML report with embedded CSS. Same schema, "
        "skill, and brief modes as create_docx. Renders cleanly in any "
        "browser, prints well, respects prefers-color-scheme. Use for "
        "shareable web reports or dashboards-as-documents."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute output path. .html extension added if missing."},
            "title": {"type": "string", "description": "Report title."},
            "subtitle": {"type": "string", "description": "Optional subtitle."},
            "author": {"type": "string", "description": "Optional author name."},
            "theme": {"type": "string", "description": _THEME_DESC},
            "skill": {"type": "string", "description": _SKILL_DESC},
            "brief": {"type": "string", "description": _BRIEF_DESC},
            "source_material": {"description": _SOURCE_MATERIAL_DESC},
            "content": {"type": "array", "description": _CONTENT_SCHEMA_DESC, "items": {"type": "object"}},
        },
        "required": ["path", "title"],
    },
}
