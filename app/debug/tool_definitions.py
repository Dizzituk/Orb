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
        "Use this when the user asks about their own files, documents, music, "
        "photos, etc. Example: query='learning roadmap' finds files with that "
        "name. You can also filter by category (documents, pictures, music, "
        "videos, desktop, screenshots) or extension (pdf, docx, mp3, etc)."
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
        "returns metadata only."
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
        "Build and install AstraBridge debug APK on the emulator. "
        "Compiles, packages, and deploys in one step. Use after fixing "
        "code to get the new version running on the emulator for testing."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
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



# Desktop Computer Use Tools
DESKTOP_SCREENSHOT_TOOL = {"name": "desktop_screenshot", "description": "Take a screenshot of the ASTRA desktop application or full screen. Optionally specify a window title to capture just that window.", "parameters": {"type": "object", "properties": {"window_title": {"type": "string", "description": "Optional window title to capture (e.g., 'Astra')."}}, "required": []}}
DESKTOP_CLICK_TOOL = {"name": "desktop_click", "description": "Click at screen coordinates. Use desktop_screenshot first to see current state. Only works within approved windows (ASTRA, Windows Sandbox, Android Studio).", "parameters": {"type": "object", "properties": {"x": {"type": "integer", "description": "X coordinate"}, "y": {"type": "integer", "description": "Y coordinate"}, "button": {"type": "string", "description": "'left', 'right', or 'middle'. Default: 'left'"}, "clicks": {"type": "integer", "description": "1=single, 2=double. Default: 1"}}, "required": ["x", "y"]}}
DESKTOP_TYPE_TOOL = {"name": "desktop_type", "description": "Type text at current cursor position. Click a text field first. Only works when an approved window is focused.", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "Text to type"}}, "required": ["text"]}}
DESKTOP_KEY_TOOL = {"name": "desktop_key", "description": "Press a key or combo (e.g., 'enter', 'tab', 'ctrl+a', 'alt+f4'). Only works when an approved window is focused.", "parameters": {"type": "object", "properties": {"key": {"type": "string", "description": "Key or combo"}}, "required": ["key"]}}
DESKTOP_SCROLL_TOOL = {"name": "desktop_scroll", "description": "Scroll at a position. Positive=up, negative=down. Only works within approved windows.", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "clicks": {"type": "integer", "description": "Scroll amount. Positive=up, negative=down."}}, "required": ["x", "y"]}}
DESKTOP_FIND_WINDOW_TOOL = {"name": "desktop_find_window", "description": "Find a window by title and get its position and size.", "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "Window title to search for (partial match)"}}, "required": ["title"]}}
DESKTOP_READ_SCREEN_TOOL = {"name": "desktop_read_screen", "description": "OCR the screen to extract visible text. Optionally specify a region.", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "width": {"type": "integer"}, "height": {"type": "integer"}}, "required": []}}

def get_universal_tools() -> List[dict]:
    """Tools available to ALL models - web search, etc.

    These are injected regardless of model trust level.
    """
    return [WEB_SEARCH_TOOL]


def get_phase1_tools() -> List[dict]:
    """Core tools: read + user file write access.

    v0.15.0: Added write_user_file and get_user_folders so ALL models
    can create/save files in the user's personal folders (Documents,
    Pictures, Desktop, etc.) while keeping codebase writes sandbox-only.
    """
    return [
        SEARCH_MY_FILES_TOOL,
        READ_USER_FILE_TOOL,
        WRITE_USER_FILE_TOOL,
        GET_USER_FOLDERS_TOOL,
        READ_FILE_TOOL,
        LIST_FILES_TOOL,
        READ_PIPELINE_STATE_TOOL,
        READ_LOGS_TOOL,
        SEARCH_FILES_TOOL,
    ]


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
