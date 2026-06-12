# FILE: app/debug/tool_defs_device.py
# Purpose: Android emulator + desktop control tool schemas (ADB, gradle, screenshots, input).
# Called-by: app.debug.tool_definitions (facade); tool registry dispatch.
# Depends-on: none (pure schema constants).
# Last-renovated: 2026-06-11 (split from tool_definitions.py, Phase 4)
from __future__ import annotations

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


DESKTOP_SCREENSHOT_TOOL = {"name": "desktop_screenshot", "description": "Take a screenshot of the ASTRA desktop application or full screen. Optionally specify a window title to capture just that window.", "parameters": {"type": "object", "properties": {"window_title": {"type": "string", "description": "Optional window title to capture (e.g., 'Astra')."}}, "required": []}}


DESKTOP_CLICK_TOOL = {"name": "desktop_click", "description": "Click at screen coordinates. Use desktop_screenshot first to see current state. Only works within approved windows (ASTRA, Windows Sandbox, Android Studio).", "parameters": {"type": "object", "properties": {"x": {"type": "integer", "description": "X coordinate"}, "y": {"type": "integer", "description": "Y coordinate"}, "button": {"type": "string", "description": "'left', 'right', or 'middle'. Default: 'left'"}, "clicks": {"type": "integer", "description": "1=single, 2=double. Default: 1"}}, "required": ["x", "y"]}}


DESKTOP_TYPE_TOOL = {"name": "desktop_type", "description": "Type text at current cursor position. Click a text field first. Only works when an approved window is focused.", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "Text to type"}}, "required": ["text"]}}


DESKTOP_KEY_TOOL = {"name": "desktop_key", "description": "Press a key or combo (e.g., 'enter', 'tab', 'ctrl+a', 'alt+f4'). Only works when an approved window is focused.", "parameters": {"type": "object", "properties": {"key": {"type": "string", "description": "Key or combo"}}, "required": ["key"]}}


DESKTOP_SCROLL_TOOL = {"name": "desktop_scroll", "description": "Scroll at a position. Positive=up, negative=down. Only works within approved windows.", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "clicks": {"type": "integer", "description": "Scroll amount. Positive=up, negative=down."}}, "required": ["x", "y"]}}


DESKTOP_FIND_WINDOW_TOOL = {"name": "desktop_find_window", "description": "Find a window by title and get its position and size.", "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "Window title to search for (partial match)"}}, "required": ["title"]}}


DESKTOP_READ_SCREEN_TOOL = {"name": "desktop_read_screen", "description": "OCR the screen to extract visible text. Optionally specify a region.", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}, "width": {"type": "integer"}, "height": {"type": "integer"}}, "required": []}}
