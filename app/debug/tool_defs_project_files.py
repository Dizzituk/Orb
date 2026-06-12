# FILE: app/debug/tool_defs_project_files.py
# Purpose: Project-file / pipeline / command tool schemas (read, list, search, write, edit, run).
# Called-by: app.debug.tool_definitions (facade); tool registry dispatch.
# Depends-on: none (pure schema constants).
# Last-renovated: 2026-06-11 (split from tool_definitions.py, Phase 4)
from __future__ import annotations

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
