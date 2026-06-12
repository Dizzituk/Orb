# FILE: app/debug/tool_defs_user_files.py
# Purpose: User-file / disk-index tool schemas (search, read, write, folders, reindex, images).
# Called-by: app.debug.tool_definitions (facade); tool registry dispatch.
# Depends-on: none (pure schema constants).
# Last-renovated: 2026-06-11 (split from tool_definitions.py, Phase 4)
from __future__ import annotations

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
