# FILE: app/debug/tool_defs_web_social.py
# Purpose: Web search / cloud / social / web-flow tool schemas.
# Called-by: app.debug.tool_definitions (facade); tool registry dispatch.
# Depends-on: none (pure schema constants).
# Last-renovated: 2026-06-11 (split from tool_definitions.py, Phase 4)
from __future__ import annotations

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
