# FILE: app/debug/tool_definitions.py
# Purpose: facade for the debug-assistant tool schema catalog + tier selector functions.
# Called-by: app.debug executors/registry, app.llm tool wiring (all original imports preserved).
# Depends-on: tool_defs_user_files, tool_defs_project_files, tool_defs_device,
#             tool_defs_web_social, tool_defs_documents.
# Last-renovated: 2026-06-11 (was 48KB flat catalog; split into 5 domain modules, Phase 4)
from __future__ import annotations

from typing import List

from app.debug.tool_defs_user_files import (  # noqa: F401 - re-exported
    SEARCH_MY_FILES_TOOL,
    READ_USER_FILE_TOOL,
    WRITE_USER_FILE_TOOL,
    GET_USER_FOLDERS_TOOL,
    RESCAN_MANIFEST_TOOL,
    REINDEX_FILE_TOOL,
    SEARCH_DISK_LIVE_TOOL,
    READ_IMAGE_TOOL,
    CREATE_FOLDER_TOOL,
    MOVE_FILE_TOOL,
    MOVE_FILES_BATCH_TOOL,
)
from app.debug.tool_defs_project_files import (  # noqa: F401 - re-exported
    READ_FILE_TOOL,
    LIST_FILES_TOOL,
    READ_PIPELINE_STATE_TOOL,
    READ_LOGS_TOOL,
    SEARCH_FILES_TOOL,
    WRITE_FILE_TOOL,
    EDIT_FILE_TOOL,
    RUN_COMMAND_TOOL,
)
from app.debug.tool_defs_device import (  # noqa: F401 - re-exported
    SCREENSHOT_TOOL,
    UI_DUMP_TOOL,
    EMULATOR_TAP_TOOL,
    EMULATOR_TYPE_TOOL,
    EMULATOR_KEY_TOOL,
    GRADLE_BUILD_TOOL,
    GRADLE_INSTALL_TOOL,
    APP_RESTART_TOOL,
    CRASH_LOG_TOOL,
    DESKTOP_SCREENSHOT_TOOL,
    DESKTOP_CLICK_TOOL,
    DESKTOP_TYPE_TOOL,
    DESKTOP_KEY_TOOL,
    DESKTOP_SCROLL_TOOL,
    DESKTOP_FIND_WINDOW_TOOL,
    DESKTOP_READ_SCREEN_TOOL,
)
from app.debug.tool_defs_web_social import (  # noqa: F401 - re-exported
    WEB_SEARCH_TOOL,
    CLOUD_UPLOAD_TOOL,
    CLOUD_LIST_TOOL,
    META_POST_TOOL,
    FLOW_RUN_TOOL,
    FLOW_SAVE_TOOL,
    FLOW_INSPECT_TOOL,
)
from app.debug.tool_defs_documents import (  # noqa: F401 - re-exported
    CREATE_DOCX_TOOL,
    CREATE_PDF_TOOL,
    CREATE_XLSX_TOOL,
    CREATE_HTML_REPORT_TOOL,
)


def get_universal_tools() -> List[dict]:
    """Tools available to ALL models - web search + memory access.

    These are injected regardless of model trust level. Memory tools
    are universal because the chat LLM needs to be able to write to
    memory in every turn, on every model. Lifestyle logging tools
    (log_weight, log_nutrition, log_workout, set_goal + the matching
    reads) are universal for the same reason: the user logs weight,
    food, and workouts by voice in ordinary chat, so every model needs
    them in front of it — otherwise it falls back to save_to_memory and
    the entry never reaches the lifestyle DB the Health dashboard reads.
    """
    from app.debug.memory_tool_definitions import get_memory_tools
    from app.debug.gemini_lifestyle_tools import LIFESTYLE_TOOL_DECLARATIONS
    # Work-day + expense logging is voice-in-ordinary-chat for this user
    # (2026-06-11): without these declared, the Health-panel model confabulated
    # "Logged" for shift closes and fuel. Subset only — the screenshot/folder
    # flows stay in the Debug-tab loop.
    from app.debug.gemini_finance_tools import FINANCE_TOOL_DECLARATIONS
    from app.debug.gemini_expense_tools import EXPENSE_TOOL_DECLARATIONS
    _work_names = {"start_work_day", "finish_work_day", "reconcile_work_day",
                   "get_work_day", "set_personal_mileage"}
    work_decls = [d for d in FINANCE_TOOL_DECLARATIONS if d.get("name") in _work_names]
    return ([WEB_SEARCH_TOOL] + get_memory_tools() + LIFESTYLE_TOOL_DECLARATIONS
            + work_decls + EXPENSE_TOOL_DECLARATIONS)


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
