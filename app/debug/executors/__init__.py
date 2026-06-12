# FILE: app/debug/executors/__init__.py
# Purpose: Executor package: groups the action_executor's tool handlers by concern.
# Called-by: app.debug.action_executor
# Depends-on: app.debug.executors.emulator, app.debug.executors.file_ops, app.debug.executors.filesystem, app.debug.executors.flow_actions (+5 more)
# Last-renovated: 2026-06-11
"""
Executor package: groups the action_executor's tool handlers by concern.

Public API: re-exports every execute_* function so any code that
historically imported from app.debug.action_executor (e.g.
behaviour_verifier, subagent_runner) continues to work without changes.

Modules:
  - _paths        : shared path constants + helpers
  - filesystem    : read/list/search/write/edit/run_command
  - user_files    : user file ops, manifest tools, vision, web, cloud
  - emulator      : ADB/gradle/pipeline-state/logs wrappers
"""
from app.debug.executors.filesystem import (
    execute_read_file,
    execute_list_files,
    execute_search_files,
    execute_write_file,
    execute_edit_file,
    execute_run_command,
)
from app.debug.executors.user_files import (
    execute_search_my_files,
    execute_read_user_file,
    execute_write_user_file,
    execute_get_user_folders,
    execute_rescan_manifest,
    execute_reindex_file,
    execute_search_disk_live,
    execute_read_image,
    execute_web_search,
    execute_cloud_upload,
    execute_cloud_list,
)
from app.debug.executors.file_ops import (
    execute_create_folder,
    execute_move_file,
    execute_move_files_batch,
)
from app.debug.executors.styled_files import (
    execute_create_docx,
    execute_create_pdf,
    execute_create_xlsx,
    execute_create_html_report,
)
from app.debug.executors.emulator import (
    execute_emulator_screenshot,
    execute_emulator_ui_dump,
    execute_emulator_tap,
    execute_emulator_type,
    execute_emulator_key,
    execute_gradle_build,
    execute_gradle_install,
    execute_app_restart,
    execute_get_crash_log,
    execute_read_pipeline_state,
    execute_read_logs,
)
from app.debug.executors.memory_tools import (
    execute_save_to_memory,
    execute_update_memory,
    execute_forget_memory,
    execute_search_memory,
    execute_save_residence,
)
from app.debug.executors.web_automation import (
    execute_web_list_sessions,
    execute_web_open_session,
    execute_web_current_state,
    execute_web_dom_snapshot,
    execute_web_navigate,
    execute_web_click,
    execute_web_type,
    execute_web_scroll,
    execute_web_extract_text,
    execute_web_screenshot,
    execute_web_vision_check,
    execute_web_upload_file,
    execute_system_keys,
)
from app.debug.executors.social_actions import (
    execute_meta_post,
)
from app.debug.executors.flow_actions import (
    execute_flow_run,
    execute_flow_save,
    execute_flow_inspect,
)

__all__ = [
    # Filesystem
    "execute_read_file", "execute_list_files", "execute_search_files",
    "execute_write_file", "execute_edit_file", "execute_run_command",
    # User files
    "execute_search_my_files", "execute_read_user_file",
    "execute_write_user_file", "execute_get_user_folders",
    "execute_rescan_manifest", "execute_reindex_file",
    "execute_search_disk_live", "execute_read_image",
    "execute_web_search", "execute_cloud_upload", "execute_cloud_list",
    # Emulator / pipeline / logs
    "execute_emulator_screenshot", "execute_emulator_ui_dump",
    "execute_emulator_tap", "execute_emulator_type", "execute_emulator_key",
    "execute_gradle_build", "execute_gradle_install",
    "execute_app_restart", "execute_get_crash_log",
    "execute_read_pipeline_state", "execute_read_logs",
    # File ops
    "execute_create_folder", "execute_move_file", "execute_move_files_batch",
    # Styled files
    "execute_create_docx", "execute_create_pdf",
    "execute_create_xlsx", "execute_create_html_report",
    # Memory tools
    "execute_save_to_memory", "execute_update_memory",
    "execute_forget_memory", "execute_search_memory",
    "execute_save_residence",
    # Web automation (logged-in browser sessions)
    "execute_web_list_sessions", "execute_web_open_session",
    "execute_web_current_state", "execute_web_dom_snapshot",
    "execute_web_navigate", "execute_web_click", "execute_web_type",
    "execute_web_scroll", "execute_web_extract_text",
    "execute_web_screenshot", "execute_web_vision_check",
    "execute_web_upload_file",
    "execute_system_keys",
    # Social media APIs (first-party HTTP integrations)
    "execute_meta_post",
    # Flow memory (cached multi-step interaction patterns with
    # per-step verification and failure isolation)
    "execute_flow_run",
    "execute_flow_save",
    "execute_flow_inspect",
]
