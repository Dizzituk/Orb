# FILE: app/debug/action_executor.py
# Purpose: Action Executor: Translates LLM tool calls into real operations.
# Called-by: app.debug.executors.flow_actions, app.debug.orchestrator.behaviour_verifier, app.debug.orchestrator.subagent_runner, app.llm.chat_tool_loop
# Depends-on: app.debug._working_set_hook, app.debug.executors, app.debug.executors._paths, app.debug.gemini_lifestyle_tools
# Last-renovated: 2026-06-11
"""
Action Executor: Translates LLM tool calls into real operations.

THIN DISPATCHER (v10.0, 2026-04-15):
This module used to be a 56 KB monolith holding 27 execute_* functions
plus all path/sandbox helpers. It has been split into:

  app/debug/executors/
    _paths.py        - shared path constants + helpers
    filesystem.py    - read/list/search/write/edit/run_command
    user_files.py    - user file ops, manifest, vision, web, cloud
    file_ops.py      - create_folder / move_file / move_files_batch
    styled_files.py  - create_docx / create_pdf / create_xlsx / create_html_report
    emulator.py      - ADB/gradle/pipeline-state/logs
    memory_tools.py  - save/update/forget/search memory + save_residence

Public API stays identical:
  - execute_tool(tool_name, params)   -- the only entry point any caller uses
  - TOOL_HANDLERS                     -- the dispatch dict

If you need to add a new tool:
  1. Add the executor function in the appropriate executors/* module.
  2. Re-export it from executors/__init__.py.
  3. Register it in the TOOL_HANDLERS dict below.
  4. Add a *_TOOL definition in app/debug/tool_definitions.py and
     wire it into get_phase1_tools / get_phase2_tools as appropriate.

Older history (sandbox routing rules, write protection, etc.) lives
in the docstring of executors/_paths.py.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

# Re-export every executor (back-compat for any code that imported
# specific execute_* functions directly from this module).
from app.debug.executors import (  # noqa: F401
    # filesystem
    execute_read_file,
    execute_list_files,
    execute_search_files,
    execute_write_file,
    execute_edit_file,
    execute_run_command,
    # user files
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
    # file ops
    execute_create_folder,
    execute_move_file,
    execute_move_files_batch,
    # styled files
    execute_create_docx,
    execute_create_pdf,
    execute_create_xlsx,
    execute_create_html_report,
    # emulator / pipeline / logs
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
    # memory tools
    execute_save_to_memory,
    execute_update_memory,
    execute_forget_memory,
    execute_search_memory,
    execute_save_residence,
    # web automation (logged-in browser sessions)
    execute_web_list_sessions,
    execute_web_open_session,
    execute_web_current_state,
    execute_web_dom_snapshot,
    execute_web_navigate,
    execute_web_wait_for,
    execute_web_click,
    execute_web_type,
    execute_web_scroll,
    execute_web_extract_text,
    execute_web_screenshot,
    execute_web_vision_check,
    execute_web_coursera_health,
    execute_web_coursera_progress,
    execute_coursera_resume,
    execute_coursera_read_lesson,
    execute_coursera_next_item,
    execute_web_upload_file,
    execute_system_keys,
    # social media APIs
    execute_meta_post,
    # flow memory
    execute_flow_run,
    execute_flow_save,
    execute_flow_inspect,
)

# Re-export path helpers under their historical names for any external
# code that reached into action_executor for them.
from app.debug.executors._paths import (  # noqa: F401
    SANDBOX_CONTROLLER_URL,
    is_host_only as _is_host_only,
    is_host_write_blocked as _is_host_write_blocked,
    resolve_sandbox_path as _resolve_sandbox_path,
    sandbox_health_status as _sandbox_health_status,
    read_host_file as _read_host_file,
)

# WI-2 (2026-06-17, sandbox-visual-selfheal): host run_command is wrapped so every
# command + its output is echoed into a persistent visible "ASTRA console" window.
# The wrapper is transparent -- it returns exactly what execute_run_command returns.
from app.debug.host_console import run_command_visible

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL DISPATCHER
# =============================================================================

TOOL_HANDLERS = {
    # User file tools
    "search_my_files":     execute_search_my_files,
    "read_user_file":      execute_read_user_file,
    "write_user_file":     execute_write_user_file,
    "get_user_folders":    execute_get_user_folders,
    # Read (sandbox, except architecture/logs/user-home on host)
    "read_file":           execute_read_file,
    "list_files":          execute_list_files,
    "read_pipeline_state": execute_read_pipeline_state,
    "read_logs":           execute_read_logs,
    "search_files":        execute_search_files,
    # Write (sandbox or host depending on path)
    "write_file":          execute_write_file,
    "edit_file":           execute_edit_file,
    "run_command":         run_command_visible,   # visible-console wrapper (WI-2)
    # ADB emulator tools
    "emulator_screenshot": execute_emulator_screenshot,
    "emulator_ui_dump":    execute_emulator_ui_dump,
    "emulator_tap":        execute_emulator_tap,
    "emulator_type":       execute_emulator_type,
    "emulator_key":        execute_emulator_key,
    "gradle_build":        execute_gradle_build,
    "gradle_install":      execute_gradle_install,
    "app_restart":         execute_app_restart,
    "get_crash_log":       execute_get_crash_log,
    # Universal tools (all models)
    "web_search":          execute_web_search,
    # Cloud storage
    "cloud_upload":        execute_cloud_upload,
    "cloud_list":          execute_cloud_list,
    # Manifest tools
    "rescan_manifest":     execute_rescan_manifest,
    "reindex_file":        execute_reindex_file,
    "search_disk_live":    execute_search_disk_live,
    # File ops
    "create_folder":       execute_create_folder,
    "move_file":           execute_move_file,
    "move_files_batch":    execute_move_files_batch,
    # Styled file creation
    "create_docx":         execute_create_docx,
    "create_pdf":          execute_create_pdf,
    "create_xlsx":         execute_create_xlsx,
    "create_html_report":  execute_create_html_report,
    # Vision
    "read_image":          execute_read_image,
    # Memory tools (tool-based save replaces background extraction for live facts)
    "save_to_memory":      execute_save_to_memory,
    "update_memory":       execute_update_memory,
    "forget_memory":       execute_forget_memory,
    "search_memory":       execute_search_memory,
    "save_residence":      execute_save_residence,
    # Web automation (browser sessions: Coursera, Meta, TikTok, YouTube, WordPress)
    "web_list_sessions":   execute_web_list_sessions,
    "web_open_session":    execute_web_open_session,
    "web_current_state":   execute_web_current_state,
    "web_dom_snapshot":    execute_web_dom_snapshot,
    "web_navigate":        execute_web_navigate,
    "web_wait_for":        execute_web_wait_for,
    "web_click":           execute_web_click,
    "web_type":            execute_web_type,
    "web_scroll":          execute_web_scroll,
    "web_extract_text":    execute_web_extract_text,
    "web_screenshot":      execute_web_screenshot,
    "web_vision_check":    execute_web_vision_check,
    # Coursera composites: login health check + vision-first progress
    # read (completion ticks are visual — text extraction can't see them)
    "web_coursera_health":   execute_web_coursera_health,
    "web_coursera_progress": execute_web_coursera_progress,
    # Coursera study loop (hands-free: resume / read lesson / next item;
    # selector-map driver, quizzes announced and never attempted)
    "coursera_resume":       execute_coursera_resume,
    "coursera_read_lesson":  execute_coursera_read_lesson,
    "coursera_next_item":    execute_coursera_next_item,
    "web_upload_file":     execute_web_upload_file,
    "system_keys":         execute_system_keys,
    # Social media APIs (first-party publish path; preferred over
    # browser flow whenever the platform has an API equivalent)
    "meta_post":           execute_meta_post,
    # Flow memory (cached, verified multi-step interaction patterns)
    "flow_run":            execute_flow_run,
    "flow_save":           execute_flow_save,
    "flow_inspect":        execute_flow_inspect,
}

# Lifestyle tools (nutrition, workouts, weight, goals). Their executors live
# in gemini_lifestyle_tools (shared with the Gemini multimodal path) and
# already have the (params) -> str shape this dispatcher expects, so they
# merge straight in. Wiring them here is what lets the TEXT chat path log to
# the lifestyle DB instead of falling back to save_to_memory.
try:
    from app.debug.gemini_lifestyle_tools import LIFESTYLE_TOOL_EXECUTORS
    TOOL_HANDLERS.update(LIFESTYLE_TOOL_EXECUTORS)
except Exception as _life_err:  # pragma: no cover - defensive
    logger.warning("[action_executor] lifestyle tools not wired: %s", _life_err)

# Work-day + expense logging by voice in ordinary chat (2026-06-11): the
# Health-panel conversation confidently claimed "Logged" while having no
# finance tools available to call — pure confabulation. Same universality
# argument as lifestyle above: the user closes his delivery shift and logs
# fuel by voice in whatever chat is open, so every model needs these.
try:
    from app.debug.gemini_finance_tools import FINANCE_TOOL_EXECUTORS
    from app.debug.gemini_expense_tools import EXPENSE_TOOL_EXECUTORS
    TOOL_HANDLERS.update(FINANCE_TOOL_EXECUTORS)
    TOOL_HANDLERS.update(EXPENSE_TOOL_EXECUTORS)
except Exception as _fin_err:  # pragma: no cover - defensive
    logger.warning("[action_executor] work/expense tools not wired: %s", _fin_err)


async def execute_tool(tool_name: str, params: Dict[str, Any]) -> str:
    """Execute a tool call from the LLM."""
    handler = TOOL_HANDLERS.get(tool_name)
    if not handler:
        return f"Unknown tool: {tool_name}"

    logger.info("[action_executor] Executing tool: %s with params: %s", tool_name, params)
    try:
        result = await handler(params)
        logger.info("[action_executor] Tool %s completed (%d chars)", tool_name, len(result))
        # v1.0 (2026-05-24): Working-set registration hook.  Records
        # the touched file into the per-project working set so future
        # turns (and other models) see it in their context block.
        # Failures are swallowed inside the hook - never break the call.
        try:
            from app.debug._working_set_hook import register_if_tracked
            register_if_tracked(tool_name, params, result)
        except Exception as _hook_err:
            logger.debug("[action_executor] working-set hook error: %s", _hook_err)
        return result
    except Exception as e:
        logger.error("[action_executor] Tool %s failed: %s", tool_name, e)
        return f"Tool execution error: {e}"
