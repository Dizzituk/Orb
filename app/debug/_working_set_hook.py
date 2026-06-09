# FILE: app/debug/_working_set_hook.py
"""
Auto-registration of tool-touched files into the project working set.

Called by execute_tool after every successful tool execution.  Maps
tool names to (path-extraction, action-verb) pairs so the working set
learns which files the conversation has just touched, without each
individual tool function needing to know the working set exists.

The mapping is deliberately narrow — only tools that read or write
user-visible files are tracked.  Codebase reads, sandbox commands,
emulator taps, social-media posts, etc. don't go in the working set
because they're not "files the user is working with in this project".

v1.0 (2026-05-24): Initial implementation.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

from app.memory import working_set

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL -> (path_extractor, action_verb) MAPPING
# =============================================================================
# path_extractor: function(params) -> Optional[str] returning the file
#   path the tool touched.  None means "ignore this call".
# action_verb: short label for the working-set entry: 'read', 'wrote',
#   'created', 'edited'.

def _path_from_params(params: Dict[str, Any]) -> Optional[str]:
    """Common case: params['path'] is the file path."""
    p = params.get("path")
    return p if isinstance(p, str) and p else None


def _source_from_params(params: Dict[str, Any]) -> Optional[str]:
    """For move_file: track the new location (destination)."""
    p = params.get("destination")
    return p if isinstance(p, str) and p else None


_TRACKED_TOOLS: Dict[str, Tuple[Callable[[Dict[str, Any]], Optional[str]], str]] = {
    # Reads
    "read_user_file":    (_path_from_params, "read"),
    "read_image":        (_path_from_params, "read"),
    # Writes
    "write_user_file":   (_path_from_params, "wrote"),
    "create_docx":       (_path_from_params, "created"),
    "create_pdf":        (_path_from_params, "created"),
    "create_xlsx":       (_path_from_params, "created"),
    "create_html_report": (_path_from_params, "created"),
    # Edits on host (the desktop image-routing path uses these)
    "edit_file":         (_path_from_params, "edited"),
    "write_file":        (_path_from_params, "wrote"),
    # Moves (destination is what survives)
    "move_file":         (_source_from_params, "wrote"),
}


def register_if_tracked(
    tool_name: str,
    params: Dict[str, Any],
    result: str,
) -> None:
    """Post-execution hook.  Called from execute_tool after a successful
    tool run.  Looks up the tool in _TRACKED_TOOLS and, if it's one we
    care about, registers the touched file in the current project's
    working set.

    Errors here are swallowed — working-set registration must never
    break a tool call."""
    try:
        entry = _TRACKED_TOOLS.get(tool_name)
        if entry is None:
            return
        path_fn, action = entry
        path = path_fn(params)
        if not path:
            return
        # Skip registration on obvious failure results.  Tool functions
        # return error strings rather than raising; we don't want to
        # cache a path that doesn't exist on disk.
        result_lower = (result or "")[:200].lower()
        if any(
            marker in result_lower
            for marker in ("error:", "failed:", "not found", "no such file")
        ):
            return
        project_id = working_set.get_current_project_id()
        model = working_set.get_current_model()
        if project_id:
            working_set.register_file(
                project_id=project_id,
                path=path,
                action=action,
                model=model,
            )
    except Exception as e:
        logger.warning(
            "[working_set_hook] Registration failed for %s: %s",
            tool_name, e,
        )
