"""
Quarantine skip check for architecture executor tasks.

Detects files that have already been quarantined by package_quarantine
and skips them during task processing to avoid redundant operations.

Extracted from step_process_task.py for file size compliance.

v5.13: Original implementation.
"""
from __future__ import annotations

import logging
from typing import Dict

from ..sandbox_client import SandboxClient
from .execution_state import ExecutionContext
from .parsing import extract_section_for_file
from .path_resolution import _resolve_multi_root_path

logger = logging.getLogger(__name__)


def check_quarantine_skip(
    rel_path: str,
    abs_path: str,
    file_info: dict,
    ctx: ExecutionContext,
    client: SandboxClient,
) -> bool:
    """Check if a MODIFY/DELETE target was already quarantined.

    Returns True if the task should be skipped (file already handled).
    """
    rel_norm = rel_path.replace("\\", "/")
    desc_lower = file_info.get("description", "").lower()

    is_delete = any(kw in desc_lower for kw in [
        "delete", "remove entirely", "superseded", "replaced by",
        "no longer exists", "remove this file",
    ])
    if not is_delete:
        section = extract_section_for_file(ctx.architecture_content, rel_path)
        if section:
            sec_lower = section.lower()
            is_delete = any(phrase in sec_lower for phrase in [
                "delete this file", "remove entirely", "removed entirely",
                "file is removed", "this file is superseded",
                "this file must be deleted", "must not exist",
                "no longer exists", "must be removed",
            ])

    if not is_delete:
        return False

    # Build quarantine path
    path_parts = rel_norm.rsplit("/", 1)
    if len(path_parts) == 2:
        q_abs = _resolve_multi_root_path(
            f"{path_parts[0]}/.quarantined/{path_parts[1]}", ctx.sandbox_base,
        )
    else:
        q_abs = _resolve_multi_root_path(
            f".quarantined/{rel_norm}", ctx.sandbox_base,
        )

    try:
        q_check = client.shell_run(
            f'if (Test-Path -Path "{q_abs}" -PathType Leaf) '
            f'{{ "QUARANTINED" }} else {{ "NONE" }}',
            timeout_seconds=10,
        )
        if not (q_check.stdout and "QUARANTINED" in q_check.stdout):
            return False

        orig_check = client.shell_run(
            f'if (Test-Path -Path "{abs_path}" -PathType Leaf) '
            f'{{ "EXISTS" }} else {{ "GONE" }}',
            timeout_seconds=10,
        )
        if orig_check.stdout and "GONE" in orig_check.stdout:
            logger.info(
                "[arch_exec] v5.13 QUARANTINE SKIP: %s — file already quarantined at %s",
                rel_path, q_abs,
            )
            print(
                f"[ARCH_EXEC] v5.13 SKIP (quarantined): {rel_path} — "
                f"already moved to .quarantined/ by package_quarantine"
            )
            ctx.add_trace("QUARANTINE_SKIP", "success", {
                "path": rel_path,
                "quarantine_path": q_abs,
                "reason": "File quarantined by package_quarantine, no action needed",
            })
            return True
    except Exception as e:
        logger.warning(
            "[arch_exec] v5.13 Quarantine check failed for %s: %s — proceeding normally",
            rel_path, e,
        )
    return False