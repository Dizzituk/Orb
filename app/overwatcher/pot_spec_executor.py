"""POT Spec Executor: Execute atomic tasks from a parsed POT spec.

Processes POT (Plan of Tasks) specs by:
1. Grouping tasks by file (batch edits per file)
2. Reading each file from sandbox
3. Applying line-specific search/replace on exact line numbers
4. Writing modified files back via sandbox
5. Verifying each change

SAFETY INVARIANT:
    - All file I/O goes through Windows Sandbox (sandbox_client)
    - NO direct host filesystem writes
    - Each edit is line-targeted, not global search/replace

v1.0 (2026-02-03): Initial implementation
    - Groups tasks by file for efficient batch processing
    - Line-number-targeted edits with content verification
    - Per-file and per-task progress tracking
    - Rollback tracking (backup original content)
v2.0 (2026-02-03): Idempotent re-run support
    - When search_term not found but replace_term already present on line,
      marks edit as 'already_applied' (idempotent success, no write needed)
    - Returns affected_files (all files in POT, whether modified or already applied)
      so build validation runs even on re-runs
    - New counters: tasks_already_applied, files_already_applied
    - Success = (files_modified > 0 OR files_already_applied > 0) AND files_failed == 0
v1.1 (2026-02-03): BOM fix
    - Fixed UTF-8 BOM corruption: Set-Content -Encoding UTF8 adds 3-byte
      BOM (EF BB BF) in PowerShell 5.1, breaking JSON/Vite parsers.
    - Now uses [System.IO.File]::WriteAllBytes() for BOM-free output.
"""

from __future__ import annotations

import base64
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.overwatcher.pot_spec_parser import POTAtomicTask, POTParseResult
from app.overwatcher.sandbox_client import (
    SandboxClient,
    SandboxError,
    get_sandbox_client,
)
from app.overwatcher._pot_spec_executor_utils_2 import MAX_CONSECUTIVE_ERRORS, POT_EXECUTOR_BUILD_ID, READ_TIMEOUT, WRITE_TIMEOUT, _build_write_command, _encode_base64, _read_file_via_sandbox, _write_file_via_sandbox
from app.overwatcher._pot_spec_executor_utils_3 import run_pot_spec_execution

logger = logging.getLogger(__name__)

# Build verification
print(f"[POT_EXECUTOR_LOADED] BUILD_ID={POT_EXECUTOR_BUILD_ID}")


# =============================================================================
# Constants
# =============================================================================


# =============================================================================
# Helpers
# =============================================================================


# =============================================================================
# Line-Targeted Edit Logic
# =============================================================================

def apply_line_edits(
    content: str,
    tasks: List[POTAtomicTask],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Apply line-targeted edits to file content.
    
    For each task:
    1. Find the line at task.line_number
    2. Verify the line contains the expected content
    3. Replace search_term with replace_term on that specific line
    
    Args:
        content: Original file content
        tasks: List of atomic tasks for this file
    
    Returns:
        Tuple of (modified_content, edit_results)
        edit_results is a list of dicts with task_id, status, details
    """
    lines = content.split('\n')
    edit_results: List[Dict[str, Any]] = []
    
    # Sort tasks by line number (descending) to avoid offset issues
    # when edits change line lengths
    sorted_tasks = sorted(tasks, key=lambda t: t.line_number, reverse=True)
    
    for task in sorted_tasks:
        result: Dict[str, Any] = {
            "task_id": task.task_id,
            "file_path": task.file_path,
            "line_number": task.line_number,
            "status": "pending",
            "original_line": None,
            "new_line": None,
            "error": None,
        }
        
        # Line numbers are 1-based, array is 0-based
        idx = task.line_number - 1
        
        # Validate line number
        if idx < 0 or idx >= len(lines):
            result["status"] = "error"
            result["error"] = (
                f"Line {task.line_number} out of range "
                f"(file has {len(lines)} lines)"
            )
            edit_results.append(result)
            logger.warning(
                "[pot_executor] %s: line %d out of range (%d lines)",
                task.file_path, task.line_number, len(lines)
            )
            continue
        
        original_line = lines[idx]
        result["original_line"] = original_line.strip()
        
        # Check if search_term exists on this line
        if task.search_term and task.search_term not in original_line:
            # Try case-insensitive check
            if task.search_term.lower() in original_line.lower():
                logger.info(
                    "[pot_executor] Case-insensitive match for '%s' on L%d",
                    task.search_term, task.line_number
                )
                # Find the actual case in the line
                lower_line = original_line.lower()
                lower_term = task.search_term.lower()
                pos = lower_line.find(lower_term)
                actual_term = original_line[pos:pos + len(task.search_term)]
                
                # Replace preserving the line's indentation and structure
                new_line = original_line[:pos] + task.replace_term + original_line[pos + len(actual_term):]
                lines[idx] = new_line
                result["new_line"] = new_line.strip()
                result["status"] = "success"
                
                logger.info(
                    "[pot_executor] L%d: '%s' → '%s' (case-adjusted)",
                    task.line_number, actual_term, task.replace_term
                )
            else:
                # Check if replace_term is already present (idempotent re-run)
                if task.replace_term and (
                    task.replace_term in original_line
                    or task.replace_term.lower() in original_line.lower()
                ):
                    result["status"] = "already_applied"
                    result["new_line"] = original_line.strip()
                    edit_results.append(result)
                    logger.info(
                        "[pot_executor] %s L%d: already applied ('%s' present)",
                        task.file_path, task.line_number,
                        task.replace_term,
                    )
                    continue

                result["status"] = "warning"
                result["error"] = (
                    f"Search term '{task.search_term}' not found on line {task.line_number}. "
                    f"Line content: '{original_line.strip()[:80]}'"
                )
                edit_results.append(result)
                logger.warning(
                    "[pot_executor] %s L%d: search term '%s' not found. Line: '%s'",
                    task.file_path, task.line_number,
                    task.search_term, original_line.strip()[:80]
                )
                continue
        else:
            # Direct replacement
            if task.search_term and task.replace_term:
                new_line = original_line.replace(task.search_term, task.replace_term, 1)
                lines[idx] = new_line
                result["new_line"] = new_line.strip()
                result["status"] = "success"
                
                logger.info(
                    "[pot_executor] L%d: '%s' → '%s'",
                    task.line_number, task.search_term, task.replace_term
                )
            elif not task.search_term:
                result["status"] = "skipped"
                result["error"] = "No search term available"
            else:
                result["status"] = "skipped"
                result["error"] = "No replace term available"
        
        edit_results.append(result)
    
    modified_content = '\n'.join(lines)
    return modified_content, edit_results


# =============================================================================
# Main Executor
# =============================================================================


__all__ = [
    "run_pot_spec_execution",
    "apply_line_edits",
]
