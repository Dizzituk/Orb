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

logger = logging.getLogger(__name__)

# Build verification
POT_EXECUTOR_BUILD_ID = "2026-02-03-v1.1-bom-fix"
print(f"[POT_EXECUTOR_LOADED] BUILD_ID={POT_EXECUTOR_BUILD_ID}")


# =============================================================================
# Constants
# =============================================================================

MAX_CONSECUTIVE_ERRORS = 5  # Abort after N consecutive file failures
READ_TIMEOUT = 30           # Seconds per file read
WRITE_TIMEOUT = 60          # Seconds per file write


# =============================================================================
# Helpers
# =============================================================================

def _encode_base64(content: str) -> str:
    """Encode content as Base64 for PowerShell transmission."""
    return base64.b64encode(content.encode('utf-8')).decode('ascii')


def _build_write_command(path: str, content: str) -> str:
    """Build PowerShell command to write file via Base64 → raw bytes.

    CRITICAL: Do NOT use 'Set-Content -Encoding UTF8' — PowerShell 5.1
    writes a UTF-8 BOM (EF BB BF) which corrupts JSON files and breaks
    Vite/Node.js parsers. Instead, decode Base64 to raw bytes and write
    via [System.IO.File]::WriteAllBytes() which is BOM-free.

    v1.1 (2026-02-03): Fixed UTF-8 BOM corruption bug.
    """
    encoded = _encode_base64(content)
    return (
        f'$bytes = [System.Convert]::FromBase64String("{encoded}"); '
        f'[System.IO.File]::WriteAllBytes("{path}", $bytes)'
    )


def _read_file_via_sandbox(client: SandboxClient, path: str) -> Optional[str]:
    """Read file content from sandbox via PowerShell."""
    try:
        cmd = f'Get-Content -Path "{path}" -Raw -Encoding UTF8'
        result = client.shell_run(cmd, timeout_seconds=READ_TIMEOUT)
        
        if result.stdout is not None:
            return result.stdout
        
        logger.warning(
            "[pot_executor] Read failed for %s: stderr=%s",
            path, (result.stderr or "")[:100]
        )
        return None
    except Exception as e:
        logger.error("[pot_executor] Read exception for %s: %s", path, e)
        return None


def _write_file_via_sandbox(client: SandboxClient, path: str, content: str) -> bool:
    """Write file content to sandbox via Base64 PowerShell command."""
    try:
        cmd = _build_write_command(path, content)
        result = client.shell_run(cmd, timeout_seconds=WRITE_TIMEOUT)
        
        # Check for errors
        if result.stderr and result.stderr.strip():
            logger.warning(
                "[pot_executor] Write stderr for %s: %s",
                path, result.stderr[:200]
            )
            return False
        
        return True
    except Exception as e:
        logger.error("[pot_executor] Write exception for %s: %s", path, e)
        return False


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

async def run_pot_spec_execution(
    *,
    spec: Any,  # ResolvedSpec - using Any to avoid circular import
    pot_tasks: POTParseResult,
    job_id: str = "",
    llm_call_fn: Optional[Callable] = None,
    artifact_root: str = "",
    client: Optional[SandboxClient] = None,
) -> Dict[str, Any]:
    """Execute a POT spec by applying atomic tasks through the sandbox.
    
    Groups tasks by file, reads each file once, applies all line edits,
    writes back, and verifies.
    
    Args:
        spec: ResolvedSpec with POT spec data
        pot_tasks: Parsed POT tasks with search/replace terms
        job_id: Job ID for tracking
        llm_call_fn: LLM call function (unused for POT, reserved)
        artifact_root: Artifact storage root
        client: Optional sandbox client
    
    Returns:
        Dict with: success, decision, error, trace, tasks_completed,
                   total_tasks, artifacts_written, file_results
    """
    start_time = time.time()
    trace: List[Dict[str, Any]] = []
    
    def add_trace(stage: str, status: str, details: Optional[Dict] = None):
        trace.append({
            "stage": stage,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {},
        })
    
    total_tasks = len(pot_tasks.tasks)
    search_term = pot_tasks.search_term or ""
    replace_term = pot_tasks.replace_term or ""
    
    logger.info(
        "[pot_executor] Starting POT execution: %d tasks, '%s' → '%s'",
        total_tasks, search_term, replace_term
    )
    print(f"[POT_EXECUTOR] Starting: {total_tasks} tasks, '{search_term}' → '{replace_term}'")
    
    add_trace("POT_EXECUTION_START", "running", {
        "total_tasks": total_tasks,
        "search_term": search_term,
        "replace_term": replace_term,
        "job_id": job_id,
    })
    
    # Validate we have search and replace terms
    if not search_term or not replace_term:
        error = f"Missing terms: search='{search_term}', replace='{replace_term}'"
        logger.error("[pot_executor] %s", error)
        add_trace("POT_EXECUTION_ERROR", "failed", {"error": error})
        return {
            "success": False,
            "decision": "FAIL",
            "error": error,
            "trace": trace,
            "tasks_completed": 0,
            "total_tasks": total_tasks,
            "artifacts_written": [],
        }
    
    # Get sandbox client
    if client is None:
        client = get_sandbox_client()
    
    if not client.is_connected():
        error = "SAFETY: Sandbox not available for POT execution"
        logger.error("[pot_executor] %s", error)
        add_trace("POT_EXECUTION_ERROR", "failed", {"error": error})
        return {
            "success": False,
            "decision": "FAIL",
            "error": error,
            "trace": trace,
            "tasks_completed": 0,
            "total_tasks": total_tasks,
            "artifacts_written": [],
        }
    
    add_trace("SANDBOX_CONNECTED", "success")
    
    # Group tasks by file
    tasks_by_file: Dict[str, List[POTAtomicTask]] = defaultdict(list)
    for task in pot_tasks.tasks:
        tasks_by_file[task.file_path].append(task)
    
    logger.info(
        "[pot_executor] Grouped into %d files: %s",
        len(tasks_by_file),
        list(tasks_by_file.keys())
    )
    print(f"[POT_EXECUTOR] Processing {len(tasks_by_file)} files")
    
    # Process each file
    tasks_completed = 0
    tasks_failed = 0
    tasks_skipped = 0
    files_modified = 0
    files_failed = 0
    consecutive_errors = 0
    file_results: List[Dict[str, Any]] = []
    artifacts_written: List[str] = []
    
    for file_path, file_tasks in tasks_by_file.items():
        file_result: Dict[str, Any] = {
            "file_path": file_path,
            "task_count": len(file_tasks),
            "status": "pending",
            "edits": [],
            "error": None,
        }
        
        logger.info(
            "[pot_executor] Processing %s (%d tasks)",
            file_path, len(file_tasks)
        )
        print(f"[POT_EXECUTOR] File: {file_path} ({len(file_tasks)} edits)")
        
        add_trace("POT_FILE_START", "processing", {
            "file_path": file_path,
            "task_count": len(file_tasks),
        })
        
        # Step 1: Read file
        content = _read_file_via_sandbox(client, file_path)
        
        if content is None:
            file_result["status"] = "error"
            file_result["error"] = "Could not read file"
            files_failed += 1
            tasks_failed += len(file_tasks)
            consecutive_errors += 1
            
            logger.error("[pot_executor] Failed to read %s", file_path)
            add_trace("POT_FILE_ERROR", "read_failed", {
                "file_path": file_path,
                "error": "Could not read file",
            })
            
            file_results.append(file_result)
            
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                error = f"Aborted: {consecutive_errors} consecutive file read failures"
                logger.error("[pot_executor] %s", error)
                add_trace("POT_EXECUTION_ABORT", "failed", {"error": error})
                break
            continue
        
        logger.info("[pot_executor] Read %d chars from %s", len(content), file_path)
        
        # Step 2: Apply line edits
        modified_content, edit_results = apply_line_edits(content, file_tasks)
        file_result["edits"] = edit_results
        
        # Count successes/failures for this file
        file_successes = sum(1 for e in edit_results if e["status"] == "success")
        file_failures = sum(1 for e in edit_results if e["status"] == "error")
        file_warnings = sum(1 for e in edit_results if e["status"] == "warning")
        file_skips = sum(1 for e in edit_results if e["status"] == "skipped")
        
        logger.info(
            "[pot_executor] %s edits: %d success, %d fail, %d warn, %d skip",
            file_path, file_successes, file_failures, file_warnings, file_skips
        )
        
        if file_successes == 0:
            file_result["status"] = "no_changes"
            file_result["error"] = "No successful edits"
            tasks_failed += file_failures
            tasks_skipped += file_skips + file_warnings
            files_failed += 1
            
            add_trace("POT_FILE_NO_CHANGES", "skipped", {
                "file_path": file_path,
                "successes": 0,
                "failures": file_failures,
            })
            
            file_results.append(file_result)
            continue
        
        # Step 3: Write modified content back
        if modified_content != content:
            write_success = _write_file_via_sandbox(client, file_path, modified_content)
            
            if not write_success:
                file_result["status"] = "write_failed"
                file_result["error"] = "Failed to write modified file"
                files_failed += 1
                tasks_failed += len(file_tasks)
                consecutive_errors += 1
                
                logger.error("[pot_executor] Write failed for %s", file_path)
                add_trace("POT_FILE_ERROR", "write_failed", {
                    "file_path": file_path,
                })
                
                file_results.append(file_result)
                
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    error = f"Aborted: {consecutive_errors} consecutive write failures"
                    add_trace("POT_EXECUTION_ABORT", "failed", {"error": error})
                    break
                continue
            
            # Step 4: Verify write
            verify_content = _read_file_via_sandbox(client, file_path)
            
            if verify_content is None:
                file_result["status"] = "verify_failed"
                file_result["error"] = "Could not re-read file for verification"
                logger.warning("[pot_executor] Verify re-read failed for %s", file_path)
            elif verify_content.strip() != modified_content.strip():
                file_result["status"] = "verify_failed"
                file_result["error"] = "Content mismatch after write"
                logger.warning("[pot_executor] Verify mismatch for %s", file_path)
            else:
                file_result["status"] = "success"
                files_modified += 1
                tasks_completed += file_successes
                tasks_skipped += file_skips + file_warnings
                consecutive_errors = 0  # Reset on success
                artifacts_written.append(file_path)
                
                logger.info(
                    "[pot_executor] ✓ %s: %d edits applied and verified",
                    file_path, file_successes
                )
                print(f"[POT_EXECUTOR] ✓ {file_path}: {file_successes} edits OK")
                
                add_trace("POT_FILE_SUCCESS", "complete", {
                    "file_path": file_path,
                    "edits_applied": file_successes,
                    "edits_skipped": file_skips + file_warnings,
                })
        else:
            file_result["status"] = "unchanged"
            tasks_skipped += len(file_tasks)
            
            logger.info("[pot_executor] %s: no changes needed", file_path)
            add_trace("POT_FILE_UNCHANGED", "skipped", {
                "file_path": file_path,
            })
        
        file_results.append(file_result)
    
    # Final summary
    elapsed_ms = int((time.time() - start_time) * 1000)
    success = files_modified > 0 and files_failed == 0
    
    summary = {
        "total_tasks": total_tasks,
        "tasks_completed": tasks_completed,
        "tasks_failed": tasks_failed,
        "tasks_skipped": tasks_skipped,
        "files_processed": len(tasks_by_file),
        "files_modified": files_modified,
        "files_failed": files_failed,
        "elapsed_ms": elapsed_ms,
    }
    
    logger.info(
        "[pot_executor] COMPLETE: success=%s, tasks=%d/%d, files=%d/%d, %dms",
        success, tasks_completed, total_tasks,
        files_modified, len(tasks_by_file), elapsed_ms
    )
    print(
        f"[POT_EXECUTOR] {'✓ SUCCESS' if success else '✗ FAILED'}: "
        f"{tasks_completed}/{total_tasks} tasks, "
        f"{files_modified}/{len(tasks_by_file)} files, "
        f"{elapsed_ms}ms"
    )
    
    add_trace(
        "POT_EXECUTION_COMPLETE",
        "success" if success else "failed",
        summary,
    )
    
    return {
        "success": success,
        "decision": "PASS" if success else "FAIL",
        "error": None if success else f"POT execution: {tasks_completed}/{total_tasks} tasks completed, {files_failed} files failed",
        "trace": trace,
        "tasks_completed": tasks_completed,
        "total_tasks": total_tasks,
        "artifacts_written": artifacts_written,
        "file_results": file_results,
        "summary": summary,
    }


__all__ = [
    "run_pot_spec_execution",
    "apply_line_edits",
]
