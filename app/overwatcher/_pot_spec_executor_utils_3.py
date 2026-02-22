from __future__ import annotations
import logging
import time
from app.overwatcher._pot_spec_executor_utils_2 import MAX_CONSECUTIVE_ERRORS, _read_file_via_sandbox, _write_file_via_sandbox
from app.overwatcher.pot_spec_parser import POTAtomicTask, POTParseResult
from app.overwatcher.sandbox_client import SandboxClient, get_sandbox_client
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


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
    files_already_applied = 0
    tasks_already_applied = 0
    consecutive_errors = 0
    file_results: List[Dict[str, Any]] = []
    artifacts_written: List[str] = []
    affected_files: List[str] = []
    
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
        file_already = sum(1 for e in edit_results if e["status"] == "already_applied")
        
        logger.info(
            "[pot_executor] %s edits: %d success, %d already, %d fail, %d warn, %d skip",
            file_path, file_successes, file_already, file_failures, file_warnings, file_skips
        )
        
        if file_successes == 0 and file_already == 0:
            # No successful edits and nothing already applied → actual failure
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
        
        if file_successes == 0 and file_already > 0:
            # All edits already applied — idempotent success, no write needed
            file_result["status"] = "already_applied"
            tasks_already_applied += file_already
            tasks_skipped += file_skips + file_warnings
            files_already_applied += 1
            affected_files.append(file_path)
            consecutive_errors = 0
            
            logger.info(
                "[pot_executor] ✓ %s: %d edits already applied (idempotent)",
                file_path, file_already
            )
            print(f"[POT_EXECUTOR] ✓ {file_path}: {file_already} edits already applied")
            
            add_trace("POT_FILE_ALREADY_APPLIED", "idempotent", {
                "file_path": file_path,
                "edits_already_applied": file_already,
                "edits_skipped": file_skips + file_warnings,
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
                tasks_already_applied += file_already
                tasks_skipped += file_skips + file_warnings
                consecutive_errors = 0  # Reset on success
                artifacts_written.append(file_path)
                affected_files.append(file_path)
                
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
    success = (files_modified > 0 or files_already_applied > 0) and files_failed == 0
    
    summary = {
        "total_tasks": total_tasks,
        "tasks_completed": tasks_completed,
        "tasks_already_applied": tasks_already_applied,
        "tasks_failed": tasks_failed,
        "tasks_skipped": tasks_skipped,
        "files_processed": len(tasks_by_file),
        "files_modified": files_modified,
        "files_already_applied": files_already_applied,
        "files_failed": files_failed,
        "elapsed_ms": elapsed_ms,
    }
    
    logger.info(
        "[pot_executor] COMPLETE: success=%s, tasks=%d/%d (already=%d), "
        "files=%d/%d (already=%d), %dms",
        success, tasks_completed, total_tasks, tasks_already_applied,
        files_modified, len(tasks_by_file), files_already_applied, elapsed_ms
    )
    
    status_label = "✓ SUCCESS"
    if not success:
        status_label = "✗ FAILED"
    elif files_already_applied > 0 and files_modified == 0:
        status_label = "✓ ALREADY APPLIED"
    
    print(
        f"[POT_EXECUTOR] {status_label}: "
        f"{tasks_completed}/{total_tasks} tasks, "
        f"{tasks_already_applied} already applied, "
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
        "tasks_already_applied": tasks_already_applied,
        "total_tasks": total_tasks,
        "artifacts_written": artifacts_written,
        "affected_files": affected_files,
        "file_results": file_results,
        "summary": summary,
    }
