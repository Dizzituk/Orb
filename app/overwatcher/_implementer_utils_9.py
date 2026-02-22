from __future__ import annotations
import logging
from app.overwatcher._implementer_utils_3 import MULTI_FILE_MAX_ERRORS, _multi_file_read_content, _multi_file_write_content
from app.overwatcher.sandbox_client import SandboxClient, get_sandbox_client
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


@dataclass
class MultiFileResult:
    """v1.11: Result from multi-file batch operations."""
    success: bool
    operation: str  # "search" or "refactor"
    search_pattern: str = ""
    replacement_pattern: str = ""
    total_files: int = 0
    total_occurrences: int = 0  # v1.11: For search operations
    files_processed: int = 0
    files_modified: int = 0
    files_unchanged: int = 0
    files_failed: int = 0
    total_replacements: int = 0
    file_preview: str = ""
    target_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    details: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    awaiting_confirmation: bool = False
    duration_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "operation": self.operation,
            "search_pattern": self.search_pattern,
            "replacement_pattern": self.replacement_pattern,
            "total_files": self.total_files,
            "total_occurrences": self.total_occurrences,
            "files_processed": self.files_processed,
            "files_modified": self.files_modified,
            "files_unchanged": self.files_unchanged,
            "files_failed": self.files_failed,
            "total_replacements": self.total_replacements,
            "file_preview": self.file_preview,
            "target_files": self.target_files,
            "errors": self.errors,
            "details": self.details,
            "error": self.error,
            "awaiting_confirmation": self.awaiting_confirmation,
            "duration_ms": self.duration_ms,
        }

async def run_multi_file_refactor(
    *,
    multi_file: Dict[str, Any],
    client: Optional[SandboxClient] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> MultiFileResult:
    """
    v1.11: Execute multi-file refactor (search and replace).
    
    Processes each file in target_files:
    1. Read current content
    2. Apply search/replace
    3. Write updated content
    4. Verify write succeeded
    5. Report progress
    
    Args:
        multi_file: Dict with multi_file data from spec
        client: Sandbox client for file operations
        progress_callback: Optional callback for progress updates
        
    Returns:
        MultiFileResult with aggregate results
    """
    import time
    import asyncio
    start_time = time.time()
    
    def elapsed_ms() -> int:
        return int((time.time() - start_time) * 1000)
    
    async def call_progress(data: Dict[str, Any]) -> None:
        """Helper to call progress callback (handles sync/async)."""
        if not progress_callback:
            return
        try:
            if asyncio.iscoroutinefunction(progress_callback):
                await progress_callback(data)
            else:
                progress_callback(data)
        except Exception as e:
            logger.warning("[implementer] v1.11 Progress callback error: %s", e)
    
    if not multi_file.get("is_multi_file"):
        return MultiFileResult(
            success=False,
            operation="refactor",
            error="Not a multi-file operation",
            duration_ms=elapsed_ms(),
        )
    
    # Check confirmation for refactor operations
    if multi_file.get("requires_confirmation") and not multi_file.get("confirmed"):
        return MultiFileResult(
            success=False,
            operation="refactor",
            error="Refactor operation requires confirmation",
            awaiting_confirmation=True,
            duration_ms=elapsed_ms(),
        )
    
    target_files = multi_file.get("target_files", [])
    search_pattern = multi_file.get("search_pattern", "")
    replacement_pattern = multi_file.get("replacement_pattern", "")
    
    if not target_files:
        return MultiFileResult(
            success=False,
            operation="refactor",
            error="No target files specified",
            search_pattern=search_pattern,
            replacement_pattern=replacement_pattern,
            duration_ms=elapsed_ms(),
        )
    
    if not search_pattern:
        return MultiFileResult(
            success=False,
            operation="refactor",
            error="No search pattern specified",
            duration_ms=elapsed_ms(),
        )
    
    logger.info(
        "[implementer] v1.11 Multi-file REFACTOR: '%s' -> '%s', files=%d",
        search_pattern,
        replacement_pattern or "(remove)",
        len(target_files),
    )
    
    # Get sandbox client
    if client is None:
        client = get_sandbox_client()
    
    if not client.is_connected():
        return MultiFileResult(
            success=False,
            operation="refactor",
            error="Sandbox not available",
            search_pattern=search_pattern,
            replacement_pattern=replacement_pattern,
            total_files=len(target_files),
            duration_ms=elapsed_ms(),
        )
    
    # Initialize results tracking
    files_modified = 0
    files_unchanged = 0
    files_failed = 0
    files_processed = 0
    total_replacements = 0
    errors: List[str] = []
    details: List[Dict[str, Any]] = []
    consecutive_errors = 0
    abort_error = None
    
    # Process each file
    for i, file_path in enumerate(target_files, 1):
        file_result: Dict[str, Any] = {
            "path": file_path,
            "status": "pending",
            "replacements": 0,
            "error": None,
        }
        
        try:
            # Progress callback: starting file
            await call_progress({
                "type": "progress",
                "current": i,
                "total": len(target_files),
                "file": file_path,
                "status": "processing",
            })
            
            # Step 1: Read file
            content = await _multi_file_read_content(client, file_path)
            
            if content is None:
                file_result["status"] = "error"
                file_result["error"] = "Could not read file"
                files_failed += 1
                errors.append(f"{file_path}: Could not read file")
                consecutive_errors += 1
                
                if consecutive_errors >= MULTI_FILE_MAX_ERRORS:
                    logger.error(
                        "[implementer] v1.11 Aborting: %d consecutive errors",
                        consecutive_errors
                    )
                    abort_error = f"Aborted after {consecutive_errors} consecutive errors"
                    details.append(file_result)
                    break
                
                details.append(file_result)
                continue
            
            # Step 2: Check if pattern exists in file
            if search_pattern not in content:
                file_result["status"] = "unchanged"
                file_result["replacements"] = 0
                files_unchanged += 1
                files_processed += 1
                consecutive_errors = 0  # Reset on success
                
                await call_progress({
                    "type": "progress",
                    "current": i,
                    "total": len(target_files),
                    "file": file_path,
                    "status": "unchanged",
                    "replacements": 0,
                })
                
                details.append(file_result)
                continue
            
            # Step 3: Count replacements and apply
            replacement_count = content.count(search_pattern)
            new_content = content.replace(search_pattern, replacement_pattern)
            
            # Step 4: Write file
            write_success = await _multi_file_write_content(client, file_path, new_content)
            
            if not write_success:
                file_result["status"] = "error"
                file_result["error"] = "Write failed"
                files_failed += 1
                errors.append(f"{file_path}: Write failed")
                consecutive_errors += 1
                
                if consecutive_errors >= MULTI_FILE_MAX_ERRORS:
                    logger.error(
                        "[implementer] v1.11 Aborting: %d consecutive errors",
                        consecutive_errors
                    )
                    abort_error = f"Aborted after {consecutive_errors} consecutive errors"
                    details.append(file_result)
                    break
                
                details.append(file_result)
                continue
            
            # Step 5: Verify write
            verify_content = await _multi_file_read_content(client, file_path)
            
            if verify_content != new_content:
                file_result["status"] = "verify_failed"
                file_result["error"] = "Verification failed - content mismatch"
                files_failed += 1
                errors.append(f"{file_path}: Verification failed")
                consecutive_errors += 1
                
                if consecutive_errors >= MULTI_FILE_MAX_ERRORS:
                    abort_error = f"Aborted after {consecutive_errors} consecutive errors"
                    details.append(file_result)
                    break
                
                details.append(file_result)
                continue
            
            # Success!
            file_result["status"] = "success"
            file_result["replacements"] = replacement_count
            files_modified += 1
            files_processed += 1
            total_replacements += replacement_count
            consecutive_errors = 0  # Reset on success
            
            logger.info(
                "[implementer] v1.11 Modified %s: %d replacements",
                file_path, replacement_count
            )
            
            # Progress callback: file complete
            await call_progress({
                "type": "progress",
                "current": i,
                "total": len(target_files),
                "file": file_path,
                "status": "success",
                "replacements": replacement_count,
            })
            
            details.append(file_result)
                
        except Exception as e:
            file_result["status"] = "error"
            file_result["error"] = str(e)[:200]
            files_failed += 1
            errors.append(f"{file_path}: {str(e)[:100]}")
            consecutive_errors += 1
            
            logger.error(
                "[implementer] v1.11 Error processing %s: %s",
                file_path, e
            )
            
            if consecutive_errors >= MULTI_FILE_MAX_ERRORS:
                abort_error = f"Aborted after {consecutive_errors} consecutive errors"
                details.append(file_result)
                break
            
            details.append(file_result)
    
    # Final success determination
    success = files_modified > 0 or (files_failed == 0 and files_unchanged > 0)
    if abort_error:
        success = False
    
    # Completion callback
    await call_progress({
        "type": "complete",
        "operation": "refactor",
        "total_files": len(target_files),
        "files_modified": files_modified,
        "files_unchanged": files_unchanged,
        "files_failed": files_failed,
        "total_replacements": total_replacements,
        "success": success,
    })
    
    logger.info(
        "[implementer] v1.11 Multi-file REFACTOR complete: "
        "modified=%d, unchanged=%d, failed=%d, replacements=%d",
        files_modified,
        files_unchanged,
        files_failed,
        total_replacements,
    )
    
    return MultiFileResult(
        success=success,
        operation="refactor",
        search_pattern=search_pattern,
        replacement_pattern=replacement_pattern,
        total_files=len(target_files),
        files_processed=files_processed,
        files_modified=files_modified,
        files_unchanged=files_unchanged,
        files_failed=files_failed,
        total_replacements=total_replacements,
        errors=errors,
        details=details,
        error=abort_error,
        duration_ms=elapsed_ms(),
    )
