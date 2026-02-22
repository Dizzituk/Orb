from __future__ import annotations
import logging
from .spec_resolution import ResolvedSpec, SpecMissingDeliverableError
from app.overwatcher._implementer_utils_3 import _is_absolute_windows_path, _is_specgate_correction_format
from app.overwatcher._implementer_utils_4 import EditTaskResult, VerificationResult, _parse_answers_from_reply, _parse_corrections
from app.overwatcher.sandbox_client import SandboxClient, SandboxError, get_sandbox_client
from pathlib import Path
from typing import Dict, List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


async def run_verification(
    *,
    impl_result: ImplementerResult,
    spec: ResolvedSpec,
    client: Optional[SandboxClient] = None,
) -> VerificationResult:
    """Verify Implementer output against spec requirements."""
    from .implementer import ImplementerResult
    try:
        expected_filename, expected_content, expected_action = spec.get_target_file()
        output_mode = spec.get_output_mode()
        insertion_format = spec.get_insertion_format()
    except SpecMissingDeliverableError as e:
        return VerificationResult(passed=False, error=str(e))
    
    mode_lower = (output_mode or "").lower()
    
    # CHAT_ONLY verification
    if mode_lower == "chat_only":
        if impl_result.write_method == "none" and impl_result.success:
            return VerificationResult(
                passed=True,
                file_exists=False,
                content_matches=True,
                filename_matches=True,
                expected_content=expected_content,
                expected_filename=expected_filename,
            )
        else:
            return VerificationResult(
                passed=False,
                error=f"CHAT_ONLY verification failed: write_method={impl_result.write_method}",
            )
    
    if not impl_result.success:
        return VerificationResult(
            passed=False,
            expected_filename=expected_filename,
            expected_content=expected_content,
            error=f"Implementation failed: {impl_result.error}",
        )
    
    if not impl_result.output_path:
        return VerificationResult(
            passed=False,
            expected_filename=expected_filename,
            expected_content=expected_content,
            error="No output path from Implementer",
        )
    
    actual_filename = Path(impl_result.output_path).name
    
    if _is_absolute_windows_path(expected_filename):
        expected_basename = Path(expected_filename).name
        filename_matches = actual_filename == expected_basename
    else:
        filename_matches = actual_filename == expected_filename
    
    if not filename_matches:
        return VerificationResult(
            passed=False,
            file_exists=True,
            content_matches=False,
            filename_matches=False,
            expected_filename=expected_filename,
            actual_filename=actual_filename,
            error=f"WRONG FILE: Expected '{expected_filename}' but got '{actual_filename}'",
        )
    
    if client is None:
        client = get_sandbox_client()
    
    try:
        if not client.is_connected():
            return VerificationResult(
                passed=False,
                expected_filename=expected_filename,
                expected_content=expected_content,
                error="Sandbox not available for verification",
            )
        
        ps_path = impl_result.output_path.replace("/", "\\")
        
        exists_result = client.shell_run(f'Test-Path -Path "{ps_path}"', timeout_seconds=10)
        file_exists = "True" in exists_result.stdout
        
        if not file_exists:
            return VerificationResult(
                passed=False,
                file_exists=False,
                filename_matches=filename_matches,
                expected_filename=expected_filename,
                expected_content=expected_content,
                error=f"File not found at {impl_result.output_path}",
            )
        
        read_result = client.shell_run(f'Get-Content -Path "{ps_path}" -Raw', timeout_seconds=10)
        
        if read_result.stderr and read_result.stderr.strip():
            return VerificationResult(
                passed=False,
                file_exists=True,
                filename_matches=filename_matches,
                expected_filename=expected_filename,
                expected_content=expected_content,
                error=f"Failed to read file: {read_result.stderr}",
            )
        
        actual_content = read_result.stdout.strip() if read_result.stdout else ""
        
        # Content verification depends on mode
        if mode_lower == "rewrite_in_place":
            # v1.10: Enhanced verification for intelligent corrections
            if _is_specgate_correction_format(expected_content):
                # Parse corrections and verify each was applied
                corrections = _parse_corrections(expected_content)
                all_present = True
                missing = []
                
                for q_num, answer in corrections.items():
                    # Check if the correction appears in the file
                    # Be flexible - check if the answer text is present
                    if answer.strip() not in actual_content:
                        all_present = False
                        missing.append(q_num)
                
                content_matches = all_present
                if not all_present:
                    logger.warning("[implementer] v1.10 Verification: missing corrections for Q%s", missing)
            else:
                # Legacy verification
                answers = _parse_answers_from_reply(expected_content)
                all_present = True
                missing = []
                
                for q_num, answer in answers.items():
                    if answer.strip() not in actual_content:
                        all_present = False
                        missing.append(q_num)
                
                content_matches = all_present
        
        elif mode_lower == "append_in_place":
            content_matches = expected_content.strip() in actual_content
        
        elif mode_lower == "overwrite_full":
            content_matches = actual_content.strip() == expected_content.strip()
        
        else:
            content_matches = actual_content == expected_content
        
        passed = content_matches and filename_matches
        
        return VerificationResult(
            passed=passed,
            file_exists=True,
            content_matches=content_matches,
            filename_matches=filename_matches,
            actual_content=actual_content,
            expected_content=expected_content,
            expected_filename=expected_filename,
            actual_filename=actual_filename,
            error=None if passed else "Content verification failed",
        )
        
    except SandboxError as e:
        return VerificationResult(
            passed=False,
            expected_filename=expected_filename,
            expected_content=expected_content,
            error=f"Sandbox error: {e}",
        )
    except Exception as e:
        return VerificationResult(
            passed=False,
            expected_filename=expected_filename,
            expected_content=expected_content,
            error=str(e),
        )

async def run_implementer_edit_task(
    *,
    path: str,
    edits: List[Dict[str, str]],
    client: Optional[SandboxClient] = None,
) -> EditTaskResult:
    """v1.13: Apply targeted edits to an existing file in the sandbox.
    
    Instead of having the LLM regenerate the entire file, this function:
    1. Reads the existing file from sandbox
    2. Applies each {old_text, new_text} replacement in order
    3. Writes the modified file back
    4. Verifies the write
    
    Each edit dict must have:
        - "old_text": exact text to find (must appear exactly once)
        - "new_text": replacement text
    
    If old_text appears 0 or 2+ times, that edit is skipped and recorded
    in failed_edits. All other edits still apply.
    
    IMPLEMENTER IS THE ONLY WRITER.
    
    Args:
        path: Absolute sandbox path to the file to edit
        edits: List of {"old_text": str, "new_text": str} dicts
        client: Optional sandbox client (uses default if None)
    
    Returns:
        EditTaskResult with per-edit success/failure tracking
    """
    from .implementer import _write_content_to_sandbox
    import time
    start_time = time.time()
    
    def elapsed() -> int:
        return int((time.time() - start_time) * 1000)
    
    logger.info(
        "[implementer] v1.13 Edit task: path=%s, edits=%d",
        path, len(edits),
    )
    print(f"[IMPLEMENTER_EDIT] MODIFY: {path} ({len(edits)} edits)")
    
    if not edits:
        return EditTaskResult(
            success=False,
            path=path,
            error="No edits provided",
            duration_ms=elapsed(),
        )
    
    if client is None:
        client = get_sandbox_client()
    
    if not client.is_connected():
        return EditTaskResult(
            success=False,
            path=path,
            error="SAFETY: Sandbox not available",
            duration_ms=elapsed(),
        )
    
    # Step 1: Read existing file
    try:
        read_cmd = f'Get-Content -Path "{path}" -Raw -Encoding UTF8'
        read_result = client.shell_run(read_cmd, timeout_seconds=30)
        
        if read_result.stdout is None or (read_result.stderr and read_result.stderr.strip()):
            return EditTaskResult(
                success=False,
                path=path,
                error=f"Cannot read file: {read_result.stderr or 'no output'}",
                duration_ms=elapsed(),
            )
        
        content = read_result.stdout
        chars_before = len(content)
        logger.info("[implementer] v1.13 Read %d chars from %s", chars_before, path)
        
    except Exception as e:
        return EditTaskResult(
            success=False,
            path=path,
            error=f"Read exception: {e}",
            duration_ms=elapsed(),
        )
    
    # Step 2: Apply edits sequentially
    edits_applied = 0
    edits_failed = 0
    failed_edits: List[Dict[str, str]] = []
    
    for i, edit in enumerate(edits, 1):
        old_text = edit.get("old_text", "")
        new_text = edit.get("new_text", "")
        
        if not old_text:
            logger.warning("[implementer] v1.13 Edit %d: empty old_text, skipping", i)
            edits_failed += 1
            failed_edits.append({"old_text": "(empty)", "reason": "empty old_text"})
            continue
        
        # Count occurrences
        count = content.count(old_text)
        
        if count == 0:
            logger.warning(
                "[implementer] v1.13 Edit %d: old_text not found (len=%d, preview='%s')",
                i, len(old_text), old_text[:80],
            )
            edits_failed += 1
            failed_edits.append({
                "old_text": old_text[:100],
                "reason": "not found in file",
            })
            continue
        
        if count > 1:
            logger.warning(
                "[implementer] v1.13 Edit %d: old_text found %d times (ambiguous), skipping",
                i, count,
            )
            edits_failed += 1
            failed_edits.append({
                "old_text": old_text[:100],
                "reason": f"found {count} times (must be unique)",
            })
            continue
        
        # Exactly 1 occurrence — apply
        content = content.replace(old_text, new_text, 1)
        edits_applied += 1
        logger.info(
            "[implementer] v1.13 Edit %d applied: -%d chars, +%d chars",
            i, len(old_text), len(new_text),
        )
    
    chars_after = len(content)
    
    if edits_applied == 0:
        return EditTaskResult(
            success=False,
            path=path,
            edits_applied=0,
            edits_failed=edits_failed,
            chars_before=chars_before,
            chars_after=chars_before,
            error="No edits could be applied",
            failed_edits=failed_edits,
            duration_ms=elapsed(),
        )
    
    # Step 3: Write modified content back
    try:
        write_result = _write_content_to_sandbox(client, path, content, timeout_seconds=60)
        
        if write_result.stderr and write_result.stderr.strip():
            return EditTaskResult(
                success=False,
                path=path,
                edits_applied=edits_applied,
                edits_failed=edits_failed,
                chars_before=chars_before,
                chars_after=chars_after,
                error=f"Write failed: {write_result.stderr[:200]}",
                failed_edits=failed_edits,
                duration_ms=elapsed(),
            )
        
        logger.info("[implementer] v1.13 Wrote %d chars to %s", chars_after, path)
        
    except Exception as e:
        return EditTaskResult(
            success=False,
            path=path,
            edits_applied=edits_applied,
            edits_failed=edits_failed,
            chars_before=chars_before,
            chars_after=chars_after,
            error=f"Write exception: {e}",
            failed_edits=failed_edits,
            duration_ms=elapsed(),
        )
    
    # Step 4: Verify
    verified = False
    try:
        verify_result = client.shell_run(
            f'Get-Content -Path "{path}" -Raw -Encoding UTF8',
            timeout_seconds=30,
        )
        if verify_result.stdout is not None:
            if verify_result.stdout.strip() == content.strip():
                verified = True
                logger.info("[implementer] v1.13 Edit verified: %s", path)
            else:
                logger.warning(
                    "[implementer] v1.13 Edit verify mismatch: wrote %d, read %d",
                    len(content), len(verify_result.stdout),
                )
    except Exception as e:
        logger.warning("[implementer] v1.13 Edit verify exception: %s", e)
    
    print(
        f"[IMPLEMENTER_EDIT] {'✓' if verified else '⚠'} "
        f"MODIFY {path}: {edits_applied}/{len(edits)} edits applied, "
        f"{chars_before} → {chars_after} chars, verified={verified}"
    )
    
    return EditTaskResult(
        success=True,
        path=path,
        edits_applied=edits_applied,
        edits_failed=edits_failed,
        chars_before=chars_before,
        chars_after=chars_after,
        failed_edits=failed_edits,
        verified=verified,
        duration_ms=elapsed(),
    )
