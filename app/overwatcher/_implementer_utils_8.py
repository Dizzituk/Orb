from __future__ import annotations
import logging
from .spec_resolution import ResolvedSpec, SpecMissingDeliverableError
from app.overwatcher._implementer_utils_2 import IMPLEMENTER_BUILD_ID, _generate_sandbox_path_candidates
from app.overwatcher._implementer_utils_3 import _apply_qa_corrections, _insert_answers_under_questions, _is_absolute_windows_path, _is_specgate_correction_format
from app.overwatcher._implementer_utils_4 import _parse_answers_from_reply
from app.overwatcher._implementer_utils_6 import ImplementerResult, _write_content_to_sandbox
from app.overwatcher.overwatcher import Decision, OverwatcherOutput
from app.overwatcher.sandbox_client import SandboxClient, SandboxError, get_sandbox_client
from pathlib import Path
from typing import Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


async def run_implementer(
    *,
    spec: ResolvedSpec,
    output: OverwatcherOutput,
    client: Optional[SandboxClient] = None,
    segment_context: Optional[dict] = None,
) -> ImplementerResult:
    """Execute approved work via Sandbox."""
    import time
    start_time = time.time()
    
    def elapsed() -> int:
        return int((time.time() - start_time) * 1000)
    
    if output.decision != Decision.PASS:
        return ImplementerResult(
            success=False,
            error=f"Overwatcher decision was {output.decision.value}",
            duration_ms=elapsed(),
        )
    
    try:
        filename, content, action = spec.get_target_file()
        target = spec.get_target()
        must_exist = spec.get_must_exist()
        output_mode = spec.get_output_mode()
        insertion_format = spec.get_insertion_format()
    except SpecMissingDeliverableError as e:
        return ImplementerResult(
            success=False,
            error=str(e),
            duration_ms=elapsed(),
        )
    
    # v1.10: Logging
    logger.info(f"[implementer] BUILD_ID={IMPLEMENTER_BUILD_ID}")
    logger.info(f"[implementer] RAW output_mode={repr(output_mode)}")
    print(f"\n>>> [IMPLEMENTER v1.10] BUILD={IMPLEMENTER_BUILD_ID} <<<")
    print(f">>> [IMPLEMENTER v1.10] RAW output_mode={repr(output_mode)} <<<\n")
    
    mode_lower = (output_mode or "").strip().lower()
    
    # CHAT_ONLY check
    if mode_lower == "chat_only":
        logger.info("[implementer] CHAT_ONLY DETECTED - RETURNING EARLY")
        return ImplementerResult(
            success=True,
            output_path=None,
            sha256=None,
            duration_ms=elapsed(),
            sandbox_used=False,
            filename=filename,
            content_written=None,
            action_taken="chat_only_noop",
            write_method="none",
        )
    
    logger.info(f"[implementer] === SPEC-DRIVEN TASK === MODE: {mode_lower}")
    logger.info(f"[implementer] Action: {action}, Filename: {filename}")
    
    if client is None:
        client = get_sandbox_client()
    
    try:
        if not client.is_connected():
            return ImplementerResult(
                success=False,
                error="SAFETY: Sandbox not available",
                duration_ms=elapsed(),
                sandbox_used=False,
            )
        
        # Build expected path
        # v1.15: NEVER hardcode WDAGUtilityAccount paths — the sandbox is a
        # clone of the host session, not a separate WDAG user profile.
        # For DESKTOP targets, query the sandbox for the actual user profile.
        if _is_absolute_windows_path(filename):
            expected_path = filename
            base_filename = Path(filename).name
            is_absolute = True
        else:
            base_filename = filename
            is_absolute = False
            if target == "DESKTOP":
                # Dynamically resolve the actual Desktop path inside the sandbox
                desktop_cmd = 'Write-Output "$env:USERPROFILE\\Desktop"'
                desktop_result = client.shell_run(desktop_cmd, timeout_seconds=10)
                if desktop_result.stdout and desktop_result.stdout.strip():
                    sandbox_desktop = desktop_result.stdout.strip()
                else:
                    # Fallback: use D:\Orb as safe default (project root)
                    sandbox_desktop = "D:\\Orb"
                    logger.warning(
                        "[implementer] v1.15 Could not resolve sandbox Desktop, "
                        "falling back to %s", sandbox_desktop,
                    )
                expected_path = f"{sandbox_desktop}\\{base_filename}"
                logger.info(
                    "[implementer] v1.15 DESKTOP target resolved to: %s",
                    expected_path,
                )
            else:
                expected_path = f"{target}\\{base_filename}"
        
        # For "modify" action with must_exist: verify file exists first
        if action == "modify" and must_exist:
            candidates = _generate_sandbox_path_candidates(expected_path)
            
            resolved_path = None
            for candidate in candidates:
                exists_cmd = f'Test-Path -Path "{candidate}"'
                exists_result = client.shell_run(exists_cmd, timeout_seconds=10)
                
                if "True" in exists_result.stdout:
                    resolved_path = candidate
                    break
            
            if resolved_path is None:
                return ImplementerResult(
                    success=False,
                    error=f"SPEC VIOLATION: File '{filename}' does not exist",
                    duration_ms=elapsed(),
                    sandbox_used=True,
                    filename=filename,
                    action_taken="existence_check_failed",
                )
            
            expected_path = resolved_path
        
        # WRITE FILE VIA SANDBOX
        if is_absolute:
            logger.info(f"[implementer] Writing via PowerShell to: {expected_path}")
            
            # REWRITE_IN_PLACE mode
            if mode_lower == "rewrite_in_place":
                logger.info("[implementer] REWRITE_IN_PLACE mode: multi-question file edit")
                
                # Read file
                read_cmd = f'Get-Content -Path "{expected_path}" -Raw'
                read_result = client.shell_run(read_cmd, timeout_seconds=30)
                
                if read_result.stderr and read_result.stderr.strip():
                    return ImplementerResult(
                        success=False,
                        error=f"Failed to read file: {read_result.stderr}",
                        duration_ms=elapsed(),
                        sandbox_used=True,
                        write_method="rewrite",
                    )
                
                original_text = read_result.stdout or ""
                logger.info(f"[implementer] Read {len(original_text)} chars from file")
                
                # =============================================================
                # v1.10: Try intelligent Q&A correction FIRST
                # =============================================================
                if _is_specgate_correction_format(content):
                    logger.info("[implementer] v1.10 Detected SpecGate correction format (Q#: [STATUS])")
                    
                    updated_text, corrections_count = _apply_qa_corrections(original_text, content)
                    
                    if corrections_count > 0:
                        logger.info("[implementer] v1.10 Applied %d intelligent corrections", corrections_count)
                        
                        # Write corrected content (v1.13: auto temp-file for large files)
                        write_result = _write_content_to_sandbox(client, expected_path, updated_text, timeout_seconds=60)
                        
                        write_success = not write_result.stderr or write_result.stderr.strip() == ""
                        if write_success:
                            logger.info("[implementer] v1.10 SUCCESS: Intelligent Q&A correction completed")
                            return ImplementerResult(
                                success=True,
                                output_path=expected_path,
                                sha256=None,
                                duration_ms=elapsed(),
                                sandbox_used=True,
                                filename=filename,
                                content_written=updated_text,
                                action_taken=action,
                                write_method="rewrite_intelligent",
                            )
                        else:
                            return ImplementerResult(
                                success=False,
                                error=f"PowerShell write failed: {write_result.stderr or write_result.stdout}",
                                duration_ms=elapsed(),
                                sandbox_used=True,
                                write_method="rewrite_intelligent",
                            )
                    else:
                        logger.warning("[implementer] v1.10 No corrections applied - falling back to legacy method")
                
                # =============================================================
                # FALLBACK: Legacy answer insertion method (v1.9)
                # =============================================================
                logger.info("[implementer] Using legacy answer insertion method")
                
                # Parse and insert answers
                answers = _parse_answers_from_reply(content)
                logger.info(f"[implementer] Parsed {len(answers)} answers: {list(answers.keys())}")
                
                fmt = insertion_format or "\n\nAnswer:\n{reply}\n"
                updated_text = _insert_answers_under_questions(original_text, answers, fmt)
                
                # v1.13: Write using shared helper (auto temp-file for large files)
                logger.info(f"[implementer] v1.13 Writing {len(updated_text)} chars")
                write_result = _write_content_to_sandbox(client, expected_path, updated_text, timeout_seconds=60)
                
                write_success = not write_result.stderr or write_result.stderr.strip() == ""
                if write_success:
                    logger.info(f"[implementer] SUCCESS: REWRITE_IN_PLACE completed")
                    return ImplementerResult(
                        success=True,
                        output_path=expected_path,
                        sha256=None,
                        duration_ms=elapsed(),
                        sandbox_used=True,
                        filename=filename,
                        content_written=updated_text,
                        action_taken=action,
                        write_method="rewrite",
                    )
                else:
                    return ImplementerResult(
                        success=False,
                        error=f"PowerShell write failed: {write_result.stderr or write_result.stdout}",
                        duration_ms=elapsed(),
                        sandbox_used=True,
                        write_method="rewrite",
                    )
            
            # APPEND_IN_PLACE mode
            elif mode_lower == "append_in_place":
                if insertion_format:
                    try:
                        append_text = insertion_format.format(reply=content)
                    except KeyError:
                        append_text = f"\n\nAnswer:\n{content}\n"
                else:
                    append_text = f"\n\nAnswer:\n{content}\n"
                
                # v1.8: Use Base64 for append too
                # Read existing content, append, write back
                read_cmd = f'Get-Content -Path "{expected_path}" -Raw'
                read_result = client.shell_run(read_cmd, timeout_seconds=30)
                existing_content = read_result.stdout or ""
                
                combined_content = existing_content + append_text
                # v1.13: auto temp-file for large files
                write_result = _write_content_to_sandbox(client, expected_path, combined_content, timeout_seconds=60)
                
                write_success = not write_result.stderr or write_result.stderr.strip() == ""
                if write_success:
                    return ImplementerResult(
                        success=True,
                        output_path=expected_path,
                        sha256=None,
                        duration_ms=elapsed(),
                        sandbox_used=True,
                        filename=filename,
                        content_written=append_text,
                        action_taken=action,
                        write_method="append",
                    )
                else:
                    return ImplementerResult(
                        success=False,
                        error=f"PowerShell write failed: {write_result.stderr}",
                        duration_ms=elapsed(),
                        sandbox_used=True,
                        write_method="append",
                    )
            
            # SEPARATE_REPLY_FILE or OVERWRITE_FULL mode
            elif mode_lower in ("separate_reply_file", "overwrite_full"):
                # v1.13: auto temp-file for large files
                write_method = "overwrite_full" if mode_lower == "overwrite_full" else "overwrite"
                write_result = _write_content_to_sandbox(client, expected_path, content, timeout_seconds=60)
                
                write_success = not write_result.stderr or write_result.stderr.strip() == ""
                if write_success:
                    return ImplementerResult(
                        success=True,
                        output_path=expected_path,
                        sha256=None,
                        duration_ms=elapsed(),
                        sandbox_used=True,
                        filename=filename,
                        content_written=content,
                        action_taken=action,
                        write_method=write_method,
                    )
                else:
                    return ImplementerResult(
                        success=False,
                        error=f"PowerShell write failed: {write_result.stderr}",
                        duration_ms=elapsed(),
                        sandbox_used=True,
                        write_method=write_method,
                    )
            
            # UNKNOWN MODE - FAIL SAFE
            else:
                logger.error(f"[implementer] SAFETY STOP: Unknown output_mode='{output_mode}'")
                return ImplementerResult(
                    success=False,
                    error=f"SAFETY: Unknown output_mode '{output_mode}'",
                    duration_ms=elapsed(),
                    sandbox_used=False,
                    filename=filename,
                    write_method=None,
                )
        else:
            # Use sandbox API for non-absolute paths
            result = client.write_file(
                target=target,
                filename=base_filename,
                content=content,
                overwrite=True,
            )
            
            if result.ok:
                return ImplementerResult(
                    success=True,
                    output_path=result.path,
                    sha256=result.sha256,
                    duration_ms=elapsed(),
                    sandbox_used=True,
                    filename=filename,
                    content_written=content,
                    action_taken=action,
                    write_method="overwrite",
                )
            else:
                return ImplementerResult(
                    success=False,
                    error=f"Sandbox write failed: {getattr(result, 'error', 'unknown')}",
                    duration_ms=elapsed(),
                    sandbox_used=True,
                )
            
    except SandboxError as e:
        return ImplementerResult(
            success=False,
            error=f"Sandbox error: {e}",
            duration_ms=elapsed(),
            sandbox_used=True,
        )
    except Exception as e:
        logger.exception(f"[implementer] Failed: {e}")
        return ImplementerResult(
            success=False,
            error=str(e),
            duration_ms=elapsed(),
        )
