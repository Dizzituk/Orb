"""
Part 2 of the file-task processing loop.

Responsibilities:
- Delegate writes to implementer (injected callables)
- Independent verification via sandbox_ops
- Optional job checker invocation
- Update counters and cross-file context accumulators

This module must remain pure orchestration logic. It does NOT import implementer, job_checker,
or interface_extraction directly — those are injected by the orchestrator (seg-10).
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Awaitable

from .sandbox_ops import _verify_file_via_sandbox, _read_existing_file
from ..sandbox_client import SandboxClient

logger = logging.getLogger(__name__)


async def run_process_task_loop_part2(
    client: SandboxClient,
    sandbox_base: str,
    file_path: str,
    attempt_data: Dict[str, Any],
    implementer_write_fn: Callable[[SandboxClient, str, str, bool], Awaitable[None]],
    implementer_edit_fn: Callable[[SandboxClient, str, str, str], Awaitable[None]],
    job_checker_fn: Optional[Callable[[SandboxClient, str], Awaitable[bool]]] = None,
    interface_extract_fn: Optional[Callable[[str, str], Optional[str]]] = None,
) -> Dict[str, Any]:
    """
    Part 2: Write file (via implementer), verify, optionally run job checker, extract interfaces.

    Args:
        client: Active SandboxClient instance
        sandbox_base: Absolute base path in sandbox (e.g. /root/project)
        file_path: Relative file path (e.g. app/foo/bar.py)
        attempt_data: Dict from Part 1 containing:
            - file_content: str (generated content)
            - edit_mode: bool
            - existing_content: Optional[str] (if edit_mode)
            - file_abs: str (absolute path in sandbox)
        implementer_write_fn: Async callable to write full file
        implementer_edit_fn: Async callable to apply edit
        job_checker_fn: Optional async callable to run job checker
        interface_extract_fn: Optional callable to extract cross-file interface context

    Returns:
        Dict with:
            - success: bool
            - file_content_final: str (actual content after write, read back)
            - verify: Dict[str, Any]
            - job_check_passed: Optional[bool]
            - last_error: Optional[str]
            - interface_context: Optional[str] (if extraction succeeded)
    """
    file_content = attempt_data["file_content"]
    edit_mode = attempt_data.get("edit_mode", False)
    existing_content = attempt_data.get("existing_content")
    file_abs = attempt_data["file_abs"]

    result: Dict[str, Any] = {
        "success": False,
        "file_content_final": "",
        "verify": {},
        "job_check_passed": None,
        "last_error": None,
        "interface_context": None,
    }

    try:
        # Step 1: Write file via implementer
        if edit_mode and existing_content is not None:
            logger.info(f"[Part2] Applying edit via implementer: {file_path}")
            await implementer_edit_fn(client, file_path, existing_content, file_content)
        else:
            logger.info(f"[Part2] Writing full file via implementer: {file_path}")
            await implementer_write_fn(client, file_path, file_content, edit_mode)

        # Step 2: Read back actual content (especially important for edit mode)
        logger.debug(f"[Part2] Reading back written file: {file_abs}")
        actual_content = await _read_existing_file(client, file_abs)
        if actual_content is None:
            err = f"File write succeeded but readback failed: {file_abs}"
            logger.error(err)
            result["last_error"] = err
            return result

        result["file_content_final"] = actual_content

        # Step 3: Independent verification via sandbox
        logger.info(f"[Part2] Running independent verification for: {file_path}")
        verify_result = await _verify_file_via_sandbox(
            client=client,
            file_path=file_path,
            sandbox_base=sandbox_base,
        )
        result["verify"] = verify_result

        if not verify_result.get("success", False):
            err = f"Verification failed: {verify_result.get('error', 'unknown')}"
            logger.warning(f"[Part2] {err}")
            result["last_error"] = err
            return result

        logger.info(f"[Part2] Verification passed for: {file_path}")

        # Step 4: Optional job checker invocation
        if job_checker_fn is not None:
            logger.info(f"[Part2] Running job checker for: {file_path}")
            try:
                check_passed = await job_checker_fn(client, file_path)
                result["job_check_passed"] = check_passed
                if not check_passed:
                    logger.warning(f"[Part2] Job checker failed for: {file_path}")
                    result["last_error"] = "Job checker test failed"
                    return result
                logger.info(f"[Part2] Job checker passed for: {file_path}")
            except Exception as e:
                logger.error(f"[Part2] Job checker error for {file_path}: {e}", exc_info=True)
                result["job_check_passed"] = False
                result["last_error"] = f"Job checker exception: {str(e)}"
                return result

        # Step 5: Extract cross-file interface context (if extractor provided)
        if interface_extract_fn is not None:
            logger.debug(f"[Part2] Extracting interface context for: {file_path}")
            try:
                ctx = interface_extract_fn(file_path, actual_content)
                if ctx:
                    result["interface_context"] = ctx
                    logger.debug(f"[Part2] Interface context extracted: {len(ctx)} chars")
            except Exception as e:
                logger.warning(f"[Part2] Interface extraction failed for {file_path}: {e}")
                # Non-fatal: continue with success

        # Success
        result["success"] = True
        logger.info(f"[Part2] All steps passed for: {file_path}")
        return result

    except Exception as e:
        logger.error(f"[Part2] Unexpected error processing {file_path}: {e}", exc_info=True)
        result["last_error"] = f"Unexpected error: {str(e)}"
        return result