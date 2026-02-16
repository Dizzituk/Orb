import logging
import os
import re
from typing import Any, Dict, List, Optional

from ..sandbox_client import SandboxClient
from .sandbox_ops import _read_existing_file

logger = logging.getLogger(__name__)

BOOT_MAX_STRIKES = 3


def _parse_broken_file_from_traceback(tb: str, artifacts: List[str]) -> Optional[str]:
    """
    Parse traceback to find the file causing the import error.
    Returns the path if it's among our artifacts, else None.
    """
    lines = tb.split("\n")
    for line in lines:
        m = re.search(r'File\s+"([^"]+)"', line)
        if m:
            path_in_tb = m.group(1).replace("\\", "/")
            for art in artifacts:
                art_norm = art.replace("\\", "/")
                if path_in_tb.endswith(art_norm) or art_norm in path_in_tb:
                    return art
    return None


async def _run_boot_check(client: SandboxClient, sb: str) -> tuple[bool, Optional[str]]:
    """
    Run boot check: python -c "from main import app; print('BOOT_CHECK_PASS')"
    Returns (passed: bool, error_output: Optional[str])
    """
    python_exe = sb + "\\.venv\\Scripts\\python.exe"
    cmd = f'"{python_exe}" -c "from main import app; print(\'BOOT_CHECK_PASS\')"'
    
    logger.info(f"Running boot check: {cmd}")
    result = await client.run_command(cmd, cwd=sb)
    
    if result["exit_code"] == 0 and "BOOT_CHECK_PASS" in result["stdout"]:
        logger.info("Boot check PASSED")
        return (True, None)
    else:
        error_output = result.get("stderr", "") or result.get("stdout", "")
        logger.warning(f"Boot check FAILED. Exit code: {result['exit_code']}")
        logger.warning(f"Error output: {error_output[:500]}")
        return (False, error_output)


async def run_process_boot_strike_loop(
    *,
    client: SandboxClient,
    sandbox_base: str,
    skip_boot_check: bool,
    success: bool,
    total_succeeded: int,
    artifacts_written: List[str],
    architecture_content: str,
    impl_provider: str,
    impl_model: str,
    llm_call_fn: Any,
    run_implementer_task_fn: Any,
) -> Dict[str, Any]:
    """
    Run boot check retry loop after file operations.
    
    Args:
        client: SandboxClient instance
        sandbox_base: Sandbox base path
        skip_boot_check: If True, skip boot check entirely
        success: Whether file operations succeeded
        total_succeeded: Count of successful file writes
        artifacts_written: List of artifact paths written
        architecture_content: Original architecture content for context
        impl_provider: LLM provider for implementer
        impl_model: LLM model for implementer
        llm_call_fn: Function to call LLM for fix generation
        run_implementer_task_fn: Function to run implementer task
        
    Returns:
        Dict with keys:
        - boot_checked: bool (whether boot check was attempted)
        - boot_passed: Optional[bool] (True if passed, False if failed, None if not checked)
        - attempts: int (number of boot check attempts)
        - last_error: Optional[str] (last error message if failed)
    """
    if skip_boot_check or not success or total_succeeded == 0:
        logger.info("Skipping boot check (skip_boot_check=%s, success=%s, total_succeeded=%d)",
                   skip_boot_check, success, total_succeeded)
        return {
            "boot_checked": False,
            "boot_passed": None,
            "attempts": 0,
            "last_error": None
        }
    
    logger.info("Starting boot check loop (max %d attempts)", BOOT_MAX_STRIKES)
    
    attempts = 0
    last_error = None
    same_error_count = 0
    prev_error_sig = None
    
    while attempts < BOOT_MAX_STRIKES:
        attempts += 1
        logger.info(f"Boot check attempt {attempts}/{BOOT_MAX_STRIKES}")
        
        passed, error_output = await _run_boot_check(client, sandbox_base)
        
        if passed:
            logger.info("Boot check passed on attempt %d", attempts)
            return {
                "boot_checked": True,
                "boot_passed": True,
                "attempts": attempts,
                "last_error": None
            }
        
        last_error = error_output or "Unknown boot error"
        logger.warning(f"Boot check failed on attempt {attempts}")
        
        error_sig = last_error[:200] if last_error else ""
        if error_sig == prev_error_sig:
            same_error_count += 1
            logger.warning(f"Same error repeated (count: {same_error_count})")
            if same_error_count >= 3:
                logger.error("Same error repeated 3 times, giving up")
                break
        else:
            same_error_count = 1
            prev_error_sig = error_sig
        
        broken_file = _parse_broken_file_from_traceback(last_error, artifacts_written)
        
        if not broken_file:
            logger.warning("Could not identify broken file from traceback, cannot auto-fix")
            break
        
        logger.info(f"Identified broken file: {broken_file}")
        
        existing_content = await _read_existing_file(client, broken_file)
        if not existing_content:
            logger.warning(f"Could not read broken file {broken_file}, cannot fix")
            break
        
        logger.info(f"Attempting to generate fix for {broken_file}")
        
        fix_prompt = f"""The following file caused a boot error:

File: {broken_file}

Current content:
```
{existing_content}
```

Error:
```
{last_error}
```

Architecture context:
```
{architecture_content[:2000]}
```

Generate a corrected version of this file that fixes the error.
Output ONLY the corrected file content, no markdown fences, no explanations.
"""
        
        try:
            fix_response = await llm_call_fn(
                prompt=fix_prompt,
                provider=impl_provider,
                model=impl_model
            )
            
            if not fix_response or "content" not in fix_response:
                logger.warning("LLM did not return fix content")
                break
            
            fixed_content = fix_response["content"]
            
            logger.info(f"Writing fixed content to {broken_file}")
            write_result = await run_implementer_task_fn(
                file_path=broken_file,
                content=fixed_content,
                retry_count=0
            )
            
            if not write_result.get("success"):
                logger.warning(f"Failed to write fixed file: {write_result.get('error')}")
                break
            
            logger.info("Fixed file written, retrying boot check")
            
        except Exception as e:
            logger.error(f"Error during auto-fix: {e}", exc_info=True)
            break
    
    logger.error(f"Boot check failed after {attempts} attempts")
    return {
        "boot_checked": True,
        "boot_passed": False,
        "attempts": attempts,
        "last_error": last_error
    }