"""
Step 6: Backend boot check with retry loop.

After all file operations, verifies the backend can still start.
If boot fails, identifies the broken file from the traceback,
feeds the error back to the Implementer for a targeted fix, and retries.

Three-strike limit per unique error.  New errors reset the counter.

Extracted from orchestrator.py monolith (v2.9–v3.2 logic).
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

from ..sandbox_client import SandboxClient
from .constants import IMPLEMENTER_MAX_TOKENS
from .context import _read_existing_file
from .execution_state import ExecutionContext
from .helpers import _extract_llm_content, _strip_markdown_fences
from .parsing import extract_section_for_file
from .prompts import IMPLEMENTER_MODIFY_FILE_SYSTEM

logger = logging.getLogger(__name__)

BOOT_MAX_STRIKES = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_boot_check(client: SandboxClient, sandbox_base: str) -> tuple:
    """Run boot check.

    Returns (passed: bool, error: str, full_output: str).

    v3.1: Fixed error reporting — prefer traceback/import errors from
    stdout, not non-fatal stderr warnings.
    """
    venv_python = sandbox_base + "\\.venv\\Scripts\\python.exe"
    boot_cmd = (
        f'cd "{sandbox_base}" ; '
        f'& "{venv_python}" -c '
        f'"import sys; sys.path.insert(0, r\'{sandbox_base}\'); '
        f'from main import app; print(\'BOOT_CHECK_PASS\')"'
    )
    result = client.shell_run(boot_cmd, timeout_seconds=30)
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    passed = "BOOT_CHECK_PASS" in stdout

    if passed:
        return passed, "", stderr

    error_keywords = (
        'Error', 'Traceback', 'ImportError', 'ModuleNotFoundError',
        'SyntaxError', 'AttributeError', 'NameError', 'TypeError',
        'File "', 'cannot import', 'No module named',
    )
    error_parts = []
    for line in (stdout + "\n" + stderr).split('\n'):
        line_s = line.strip()
        if any(kw in line_s for kw in error_keywords):
            error_parts.append(line_s)

    error_msg = (
        '\n'.join(error_parts[:10])
        if error_parts
        else f"stdout(tail): {stdout[-500:]}\nstderr(tail): {stderr[-500:]}"
    )
    full_output = stdout + "\n---STDERR---\n" + stderr
    return passed, error_msg[:1000], full_output


def _parse_broken_file(traceback: str, written: List[str]) -> Optional[str]:
    """Extract the broken file path from a Python traceback.

    Only returns paths that were written by this job (artifacts_written).
    """
    file_matches = re.findall(r'File "([^"]+)"', traceback)
    written_set = {p.replace("/", "\\") for p in written}
    for fpath in reversed(file_matches):
        normalised = fpath.replace("/", "\\")
        if normalised in written_set:
            return normalised
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_boot_check_loop(
    ctx: ExecutionContext,
    client: SandboxClient,
) -> None:
    """Run backend boot check with fix-retry loop.

    Updates ctx.files_failed and ctx.add_trace on failure.
    """
    if ctx.skip_boot_check:
        logger.info("[arch_exec] v3.2 Boot check SKIPPED (skip_boot_check=True, intermediate segment)")
        print("[ARCH_EXEC] ⏭️ Boot check skipped (intermediate segment)")
        ctx.add_trace("BOOT_CHECK_COMPLETE", "skipped_intermediate")
        return

    if not (ctx.success or ctx.total_succeeded > 0):
        return

    logger.info("[arch_exec] v2.9 Running backend boot check...")
    print("[ARCH_EXEC] 🔍 Running backend boot check...")
    ctx.add_trace("BOOT_CHECK_START", "running")

    boot_passed = False
    last_boot_error: Optional[str] = None
    same_error_count = 0

    try:
        for boot_strike in range(1, BOOT_MAX_STRIKES + 1):
            passed, boot_error, full_stderr = _run_boot_check(client, ctx.sandbox_base)

            if passed:
                logger.info("[arch_exec] v2.9 ✓ Backend boot check PASSED (strike %d)", boot_strike)
                print(f"[ARCH_EXEC] ✅ Backend boot check PASSED (attempt {boot_strike})")
                ctx.add_trace("BOOT_CHECK_COMPLETE", "pass", {"attempt": boot_strike})
                boot_passed = True
                break

            logger.error("[arch_exec] v2.9 ✗ Boot check FAILED (strike %d): %s", boot_strike, boot_error[:200])
            print(f"[ARCH_EXEC] ❌ Boot check FAILED (attempt {boot_strike}/{BOOT_MAX_STRIKES}): {boot_error[:200]}")
            ctx.add_trace("BOOT_CHECK_FAIL", f"strike_{boot_strike}", {"error": boot_error[:500]})

            # Track same-error vs new-error
            if boot_error == last_boot_error:
                same_error_count += 1
            else:
                same_error_count = 1
                last_boot_error = boot_error

            if same_error_count >= BOOT_MAX_STRIKES:
                logger.error("[arch_exec] v2.9 Same boot error %d times — giving up", same_error_count)
                break

            if boot_strike >= BOOT_MAX_STRIKES:
                break

            # --- Attempt fix ---
            broken_file = _parse_broken_file(full_stderr, ctx.artifacts_written)
            if not broken_file:
                logger.warning("[arch_exec] v2.9 Cannot identify broken file from traceback")
                print("[ARCH_EXEC] ⚠️ Cannot identify broken file from traceback")
                break

            logger.info("[arch_exec] v2.9 Broken file identified: %s — attempting fix", broken_file)
            print(f"[ARCH_EXEC] 🔧 Attempting fix on: {broken_file}")
            ctx.add_trace("BOOT_FIX_ATTEMPT", f"strike_{boot_strike}", {
                "broken_file": broken_file, "error": boot_error[:300],
            })

            broken_content = await _read_existing_file(client, broken_file)
            if not broken_content:
                logger.warning("[arch_exec] v2.9 Cannot read broken file: %s", broken_file)
                break

            # Resolve relative path for architecture lookup
            broken_rel = broken_file
            for prefix in [ctx.sandbox_base + "\\", "D:\\orb-desktop\\"]:
                if broken_file.startswith(prefix):
                    broken_rel = broken_file[len(prefix):]
                    break
            arch_section = extract_section_for_file(ctx.architecture_content, broken_rel)

            fix_prompt = (
                f"## BOOT CHECK FIX — Strike {boot_strike}\n\n"
                f"The backend failed to start after your changes. "
                f"You MUST fix this file while preserving ALL existing functionality.\n\n"
                f"### Boot Error\n```\n{boot_error}\n```\n\n"
                f"### Full Traceback\n```\n{full_stderr[:2000]}\n```\n\n"
                f"### Current File Content (broken)\n```\n{broken_content}\n```\n\n"
                f"### Architecture Specification For This File\n{arch_section}\n\n"
                f"### CRITICAL RULES\n"
                f"1. Output ONLY the complete fixed file — no markdown fences, no explanations.\n"
                f"2. Fix the boot error shown above.\n"
                f"3. DO NOT remove or break any existing imports, functions, or functionality.\n"
                f"4. The fix must integrate the new feature while keeping everything that already worked.\n"
                f"5. If an import path doesn't exist, remove it or fix it — don't guess.\n"
                f"6. Preserve the file's existing code style and patterns.\n"
            )

            from app.overwatcher.implementer import run_implementer_task
            try:
                fix_result = await ctx.llm_call_fn(
                    provider_id=ctx.impl_provider,
                    model_id=ctx.impl_model,
                    messages=[
                        {"role": "system", "content": IMPLEMENTER_MODIFY_FILE_SYSTEM},
                        {"role": "user", "content": fix_prompt},
                    ],
                    max_tokens=ctx.impl_max_tokens,
                    timeout_seconds=600,
                )
                fix_content = _extract_llm_content(fix_result)
                fix_content = _strip_markdown_fences(fix_content)

                if not fix_content or len(fix_content) < 50:
                    logger.warning("[arch_exec] v2.9 Fix produced empty/minimal content")
                    continue

                write_result = await run_implementer_task(
                    path=broken_file, content=fix_content,
                    action="modify", client=client,
                )
                if write_result.success:
                    logger.info("[arch_exec] v2.9 Fix written: %s (%d chars)", broken_file, len(fix_content))
                    print(f"[ARCH_EXEC] ✓ Fix applied to {broken_file} ({len(fix_content)} chars)")
                else:
                    logger.error("[arch_exec] v2.9 Fix write failed: %s", write_result.error)
                    break
            except Exception as e:
                logger.error("[arch_exec] v2.9 Fix LLM call failed: %s", e)
                break

        if not boot_passed:
            error_final = last_boot_error or "Boot check failed"
            ctx.add_trace("BOOT_CHECK_COMPLETE", "fail", {
                "error": error_final[:500], "strikes": boot_strike,
            })
            ctx.files_failed += ctx.total_succeeded

    except Exception as e:
        logger.warning("[arch_exec] v2.9 Boot check could not run: %s", e)
        print(f"[ARCH_EXEC] ⚠️ Boot check skipped: {e}")
        ctx.add_trace("BOOT_CHECK_COMPLETE", "skipped", {"error": str(e)})
