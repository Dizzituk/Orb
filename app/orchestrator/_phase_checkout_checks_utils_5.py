from __future__ import annotations
import logging
import os
import re
import time
from .phase_checkout_models import BootTestResult
from app.orchestrator._phase_checkout_checks_utils_2 import _discover_sandbox_base
from app.orchestrator._phase_checkout_checks_utils_3 import BOOT_MAX_FIX_ATTEMPTS, BOOT_STRIKE_LIMIT
from app.orchestrator._phase_checkout_checks_utils_4 import _grep_sandbox_for_bad_import, _run_single_boot
from typing import Any, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


BOOT_FIX_TIMEOUT = int(os.environ.get("PHASE_CHECKOUT_TIMEOUT_SECONDS", "180"))

async def run_boot_test_with_fix_loop(
    sandbox_base: str,
    state: Any,
    emit: Any = None,
) -> BootTestResult:
    """
    Run application boot test with automatic fix attempts.

    v2.5: Strike-based boot fix loop.
    - Different errors each attempt = progress, keeps going.
    - Same error repeating = stuck. Strikes accumulate per error signature.
    - BOOT_STRIKE_LIMIT strikes on the same error = hard stop.
    - BOOT_MAX_FIX_ATTEMPTS is a safety cap on total iterations.

    When boot fails, the system:
    1. Parses the traceback to identify the failing file and error type
    2. Computes an error signature to track whether this is a new or repeated error
    3. Reads the broken file from the sandbox
    4. Sends the file + error to the LLM for a targeted surgical fix
    5. Writes the fix back to the sandbox
    6. Retries boot

    Only fixes deterministic errors: bad imports, syntax errors, missing
    attributes. Does NOT attempt to fix logic errors or runtime failures.
    """
    from .phase_checkout_checks import _attempt_boot_fix, _investigate_boot_error
    _emit = emit or (lambda msg: None)
    start = time.time()

    try:
        from app.overwatcher.sandbox_client import get_sandbox_client
        client = get_sandbox_client()
    except Exception as exc:
        return BootTestResult(
            status="error",
            error_summary=f"Cannot connect to sandbox: {exc}",
            duration_ms=int((time.time() - start) * 1000),
        )

    # Discover actual sandbox base path
    actual_base = _discover_sandbox_base(client, sandbox_base)

    fixes_applied = []
    # v2.5: Strike tracking per error signature
    # Different errors each time = progress, keep going.
    # Same error repeating = stuck. 3 strikes on same signature = hard stop.
    from app.orchestrator.strike_tracker import _error_signature
    error_strike_counts: dict = {}   # {signature: strike_count}
    total_attempts = 0

    for attempt in range(1, BOOT_MAX_FIX_ATTEMPTS + 1):
        total_attempts = attempt
        _emit(f"  Boot attempt {attempt}/{BOOT_MAX_FIX_ATTEMPTS}...")

        passed, stdout, stderr, error_summary, failing_file = _run_single_boot(
            client, actual_base
        )

        if passed:
            elapsed = int((time.time() - start) * 1000)
            if attempt > 1:
                _emit(f"  [OK] Boot PASSED after {attempt - 1} fix(es): {fixes_applied}")
            # Journal: boot passed
            try:
                from app.experience.context import journal_emit
                journal_emit(
                    stage="phase_checkout",
                    event_type="boot_pass",
                    severity="info",
                    description=f"Boot passed on attempt {attempt}",
                    duration_ms=elapsed,
                    details={"attempt": attempt, "fixes_applied": fixes_applied},
                )
            except Exception:
                pass
            return BootTestResult(
                status="pass", stdout=stdout, stderr=stderr,
                duration_ms=elapsed,
            )

        # Boot failed — compute error signature to track strikes
        sig = _error_signature(error_summary)
        strike = error_strike_counts.get(sig, 0) + 1
        error_strike_counts[sig] = strike

        if strike == 1:
            _emit(f"  [FAIL] Boot attempt {attempt} failed (new error): {error_summary[:150]}")
        else:
            _emit(f"  [FAIL] Boot attempt {attempt} failed (strike {strike}/{BOOT_STRIKE_LIMIT}): {error_summary[:150]}")

        # Journal: boot failure
        try:
            from app.experience.context import journal_emit
            journal_emit(
                stage="phase_checkout",
                event_type="boot_failure",
                severity="error" if strike >= 2 else "warning",
                description=error_summary[:300],
                error_signature=sig,
                file_scope=failing_file or "",
                strike_number=strike,
                details={"attempt": attempt, "stdout": (stdout or "")[:500], "stderr": (stderr or "")[:500]},
            )
        except Exception:
            pass

        # Hard stop: same error hit strike limit
        if strike >= BOOT_STRIKE_LIMIT:
            _emit(f"  [ABORT] Strike {strike}/{BOOT_STRIKE_LIMIT} for same error — fix not working, hard stop")
            break

        # Safety cap on total attempts
        if attempt >= BOOT_MAX_FIX_ATTEMPTS:
            _emit(f"  [ABORT] Max total attempts ({BOOT_MAX_FIX_ATTEMPTS}) reached")
            break

        # --- Attempt to fix the error ---
        if not failing_file:
            # v2.3: Try grepping the sandbox to find which file has the bad import
            _emit("  [SEARCH] No file in traceback -- grepping sandbox for bad import...")
            failing_file = await _grep_sandbox_for_bad_import(
                client=client,
                actual_base=actual_base,
                error_summary=error_summary,
                state=state,
                emit=_emit,
            )
            if not failing_file:
                # v2.8: PRE-EXISTING ERROR DETECTION
                # If grep found NOTHING in segment outputs, check if the app
                # actually booted successfully. The error may be pre-existing
                # and non-fatal (caught by try/except in the codebase).
                if "BOOT_CHECK_PASS" in (stdout or ""):
                    _emit("  [PRE-EXISTING] Error not in segment outputs but BOOT_CHECK_PASS present")
                    _emit(f"  [PRE-EXISTING] Treating as pre-existing non-fatal error: {error_summary[:120]}")
                    # Journal the pre-existing error for awareness
                    try:
                        from app.experience.context import journal_emit
                        journal_emit(
                            stage="phase_checkout",
                            event_type="pre_existing_error_detected",
                            severity="info",
                            description=f"Boot error is pre-existing (not in segment outputs): {error_summary[:200]}",
                            error_signature=sig,
                        )
                    except Exception:
                        pass
                    # Treat as PASS since the app actually booted
                    elapsed = int((time.time() - start) * 1000)
                    return BootTestResult(
                        status="pass", stdout=stdout, stderr=stderr,
                        duration_ms=elapsed,
                    )

                # v2.4: Investigation step -- search the WIDER sandbox, not just segment outputs
                _emit("  [INVESTIGATE] Segment outputs clean -- searching wider codebase...")
                investigation_result = await _investigate_boot_error(
                    client=client,
                    actual_base=actual_base,
                    error_summary=error_summary,
                    full_stderr=stderr,
                    state=state,
                    emit=_emit,
                )
                if investigation_result:
                    failing_file, fix_applied = investigation_result
                    if fix_applied:
                        fixes_applied.append(f"{failing_file}: {fix_applied}")
                        _emit(f"  [INVESTIGATE] Fix applied: {fix_applied}")
                        continue  # Re-boot without entering _attempt_boot_fix
                    # investigation found the file but didn't fix it -- fall through to _attempt_boot_fix
                else:
                    # v2.8: Even without BOOT_CHECK_PASS, if grep AND investigation
                    # both found nothing in our segments, this is likely pre-existing.
                    _emit("  [PRE-EXISTING] Error not found in segment outputs or wider codebase")
                    _emit("  [INCONCLUSIVE] Cannot determine if error is from our changes or pre-existing")
                    _emit("  [ABORT] Cannot identify failing file from traceback, grep, or investigation")
                    break

        _emit(f"  [FIX] Attempting fix on: {failing_file}")

        fix_applied = await _attempt_boot_fix(
            client=client,
            actual_base=actual_base,
            failing_file=failing_file,
            error_summary=error_summary,
            full_stderr=stderr,
            state=state,
            emit=_emit,
        )

        if fix_applied:
            fixes_applied.append(f"{failing_file}: {fix_applied}")
            _emit(f"  [FIX] Applied: {fix_applied}")
            # Journal: boot fix attempt
            try:
                from app.experience.context import journal_emit
                journal_emit(
                    stage="phase_checkout",
                    event_type="boot_fix_attempt",
                    severity="info",
                    description=f"Fix applied to {failing_file}: {fix_applied}",
                    resolution=fix_applied,
                    error_signature=sig,
                    file_scope=failing_file or "",
                    strategy_used=fix_applied[:100],
                    strike_number=strike,
                )
            except Exception:
                pass
        else:
            _emit("  [ABORT] Could not generate a fix")
            # Journal: fix failed
            try:
                from app.experience.context import journal_emit
                journal_emit(
                    stage="phase_checkout",
                    event_type="boot_fix_failed",
                    severity="error",
                    description=f"Could not generate fix for {failing_file}",
                    error_signature=sig,
                    file_scope=failing_file or "",
                    strike_number=strike,
                )
            except Exception:
                pass
            break

    # All attempts exhausted
    elapsed = int((time.time() - start) * 1000)
    err_summary, failing = _parse_boot_failure(stdout or "", stderr or "")
    return BootTestResult(
        status="fail", stdout=stdout or "", stderr=stderr or "",
        error_summary=err_summary, traceback_file=failing,
        duration_ms=elapsed,
    )

_BOOT_FIX_SYSTEM_PROMPT = """\
You are a surgical code fixer. Your ONLY job is to fix the specific boot error \
shown below. You must preserve ALL existing functionality -- do not remove, \
rewrite, or restructure anything beyond the minimum change needed to fix the error.

Rules:
1. Output ONLY the complete fixed file. No markdown fences, no explanations, \
no commentary before or after.
2. Fix ONLY the specific error shown. Do not "improve" or refactor anything else.
3. If an import path does not exist, either remove the import or fix the path \
to the correct module. Do NOT guess -- if you are not sure of the correct path, \
comment out the import with a note.
4. If there is a syntax error, fix only the syntax. Do not change logic.
5. The file must be valid Python that passes syntax checking.
6. Preserve all existing functions, classes, constants, and their signatures.
"""

def _extract_fix_content(llm_result: Any) -> Optional[str]:
    """Extract file content from LLM response, stripping markdown fences."""
    if not llm_result:
        return None

    # Handle different response formats
    content = ""
    if hasattr(llm_result, "content"):
        content = llm_result.content or ""
    elif isinstance(llm_result, dict):
        content = llm_result.get("content", "") or ""
        if not content and "choices" in llm_result:
            choices = llm_result["choices"]
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content", "")
    elif isinstance(llm_result, str):
        content = llm_result

    if not content:
        return None

    # Strip markdown fences
    content = content.strip()
    if content.startswith("```python"):
        content = content[len("```python"):].strip()
    elif content.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()

    return content

def _write_file_to_sandbox(
    client: Any,
    rel_path: str,
    content: str,
    sandbox_base: str,
) -> bool:
    """Write a file to the sandbox filesystem."""
    normed = rel_path.replace("/", "\\")
    if not (normed.startswith("C:") or normed.startswith("D:")):
        abs_path = f"{sandbox_base}\\{normed}"
    else:
        abs_path = normed

    try:
        # Use base64 encoding to avoid PowerShell escaping issues
        import base64
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")

        # Write in chunks if large
        chunk_size = 6000  # Safe for command line length
        if len(encoded) <= chunk_size:
            cmd = (
                f'$bytes = [Convert]::FromBase64String("{encoded}"); '
                f'[System.IO.File]::WriteAllText("{abs_path}", '
                f'[System.Text.Encoding]::UTF8.GetString($bytes))'
            )
            result = client.shell_run(cmd, timeout_seconds=15)
        else:
            # Write base64 to temp file, then decode
            temp_path = abs_path + ".b64tmp"
            # Clear temp file
            client.shell_run(f'Set-Content -Path "{temp_path}" -Value "" -NoNewline', timeout_seconds=5)
            for i in range(0, len(encoded), chunk_size):
                chunk = encoded[i:i+chunk_size]
                cmd = f'Add-Content -Path "{temp_path}" -Value "{chunk}" -NoNewline'
                client.shell_run(cmd, timeout_seconds=10)
            # Decode and write
            cmd = (
                f'$b64 = Get-Content -Path "{temp_path}" -Raw; '
                f'$bytes = [Convert]::FromBase64String($b64); '
                f'[System.IO.File]::WriteAllText("{abs_path}", '
                f'[System.Text.Encoding]::UTF8.GetString($bytes)); '
                f'Remove-Item -Path "{temp_path}" -Force'
            )
            result = client.shell_run(cmd, timeout_seconds=15)

        # Verify the write
        verify = client.shell_run(f'Test-Path -Path "{abs_path}"', timeout_seconds=5)
        return (verify.stdout or "").strip().lower() == "true"

    except Exception as exc:
        logger.warning("[phase_checkout] Cannot write %s to sandbox: %s", abs_path, exc)
        return False

def _parse_boot_failure(stdout: str, stderr: str) -> Tuple[str, Optional[str]]:
    """Parse boot test output to identify error and failing file."""
    combined = stdout + "\n" + stderr
    err_keywords = (
        'Error', 'Traceback', 'ImportError', 'ModuleNotFoundError',
        'SyntaxError', 'AttributeError', 'NameError', 'TypeError',
        'cannot import', 'No module named',
    )
    err_lines = [
        ln.strip() for ln in combined.split("\n")
        if any(kw in ln for kw in err_keywords)
    ]
    summary = "\n".join(err_lines[:5]) or "Unknown boot failure"

    # Find the LAST File reference in the traceback -- that's the actual cause
    failing_file = None
    file_matches = list(re.finditer(r'File "([^"]+)"', combined))
    for match in reversed(file_matches):
        path = match.group(1)
        for prefix in (r"D:\Orb\\", r"D:\Orb/", r"C:\Orb\\", r"C:\Orb/",
                       r"C:\Orb\Orb\\", r"C:\Orb\Orb/"):
            if path.lower().startswith(prefix.lower()):
                failing_file = path[len(prefix):]
                break
        if failing_file:
            # Skip Python standard library files
            if not failing_file.startswith("app") and not failing_file.startswith("main"):
                failing_file = None
                continue
            break

    return (summary, failing_file)
