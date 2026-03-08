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
        from app.sandbox.client import get_sandbox_client
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

# v3.3-fix: Frontend path prefixes that must resolve to D:\orb-desktop
_PC_FRONTEND_PREFIX = "orb-desktop/"
_PC_FRONTEND_ROOT = r"D:\orb-desktop"
_PC_FRONTEND_BARE_PREFIXES = ("src/", "public/")

def _write_file_to_sandbox(
    client: Any,
    rel_path: str,
    content: str,
    sandbox_base: str,
) -> bool:
    """Write a file to the sandbox filesystem.

    v3.3-fix: Now resolves frontend paths (orb-desktop/ prefix or bare
    src/, public/) to D:\\orb-desktop instead of D:\\Orb\\src.
    """
    normed = rel_path.replace("/", "\\")
    normalized_fwd = rel_path.replace("\\", "/")

    if normed.startswith("C:") or normed.startswith("D:"):
        abs_path = normed
    elif normalized_fwd.startswith(_PC_FRONTEND_PREFIX):
        frontend_rel = normalized_fwd[len(_PC_FRONTEND_PREFIX):]
        abs_path = _PC_FRONTEND_ROOT + "\\" + frontend_rel.replace("/", "\\")
    elif any(normalized_fwd.startswith(bp) for bp in _PC_FRONTEND_BARE_PREFIXES):
        abs_path = _PC_FRONTEND_ROOT + "\\" + normed
    else:
        abs_path = f"{sandbox_base}\\{normed}"

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
    """Parse boot test output to identify error and failing file.

    v3.2: For AttributeError on imported types (e.g. 'type object X has no
    attribute Y'), traces the import chain to find the file that *defines*
    the broken class, not just the file that *uses* it. This is critical
    because the fix needs to be applied to the defining module.
    """
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

    # Collect all File references in the traceback as relative paths
    all_files_in_trace: list = []
    file_matches = list(re.finditer(r'File "([^"]+)"', combined))
    for match in file_matches:
        path = match.group(1)
        rel = _strip_sandbox_prefix(path)
        if rel and (rel.startswith("app") or rel.startswith("main")):
            all_files_in_trace.append(rel)

    # v3.2: For AttributeError on a type/class, find the IMPORT SOURCE.
    # Pattern: "from app.builds.models import PipelineStage" in the using file,
    # then "AttributeError: type object 'PipelineStage' has no attribute 'weaver'"
    # The fix needs to go to app/builds/models.py, not the using file.
    attr_match = re.search(
        r"AttributeError: type object '(\w+)' has no attribute '(\w+)'",
        combined,
    )
    if attr_match and all_files_in_trace:
        class_name = attr_match.group(1)
        # The last file in the traceback is the using file — look for
        # where it imports the class FROM.
        using_file = all_files_in_trace[-1] if all_files_in_trace else None
        import_source = _trace_import_source(combined, using_file, class_name)
        if import_source:
            logger.info(
                "[phase_checkout] v3.2 AttributeError traced: %s defined in %s (used in %s)",
                class_name, import_source, using_file,
            )
            return (summary, import_source)

    # Default: last File reference in the traceback is the cause
    failing_file = None
    for match in reversed(file_matches):
        path = match.group(1)
        rel = _strip_sandbox_prefix(path)
        if rel and (rel.startswith("app") or rel.startswith("main")):
            failing_file = rel
            break

    return (summary, failing_file)


def _strip_sandbox_prefix(path: str) -> Optional[str]:
    """Strip D:\\Orb\\ or similar sandbox prefix from an absolute path."""
    for prefix in (
        r"D:\Orb\\", "D:\\Orb\\", r"D:\Orb/", "D:/Orb/",
        r"C:\Orb\\", r"C:\Orb/",
        r"C:\Orb\Orb\\", r"C:\Orb\Orb/",
    ):
        if path.lower().startswith(prefix.lower()):
            return path[len(prefix):]
    # Also handle double-backslash paths from traceback formatting
    m = re.match(r'[CD]:\\+Orb\\+(.*)', path, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _trace_import_source(
    combined_output: str, using_file: Optional[str], class_name: str,
) -> Optional[str]:
    """v3.2: Trace where a class is imported from by scanning the traceback.

    Looks for 'from X import <class_name>' in the traceback context or
    infers the source module from the import chain visible in the File
    references.

    For example, if the traceback shows:
      File "D:\\Orb\\app\\builds\\service.py", line 13, in <module>
        from app.builds.models import BuildProject, PipelineStage
    and the error is AttributeError on PipelineStage, we return
    'app/builds/models.py' (converted to relative path with os.sep).
    """
    # Strategy 1: Look for explicit 'from X import class_name' in traceback
    import_pattern = re.compile(
        r'from\s+([\w.]+)\s+import\s+[\w\s,]*\b' + re.escape(class_name) + r'\b',
    )
    m = import_pattern.search(combined_output)
    if m:
        module_path = m.group(1).replace('.', os.sep) + '.py'
        return module_path

    # Strategy 2: If the traceback has the file chain A -> B -> error,
    # and B is the last file, check if the PREVIOUS file in the chain
    # is the module that defines the type.
    file_matches = list(re.finditer(r'File "([^"]+)"', combined_output))
    app_files = []
    for fm in file_matches:
        rel = _strip_sandbox_prefix(fm.group(1))
        if rel and (rel.startswith("app") or rel.startswith("main")):
            app_files.append(rel)

    # If the import chain shows: router.py -> service.py -> (error on line
    # that references PipelineStage), then service.py is where the import
    # statement is. We already handled this with Strategy 1 above.
    # As a fallback, look at the penultimate file.
    if len(app_files) >= 2:
        penultimate = app_files[-2]
        # If the penultimate file's module name contains the class name
        # (e.g. models.py likely defines PipelineStage), prefer it
        penult_stem = os.path.basename(penultimate).replace('.py', '').lower()
        if penult_stem in ('models', 'schemas', 'types', 'enums', 'constants'):
            logger.info(
                "[phase_checkout] v3.2 Penultimate file %s is likely defining module",
                penultimate,
            )
            return penultimate

    return None
