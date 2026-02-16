# FILE: app/orchestrator/phase_checkout_checks.py
"""
Phase Checkout -- Individual Verification Checks.

Contains the implementation of each check run during Phase Checkout:
1. Output file size validation (informational -- only checks segment outputs)
2. Skeleton contract verification (informational -- checks via sandbox)
3. Application boot test with fix loop (pass/fail gate)

v2.0 (2026-02-15): Boot test is now a fix loop. When boot fails, the system:
    1. Parses the traceback to identify the failing file and error type
    2. Reads the broken file from the sandbox
    3. Sends the file + error to the LLM for a targeted fix
    4. Writes the fix back to the sandbox
    5. Retries boot (up to BOOT_MAX_FIX_ATTEMPTS times)
    Size check now only validates segment-produced output files, not
    pre-existing source files. Contract check resolves paths via sandbox.
v2.7 (2026-02-16): Silent import failure detection hardened. Boot check
    now scans BOTH stdout and stderr for import warnings (not just stderr).
    Fixes regex group mismatch: "cannot import name X from Y" now correctly
    checks Y (the module path) for app.* prefix, not X (the symbol name).
    Journal emit on silent import failure for learning system.
v2.6 (2026-02-15): Relative import awareness. When boot fails with
    "cannot import name 'X' from 'a.b.c.module'", grep and investigation
    now also search for relative import forms like "from .module import X".
    Previously only searched for the full dotted absolute path, which missed
    __init__.py files and sibling modules using relative imports.
v2.5 (2026-02-15): Strike-based boot fix loop. Different errors = progress,
    keeps going. Same error repeating = strikes accumulate per error signature.
    3 strikes on same error = hard stop. Safety cap of 12 total attempts.
    Default raised from 3 to 12 max attempts.
v2.4 (2026-02-15): Investigation step before abort. When failing file
    not found in segment outputs, searches the wider sandbox codebase,
    reads the importing file, and hands it to the LLM to investigate
    and fix -- same as a human would do.
v1.0 (2026-02-14): Extracted from phase_checkout.py for cap compliance.
"""

from __future__ import annotations

import ast
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from app.pot_spec.grounded.size_models import (
    MAX_FILE_KB,
    MAX_FILE_LINES,
    MAX_FUNCTION_LINES,
)
from .phase_checkout_models import (
    BootTestResult,
    ContractCheckResult,
    ContractViolation,
    SizeValidationResult,
    SizeViolation,
)

logger = logging.getLogger(__name__)

BOOT_MAX_FIX_ATTEMPTS = int(os.environ.get("PHASE_CHECKOUT_MAX_FIX_ATTEMPTS", "12"))
BOOT_FIX_TIMEOUT = int(os.environ.get("PHASE_CHECKOUT_TIMEOUT_SECONDS", "180"))
BOOT_STRIKE_LIMIT = int(os.environ.get("PHASE_CHECKOUT_STRIKE_LIMIT", "3"))


# =============================================================================
# CHECK 1: OUTPUT FILE SIZE VALIDATION
# =============================================================================

def check_output_file_sizes(
    state: Any,
    sandbox_base: str,
) -> SizeValidationResult:
    """
    Scan segment-produced output files for size constraint violations.

    v2.0: Only checks files that segments actually produced (output_files).
    Pre-existing source files that are being replaced/decomposed are NOT checked.

    Checks:
    - File line count <= MAX_FILE_LINES (400)
    - File size <= MAX_FILE_KB (15 KB)
    - Largest function body <= MAX_FUNCTION_LINES (200)
    """
    violations: List[SizeViolation] = []
    files_checked = 0

    # v2.0: Check files via sandbox, not host filesystem
    try:
        from app.overwatcher.sandbox_client import get_sandbox_client
        client = get_sandbox_client()
        use_sandbox = client.is_connected()
    except Exception:
        use_sandbox = False
        client = None

    for seg_id, seg_state in state.segments.items():
        for rel_path in (seg_state.output_files or []):
            # Read file content -- prefer sandbox, fallback to host
            content = None
            if use_sandbox and client:
                content = _read_file_via_sandbox(client, rel_path, sandbox_base)
            if content is None:
                abs_path = _resolve_output_path(rel_path, sandbox_base)
                if abs_path and os.path.isfile(abs_path):
                    try:
                        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                    except Exception:
                        continue

            if content is None:
                continue

            files_checked += 1
            line_count = content.count("\n") + 1
            kb_size = round(len(content.encode("utf-8")) / 1024, 1)

            if line_count > MAX_FILE_LINES:
                violations.append(SizeViolation(
                    file_path=rel_path, line_count=line_count,
                    kb_size=kb_size, produced_by_segment=seg_id,
                    violation_type="file_too_large",
                ))
            elif kb_size > MAX_FILE_KB:
                violations.append(SizeViolation(
                    file_path=rel_path, line_count=line_count,
                    kb_size=kb_size, produced_by_segment=seg_id,
                    violation_type="file_too_large_kb",
                ))

            # Function-level check (Python only)
            if rel_path.endswith(".py"):
                max_fn_lines, max_fn_name = _find_largest_function(content)
                if max_fn_lines > MAX_FUNCTION_LINES:
                    violations.append(SizeViolation(
                        file_path=rel_path, line_count=line_count,
                        kb_size=kb_size, max_function_lines=max_fn_lines,
                        max_function_name=max_fn_name,
                        produced_by_segment=seg_id,
                        violation_type="function_too_large",
                    ))

    return SizeValidationResult(
        status="fail" if violations else "pass",
        files_checked=files_checked,
        violations=violations,
    )


def _find_largest_function(source_code: str) -> Tuple[int, str]:
    """Find the largest function body in a Python file."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return (0, "")

    max_lines = 0
    max_name = ""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if hasattr(node, "end_lineno") and node.end_lineno:
                fn_lines = node.end_lineno - node.lineno + 1
                if fn_lines > max_lines:
                    max_lines = fn_lines
                    max_name = node.name
    return (max_lines, max_name)


# =============================================================================
# CHECK 2: SKELETON CONTRACT VERIFICATION
# =============================================================================

def check_skeleton_contracts(
    state: Any,
    skeleton: Any,
    sandbox_base: str,
) -> ContractCheckResult:
    """
    Verify each segment delivered its skeleton-promised exports.

    v2.0: Checks file existence via sandbox (not host filesystem).
    Scope violations for package creation are handled -- if a segment
    creates a package from a monolith, the package path is valid scope.
    """
    violations: List[ContractViolation] = []

    # v2.0: Use sandbox for file existence checks
    try:
        from app.overwatcher.sandbox_client import get_sandbox_client
        client = get_sandbox_client()
        use_sandbox = client.is_connected()
    except Exception:
        use_sandbox = False
        client = None

    for skel in skeleton.skeletons:
        seg_id = skel.segment_id

        for export in skel.exports:
            exists = False
            if use_sandbox and client:
                exists = _file_exists_in_sandbox(client, export.file_path, sandbox_base)
            else:
                abs_path = _resolve_output_path(export.file_path, sandbox_base)
                exists = abs_path and os.path.isfile(abs_path)

            if not exists:
                violations.append(ContractViolation(
                    segment_id=seg_id,
                    violation_type="missing_export",
                    detail=f"Export '{export.file_path}' not found in sandbox",
                ))

        # v2.0: Scope check -- allow package paths that extend the original scope
        seg_state = state.segments.get(seg_id)
        if seg_state and seg_state.output_files:
            scope_set = {_norm(f) for f in skel.file_scope}
            for out_file in seg_state.output_files:
                normed = _norm(out_file)
                if normed not in scope_set:
                    # v2.0: Check if this is a package expansion -- if the file
                    # is under a directory that matches a scope entry (minus .py),
                    # it's a valid package decomposition, not a scope violation.
                    is_package_expansion = False
                    for scope_entry in scope_set:
                        # e.g. scope has "app/overwatcher/architecture_executor.py"
                        # output is "app/overwatcher/architecture_executor/constants.py"
                        base_no_ext = scope_entry.rsplit(".", 1)[0] if "." in scope_entry else scope_entry
                        if normed.startswith(base_no_ext + "/"):
                            is_package_expansion = True
                            break
                    if not is_package_expansion:
                        violations.append(ContractViolation(
                            segment_id=seg_id,
                            violation_type="scope_violation",
                            detail=f"Output '{out_file}' not in segment file_scope",
                        ))

    return ContractCheckResult(
        status="fail" if violations else "pass",
        violations=violations,
    )


# =============================================================================
# CHECK 3: BOOT TEST WITH FIX LOOP
# =============================================================================

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
                # v2.4: Investigation step -- search the WIDER sandbox, not just segment outputs
                _emit("  [INVESTIGATE] Segment outputs clean -- searching wider codebase...")
                investigation_result = await _investigate_boot_error(
                    client=client,
                    actual_base=actual_base,
                    error_summary=error_summary,
                    full_stderr=stderr,
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


def _run_single_boot(
    client: Any,
    actual_base: str,
) -> Tuple[bool, str, str, str, Optional[str]]:
    """
    Run a single boot test. Returns (passed, stdout, stderr, error_summary, failing_file).
    """
    venv = actual_base + r"\.venv\Scripts\python.exe"
    # v2.2: Force UTF-8 to prevent UnicodeEncodeError from emoji in generated code
    cmd = (
        f'$env:PYTHONIOENCODING="ascii:replace"; $env:PYTHONUTF8="0" ; '
        f'cd "{actual_base}" ; '
        f'& "{venv}" -c '
        f'"import sys; sys.path.insert(0, r\'{actual_base}\'); '
        f'from main import app; print(\'BOOT_CHECK_PASS\')"'
    )

    try:
        shell_result = client.shell_run(cmd, timeout_seconds=30)
    except Exception as exc:
        return (False, "", str(exc), f"Shell execution failed: {exc}", None)

    stdout = (shell_result.stdout or "").strip()
    stderr = (shell_result.stderr or "").strip()

    if "BOOT_CHECK_PASS" in stdout:
        # v2.7: Scan BOTH stdout and stderr for silent import failures.
        # The app may boot (caught in try/except) but still have broken imports.
        # These are real bugs that need fixing. Warnings appear in stdout (via
        # print) and/or stderr (via logger.warning).
        silent_import_error = _check_stderr_for_silent_import_failures(stderr, stdout)
        if silent_import_error:
            # v2.7: Journal the silent failure so the learning system captures it
            try:
                from app.experience.context import journal_emit
                journal_emit(
                    stage="phase_checkout",
                    event_type="silent_import_failure",
                    severity="error",
                    description=f"Boot passed but critical import failed silently: {silent_import_error[:200]}",
                    error_signature=silent_import_error[:100],
                    root_cause="LLM-generated code imports a symbol that does not exist in the target module",
                    resolution="Boot check now treats silent import failures as boot failures",
                )
            except Exception:
                pass
            return (False, stdout, stderr, silent_import_error, None)
        return (True, stdout, stderr, "", None)

    err_summary, failing_file = _parse_boot_failure(stdout, stderr)
    return (False, stdout, stderr, err_summary, failing_file)


async def _attempt_boot_fix(
    client: Any,
    actual_base: str,
    failing_file: str,
    error_summary: str,
    full_stderr: str,
    state: Any,
    emit: Any,
) -> Optional[str]:
    """
    Attempt to fix a boot failure by reading the broken file, sending it
    to the LLM with the error context, and writing the fix back.

    Returns a short description of the fix applied, or None if fix failed.
    """
    _emit = emit or (lambda msg: None)

    # Determine error type for targeted fix
    err_lower = error_summary.lower()
    is_import_error = ("modulenotfounderror" in err_lower or "importerror" in err_lower
                       or "cannot import" in err_lower or "no module named" in err_lower)
    is_syntax_error = "syntaxerror" in err_lower
    is_attribute_error = "attributeerror" in err_lower
    is_name_error = "nameerror" in err_lower

    if not (is_import_error or is_syntax_error or is_attribute_error or is_name_error):
        _emit(f"    Error type not fixable by phase checkout: {error_summary[:100]}")
        return None

    # Read the broken file from sandbox
    broken_content = _read_file_via_sandbox(client, failing_file, actual_base)
    if not broken_content:
        _emit(f"    Cannot read broken file: {failing_file}")
        return None

    # --- For import errors, try smart reconciliation first (no LLM needed) ---
    if is_import_error:
        # v2.1: Try post-execution reconciliation to find the correct import name
        # instead of just commenting out the broken import.
        recon_fix = _try_reconciliation_import_fix(
            client, actual_base, failing_file, broken_content, error_summary, state, _emit,
        )
        if recon_fix:
            fixed_content, fix_desc = recon_fix
            success = _write_file_to_sandbox(client, failing_file, fixed_content, actual_base)
            if success:
                return fix_desc

        # Fallback: comment out the broken import as a last resort
        deterministic_fix = _try_deterministic_import_fix(
            broken_content, error_summary, full_stderr
        )
        if deterministic_fix:
            fixed_content, fix_desc = deterministic_fix
            success = _write_file_to_sandbox(client, failing_file, fixed_content, actual_base)
            if success:
                return fix_desc

    # --- Fall back to LLM-based fix ---
    fix_prompt = _build_fix_prompt(
        failing_file=failing_file,
        broken_content=broken_content,
        error_summary=error_summary,
        full_stderr=full_stderr,
        error_type="import" if is_import_error else
                   "syntax" if is_syntax_error else
                   "attribute" if is_attribute_error else "name",
    )

    try:
        from app.providers.registry import get_provider_registry
        registry = get_provider_registry()

        # Use a capable model for the fix
        provider_id = _pick_boot_fix_provider()
        model_id = _pick_boot_fix_model()
        llm_result = await registry.llm_call(
            provider_id=provider_id,
            model_id=model_id,
            messages=[
                {"role": "system", "content": _BOOT_FIX_SYSTEM_PROMPT},
                {"role": "user", "content": fix_prompt},
            ],
            max_tokens=_pick_boot_fix_max_tokens(),
            timeout_seconds=BOOT_FIX_TIMEOUT,
        )

        fixed_content = _extract_fix_content(llm_result)
        if not fixed_content or len(fixed_content) < 20:
            _emit("    LLM fix produced empty/minimal content")
            return None

        # Validate the fix isn't destroying the file
        if len(fixed_content) < len(broken_content) * 0.5:
            _emit("    LLM fix removed too much content (>50% reduction) -- rejecting")
            return None

        success = _write_file_to_sandbox(client, failing_file, fixed_content, actual_base)
        if success:
            return f"LLM fix applied ({len(fixed_content)} chars)"
        return None

    except Exception as exc:
        _emit(f"    LLM fix failed: {exc}")
        logger.warning("[phase_checkout] LLM boot fix failed: %s", exc)
        return None


async def _grep_sandbox_for_bad_import(
    client: Any,
    actual_base: str,
    error_summary: str,
    state: Any,
    emit: Any,
) -> Optional[str]:
    """
    v2.2: When _parse_boot_failure can't find a File path in the traceback,
    grep the sandbox for files that contain the offending import.

    Parses error_summary for patterns like:
      - "No module named 'app.logger'"
      - "cannot import name 'X' from 'Y'"

    Then runs Select-String on the sandbox to find which .py file has that import.
    Returns the relative file path, or None.
    """
    _emit = emit or (lambda msg: None)

    # Extract the bad module name
    no_module = re.search(r"No module named '([^']+)'", error_summary)
    cannot_import = re.search(r"cannot import name '([^']+)' from '([^']+)'", error_summary)

    search_pattern = None
    if no_module:
        bad_module = no_module.group(1)
        # Search for 'from app.logger' or 'import app.logger'
        search_pattern = bad_module.replace(".", r"\.")
    elif cannot_import:
        bad_name = cannot_import.group(1)
        bad_source = cannot_import.group(2)
        # v2.6: Also search for RELATIVE imports (e.g. "from .constants import X")
        # When bad_source is "app.overwatcher.architecture_executor.constants",
        # the file doing the import might use "from .constants import X" (relative).
        rel_module = bad_source.rsplit(".", 1)[-1]  # "constants"
        search_pattern = (
            f"{bad_name}.*{bad_source.replace('.', r'\.')}|"
            f"{bad_source.replace('.', r'\.')}.*{bad_name}|"
            f"from \\.{rel_module}.*import.*{bad_name}"
        )

    if not search_pattern:
        return None

    _emit(f"  [GREP] Searching sandbox for files referencing '{search_pattern}'...")

    # Grep across all segment output files on the sandbox
    # Build a list of paths to search from the state
    search_paths = []
    for seg_id, seg_state in state.segments.items():
        for rel_path in (seg_state.output_files or []):
            if rel_path.endswith(".py"):
                normed = rel_path.replace("/", "\\")
                if not (normed.startswith("C:") or normed.startswith("D:")):
                    search_paths.append(f"{actual_base}\\{normed}")
                else:
                    search_paths.append(normed)

    if not search_paths:
        _emit("  [GREP] No output files to search")
        return None

    # Run Select-String on the sandbox — batch the paths
    # Use a simple grep command
    paths_str = ",".join(f'"{p}"' for p in search_paths[:30])  # Cap at 30 files
    cmd = (
        f'Select-String -Path {paths_str} '
        f'-Pattern "{search_pattern}" -SimpleMatch '
        f'| Select-Object -First 3 -Property Filename, LineNumber, Line '
        f'| Format-List'
    )

    try:
        result = client.shell_run(cmd, timeout_seconds=15)
        stdout = (result.stdout or "").strip()

        if not stdout:
            # SimpleMatch didn't find it — try without SimpleMatch for regex
            cmd2 = (
                f'Select-String -Path {paths_str} '
                f'-Pattern "{search_pattern}" '
                f'| Select-Object -First 3 -Property Filename, LineNumber, Line '
                f'| Format-List'
            )
            result2 = client.shell_run(cmd2, timeout_seconds=15)
            stdout = (result2.stdout or "").strip()

        if not stdout:
            _emit("  [GREP] No matches found on sandbox")
            return None

        _emit(f"  [GREP] Found: {stdout[:200]}")

        # Extract filename from the result
        fn_match = re.search(r'Filename\s*:\s*(\S+)', stdout)
        if fn_match:
            filename = fn_match.group(1)
            # Find the full relative path from state
            for seg_id, seg_state in state.segments.items():
                for rel_path in (seg_state.output_files or []):
                    if rel_path.endswith(filename):
                        _emit(f"  [GREP] Identified offending file: {rel_path} (from {seg_id})")
                        return rel_path
            # Fallback: return just the filename
            _emit(f"  [GREP] Found file {filename} but can't map to segment")
            return filename

        return None

    except Exception as exc:
        _emit(f"  [GREP] Sandbox search failed: {exc}")
        return None


# =============================================================================
# INVESTIGATION STEP (v2.4)
# =============================================================================

async def _investigate_boot_error(
    client: Any,
    actual_base: str,
    error_summary: str,
    full_stderr: str,
    emit: Any,
) -> Optional[Tuple[str, Optional[str]]]:
    """
    v2.4: When the failing file is not in segment outputs, investigate the
    wider sandbox codebase. This is what a human would do: search the whole
    project for who imports the bad module, read that file, understand the
    context, and decide what to fix.

    Steps:
    1. Search the ENTIRE sandbox (not just segment outputs) for the bad import
    2. Read the file that imports it
    3. Look at the target -- does the module exist? Was it renamed?
    4. Hand everything to the LLM to decide the fix
    5. Apply the fix

    Returns (failing_file, fix_description) if fixed, (failing_file, None) if
    file found but not fixed, or None if investigation found nothing.
    """
    _emit = emit or (lambda msg: None)

    # Extract the bad module name
    no_module = re.search(r"No module named '([^']+)'", error_summary)
    cannot_import = re.search(r"cannot import name '([^']+)' from '([^']+)'", error_summary)

    if not no_module and not cannot_import:
        return None

    if no_module:
        bad_module = no_module.group(1)
        search_term = bad_module
    else:
        bad_name = cannot_import.group(1)
        bad_source = cannot_import.group(2)
        search_term = bad_source
        # v2.6: Also search for relative import form
        rel_module = bad_source.rsplit(".", 1)[-1]
        rel_search_term = f".{rel_module}"

    # Step 1: Search the sandbox app directory for files importing this module
    # Scoped to app/ only — excludes .venv, node_modules, etc. which cause timeouts
    _emit(f"  [INVESTIGATE] Searching sandbox app directory for imports of '{search_term}' (and relative '{rel_search_term}')...")
    try:
        app_dir = f"{actual_base}\\app"
        grep_cmd = (
            f'Get-ChildItem -Path "{app_dir}" -Filter "*.py" -Recurse '
            f'-ErrorAction SilentlyContinue '
            f'| Select-String -Pattern "{search_term}" -SimpleMatch '
            f'| Select-Object -First 5 -Property Path, LineNumber, Line '
            f'| Format-List'
        )
        result = client.shell_run(grep_cmd, timeout_seconds=30)
        grep_output = (result.stdout or "").strip()

        # v2.6: If nothing found with absolute path, try relative import form
        if not grep_output and cannot_import:
            rel_grep_cmd = (
                f'Get-ChildItem -Path "{app_dir}" -Filter "*.py" -Recurse '
                f'-ErrorAction SilentlyContinue '
                f'| Select-String -Pattern "{rel_search_term}" -SimpleMatch '
                f'| Select-Object -First 5 -Property Path, LineNumber, Line '
                f'| Format-List'
            )
            result_rel = client.shell_run(rel_grep_cmd, timeout_seconds=30)
            grep_output = (result_rel.stdout or "").strip()
            if grep_output:
                _emit(f"  [INVESTIGATE] Found via relative import pattern '{rel_search_term}'")

        # If nothing in app/, also check main.py and top-level files
        if not grep_output:
            top_cmd = (
                f'Select-String -Path "{actual_base}\\*.py" '
                f'-Pattern "{search_term}" -SimpleMatch -ErrorAction SilentlyContinue '
                f'| Select-Object -First 3 -Property Path, LineNumber, Line '
                f'| Format-List'
            )
            result2 = client.shell_run(top_cmd, timeout_seconds=10)
            grep_output = (result2.stdout or "").strip()
    except Exception as exc:
        _emit(f"  [INVESTIGATE] Sandbox search failed: {exc}")
        return None

    if not grep_output:
        _emit("  [INVESTIGATE] No files in entire sandbox reference this module")
        return None

    _emit(f"  [INVESTIGATE] Found references:\n{grep_output[:500]}")

    # Step 2: Extract the file path(s) that import the bad module
    path_matches = re.findall(r'Path\s*:\s*(.+?\.py)', grep_output)
    if not path_matches:
        _emit("  [INVESTIGATE] Could not parse file paths from search results")
        return None

    # Pick the first importing file (most likely the one causing the boot error)
    importing_file_abs = path_matches[0].strip()
    _emit(f"  [INVESTIGATE] Importing file: {importing_file_abs}")

    # Read the importing file
    importing_content = None
    try:
        result = client.shell_run(
            f'Get-Content -Path "{importing_file_abs}" -Raw -Encoding UTF8',
            timeout_seconds=15,
        )
        importing_content = (result.stdout or "").strip()
    except Exception:
        pass

    if not importing_content:
        _emit(f"  [INVESTIGATE] Cannot read {importing_file_abs}")
        return None

    # Step 3: Check if the target module/file exists (maybe it was renamed)
    target_investigation = ""
    if no_module:
        # Convert module path to file path: app.logger_setup -> app/logger_setup.py
        module_as_path = bad_module.replace(".", "\\") + ".py"
        module_as_pkg = bad_module.replace(".", "\\") + "\\__init__.py"
        target_file_path = f"{actual_base}\\{module_as_path}"
        target_pkg_path = f"{actual_base}\\{module_as_pkg}"

        # Check if target exists
        try:
            r1 = client.shell_run(f'Test-Path -Path "{target_file_path}" -PathType Leaf', timeout_seconds=10)
            r2 = client.shell_run(f'Test-Path -Path "{target_pkg_path}" -PathType Leaf', timeout_seconds=10)
            target_exists_file = (r1.stdout or "").strip().lower() == "true"
            target_exists_pkg = (r2.stdout or "").strip().lower() == "true"
        except Exception:
            target_exists_file = False
            target_exists_pkg = False

        if target_exists_file or target_exists_pkg:
            target_investigation += f"Target module file EXISTS at {target_file_path if target_exists_file else target_pkg_path}. The import should work -- investigate why it fails.\n"
        else:
            target_investigation += f"Target module file does NOT exist at {target_file_path} or {target_pkg_path}.\n"
            # Look for similar files in the same directory
            parent_dir = os.path.dirname(target_file_path)
            try:
                ls_result = client.shell_run(
                    f'Get-ChildItem -Path "{parent_dir}" -Filter "*.py" -ErrorAction SilentlyContinue '
                    f'| Select-Object -ExpandProperty Name',
                    timeout_seconds=10,
                )
                nearby_files = (ls_result.stdout or "").strip()
                if nearby_files:
                    target_investigation += f"Files that DO exist in {parent_dir}:\n{nearby_files}\n"
            except Exception:
                pass

    # Step 4: Hand everything to the LLM for investigation and fix
    _emit("  [INVESTIGATE] Sending to LLM for analysis and fix...")

    investigation_prompt = (
        f"BOOT ERROR: {error_summary}\n\n"
        f"FULL STDERR (last 2000 chars):\n{full_stderr[-2000:]}\n\n"
        f"INVESTIGATION RESULTS:\n{target_investigation}\n"
        f"IMPORTING FILE ({importing_file_abs}):\n"
        f"```python\n{importing_content[:6000]}\n```\n\n"
        f"YOUR TASK:\n"
        f"1. Understand WHY this boot error is happening\n"
        f"2. Determine the minimal fix needed\n"
        f"3. Output the COMPLETE fixed file content\n\n"
        f"Common causes:\n"
        f"- Import references a module that was renamed or moved\n"
        f"- Import is for a module that doesn't exist (hallucinated by code generator)\n"
        f"- Import is wrapped in try/except but the except handler also fails\n"
        f"- Import path is wrong (e.g. 'app.logger_setup' should be 'app.logging_config')\n\n"
        f"Output ONLY the complete fixed file. No explanations, no markdown fences."
    )

    try:
        from app.providers.registry import get_provider_registry
        registry = get_provider_registry()

        provider_id = _pick_boot_fix_provider()
        model_id = _pick_boot_fix_model()
        llm_result = await registry.llm_call(
            provider_id=provider_id,
            model_id=model_id,
            messages=[
                {"role": "system", "content": _BOOT_FIX_SYSTEM_PROMPT},
                {"role": "user", "content": investigation_prompt},
            ],
            max_tokens=_pick_boot_fix_max_tokens(),
            timeout_seconds=BOOT_FIX_TIMEOUT + 60,  # Extra time for investigation
        )

        fixed_content = _extract_fix_content(llm_result)
        if not fixed_content or len(fixed_content) < 20:
            _emit("  [INVESTIGATE] LLM investigation produced empty/minimal content")
            # Return the file path so _attempt_boot_fix can try its own approach
            rel_path = importing_file_abs.replace(actual_base + "\\", "").replace(actual_base + "/", "")
            return (rel_path, None)

        # Sanity check: don't let LLM destroy the file
        if len(fixed_content) < len(importing_content) * 0.5:
            _emit("  [INVESTIGATE] LLM fix removed too much content (>50% reduction) -- rejecting")
            rel_path = importing_file_abs.replace(actual_base + "\\", "").replace(actual_base + "/", "")
            return (rel_path, None)

        # Write the fix to the sandbox
        success = _write_file_to_sandbox_abs(client, importing_file_abs, fixed_content)
        if success:
            rel_path = importing_file_abs.replace(actual_base + "\\", "").replace(actual_base + "/", "")
            _emit(f"  [INVESTIGATE] Fix written to {rel_path}")
            return (rel_path, f"Investigation fix: {error_summary[:80]}")
        else:
            _emit("  [INVESTIGATE] Failed to write fix to sandbox")
            return None

    except Exception as exc:
        _emit(f"  [INVESTIGATE] LLM investigation failed: {exc}")
        logger.warning("[phase_checkout] Investigation LLM call failed: %s", exc)
        rel_path = importing_file_abs.replace(actual_base + "\\", "").replace(actual_base + "/", "")
        return (rel_path, None)


def _write_file_to_sandbox_abs(
    client: Any,
    abs_path: str,
    content: str,
) -> bool:
    """Write a file to sandbox using absolute path (for investigation fixes)."""
    import base64
    try:
        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        tmp_path = abs_path.rsplit("\\", 1)[0] + "\\_orb_phase_fix.b64"
        # Write base64
        chunk_size = 60000
        for i in range(0, len(b64), chunk_size):
            chunk = b64[i:i + chunk_size]
            op = "-NoNewline" if i == 0 else "-Append -NoNewline"
            client.shell_run(
                f'Set-Content -Path "{tmp_path}" -Value "{chunk}" {op} -Encoding ASCII',
                timeout_seconds=15,
            )
        # Decode and write
        client.shell_run(
            f'$b64 = [System.IO.File]::ReadAllText("{tmp_path}"); '
            f'$bytes = [System.Convert]::FromBase64String($b64); '
            f'[System.IO.File]::WriteAllBytes("{abs_path}", $bytes); '
            f'Remove-Item -Path "{tmp_path}" -Force -ErrorAction SilentlyContinue; '
            f'"WRITE_OK"',
            timeout_seconds=15,
        )
        return True
    except Exception as exc:
        logger.warning("[phase_checkout] Write to sandbox failed: %s", exc)
        return False


# =============================================================================
# SILENT IMPORT FAILURE DETECTION (v2.3)
# =============================================================================

# Import patterns that indicate real failures (not just warnings)
_SILENT_IMPORT_PATTERNS = [
    re.compile(r"No module named '([^']+)'"),
    re.compile(r"cannot import name '([^']+)' from '([^']+)'"),
    re.compile(r"ImportError: ([^\n]+)"),
    re.compile(r"ModuleNotFoundError: ([^\n]+)"),
]

# Known pre-existing import failures to ignore (not from our segments)
_KNOWN_PREEXISTING_FAILURES = {
    "numpy",
    "scipy",
    "pandas",
    "cv2",
    "PIL",
    "torch",
    "tensorflow",
}


def _check_stderr_for_silent_import_failures(
    stderr: str,
    stdout: str = "",
) -> Optional[str]:
    """
    v2.7: Scan boot output for import failures that were silently caught.

    The boot test passed (BOOT_CHECK_PASS in stdout) but output may contain
    warnings about modules that failed to import. These are real bugs —
    the module loaded but some functionality is broken.

    v2.7 changes:
    - Also scans stdout (print-based warnings, not just logging stderr)
    - Fixes regex group mismatch: for "cannot import name X from Y",
      now correctly checks Y (the module) not X (the name) for app.* prefix
    - Checks both group(1) and group(2) where available

    Only flags import failures for project modules (app.*). Ignores
    known pre-existing third-party failures (numpy, scipy, etc.).

    Returns an error summary string if a silent failure is found, or None.
    """
    # Scan both stderr and stdout — import warnings can appear in either
    combined = f"{stderr or ''}\n{stdout or ''}"
    if not combined.strip():
        return None

    for pattern in _SILENT_IMPORT_PATTERNS:
        for match in pattern.finditer(combined):
            full_match = match.group(0)

            # Collect all captured groups as potential module references
            groups = [g for g in match.groups() if g]

            # Check if ANY captured group references a project module
            is_project_module = False
            for g in groups:
                g_root = g.split(".")[0]
                if g_root in _KNOWN_PREEXISTING_FAILURES:
                    break  # Known third-party — skip entire match
                if g.startswith("app.") or g.startswith("src."):
                    is_project_module = True
            else:
                # Loop completed without break (not a known pre-existing failure)
                if is_project_module:
                    return full_match

                # Also flag if output explicitly mentions architecture_executor
                if "architecture_executor" in full_match.lower():
                    return full_match

    return None


def _try_reconciliation_import_fix(
    client: Any,
    actual_base: str,
    failing_file: str,
    broken_content: str,
    error_summary: str,
    state: Any,
    emit: Any,
) -> Optional[Tuple[str, str]]:
    """
    v2.1: Use post-execution reconciliation to fix import errors smartly.

    Instead of commenting out a broken import, this reads the target module
    from the sandbox, extracts what it actually exports, and rewrites the
    import line with the correct name.

    e.g. "cannot import name 'collect_file_inventory' from 'init_files'"
    -> reads init_files.py, finds it exports '_ensure_python_init_files'
    -> rewrites the import line

    Returns (fixed_content, description) or None.
    """
    _emit = emit or (lambda msg: None)

    try:
        from app.orchestrator.post_execution_reconciliation import (
            _build_export_registry,
            detect_import_mismatches,
            apply_import_fixes,
        )
    except ImportError:
        return None

    # Parse the error to find what module was being imported from
    cannot_import = re.search(r"cannot import name '([^']+)' from '([^']+)'", error_summary)
    if not cannot_import:
        return None

    bad_name = cannot_import.group(1)
    source_module = cannot_import.group(2)

    # Find the source module's file on the sandbox
    # The module path might be dotted: app.overwatcher.architecture_executor.init_files
    module_stem = source_module.rsplit(".", 1)[-1]  # e.g. "init_files"

    # Collect all segment output files to build registry
    all_contents: Dict[str, str] = {}
    for seg_id, seg_state in state.segments.items():
        for rel_path in (seg_state.output_files or []):
            if not rel_path.endswith(".py"):
                continue
            content = _read_file_via_sandbox(client, rel_path, actual_base)
            if content:
                all_contents[rel_path] = content

    if not all_contents:
        return None

    # Build registry and detect mismatches
    registry = _build_export_registry(all_contents)
    fixes = detect_import_mismatches(
        file_path=failing_file,
        file_content=broken_content,
        export_registry=registry,
    )

    # Filter to fixes for the specific error we're trying to fix
    relevant_fixes = [f for f in fixes if f.wrong_name == bad_name]

    if not relevant_fixes:
        _emit(f"    [RECON] No match found for '{bad_name}' in module exports")
        return None

    # Apply the fix(es)
    fixed_content = apply_import_fixes(broken_content, relevant_fixes)
    if fixed_content == broken_content:
        return None

    fix_desc = "; ".join(
        f"'{f.wrong_name}'->'{f.correct_name}' ({f.fix_method})"
        for f in relevant_fixes
    )
    _emit(f"    [RECON] Smart fix: {fix_desc}")
    return (fixed_content, f"Reconciliation: {fix_desc}")


def _try_deterministic_import_fix(
    content: str,
    error_summary: str,
    full_stderr: str,
) -> Optional[Tuple[str, str]]:
    """
    Try to fix import errors without LLM -- pure string manipulation.

    Handles:
    - "No module named 'app.models'" -> comment out or remove the import line
    - "cannot import name 'X' from 'Y'" -> comment out the import line

    Returns (fixed_content, description) or None.
    """
    # Extract the problematic module/name from the error
    no_module_match = re.search(r"No module named '([^']+)'", error_summary)
    cannot_import_match = re.search(r"cannot import name '([^']+)' from '([^']+)'", error_summary)

    if no_module_match:
        bad_module = no_module_match.group(1)
        # Find and comment out import lines referencing this module
        lines = content.split("\n")
        fixed_lines = []
        changes = 0
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith("from ") or stripped.startswith("import ")) and bad_module in stripped:
                fixed_lines.append(f"# PHASE_CHECKOUT_FIX: removed bad import: {line.strip()}")
                changes += 1
            else:
                fixed_lines.append(line)

        if changes > 0:
            return ("\n".join(fixed_lines), f"Commented out {changes} import(s) of '{bad_module}'")

    if cannot_import_match:
        bad_name = cannot_import_match.group(1)
        bad_source = cannot_import_match.group(2)
        lines = content.split("\n")
        fixed_lines = []
        changes = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("from ") and bad_source in stripped and bad_name in stripped:
                fixed_lines.append(f"# PHASE_CHECKOUT_FIX: removed bad import: {line.strip()}")
                changes += 1
            else:
                fixed_lines.append(line)

        if changes > 0:
            return ("\n".join(fixed_lines), f"Commented out import of '{bad_name}' from '{bad_source}'")

    return None


# =============================================================================
# BOOT FIX PROMPTS
# =============================================================================

def _pick_boot_fix_provider() -> str:
    """Pick the LLM provider for boot fix from env."""
    return os.environ.get("PHASE_CHECKOUT_PROVIDER", "anthropic")


def _pick_boot_fix_model() -> str:
    """Pick the LLM model for boot fix from env."""
    return os.environ.get("PHASE_CHECKOUT_MODEL", "claude-opus-4-6")


def _pick_boot_fix_max_tokens() -> int:
    """Pick the max output tokens for boot fix from env."""
    return int(os.environ.get("PHASE_CHECKOUT_MAX_OUTPUT_TOKENS", "8000"))


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


def _build_fix_prompt(
    failing_file: str,
    broken_content: str,
    error_summary: str,
    full_stderr: str,
    error_type: str,
) -> str:
    """Build the LLM prompt for a targeted boot fix."""
    # Truncate content/stderr to fit context
    max_content = 15000
    max_stderr = 3000
    content_display = broken_content[:max_content]
    if len(broken_content) > max_content:
        content_display += f"\n\n... [truncated, {len(broken_content)} chars total]"
    stderr_display = full_stderr[:max_stderr]

    # v3.0: Check experience memory for matching boot fix patterns
    _memory_section = ""
    try:
        from app.experience.retrieval import retrieve_for_stage, format_injection
        from app.db import get_db_session
        from app.orchestrator.strike_tracker import _error_signature
        _mem_db = get_db_session()
        _boot_patterns = retrieve_for_stage(
            _mem_db, stage="phase_checkout",
            context=f"Boot fix for {failing_file}: {error_summary[:100]}",
            error_signature=_error_signature(error_summary),
            max_results=3,
        )
        if _boot_patterns:
            _memory_section = "\n\n" + format_injection(_boot_patterns, stage="phase_checkout")
        _mem_db.close()
    except Exception:
        pass

    return (
        f"## BOOT FIX REQUIRED\n\n"
        f"**Error type:** {error_type}\n"
        f"**Failing file:** {failing_file}\n\n"
        f"### Error Summary\n```\n{error_summary}\n```\n\n"
        f"### Full Traceback\n```\n{stderr_display}\n```\n\n"
        f"### Current File Content (broken)\n```python\n{content_display}\n```\n\n"
        f"{_memory_section}\n\n"
        f"Output the complete fixed file now. Nothing else."
    )


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


# =============================================================================
# SANDBOX HELPERS
# =============================================================================

def _discover_sandbox_base(client: Any, default_base: str) -> str:
    """Find the actual repo base path inside the sandbox."""
    for candidate in [r"C:\Orb\Orb", r"C:\Orb", r"D:\Orb"]:
        try:
            test = client.shell_run(
                f'Test-Path -Path "{candidate}\\main.py"',
                timeout_seconds=10,
            )
            if (test.stdout or "").strip().lower() == "true":
                return candidate
        except Exception:
            continue
    return default_base


def _read_file_via_sandbox(
    client: Any,
    rel_path: str,
    sandbox_base: str,
) -> Optional[str]:
    """Read a file from the sandbox filesystem."""
    # Normalise path separators
    normed = rel_path.replace("/", "\\")
    # Build absolute path if not already
    if not (normed.startswith("C:") or normed.startswith("D:")):
        abs_path = f"{sandbox_base}\\{normed}"
    else:
        abs_path = normed

    try:
        result = client.shell_run(
            f'Get-Content -Path "{abs_path}" -Raw -Encoding UTF8',
            timeout_seconds=15,
        )
        content = result.stdout
        if content is not None and len(content.strip()) > 0:
            return content
    except Exception as exc:
        logger.debug("[phase_checkout] Cannot read %s from sandbox: %s", abs_path, exc)

    return None


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


def _file_exists_in_sandbox(
    client: Any,
    rel_path: str,
    sandbox_base: str,
) -> bool:
    """Check if a file exists in the sandbox."""
    normed = rel_path.replace("/", "\\")
    if not (normed.startswith("C:") or normed.startswith("D:")):
        abs_path = f"{sandbox_base}\\{normed}"
    else:
        abs_path = normed

    try:
        result = client.shell_run(
            f'Test-Path -Path "{abs_path}" -PathType Leaf',
            timeout_seconds=10,
        )
        return (result.stdout or "").strip().lower() == "true"
    except Exception:
        return False


# =============================================================================
# PARSING HELPERS
# =============================================================================

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


def _resolve_output_path(rel_path: str, sandbox_base: str) -> Optional[str]:
    """Resolve a relative file path to absolute using sandbox base."""
    normalised = rel_path.replace("/", os.sep).replace("\\", os.sep)
    return os.path.join(sandbox_base, normalised)


def _norm(path: str) -> str:
    """Normalise path for comparison."""
    return path.replace("\\", "/").lower().strip("/")


def map_file_to_segment(
    file_path: Optional[str],
    state: Any,
) -> Optional[str]:
    """Map a failing file path to the segment that produced it."""
    if not file_path:
        return None
    target = _norm(file_path)
    for seg_id, seg_state in state.segments.items():
        for out_file in (seg_state.output_files or []):
            if _norm(out_file) == target:
                return seg_id
    return None
