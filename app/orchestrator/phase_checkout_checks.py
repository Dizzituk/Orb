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

BOOT_MAX_FIX_ATTEMPTS = 3
BOOT_FIX_TIMEOUT = 120  # seconds per LLM fix call


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

    v2.0: When boot fails, the system:
    1. Parses the traceback to identify the failing file and error type
    2. Reads the broken file from the sandbox
    3. Sends the file + error to the LLM for a targeted surgical fix
    4. Writes the fix back to the sandbox
    5. Retries boot (up to BOOT_MAX_FIX_ATTEMPTS times)

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

    last_error = None
    same_error_count = 0
    fixes_applied = []

    for attempt in range(1, BOOT_MAX_FIX_ATTEMPTS + 1):
        _emit(f"  Boot attempt {attempt}/{BOOT_MAX_FIX_ATTEMPTS}...")

        passed, stdout, stderr, error_summary, failing_file = _run_single_boot(
            client, actual_base
        )

        if passed:
            elapsed = int((time.time() - start) * 1000)
            if attempt > 1:
                _emit(f"  [OK] Boot PASSED after {attempt - 1} fix(es): {fixes_applied}")
            return BootTestResult(
                status="pass", stdout=stdout, stderr=stderr,
                duration_ms=elapsed,
            )

        # Boot failed
        _emit(f"  [FAIL] Boot attempt {attempt} failed: {error_summary[:150]}")

        # Track repeated errors -- if same error 2x, the fix didn't work
        if error_summary == last_error:
            same_error_count += 1
        else:
            same_error_count = 1
            last_error = error_summary

        if same_error_count >= 2:
            _emit(f"  [ABORT] Same error repeated {same_error_count} times -- fix not working")
            break

        # Last attempt -- don't try to fix, just report
        if attempt >= BOOT_MAX_FIX_ATTEMPTS:
            _emit(f"  [ABORT] Max fix attempts ({BOOT_MAX_FIX_ATTEMPTS}) reached")
            break

        # --- Attempt to fix the error ---
        if not failing_file:
            _emit("  [ABORT] Cannot identify failing file from traceback -- cannot auto-fix")
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
        else:
            _emit("  [ABORT] Could not generate a fix")
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
    cmd = (
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
            max_tokens=8000,
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
    """Pick the LLM provider for boot fix. Env override or default."""
    return os.environ.get("ASTRA_BOOT_FIX_PROVIDER", "anthropic")


def _pick_boot_fix_model() -> str:
    """Pick the LLM model for boot fix. Env override or default.
    Uses Claude Opus as default -- phase checkout needs a strong model
    that can read tracebacks and surgically fix code.
    """
    return os.environ.get("ASTRA_BOOT_FIX_MODEL", "claude-opus-4-6")


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

    return (
        f"## BOOT FIX REQUIRED\n\n"
        f"**Error type:** {error_type}\n"
        f"**Failing file:** {failing_file}\n\n"
        f"### Error Summary\n```\n{error_summary}\n```\n\n"
        f"### Full Traceback\n```\n{stderr_display}\n```\n\n"
        f"### Current File Content (broken)\n```python\n{content_display}\n```\n\n"
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
