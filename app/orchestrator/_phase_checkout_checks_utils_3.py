from __future__ import annotations
import logging
import os
import re
from .phase_checkout_models import ContractCheckResult, ContractViolation, SizeValidationResult, SizeViolation
from app.orchestrator._phase_checkout_checks_utils_2 import _KNOWN_PREEXISTING_FAILURES, _SILENT_IMPORT_PATTERNS, _file_exists_in_sandbox, _find_largest_function
from app.pot_spec.grounded.size_models import MAX_FILE_KB, MAX_FILE_LINES, MAX_FUNCTION_LINES
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


BOOT_MAX_FIX_ATTEMPTS = int(os.environ.get("PHASE_CHECKOUT_MAX_FIX_ATTEMPTS", "12"))

BOOT_STRIKE_LIMIT = int(os.environ.get("PHASE_CHECKOUT_STRIKE_LIMIT", "3"))

def check_output_file_sizes(
    state: Any,
    sandbox_base: str,
    baseline_function_sizes: Optional[Dict[str, int]] = None,
) -> SizeValidationResult:
    """
    Scan segment-produced output files for size constraint violations.

    v2.0: Only checks files that segments actually produced (output_files).
    Pre-existing source files that are being replaced/decomposed are NOT checked.

    v6.1 FIX 3: baseline_function_sizes is a dict of {function_name: line_count}
    from the source monolith scan. If a transplanted function is <= its baseline
    size, it's a pre-existing violation, not a new one. Skip it.

    Checks:
    - File line count <= MAX_FILE_LINES (400)
    - File size <= MAX_FILE_KB (15 KB)
    - Largest function body <= MAX_FUNCTION_LINES (200)
    """
    from .phase_checkout_checks import _read_file_via_sandbox, _resolve_output_path
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
                    # v6.1 FIX 3: Check if this is a pre-existing oversized function
                    _is_baseline = False
                    if baseline_function_sizes and max_fn_name:
                        _baseline_size = baseline_function_sizes.get(max_fn_name, 0)
                        if _baseline_size > 0 and max_fn_lines <= _baseline_size:
                            _is_baseline = True
                            logger.info(
                                "[phase_checkout] v6.1 Skipping pre-existing size violation: "
                                "%s (%d lines, baseline %d)",
                                max_fn_name, max_fn_lines, _baseline_size,
                            )

                    if not _is_baseline:
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
    from .phase_checkout_checks import _resolve_output_path
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
        # v3.0: Strip absolute path prefix before comparison. Output files have
        # absolute Windows paths (D:\Orb\app\...) while file_scope has relative
        # paths (app/...). Without stripping, every file is a false positive.
        seg_state = state.segments.get(seg_id)
        if seg_state and seg_state.output_files:
            scope_set = {_norm(f) for f in skel.file_scope}
            _base_prefix = _norm(sandbox_base).rstrip("/") + "/"
            for out_file in seg_state.output_files:
                normed = _norm(out_file)
                # v3.0: Strip sandbox base prefix to get relative path
                if normed.startswith(_base_prefix):
                    normed = normed[len(_base_prefix):]
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
    from .phase_checkout_checks import _read_file_via_sandbox
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

def _norm(path: str) -> str:
    """Normalise path for comparison."""
    return path.replace("\\", "/").lower().strip("/")
