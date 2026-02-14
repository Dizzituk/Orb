# FILE: app/orchestrator/phase_checkout_checks.py
"""
Phase Checkout — Individual Verification Checks.

Contains the implementation of each check run during Phase Checkout:
1. Output file size validation
2. Skeleton contract verification
3. Application boot test

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


# =============================================================================
# CHECK 1: OUTPUT FILE SIZE VALIDATION
# =============================================================================

def check_output_file_sizes(
    state: Any,
    sandbox_base: str,
) -> SizeValidationResult:
    """
    Scan all output files for size constraint violations.

    Checks:
    - File line count <= MAX_FILE_LINES (400)
    - File size <= MAX_FILE_KB (15 KB)
    - Largest function body <= MAX_FUNCTION_LINES (200)
    """
    violations: List[SizeViolation] = []
    files_checked = 0

    for seg_id, seg_state in state.segments.items():
        for rel_path in (seg_state.output_files or []):
            abs_path = _resolve_output_path(rel_path, sandbox_base)
            if not abs_path or not os.path.isfile(abs_path):
                continue

            files_checked += 1
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception:
                continue

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

    Checks export files exist and output files are within file_scope.
    """
    violations: List[ContractViolation] = []

    for skel in skeleton.skeletons:
        seg_id = skel.segment_id

        for export in skel.exports:
            abs_path = _resolve_output_path(export.file_path, sandbox_base)
            if not abs_path or not os.path.isfile(abs_path):
                violations.append(ContractViolation(
                    segment_id=seg_id,
                    violation_type="missing_export",
                    detail=f"Export '{export.file_path}' not found on disk",
                ))

        seg_state = state.segments.get(seg_id)
        if seg_state and seg_state.output_files:
            scope_set = {_norm(f) for f in skel.file_scope}
            for out_file in seg_state.output_files:
                if _norm(out_file) not in scope_set:
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
# CHECK 3: BOOT TEST
# =============================================================================

def run_boot_test(sandbox_base: str) -> BootTestResult:
    """
    Run application boot test via sandbox.

    Imports main.py and checks for BOOT_CHECK_PASS.
    On failure, parses traceback to identify the failing file.
    """
    start = time.time()
    try:
        from app.overwatcher.sandbox_client import get_sandbox_client
        client = get_sandbox_client()

        actual_base = sandbox_base
        for candidate in [r"C:\Orb\Orb", r"C:\Orb", r"D:\Orb"]:
            try:
                test = client.shell_run(
                    f'Test-Path -Path "{candidate}\\main.py"',
                    timeout_seconds=10,
                )
                if (test.stdout or "").strip().lower() == "true":
                    actual_base = candidate
                    break
            except Exception:
                continue

        venv = actual_base + r"\.venv\Scripts\python.exe"
        cmd = (
            f'cd "{actual_base}" ; '
            f'& "{venv}" -c '
            f'"import sys; sys.path.insert(0, r\'{actual_base}\'); '
            f'from main import app; print(\'BOOT_CHECK_PASS\')"'
        )
        shell_result = client.shell_run(cmd, timeout_seconds=30)
        stdout = (shell_result.stdout or "").strip()
        stderr = (shell_result.stderr or "").strip()
        elapsed = int((time.time() - start) * 1000)

        if "BOOT_CHECK_PASS" in stdout:
            return BootTestResult(
                status="pass", stdout=stdout, stderr=stderr,
                duration_ms=elapsed,
            )

        err_summary, failing_file = _parse_boot_failure(stdout, stderr)
        return BootTestResult(
            status="fail", stdout=stdout, stderr=stderr,
            error_summary=err_summary, traceback_file=failing_file,
            duration_ms=elapsed,
        )

    except Exception as exc:
        return BootTestResult(
            status="error",
            error_summary=str(exc)[:300],
            duration_ms=int((time.time() - start) * 1000),
        )


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

    file_match = re.search(r'File "([^"]+)"', combined)
    failing_file = None
    if file_match:
        path = file_match.group(1)
        for prefix in (r"D:\Orb\\", r"C:\Orb\\", r"C:\Orb\Orb\\"):
            if path.lower().startswith(prefix.lower()):
                failing_file = path[len(prefix):]
                break

    return (summary, failing_file)


# =============================================================================
# HELPERS
# =============================================================================

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
