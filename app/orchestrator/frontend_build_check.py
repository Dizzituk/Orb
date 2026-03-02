# FILE: app/orchestrator/frontend_build_check.py
"""
Deterministic Frontend Build Check.

v1.0 (2026-03-01): Runs `npx tsc --noEmit` in the sandbox to catch
TypeScript/TSX compilation errors BEFORE any LLM-based review.

This is a hard gate — if TypeScript doesn't compile, the code is broken.
No LLM opinion needed. The compiler output gives exact file, line, column,
and error message for every issue.

Replaces the heuristic-only `check_frontend_syntax` for actual compilation
validation. The heuristic check (SSE contamination detection) still runs
separately as it catches a different class of error.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# v1.1: Baseline filter for pre-existing tsc errors
try:
    from .tsc_error_baseline import filter_baseline_errors
    _HAS_BASELINE = True
except ImportError:
    _HAS_BASELINE = False
    filter_baseline_errors = None

FRONTEND_BUILD_CHECK_BUILD_ID = "2026-03-01-v1.0-deterministic-tsc"
print(f"[FRONTEND_BUILD_CHECK_LOADED] BUILD_ID={FRONTEND_BUILD_CHECK_BUILD_ID}")

# Regex to parse tsc error output lines like:
# src/components/education/EducationCourseGrid.tsx(4,10): error TS2300: Duplicate identifier 'courses'.
_TSC_ERROR_RE = re.compile(
    r"^(.+?)\((\d+),(\d+)\):\s+error\s+(TS\d+):\s+(.+)$"
)


@dataclass
class TscError:
    """Single TypeScript compiler error."""
    file: str
    line: int
    column: int
    code: str        # e.g. "TS2300"
    message: str     # e.g. "Duplicate identifier 'courses'."

    @property
    def signature(self) -> str:
        """Stable signature for deduplication and strike tracking."""
        return f"{self.file}:{self.line}:{self.code}"

    def __str__(self) -> str:
        return f"{self.file}({self.line},{self.column}): {self.code}: {self.message}"


@dataclass
class FrontendBuildResult:
    """Result of a frontend build check."""
    status: str                        # "pass" | "fail" | "error" | "skipped"
    errors: List[TscError] = field(default_factory=list)
    error_count: int = 0
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    error_summary: str = ""


def parse_tsc_errors(output: str) -> List[TscError]:
    """Parse TypeScript compiler output into structured errors.

    Args:
        output: Raw stdout/stderr from `npx tsc --noEmit`.

    Returns:
        List of TscError objects, one per error line.
    """
    errors: List[TscError] = []
    for line in output.splitlines():
        line = line.strip()
        m = _TSC_ERROR_RE.match(line)
        if m:
            errors.append(TscError(
                file=m.group(1).replace("\\", "/"),
                line=int(m.group(2)),
                column=int(m.group(3)),
                code=m.group(4),
                message=m.group(5).strip(),
            ))
    return errors


def run_frontend_build_check(
    client: Any,
    frontend_base: str = r"D:\orb-desktop",
    timeout_seconds: int = 60,
    emit: Optional[Any] = None,
) -> FrontendBuildResult:
    """Run `npx tsc --noEmit` in the sandbox and parse results.

    This is a pure deterministic check — no LLM involvement.
    The TypeScript compiler is the single source of truth.

    Args:
        client: SandboxClient instance.
        frontend_base: Path to the frontend repo in the sandbox.
        timeout_seconds: Max time for tsc to run.
        emit: Optional SSE callback for progress messages.

    Returns:
        FrontendBuildResult with status, parsed errors, and raw output.
    """
    _emit = emit or (lambda msg: None)
    import time
    start = time.time()

    # Run tsc --noEmit from the frontend directory
    cmd = (
        f'cd "{frontend_base}" ; '
        f'npx tsc --noEmit --pretty false 2>&1'
    )

    _emit("  [TSC] Running TypeScript compiler check...")
    logger.info("[frontend_build] Running tsc --noEmit in %s", frontend_base)

    try:
        result = client.shell_run(
            cmd,
            cwd_target="REPO",
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        elapsed = int((time.time() - start) * 1000)
        logger.error("[frontend_build] Shell execution failed: %s", exc)
        return FrontendBuildResult(
            status="error",
            error_summary=f"Failed to run tsc: {exc}",
            duration_ms=elapsed,
        )

    elapsed = int((time.time() - start) * 1000)
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    combined = f"{stdout}\n{stderr}".strip()

    # Parse errors from output
    errors = parse_tsc_errors(combined)

    # v1.1: Filter out known pre-existing baseline errors before pass/fail decision.
    # Sandbox-specific errors (missing numpy, whisper, etc.) should not trigger failures.
    baseline_count = 0
    if errors and _HAS_BASELINE and filter_baseline_errors:
        errors, baseline_errs = filter_baseline_errors(errors)
        baseline_count = len(baseline_errs)
        if baseline_count:
            logger.info(
                "[frontend_build] Filtered %d baseline error(s) (pre-existing)",
                baseline_count,
            )

    if result.exit_code == 0 and not errors:
        _emit(f"  [TSC] ✅ TypeScript compilation PASSED ({elapsed}ms)")
        logger.info("[frontend_build] tsc PASSED in %dms", elapsed)
        return FrontendBuildResult(
            status="pass",
            stdout=stdout,
            stderr=stderr,
            duration_ms=elapsed,
        )

    # v1.1: If all errors were baseline, treat as pass
    if not errors and baseline_count > 0:
        _emit(f"  [TSC] ✅ TypeScript compilation PASSED ({baseline_count} pre-existing error(s) filtered)")
        logger.info("[frontend_build] tsc PASSED in %dms (%d baseline errors filtered)", elapsed, baseline_count)
        return FrontendBuildResult(
            status="pass",
            stdout=stdout,
            stderr=stderr,
            duration_ms=elapsed,
        )

    # Compilation failed
    _emit(f"  [TSC] ❌ TypeScript compilation FAILED: {len(errors)} error(s)")
    for err in errors[:10]:  # Show first 10
        _emit(f"    {err}")

    error_summary = "; ".join(str(e) for e in errors[:5])
    logger.warning(
        "[frontend_build] tsc FAILED in %dms: %d errors. First: %s",
        elapsed, len(errors), errors[0] if errors else "unknown",
    )

    return FrontendBuildResult(
        status="fail",
        errors=errors,
        error_count=len(errors),
        stdout=stdout,
        stderr=stderr,
        duration_ms=elapsed,
        error_summary=error_summary,
    )


def filter_errors_by_segment(
    errors: List[TscError],
    segment_files: List[str],
) -> List[TscError]:
    """Filter tsc errors to only those in segment-produced files.

    Pre-existing errors in files the segment didn't touch should not
    block the segment. Only errors in files this segment created or
    modified are actionable.

    Args:
        errors: All tsc errors from the build.
        segment_files: List of file paths this segment owns.

    Returns:
        Filtered list of errors only in segment files.
    """
    # Normalise paths for comparison
    norm_seg = {f.replace("\\", "/").lower() for f in segment_files}

    return [
        e for e in errors
        if e.file.lower() in norm_seg
        or any(e.file.lower().endswith(sf.lower()) for sf in norm_seg)
    ]
