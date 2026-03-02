# FILE: app/orchestrator/frontend_fix_loop.py
"""
Frontend Build Fix Loop — Tiered error resolution.

v1.0 (2026-03-01): Three-tier fix approach:
  Tier 1: Deterministic auto-fix (duplicate imports, unused imports). Free, instant.
  Tier 2: Targeted LLM fix (Sonnet-level, single file + exact error). Cheap.
  Tier 3: Opus escalation (multi-file issues). Expensive, rare.

Only Tier 1 is implemented in v1.0. Tier 2/3 are stubbed for future work.
The loop runs: check → fix → recheck → done (or escalate).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .frontend_build_check import (
    FrontendBuildResult,
    TscError,
    filter_errors_by_segment,
    parse_tsc_errors,
    run_frontend_build_check,
)
from .deterministic_import_fixer import apply_deterministic_fixes

logger = logging.getLogger(__name__)

FRONTEND_FIX_LOOP_BUILD_ID = "2026-03-01-v1.0-tiered-fix"
print(f"[FRONTEND_FIX_LOOP_LOADED] BUILD_ID={FRONTEND_FIX_LOOP_BUILD_ID}")

# Max deterministic fix attempts before giving up
MAX_DETERMINISTIC_ROUNDS = 3


@dataclass
class FrontendFixResult:
    """Result of the full frontend fix loop."""
    status: str                    # "pass" | "fail" | "fixed" | "error"
    initial_errors: int = 0
    remaining_errors: int = 0
    fixes_applied: List[str] = field(default_factory=list)
    duration_ms: int = 0
    tsc_result: Optional[FrontendBuildResult] = None


async def run_frontend_fix_loop(
    segment_files: List[str],
    frontend_base: str = r"D:\orb-desktop",
    emit: Optional[Any] = None,
) -> FrontendFixResult:
    """Run the frontend build check with tiered auto-fix.

    1. Run tsc --noEmit
    2. If errors, filter to segment files only
    3. Apply Tier 1 deterministic fixes (import merging, unused removal)
    4. Write fixed files back to sandbox
    5. Re-run tsc to verify
    6. If still failing, report remaining errors (Tier 2/3 not yet implemented)

    Args:
        segment_files: List of file paths this segment produced.
        frontend_base: Path to frontend repo in sandbox.
        emit: Optional SSE callback.

    Returns:
        FrontendFixResult with status and details.
    """
    _emit = emit or (lambda msg: None)
    start = time.time()

    # Connect to sandbox
    try:
        from app.overwatcher.sandbox_client import get_sandbox_client
        client = get_sandbox_client()
    except Exception as exc:
        return FrontendFixResult(
            status="error",
            duration_ms=int((time.time() - start) * 1000),
        )

    # Normalise segment file paths to include frontend base
    norm_files = []
    for f in segment_files:
        if not f.startswith(("D:", "C:", "/")):
            norm_files.append(f"src/{f}" if not f.startswith("src") else f)
        else:
            norm_files.append(f)

    # --- Initial build check ---
    _emit("[FRONTEND] Running TypeScript compilation check...")
    check = run_frontend_build_check(client, frontend_base, emit=_emit)

    if check.status == "pass":
        return FrontendFixResult(
            status="pass",
            duration_ms=int((time.time() - start) * 1000),
            tsc_result=check,
        )

    if check.status == "error":
        return FrontendFixResult(
            status="error",
            duration_ms=int((time.time() - start) * 1000),
            tsc_result=check,
        )

    # Filter to segment-owned errors only
    seg_errors = filter_errors_by_segment(check.errors, norm_files)
    if not seg_errors:
        # All errors are in pre-existing files, not ours
        _emit(
            f"  [TSC] {len(check.errors)} error(s) found but none in segment files "
            f"— treating as pre-existing"
        )
        return FrontendFixResult(
            status="pass",
            initial_errors=len(check.errors),
            duration_ms=int((time.time() - start) * 1000),
            tsc_result=check,
        )

    initial_error_count = len(seg_errors)
    _emit(
        f"  [TSC] {initial_error_count} error(s) in segment files — "
        f"attempting Tier 1 deterministic fix..."
    )

    all_fixes: List[str] = []

    # --- Tier 1: Deterministic fix loop ---
    for round_num in range(1, MAX_DETERMINISTIC_ROUNDS + 1):
        # Group errors by file
        errors_by_file: Dict[str, List[TscError]] = {}
        for err in seg_errors:
            errors_by_file.setdefault(err.file, []).append(err)

        round_fixes: List[str] = []

        for rel_path, file_errors in errors_by_file.items():
            # Read the file from sandbox
            abs_path = os.path.join(frontend_base, rel_path).replace("/", "\\")
            try:
                read_result = client._request(
                    "POST", "/fs/contents",
                    json_body={
                        "paths": [abs_path],
                        "max_file_size": 50000,
                    },
                )
                files = read_result.get("files", [])
                if not files or "error" in files[0]:
                    _emit(f"    [SKIP] Cannot read {rel_path}")
                    continue
                content = files[0]["content"]
            except Exception as exc:
                _emit(f"    [SKIP] Read failed for {rel_path}: {exc}")
                continue

            # Apply deterministic fixes
            fixed_content, fixes = apply_deterministic_fixes(content, file_errors)

            if not fixes:
                continue

            # Write fixed file back to sandbox
            try:
                client.write_file(abs_path, fixed_content)
                round_fixes.extend(fixes)
                _emit(f"    [FIX] {rel_path}: {'; '.join(fixes)}")
                logger.info(
                    "[frontend_fix] Tier 1 fixed %s: %s",
                    rel_path, "; ".join(fixes),
                )
            except Exception as exc:
                _emit(f"    [FAIL] Write failed for {rel_path}: {exc}")
                logger.error("[frontend_fix] Write failed for %s: %s", rel_path, exc)

        if not round_fixes:
            # No more deterministic fixes possible
            _emit(
                f"  [TSC] Tier 1 exhausted after {round_num} round(s) — "
                f"no further deterministic fixes available"
            )
            break

        all_fixes.extend(round_fixes)

        # Re-run tsc to check if fixes resolved the errors
        _emit(f"  [TSC] Re-checking after Tier 1 round {round_num}...")
        recheck = run_frontend_build_check(client, frontend_base, emit=_emit)

        if recheck.status == "pass":
            elapsed = int((time.time() - start) * 1000)
            _emit(
                f"  [TSC] ✅ FIXED — {initial_error_count} error(s) resolved "
                f"deterministically in {round_num} round(s)"
            )
            return FrontendFixResult(
                status="fixed",
                initial_errors=initial_error_count,
                remaining_errors=0,
                fixes_applied=all_fixes,
                duration_ms=elapsed,
                tsc_result=recheck,
            )

        # Still failing — update seg_errors for next round
        seg_errors = filter_errors_by_segment(recheck.errors, norm_files)
        if not seg_errors:
            elapsed = int((time.time() - start) * 1000)
            _emit("  [TSC] ✅ Segment errors resolved (remaining are pre-existing)")
            return FrontendFixResult(
                status="fixed",
                initial_errors=initial_error_count,
                remaining_errors=len(recheck.errors),
                fixes_applied=all_fixes,
                duration_ms=elapsed,
                tsc_result=recheck,
            )

    # --- Tier 1 exhausted, errors remain ---
    elapsed = int((time.time() - start) * 1000)
    remaining = len(seg_errors)
    _emit(
        f"  [TSC] ⚠️ {remaining} error(s) remain after Tier 1 fixes. "
        f"Tier 2 (LLM fix) not yet implemented — reporting as failure."
    )

    # Journal for learning
    try:
        from app.experience.context import journal_emit
        journal_emit(
            stage="frontend_build_check",
            event_type="tier1_exhausted",
            severity="warning",
            description=(
                f"Tier 1 deterministic fix resolved {initial_error_count - remaining}/"
                f"{initial_error_count} errors. {remaining} remain."
            ),
            details={
                "initial_errors": initial_error_count,
                "remaining_errors": remaining,
                "fixes_applied": all_fixes,
                "remaining_codes": [e.code for e in seg_errors],
            },
        )
    except Exception:
        pass

    return FrontendFixResult(
        status="fail",
        initial_errors=initial_error_count,
        remaining_errors=remaining,
        fixes_applied=all_fixes,
        duration_ms=elapsed,
        tsc_result=check,
    )
