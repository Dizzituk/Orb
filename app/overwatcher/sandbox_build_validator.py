"""Sandbox Build Validator: Post-POT build verification for sandbox projects.

Detects project types in the Windows Sandbox, runs the appropriate build
commands, parses error output, and returns structured results that the
Overwatcher can reason about for diagnostic/retry loops.

Supported project types:
    - vite_react: Vite + React + TypeScript (orb-desktop)
    - python_backend: Python/FastAPI backend (Orb)

v1.1 (2026-02-04): Path inference from actual file paths
    - Build commands now run in the directory where files were actually written
      (infers project root from POT file paths instead of hardcoded constants)
    - Fixes path mismatch: POT specs may use host paths (D:\\orb-desktop)
      while sandbox defaults are C:\\Orb\\orb-desktop
    - Inferred paths passed through fix execution chain
    - Diagnostic prompt no longer hardcodes C:\\Orb paths
    - Added BOM corruption hint to diagnostic system prompt
v1.0 (2026-02-03): Initial implementation
    - Project type detection from modified file paths
    - Build command execution via sandbox_client.shell_run()
    - Error output parsing (file paths, error types, structured summaries)
    - Multi-project validation (detects which projects were affected)
    - Diagnostic reasoning with LLM-powered fix generation
    - Bounded retry loop (max 3 attempts, configurable)

SAFETY INVARIANT:
    - All I/O goes through Windows Sandbox (sandbox_client)
    - NO direct host filesystem writes
    - If sandbox unavailable → FAIL (no local fallback)
    - Fix types constrained to known safe operations
    - No execution of arbitrary LLM-generated system commands
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.overwatcher.sandbox_client import (
    SandboxClient,
    SandboxError,
    ShellResult,
    get_sandbox_client,
)
from app.overwatcher._sandbox_build_validator_utils_2 import BUILD_VALIDATION_ENABLED, BUILD_VALIDATOR_BUILD_ID, DIAGNOSTIC_SYSTEM_PROMPT, DIAGNOSTIC_USER_PROMPT, ERROR_TYPE_PATTERNS, FILE_PATH_PATTERNS, MAX_DIAGNOSTIC_PROMPT_CHARS, PROJECT_UNKNOWN
from app.overwatcher._sandbox_build_validator_utils_3 import ALLOWED_FIX_COMMANDS, MAX_BUILD_FIX_ATTEMPTS, MAX_BUILD_OUTPUT_CHARS, _is_safe_command, _parse_diagnostic_response, _truncate_output, detect_project_from_path, parse_build_error_output
from app.overwatcher._sandbox_build_validator_utils_4 import BUILD_VALIDATION_TIMEOUT, BuildFixAction, DiagnosticResult, detect_affected_projects, detect_project_type_from_sandbox, diagnose_build_failure, execute_all_fixes, validate_all_affected_projects
from app.overwatcher._sandbox_build_validator_utils_5 import BuildValidationResult, PROJECT_PYTHON_BACKEND, PROJECT_VITE_REACT, _infer_project_path, execute_build_fix, validate_build

logger = logging.getLogger(__name__)

# Build verification
print(f"[BUILD_VALIDATOR_LOADED] BUILD_ID={BUILD_VALIDATOR_BUILD_ID}")


# =============================================================================
# Configuration (env-driven, following existing patterns)
# =============================================================================

# Truncation limits for LLM prompts (follows evidence.py cost guardrails)

# Sandbox project paths (matches sandbox/manager.py paths)
SANDBOX_FRONTEND_PATH = r"C:\Orb\orb-desktop"
SANDBOX_BACKEND_PATH = r"C:\Orb\Orb"

# Project type identifiers

# Allowed fix command prefixes (safety constraint — Section 11.5 of job spec)


# =============================================================================
# Result Data Classes
# =============================================================================


# =============================================================================
# Project Type Detection
# =============================================================================


# =============================================================================
# Error Parsing
# =============================================================================

# Patterns for extracting file paths from build error output

# Patterns for extracting error types


# =============================================================================
# Build Validation
# =============================================================================


# =============================================================================
# Diagnostic Reasoning (LLM-powered)
# =============================================================================


# =============================================================================
# Fix Execution
# =============================================================================


# =============================================================================
# Full Validation + Retry Loop (called from overwatcher_command.py)
# =============================================================================

async def run_build_validation_loop(
    *,
    client: SandboxClient,
    modified_files: List[str],
    spec_content: str,
    pot_result: Dict[str, Any],
    llm_call_fn: Callable,
    add_trace: Callable,
) -> Tuple[bool, List[BuildValidationResult], List[Dict[str, Any]]]:
    """Run the full build validation + diagnostic/retry loop.

    This is the main entry point called from overwatcher_command.py
    after POT execution succeeds.

    Args:
        client: SandboxClient instance
        modified_files: Files written by POT executor
        spec_content: Original spec content
        pot_result: POT execution result dict
        llm_call_fn: Async LLM call function
        add_trace: Trace logging callback

    Returns:
        Tuple of:
            - passed: bool (final result)
            - build_results: List of final BuildValidationResults
            - fix_history: List of fix attempt dicts (for evidence)
    """
    if not BUILD_VALIDATION_ENABLED:
        logger.info("[build_validator] Build validation disabled via env — skipping")
        add_trace("BUILD_VALIDATION", "disabled", {
            "reason": "OVERWATCHER_BUILD_VALIDATION_ENABLED=0",
        })
        return True, [], []

    # Initial build validation
    logger.info(
        "[build_validator] Starting build validation for %d modified files",
        len(modified_files),
    )
    add_trace("BUILD_VALIDATION", "started", {
        "modified_files_count": len(modified_files),
        "modified_files": modified_files[:20],  # Truncate for trace
    })

    build_results = await validate_all_affected_projects(client, modified_files)

    if not build_results:
        # No projects detected — treat as warning, not failure
        logger.warning(
            "[build_validator] No affected projects detected — treating as pass (warning)"
        )
        add_trace("BUILD_VALIDATION", "warning_no_projects", {
            "modified_files": modified_files[:10],
        })
        return True, [], []

    # Pre-compute inferred project paths for fix execution later
    affected = detect_affected_projects(modified_files)
    inferred_frontend = _infer_project_path(
        PROJECT_VITE_REACT,
        affected.get(PROJECT_VITE_REACT, []),
    ) if PROJECT_VITE_REACT in affected else SANDBOX_FRONTEND_PATH
    inferred_backend = _infer_project_path(
        PROJECT_PYTHON_BACKEND,
        affected.get(PROJECT_PYTHON_BACKEND, []),
    ) if PROJECT_PYTHON_BACKEND in affected else SANDBOX_BACKEND_PATH

    # Check if all builds passed
    all_passed = all(r.passed for r in build_results)
    if all_passed:
        add_trace("BUILD_VALIDATION", "passed", {
            "projects_validated": [
                {"type": r.project_type, "duration_ms": r.duration_ms}
                for r in build_results
            ],
        })
        logger.info("[build_validator] ✓ All builds passed on first try")
        return True, build_results, []

    # Build failed — enter diagnostic/retry loop
    logger.warning(
        "[build_validator] Build failed — entering diagnostic loop (max %d attempts)",
        MAX_BUILD_FIX_ATTEMPTS,
    )
    add_trace("BUILD_VALIDATION", "failed_entering_retry", {
        "failed_projects": [
            r.to_dict() for r in build_results if not r.passed
        ],
    })

    fix_history: List[Dict[str, Any]] = []
    previous_fixes: List[Dict] = []

    for attempt in range(1, MAX_BUILD_FIX_ATTEMPTS + 1):
        logger.info(
            "[build_validator] === Fix attempt %d/%d ===",
            attempt, MAX_BUILD_FIX_ATTEMPTS,
        )
        add_trace("BUILD_FIX_ATTEMPT", "started", {"attempt": attempt})

        # Diagnose
        diagnostic = await diagnose_build_failure(
            llm_call_fn=llm_call_fn,
            spec_content=spec_content,
            pot_result=pot_result,
            build_results=build_results,
            attempt=attempt,
            previous_fixes=previous_fixes,
        )

        logger.info(
            "[build_validator] Diagnosis: %s (root_cause=%s, confidence=%.2f, fixes=%d)",
            diagnostic.diagnosis,
            diagnostic.root_cause,
            diagnostic.confidence,
            len(diagnostic.fixes),
        )

        add_trace("BUILD_FIX_DIAGNOSIS", "complete", {
            "attempt": attempt,
            "diagnosis": diagnostic.diagnosis,
            "root_cause": diagnostic.root_cause,
            "confidence": diagnostic.confidence,
            "fix_count": len(diagnostic.fixes),
        })

        if not diagnostic.fixes:
            logger.warning(
                "[build_validator] No fixes suggested by LLM — attempt %d failed",
                attempt,
            )
            add_trace("BUILD_FIX_ATTEMPT", "no_fixes", {"attempt": attempt})
            fix_history.append({
                "attempt": attempt,
                "diagnosis": diagnostic.diagnosis,
                "root_cause": diagnostic.root_cause,
                "fixes": [],
                "result": "no_fixes_suggested",
            })
            previous_fixes.append({
                "attempt": attempt,
                "diagnosis": diagnostic.diagnosis,
                "fix_type": "none",
            })
            continue

        # Execute fixes (using inferred project paths for correct directory)
        fix_results = await execute_all_fixes(
            client, diagnostic,
            inferred_frontend_path=inferred_frontend,
            inferred_backend_path=inferred_backend,
        )

        add_trace("BUILD_FIX_EXECUTED", "complete", {
            "attempt": attempt,
            "fixes_executed": len(fix_results),
            "fixes_succeeded": sum(1 for r in fix_results if r.get("success")),
        })

        fix_history.append({
            "attempt": attempt,
            "diagnosis": diagnostic.diagnosis,
            "root_cause": diagnostic.root_cause,
            "fixes": [f.to_dict() for f in diagnostic.fixes],
            "fix_results": fix_results,
        })

        for fix_data in diagnostic.fixes:
            previous_fixes.append({
                "attempt": attempt,
                "diagnosis": diagnostic.diagnosis,
                "fix_type": fix_data.fix_type,
            })

        # Re-validate build
        build_results = await validate_all_affected_projects(client, modified_files)
        all_passed = all(r.passed for r in build_results)

        if all_passed:
            logger.info(
                "[build_validator] ✓ Build passed after fix attempt %d", attempt
            )
            add_trace("BUILD_FIX_SUCCESS", "passed", {
                "attempt": attempt,
                "diagnosis": diagnostic.diagnosis,
            })
            return True, build_results, fix_history

        logger.warning(
            "[build_validator] Build still failing after attempt %d", attempt
        )
        add_trace("BUILD_FIX_ATTEMPT", "still_failing", {
            "attempt": attempt,
            "remaining_errors": [
                r.error_summary for r in build_results if not r.passed
            ],
        })

    # Exhausted all retries
    logger.error(
        "[build_validator] ✗ Build validation FAILED after %d fix attempts",
        MAX_BUILD_FIX_ATTEMPTS,
    )
    add_trace("BUILD_FIX_EXHAUSTED", "failed", {
        "total_attempts": MAX_BUILD_FIX_ATTEMPTS,
        "final_errors": [r.to_dict() for r in build_results if not r.passed],
        "fix_history": fix_history,
    })

    return False, build_results, fix_history


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    "MAX_BUILD_FIX_ATTEMPTS",
    "BUILD_VALIDATION_TIMEOUT",
    "BUILD_VALIDATION_ENABLED",
    "SANDBOX_FRONTEND_PATH",
    "SANDBOX_BACKEND_PATH",
    # Data classes
    "BuildValidationResult",
    "BuildFixAction",
    "DiagnosticResult",
    # Detection
    "detect_project_from_path",
    "detect_affected_projects",
    "detect_project_type_from_sandbox",
    "_infer_project_path",
    # Parsing
    "parse_build_error_output",
    # Validation
    "validate_build",
    "validate_all_affected_projects",
    # Diagnostic
    "diagnose_build_failure",
    "execute_build_fix",
    "execute_all_fixes",
    # Main entry point
    "run_build_validation_loop",
]
