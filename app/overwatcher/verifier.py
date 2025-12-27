# FILE: app/overwatcher/verifier.py
"""Block 9: Verification Gate.

Runs tests, lint, and type checks on touched files after chunk implementation.

Key behaviors:
- Blocking: tests + lint + types for touched files
- Tracked: existing legacy failures as backlog
- Evidence storage for audit trail
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from app.overwatcher.schemas import (
    Chunk,
    CommandResult,
    VerificationResult,
    VerificationStatus,
)

logger = logging.getLogger(__name__)

# Configuration
VERIFICATION_TIMEOUT = int(os.getenv("ORB_VERIFICATION_TIMEOUT", "120"))
PYTEST_CMD = os.getenv("ORB_PYTEST_CMD", "python -m pytest")
RUFF_CMD = os.getenv("ORB_RUFF_CMD", "ruff check")
MYPY_CMD = os.getenv("ORB_MYPY_CMD", "python -m mypy")


# =============================================================================
# Command Execution
# =============================================================================

def run_command(
    command: str,
    cwd: str,
    timeout: int = VERIFICATION_TIMEOUT,
    env: Optional[Dict[str, str]] = None,
) -> CommandResult:
    """Run a shell command and capture results.
    
    Args:
        command: Command to run
        cwd: Working directory
        timeout: Timeout in seconds
        env: Environment variables
    
    Returns:
        CommandResult with exit code and output
    """
    start_time = time.time()
    
    try:
        # Use shell on Windows
        use_shell = os.name == "nt"
        
        result = subprocess.run(
            command,
            shell=use_shell,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, **(env or {})},
        )
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        return CommandResult(
            command=command,
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_ms=duration_ms,
            passed=result.returncode == 0,
        )
        
    except subprocess.TimeoutExpired:
        duration_ms = int((time.time() - start_time) * 1000)
        return CommandResult(
            command=command,
            exit_code=-1,
            stdout="",
            stderr=f"Command timed out after {timeout}s",
            duration_ms=duration_ms,
            passed=False,
        )
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        return CommandResult(
            command=command,
            exit_code=-1,
            stdout="",
            stderr=str(e),
            duration_ms=duration_ms,
            passed=False,
        )


# =============================================================================
# Test Discovery
# =============================================================================

def find_tests_for_files(
    touched_files: List[str],
    repo_path: str,
    test_dir: str = "tests",
) -> List[str]:
    """Find test files related to touched source files.
    
    Uses conventions:
    - app/foo/bar.py -> tests/test_bar.py
    - app/foo/bar.py -> tests/foo/test_bar.py
    """
    test_files = []
    
    for file_path in touched_files:
        p = Path(file_path)
        
        # Skip non-Python files
        if p.suffix != ".py":
            continue
        
        # Skip test files themselves
        if p.name.startswith("test_") or "/tests/" in str(p) or "\\tests\\" in str(p):
            test_files.append(str(p))
            continue
        
        # Try to find corresponding test file
        base_name = p.stem
        possible_tests = [
            Path(repo_path) / test_dir / f"test_{base_name}.py",
            Path(repo_path) / test_dir / p.parent.name / f"test_{base_name}.py",
        ]
        
        for test_path in possible_tests:
            if test_path.exists():
                rel_path = test_path.relative_to(repo_path)
                test_files.append(str(rel_path))
    
    return list(set(test_files))


# =============================================================================
# Individual Verifiers
# =============================================================================

def run_pytest(
    repo_path: str,
    test_files: Optional[List[str]] = None,
    extra_args: str = "",
) -> Tuple[CommandResult, int, int]:
    """Run pytest on specified files.
    
    Returns (result, passed_count, failed_count)
    """
    if test_files:
        files_arg = " ".join(test_files)
        command = f"{PYTEST_CMD} {files_arg} {extra_args} -v"
    else:
        command = f"{PYTEST_CMD} {extra_args} -v"
    
    result = run_command(command, repo_path)
    
    # Parse output for pass/fail counts
    passed = 0
    failed = 0
    
    # Look for summary line like "5 passed, 2 failed"
    match = re.search(r"(\d+) passed", result.stdout)
    if match:
        passed = int(match.group(1))
    
    match = re.search(r"(\d+) failed", result.stdout)
    if match:
        failed = int(match.group(1))
    
    return result, passed, failed


def run_ruff(
    repo_path: str,
    files: Optional[List[str]] = None,
) -> Tuple[CommandResult, int]:
    """Run ruff linter on specified files.
    
    Returns (result, error_count)
    """
    if files:
        # Filter to Python files
        py_files = [f for f in files if f.endswith(".py")]
        if not py_files:
            return CommandResult(
                command="ruff (skipped)",
                exit_code=0,
                stdout="No Python files to check",
                stderr="",
                duration_ms=0,
                passed=True,
            ), 0
        files_arg = " ".join(py_files)
        command = f"{RUFF_CMD} {files_arg}"
    else:
        command = f"{RUFF_CMD} ."
    
    result = run_command(command, repo_path)
    
    # Count errors (each non-empty line is typically an error)
    error_count = 0
    if result.stdout:
        error_count = len([l for l in result.stdout.strip().split("\n") if l.strip()])
    
    return result, error_count


def run_mypy(
    repo_path: str,
    files: Optional[List[str]] = None,
) -> Tuple[CommandResult, int]:
    """Run mypy type checker on specified files.
    
    Returns (result, error_count)
    """
    if files:
        py_files = [f for f in files if f.endswith(".py")]
        if not py_files:
            return CommandResult(
                command="mypy (skipped)",
                exit_code=0,
                stdout="No Python files to check",
                stderr="",
                duration_ms=0,
                passed=True,
            ), 0
        files_arg = " ".join(py_files)
        command = f"{MYPY_CMD} {files_arg} --ignore-missing-imports"
    else:
        command = f"{MYPY_CMD} . --ignore-missing-imports"
    
    result = run_command(command, repo_path)
    
    # Count errors
    error_count = 0
    if result.stdout:
        # Count lines with "error:" in them
        error_count = len([l for l in result.stdout.split("\n") if "error:" in l])
    
    return result, error_count


# =============================================================================
# Legacy Failure Tracking
# =============================================================================

def load_legacy_failures(
    repo_path: str,
    failures_file: str = ".orb_legacy_failures.json",
) -> Dict[str, List[str]]:
    """Load known legacy failures that predate this implementation.
    
    Returns dict of category -> list of failure identifiers
    """
    failures_path = Path(repo_path) / failures_file
    
    if not failures_path.exists():
        return {"tests": [], "lint": [], "types": []}
    
    try:
        return json.loads(failures_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"[verifier] Failed to load legacy failures: {e}")
        return {"tests": [], "lint": [], "types": []}


def save_legacy_failures(
    repo_path: str,
    failures: Dict[str, List[str]],
    failures_file: str = ".orb_legacy_failures.json",
) -> None:
    """Save legacy failures for future reference."""
    failures_path = Path(repo_path) / failures_file
    failures_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")


def filter_new_failures(
    all_failures: List[str],
    legacy_failures: List[str],
) -> List[str]:
    """Return only failures that aren't in the legacy list."""
    legacy_set = set(legacy_failures)
    return [f for f in all_failures if f not in legacy_set]


# =============================================================================
# Main Verification Gate
# =============================================================================

async def verify_chunk(
    chunk: Chunk,
    repo_path: str,
    touched_files: List[str],
    job_artifact_root: Optional[str] = None,
) -> VerificationResult:
    """Run verification gate for a chunk.
    
    Args:
        chunk: The chunk that was implemented
        repo_path: Path to repository
        touched_files: Files that were touched
        job_artifact_root: Path to store evidence
    
    Returns:
        VerificationResult with pass/fail and evidence
    """
    logger.info(f"[verifier] Running verification for chunk {chunk.chunk_id}")
    
    results: List[CommandResult] = []
    tests_passed = 0
    tests_failed = 0
    lint_errors = 0
    type_errors = 0
    legacy_failures: List[str] = []
    evidence_paths: List[str] = []
    
    # Load legacy failures
    legacy = load_legacy_failures(repo_path)
    
    # 1. Run chunk-specific verification commands first
    for cmd in chunk.verification.commands:
        result = run_command(cmd, repo_path, timeout=chunk.verification.timeout_seconds)
        results.append(result)
    
    # 2. Run pytest on related tests
    test_files = find_tests_for_files(touched_files, repo_path)
    if test_files:
        pytest_result, passed, failed = run_pytest(repo_path, test_files)
        results.append(pytest_result)
        tests_passed += passed
        tests_failed += failed
    
    # 3. Run ruff on touched files
    ruff_result, ruff_errors = run_ruff(repo_path, touched_files)
    results.append(ruff_result)
    lint_errors = ruff_errors
    
    # 4. Run mypy on touched files
    mypy_result, mypy_errors = run_mypy(repo_path, touched_files)
    results.append(mypy_result)
    type_errors = mypy_errors
    
    # Store evidence
    if job_artifact_root:
        evidence_dir = Path(job_artifact_root) / "jobs" / chunk.chunk_id / "verification"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        
        for i, result in enumerate(results):
            evidence_path = evidence_dir / f"cmd_{i}.json"
            evidence_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            evidence_paths.append(str(evidence_path))
    
    # Determine overall status
    all_passed = all(r.passed for r in results)
    
    # Check for new failures (not in legacy)
    # For now, we're strict: any failure blocks
    if all_passed and tests_failed == 0 and lint_errors == 0 and type_errors == 0:
        status = VerificationStatus.PASSED
    else:
        status = VerificationStatus.FAILED
    
    return VerificationResult(
        chunk_id=chunk.chunk_id,
        status=status,
        command_results=results,
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        lint_errors=lint_errors,
        type_errors=type_errors,
        evidence_paths=evidence_paths,
        legacy_failures=legacy_failures,
    )


def run_full_verification(
    repo_path: str,
    timeout: int = 300,
) -> Tuple[bool, CommandResult]:
    """Run full test suite (for smoke/quarantine verification).
    
    Returns (passed, result)
    """
    command = f"{PYTEST_CMD} -x -q"  # Stop on first failure, quiet
    result = run_command(command, repo_path, timeout=timeout)
    return result.passed, result


def run_smoke_boot(
    repo_path: str,
    entry_point: str = "app.main:app",
    timeout: int = 30,
) -> Tuple[bool, CommandResult]:
    """Try to import the main application to verify it boots.
    
    Returns (passed, result)
    """
    # Try to import the module
    module = entry_point.split(":")[0]
    command = f"python -c \"import {module}\""
    result = run_command(command, repo_path, timeout=timeout)
    return result.passed, result


__all__ = [
    # Command execution
    "run_command",
    # Test discovery
    "find_tests_for_files",
    # Individual verifiers
    "run_pytest",
    "run_ruff",
    "run_mypy",
    # Legacy tracking
    "load_legacy_failures",
    "save_legacy_failures",
    "filter_new_failures",
    # Main verification
    "verify_chunk",
    "run_full_verification",
    "run_smoke_boot",
]
