# FILE: app/overwatcher/sandbox_verifier.py
"""Sandbox Verifier: Block 9 verification via isolated sandbox.

Runs pytest, ruff, mypy in Windows Sandbox for safe execution of
potentially untrusted code changes.

Key behaviors:
- Falls back to local execution if sandbox unavailable
- Converts sandbox ShellResult to CommandResult for compatibility
- Supports both sandbox and local verification modes
"""

from __future__ import annotations

import logging
import os
import re
from typing import List, Optional, Tuple

from app.overwatcher.schemas import (
    Chunk,
    CommandResult,
    VerificationResult,
    VerificationStatus,
)
from app.overwatcher.sandbox_client import (
    SandboxClient,
    SandboxError,
    ShellResult,
    get_sandbox_client,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Use sandbox by default if available
USE_SANDBOX = os.getenv("ORB_USE_SANDBOX", "1").lower() in {"1", "true", "yes"}

# Verification timeouts
PYTEST_TIMEOUT = int(os.getenv("ORB_PYTEST_TIMEOUT", "120"))
RUFF_TIMEOUT = int(os.getenv("ORB_RUFF_TIMEOUT", "60"))
MYPY_TIMEOUT = int(os.getenv("ORB_MYPY_TIMEOUT", "120"))


# =============================================================================
# Result Conversion
# =============================================================================

def shell_to_command_result(
    shell_result: ShellResult,
    command: str,
) -> CommandResult:
    """Convert SandboxClient ShellResult to verifier CommandResult."""
    return CommandResult(
        command=command,
        exit_code=shell_result.exit_code,
        stdout=shell_result.stdout,
        stderr=shell_result.stderr,
        duration_ms=shell_result.duration_ms,
        timed_out=not shell_result.ok and shell_result.exit_code == -1,
    )


def command_passed(result: CommandResult) -> bool:
    """Check if a CommandResult represents a passing command."""
    return result.exit_code == 0 and not result.timed_out


def parse_pytest_counts(output: str) -> Tuple[int, int]:
    """Parse pytest output for pass/fail counts.
    
    Returns:
        (passed_count, failed_count)
    """
    passed = 0
    failed = 0
    
    # Look for summary line like "5 passed, 2 failed"
    match = re.search(r"(\d+) passed", output)
    if match:
        passed = int(match.group(1))
    
    match = re.search(r"(\d+) failed", output)
    if match:
        failed = int(match.group(1))
    
    return passed, failed


def parse_lint_error_count(output: str) -> int:
    """Parse ruff output for error count."""
    if not output.strip():
        return 0
    
    # Each non-empty line is typically an error
    return len([line for line in output.strip().split("\n") if line.strip()])


def parse_type_error_count(output: str) -> int:
    """Parse mypy output for error count."""
    if not output:
        return 0
    
    # Count lines with "error:" in them
    return len([line for line in output.split("\n") if "error:" in line])


# =============================================================================
# Sandbox Verification Functions
# =============================================================================

def sandbox_run_pytest(
    client: SandboxClient,
    test_files: Optional[List[str]] = None,
    extra_args: str = "-v",
    timeout: int = PYTEST_TIMEOUT,
) -> Tuple[CommandResult, int, int]:
    """Run pytest via sandbox.
    
    Args:
        client: SandboxClient instance
        test_files: Specific test files to run (None = all)
        extra_args: Additional pytest arguments
        timeout: Command timeout
    
    Returns:
        (CommandResult, passed_count, failed_count)
    """
    if test_files:
        files_arg = " ".join(test_files)
        command = f"python -m pytest {files_arg} {extra_args}"
    else:
        command = f"python -m pytest {extra_args}"
    
    try:
        result = client.shell_run(command, timeout_seconds=timeout)
        cmd_result = shell_to_command_result(result, command)
        
        # Parse counts from output
        output = result.stdout + "\n" + result.stderr
        passed, failed = parse_pytest_counts(output)
        
        return cmd_result, passed, failed
        
    except SandboxError as e:
        logger.warning(f"[sandbox_verifier] pytest failed: {e}")
        return CommandResult(
            command=command,
            exit_code=-1,
            stdout="",
            stderr=str(e),
            duration_ms=0,
            timed_out=True,
        ), 0, 0


def sandbox_run_ruff(
    client: SandboxClient,
    files: Optional[List[str]] = None,
    timeout: int = RUFF_TIMEOUT,
) -> Tuple[CommandResult, int]:
    """Run ruff linter via sandbox.
    
    Args:
        client: SandboxClient instance
        files: Specific files to lint (None = all)
        timeout: Command timeout
    
    Returns:
        (CommandResult, error_count)
    """
    if files:
        py_files = [f for f in files if f.endswith(".py")]
        if not py_files:
            return CommandResult(
                command="ruff (skipped)",
                exit_code=0,
                stdout="No Python files to check",
                stderr="",
                duration_ms=0,
                timed_out=False,
            ), 0
        target = " ".join(py_files)
    else:
        target = "."
    
    command = f"ruff check {target}"
    
    try:
        result = client.shell_run(command, timeout_seconds=timeout)
        cmd_result = shell_to_command_result(result, command)
        error_count = parse_lint_error_count(result.stdout)
        
        return cmd_result, error_count
        
    except SandboxError as e:
        logger.warning(f"[sandbox_verifier] ruff failed: {e}")
        return CommandResult(
            command=command,
            exit_code=-1,
            stdout="",
            stderr=str(e),
            duration_ms=0,
            timed_out=True,
        ), 0


def sandbox_run_mypy(
    client: SandboxClient,
    files: Optional[List[str]] = None,
    timeout: int = MYPY_TIMEOUT,
) -> Tuple[CommandResult, int]:
    """Run mypy type checker via sandbox.
    
    Args:
        client: SandboxClient instance
        files: Specific files to check (None = all)
        timeout: Command timeout
    
    Returns:
        (CommandResult, error_count)
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
                timed_out=False,
            ), 0
        target = " ".join(py_files)
    else:
        target = "."
    
    command = f"python -m mypy {target} --ignore-missing-imports"
    
    try:
        result = client.shell_run(command, timeout_seconds=timeout)
        cmd_result = shell_to_command_result(result, command)
        error_count = parse_type_error_count(result.stdout)
        
        return cmd_result, error_count
        
    except SandboxError as e:
        logger.warning(f"[sandbox_verifier] mypy failed: {e}")
        return CommandResult(
            command=command,
            exit_code=-1,
            stdout="",
            stderr=str(e),
            duration_ms=0,
            timed_out=True,
        ), 0


def sandbox_run_command(
    client: SandboxClient,
    command: str,
    timeout: int = 60,
) -> CommandResult:
    """Run arbitrary command via sandbox.
    
    Args:
        client: SandboxClient instance
        command: Shell command to run
        timeout: Command timeout
    
    Returns:
        CommandResult
    """
    try:
        result = client.shell_run(command, timeout_seconds=timeout)
        return shell_to_command_result(result, command)
    except SandboxError as e:
        logger.warning(f"[sandbox_verifier] Command failed: {e}")
        return CommandResult(
            command=command,
            exit_code=-1,
            stdout="",
            stderr=str(e),
            duration_ms=0,
            timed_out=True,
        )


# =============================================================================
# Main Verification Gate (Sandbox-Aware)
# =============================================================================

async def verify_chunk_sandbox(
    chunk: Chunk,
    touched_files: List[str],
    job_artifact_root: Optional[str] = None,
    client: Optional[SandboxClient] = None,
) -> VerificationResult:
    """Run verification gate for a chunk via sandbox.
    
    This is the sandbox-aware version of verify_chunk from verifier.py.
    
    Args:
        chunk: The chunk that was implemented
        touched_files: Files that were touched
        job_artifact_root: Path to store evidence
        client: SandboxClient instance (auto-creates if None)
    
    Returns:
        VerificationResult with pass/fail and evidence
    """
    from app.overwatcher.verifier import find_tests_for_files
    
    logger.info(f"[sandbox_verifier] Running verification for chunk {chunk.chunk_id}")
    
    # Get or create sandbox client
    if client is None:
        client = get_sandbox_client()
    
    # Check sandbox connection
    if not client.is_connected():
        logger.warning("[sandbox_verifier] Sandbox not available, falling back to local")
        # Fall back to local verification
        from app.overwatcher.verifier import verify_chunk
        return await verify_chunk(
            chunk=chunk,
            repo_path=".",  # Assume current directory
            touched_files=touched_files,
            job_artifact_root=job_artifact_root,
        )
    
    results: List[CommandResult] = []
    tests_passed = 0
    tests_failed = 0
    lint_errors = 0
    type_errors = 0
    evidence_paths: List[str] = []
    
    # 1. Run chunk-specific verification commands
    for cmd in chunk.verification.commands:
        result = sandbox_run_command(
            client, cmd, timeout=chunk.verification.timeout_seconds
        )
        results.append(result)
    
    # 2. Run pytest on related tests
    test_files = find_tests_for_files(touched_files, ".")
    if test_files:
        pytest_result, passed, failed = sandbox_run_pytest(client, test_files)
        results.append(pytest_result)
        tests_passed += passed
        tests_failed += failed
    
    # 3. Run ruff on touched files
    ruff_result, ruff_errors = sandbox_run_ruff(client, touched_files)
    results.append(ruff_result)
    lint_errors = ruff_errors
    
    # 4. Run mypy on touched files
    mypy_result, mypy_errors = sandbox_run_mypy(client, touched_files)
    results.append(mypy_result)
    type_errors = mypy_errors
    
    # Store evidence
    if job_artifact_root:
        import json
        from pathlib import Path
        
        evidence_dir = Path(job_artifact_root) / "jobs" / chunk.chunk_id / "verification"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        
        for i, result in enumerate(results):
            evidence_path = evidence_dir / f"sandbox_cmd_{i}.json"
            evidence_path.write_text(
                json.dumps(result.to_dict(), indent=2),
                encoding="utf-8"
            )
            evidence_paths.append(str(evidence_path))
    
    # Determine overall status
    all_passed = all(command_passed(r) for r in results)
    
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
    )


async def run_full_verification_sandbox(
    client: Optional[SandboxClient] = None,
    timeout: int = 300,
) -> Tuple[bool, CommandResult]:
    """Run full test suite via sandbox.
    
    Args:
        client: SandboxClient instance
        timeout: Timeout for test run
    
    Returns:
        (passed, CommandResult)
    """
    if client is None:
        client = get_sandbox_client()
    
    if not client.is_connected():
        logger.warning("[sandbox_verifier] Sandbox not available")
        return False, CommandResult(
            command="pytest (sandbox unavailable)",
            exit_code=-1,
            stdout="",
            stderr="Sandbox not available",
            duration_ms=0,
            timed_out=False,
        )
    
    result, passed, failed = sandbox_run_pytest(
        client,
        test_files=None,
        extra_args="-x -q",  # Stop on first failure, quiet
        timeout=timeout,
    )
    
    return command_passed(result) and failed == 0, result


async def run_smoke_boot_sandbox(
    client: Optional[SandboxClient] = None,
    entry_point: str = "app.main",
    timeout: int = 30,
) -> Tuple[bool, CommandResult]:
    """Try to import the main application via sandbox.
    
    Args:
        client: SandboxClient instance
        entry_point: Module to import
        timeout: Timeout
    
    Returns:
        (passed, CommandResult)
    """
    if client is None:
        client = get_sandbox_client()
    
    if not client.is_connected():
        logger.warning("[sandbox_verifier] Sandbox not available")
        return False, CommandResult(
            command="smoke boot (sandbox unavailable)",
            exit_code=-1,
            stdout="",
            stderr="Sandbox not available",
            duration_ms=0,
            timed_out=False,
        )
    
    module = entry_point.split(":")[0]
    command = f'python -c "import {module}"'
    result = sandbox_run_command(client, command, timeout=timeout)
    
    return command_passed(result), result


__all__ = [
    # Conversion
    "shell_to_command_result",
    "command_passed",
    "parse_pytest_counts",
    "parse_lint_error_count",
    "parse_type_error_count",
    # Individual verifiers
    "sandbox_run_pytest",
    "sandbox_run_ruff",
    "sandbox_run_mypy",
    "sandbox_run_command",
    # Main verification
    "verify_chunk_sandbox",
    "run_full_verification_sandbox",
    "run_smoke_boot_sandbox",
]
