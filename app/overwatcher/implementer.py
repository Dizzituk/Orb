# FILE: app/overwatcher/implementer.py
"""Implementer: Executes approved work and verifies results.

Handles:
- Writing files to sandbox based on spec
- Enforcing must_exist constraint for modify actions
- Verifying output matches spec requirements
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from app.overwatcher.overwatcher import OverwatcherOutput, Decision
from app.overwatcher.sandbox_client import (
    SandboxClient,
    SandboxError,
    get_sandbox_client,
)

from .spec_resolution import ResolvedSpec, SpecMissingDeliverableError

logger = logging.getLogger(__name__)


@dataclass
class ImplementerResult:
    """Result from Implementer execution."""
    success: bool
    output_path: Optional[str] = None
    sha256: Optional[str] = None
    error: Optional[str] = None
    duration_ms: int = 0
    sandbox_used: bool = False
    filename: Optional[str] = None
    content_written: Optional[str] = None
    action_taken: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output_path": self.output_path,
            "sha256": self.sha256,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "sandbox_used": self.sandbox_used,
            "filename": self.filename,
            "content_written": self.content_written,
            "action_taken": self.action_taken,
        }


@dataclass
class VerificationResult:
    """Result from verification step."""
    passed: bool
    file_exists: bool = False
    content_matches: bool = False
    filename_matches: bool = False
    actual_content: Optional[str] = None
    expected_content: Optional[str] = None
    expected_filename: Optional[str] = None
    actual_filename: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "file_exists": self.file_exists,
            "content_matches": self.content_matches,
            "filename_matches": self.filename_matches,
            "actual_content": self.actual_content,
            "expected_content": self.expected_content,
            "expected_filename": self.expected_filename,
            "actual_filename": self.actual_filename,
            "error": self.error,
        }


async def run_implementer(
    *,
    spec: ResolvedSpec,
    output: OverwatcherOutput,
    client: Optional[SandboxClient] = None,
) -> ImplementerResult:
    """Execute approved work via Sandbox.
    
    For action="modify" with must_exist=True:
    - Checks file exists BEFORE writing
    - Fails if file doesn't exist (does NOT create it)
    """
    import time
    start_time = time.time()
    
    def elapsed() -> int:
        return int((time.time() - start_time) * 1000)
    
    if output.decision != Decision.PASS:
        return ImplementerResult(
            success=False,
            error=f"Overwatcher decision was {output.decision.value}",
            duration_ms=elapsed(),
        )
    
    # Get spec-driven file details
    try:
        filename, content, action = spec.get_target_file()
        target = spec.get_target()
        must_exist = spec.get_must_exist()
    except SpecMissingDeliverableError as e:
        return ImplementerResult(
            success=False,
            error=str(e),
            duration_ms=elapsed(),
        )
    
    logger.info(f"[implementer] === SPEC-DRIVEN TASK ===")
    logger.info(f"[implementer] Action: {action}")
    logger.info(f"[implementer] Filename: {filename}")
    logger.info(f"[implementer] Target: {target}")
    logger.info(f"[implementer] Content: '{content}'")
    logger.info(f"[implementer] Must exist: {must_exist}")
    
    # Get sandbox client
    if client is None:
        client = get_sandbox_client()
    
    try:
        if not client.is_connected():
            return ImplementerResult(
                success=False,
                error="SAFETY: Sandbox not available",
                duration_ms=elapsed(),
                sandbox_used=False,
            )
        
        # Build expected path
        if target == "DESKTOP":
            expected_path = f"C:\\Users\\WDAGUtilityAccount\\Desktop\\{filename}"
        else:
            expected_path = f"{target}\\{filename}"
        
        # For "modify" action with must_exist: verify file exists first
        if action == "modify" and must_exist:
            logger.info(f"[implementer] Checking existence: {expected_path}")
            
            exists_cmd = f'Test-Path -Path "{expected_path}"'
            exists_result = client.shell_run(exists_cmd, timeout_seconds=10)
            
            file_exists = exists_result.ok and "True" in exists_result.stdout
            logger.info(f"[implementer] File exists check: {file_exists}")
            
            if not file_exists:
                return ImplementerResult(
                    success=False,
                    error=f"SPEC VIOLATION: File '{filename}' does not exist at {expected_path}. "
                          f"Spec requires modifying an existing file (action=modify, must_exist=True). "
                          f"Cannot create a new file.",
                    duration_ms=elapsed(),
                    sandbox_used=True,
                    filename=filename,
                    action_taken="existence_check_failed",
                )
            
            logger.info(f"[implementer] File exists, proceeding with modify")
        
        # Write file via sandbox
        logger.info(f"[implementer] Writing to sandbox: {filename} -> {target}")
        result = client.write_file(
            target=target,
            filename=filename,
            content=content,
            overwrite=True,
        )
        
        if result.ok:
            logger.info(f"[implementer] SUCCESS: {result.path}")
            return ImplementerResult(
                success=True,
                output_path=result.path,
                sha256=result.sha256,
                duration_ms=elapsed(),
                sandbox_used=True,
                filename=filename,
                content_written=content,
                action_taken=action,
            )
        else:
            return ImplementerResult(
                success=False,
                error=f"Sandbox write failed: {getattr(result, 'error', 'unknown')}",
                duration_ms=elapsed(),
                sandbox_used=True,
            )
            
    except SandboxError as e:
        return ImplementerResult(
            success=False,
            error=f"Sandbox error: {e}",
            duration_ms=elapsed(),
            sandbox_used=True,
        )
    except Exception as e:
        logger.exception(f"[implementer] Failed: {e}")
        return ImplementerResult(
            success=False,
            error=str(e),
            duration_ms=elapsed(),
        )


async def run_verification(
    *,
    impl_result: ImplementerResult,
    spec: ResolvedSpec,
    client: Optional[SandboxClient] = None,
) -> VerificationResult:
    """Verify Implementer output against spec requirements.
    
    Checks:
    1. Correct filename (not some other file like hello.txt)
    2. Content matches spec exactly
    3. File exists at expected path
    """
    try:
        expected_filename, expected_content, expected_action = spec.get_target_file()
    except SpecMissingDeliverableError as e:
        return VerificationResult(
            passed=False,
            error=str(e),
        )
    
    logger.info(f"[verification] === SPEC VERIFICATION ===")
    logger.info(f"[verification] Expected filename: {expected_filename}")
    logger.info(f"[verification] Expected content: '{expected_content}'")
    logger.info(f"[verification] Expected action: {expected_action}")
    
    if not impl_result.success:
        return VerificationResult(
            passed=False,
            expected_filename=expected_filename,
            expected_content=expected_content,
            error=f"Implementation failed: {impl_result.error}",
        )
    
    if not impl_result.output_path:
        return VerificationResult(
            passed=False,
            expected_filename=expected_filename,
            expected_content=expected_content,
            error="No output path from Implementer",
        )
    
    # Check filename matches spec (CRITICAL: catch wrong file like hello.txt)
    actual_filename = Path(impl_result.output_path).name
    filename_matches = actual_filename == expected_filename
    
    logger.info(f"[verification] Actual filename: {actual_filename}")
    logger.info(f"[verification] Filename matches: {filename_matches}")
    
    if not filename_matches:
        return VerificationResult(
            passed=False,
            file_exists=True,
            content_matches=False,
            filename_matches=False,
            expected_filename=expected_filename,
            actual_filename=actual_filename,
            error=f"WRONG FILE: Spec requires '{expected_filename}' but got '{actual_filename}'. "
                  f"This is a spec violation.",
        )
    
    # Verify content via sandbox
    if client is None:
        client = get_sandbox_client()
    
    try:
        if not client.is_connected():
            return VerificationResult(
                passed=False,
                expected_filename=expected_filename,
                expected_content=expected_content,
                error="Sandbox not available for verification",
            )
        
        ps_path = impl_result.output_path.replace("/", "\\")
        
        # Check exists
        exists_result = client.shell_run(f'Test-Path -Path "{ps_path}"', timeout_seconds=10)
        file_exists = exists_result.ok and "True" in exists_result.stdout
        
        if not file_exists:
            return VerificationResult(
                passed=False,
                file_exists=False,
                filename_matches=filename_matches,
                expected_filename=expected_filename,
                expected_content=expected_content,
                error=f"File not found at {impl_result.output_path}",
            )
        
        # Read content
        read_result = client.shell_run(f'Get-Content -Path "{ps_path}" -Raw', timeout_seconds=10)
        
        if not read_result.ok:
            return VerificationResult(
                passed=False,
                file_exists=True,
                filename_matches=filename_matches,
                expected_filename=expected_filename,
                expected_content=expected_content,
                error=f"Failed to read file: {read_result.stderr}",
            )
        
        actual_content = read_result.stdout.strip()
        content_matches = actual_content == expected_content
        
        logger.info(f"[verification] Actual content: '{actual_content}'")
        logger.info(f"[verification] Content matches: {content_matches}")
        
        passed = content_matches and filename_matches
        
        return VerificationResult(
            passed=passed,
            file_exists=True,
            content_matches=content_matches,
            filename_matches=filename_matches,
            actual_content=actual_content,
            expected_content=expected_content,
            expected_filename=expected_filename,
            actual_filename=actual_filename,
            error=None if passed else f"Content mismatch: expected '{expected_content}', got '{actual_content}'",
        )
        
    except SandboxError as e:
        return VerificationResult(
            passed=False,
            expected_filename=expected_filename,
            expected_content=expected_content,
            error=f"Sandbox error: {e}",
        )
    except Exception as e:
        return VerificationResult(
            passed=False,
            expected_filename=expected_filename,
            expected_content=expected_content,
            error=str(e),
        )


__all__ = [
    "ImplementerResult",
    "VerificationResult",
    "run_implementer",
    "run_verification",
]
