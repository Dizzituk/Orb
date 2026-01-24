# FILE: app/overwatcher/implementer.py
"""Implementer: Executes approved work and verifies results.

Handles:
- Writing files to sandbox based on spec
- Enforcing must_exist constraint for modify actions
- Verifying output matches spec requirements
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.overwatcher.overwatcher import OverwatcherOutput, Decision
from app.overwatcher.sandbox_client import (
    SandboxClient,
    SandboxError,
    get_sandbox_client,
)

from .spec_resolution import ResolvedSpec, SpecMissingDeliverableError

logger = logging.getLogger(__name__)


def _is_absolute_windows_path(path: str) -> bool:
    """Check if path is an absolute Windows path (e.g., C:\\..., D:\\...)."""
    if len(path) >= 3:
        return path[1] == ':' and path[2] in ('\\', '/')
    return False


def _escape_powershell_string(s: str) -> str:
    """Escape a string for use in PowerShell double-quoted strings."""
    # Escape backticks first, then quotes, then dollar signs
    return s.replace('`', '``').replace('"', '`"').replace('$', '`$')


def _generate_sandbox_path_candidates(path: str) -> List[str]:
    """Generate candidate paths for sandbox resolution.
    
    For Desktop paths, tries:
    1. Original path as-is
    2. Same user, non-OneDrive Desktop (if original has OneDrive)
    3. WDAGUtilityAccount OneDrive Desktop (if original has OneDrive)
    4. WDAGUtilityAccount Desktop
    
    Args:
        path: Original absolute Windows path
        
    Returns:
        List of candidate paths to try (in priority order)
    """
    candidates = [path]
    
    # Match: C:\Users\<username>\OneDrive\Desktop\<rest>
    onedrive_match = re.match(
        r'^([A-Za-z]):\\Users\\([^\\]+)\\OneDrive\\Desktop\\(.*)$',
        path,
        re.IGNORECASE
    )
    if onedrive_match:
        drive = onedrive_match.group(1)
        username = onedrive_match.group(2)
        rest = onedrive_match.group(3)
        
        # Candidate 2: Same user, non-OneDrive Desktop
        non_onedrive = f"{drive}:\\Users\\{username}\\Desktop\\{rest}"
        if non_onedrive not in candidates:
            candidates.append(non_onedrive)
        
        # Candidate 3: WDAGUtilityAccount OneDrive Desktop
        wdag_onedrive = f"{drive}:\\Users\\WDAGUtilityAccount\\OneDrive\\Desktop\\{rest}"
        if wdag_onedrive not in candidates:
            candidates.append(wdag_onedrive)
        
        # Candidate 4: WDAGUtilityAccount Desktop
        wdag = f"{drive}:\\Users\\WDAGUtilityAccount\\Desktop\\{rest}"
        if wdag not in candidates:
            candidates.append(wdag)
        
        return candidates
    
    # Match: C:\Users\<username>\Desktop\<rest> (non-OneDrive)
    desktop_match = re.match(
        r'^([A-Za-z]):\\Users\\([^\\]+)\\Desktop\\(.*)$',
        path,
        re.IGNORECASE
    )
    if desktop_match:
        drive = desktop_match.group(1)
        username = desktop_match.group(2)
        rest = desktop_match.group(3)
        
        # Candidate 2: WDAGUtilityAccount Desktop
        wdag = f"{drive}:\\Users\\WDAGUtilityAccount\\Desktop\\{rest}"
        if wdag not in candidates:
            candidates.append(wdag)
        
        return candidates
    
    return candidates


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
        # If filename is already absolute (from SpecGate sandbox paths), use as-is
        if _is_absolute_windows_path(filename):
            expected_path = filename
            base_filename = Path(filename).name
            is_absolute = True
            logger.info(f"[implementer] Using absolute path as-is: {expected_path}")
        else:
            base_filename = filename
            is_absolute = False
            if target == "DESKTOP":
                expected_path = f"C:\\Users\\WDAGUtilityAccount\\Desktop\\{base_filename}"
            else:
                expected_path = f"{target}\\{base_filename}"
        
        # For "modify" action with must_exist: verify file exists first
        # Try multiple candidate paths (original + sandbox remapped)
        if action == "modify" and must_exist:
            # Log sandbox environment for debugging
            sandbox_whoami = "<unknown>"
            sandbox_userprofile = "<unknown>"
            try:
                whoami_result = client.shell_run("whoami", timeout_seconds=5)
                sandbox_whoami = whoami_result.stdout.strip() if whoami_result.stdout else f"<error: {whoami_result.stderr}>"
                userprofile_result = client.shell_run("echo $env:USERPROFILE", timeout_seconds=5)
                sandbox_userprofile = userprofile_result.stdout.strip() if userprofile_result.stdout else f"<error: {userprofile_result.stderr}>"
                logger.info(f"[implementer] Sandbox env: whoami={sandbox_whoami}")
                logger.info(f"[implementer] Sandbox env: USERPROFILE={sandbox_userprofile}")
            except Exception as e:
                logger.warning(f"[implementer] Could not log sandbox env: {e}")
            
            candidates = _generate_sandbox_path_candidates(expected_path)
            
            resolved_path = None
            candidate_results = []  # Track results for diagnostics
            for candidate in candidates:
                logger.info(f"[implementer] Checking existence: {candidate}")
                exists_cmd = f'Test-Path -Path "{candidate}"'
                exists_result = client.shell_run(exists_cmd, timeout_seconds=10)
                
                file_exists = "True" in exists_result.stdout
                candidate_results.append((candidate, file_exists))
                logger.info(f"[implementer] Exists? {candidate} -> {file_exists}")
                
                if file_exists:
                    resolved_path = candidate
                    break
            
            if resolved_path is None:
                # Build detailed diagnostic error message
                candidate_lines = "\n".join(f"  - {c} -> {r}" for c, r in candidate_results)
                error_msg = (
                    f"SPEC VIOLATION: File '{filename}' does not exist at any candidate path.\n"
                    f"Tried:\n{candidate_lines}\n"
                    f"Sandbox env:\n"
                    f"  - whoami: {sandbox_whoami}\n"
                    f"  - USERPROFILE: {sandbox_userprofile}\n"
                    f"Spec requires modifying an existing file (action=modify, must_exist=True). "
                    f"Cannot create a new file."
                )
                return ImplementerResult(
                    success=False,
                    error=error_msg,
                    duration_ms=elapsed(),
                    sandbox_used=True,
                    filename=filename,
                    action_taken="existence_check_failed",
                )
            
            # Use resolved path for subsequent operations
            expected_path = resolved_path
            logger.info(f"[implementer] File exists at: {expected_path}, proceeding with modify")
        
        # Write file via sandbox
        # For absolute paths, use PowerShell Set-Content directly
        # For relative paths, use the sandbox write_file API
        if is_absolute:
            logger.info(f"[implementer] Writing via PowerShell to absolute path: {expected_path}")
            escaped_content = _escape_powershell_string(content)
            write_cmd = f'Set-Content -Path "{expected_path}" -Value "{escaped_content}" -NoNewline'
            write_result = client.shell_run(write_cmd, timeout_seconds=30)
            
            # Check success: no stderr means success for Set-Content
            write_success = not write_result.stderr or write_result.stderr.strip() == ""
            if write_success:
                logger.info(f"[implementer] SUCCESS: {expected_path}")
                return ImplementerResult(
                    success=True,
                    output_path=expected_path,
                    sha256=None,  # Not computed for direct writes
                    duration_ms=elapsed(),
                    sandbox_used=True,
                    filename=filename,
                    content_written=content,
                    action_taken=action,
                )
            else:
                return ImplementerResult(
                    success=False,
                    error=f"PowerShell write failed: {write_result.stderr or write_result.stdout}",
                    duration_ms=elapsed(),
                    sandbox_used=True,
                )
        else:
            logger.info(f"[implementer] Writing via sandbox API: {base_filename} -> {target}")
            result = client.write_file(
                target=target,
                filename=base_filename,
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
    
    # For absolute paths, compare basenames; otherwise compare full filename
    if _is_absolute_windows_path(expected_filename):
        expected_basename = Path(expected_filename).name
        filename_matches = actual_filename == expected_basename
    else:
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
        file_exists = "True" in exists_result.stdout
        
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
        
        if read_result.stderr and read_result.stderr.strip():
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
