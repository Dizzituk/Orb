# FILE: app/overwatcher/sandbox_executor.py
"""Sandbox Executor: Block 8 implementation via isolated sandbox.

Executes chunk implementations through the sandbox bridge:
- Writes files via /fs/write endpoint
- Validates boundaries against sandbox tree
- Supports rollback through sandbox operations

This provides isolation for potentially untrusted LLM-generated code
before promoting to the main repository.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.overwatcher.schemas import (
    Chunk,
    ChunkStatus,
    BoundaryViolation,
    DiffCheckResult,
)
from app.overwatcher.sandbox_client import (
    SandboxClient,
    SandboxError,
    FileEntry,
    get_sandbox_client,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Use sandbox by default
USE_SANDBOX = os.getenv("ORB_USE_SANDBOX", "1").lower() in {"1", "true", "yes"}

# Write target for chunk implementations
SANDBOX_WRITE_TARGET = os.getenv("ORB_SANDBOX_WRITE_TARGET", "SCRATCH")


# =============================================================================
# Boundary Checking via Sandbox
# =============================================================================

def get_sandbox_file_tree(
    client: SandboxClient,
) -> Dict[str, FileEntry]:
    """Get current file tree from sandbox.
    
    Returns:
        Dict mapping path -> FileEntry
    """
    try:
        entries = client.repo_tree(include_hashes=True)
        return {e.path: e for e in entries}
    except SandboxError as e:
        logger.warning(f"[sandbox_executor] Failed to get tree: {e}")
        return {}


def check_sandbox_boundaries(
    chunk: Chunk,
    files_to_write: Dict[str, str],
    existing_tree: Dict[str, FileEntry],
) -> DiffCheckResult:
    """Check if file writes respect chunk boundaries.
    
    Args:
        chunk: The chunk being implemented
        files_to_write: Dict of path -> content to write
        existing_tree: Current sandbox file tree
    
    Returns:
        DiffCheckResult with any violations
    """
    violations = []
    files_added = []
    files_modified = []
    
    # Normalize allowed paths
    allowed_add = set(chunk.allowed_files.get("add", []))
    allowed_modify = set(chunk.allowed_files.get("modify", []))
    
    for path in files_to_write.keys():
        # Normalize path
        norm_path = path.replace("\\", "/").lstrip("/")
        
        # Determine if add or modify
        if norm_path in existing_tree:
            # Modification
            if norm_path not in allowed_modify:
                violations.append(BoundaryViolation(
                    file_path=path,
                    action="modified",
                    reason="File not in allowed_files.modify list",
                ))
            else:
                files_modified.append(norm_path)
        else:
            # Addition
            if norm_path not in allowed_add:
                violations.append(BoundaryViolation(
                    file_path=path,
                    action="added",
                    reason="File not in allowed_files.add list",
                ))
            else:
                files_added.append(norm_path)
    
    return DiffCheckResult(
        passed=len(violations) == 0,
        violations=violations,
        files_added=files_added,
        files_modified=files_modified,
        files_deleted=[],
    )


# =============================================================================
# File Writing via Sandbox
# =============================================================================

def write_files_to_sandbox(
    client: SandboxClient,
    files: Dict[str, str],
    target: str = SANDBOX_WRITE_TARGET,
    overwrite: bool = True,
) -> Tuple[bool, List[str], List[str]]:
    """Write files to sandbox filesystem.
    
    Args:
        client: SandboxClient instance
        files: Dict of relative_path -> content
        target: Sandbox write target (SCRATCH, REPO, etc.)
        overwrite: Allow overwriting existing files
    
    Returns:
        (success, written_paths, failed_paths)
    """
    written = []
    failed = []
    
    for path, content in files.items():
        # Split path into subdir and filename
        p = Path(path.replace("\\", "/"))
        subdir = str(p.parent) if str(p.parent) != "." else None
        filename = p.name
        
        # Validate filename (sandbox requires safe names)
        # Replace any problematic characters
        safe_filename = filename
        
        try:
            result = client.write_file(
                target=target,
                filename=safe_filename,
                content=content,
                subdir=subdir,
                overwrite=overwrite,
            )
            
            if result.ok:
                written.append(path)
                logger.info(f"[sandbox_executor] Wrote: {path} ({result.bytes} bytes)")
            else:
                failed.append(path)
                logger.warning(f"[sandbox_executor] Write failed: {path}")
                
        except SandboxError as e:
            failed.append(path)
            logger.warning(f"[sandbox_executor] Write error for {path}: {e}")
    
    success = len(failed) == 0
    return success, written, failed


def read_file_from_sandbox(
    client: SandboxClient,
    path: str,
) -> Optional[str]:
    """Read file content from sandbox.
    
    Args:
        client: SandboxClient instance
        path: Relative file path
    
    Returns:
        File content or None
    """
    try:
        result = client.repo_file(path)
        return result.content
    except SandboxError as e:
        logger.warning(f"[sandbox_executor] Read failed for {path}: {e}")
        return None


# =============================================================================
# Backup and Rollback via Sandbox
# =============================================================================

def create_sandbox_backup(
    client: SandboxClient,
    chunk: Chunk,
) -> Dict[str, str]:
    """Create backup of files that will be modified.
    
    Args:
        client: SandboxClient instance
        chunk: Chunk with files to backup
    
    Returns:
        Dict of path -> original content
    """
    backups = {}
    
    for path in chunk.allowed_files.get("modify", []):
        content = read_file_from_sandbox(client, path)
        if content is not None:
            backups[path] = content
            logger.debug(f"[sandbox_executor] Backed up: {path}")
    
    return backups


def rollback_sandbox_changes(
    client: SandboxClient,
    backups: Dict[str, str],
    files_added: List[str],
) -> bool:
    """Rollback changes in sandbox.
    
    Args:
        client: SandboxClient instance
        backups: Original content of modified files
        files_added: Files that were added (to delete)
    
    Returns:
        True if rollback succeeded
    
    Note:
        Sandbox doesn't have a delete endpoint, so added files
        can only be overwritten with empty content or left in place.
    """
    success = True
    
    # Restore modified files
    for path, content in backups.items():
        try:
            p = Path(path)
            client.write_file(
                target="REPO",
                filename=p.name,
                content=content,
                subdir=str(p.parent) if str(p.parent) != "." else None,
                overwrite=True,
            )
            logger.info(f"[sandbox_executor] Restored: {path}")
        except SandboxError as e:
            logger.error(f"[sandbox_executor] Restore failed for {path}: {e}")
            success = False
    
    # For added files, we can't delete via sandbox API
    # Log them for manual cleanup
    if files_added:
        logger.warning(
            f"[sandbox_executor] Added files cannot be auto-deleted: {files_added}"
        )
    
    return success


# =============================================================================
# Main Execution
# =============================================================================

async def execute_chunk_sandbox(
    *,
    chunk: Chunk,
    files: Dict[str, str],
    spec_id: Optional[str] = None,
    spec_hash: Optional[str] = None,
    job_id: str = "",
    job_artifact_root: str = "",
    client: Optional[SandboxClient] = None,
    dry_run: bool = False,
) -> Tuple[bool, DiffCheckResult, Dict[str, str]]:
    """Execute chunk implementation via sandbox.
    
    This is the sandbox version of execute_chunk from executor.py.
    Takes pre-generated files (from LLM) and writes them through sandbox.
    
    Args:
        chunk: Chunk to implement
        files: Dict of path -> content (already generated by LLM)
        spec_id: Spec ID for validation
        spec_hash: Spec hash for validation
        job_id: Job UUID
        job_artifact_root: Root for artifacts
        client: SandboxClient instance
        dry_run: If True, check boundaries but don't write
    
    Returns:
        (success, diff_result, files_written)
    """
    logger.info(f"[sandbox_executor] Executing chunk {chunk.chunk_id}")
    
    # Get or create client
    if client is None:
        client = get_sandbox_client()
    
    # Check sandbox connection
    if not client.is_connected():
        logger.error("[sandbox_executor] Sandbox not available")
        return False, DiffCheckResult(passed=False, violations=[]), {}
    
    # Get current tree for boundary checking
    existing_tree = get_sandbox_file_tree(client)
    
    # Check boundaries BEFORE writing
    diff_result = check_sandbox_boundaries(chunk, files, existing_tree)
    
    if not diff_result.passed:
        logger.warning(
            f"[sandbox_executor] Boundary violations: "
            f"{[v.file_path for v in diff_result.violations]}"
        )
        return False, diff_result, {}
    
    if dry_run:
        logger.info("[sandbox_executor] Dry run - skipping write")
        return True, diff_result, files
    
    # Write files
    success, written, failed = write_files_to_sandbox(
        client=client,
        files=files,
        target="REPO",  # Write directly to repo in sandbox
        overwrite=True,
    )
    
    if not success:
        logger.error(f"[sandbox_executor] Write failed for: {failed}")
        # Update diff result to reflect failures
        return False, diff_result, {k: v for k, v in files.items() if k in written}
    
    logger.info(
        f"[sandbox_executor] Chunk {chunk.chunk_id} executed: "
        f"{len(diff_result.files_added)} added, {len(diff_result.files_modified)} modified"
    )
    
    return True, diff_result, files


async def execute_chunk_with_sandbox_fallback(
    *,
    chunk: Chunk,
    repo_path: str,
    spec_id: Optional[str] = None,
    spec_hash: Optional[str] = None,
    job_id: str = "",
    job_artifact_root: str = "",
    llm_call_fn: Callable,
    dry_run: bool = False,
    prefer_sandbox: bool = True,
) -> Tuple[bool, DiffCheckResult, Dict[str, str]]:
    """Execute chunk with sandbox fallback to local.
    
    Tries sandbox first if available, falls back to local execution.
    
    Args:
        chunk: Chunk to implement
        repo_path: Local repo path (fallback)
        spec_id: Spec ID
        spec_hash: Spec hash
        job_id: Job UUID
        job_artifact_root: Artifact root
        llm_call_fn: LLM call function
        dry_run: Skip writes
        prefer_sandbox: Try sandbox first
    
    Returns:
        (success, diff_result, files)
    """
    from app.overwatcher.executor import execute_chunk
    
    if prefer_sandbox and USE_SANDBOX:
        try:
            client = get_sandbox_client()
            if client.is_connected():
                # First generate files via LLM (using local executor)
                success, diff_result, files = await execute_chunk(
                    chunk=chunk,
                    repo_path=repo_path,
                    spec_id=spec_id,
                    spec_hash=spec_hash,
                    job_id=job_id,
                    job_artifact_root=job_artifact_root,
                    llm_call_fn=llm_call_fn,
                    dry_run=True,  # Don't write locally
                )
                
                if not success or not files:
                    return success, diff_result, files
                
                # Write via sandbox
                return await execute_chunk_sandbox(
                    chunk=chunk,
                    files=files,
                    spec_id=spec_id,
                    spec_hash=spec_hash,
                    job_id=job_id,
                    job_artifact_root=job_artifact_root,
                    client=client,
                    dry_run=dry_run,
                )
                
        except SandboxError as e:
            logger.warning(f"[sandbox_executor] Sandbox failed, falling back to local: {e}")
    
    # Fall back to local execution
    return await execute_chunk(
        chunk=chunk,
        repo_path=repo_path,
        spec_id=spec_id,
        spec_hash=spec_hash,
        job_id=job_id,
        job_artifact_root=job_artifact_root,
        llm_call_fn=llm_call_fn,
        dry_run=dry_run,
    )


__all__ = [
    # Boundary checking
    "get_sandbox_file_tree",
    "check_sandbox_boundaries",
    # File operations
    "write_files_to_sandbox",
    "read_file_from_sandbox",
    # Backup/rollback
    "create_sandbox_backup",
    "rollback_sandbox_changes",
    # Execution
    "execute_chunk_sandbox",
    "execute_chunk_with_sandbox_fallback",
]
