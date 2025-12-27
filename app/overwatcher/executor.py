# FILE: app/overwatcher/executor.py
"""Block 8: Implementation Executor with Diff Boundary Enforcement.

Sonnet applies a chunk; Overwatcher rejects changes outside allowed files.

Key behaviors:
- Execute chunk implementation via Sonnet
- Check diff against allowed_files boundaries
- Reject violations before applying
- Support rollback on failure
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from app.overwatcher.schemas import (
    Chunk,
    ChunkStatus,
    BoundaryViolation,
    DiffCheckResult,
    FileAction,
)

logger = logging.getLogger(__name__)

# Configuration
SONNET_MODEL = os.getenv("ORB_SONNET_MODEL", "claude-sonnet-4-20250514")
SANDBOX_ROOT = os.getenv("ORB_SANDBOX_ROOT", "D:\\SandboxOrb")


# =============================================================================
# Diff Boundary Checking
# =============================================================================

def normalize_path(path: str, base_dir: str = "") -> str:
    """Normalize path for comparison."""
    p = path.replace("\\", "/").strip()
    if base_dir:
        base = base_dir.replace("\\", "/").rstrip("/")
        if p.startswith(base):
            p = p[len(base):].lstrip("/")
    return p


def check_diff_boundaries(
    chunk: Chunk,
    files_added: List[str],
    files_modified: List[str],
    files_deleted: List[str],
    base_dir: str = "",
) -> DiffCheckResult:
    """Check if file changes respect chunk boundaries.
    
    Args:
        chunk: The chunk being implemented
        files_added: Paths of files added
        files_modified: Paths of files modified
        files_deleted: Paths of files deleted
        base_dir: Base directory to normalize paths against
    
    Returns:
        DiffCheckResult with any violations
    """
    violations = []
    
    # Normalize allowed paths
    allowed_add = {normalize_path(p, base_dir) for p in chunk.allowed_files.get("add", [])}
    allowed_modify = {normalize_path(p, base_dir) for p in chunk.allowed_files.get("modify", [])}
    allowed_delete = {normalize_path(p, base_dir) for p in chunk.allowed_files.get("delete_candidates", [])}
    
    # Check added files
    normalized_added = []
    for f in files_added:
        nf = normalize_path(f, base_dir)
        normalized_added.append(nf)
        if nf not in allowed_add:
            violations.append(BoundaryViolation(
                file_path=f,
                action="added",
                reason=f"File not in allowed_files.add list",
            ))
    
    # Check modified files
    normalized_modified = []
    for f in files_modified:
        nf = normalize_path(f, base_dir)
        normalized_modified.append(nf)
        if nf not in allowed_modify:
            violations.append(BoundaryViolation(
                file_path=f,
                action="modified",
                reason=f"File not in allowed_files.modify list",
            ))
    
    # Check deleted files
    normalized_deleted = []
    for f in files_deleted:
        nf = normalize_path(f, base_dir)
        normalized_deleted.append(nf)
        if nf not in allowed_delete:
            violations.append(BoundaryViolation(
                file_path=f,
                action="deleted",
                reason=f"File not in allowed_files.delete_candidates list",
            ))
    
    return DiffCheckResult(
        passed=len(violations) == 0,
        violations=violations,
        files_added=normalized_added,
        files_modified=normalized_modified,
        files_deleted=normalized_deleted,
    )


def parse_git_diff_stat(diff_output: str) -> Tuple[List[str], List[str], List[str]]:
    """Parse git diff --stat output to get file lists.
    
    Returns (added, modified, deleted)
    """
    added = []
    modified = []
    deleted = []
    
    for line in diff_output.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("create mode"):
            continue
        
        # Parse status lines like "A  path/to/file.py" or "M  path/to/file.py"
        if line.startswith("A\t") or line.startswith("A "):
            added.append(line[2:].strip())
        elif line.startswith("M\t") or line.startswith("M "):
            modified.append(line[2:].strip())
        elif line.startswith("D\t") or line.startswith("D "):
            deleted.append(line[2:].strip())
    
    return added, modified, deleted


def get_git_changes(repo_path: str, base_ref: str = "HEAD") -> Tuple[List[str], List[str], List[str]]:
    """Get file changes from git.
    
    Returns (added, modified, deleted)
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-status", base_ref],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode != 0:
            logger.warning(f"[executor] git diff failed: {result.stderr}")
            return [], [], []
        
        return parse_git_diff_stat(result.stdout)
        
    except Exception as e:
        logger.warning(f"[executor] Failed to get git changes: {e}")
        return [], [], []


def get_working_tree_changes(repo_path: str) -> Tuple[List[str], List[str], List[str]]:
    """Get uncommitted changes from working tree.
    
    Returns (added, modified, deleted)
    """
    try:
        # Get staged changes
        staged = subprocess.run(
            ["git", "diff", "--name-status", "--cached"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # Get unstaged changes
        unstaged = subprocess.run(
            ["git", "diff", "--name-status"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # Get untracked files
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        added = []
        modified = []
        deleted = []
        
        # Parse staged
        a, m, d = parse_git_diff_stat(staged.stdout)
        added.extend(a)
        modified.extend(m)
        deleted.extend(d)
        
        # Parse unstaged
        a, m, d = parse_git_diff_stat(unstaged.stdout)
        added.extend(a)
        modified.extend(m)
        deleted.extend(d)
        
        # Add untracked as added
        for line in untracked.stdout.strip().split("\n"):
            if line.strip():
                added.append(line.strip())
        
        # Deduplicate
        return list(set(added)), list(set(modified)), list(set(deleted))
        
    except Exception as e:
        logger.warning(f"[executor] Failed to get working tree changes: {e}")
        return [], [], []


# =============================================================================
# Implementation Prompt
# =============================================================================

IMPLEMENTATION_PROMPT = """You are implementing a specific chunk of an architecture.

CHUNK DETAILS:
- ID: {chunk_id}
- Title: {title}
- Objective: {objective}

ALLOWED FILES:
- Add: {allowed_add}
- Modify: {allowed_modify}
- Delete candidates: {allowed_delete}

STEPS:
{steps_text}

IMPORTANT RULES:
1. You may ONLY touch files listed in ALLOWED FILES
2. For modified files, output the COMPLETE new file content
3. Use clear file markers: ```python:path/to/file.py
4. Do not touch any files outside the allowed list
5. After implementation, all verification commands should pass

VERIFICATION COMMANDS:
{verification_commands}

Implement this chunk. For each file, provide the complete content."""


def build_implementation_prompt(chunk: Chunk) -> str:
    """Build prompt for Sonnet to implement chunk."""
    steps_text = ""
    for i, step in enumerate(chunk.steps, 1):
        steps_text += f"{i}. [{step.action.value}] {step.file_path}: {step.description}\n"
        if step.details:
            steps_text += f"   Details: {step.details}\n"
    
    verification_commands = "\n".join(chunk.verification.commands) if chunk.verification.commands else "None specified"
    
    return IMPLEMENTATION_PROMPT.format(
        chunk_id=chunk.chunk_id,
        title=chunk.title,
        objective=chunk.objective,
        allowed_add=", ".join(chunk.allowed_files.get("add", [])) or "None",
        allowed_modify=", ".join(chunk.allowed_files.get("modify", [])) or "None",
        allowed_delete=", ".join(chunk.allowed_files.get("delete_candidates", [])) or "None",
        steps_text=steps_text or "No specific steps defined",
        verification_commands=verification_commands,
    )


# =============================================================================
# File Extraction from LLM Output
# =============================================================================

def extract_files_from_output(output: str) -> Dict[str, str]:
    """Extract file contents from LLM output.
    
    Looks for patterns like:
    ```python:path/to/file.py
    content
    ```
    
    Returns dict of path -> content
    """
    files = {}
    
    # Pattern: ```language:path\ncontent\n```
    pattern = r"```(?:\w+)?:([^\n]+)\n([\s\S]*?)```"
    matches = re.findall(pattern, output)
    
    for path, content in matches:
        path = path.strip()
        files[path] = content
    
    # Also try simpler pattern: # FILE: path\n```\ncontent\n```
    pattern2 = r"#\s*FILE:\s*([^\n]+)\n```(?:\w+)?\n([\s\S]*?)```"
    matches2 = re.findall(pattern2, output)
    
    for path, content in matches2:
        path = path.strip()
        if path not in files:
            files[path] = content
    
    return files


# =============================================================================
# Chunk Execution
# =============================================================================

async def execute_chunk(
    chunk: Chunk,
    repo_path: str,
    llm_call_fn,
    provider_id: str = "anthropic",
    model_id: str = None,
    dry_run: bool = False,
) -> Tuple[bool, DiffCheckResult, Dict[str, str]]:
    """Execute a chunk implementation.
    
    Args:
        chunk: Chunk to implement
        repo_path: Path to repository
        llm_call_fn: Async function to call LLM
        provider_id: LLM provider
        model_id: LLM model (defaults to SONNET_MODEL)
        dry_run: If True, don't write files
    
    Returns:
        (success, diff_result, file_contents)
    """
    model_id = model_id or SONNET_MODEL
    
    logger.info(f"[executor] Executing chunk {chunk.chunk_id}: {chunk.title}")
    
    # Build implementation prompt
    prompt = build_implementation_prompt(chunk)
    
    messages = [
        {"role": "system", "content": "You are an expert programmer implementing a specific chunk. Follow the file permissions exactly."},
        {"role": "user", "content": prompt},
    ]
    
    try:
        result = await llm_call_fn(
            provider_id=provider_id,
            model_id=model_id,
            messages=messages,
        )
        
        raw_output = result.content if hasattr(result, "content") else str(result)
        
        # Extract files from output
        files = extract_files_from_output(raw_output)
        
        if not files:
            logger.warning(f"[executor] No files extracted from output")
            return False, DiffCheckResult(passed=False, violations=[]), {}
        
        # Determine which files are added vs modified
        files_added = []
        files_modified = []
        
        for path in files.keys():
            full_path = Path(repo_path) / path
            if full_path.exists():
                files_modified.append(path)
            else:
                files_added.append(path)
        
        # Check boundaries BEFORE writing
        diff_result = check_diff_boundaries(
            chunk=chunk,
            files_added=files_added,
            files_modified=files_modified,
            files_deleted=[],  # Sonnet doesn't delete via output
            base_dir=repo_path,
        )
        
        if not diff_result.passed:
            logger.warning(f"[executor] Boundary violations: {[v.file_path for v in diff_result.violations]}")
            return False, diff_result, files
        
        # Write files if not dry run
        if not dry_run:
            for path, content in files.items():
                full_path = Path(repo_path) / path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content, encoding="utf-8")
                logger.info(f"[executor] Wrote: {path}")
        
        return True, diff_result, files
        
    except Exception as e:
        logger.error(f"[executor] Chunk execution failed: {e}")
        return False, DiffCheckResult(passed=False, violations=[]), {}


# =============================================================================
# Rollback Support
# =============================================================================

def create_backup(
    chunk: Chunk,
    repo_path: str,
    backup_dir: str,
) -> Dict[str, str]:
    """Create backup of files that will be modified.
    
    Returns dict of path -> original content
    """
    backups = {}
    
    for path in chunk.allowed_files.get("modify", []):
        full_path = Path(repo_path) / path
        if full_path.exists():
            try:
                backups[path] = full_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning(f"[executor] Failed to backup {path}: {e}")
    
    # Also store in backup directory
    if backup_dir:
        backup_path = Path(backup_dir) / chunk.chunk_id
        backup_path.mkdir(parents=True, exist_ok=True)
        
        for path, content in backups.items():
            bp = backup_path / path.replace("/", "_").replace("\\", "_")
            bp.write_text(content, encoding="utf-8")
    
    return backups


def rollback_chunk(
    chunk: Chunk,
    repo_path: str,
    backups: Dict[str, str],
    files_added: List[str],
) -> bool:
    """Rollback a chunk implementation.
    
    Args:
        chunk: The chunk that was implemented
        repo_path: Path to repository
        backups: Original content of modified files
        files_added: Files that were added (to delete)
    
    Returns:
        True if rollback succeeded
    """
    success = True
    
    # Restore modified files
    for path, content in backups.items():
        try:
            full_path = Path(repo_path) / path
            full_path.write_text(content, encoding="utf-8")
            logger.info(f"[executor] Restored: {path}")
        except Exception as e:
            logger.error(f"[executor] Failed to restore {path}: {e}")
            success = False
    
    # Delete added files
    for path in files_added:
        try:
            full_path = Path(repo_path) / path
            if full_path.exists():
                full_path.unlink()
                logger.info(f"[executor] Deleted: {path}")
        except Exception as e:
            logger.error(f"[executor] Failed to delete {path}: {e}")
            success = False
    
    return success


__all__ = [
    # Boundary checking
    "check_diff_boundaries",
    "get_git_changes",
    "get_working_tree_changes",
    # Execution
    "build_implementation_prompt",
    "extract_files_from_output",
    "execute_chunk",
    # Rollback
    "create_backup",
    "rollback_chunk",
]
