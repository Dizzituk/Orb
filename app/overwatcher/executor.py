# FILE: app/overwatcher/executor.py
"""Block 8: Implementation Executor with Diff Boundary Enforcement.

Spec v2.3 Block 8:
- Model: Claude Sonnet (claude-sonnet-4-5-20250514)
- Fallback: GPT-5.2 Thinking (gpt-5.2-thinking)
- Purpose: Generate code for each chunk (full files only for touched files)

Key behaviors:
- Execute chunk implementation via Sonnet
- Check diff against allowed_files boundaries
- Reject violations before applying
- Support rollback on failure
- Emit proper ledger events
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from app.overwatcher.schemas import (
    Chunk,
    ChunkStatus,
    BoundaryViolation,
    DiffCheckResult,
    FileAction,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration (Spec v2.3 §10 and §3.3)
# =============================================================================

# Block 8 uses Claude Sonnet for implementation
IMPLEMENTER_PROVIDER = os.getenv("ORB_IMPLEMENTER_PROVIDER", "anthropic")
IMPLEMENTER_MODEL = os.getenv("ORB_IMPLEMENTER_MODEL", "claude-sonnet-4-5-20250514")

# Fallback per spec §3.3
IMPLEMENTER_FALLBACK_PROVIDER = os.getenv("ORB_IMPLEMENTER_FALLBACK_PROVIDER", "openai")
IMPLEMENTER_FALLBACK_MODEL = os.getenv("ORB_IMPLEMENTER_FALLBACK_MODEL", "gpt-5.2-thinking")

# Sandbox configuration
SANDBOX_ROOT = os.getenv("ORB_SANDBOX_ROOT", "D:\\SandboxOrb")

# Implementation output limits
IMPLEMENTER_MAX_OUTPUT_TOKENS = int(os.getenv("ORB_IMPLEMENTER_MAX_OUTPUT_TOKENS", "16000"))


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
    
    Spec v2.3: Boundary violations trigger rollback.
    
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
    """Parse git diff --name-status output to get file lists.
    
    Returns (added, modified, deleted)
    """
    added = []
    modified = []
    deleted = []
    
    for line in diff_output.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("create mode"):
            continue
        
        # Parse status lines like "A\tpath/to/file.py" or "M\tpath/to/file.py"
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

IMPLEMENTATION_SYSTEM = """You are an expert programmer implementing a specific chunk of an approved architecture.

CRITICAL RULES:
1. You may ONLY touch files listed in ALLOWED FILES
2. For each file, output the COMPLETE file content
3. Use markers: # FILE: path/to/file.py followed by code block
4. Do not touch any files outside the allowed list
5. Verify your implementation will pass the listed verification commands

You must echo these exact lines at the start of your response:
SPEC_ID: {spec_id}
SPEC_HASH: {spec_hash}

Then provide the implementation."""

IMPLEMENTATION_PROMPT = """Implement this chunk:

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

VERIFICATION COMMANDS (must pass after implementation):
{verification_commands}

OUTPUT FORMAT:
For each file, use this exact format:

# FILE: path/to/file.py
```python
<complete file content>
```

Implement this chunk now. Provide complete file contents for all touched files."""


def build_implementation_prompt(
    chunk: Chunk,
    spec_id: str,
    spec_hash: str,
) -> Tuple[str, str]:
    """Build system and user prompts for implementation.
    
    Returns:
        (system_prompt, user_prompt)
    """
    steps_text = ""
    for i, step in enumerate(chunk.steps, 1):
        action = step.action.value if isinstance(step.action, FileAction) else step.action
        steps_text += f"{i}. [{action}] {step.file_path}: {step.description}\n"
        if step.details:
            steps_text += f"   Details: {step.details}\n"
    
    verification_commands = "\n".join(chunk.verification.commands) if chunk.verification.commands else "None specified"
    
    system = IMPLEMENTATION_SYSTEM.format(
        spec_id=spec_id,
        spec_hash=spec_hash,
    )
    
    user = IMPLEMENTATION_PROMPT.format(
        chunk_id=chunk.chunk_id,
        title=chunk.title,
        objective=chunk.objective,
        allowed_add=", ".join(chunk.allowed_files.get("add", [])) or "None",
        allowed_modify=", ".join(chunk.allowed_files.get("modify", [])) or "None",
        allowed_delete=", ".join(chunk.allowed_files.get("delete_candidates", [])) or "None",
        steps_text=steps_text or "No specific steps defined",
        verification_commands=verification_commands,
    )
    
    return system, user


# =============================================================================
# File Extraction from LLM Output
# =============================================================================

def extract_files_from_output(output: str) -> Dict[str, str]:
    """Extract file contents from LLM output.
    
    Looks for patterns like:
    # FILE: path/to/file.py
    ```python
    content
    ```
    
    Returns dict of path -> content
    """
    files = {}
    
    # Pattern: # FILE: path\n```language\ncontent\n```
    pattern = r"#\s*FILE:\s*([^\n]+)\n```(?:\w+)?\n([\s\S]*?)```"
    matches = re.findall(pattern, output)
    
    for path, content in matches:
        path = path.strip()
        files[path] = content.rstrip()
    
    # Also try: ```language:path\ncontent\n```
    pattern2 = r"```(?:\w+)?:([^\n]+)\n([\s\S]*?)```"
    matches2 = re.findall(pattern2, output)
    
    for path, content in matches2:
        path = path.strip()
        if path not in files:
            files[path] = content.rstrip()
    
    return files


def parse_spec_headers(output: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse SPEC_ID and SPEC_HASH headers from output.
    
    Returns:
        (spec_id, spec_hash)
    """
    spec_id = None
    spec_hash = None
    
    lines = output.strip().split("\n") if output else []
    for line in lines[:5]:
        if line.strip().startswith("SPEC_ID:"):
            spec_id = line.split(":", 1)[1].strip()
        elif line.strip().startswith("SPEC_HASH:"):
            spec_hash = line.split(":", 1)[1].strip()
    
    return spec_id, spec_hash


# =============================================================================
# Chunk Execution
# =============================================================================

async def execute_chunk(
    *,
    chunk: Chunk,
    repo_path: str,
    spec_id: str,
    spec_hash: str,
    job_id: str,
    job_artifact_root: str,
    llm_call_fn: Callable,
    provider_id: str = None,
    model_id: str = None,
    dry_run: bool = False,
) -> Tuple[bool, DiffCheckResult, Dict[str, str]]:
    """Execute a chunk implementation.
    
    Spec v2.3 Block 8:
    - Primary: Claude Sonnet
    - Fallback: GPT-5.2 Thinking
    - Emits CHUNK_IMPLEMENTED or BOUNDARY_VIOLATION
    
    Args:
        chunk: Chunk to implement
        repo_path: Path to repository
        spec_id: Spec ID for header echo
        spec_hash: Spec hash for header echo
        job_id: Job UUID
        job_artifact_root: Root for artifacts
        llm_call_fn: Async function to call LLM
        provider_id: LLM provider (default: anthropic)
        model_id: LLM model (default: claude-sonnet)
        dry_run: If True, don't write files
    
    Returns:
        (success, diff_result, file_contents)
    """
    from app.pot_spec.ledger import (
        emit_chunk_implemented,
        emit_boundary_violation,
        emit_stage_started,
        emit_provider_fallback,
        emit_stage_failed,
    )
    
    stage_run_id = str(uuid4())
    
    # Default to spec-mandated models
    provider_id = provider_id or IMPLEMENTER_PROVIDER
    model_id = model_id or IMPLEMENTER_MODEL
    
    logger.info(f"[executor] Executing chunk {chunk.chunk_id}: {chunk.title}")
    
    # Emit stage started
    try:
        emit_stage_started(
            job_artifact_root=job_artifact_root,
            job_id=job_id,
            stage_id="implementation",
            stage_run_id=stage_run_id,
        )
    except Exception as e:
        logger.warning(f"[executor] Failed to emit stage started: {e}")
    
    # Build implementation prompt
    system_prompt, user_prompt = build_implementation_prompt(chunk, spec_id, spec_hash)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    # Try primary model
    result = None
    used_fallback = False
    
    try:
        result = await llm_call_fn(
            provider_id=provider_id,
            model_id=model_id,
            messages=messages,
            max_tokens=IMPLEMENTER_MAX_OUTPUT_TOKENS,
        )
    except Exception as e:
        logger.warning(f"[executor] Primary model failed ({provider_id}/{model_id}): {e}")
        
        # Try fallback
        if IMPLEMENTER_FALLBACK_PROVIDER and IMPLEMENTER_FALLBACK_MODEL:
            try:
                emit_provider_fallback(
                    job_artifact_root=job_artifact_root,
                    job_id=job_id,
                    from_provider=provider_id,
                    from_model=model_id,
                    to_provider=IMPLEMENTER_FALLBACK_PROVIDER,
                    to_model=IMPLEMENTER_FALLBACK_MODEL,
                    reason=str(e),
                )
            except Exception:
                pass
            
            try:
                result = await llm_call_fn(
                    provider_id=IMPLEMENTER_FALLBACK_PROVIDER,
                    model_id=IMPLEMENTER_FALLBACK_MODEL,
                    messages=messages,
                    max_tokens=IMPLEMENTER_MAX_OUTPUT_TOKENS,
                )
                used_fallback = True
                provider_id = IMPLEMENTER_FALLBACK_PROVIDER
                model_id = IMPLEMENTER_FALLBACK_MODEL
            except Exception as e2:
                logger.error(f"[executor] Fallback model also failed: {e2}")
    
    if result is None:
        try:
            emit_stage_failed(
                job_artifact_root=job_artifact_root,
                job_id=job_id,
                stage_id="implementation",
                error_type="llm_call_failed",
                error_message="Both primary and fallback models failed",
            )
        except Exception:
            pass
        return False, DiffCheckResult(passed=False, violations=[]), {}
    
    raw_output = result.content if hasattr(result, "content") else str(result)
    
    # Verify spec hash
    returned_spec_id, returned_spec_hash = parse_spec_headers(raw_output)
    if returned_spec_hash and returned_spec_hash != spec_hash:
        logger.error(f"[executor] SPEC_HASH mismatch: expected {spec_hash[:16]}..., got {returned_spec_hash[:16]}...")
        try:
            emit_stage_failed(
                job_artifact_root=job_artifact_root,
                job_id=job_id,
                stage_id="implementation",
                error_type="spec_hash_mismatch",
                error_message=f"Expected {spec_hash}, got {returned_spec_hash}",
            )
        except Exception:
            pass
        return False, DiffCheckResult(passed=False, violations=[]), {}
    
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
        files_deleted=[],  # LLM doesn't delete via output
        base_dir=repo_path,
    )
    
    if not diff_result.passed:
        logger.warning(f"[executor] Boundary violations: {[v.file_path for v in diff_result.violations]}")
        try:
            emit_boundary_violation(
                job_artifact_root=job_artifact_root,
                job_id=job_id,
                chunk_id=chunk.chunk_id,
                violations=[v.to_dict() for v in diff_result.violations],
            )
        except Exception as e:
            logger.warning(f"[executor] Failed to emit boundary violation: {e}")
        return False, diff_result, files
    
    # Write files if not dry run
    if not dry_run:
        for path, content in files.items():
            full_path = Path(repo_path) / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            logger.info(f"[executor] Wrote: {path}")
    
    # Emit success event
    try:
        emit_chunk_implemented(
            job_artifact_root=job_artifact_root,
            job_id=job_id,
            chunk_id=chunk.chunk_id,
            files_added=files_added,
            files_modified=files_modified,
            model=model_id,
        )
    except Exception as e:
        logger.warning(f"[executor] Failed to emit chunk implemented: {e}")
    
    logger.info(f"[executor] Chunk {chunk.chunk_id} implemented: {len(files_added)} added, {len(files_modified)} modified (model={model_id}, fallback={used_fallback})")
    return True, diff_result, files


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
    # Configuration
    "IMPLEMENTER_PROVIDER",
    "IMPLEMENTER_MODEL",
    "IMPLEMENTER_FALLBACK_PROVIDER",
    "IMPLEMENTER_FALLBACK_MODEL",
    # Boundary checking
    "normalize_path",
    "check_diff_boundaries",
    "get_git_changes",
    "get_working_tree_changes",
    # Prompt building
    "build_implementation_prompt",
    # File extraction
    "extract_files_from_output",
    "parse_spec_headers",
    # Execution
    "execute_chunk",
    # Rollback
    "create_backup",
    "rollback_chunk",
]
