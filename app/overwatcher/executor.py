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
from app.overwatcher._executor_utils import CODE_BLOCK_PATTERN, FILE_HEADER_PATTERN, IMPLEMENTER_MAX_OUTPUT_TOKENS, SANDBOX_ROOT, emit_boundary_violation, emit_chunk_implemented, emit_provider_fallback, emit_stage_failed
from app.overwatcher._executor_utils_1 import IMPLEMENTER_FALLBACK_MODEL, IMPLEMENTER_FALLBACK_PROVIDER, IMPLEMENTER_MODEL, IMPLEMENTER_PROVIDER, create_backup, get_git_changes, get_working_tree_changes, rollback_chunk

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration (Spec v2.3 §10 and §3.3)
# =============================================================================

# Block 8 uses Claude Sonnet for implementation

# Fallback per spec §3.3

# Sandbox configuration

# Implementation output limits


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
    for f in files_added:
        nf = normalize_path(f, base_dir)
        if nf not in allowed_add:
            violations.append(BoundaryViolation(
                file_path=f,
                violation_type="unauthorized_add",
                details="File not in allowed_files.add list",
            ))
    
    # Check modified files
    for f in files_modified:
        nf = normalize_path(f, base_dir)
        if nf not in allowed_modify:
            violations.append(BoundaryViolation(
                file_path=f,
                violation_type="unauthorized_modify",
                details="File not in allowed_files.modify list",
            ))
    
    # Check deleted files
    for f in files_deleted:
        nf = normalize_path(f, base_dir)
        if nf not in allowed_delete:
            violations.append(BoundaryViolation(
                file_path=f,
                violation_type="unauthorized_delete",
                details="File not in allowed_files.delete_candidates list",
            ))
    
    return DiffCheckResult(
        allowed=len(violations) == 0,
        violations=violations,
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


# =============================================================================
# File Extraction from LLM Output
# =============================================================================

# Pattern to match file headers like "# FILE: path/to/file.py"

# Pattern to extract code blocks


def extract_files_from_output(output: str) -> Dict[str, str]:
    """Extract files from LLM output.
    
    Expected format:
    # FILE: path/to/file.py
    ```python
    <content>
    ```
    
    Returns dict of path -> content
    """
    if not output:
        return {}
    
    files = {}
    
    # Find all file headers
    headers = list(FILE_HEADER_PATTERN.finditer(output))
    
    for i, match in enumerate(headers):
        file_path = match.group(1).strip()
        
        # Get content between this header and next (or end)
        start = match.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(output)
        section = output[start:end]
        
        # Extract code block content
        code_match = CODE_BLOCK_PATTERN.search(section)
        if code_match:
            content = code_match.group(1)
            # Remove trailing newline if present
            if content.endswith('\n'):
                content = content[:-1]
            files[file_path] = content
    
    return files


def parse_spec_headers(output: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse SPEC_ID and SPEC_HASH from LLM output.
    
    Returns (spec_id, spec_hash)
    """
    if not output:
        return None, None
    
    spec_id = None
    spec_hash = None
    
    # Look for SPEC_ID: <value>
    match = re.search(r'SPEC_ID:\s*(\S+)', output)
    if match:
        spec_id = match.group(1)
    
    # Look for SPEC_HASH: <value>
    match = re.search(r'SPEC_HASH:\s*(\S+)', output)
    if match:
        spec_hash = match.group(1)
    
    return spec_id, spec_hash


# =============================================================================
# Implementation Prompt Building
# =============================================================================

def build_implementation_prompt(
    chunk: Chunk,
    spec_id: str,
    spec_hash: str,
    fix_actions_context: Optional[str] = None,
) -> Tuple[str, str]:
    """Build the implementation prompt for a chunk.
    
    Returns (system_prompt, user_prompt)
    """
    # System prompt with spec echo requirement
    system_prompt = f"""You are an expert code implementer. Your task is to implement code changes according to specifications.

IMPORTANT: You MUST include these exact headers at the start of your response:
SPEC_ID: {spec_id}
SPEC_HASH: {spec_hash}

For each file you create or modify, use this format:
# FILE: path/to/file.py
```python
<complete file content>
```

Rules:
1. Output COMPLETE file contents, not patches
2. Only modify files listed in allowed_files
3. Follow the verification commands to ensure correctness
4. Include all necessary imports and dependencies
"""
    
    # User prompt with chunk details
    allowed_files_str = json.dumps(chunk.allowed_files, indent=2)
    
    steps_str = ""
    for step in chunk.steps:
        steps_str += f"\n- {step.description}"
        if step.details:
            steps_str += f"\n  Details: {step.details}"
    
    verification_str = "\n".join(f"- {cmd}" for cmd in chunk.verification.commands)
    
    user_prompt = f"""Implement the following chunk:

Chunk ID: {chunk.chunk_id}
Title: {chunk.title}
Objective: {chunk.objective}

Allowed Files:
{allowed_files_str}

Steps:{steps_str}

Verification Commands:
{verification_str}

Please implement all required changes. Output complete file contents for each file.
"""
    
    # Inject Overwatcher FIX_ACTIONS if provided (from previous failure diagnosis)
    if fix_actions_context:
        user_prompt = f"""IMPORTANT - OVERWATCHER GUIDANCE FROM PREVIOUS FAILURE:
{fix_actions_context}

---

{user_prompt}"""
    
    return system_prompt, user_prompt


# =============================================================================
# Ledger Event Emission (stubs for event system)
# =============================================================================


# =============================================================================
# Main Execution
# =============================================================================

async def execute_chunk(
    *,
    chunk: Chunk,
    repo_path: str,
    spec_id: Optional[str] = None,
    spec_hash: Optional[str] = None,
    job_id: str = "",
    job_artifact_root: str = "",
    llm_call_fn: Callable,
    dry_run: bool = False,
    provider_id: str = IMPLEMENTER_PROVIDER,
    model_id: str = IMPLEMENTER_MODEL,
    fix_actions_context: Optional[str] = None,
) -> Tuple[bool, DiffCheckResult, Dict[str, str]]:
    """Execute implementation for a single chunk.
    
    Args:
        chunk: The chunk to implement
        repo_path: Path to repository root
        spec_id: Spec ID for validation
        spec_hash: Spec hash for validation  
        job_id: Job UUID for tracking
        job_artifact_root: Root directory for artifacts
        llm_call_fn: Async function to call LLM
        dry_run: If True, check boundaries but don't write files
        provider_id: LLM provider to use
        model_id: LLM model to use
    
    Returns:
        (success, diff_result, files_dict)
    """
    logger.info(f"[executor] Executing chunk {chunk.chunk_id}")
    
    # Build implementation prompt (includes Overwatcher FIX_ACTIONS if retrying)
    system_prompt, user_prompt = build_implementation_prompt(
        chunk, spec_id, spec_hash, fix_actions_context
    )
    
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
        return False, DiffCheckResult(allowed=False, violations=[]), {}
    
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
        return False, DiffCheckResult(allowed=False, violations=[]), {}
    
    # Extract files from output
    files = extract_files_from_output(raw_output)
    
    if not files:
        logger.warning(f"[executor] No files extracted from output")
        return False, DiffCheckResult(allowed=False, violations=[]), {}
    
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
    
    if not diff_result.allowed:
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
