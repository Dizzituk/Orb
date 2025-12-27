# FILE: app/overwatcher/orchestrator.py
"""Overwatcher Orchestrator: Full pipeline controller.

Enforces:
1. Fail-fast hash verification BEFORE any artifact storage
2. Spec hash echo in ALL stages (including revision)
3. Rollback on boundary violation OR verification failure
4. Proper ledger events at each step

Pipeline flow:
  Spec Gate → [hash lock] → Architecture → [hash verify] → Critique Loop →
  [hash verify each revision] → Chunk Plan → Execute Loop →
  [boundary check + verify + rollback] → Quarantine → Delete → Replay Pack
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from app.pot_spec.canonical import compute_spec_hash, verify_hash
from app.overwatcher.schemas import (
    Chunk,
    ChunkPlan,
    ChunkStatus,
    DiffCheckResult,
    VerificationResult,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """Tracks state across pipeline stages."""
    job_id: str
    spec_id: Optional[str] = None
    spec_hash: Optional[str] = None
    spec_dict: Optional[Dict[str, Any]] = None
    arch_id: Optional[str] = None
    arch_version: int = 0
    arch_content: Optional[str] = None
    plan: Optional[ChunkPlan] = None
    current_chunk_idx: int = 0
    completed_chunks: List[str] = field(default_factory=list)
    failed_chunks: List[str] = field(default_factory=list)
    rollback_stack: List[Dict[str, Any]] = field(default_factory=list)


class HashVerificationError(Exception):
    """Raised when spec hash verification fails."""
    def __init__(self, expected: str, observed: Optional[str], stage: str):
        self.expected = expected
        self.observed = observed
        self.stage = stage
        super().__init__(f"Hash mismatch at {stage}: expected={expected}, observed={observed}")


class BoundaryViolationError(Exception):
    """Raised when diff boundary check fails."""
    def __init__(self, chunk_id: str, violations: List[Dict[str, Any]]):
        self.chunk_id = chunk_id
        self.violations = violations
        super().__init__(f"Boundary violation in {chunk_id}: {len(violations)} files")


class VerificationFailedError(Exception):
    """Raised when verification gate fails."""
    def __init__(self, chunk_id: str, result: VerificationResult):
        self.chunk_id = chunk_id
        self.result = result
        super().__init__(f"Verification failed for {chunk_id}")


# =============================================================================
# Hash Verification (Fail-Fast)
# =============================================================================

def verify_stage_hash(
    *,
    stage_name: str,
    spec_id: str,
    expected_hash: str,
    stage_output: str,
    job_id: str,
    job_artifact_root: str,
) -> Tuple[str, str]:
    """Verify spec hash from stage output BEFORE any artifact storage.
    
    This is fail-fast: raises immediately if hash doesn't match.
    Ledger event is emitted regardless of success/failure.
    
    Args:
        stage_name: Name of the stage
        spec_id: Expected spec ID
        expected_hash: Expected spec hash
        stage_output: Raw output from stage (with header)
        job_id: Job UUID
        job_artifact_root: Root path for artifacts
    
    Returns:
        (returned_spec_id, returned_hash) on success
    
    Raises:
        HashVerificationError: If hash doesn't match (BEFORE any storage)
    """
    from app.pot_spec.ledger import (
        emit_spec_hash_computed,
        emit_spec_hash_verified,
        emit_spec_hash_mismatch,
    )
    from app.jobs.stage3_locks import parse_spec_echo_headers
    
    # Emit that we're computing/checking
    emit_spec_hash_computed(
        job_artifact_root=job_artifact_root,
        job_id=job_id,
        stage_name=stage_name,
        spec_id=spec_id,
        expected_spec_hash=expected_hash,
    )
    
    # Parse headers
    returned_id, returned_hash, parse_note = parse_spec_echo_headers(stage_output)
    
    # Check for parse failure
    if parse_note != "ok" or not returned_hash:
        emit_spec_hash_mismatch(
            job_artifact_root=job_artifact_root,
            job_id=job_id,
            stage_name=stage_name,
            spec_id=spec_id,
            expected_spec_hash=expected_hash,
            observed_spec_hash=returned_hash,
            reason=parse_note,
        )
        raise HashVerificationError(expected_hash, returned_hash, stage_name)
    
    # Check hash match
    if returned_hash != expected_hash:
        emit_spec_hash_mismatch(
            job_artifact_root=job_artifact_root,
            job_id=job_id,
            stage_name=stage_name,
            spec_id=spec_id,
            expected_spec_hash=expected_hash,
            observed_spec_hash=returned_hash,
            reason="hash_mismatch",
        )
        raise HashVerificationError(expected_hash, returned_hash, stage_name)
    
    # Success - emit verified
    emit_spec_hash_verified(
        job_artifact_root=job_artifact_root,
        job_id=job_id,
        stage_name=stage_name,
        spec_id=spec_id,
        spec_hash=returned_hash,
    )
    
    return returned_id, returned_hash


# =============================================================================
# Chunk Execution with Rollback
# =============================================================================

async def execute_chunk_with_rollback(
    *,
    chunk: Chunk,
    repo_path: str,
    state: PipelineState,
    job_artifact_root: str,
    llm_call_fn,
) -> VerificationResult:
    """Execute a chunk with automatic rollback on failure.
    
    Rollback triggers:
    1. Boundary violation (files touched outside allowed list)
    2. Verification failure (tests/lint/types fail)
    
    Args:
        chunk: Chunk to execute
        repo_path: Path to repository
        state: Pipeline state for rollback tracking
        job_artifact_root: Root for artifacts
        llm_call_fn: Async LLM call function
    
    Returns:
        VerificationResult on success
    
    Raises:
        BoundaryViolationError: If boundary check fails (after rollback)
        VerificationFailedError: If verification fails (after rollback)
    """
    from app.overwatcher.executor import (
        create_backup,
        execute_chunk,
        rollback_chunk,
        check_diff_boundaries,
    )
    from app.overwatcher.verifier import verify_chunk
    from app.pot_spec.ledger import (
        emit_chunk_implemented,
        emit_boundary_violation,
        emit_verify_pass,
        emit_verify_fail,
    )
    
    logger.info(f"[orchestrator] Executing chunk {chunk.chunk_id}")
    
    # 1. Create backup BEFORE any changes
    backup_dir = str(Path(job_artifact_root) / "jobs" / state.job_id / "backups")
    backups = create_backup(chunk, repo_path, backup_dir)
    
    # Track for rollback
    rollback_info = {
        "chunk_id": chunk.chunk_id,
        "backups": backups,
        "files_added": [],
    }
    state.rollback_stack.append(rollback_info)
    
    try:
        # 2. Execute chunk (Sonnet generates code)
        success, diff_result, files = await execute_chunk(
            chunk=chunk,
            repo_path=repo_path,
            llm_call_fn=llm_call_fn,
            dry_run=False,  # Actually write files
        )
        
        rollback_info["files_added"] = diff_result.files_added
        
        # 3. Check boundary violations FIRST
        if not diff_result.passed:
            emit_boundary_violation(
                job_artifact_root=job_artifact_root,
                job_id=state.job_id,
                chunk_id=chunk.chunk_id,
                violations=[v.to_dict() for v in diff_result.violations],
            )
            # ROLLBACK
            rollback_chunk(chunk, repo_path, backups, diff_result.files_added)
            state.rollback_stack.pop()
            raise BoundaryViolationError(chunk.chunk_id, [v.to_dict() for v in diff_result.violations])
        
        # 4. Emit implementation event
        emit_chunk_implemented(
            job_artifact_root=job_artifact_root,
            job_id=state.job_id,
            chunk_id=chunk.chunk_id,
            files_added=diff_result.files_added,
            files_modified=diff_result.files_modified,
        )
        
        # 5. Run verification gate
        touched_files = diff_result.files_added + diff_result.files_modified
        verification = await verify_chunk(
            chunk=chunk,
            repo_path=repo_path,
            touched_files=touched_files,
            job_artifact_root=job_artifact_root,
        )
        
        # 6. Check verification result
        if verification.status != VerificationStatus.PASSED:
            emit_verify_fail(
                job_artifact_root=job_artifact_root,
                job_id=state.job_id,
                chunk_id=chunk.chunk_id,
                tests_failed=verification.tests_failed,
                lint_errors=verification.lint_errors,
                type_errors=verification.type_errors,
                failure_summary=f"Tests: {verification.tests_failed} failed, Lint: {verification.lint_errors}, Types: {verification.type_errors}",
                evidence_paths=verification.evidence_paths,
            )
            # ROLLBACK
            rollback_chunk(chunk, repo_path, backups, diff_result.files_added)
            state.rollback_stack.pop()
            raise VerificationFailedError(chunk.chunk_id, verification)
        
        # 7. Success!
        emit_verify_pass(
            job_artifact_root=job_artifact_root,
            job_id=state.job_id,
            chunk_id=chunk.chunk_id,
            tests_passed=verification.tests_passed,
            lint_errors=0,
            type_errors=0,
            evidence_paths=verification.evidence_paths,
        )
        
        # Remove from rollback stack (no longer needed)
        state.rollback_stack.pop()
        state.completed_chunks.append(chunk.chunk_id)
        chunk.status = ChunkStatus.VERIFIED
        
        return verification
        
    except (BoundaryViolationError, VerificationFailedError):
        state.failed_chunks.append(chunk.chunk_id)
        chunk.status = ChunkStatus.FAILED
        raise
    except Exception as e:
        # Unexpected error - rollback and re-raise
        logger.error(f"[orchestrator] Unexpected error in chunk {chunk.chunk_id}: {e}")
        if rollback_info in state.rollback_stack:
            rollback_chunk(chunk, repo_path, backups, rollback_info.get("files_added", []))
            state.rollback_stack.remove(rollback_info)
        state.failed_chunks.append(chunk.chunk_id)
        chunk.status = ChunkStatus.FAILED
        raise


# =============================================================================
# Build System Prompt with Spec Hash (for revision stages)
# =============================================================================

def build_hash_header(spec_id: str, spec_hash: str) -> str:
    """Build the 2-line header that must be echoed by LLM.
    
    Used for:
    - Architecture generation
    - Architecture revision (each iteration)
    - Chunk planning
    """
    return f"""You must echo these exact lines at the start of your response:
SPEC_ID: {spec_id}
SPEC_HASH: {spec_hash}

Then proceed with your response."""


__all__ = [
    "PipelineState",
    "HashVerificationError",
    "BoundaryViolationError",
    "VerificationFailedError",
    "verify_stage_hash",
    "execute_chunk_with_rollback",
    "build_hash_header",
]
