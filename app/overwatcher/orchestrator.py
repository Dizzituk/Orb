# FILE: app/overwatcher/orchestrator.py
"""Overwatcher Orchestrator: Full pipeline controller.

Enforces:
1. Fail-fast hash verification BEFORE any artifact storage
2. Spec hash echo in ALL stages (including revision)
3. Rollback on boundary violation OR verification failure
4. Three-strike error handling (Spec §9.4)
5. Proper ledger events at each step

Pipeline flow:
  Spec Gate → [hash lock] → Architecture → [hash verify] → Critique Loop →
  [hash verify each revision] → Chunk Plan → Execute Loop →
  [boundary check + verify + rollback] → Quarantine → Delete → Replay Pack

Strike rules (Spec §9.4):
- Strike 1: Internal knowledge only
- Strike 2 (same ErrorSignature): Deep Research allowed
- Strike 3 (same ErrorSignature): HARD STOP, quarantine
- Different ErrorSignature resets strikes to 1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
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
from app.overwatcher._orchestrator_utils import BoundaryViolationError, HashVerificationError, StrikeState, StrikeThreeError, VerificationFailedError, build_hash_header, run_implementation_loop, verify_stage_hash

logger = logging.getLogger(__name__)

# Block 9 integration - verified execution loop with Overwatcher
try:
    from app.overwatcher.block9_loop import (
        Block9State,
        run_chunk_block9,
        run_chunks_block9,
    )
    BLOCK9_AVAILABLE = True
except ImportError:
    BLOCK9_AVAILABLE = False
    logger.warning("[orchestrator] block9_loop not available, using legacy strike loop")


# =============================================================================
# Pipeline State
# =============================================================================

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


# =============================================================================
# Exceptions
# =============================================================================


# =============================================================================
# Strike Tracker (Spec §9.4)
# =============================================================================


# =============================================================================
# Hash Verification (Fail-Fast)
# =============================================================================


# =============================================================================
# Build System Prompt with Spec Hash (for revision stages)
# =============================================================================


# =============================================================================
# Chunk Execution with Rollback
# =============================================================================

async def execute_chunk_with_rollback(
    *,
    chunk: Chunk,
    repo_path: str,
    state: PipelineState,
    job_artifact_root: str,
    llm_call_fn: Callable,
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
            spec_id=state.spec_id,
            spec_hash=state.spec_hash,
            job_id=state.job_id,
            job_artifact_root=job_artifact_root,
            llm_call_fn=llm_call_fn,
            dry_run=False,
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
            model="claude-sonnet-4-5-20250514",
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
# Three-Strike Loop (Spec §9.4)
# =============================================================================

async def run_chunk_with_strikes(
    *,
    chunk: Chunk,
    repo_path: str,
    state: PipelineState,
    job_artifact_root: str,
    llm_call_fn: Callable,
) -> Tuple[bool, Optional[VerificationResult]]:
    """Run chunk implementation with three-strike error handling.
    
    Spec v2.3 §9.4:
    - Strike 1: Internal knowledge only
    - Strike 2 (same error): Deep Research allowed
    - Strike 3 (same error): HARD STOP, quarantine
    
    Args:
        chunk: Chunk to implement
        repo_path: Path to repository
        state: Pipeline state
        job_artifact_root: Root for artifacts
        llm_call_fn: Async function to call LLM
    
    Returns:
        (success, final_verification_result)
    
    Raises:
        StrikeThreeError: If chunk exhausts all three strikes
    """
    from app.overwatcher.error_signature import compute_error_signature, ErrorSignature
    from app.overwatcher.deep_research import run_deep_research
    from app.pot_spec.ledger_overwatcher import emit_strike_recorded, emit_chunk_abandoned
    
    strike_state = StrikeState(chunk_id=chunk.chunk_id)
    last_result: Optional[VerificationResult] = None
    deep_research_context: Optional[str] = None
    
    while not strike_state.is_exhausted():
        current_strike = strike_state.strike_count + 1
        logger.info(f"[orchestrator] Chunk {chunk.chunk_id} attempt {current_strike}/3")
        
        try:
            result = await execute_chunk_with_rollback(
                chunk=chunk,
                repo_path=repo_path,
                state=state,
                job_artifact_root=job_artifact_root,
                llm_call_fn=llm_call_fn,
            )
            
            # Success!
            logger.info(f"[orchestrator] Chunk {chunk.chunk_id} PASSED on strike {current_strike}")
            return True, result
            
        except (BoundaryViolationError, VerificationFailedError) as e:
            last_result = e.result if isinstance(e, VerificationFailedError) else None
            
            # Compute error signature
            if isinstance(e, VerificationFailedError) and last_result:
                error_text = last_result.command_results[0].stderr if last_result.command_results else str(e)
            else:
                error_text = str(e)
            
            new_signature = compute_error_signature(error_text)
            diagnosis = str(e)
            
            strike_count = strike_state.record_strike(new_signature, diagnosis)
            
            # Emit strike recorded
            try:
                emit_strike_recorded(
                    job_artifact_root=job_artifact_root,
                    job_id=state.job_id,
                    chunk_id=chunk.chunk_id,
                    strike_number=strike_count,
                    reason=diagnosis[:200],
                )
            except Exception as emit_err:
                logger.warning(f"[orchestrator] Failed to emit strike: {emit_err}")
            
            logger.warning(f"[orchestrator] Chunk {chunk.chunk_id} Strike {strike_count}: {diagnosis[:100]}")
            
            # Strike 2: Run Deep Research if same error
            if strike_count == 2 and strike_state.current_signature:
                logger.info(f"[orchestrator] Strike 2 - running Deep Research")
                
                try:
                    research_result = await run_deep_research(
                        error_signature=strike_state.current_signature,
                        stack_trace=error_text,
                        context=chunk.objective,
                        job_id=state.job_id,
                        chunk_id=chunk.chunk_id,
                        job_artifact_root=job_artifact_root,
                        llm_call_fn=llm_call_fn,
                    )
                    
                    deep_research_context = research_result.to_context_string()
                    logger.info(f"[orchestrator] Deep Research complete: {research_result.likely_cause[:100]}")
                except Exception as research_err:
                    logger.warning(f"[orchestrator] Deep Research failed: {research_err}")
    
    # Strike 3 - HARD STOP
    logger.error(f"[orchestrator] Chunk {chunk.chunk_id} exhausted all strikes")
    
    try:
        emit_chunk_abandoned(
            job_artifact_root=job_artifact_root,
            job_id=state.job_id,
            chunk_id=chunk.chunk_id,
            strike_count=strike_state.strike_count,
            final_error=str(strike_state.strike_history[-1]) if strike_state.strike_history else "Unknown",
        )
    except Exception as emit_err:
        logger.warning(f"[orchestrator] Failed to emit chunk abandoned: {emit_err}")
    
    chunk.status = ChunkStatus.QUARANTINED
    raise StrikeThreeError(chunk.chunk_id, strike_state.current_signature)


# =============================================================================
# Full Implementation Loop
# =============================================================================


__all__ = [
    # State
    "PipelineState",
    "StrikeState",
    # Exceptions
    "HashVerificationError",
    "BoundaryViolationError",
    "VerificationFailedError",
    "StrikeThreeError",
    # Hash verification
    "verify_stage_hash",
    "build_hash_header",
    # Execution
    "execute_chunk_with_rollback",
    "run_chunk_with_strikes",
    "run_implementation_loop",
]
