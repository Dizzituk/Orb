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

logger = logging.getLogger(__name__)


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


class StrikeThreeError(Exception):
    """Raised when chunk exhausts all three strikes."""
    def __init__(self, chunk_id: str, error_signature: Any):
        self.chunk_id = chunk_id
        self.error_signature = error_signature
        super().__init__(f"Chunk {chunk_id} exhausted 3 strikes")


# =============================================================================
# Strike Tracker (Spec §9.4)
# =============================================================================

@dataclass
class StrikeState:
    """Tracks strikes for a chunk.
    
    Spec §9.4: Same ErrorSignature accumulates strikes.
    Different ErrorSignature resets to 1.
    """
    
    chunk_id: str
    current_signature: Optional[Any] = None  # ErrorSignature
    strike_count: int = 0
    strike_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def record_strike(
        self,
        new_signature: Any,  # ErrorSignature
        diagnosis: str,
    ) -> int:
        """Record a new strike and return the new count.
        
        Returns:
            Current strike count (1, 2, or 3)
        """
        from app.overwatcher.error_signature import signatures_match
        
        # Check if same error
        if self.current_signature and signatures_match(self.current_signature, new_signature):
            # Same error - increment strike
            self.strike_count += 1
        else:
            # Different error - reset to 1
            self.strike_count = 1
            self.current_signature = new_signature
        
        # Record history
        self.strike_history.append({
            "strike_number": self.strike_count,
            "signature_hash": new_signature.signature_hash if new_signature else None,
            "diagnosis": diagnosis,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        return self.strike_count
    
    def is_exhausted(self) -> bool:
        """Check if chunk has exhausted all strikes."""
        return self.strike_count >= 3


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

async def run_implementation_loop(
    *,
    plan: ChunkPlan,
    repo_path: str,
    state: PipelineState,
    job_artifact_root: str,
    llm_call_fn: Callable,
    stop_on_failure: bool = True,
) -> Tuple[List[str], List[str]]:
    """Run the full implementation loop for all chunks.
    
    Args:
        plan: Chunk plan from Block 7
        repo_path: Path to repository
        state: Pipeline state
        job_artifact_root: Root for artifacts
        llm_call_fn: Async function to call LLM
        stop_on_failure: If True, stop on first chunk failure
    
    Returns:
        (passed_chunks, failed_chunks)
    """
    from app.overwatcher.planner import topological_sort_chunks
    from app.pot_spec.ledger import emit_job_completed, emit_job_failed
    
    sorted_chunks = topological_sort_chunks(plan.chunks)
    
    passed_chunks = []
    failed_chunks = []
    
    for chunk in sorted_chunks:
        logger.info(f"[orchestrator] Processing chunk {chunk.chunk_id}: {chunk.title}")
        
        try:
            success, _ = await run_chunk_with_strikes(
                chunk=chunk,
                repo_path=repo_path,
                state=state,
                job_artifact_root=job_artifact_root,
                llm_call_fn=llm_call_fn,
            )
            
            if success:
                passed_chunks.append(chunk.chunk_id)
                logger.info(f"[orchestrator] Chunk {chunk.chunk_id} VERIFIED")
            else:
                failed_chunks.append(chunk.chunk_id)
                
                if stop_on_failure:
                    logger.error(f"[orchestrator] Stopping due to chunk failure")
                    break
                    
        except StrikeThreeError as e:
            failed_chunks.append(e.chunk_id)
            
            if stop_on_failure:
                logger.error(f"[orchestrator] Stopping due to Strike 3 on {e.chunk_id}")
                break
    
    # Emit job completion
    try:
        if not failed_chunks:
            emit_job_completed(
                job_artifact_root=job_artifact_root,
                job_id=state.job_id,
                final_arch_version=plan.arch_version,
                total_iterations=len(passed_chunks),
            )
        else:
            emit_job_failed(
                job_artifact_root=job_artifact_root,
                job_id=state.job_id,
                error_type="chunk_failures",
                error_message=f"Failed chunks: {failed_chunks}",
                failed_stage_id="implementation",
            )
    except Exception as e:
        logger.warning(f"[orchestrator] Failed to emit job completion: {e}")
    
    return passed_chunks, failed_chunks


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
