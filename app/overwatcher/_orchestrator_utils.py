from __future__ import annotations
import logging
from app.overwatcher.schemas import ChunkPlan, VerificationResult
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
BLOCK9_AVAILABLE = True


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
    
    # Use Block 9 loop if available (integrates Overwatcher diagnosis)
    if BLOCK9_AVAILABLE:
        block9_state = Block9State(
            job_id=state.job_id,
            spec_id=state.spec_id or "",
            spec_hash=state.spec_hash or "",
            repo_path=repo_path,
            job_artifact_root=job_artifact_root,
        )
        
        passed_chunks, failed_chunks = await run_chunks_block9(
            chunks=sorted_chunks,
            state=block9_state,
            llm_call_fn=llm_call_fn,
            db_session=None,  # Pass db session if available
            stop_on_failure=stop_on_failure,
        )
        
        # Update PipelineState for compatibility
        state.completed_chunks = passed_chunks
        state.failed_chunks = failed_chunks
    else:
        # Legacy path - run_chunk_with_strikes without Overwatcher integration
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
