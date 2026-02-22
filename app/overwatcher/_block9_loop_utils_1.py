from __future__ import annotations
import logging
from app.overwatcher.block9_loop import Block9State, logger, run_chunk_block9
from app.overwatcher.error_signature import ErrorSignature, compute_error_signature
from app.overwatcher.evidence import EvidenceBundle, FileChange
from app.overwatcher.incident_report import IncidentReport
from app.overwatcher.schemas import Chunk, VerificationResult
from app.overwatcher.strike_state import StrikeRecord
from app.overwatcher.validated_overwatcher import run_validated_overwatcher
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
from uuid import uuid4
from .block9_loop import emit_fix_actions_issued, emit_overwatcher_called, emit_strike_recorded, record_overwatch_intervention
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
ASTRA_MEMORY_AVAILABLE = True
LEDGER_AVAILABLE = True


MAX_STRIKES = 3

DEEP_RESEARCH_ON_STRIKE = 2  # Run deep research on strike 2

@dataclass 
class ChunkExecutionResult:
    """Result of executing a chunk through Block 9."""
    success: bool
    chunk_id: str
    strikes_used: int
    final_verification: Optional[VerificationResult] = None
    incident_report: Optional[IncidentReport] = None
    error_message: Optional[str] = None

def build_failure_evidence(
    *,
    state: Block9State,
    chunk: Chunk,
    verification_result: VerificationResult,
    strike_number: int,
    previous_signature: Optional[ErrorSignature],
    touched_files: List[str],
) -> EvidenceBundle:
    """Build evidence bundle from verification failure."""
    
    # Build file changes from chunk
    file_changes = []
    for path in chunk.allowed_files.get("modify", []):
        file_changes.append(FileChange(
            path=path,
            action="modify",
            intent=chunk.objective or "Implementation",
        ))
    for path in chunk.allowed_files.get("add", []):
        file_changes.append(FileChange(
            path=path,
            action="add", 
            intent=chunk.objective or "New file",
        ))
    
    # Extract error output from verification
    error_output = ""
    stack_trace = ""
    if verification_result.command_results:
        for cmd_result in verification_result.command_results:
            if not cmd_result.passed:
                error_output += f"\n{cmd_result.stderr or cmd_result.stdout}"
                if "Traceback" in (cmd_result.stderr or ""):
                    stack_trace = cmd_result.stderr
    
    # Compute current error signature
    current_signature = compute_error_signature(error_output) if error_output else None
    
    return EvidenceBundle(
        job_id=state.job_id,
        chunk_id=chunk.chunk_id,
        stage_run_id=str(uuid4()),
        spec_id=state.spec_id,
        spec_hash=state.spec_hash,
        strike_number=strike_number,
        previous_error_signature=previous_signature,
        file_changes=file_changes,
        test_result=None,  # Could populate from verification_result
        lint_results=[],
        error_output=error_output[:5000],  # Truncate
        stack_trace=stack_trace[:2000],
        current_error_signature=current_signature,
        chunk_title=chunk.title,
        chunk_objective=chunk.objective,
        verification_commands=chunk.verification.commands if chunk.verification else [],
    )

async def call_overwatcher_for_diagnosis(
    *,
    state: Block9State,
    chunk: Chunk,
    evidence: EvidenceBundle,
    llm_call_fn: Callable,
    db_session=None,
) -> Tuple[Optional[str], Optional[ErrorSignature]]:
    """Call Overwatcher to diagnose failure and get FIX_ACTIONS.
    
    Returns:
        (fix_actions_text, error_signature)
    """
    state.total_overwatcher_calls += 1
    
    # Emit ledger event
    if LEDGER_AVAILABLE:
        try:
            emit_overwatcher_called(
                job_artifact_root=state.job_artifact_root,
                job_id=state.job_id,
                chunk_id=chunk.chunk_id,
                strike_number=evidence.strike_number,
            )
        except Exception as e:
            logger.warning(f"[block9] Failed to emit overwatcher_called: {e}")
    
    # Call validated overwatcher (Job 3)
    result = await run_validated_overwatcher(
        evidence=evidence,
        llm_call_fn=llm_call_fn,
        job_artifact_root=state.job_artifact_root,
        deep_research_context=state.deep_research_context,
        db_session=db_session,
    )
    
    # Record to astra_memory
    if ASTRA_MEMORY_AVAILABLE and db_session:
        try:
            intervention_type = "warning" if evidence.strike_number < 3 else "block"
            record_overwatch_intervention(
                db=db_session,
                job_id=state.job_id,
                intervention_type=intervention_type,
                reason=result.output.diagnosis[:200],
                error_signature=evidence.current_error_signature.signature_hash if evidence.current_error_signature else None,
            )
        except Exception as e:
            logger.warning(f"[block9] Failed to record intervention: {e}")
    
    # Build FIX_ACTIONS context for Implementer
    fix_actions_text = None
    if result.output.fix_actions:
        lines = ["OVERWATCHER FIX_ACTIONS (implement these):"]
        for fa in result.output.fix_actions:
            lines.append(f"  {fa.order}. [{fa.target_file}] {fa.action_type}: {fa.description}")
        if result.output.constraints:
            lines.append("CONSTRAINTS (do NOT violate):")
            for c in result.output.constraints:
                lines.append(f"  - {c}")
        fix_actions_text = "\n".join(lines)
        
        # Emit fix actions
        if LEDGER_AVAILABLE:
            try:
                emit_fix_actions_issued(
                    job_artifact_root=state.job_artifact_root,
                    job_id=state.job_id,
                    chunk_id=chunk.chunk_id,
                    action_count=len(result.output.fix_actions),
                )
            except Exception:
                pass
    
    return fix_actions_text, evidence.current_error_signature

def record_strike_to_manager(
    *,
    state: Block9State,
    chunk: Chunk,
    error_signature: ErrorSignature,
    diagnosis: str,
    verification_result: VerificationResult,
) -> StrikeRecord:
    """Record strike to StrikeManager and return record."""
    
    record = state.strike_manager.record_strike(
        job_id=state.job_id,
        stage="verification",
        error_signature=error_signature,
        diagnosis=diagnosis,
        evidence={
            "chunk_id": chunk.chunk_id,
            "tests_failed": verification_result.tests_failed,
            "lint_errors": verification_result.lint_errors,
        },
    )
    
    # Emit ledger event
    if LEDGER_AVAILABLE:
        try:
            emit_strike_recorded(
                job_artifact_root=state.job_artifact_root,
                job_id=state.job_id,
                chunk_id=chunk.chunk_id,
                strike_number=record.strike_count,
                reason=diagnosis[:200],
            )
        except Exception:
            pass
    
    return record

async def run_chunks_block9(
    *,
    chunks: List[Chunk],
    state: Block9State,
    llm_call_fn: Callable,
    db_session=None,
    stop_on_failure: bool = True,
) -> Tuple[List[str], List[str]]:
    """Run multiple chunks through Block 9.
    
    Args:
        chunks: Chunks to execute (should be topologically sorted)
        state: Block 9 state
        llm_call_fn: Async LLM call function
        db_session: Database session
        stop_on_failure: Stop on first chunk failure
    
    Returns:
        (passed_chunk_ids, failed_chunk_ids)
    """
    passed = []
    failed = []
    
    for chunk in chunks:
        result = await run_chunk_block9(
            chunk=chunk,
            state=state,
            llm_call_fn=llm_call_fn,
            db_session=db_session,
        )
        
        if result.success:
            passed.append(chunk.chunk_id)
        else:
            failed.append(chunk.chunk_id)
            if stop_on_failure:
                logger.error(f"[block9] Stopping due to chunk failure: {chunk.chunk_id}")
                break
    
    # Log summary
    logger.info(
        f"[block9] Complete: {len(passed)} passed, {len(failed)} failed, "
        f"overwatcher_calls={state.total_overwatcher_calls}, "
        f"implementer_calls={state.total_implementer_calls}"
    )
    
    return passed, failed
