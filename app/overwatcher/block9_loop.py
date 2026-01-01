# FILE: app/overwatcher/block9_loop.py
"""Block 9: Verified Execution Loop with Overwatcher Integration.

Wires together:
- Executor (Sonnet) for implementation
- Verifier for tests/lint
- Validated Overwatcher (GPT-5.2 Pro) for diagnosis on failure
- Strike tracking by ErrorSignature
- Incident reports on Strike 3
- Cost guard enforcement
- Persistence to astra_memory

Flow:
  1. Execute chunk (Implementer/Sonnet)
  2. Verify (tests + lint)
  3. ON PASS: done
  4. ON FAIL:
     a. Build evidence bundle
     b. Call Overwatcher → get FIX_ACTIONS
     c. Record strike (same signature = increment, different = reset)
     d. Strike 2: deep research allowed
     e. Strike 3: HARD_STOP + incident report
     f. Pass FIX_ACTIONS context to next attempt

Spec v2.3 §9.4: Three-strike by ErrorSignature, not attempt count.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# =============================================================================
# Imports - Core Components
# =============================================================================

from app.overwatcher.schemas import (
    Chunk,
    ChunkStatus,
    VerificationResult,
    VerificationStatus,
)
from app.overwatcher.evidence import (
    EvidenceBundle,
    FileChange,
    build_evidence_bundle,
)
from app.overwatcher.error_signature import (
    ErrorSignature,
    compute_error_signature,
    signatures_match,
)

# Job 1: Strike state + incident reports
from app.overwatcher.strike_state import (
    StrikeManager,
    StrikeRecord,
    StrikeOutcome,
)
from app.overwatcher.incident_report import (
    IncidentReport,
    build_incident_report,
    store_incident_report,
)

# Job 3: Validated Overwatcher + Cost Guard
from app.overwatcher.validated_overwatcher import (
    run_validated_overwatcher,
    ValidatedOverwatcherResult,
)
from app.overwatcher.cost_guard import (
    ModelRole,
    record_usage,
    get_cost_guard,
)

# Job 5: Memory persistence
try:
    from app.astra_memory.service import (
        create_job,
        update_job_status,
        project_event_to_db,
        record_overwatch_intervention,
        record_overwatch_pattern,
        create_chunk,
        update_chunk_status,
        get_or_create_overwatch_summary,
    )
    ASTRA_MEMORY_AVAILABLE = True
except ImportError:
    ASTRA_MEMORY_AVAILABLE = False
    logger.warning("[block9] astra_memory not available")

# Existing modules
from app.overwatcher.executor import (
    execute_chunk,
    create_backup,
    rollback_chunk,
)
from app.overwatcher.verifier import verify_chunk
from app.overwatcher.deep_research import run_deep_research

# Ledger events
try:
    from app.pot_spec.ledger_overwatcher import (
        emit_strike_recorded,
        emit_chunk_abandoned,
        emit_overwatcher_called,
        emit_fix_actions_issued,
    )
    LEDGER_AVAILABLE = True
except ImportError:
    LEDGER_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

MAX_STRIKES = 3
DEEP_RESEARCH_ON_STRIKE = 2  # Run deep research on strike 2


# =============================================================================
# Block 9 State
# =============================================================================

@dataclass
class Block9State:
    """State for Block 9 execution loop."""
    job_id: str
    spec_id: str
    spec_hash: str
    repo_path: str
    job_artifact_root: str
    
    # Strike manager (Job 1)
    strike_manager: StrikeManager = field(default_factory=StrikeManager)
    
    # Accumulated context from Overwatcher
    fix_actions_context: Optional[str] = None
    deep_research_context: Optional[str] = None
    
    # Tracking
    total_overwatcher_calls: int = 0
    total_implementer_calls: int = 0


@dataclass 
class ChunkExecutionResult:
    """Result of executing a chunk through Block 9."""
    success: bool
    chunk_id: str
    strikes_used: int
    final_verification: Optional[VerificationResult] = None
    incident_report: Optional[IncidentReport] = None
    error_message: Optional[str] = None


# =============================================================================
# Evidence Bundle Builder
# =============================================================================

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


# =============================================================================
# Overwatcher Integration
# =============================================================================

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


# =============================================================================
# Strike Recording
# =============================================================================

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


# =============================================================================
# Main Execution Loop
# =============================================================================

async def run_chunk_block9(
    *,
    chunk: Chunk,
    state: Block9State,
    llm_call_fn: Callable,
    db_session=None,
) -> ChunkExecutionResult:
    """Execute a chunk with full Block 9 loop.
    
    Flow:
    1. Backup files
    2. Execute (Implementer)
    3. Verify
    4. On failure: Overwatcher → strike → retry or HARD_STOP
    
    Args:
        chunk: Chunk to execute
        state: Block 9 state
        llm_call_fn: Async LLM call function
        db_session: Database session for persistence
    
    Returns:
        ChunkExecutionResult
    """
    logger.info(f"[block9] Starting chunk {chunk.chunk_id}: {chunk.title}")
    
    # Record chunk start in astra_memory
    if ASTRA_MEMORY_AVAILABLE and db_session:
        try:
            create_chunk(
                db=db_session,
                job_id=state.job_id,
                chunk_id=chunk.chunk_id,
                sequence=chunk.sequence,
                target_path=chunk.allowed_files.get("modify", [""])[0] if chunk.allowed_files.get("modify") else "",
                description=chunk.objective,
            )
        except Exception as e:
            logger.warning(f"[block9] Failed to create chunk record: {e}")
    
    # Backup files before modification
    backup_dir = str(Path(state.job_artifact_root) / "backups") if state.job_artifact_root else ""
    backups = create_backup(chunk, state.repo_path, backup_dir)
    
    # Track current signature for this chunk
    current_signature: Optional[ErrorSignature] = None
    strike_count = 0
    last_verification: Optional[VerificationResult] = None
    
    while strike_count < MAX_STRIKES:
        attempt = strike_count + 1
        logger.info(f"[block9] Chunk {chunk.chunk_id} attempt {attempt}/{MAX_STRIKES}")
        
        state.total_implementer_calls += 1
        
        # === EXECUTE ===
        # Pass fix_actions context if available from previous Overwatcher call
        exec_success, diff_result, files_written = await execute_chunk(
            chunk=chunk,
            repo_path=state.repo_path,
            spec_id=state.spec_id,
            spec_hash=state.spec_hash,
            job_id=state.job_id,
            job_artifact_root=state.job_artifact_root,
            llm_call_fn=llm_call_fn,
            fix_actions_context=state.fix_actions_context,
        )
        
        if not exec_success:
            # Execution failed (boundary violation or LLM error)
            logger.warning(f"[block9] Execution failed for {chunk.chunk_id}")
            
            # Rollback
            files_added = list(files_written.keys()) if files_written else []
            rollback_chunk(chunk, state.repo_path, backups, files_added)
            
            # Build minimal evidence for Overwatcher
            verification_result = VerificationResult(
                chunk_id=chunk.chunk_id,
                status=VerificationStatus.FAILED,
                command_results=[],
                tests_passed=0,
                tests_failed=0,
                lint_errors=0,
                type_errors=0,
            )
            
            # Compute signature from boundary violations
            if diff_result.violations:
                error_text = f"Boundary violations: {[v.file_path for v in diff_result.violations]}"
                current_signature = compute_error_signature(error_text)
            
            strike_count += 1
            last_verification = verification_result
            continue
        
        # === VERIFY ===
        touched_files = list(files_written.keys()) if files_written else []
        verification_result = await verify_chunk(
            chunk=chunk,
            repo_path=state.repo_path,
            touched_files=touched_files,
            job_artifact_root=state.job_artifact_root,
        )
        
        last_verification = verification_result
        
        if verification_result.status == VerificationStatus.PASSED:
            # SUCCESS!
            logger.info(f"[block9] Chunk {chunk.chunk_id} VERIFIED on attempt {attempt}")
            
            # Update astra_memory
            if ASTRA_MEMORY_AVAILABLE and db_session:
                try:
                    update_chunk_status(
                        db=db_session,
                        job_id=state.job_id,
                        chunk_id=chunk.chunk_id,
                        status="completed",
                        tests_passed=verification_result.tests_passed,
                        tests_failed=verification_result.tests_failed,
                        lint_errors=verification_result.lint_errors,
                    )
                except Exception:
                    pass
            
            return ChunkExecutionResult(
                success=True,
                chunk_id=chunk.chunk_id,
                strikes_used=strike_count,
                final_verification=verification_result,
            )
        
        # === VERIFICATION FAILED ===
        logger.warning(f"[block9] Verification failed for {chunk.chunk_id}")
        
        # Rollback before diagnosis
        files_added = list(files_written.keys()) if files_written else []
        rollback_chunk(chunk, state.repo_path, backups, files_added)
        
        # Build evidence bundle
        evidence = build_failure_evidence(
            state=state,
            chunk=chunk,
            verification_result=verification_result,
            strike_number=strike_count + 1,
            previous_signature=current_signature,
            touched_files=touched_files,
        )
        
        # === CALL OVERWATCHER ===
        fix_actions_text, new_signature = await call_overwatcher_for_diagnosis(
            state=state,
            chunk=chunk,
            evidence=evidence,
            llm_call_fn=llm_call_fn,
            db_session=db_session,
        )
        
        # Update state with fix actions for next attempt
        state.fix_actions_context = fix_actions_text
        
        # === RECORD STRIKE ===
        # Check if same signature
        if current_signature and new_signature and signatures_match(current_signature, new_signature):
            strike_count += 1
        else:
            # Different error - reset to 1
            strike_count = 1
            current_signature = new_signature
        
        # Record to strike manager
        if new_signature:
            record = record_strike_to_manager(
                state=state,
                chunk=chunk,
                error_signature=new_signature,
                diagnosis=evidence.error_output[:500] if evidence.error_output else "Verification failed",
                verification_result=verification_result,
            )
        
        logger.warning(f"[block9] Strike {strike_count}/{MAX_STRIKES} for {chunk.chunk_id}")
        
        # === STRIKE 2: DEEP RESEARCH ===
        if strike_count == DEEP_RESEARCH_ON_STRIKE and new_signature:
            logger.info(f"[block9] Strike 2 - running deep research")
            try:
                research_result = await run_deep_research(
                    error_signature=new_signature,
                    stack_trace=evidence.stack_trace or "",
                    context=chunk.objective or "",
                    job_id=state.job_id,
                    chunk_id=chunk.chunk_id,
                    job_artifact_root=state.job_artifact_root,
                    llm_call_fn=llm_call_fn,
                )
                state.deep_research_context = research_result.to_context_string()
                logger.info(f"[block9] Deep research complete")
            except Exception as e:
                logger.warning(f"[block9] Deep research failed: {e}")
    
    # === STRIKE 3: HARD STOP ===
    logger.error(f"[block9] Chunk {chunk.chunk_id} exhausted {MAX_STRIKES} strikes - HARD STOP")
    
    # Generate incident report (Job 1)
    incident = build_incident_report(
        job_id=state.job_id,
        stage_name="verification",
        signature=current_signature,
        strike_history=state.strike_manager.get_history_for_signature(
            current_signature.signature_hash if current_signature else ""
        ),
        diagnosis_summary=state.fix_actions_context or "No diagnosis available",
    )
    
    # Store incident report
    if state.job_artifact_root:
        store_incident_report(incident, state.job_artifact_root)
    
    # Emit chunk abandoned
    if LEDGER_AVAILABLE:
        try:
            emit_chunk_abandoned(
                job_artifact_root=state.job_artifact_root,
                job_id=state.job_id,
                chunk_id=chunk.chunk_id,
                strike_count=strike_count,
                final_error=incident.diagnosis_summary[:200],
            )
        except Exception:
            pass
    
    # Update astra_memory
    if ASTRA_MEMORY_AVAILABLE and db_session:
        try:
            update_chunk_status(
                db=db_session,
                job_id=state.job_id,
                chunk_id=chunk.chunk_id,
                status="quarantined",
            )
            # Record pattern for cross-job learning
            if current_signature:
                record_overwatch_pattern(
                    db=db_session,
                    pattern_type="repeated_failure",
                    job_id=state.job_id,
                    error_signature=current_signature.signature_hash,
                )
        except Exception:
            pass
    
    chunk.status = ChunkStatus.QUARANTINED
    
    return ChunkExecutionResult(
        success=False,
        chunk_id=chunk.chunk_id,
        strikes_used=strike_count,
        final_verification=last_verification,
        incident_report=incident,
        error_message=f"Exhausted {MAX_STRIKES} strikes",
    )


# =============================================================================
# Convenience: Run Multiple Chunks
# =============================================================================

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


# =============================================================================
# Import fix for Path
# =============================================================================

from pathlib import Path


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # State
    "Block9State",
    "ChunkExecutionResult",
    # Evidence
    "build_failure_evidence",
    # Main functions
    "run_chunk_block9",
    "run_chunks_block9",
    # Config
    "MAX_STRIKES",
    "DEEP_RESEARCH_ON_STRIKE",
]
