# FILE: app/pot_spec/spec_gate_v2.py
"""
Spec Gate v2 - Wraps spec_gate with clarification state management.

This module provides run_spec_gate_v2() which:
1. Calls the original run_spec_gate()
2. Filters questions through clarification state (dedupe, 3-round cap)
3. Returns SpecGateResult with ready_for_pipeline flag
4. Provides confirmation handler for critical pipeline gate
5. Persists validated specs to DB for restart survival

v1.4 (2026-01): FIXED import error (Spec as SpecSchema) and path mismatch (added "jobs" segment)
v1.3 (2026-01): Fixed jobs/jobs/ double path bug in _get_spec_file_path
v1.2 (2026-01): Added DB persistence for validated specs (fixes restart lookup)
v1.1 (2026-01): Added round 3 directive to force spec output (no questions)
v1.0 (2026-01): Initial implementation

Usage:
    # Instead of:
    spec_id, spec_hash, questions = await run_spec_gate(...)
    
    # Use:
    result = await run_spec_gate_v2(...)
    if result.ready_for_pipeline:
        print(result.spec_summary_markdown)
        # await user confirmation
    elif result.hard_stopped:
        print(f"HARD STOP: {result.hard_stop_reason}")
    else:
        # show result.open_questions to user
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# Optional Imports (keep module resilient)
# =============================================================================

# Import original spec_gate
try:
    from app.pot_spec.spec_gate import run_spec_gate, _artifact_root, _append_event, _utc_ts
    _SPEC_GATE_AVAILABLE = True
except ImportError:
    _SPEC_GATE_AVAILABLE = False
    run_spec_gate = None
    def _artifact_root():
        return os.path.abspath(os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"))
    def _append_event(*args, **kwargs):
        pass
    def _utc_ts():
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# Import clarification state
try:
    from app.pot_spec.clarification_state import (
        ClarificationState,
        ClarificationDecision,
        load_clarification_state,
        save_clarification_state,
        format_spec_summary_markdown,
        compute_question_signature_simple,
    )
    _CLARIFICATION_AVAILABLE = True
except ImportError:
    _CLARIFICATION_AVAILABLE = False
    logger.warning("[spec_gate_v2] clarification_state module not available")

# Import specs service for DB persistence (CRITICAL for restart survival)
# v1.4 FIX: Import Spec as SpecSchema (the class is named "Spec" not "SpecSchema")
try:
    from app.specs.service import create_spec, update_spec_status
    from app.specs.schema import Spec as SpecSchema, SpecStatus
    _SPECS_SERVICE_AVAILABLE = True
    logger.info("[spec_gate_v2] specs.service module loaded successfully")
except ImportError as e:
    _SPECS_SERVICE_AVAILABLE = False
    logger.warning(f"[spec_gate_v2] specs.service module not available - validated specs won't persist to DB: {e}")


# =============================================================================
# Result Dataclass
# =============================================================================

@dataclass
class SpecGateResult:
    """Result from run_spec_gate_v2 with clarification state."""
    spec_id: str
    spec_hash: str
    open_questions: List[str]
    spec_version: int
    ready_for_pipeline: bool
    hard_stopped: bool = False
    hard_stop_reason: Optional[str] = None
    clarification_round: int = 0
    spec_summary_markdown: Optional[str] = None
    spec_file_path: Optional[str] = None
    db_persisted: bool = False  # v1.2: Track if spec was saved to DB
    
    def __iter__(self):
        """Allow unpacking as (spec_id, spec_hash, open_questions)."""
        return iter([self.spec_id, self.spec_hash, self.open_questions])
    
    def __getitem__(self, idx):
        """Allow indexing like tuple."""
        return [self.spec_id, self.spec_hash, self.open_questions][idx]


# =============================================================================
# Helper Functions
# =============================================================================

def _get_artifact_root() -> str:
    if _SPEC_GATE_AVAILABLE:
        return _artifact_root()
    return os.path.abspath(os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"))


def _get_spec_file_path(job_root: str, job_id: str, spec_version: int) -> str:
    """
    Get the path to a spec file.
    
    v1.5 FIX: Always look for spec_v1.json since spec_gate.py hardcodes version=1.
    The spec_version parameter is kept for API compatibility but not used in path.
    spec_gate.py writes to: job_root/jobs/<job_id>/spec/spec_v1.json (always v1)
    
    TODO: Fix spec_gate.py to accept and use spec_version parameter, then 
    revert this to use spec_version in the path.
    """
    # Always use v1 since spec_gate.py hardcodes spec_version=1
    return os.path.join(job_root, "jobs", job_id, "spec", "spec_v1.json")


def _filter_questions_with_clarification(
    questions: List[str],
    job_artifact_root: str,
    job_id: str,
    spec_version: int,
) -> Tuple[List[str], "ClarificationDecision", Optional["ClarificationState"]]:
    """Filter questions through clarification state."""
    if not _CLARIFICATION_AVAILABLE:
        # Stub for when module unavailable
        class _StubDecision:
            CONTINUE = "continue"
            READY_FOR_CONFIRM = "ready"
        if questions:
            return questions, _StubDecision.CONTINUE, None
        return [], _StubDecision.READY_FOR_CONFIRM, None
    
    state = load_clarification_state(job_artifact_root, job_id)
    decision = state.record_questions(questions, spec_version)
    save_clarification_state(job_artifact_root, job_id, state)
    
    if decision == ClarificationDecision.HARD_STOP:
        return [], decision, state
    
    if decision == ClarificationDecision.ALREADY_ASKED:
        return [], ClarificationDecision.READY_FOR_CONFIRM, state
    
    if decision == ClarificationDecision.READY_FOR_CONFIRM:
        return [], decision, state
    
    if state.rounds:
        return state.rounds[-1].questions_asked, decision, state
    
    return questions, decision, state


def _persist_validated_spec_to_db(
    db: Any,
    project_id: int,
    spec_id: str,
    spec_hash: str,
    spec_version: int,
    spec_file_path: str,
    job_id: str,
    job_root: str,
) -> bool:
    """
    Persist a validated spec to the specs DB table.
    
    This is CRITICAL for restart survival - without this, get_latest_validated_spec()
    returns None after app restart because it only queries the DB, not filesystem.
    
    Returns True if successfully persisted, False otherwise.
    """
    if not _SPECS_SERVICE_AVAILABLE:
        logger.warning("[spec_gate_v2] Cannot persist spec to DB - specs.service not available")
        return False
    
    if not project_id:
        logger.error("[spec_gate_v2] Cannot persist spec to DB - project_id not provided")
        return False
    
    try:
        # Load spec payload from filesystem
        if not os.path.exists(spec_file_path):
            logger.error(f"[spec_gate_v2] Spec file not found: {spec_file_path}")
            return False
            
        with open(spec_file_path, "r", encoding="utf-8") as f:
            spec_payload = json.load(f)
        
        # Extract fields for SpecSchema
        goal = spec_payload.get("goal", "")
        title = goal[:200] if goal else f"Spec {spec_id[:8]}"
        created_by_model = spec_payload.get("created_by_model", "spec_gate_v2")
        
        # Build content_markdown from payload if possible
        content_markdown = None
        if spec_payload.get("requirements"):
            md_parts = [f"# {title}", "", "## Goal", goal, ""]
            reqs = spec_payload.get("requirements", {})
            if reqs.get("must"):
                md_parts.append("## Must Have")
                for r in reqs["must"]:
                    md_parts.append(f"- {r}")
                md_parts.append("")
            if reqs.get("should"):
                md_parts.append("## Should Have")
                for r in reqs["should"]:
                    md_parts.append(f"- {r}")
                md_parts.append("")
            if reqs.get("can"):
                md_parts.append("## Nice to Have")
                for r in reqs["can"]:
                    md_parts.append(f"- {r}")
                md_parts.append("")
            content_markdown = "\n".join(md_parts)
        
        # Create SpecSchema object
        from app.specs.schema import SpecProvenance, SpecRequirements, SpecConstraints, SpecSafety
        
        spec_schema = SpecSchema(
            spec_id=spec_id,
            spec_version=str(spec_version),
            title=title,
            summary=goal[:500] if goal else "",
            objective=goal,
        )
        
        # Set provenance
        spec_schema.provenance = SpecProvenance(
            job_id=job_id,
            generator_model=created_by_model,
            created_at=spec_payload.get("created_at", datetime.now(timezone.utc).isoformat()),
        )
        
        # Create in DB
        db_spec = create_spec(
            db=db,
            project_id=project_id,
            spec_schema=spec_schema,
            generator_model=created_by_model,
        )
        
        # Update status to VALIDATED
        update_spec_status(
            db=db,
            spec_id=spec_id,
            new_status=SpecStatus.VALIDATED.value,
            validation_result={
                "valid": True,
                "validated_at": _utc_ts(),
                "spec_hash": spec_hash,
                "spec_file": spec_file_path,
            },
            triggered_by="spec_gate_v2",
        )
        
        logger.info(f"[spec_gate_v2] Spec {spec_id} persisted to DB with status=validated")
        return True
        
    except Exception as e:
        logger.exception(f"[spec_gate_v2] Failed to persist spec to DB: {e}")
        return False


# =============================================================================
# Main Entry Point
# =============================================================================

async def run_spec_gate_v2(
    db: Any,
    job_id: str,
    user_intent: str,
    provider_id: str,
    model_id: str,
    project_id: Optional[int] = None,
    repo_snapshot: Optional[dict] = None,
    constraints_hint: Optional[dict] = None,
    spec_version: int = 1,
    **kwargs,
) -> SpecGateResult:
    """
    Run Spec Gate with clarification state management.
    
    Args:
        db: Database session
        job_id: Job identifier
        user_intent: User's request/intent text
        provider_id: LLM provider (openai, anthropic, google)
        model_id: Model identifier
        project_id: Project ID - REQUIRED for DB persistence and restart survival
        repo_snapshot: Optional repository state snapshot
        constraints_hint: Optional constraints for spec generation
        spec_version: Spec version number (increments with clarification rounds)
        **kwargs: Additional arguments passed to run_spec_gate
    
    Returns:
        SpecGateResult with spec details, questions, and pipeline readiness
    """
    if not _SPEC_GATE_AVAILABLE:
        raise RuntimeError("spec_gate module not available")
    
    # Warn if project_id not provided (will break restart survival)
    if project_id is None:
        logger.warning("[spec_gate_v2] project_id not provided - spec won't persist to DB for restart survival")
    
    job_root = _get_artifact_root()
    
    # =========================================================================
    # ROUND 3 DIRECTIVE: Force spec output, no more questions
    # =========================================================================
    effective_intent = user_intent
    if spec_version >= 3:
        round3_directive = """
[FINAL ROUND - MANDATORY SPEC OUTPUT]
This is clarification round 3 of 3. You MUST NOT ask any more questions.
You MUST output the complete, final spec based on all information gathered so far.
If any details are missing, make reasonable assumptions and note them in the spec.
DO NOT return questions. Return the complete spec JSON only.

User context and answers:
"""
        effective_intent = round3_directive + user_intent
        logger.info(f"[spec_gate_v2] Round 3: Forcing spec output (no questions)")
    
    # =========================================================================
    # Call underlying spec_gate
    # =========================================================================
    spec_id, spec_hash, raw_questions = await run_spec_gate(
        db,
        job_id,
        effective_intent,
        provider_id,
        model_id,
        repo_snapshot=repo_snapshot,
        constraints_hint=constraints_hint,
        **kwargs,
    )
    
    # =========================================================================
    # Filter questions through clarification state
    # =========================================================================
    filtered_questions, decision, clarification_state = _filter_questions_with_clarification(
        questions=raw_questions,
        job_artifact_root=job_root,
        job_id=job_id,
        spec_version=spec_version,
    )
    
    # ROUND 3 OVERRIDE: Force ready_for_pipeline even if questions returned
    round3_forced = False
    if spec_version >= 3 and filtered_questions:
        logger.warning(f"[spec_gate_v2] Round 3 but got {len(filtered_questions)} questions - forcing ready_for_pipeline")
        decision = ClarificationDecision.READY_FOR_CONFIRM if _CLARIFICATION_AVAILABLE else "ready"
        round3_forced = True
        # Clear questions so we proceed (they're logged for reference)
        filtered_questions = []
    
    # =========================================================================
    # Determine pipeline readiness
    # =========================================================================
    ready_for_pipeline = (decision == ClarificationDecision.READY_FOR_CONFIRM) if _CLARIFICATION_AVAILABLE else (decision == "ready")
    hard_stopped = (decision == ClarificationDecision.HARD_STOP) if _CLARIFICATION_AVAILABLE else False
    hard_stop_reason = clarification_state.hard_stop_reason if (hard_stopped and clarification_state) else None
    
    spec_file_path = _get_spec_file_path(job_root, job_id, spec_version)
    
    # =========================================================================
    # CRITICAL: Persist validated spec to DB for restart survival
    # =========================================================================
    db_persisted = False
    if ready_for_pipeline and project_id:
        db_persisted = _persist_validated_spec_to_db(
            db=db,
            project_id=project_id,
            spec_id=spec_id,
            spec_hash=spec_hash,
            spec_version=spec_version,
            spec_file_path=spec_file_path,
            job_id=job_id,
            job_root=job_root,
        )
    
    # =========================================================================
    # Generate spec summary markdown
    # =========================================================================
    spec_summary_markdown = None
    if ready_for_pipeline and _CLARIFICATION_AVAILABLE:
        try:
            with open(spec_file_path, "r", encoding="utf-8") as f:
                spec_payload = json.load(f)
            spec_payload["spec_hash"] = spec_hash
            spec_summary_markdown = format_spec_summary_markdown(
                spec_payload=spec_payload,
                spec_file_path=spec_file_path,
                clarification_state=clarification_state,
            )
        except Exception as e:
            logger.warning(f"[spec_gate_v2] Failed to build summary: {e}")
    
    # =========================================================================
    # Emit ledger events
    # =========================================================================
    if filtered_questions:
        _append_event(job_root, job_id, {
            "event": "SPEC_QUESTIONS_FILTERED",
            "job_id": job_id,
            "spec_id": spec_id,
            "raw_count": len(raw_questions),
            "filtered_count": len(filtered_questions),
            "clarification_round": clarification_state.current_round if clarification_state else 0,
            "status": "ok",
            "ts": _utc_ts(),
        })
    
    if ready_for_pipeline:
        _append_event(job_root, job_id, {
            "event": "SPEC_READY_FOR_PIPELINE",
            "job_id": job_id,
            "spec_id": spec_id,
            "spec_version": spec_version,
            "awaiting_confirmation": True,
            "db_persisted": db_persisted,
            "status": "ok",
            "ts": _utc_ts(),
        })
    
    if hard_stopped:
        _append_event(job_root, job_id, {
            "event": "SPEC_GATE_HARD_STOP",
            "job_id": job_id,
            "reason": hard_stop_reason,
            "status": "hard_stop",
            "ts": _utc_ts(),
        })
    
    return SpecGateResult(
        spec_id=spec_id,
        spec_hash=spec_hash,
        open_questions=filtered_questions,
        spec_version=spec_version,
        ready_for_pipeline=ready_for_pipeline,
        hard_stopped=hard_stopped,
        hard_stop_reason=hard_stop_reason,
        clarification_round=clarification_state.current_round if clarification_state else 0,
        spec_summary_markdown=spec_summary_markdown,
        spec_file_path=spec_file_path,
        db_persisted=db_persisted,
    )


# =============================================================================
# Confirmation Handler
# =============================================================================

async def confirm_spec_for_pipeline(
    job_id: str,
    spec_version: int,
    spec_hash: str,
    user_confirmed: bool = False,
) -> Tuple[bool, str]:
    """Handle user confirmation before critical pipeline."""
    if not _CLARIFICATION_AVAILABLE:
        return user_confirmed, "Clarification module not available"
    
    job_root = _get_artifact_root()
    state = load_clarification_state(job_root, job_id)
    
    if not state.ready_for_pipeline:
        return False, "Spec not ready (has open questions)"
    
    if state.hard_stopped:
        return False, f"Hard stopped: {state.hard_stop_reason}"
    
    if not user_confirmed:
        return False, "Awaiting user confirmation"
    
    confirmed = state.confirm_for_pipeline(spec_version=spec_version, spec_hash=spec_hash)
    
    if confirmed:
        save_clarification_state(job_root, job_id, state)
        _append_event(job_root, job_id, {
            "event": "SPEC_CONFIRMED_FOR_PIPELINE",
            "job_id": job_id,
            "spec_version": spec_version,
            "spec_hash": spec_hash,
            "confirmed_at": state.confirmed_at,
            "status": "ok",
            "ts": _utc_ts(),
        })
        return True, "Confirmed - proceeding to critical pipeline"
    
    return False, "Confirmation failed"


__all__ = ["SpecGateResult", "run_spec_gate_v2", "confirm_spec_for_pipeline"]