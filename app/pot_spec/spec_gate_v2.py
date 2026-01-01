# FILE: app/pot_spec/spec_gate_v2.py
"""
Spec Gate v2 - Wraps spec_gate with clarification state management.

This module provides run_spec_gate_v2() which:
1. Calls the original run_spec_gate()
2. Filters questions through clarification state (dedupe, 3-round cap)
3. Returns SpecGateResult with ready_for_pipeline flag
4. Provides confirmation handler for critical pipeline gate

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

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

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
    return os.path.join(job_root, "jobs", job_id, "spec", f"spec_v{spec_version}.json")


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


# =============================================================================
# Main Function
# =============================================================================

async def run_spec_gate_v2(
    db: Any,
    job_id: str,
    user_intent: str,
    provider_id: str,
    model_id: str,
    *,
    repo_snapshot: Optional[dict] = None,
    constraints_hint: Optional[dict] = None,
    spec_version: int = 1,
    **kwargs,
) -> SpecGateResult:
    """Run Spec Gate with clarification state management."""
    if not _SPEC_GATE_AVAILABLE:
        raise RuntimeError("spec_gate module not available")
    
    job_root = _get_artifact_root()
    
    spec_id, spec_hash, raw_questions = await run_spec_gate(
        db,
        job_id,
        user_intent,
        provider_id,
        model_id,
        repo_snapshot=repo_snapshot,
        constraints_hint=constraints_hint,
        **kwargs,
    )
    
    filtered_questions, decision, clarification_state = _filter_questions_with_clarification(
        questions=raw_questions,
        job_artifact_root=job_root,
        job_id=job_id,
        spec_version=spec_version,
    )
    
    ready_for_pipeline = (decision == ClarificationDecision.READY_FOR_CONFIRM)
    hard_stopped = (decision == ClarificationDecision.HARD_STOP)
    hard_stop_reason = clarification_state.hard_stop_reason if (hard_stopped and clarification_state) else None
    
    spec_file_path = _get_spec_file_path(job_root, job_id, spec_version)
    
    spec_summary_markdown = None
    if ready_for_pipeline and _CLARIFICATION_AVAILABLE:
        try:
            import json
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
    
    # Emit events
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
