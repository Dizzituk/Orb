# FILE: app/llm/spec_flow_state.py
"""
Spec Flow State Management for ASTRA Command Flow.

Tracks active spec flows per project to ensure:
1. Follow-up messages route to the correct handler (not chat)
2. SPoT (spec_id, spec_hash) persists across stages
3. Flow stages execute in order with proper context

State Lifecycle:
1. Weaver creates spec → state = "awaiting_spec_gate_confirm"
2. User confirms → Spec Gate runs → state = "spec_gate_questions" or "spec_validated"
3. User answers questions → route back to Spec Gate
4. Spec validated → state = "awaiting_critical_pipeline"
5. User confirms → Critical Pipeline runs → state = "awaiting_overwatcher"
6. User confirms → Overwatcher runs → state = "complete"

v1.0 (2026-01): Initial implementation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class SpecFlowStage(str, Enum):
    """Current stage in the spec flow."""
    # After Weaver generates spec, awaiting user to send to Spec Gate
    AWAITING_SPEC_GATE_CONFIRM = "awaiting_spec_gate_confirm"
    
    # Spec Gate is asking clarification questions
    SPEC_GATE_QUESTIONS = "spec_gate_questions"
    
    # Spec validated, awaiting user to run critical pipeline
    SPEC_VALIDATED = "spec_validated"
    
    # Critical Pipeline complete, awaiting Overwatcher
    AWAITING_OVERWATCHER = "awaiting_overwatcher"
    
    # Flow complete
    COMPLETE = "complete"
    
    # Flow cancelled/abandoned
    CANCELLED = "cancelled"


@dataclass
class SpecFlowState:
    """State of an active spec flow for a project."""
    project_id: int
    stage: SpecFlowStage
    
    # Job tracking
    job_id: Optional[str] = None
    
    # Weaver output (v3.0 - simple text, not spec)
    weaver_spec_id: Optional[str] = None
    weaver_job_description: Optional[str] = None  # v3.0: Simple organized text from Weaver
    
    # Spec Gate output (SPoT - Singular Point of Truth)
    spec_id: Optional[str] = None
    spec_hash: Optional[str] = None
    spec_version: int = 1
    
    # Clarification state
    clarification_round: int = 0
    open_questions: list = field(default_factory=list)
    
    # Work artifacts from Critical Pipeline
    work_artifacts: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for storage."""
        return {
            "project_id": self.project_id,
            "stage": self.stage.value,
            "job_id": self.job_id,
            "weaver_spec_id": self.weaver_spec_id,
            "weaver_job_description": self.weaver_job_description,
            "spec_id": self.spec_id,
            "spec_hash": self.spec_hash,
            "spec_version": self.spec_version,
            "clarification_round": self.clarification_round,
            "open_questions": self.open_questions,
            "work_artifacts": self.work_artifacts,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpecFlowState":
        """Deserialize from dict."""
        return cls(
            project_id=data["project_id"],
            stage=SpecFlowStage(data["stage"]),
            job_id=data.get("job_id"),
            weaver_spec_id=data.get("weaver_spec_id"),
            weaver_job_description=data.get("weaver_job_description"),
            spec_id=data.get("spec_id"),
            spec_hash=data.get("spec_hash"),
            spec_version=data.get("spec_version", 1),
            clarification_round=data.get("clarification_round", 0),
            open_questions=data.get("open_questions", []),
            work_artifacts=data.get("work_artifacts", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(timezone.utc),
        )


# =============================================================================
# IN-MEMORY STATE STORE (Simple implementation)
# For production, consider Redis or DB-backed store
# =============================================================================

_FLOW_STATES: Dict[int, SpecFlowState] = {}


def get_active_flow(project_id: int) -> Optional[SpecFlowState]:
    """Get active spec flow for a project, if any."""
    state = _FLOW_STATES.get(project_id)
    if state and state.stage not in (SpecFlowStage.COMPLETE, SpecFlowStage.CANCELLED):
        return state
    return None


def set_flow_state(state: SpecFlowState) -> None:
    """Set/update flow state for a project."""
    state.updated_at = datetime.now(timezone.utc)
    _FLOW_STATES[state.project_id] = state
    logger.debug(f"[spec_flow] Set state for project {state.project_id}: {state.stage.value}")


def clear_flow_state(project_id: int) -> None:
    """Clear flow state for a project."""
    if project_id in _FLOW_STATES:
        del _FLOW_STATES[project_id]
        logger.debug(f"[spec_flow] Cleared state for project {project_id}")


def start_weaver_flow(
    project_id: int,
    weaver_spec_id: str,
    weaver_job_description: Optional[str] = None,
) -> SpecFlowState:
    """Start a new flow after Weaver generates a spec/job description.
    
    v3.0: Now accepts weaver_job_description for simple Weaver output.
    """
    state = SpecFlowState(
        project_id=project_id,
        stage=SpecFlowStage.AWAITING_SPEC_GATE_CONFIRM,
        weaver_spec_id=weaver_spec_id,
        weaver_job_description=weaver_job_description,
    )
    set_flow_state(state)
    return state


def advance_to_spec_gate_questions(
    project_id: int,
    job_id: str,
    spec_id: str,
    spec_hash: str,
    questions: list,
    clarification_round: int = 1,
) -> Optional[SpecFlowState]:
    """Advance flow to Spec Gate questions stage."""
    state = _FLOW_STATES.get(project_id)
    if not state:
        # Create new state if none exists
        state = SpecFlowState(project_id=project_id, stage=SpecFlowStage.SPEC_GATE_QUESTIONS)
    
    state.stage = SpecFlowStage.SPEC_GATE_QUESTIONS
    state.job_id = job_id
    state.spec_id = spec_id
    state.spec_hash = spec_hash
    state.open_questions = questions
    state.clarification_round = clarification_round
    set_flow_state(state)
    return state


def advance_to_spec_validated(
    project_id: int,
    spec_id: str,
    spec_hash: str,
    spec_version: int = 1,
) -> Optional[SpecFlowState]:
    """Advance flow to spec validated stage (SPoT ready)."""
    state = _FLOW_STATES.get(project_id)
    if not state:
        state = SpecFlowState(project_id=project_id, stage=SpecFlowStage.SPEC_VALIDATED)
    
    state.stage = SpecFlowStage.SPEC_VALIDATED
    state.spec_id = spec_id
    state.spec_hash = spec_hash
    state.spec_version = spec_version
    state.open_questions = []
    set_flow_state(state)
    return state


def advance_to_awaiting_overwatcher(
    project_id: int,
    work_artifacts: Dict[str, Any],
) -> Optional[SpecFlowState]:
    """Advance flow to awaiting Overwatcher stage."""
    state = _FLOW_STATES.get(project_id)
    if not state:
        return None
    
    state.stage = SpecFlowStage.AWAITING_OVERWATCHER
    state.work_artifacts = work_artifacts
    set_flow_state(state)
    return state


def complete_flow(project_id: int) -> None:
    """Mark flow as complete."""
    state = _FLOW_STATES.get(project_id)
    if state:
        state.stage = SpecFlowStage.COMPLETE
        set_flow_state(state)


def cancel_flow(project_id: int) -> None:
    """Cancel/abandon flow."""
    state = _FLOW_STATES.get(project_id)
    if state:
        state.stage = SpecFlowStage.CANCELLED
        set_flow_state(state)


# =============================================================================
# ROUTING HELPERS
# =============================================================================

def should_route_to_spec_gate(project_id: int) -> bool:
    """Check if message should route to Spec Gate (mid-clarification)."""
    state = get_active_flow(project_id)
    if not state:
        return False
    return state.stage == SpecFlowStage.SPEC_GATE_QUESTIONS


def should_route_to_critical_pipeline(project_id: int) -> bool:
    """Check if message should route to Critical Pipeline."""
    state = get_active_flow(project_id)
    if not state:
        return False
    return state.stage == SpecFlowStage.SPEC_VALIDATED


def should_route_to_overwatcher(project_id: int) -> bool:
    """Check if message should route to Overwatcher."""
    state = get_active_flow(project_id)
    if not state:
        return False
    return state.stage == SpecFlowStage.AWAITING_OVERWATCHER


def get_spot_for_project(project_id: int) -> Optional[Dict[str, Any]]:
    """Get SPoT (spec_id, spec_hash) for a project if available."""
    state = get_active_flow(project_id)
    if not state or not state.spec_id:
        return None
    return {
        "spec_id": state.spec_id,
        "spec_hash": state.spec_hash,
        "spec_version": state.spec_version,
    }


__all__ = [
    "SpecFlowStage",
    "SpecFlowState",
    "get_active_flow",
    "set_flow_state",
    "clear_flow_state",
    "start_weaver_flow",
    "advance_to_spec_gate_questions",
    "advance_to_spec_validated",
    "advance_to_awaiting_overwatcher",
    "complete_flow",
    "cancel_flow",
    "should_route_to_spec_gate",
    "should_route_to_critical_pipeline",
    "should_route_to_overwatcher",
    "get_spot_for_project",
]
