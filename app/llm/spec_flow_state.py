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

v1.3 (2026-01-20): HASH-BASED DELTA TRACKING
- Added woven_user_hashes field for durable message deduplication
- Index-based slicing was brittle and caused wrong messages to be extracted
- Hash-based tracking guarantees correct delta detection

v1.2 (2026-01-20): Persistent prefs and checkpoints
v1.1 (2026-01-20): Added WEAVER_DESIGN_QUESTIONS stage for design question flow
v1.0 (2026-01): Initial implementation
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Any, List, Tuple, Set
from enum import Enum
from app.llm._spec_flow_state_utils import clear_confirmed_design_prefs, clear_flow_state, clear_weave_checkpoint, clear_woven_user_hashes, get_confirmed_design_prefs, get_spot_for_project, get_weave_checkpoint, get_woven_user_hashes
from app.llm._spec_flow_state_utils import cancel_flow, check_weaver_answer_keywords, clear_weaver_design_questions, complete_flow, get_weaver_design_state, should_route_to_critical_pipeline, should_route_to_overwatcher, should_route_to_spec_gate
from app.llm._spec_flow_state_utils import advance_to_awaiting_overwatcher, advance_to_spec_gate_questions, advance_to_spec_validated, capture_weaver_answers, save_confirmed_design_prefs, save_weave_checkpoint, save_woven_user_hashes, should_route_to_weaver_continuation

logger = logging.getLogger(__name__)


class SpecFlowStage(str, Enum):
    """Current stage in the spec flow."""
    # Weaver is waiting for design question answers
    WEAVER_DESIGN_QUESTIONS = "weaver_design_questions"
    
    # After Weaver generates spec, awaiting user to send to Spec Gate
    AWAITING_SPEC_GATE_CONFIRM = "awaiting_spec_gate_confirm"
    
    # Spec Gate is asking clarification questions
    SPEC_GATE_QUESTIONS = "spec_gate_questions"
    
    # Spec validated, awaiting user to run critical pipeline
    SPEC_VALIDATED = "spec_validated"
    
    # Spec segmented, awaiting user to run segment loop (Phase 2)
    SPEC_SEGMENTED = "spec_segmented"
    
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
    
    # v3.9.1: Vision context from Gemini screenshot analysis
    # This allows SpecGate classifier to identify USER-VISIBLE UI elements
    weaver_vision_context: Optional[str] = None
    
    # v1.1: Weaver design questions state (temporary during question flow)
    weaver_pending_questions: Dict[str, str] = field(default_factory=dict)  # type → question text
    weaver_answer_keywords: Dict[str, List[str]] = field(default_factory=dict)  # type → keywords
    weaver_captured_answers: Dict[str, str] = field(default_factory=dict)  # type → captured answer
    
    # v1.2: Persistent design prefs (survives across weave runs)
    confirmed_design_prefs: Dict[str, str] = field(default_factory=dict)  # type → confirmed value
    
    # v1.2: Incremental weave tracking (DEPRECATED - kept for compatibility)
    last_weave_message_count: int = 0  # How many messages were processed in last weave
    last_weave_output: Optional[str] = None  # The previous woven job description
    
    # v1.3: Hash-based delta tracking (replaces index-based slicing)
    woven_user_hashes: Set[str] = field(default_factory=set)  # Hashes of already-woven user messages
    
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
            "weaver_vision_context": self.weaver_vision_context,
            "weaver_pending_questions": self.weaver_pending_questions,
            "weaver_answer_keywords": self.weaver_answer_keywords,
            "weaver_captured_answers": self.weaver_captured_answers,
            "confirmed_design_prefs": self.confirmed_design_prefs,
            "last_weave_message_count": self.last_weave_message_count,
            "last_weave_output": self.last_weave_output,
            "woven_user_hashes": list(self.woven_user_hashes),  # Convert set to list for JSON
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
            weaver_vision_context=data.get("weaver_vision_context"),
            weaver_pending_questions=data.get("weaver_pending_questions", {}),
            weaver_answer_keywords=data.get("weaver_answer_keywords", {}),
            weaver_captured_answers=data.get("weaver_captured_answers", {}),
            confirmed_design_prefs=data.get("confirmed_design_prefs", {}),
            last_weave_message_count=data.get("last_weave_message_count", 0),
            last_weave_output=data.get("last_weave_output"),
            woven_user_hashes=set(data.get("woven_user_hashes", [])),  # Convert list back to set
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


def start_weaver_flow(
    project_id: int,
    weaver_spec_id: str,
    weaver_job_description: Optional[str] = None,
    vision_context: Optional[str] = None,
) -> SpecFlowState:
    """Start a new flow after Weaver generates a spec/job description.
    
    v3.0: Now accepts weaver_job_description for simple Weaver output.
    v1.2: PRESERVES existing confirmed_design_prefs and weave checkpoint!
    v1.3: PRESERVES woven_user_hashes for hash-based delta tracking!
    v3.9.1: Now accepts vision_context for intelligent UI classification in SpecGate.
    """
    # Get existing state to preserve prefs and checkpoint
    existing = _FLOW_STATES.get(project_id)
    
    if existing:
        # UPDATE existing state, preserving prefs, checkpoint, AND hashes
        existing.stage = SpecFlowStage.AWAITING_SPEC_GATE_CONFIRM
        existing.weaver_spec_id = weaver_spec_id
        existing.weaver_job_description = weaver_job_description
        existing.weaver_vision_context = vision_context  # v3.9.1
        # KEEP: confirmed_design_prefs, last_weave_message_count, last_weave_output, woven_user_hashes
        set_flow_state(existing)
        print(f"[FLOW_STATE] Updated flow for project {project_id}, preserving prefs: {list(existing.confirmed_design_prefs.keys())}, hashes: {len(existing.woven_user_hashes)}, vision_context: {len(vision_context or '')} chars")
        return existing
    else:
        # Create new state
        state = SpecFlowState(
            project_id=project_id,
            stage=SpecFlowStage.AWAITING_SPEC_GATE_CONFIRM,
            weaver_spec_id=weaver_spec_id,
            weaver_job_description=weaver_job_description,
            weaver_vision_context=vision_context,  # v3.9.1
        )
        set_flow_state(state)
        return state


# =============================================================================
# WEAVER DESIGN QUESTIONS FLOW (v1.1)
# =============================================================================

def extract_keywords_from_question(question_text: str) -> List[str]:
    """
    Extract answer keywords from a question's example text.
    
    From: "Do you have a preferred layout? (e.g., sidebar, top nav, centered, grid)"
    Extract: ["sidebar", "top nav", "centered", "grid"]
    
    Also handles:
    - Parentheses: (e.g., X, Y, Z)
    - "such as": such as X, Y, or Z
    - "like": like X, Y, Z
    """
    keywords = []
    text = question_text.lower()
    
    # Pattern 1: (e.g., X, Y, Z) or (X, Y, Z)
    paren_match = re.search(r'\((?:e\.?g\.?,?\s*)?([^)]+)\)', text)
    if paren_match:
        inner = paren_match.group(1)
        # Split on commas, "or", clean up
        parts = re.split(r',\s*|\s+or\s+', inner)
        keywords.extend([p.strip() for p in parts if p.strip()])
    
    # Pattern 2: "such as X, Y, or Z"
    such_as_match = re.search(r'such as\s+([^?.]+)', text)
    if such_as_match:
        inner = such_as_match.group(1)
        parts = re.split(r',\s*|\s+or\s+', inner)
        keywords.extend([p.strip() for p in parts if p.strip()])
    
    # Pattern 3: "like X, Y, or Z"
    like_match = re.search(r'\blike\s+([^?.]+)', text)
    if like_match:
        inner = like_match.group(1)
        parts = re.split(r',\s*|\s+or\s+', inner)
        keywords.extend([p.strip() for p in parts if p.strip()])
    
    # Deduplicate while preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)
    
    return unique_keywords


def set_weaver_design_questions(
    project_id: int,
    questions: Dict[str, str],
) -> SpecFlowState:
    """
    Set weaver to waiting state with pending design questions.
    
    Args:
        project_id: Project ID
        questions: Dict mapping question_type (color/style/layout) to question text
    
    Returns:
        Updated flow state
    """
    state = _FLOW_STATES.get(project_id)
    if not state:
        state = SpecFlowState(project_id=project_id, stage=SpecFlowStage.WEAVER_DESIGN_QUESTIONS)
    
    state.stage = SpecFlowStage.WEAVER_DESIGN_QUESTIONS
    state.weaver_pending_questions = questions
    
    # Extract keywords from each question
    state.weaver_answer_keywords = {}
    for q_type, q_text in questions.items():
        keywords = extract_keywords_from_question(q_text)
        state.weaver_answer_keywords[q_type] = keywords
        print(f"[FLOW_STATE] Extracted keywords for {q_type}: {keywords}")
    
    set_flow_state(state)
    print(f"[FLOW_STATE] Set WEAVER_DESIGN_QUESTIONS for project {project_id}")
    return state


# =============================================================================
# CONFIRMED DESIGN PREFS (v1.2) - Persist across weave runs
# =============================================================================


# =============================================================================
# WEAVE CHECKPOINT (v1.2) - For incremental weaving
# =============================================================================


# =============================================================================
# WOVEN USER HASHES (v1.3) - Hash-based delta tracking
# =============================================================================


# =============================================================================
# SPEC GATE FLOW FUNCTIONS (existing)
# =============================================================================


def advance_to_spec_segmented(
    project_id: int,
    spec_id: str,
    spec_hash: str,
    job_id: str,
    total_segments: int,
    spec_version: int = 1,
) -> Optional[SpecFlowState]:
    """Advance flow to spec segmented stage (Phase 2 — segments ready for execution)."""
    state = _FLOW_STATES.get(project_id)
    if not state:
        state = SpecFlowState(project_id=project_id, stage=SpecFlowStage.SPEC_SEGMENTED)

    state.stage = SpecFlowStage.SPEC_SEGMENTED
    state.spec_id = spec_id
    state.spec_hash = spec_hash
    state.spec_version = spec_version
    state.open_questions = []
    # Store segment metadata in work_artifacts for downstream access
    state.work_artifacts = {
        "job_id": job_id,
        "total_segments": total_segments,
        "segmented": True,
    }
    set_flow_state(state)
    return state


# =============================================================================
# ROUTING HELPERS
# =============================================================================


__all__ = [
    "SpecFlowStage",
    "SpecFlowState",
    "get_active_flow",
    "set_flow_state",
    "clear_flow_state",
    "start_weaver_flow",
    # Weaver design questions (v1.1)
    "extract_keywords_from_question",
    "set_weaver_design_questions",
    "check_weaver_answer_keywords",
    "capture_weaver_answers",
    "get_weaver_design_state",
    "clear_weaver_design_questions",
    "should_route_to_weaver_continuation",
    # Confirmed design prefs (v1.2)
    "save_confirmed_design_prefs",
    "get_confirmed_design_prefs",
    "clear_confirmed_design_prefs",
    # Weave checkpoint (v1.2)
    "save_weave_checkpoint",
    "get_weave_checkpoint",
    "clear_weave_checkpoint",
    # Woven user hashes (v1.3)
    "save_woven_user_hashes",
    "get_woven_user_hashes",
    "clear_woven_user_hashes",
    # Spec Gate flow
    "advance_to_spec_gate_questions",
    "advance_to_spec_validated",
    "advance_to_spec_segmented",
    "advance_to_awaiting_overwatcher",
    "complete_flow",
    "cancel_flow",
    "should_route_to_spec_gate",
    "should_route_to_critical_pipeline",
    "should_route_to_overwatcher",
    "get_spot_for_project",
]
