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
    v1.2: PRESERVES existing confirmed_design_prefs and weave checkpoint!
    v1.3: PRESERVES woven_user_hashes for hash-based delta tracking!
    """
    # Get existing state to preserve prefs and checkpoint
    existing = _FLOW_STATES.get(project_id)
    
    if existing:
        # UPDATE existing state, preserving prefs, checkpoint, AND hashes
        existing.stage = SpecFlowStage.AWAITING_SPEC_GATE_CONFIRM
        existing.weaver_spec_id = weaver_spec_id
        existing.weaver_job_description = weaver_job_description
        # KEEP: confirmed_design_prefs, last_weave_message_count, last_weave_output, woven_user_hashes
        set_flow_state(existing)
        print(f"[FLOW_STATE] Updated flow for project {project_id}, preserving prefs: {list(existing.confirmed_design_prefs.keys())}, hashes: {len(existing.woven_user_hashes)}")
        return existing
    else:
        # Create new state
        state = SpecFlowState(
            project_id=project_id,
            stage=SpecFlowStage.AWAITING_SPEC_GATE_CONFIRM,
            weaver_spec_id=weaver_spec_id,
            weaver_job_description=weaver_job_description,
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


def check_weaver_answer_keywords(
    project_id: int,
    message: str,
) -> Tuple[bool, Dict[str, str]]:
    """
    Check if user message contains answer keywords for pending questions.
    
    Args:
        project_id: Project ID
        message: User's message
    
    Returns:
        Tuple of (has_any_keywords, captured_answers_dict)
        captured_answers_dict maps question_type → matched keyword
    """
    state = _FLOW_STATES.get(project_id)
    if not state or state.stage != SpecFlowStage.WEAVER_DESIGN_QUESTIONS:
        return False, {}
    
    msg_lower = message.lower()
    captured = {}
    
    for q_type, keywords in state.weaver_answer_keywords.items():
        # Skip if already answered
        if q_type in state.weaver_captured_answers:
            continue
        
        # Check each keyword
        for kw in keywords:
            # Match as whole word or phrase
            # Handle both "sidebar" and "side bar" variants
            kw_variants = [kw, kw.replace(" ", ""), kw.replace("-", " ")]
            
            for variant in kw_variants:
                if variant in msg_lower:
                    captured[q_type] = kw  # Store original keyword
                    print(f"[FLOW_STATE] Captured answer for {q_type}: '{kw}' (matched '{variant}')")
                    break
            
            if q_type in captured:
                break
    
    has_any = len(captured) > 0
    return has_any, captured


def capture_weaver_answers(
    project_id: int,
    answers: Dict[str, str],
) -> Optional[SpecFlowState]:
    """
    Store captured answers in flow state.
    
    Args:
        project_id: Project ID
        answers: Dict mapping question_type → captured answer
    
    Returns:
        Updated flow state or None if no active flow
    """
    state = _FLOW_STATES.get(project_id)
    if not state:
        return None
    
    # Merge new answers with existing
    state.weaver_captured_answers.update(answers)
    
    # Remove answered questions from pending
    for q_type in answers:
        if q_type in state.weaver_pending_questions:
            del state.weaver_pending_questions[q_type]
    
    set_flow_state(state)
    
    remaining = len(state.weaver_pending_questions)
    print(f"[FLOW_STATE] Captured {len(answers)} answers, {remaining} questions remaining")
    
    return state


def get_weaver_design_state(project_id: int) -> Optional[Dict[str, Any]]:
    """
    Get current weaver design question state.
    
    Returns dict with:
        - pending_questions: questions not yet answered
        - captured_answers: answers already captured
        - all_answered: bool whether all questions are answered
    """
    state = _FLOW_STATES.get(project_id)
    if not state or state.stage != SpecFlowStage.WEAVER_DESIGN_QUESTIONS:
        return None
    
    return {
        "pending_questions": state.weaver_pending_questions,
        "captured_answers": state.weaver_captured_answers,
        "answer_keywords": state.weaver_answer_keywords,
        "all_answered": len(state.weaver_pending_questions) == 0,
    }


def clear_weaver_design_questions(project_id: int) -> None:
    """Clear weaver design question state (after weave completes)."""
    state = _FLOW_STATES.get(project_id)
    if state:
        state.weaver_pending_questions = {}
        state.weaver_answer_keywords = {}
        state.weaver_captured_answers = {}
        # Don't change stage here - let weaver do that
        set_flow_state(state)
        print(f"[FLOW_STATE] Cleared weaver design questions for project {project_id}")


# =============================================================================
# CONFIRMED DESIGN PREFS (v1.2) - Persist across weave runs
# =============================================================================

def save_confirmed_design_prefs(
    project_id: int,
    prefs: Dict[str, str],
) -> Optional[SpecFlowState]:
    """
    Save confirmed design prefs that persist across weave runs.
    
    These are NOT cleared when weave completes - they stick for the project.
    """
    state = _FLOW_STATES.get(project_id)
    if not state:
        state = SpecFlowState(project_id=project_id, stage=SpecFlowStage.AWAITING_SPEC_GATE_CONFIRM)
    
    # Merge with existing prefs (new values override)
    state.confirmed_design_prefs.update(prefs)
    set_flow_state(state)
    
    print(f"[FLOW_STATE] Saved confirmed design prefs for project {project_id}: {prefs}")
    return state


def get_confirmed_design_prefs(project_id: int) -> Dict[str, str]:
    """
    Get confirmed design prefs for a project.
    
    Returns empty dict if no prefs saved.
    """
    state = _FLOW_STATES.get(project_id)
    if not state:
        return {}
    return state.confirmed_design_prefs.copy()


def clear_confirmed_design_prefs(project_id: int) -> None:
    """
    Clear confirmed design prefs (e.g., when starting a completely new job).
    """
    state = _FLOW_STATES.get(project_id)
    if state:
        state.confirmed_design_prefs = {}
        set_flow_state(state)
        print(f"[FLOW_STATE] Cleared confirmed design prefs for project {project_id}")


# =============================================================================
# WEAVE CHECKPOINT (v1.2) - For incremental weaving
# =============================================================================

def save_weave_checkpoint(
    project_id: int,
    message_count: int,
    weave_output: str,
) -> Optional[SpecFlowState]:
    """
    Save checkpoint after weave completes.
    
    This allows subsequent weaves to only process NEW messages.
    NOTE: v1.3 uses hash-based tracking instead of message_count for delta detection.
    """
    state = _FLOW_STATES.get(project_id)
    if not state:
        state = SpecFlowState(project_id=project_id, stage=SpecFlowStage.AWAITING_SPEC_GATE_CONFIRM)
    
    state.last_weave_message_count = message_count
    state.last_weave_output = weave_output
    set_flow_state(state)
    
    print(f"[FLOW_STATE] Saved weave checkpoint for project {project_id}: {message_count} messages")
    return state


def get_weave_checkpoint(project_id: int) -> Optional[Dict[str, Any]]:
    """
    Get weave checkpoint for a project.
    
    Returns dict with:
        - message_count: how many messages were in last weave
        - last_output: the previous woven job description
    
    Returns None if no checkpoint exists.
    """
    state = _FLOW_STATES.get(project_id)
    if not state or state.last_weave_message_count == 0:
        return None
    
    return {
        "message_count": state.last_weave_message_count,
        "last_output": state.last_weave_output,
    }


def clear_weave_checkpoint(project_id: int) -> None:
    """
    Clear weave checkpoint (e.g., when starting a completely new job).
    """
    state = _FLOW_STATES.get(project_id)
    if state:
        state.last_weave_message_count = 0
        state.last_weave_output = None
        set_flow_state(state)
        print(f"[FLOW_STATE] Cleared weave checkpoint for project {project_id}")


# =============================================================================
# WOVEN USER HASHES (v1.3) - Hash-based delta tracking
# =============================================================================

def save_woven_user_hashes(
    project_id: int,
    hashes: Set[str],
) -> Optional[SpecFlowState]:
    """
    Save the set of user message hashes that have been woven.
    
    This provides durable tracking of which messages are already in the spec,
    regardless of message ordering or count drift.
    """
    state = _FLOW_STATES.get(project_id)
    if not state:
        state = SpecFlowState(project_id=project_id, stage=SpecFlowStage.AWAITING_SPEC_GATE_CONFIRM)
    
    # Union with existing hashes (don't replace - accumulate)
    state.woven_user_hashes = state.woven_user_hashes.union(hashes)
    set_flow_state(state)
    
    print(f"[FLOW_STATE] Saved woven user hashes for project {project_id}: {len(state.woven_user_hashes)} total")
    return state


def get_woven_user_hashes(project_id: int) -> Set[str]:
    """
    Get the set of user message hashes that have been woven.
    
    Returns empty set if no hashes saved.
    """
    state = _FLOW_STATES.get(project_id)
    if not state:
        return set()
    return state.woven_user_hashes.copy()


def clear_woven_user_hashes(project_id: int) -> None:
    """
    Clear woven user hashes (e.g., when starting a completely new job).
    """
    state = _FLOW_STATES.get(project_id)
    if state:
        state.woven_user_hashes = set()
        set_flow_state(state)
        print(f"[FLOW_STATE] Cleared woven user hashes for project {project_id}")


def should_route_to_weaver_continuation(project_id: int) -> bool:
    """Check if message should route to Weaver continuation (mid-design-questions)."""
    state = get_active_flow(project_id)
    if not state:
        return False
    return state.stage == SpecFlowStage.WEAVER_DESIGN_QUESTIONS


# =============================================================================
# SPEC GATE FLOW FUNCTIONS (existing)
# =============================================================================

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
    "advance_to_awaiting_overwatcher",
    "complete_flow",
    "cancel_flow",
    "should_route_to_spec_gate",
    "should_route_to_critical_pipeline",
    "should_route_to_overwatcher",
    "get_spot_for_project",
]
