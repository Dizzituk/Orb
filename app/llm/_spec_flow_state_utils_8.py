from __future__ import annotations
from typing import Any, Dict, Optional, Set


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
    from .spec_flow_state import SpecFlowState, _FLOW_STATES, set_flow_state
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

def save_confirmed_design_prefs(
    project_id: int,
    prefs: Dict[str, str],
) -> Optional[SpecFlowState]:
    """
    Save confirmed design prefs that persist across weave runs.
    
    These are NOT cleared when weave completes - they stick for the project.
    """
    from .spec_flow_state import SpecFlowStage, SpecFlowState, _FLOW_STATES, set_flow_state
    state = _FLOW_STATES.get(project_id)
    if not state:
        state = SpecFlowState(project_id=project_id, stage=SpecFlowStage.AWAITING_SPEC_GATE_CONFIRM)
    
    # Merge with existing prefs (new values override)
    state.confirmed_design_prefs.update(prefs)
    set_flow_state(state)
    
    print(f"[FLOW_STATE] Saved confirmed design prefs for project {project_id}: {prefs}")
    return state

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
    from .spec_flow_state import SpecFlowStage, SpecFlowState, _FLOW_STATES, set_flow_state
    state = _FLOW_STATES.get(project_id)
    if not state:
        state = SpecFlowState(project_id=project_id, stage=SpecFlowStage.AWAITING_SPEC_GATE_CONFIRM)
    
    state.last_weave_message_count = message_count
    state.last_weave_output = weave_output
    set_flow_state(state)
    
    print(f"[FLOW_STATE] Saved weave checkpoint for project {project_id}: {message_count} messages")
    return state

def save_woven_user_hashes(
    project_id: int,
    hashes: Set[str],
) -> Optional[SpecFlowState]:
    """
    Save the set of user message hashes that have been woven.
    
    This provides durable tracking of which messages are already in the spec,
    regardless of message ordering or count drift.
    """
    from .spec_flow_state import SpecFlowStage, SpecFlowState, _FLOW_STATES, set_flow_state
    state = _FLOW_STATES.get(project_id)
    if not state:
        state = SpecFlowState(project_id=project_id, stage=SpecFlowStage.AWAITING_SPEC_GATE_CONFIRM)
    
    # Union with existing hashes (don't replace - accumulate)
    state.woven_user_hashes = state.woven_user_hashes.union(hashes)
    set_flow_state(state)
    
    print(f"[FLOW_STATE] Saved woven user hashes for project {project_id}: {len(state.woven_user_hashes)} total")
    return state

def should_route_to_weaver_continuation(project_id: int) -> bool:
    """Check if message should route to Weaver continuation (mid-design-questions)."""
    from .spec_flow_state import SpecFlowStage, get_active_flow
    state = get_active_flow(project_id)
    if not state:
        return False
    return state.stage == SpecFlowStage.WEAVER_DESIGN_QUESTIONS

def advance_to_spec_gate_questions(
    project_id: int,
    job_id: str,
    spec_id: str,
    spec_hash: str,
    questions: list,
    clarification_round: int = 1,
) -> Optional[SpecFlowState]:
    """Advance flow to Spec Gate questions stage."""
    from .spec_flow_state import SpecFlowStage, SpecFlowState, _FLOW_STATES, set_flow_state
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
    from .spec_flow_state import SpecFlowStage, SpecFlowState, _FLOW_STATES, set_flow_state
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
    from .spec_flow_state import SpecFlowStage, SpecFlowState, _FLOW_STATES, set_flow_state
    state = _FLOW_STATES.get(project_id)
    if not state:
        return None
    
    state.stage = SpecFlowStage.AWAITING_OVERWATCHER
    state.work_artifacts = work_artifacts
    set_flow_state(state)
    return state
