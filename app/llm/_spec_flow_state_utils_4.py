from __future__ import annotations
from app.llm.spec_flow_state import SpecFlowStage, get_active_flow, set_flow_state
from typing import Any, Dict, Optional, Tuple
from .spec_flow_state import _FLOW_STATES


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
