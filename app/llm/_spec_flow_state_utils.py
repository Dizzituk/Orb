from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Set
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


def clear_flow_state(project_id: int) -> None:
    """Clear flow state for a project."""
    if project_id in _FLOW_STATES:
        del _FLOW_STATES[project_id]
        logger.debug(f"[spec_flow] Cleared state for project {project_id}")

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


# Auto-generated re-exports for symbols in numbered _utils files
_REEXPORT_MAP = {
    "cancel_flow": "_spec_flow_state_utils_4",
    "check_weaver_answer_keywords": "_spec_flow_state_utils_4",
    "clear_weaver_design_questions": "_spec_flow_state_utils_4",
    "complete_flow": "_spec_flow_state_utils_4",
    "get_weaver_design_state": "_spec_flow_state_utils_4",
    "should_route_to_critical_pipeline": "_spec_flow_state_utils_4",
    "should_route_to_overwatcher": "_spec_flow_state_utils_4",
    "should_route_to_spec_gate": "_spec_flow_state_utils_4",
    "advance_to_awaiting_overwatcher": "_spec_flow_state_utils_5",
    "advance_to_spec_gate_questions": "_spec_flow_state_utils_5",
    "advance_to_spec_validated": "_spec_flow_state_utils_5",
    "capture_weaver_answers": "_spec_flow_state_utils_5",
    "save_confirmed_design_prefs": "_spec_flow_state_utils_5",
    "save_weave_checkpoint": "_spec_flow_state_utils_5",
    "save_woven_user_hashes": "_spec_flow_state_utils_5",
    "should_route_to_weaver_continuation": "_spec_flow_state_utils_5",
}

def __getattr__(name):
    if name in _REEXPORT_MAP:
        import importlib
        mod = importlib.import_module(f"app.llm.{_REEXPORT_MAP[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
