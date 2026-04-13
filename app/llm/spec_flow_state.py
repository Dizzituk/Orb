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

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, List, Set
from enum import Enum
from app.llm._spec_flow_state_utils_6 import clear_confirmed_design_prefs, clear_flow_state, clear_weave_checkpoint, clear_woven_user_hashes, get_confirmed_design_prefs, get_spot_for_project, get_weave_checkpoint, get_woven_user_hashes
from app.llm._spec_flow_state_utils_7 import cancel_flow, check_weaver_answer_keywords, clear_weaver_design_questions, complete_flow, get_weaver_design_state, should_route_to_critical_pipeline, should_route_to_overwatcher, should_route_to_spec_gate
from app.llm._spec_flow_state_utils_8 import advance_to_awaiting_overwatcher, advance_to_spec_gate_questions, advance_to_spec_validated, capture_weaver_answers, save_confirmed_design_prefs, save_weave_checkpoint, save_woven_user_hashes, should_route_to_weaver_continuation

logger = logging.getLogger(__name__)

_STATE_DIR = Path(__file__).resolve().parent / "_spec_flow_state"
_STATE_DIR.mkdir(parents=True, exist_ok=True)


class SpecFlowStage(str, Enum):
    """Current stage in the spec flow."""
    WEAVER_DESIGN_QUESTIONS = "weaver_design_questions"
    AWAITING_SPEC_GATE_CONFIRM = "awaiting_spec_gate_confirm"
    SPEC_GATE_QUESTIONS = "spec_gate_questions"
    SPEC_VALIDATED = "spec_validated"
    SPEC_SEGMENTED = "spec_segmented"
    AWAITING_OVERWATCHER = "awaiting_overwatcher"
    COMPLETE = "complete"
    CANCELLED = "cancelled"


@dataclass
class SpecFlowState:
    project_id: int
    stage: SpecFlowStage
    job_id: Optional[str] = None
    weaver_spec_id: Optional[str] = None
    weaver_job_description: Optional[str] = None
    weaver_vision_context: Optional[str] = None
    weaver_pending_questions: Dict[str, str] = field(default_factory=dict)
    weaver_answer_keywords: Dict[str, List[str]] = field(default_factory=dict)
    weaver_captured_answers: Dict[str, str] = field(default_factory=dict)
    confirmed_design_prefs: Dict[str, str] = field(default_factory=dict)
    last_weave_message_count: int = 0
    last_weave_output: Optional[str] = None
    woven_user_hashes: Set[str] = field(default_factory=set)
    spec_id: Optional[str] = None
    spec_hash: Optional[str] = None
    spec_version: int = 1
    clarification_round: int = 0
    open_questions: list = field(default_factory=list)
    work_artifacts: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
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
            "woven_user_hashes": list(self.woven_user_hashes),
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
            woven_user_hashes=set(data.get("woven_user_hashes", [])),
            spec_id=data.get("spec_id"),
            spec_hash=data.get("spec_hash"),
            spec_version=data.get("spec_version", 1),
            clarification_round=data.get("clarification_round", 0),
            open_questions=data.get("open_questions", []),
            work_artifacts=data.get("work_artifacts", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(timezone.utc),
        )


def _state_path(project_id: int) -> Path:
    return _STATE_DIR / f"project_{project_id}.json"


def _persist_flow_state(state: SpecFlowState) -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    _state_path(state.project_id).write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def _load_flow_state(project_id: int) -> Optional[SpecFlowState]:
    path = _state_path(project_id)
    if not path.exists():
        return None
    try:
        return SpecFlowState.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        logger.exception("[spec_flow] Failed to load state for project %s", project_id)
        return None


def _is_active(state: Optional[SpecFlowState]) -> bool:
    return bool(state and state.stage not in (SpecFlowStage.COMPLETE, SpecFlowStage.CANCELLED))


_FLOW_STATES: Dict[int, SpecFlowState] = {}


def get_active_flow(project_id: int) -> Optional[SpecFlowState]:
    state = _FLOW_STATES.get(project_id)
    if not _is_active(state):
        state = _load_flow_state(project_id)
        if _is_active(state):
            _FLOW_STATES[project_id] = state
        else:
            return None
    return state


def set_flow_state(state: SpecFlowState) -> None:
    state.updated_at = datetime.now(timezone.utc)
    _FLOW_STATES[state.project_id] = state
    _persist_flow_state(state)
    logger.debug(f"[spec_flow] Set state for project {state.project_id}: {state.stage.value}")


def start_weaver_flow(
    project_id: int,
    weaver_spec_id: str,
    weaver_job_description: Optional[str] = None,
    vision_context: Optional[str] = None,
) -> SpecFlowState:
    existing = get_active_flow(project_id)
    if existing:
        existing.stage = SpecFlowStage.AWAITING_SPEC_GATE_CONFIRM
        existing.weaver_spec_id = weaver_spec_id
        existing.weaver_job_description = weaver_job_description
        existing.weaver_vision_context = vision_context
        set_flow_state(existing)
        print(f"[FLOW_STATE] Updated flow for project {project_id}, preserving prefs: {list(existing.confirmed_design_prefs.keys())}, hashes: {len(existing.woven_user_hashes)}, vision_context: {len(vision_context or '')} chars")
        return existing
    state = SpecFlowState(project_id=project_id, stage=SpecFlowStage.AWAITING_SPEC_GATE_CONFIRM, weaver_spec_id=weaver_spec_id, weaver_job_description=weaver_job_description, weaver_vision_context=vision_context)
    set_flow_state(state)
    return state


def extract_keywords_from_question(question_text: str) -> List[str]:
    keywords = []
    text = question_text.lower()
    paren_match = re.search(r'\((?:e\.?g\.?\,?\s*)?([^)]+)\)', text)
    if paren_match:
        parts = re.split(r',\s*|\s+or\s+', paren_match.group(1))
        keywords.extend([p.strip() for p in parts if p.strip()])
    such_as_match = re.search(r'such as\s+([^?.]+)', text)
    if such_as_match:
        parts = re.split(r',\s*|\s+or\s+', such_as_match.group(1))
        keywords.extend([p.strip() for p in parts if p.strip()])
    like_match = re.search(r'\blike\s+([^?.]+)', text)
    if like_match:
        parts = re.split(r',\s*|\s+or\s+', like_match.group(1))
        keywords.extend([p.strip() for p in parts if p.strip()])
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)
    return unique_keywords


def set_weaver_design_questions(project_id: int, questions: Dict[str, str]) -> SpecFlowState:
    state = get_active_flow(project_id) or SpecFlowState(project_id=project_id, stage=SpecFlowStage.WEAVER_DESIGN_QUESTIONS)
    state.stage = SpecFlowStage.WEAVER_DESIGN_QUESTIONS
    state.weaver_pending_questions = questions
    state.weaver_answer_keywords = {}
    for q_type, q_text in questions.items():
        keywords = extract_keywords_from_question(q_text)
        state.weaver_answer_keywords[q_type] = keywords
        print(f"[FLOW_STATE] Extracted keywords for {q_type}: {keywords}")
    set_flow_state(state)
    print(f"[FLOW_STATE] Set WEAVER_DESIGN_QUESTIONS for project {project_id}")
    return state


def advance_to_spec_segmented(project_id: int, spec_id: str, spec_hash: str, job_id: str, total_segments: int, spec_version: int = 1) -> Optional[SpecFlowState]:
    state = get_active_flow(project_id) or SpecFlowState(project_id=project_id, stage=SpecFlowStage.SPEC_SEGMENTED)
    state.stage = SpecFlowStage.SPEC_SEGMENTED
    state.spec_id = spec_id
    state.spec_hash = spec_hash
    state.spec_version = spec_version
    state.open_questions = []
    state.work_artifacts = {"job_id": job_id, "total_segments": total_segments, "segmented": True}
    set_flow_state(state)
    return state


__all__ = [
    "SpecFlowStage",
    "SpecFlowState",
    "get_active_flow",
    "set_flow_state",
    "clear_flow_state",
    "start_weaver_flow",
    "extract_keywords_from_question",
    "set_weaver_design_questions",
    "check_weaver_answer_keywords",
    "capture_weaver_answers",
    "get_weaver_design_state",
    "clear_weaver_design_questions",
    "should_route_to_weaver_continuation",
    "save_confirmed_design_prefs",
    "get_confirmed_design_prefs",
    "clear_confirmed_design_prefs",
    "save_weave_checkpoint",
    "get_weave_checkpoint",
    "clear_weave_checkpoint",
    "save_woven_user_hashes",
    "get_woven_user_hashes",
    "clear_woven_user_hashes",
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
