# FILE: app/pot_spec/clarification_state.py
"""Clarification State Management (Job 4)

Tracks:
- Asked question signatures (dedupe)
- Rounds per hole signature (3-round cap)
- Confirmation gate state

Storage: jobs/<job_id>/governance/clarification_state.json

This is governance metadata, NOT part of the immutable PoT Spec.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

MAX_ROUNDS_PER_HOLE = 3  # After 3 rounds on same hole → HARD_STOP
GOVERNANCE_DIR = "governance"
CLARIFICATION_FILE = "clarification_state.json"


class ClarificationDecision(str, Enum):
    """Decision after evaluating clarification state."""
    CONTINUE = "continue"           # Can ask more questions
    READY_FOR_CONFIRM = "ready"     # No more questions, show spec + await confirm
    HARD_STOP = "hard_stop"         # 3-round cap hit, escalate to human
    ALREADY_ASKED = "already_asked" # Question was already asked (dedupe)


# =============================================================================
# Hole Signature (reuse from overwatcher_schemas pattern)
# =============================================================================

class HoleType(str, Enum):
    """Types of spec holes that can trigger clarification."""
    MISSING_INFO = "missing_info"
    CONTRADICTION = "contradiction"
    AMBIGUITY = "ambiguity"
    SAFETY_GAP = "safety_gap"
    SCOPE_UNCLEAR = "scope_unclear"
    DEPENDENCY_UNKNOWN = "dependency_unknown"


def compute_question_signature(
    hole_type: HoleType,
    question_text: str,
    context_excerpt: str = "",
) -> str:
    """Compute a stable signature for a clarification question.
    
    Used for:
    - Dedupe: Don't ask the same question twice
    - Round tracking: Count attempts per unique hole
    
    Args:
        hole_type: Category of the spec hole
        question_text: The actual question being asked
        context_excerpt: Optional context (section of spec, etc.)
    
    Returns:
        SHA256 hash (first 16 chars) as signature
    """
    # Normalize
    q_norm = " ".join(question_text.lower().split())
    ctx_norm = " ".join(context_excerpt.lower().split())[:200] if context_excerpt else ""
    
    raw = f"{hole_type.value}|{q_norm}|{ctx_norm}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def compute_question_signature_simple(question_text: str) -> str:
    """Simplified signature for questions without hole type classification.
    
    Used when LLM generates open_questions without explicit hole categorization.
    """
    q_norm = " ".join(question_text.lower().split())
    return hashlib.sha256(q_norm.encode()).hexdigest()[:16]


# =============================================================================
# Clarification State Dataclass
# =============================================================================

@dataclass
class ClarificationRound:
    """Record of a single clarification round."""
    round_number: int
    spec_version: int
    questions_asked: List[str]
    question_sigs: List[str]
    asked_at: str  # ISO timestamp
    user_response: Optional[str] = None
    responded_at: Optional[str] = None


@dataclass
class ClarificationState:
    """Tracks clarification history for a job.
    
    Stored in: jobs/<job_id>/governance/clarification_state.json
    """
    job_id: str
    current_round: int = 0
    total_questions_asked: int = 0
    
    # Dedupe: all question signatures ever asked
    asked_sigs: Set[str] = field(default_factory=set)
    
    # Round tracking per hole signature
    rounds_by_sig: Dict[str, int] = field(default_factory=dict)
    
    # History of rounds
    rounds: List[ClarificationRound] = field(default_factory=list)
    
    # Confirmation gate
    ready_for_pipeline: bool = False
    user_confirmed: bool = False
    confirmed_at: Optional[str] = None
    confirmed_spec_version: Optional[int] = None
    confirmed_spec_hash: Optional[str] = None
    
    # Hard stop tracking
    hard_stopped: bool = False
    hard_stop_reason: Optional[str] = None
    hard_stop_sig: Optional[str] = None
    
    def is_question_asked(self, sig: str) -> bool:
        """Check if question was already asked (dedupe)."""
        return sig in self.asked_sigs
    
    def get_rounds_for_sig(self, sig: str) -> int:
        """Get number of rounds for a specific hole signature."""
        return self.rounds_by_sig.get(sig, 0)
    
    def would_exceed_cap(self, sig: str) -> bool:
        """Check if asking about this sig would exceed 3-round cap."""
        return self.get_rounds_for_sig(sig) >= MAX_ROUNDS_PER_HOLE
    
    def record_questions(
        self,
        questions: List[str],
        spec_version: int,
    ) -> ClarificationDecision:
        """Record new questions being asked.
        
        Returns decision about whether to proceed.
        """
        if self.hard_stopped:
            return ClarificationDecision.HARD_STOP
        
        if not questions:
            # No questions = ready for confirmation
            self.ready_for_pipeline = True
            return ClarificationDecision.READY_FOR_CONFIRM
        
        # Compute signatures for new questions
        new_sigs = []
        for q in questions:
            sig = compute_question_signature_simple(q)
            
            # Check dedupe
            if sig in self.asked_sigs:
                logger.warning(f"[clarification] Dedupe: question already asked (sig={sig})")
                continue  # Skip duplicate
            
            # Check 3-round cap
            if self.would_exceed_cap(sig):
                self.hard_stopped = True
                self.hard_stop_reason = f"Exceeded {MAX_ROUNDS_PER_HOLE}-round cap for hole"
                self.hard_stop_sig = sig
                logger.error(f"[clarification] HARD_STOP: 3-round cap exceeded for sig={sig}")
                return ClarificationDecision.HARD_STOP
            
            new_sigs.append(sig)
        
        if not new_sigs:
            # All questions were duplicates
            return ClarificationDecision.ALREADY_ASKED
        
        # Record the round
        self.current_round += 1
        now_iso = datetime.now(timezone.utc).isoformat()
        
        # Filter to only new (non-duplicate) questions
        new_questions = [q for q in questions if compute_question_signature_simple(q) in new_sigs]
        
        round_record = ClarificationRound(
            round_number=self.current_round,
            spec_version=spec_version,
            questions_asked=new_questions,
            question_sigs=new_sigs,
            asked_at=now_iso,
        )
        self.rounds.append(round_record)
        
        # Update tracking
        for sig in new_sigs:
            self.asked_sigs.add(sig)
            self.rounds_by_sig[sig] = self.rounds_by_sig.get(sig, 0) + 1
        
        self.total_questions_asked += len(new_questions)
        self.ready_for_pipeline = False
        
        return ClarificationDecision.CONTINUE
    
    def record_user_response(self, response: str) -> None:
        """Record user's response to the latest round."""
        if self.rounds:
            self.rounds[-1].user_response = response
            self.rounds[-1].responded_at = datetime.now(timezone.utc).isoformat()
    
    def confirm_for_pipeline(
        self,
        spec_version: int,
        spec_hash: str,
    ) -> bool:
        """Record user confirmation to proceed with critical pipeline.
        
        Returns True if confirmation recorded, False if not ready.
        """
        if not self.ready_for_pipeline:
            logger.warning("[clarification] Cannot confirm: not ready for pipeline")
            return False
        
        if self.hard_stopped:
            logger.warning("[clarification] Cannot confirm: hard stopped")
            return False
        
        self.user_confirmed = True
        self.confirmed_at = datetime.now(timezone.utc).isoformat()
        self.confirmed_spec_version = spec_version
        self.confirmed_spec_hash = spec_hash
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "job_id": self.job_id,
            "current_round": self.current_round,
            "total_questions_asked": self.total_questions_asked,
            "asked_sigs": list(self.asked_sigs),
            "rounds_by_sig": self.rounds_by_sig,
            "rounds": [
                {
                    "round_number": r.round_number,
                    "spec_version": r.spec_version,
                    "questions_asked": r.questions_asked,
                    "question_sigs": r.question_sigs,
                    "asked_at": r.asked_at,
                    "user_response": r.user_response,
                    "responded_at": r.responded_at,
                }
                for r in self.rounds
            ],
            "ready_for_pipeline": self.ready_for_pipeline,
            "user_confirmed": self.user_confirmed,
            "confirmed_at": self.confirmed_at,
            "confirmed_spec_version": self.confirmed_spec_version,
            "confirmed_spec_hash": self.confirmed_spec_hash,
            "hard_stopped": self.hard_stopped,
            "hard_stop_reason": self.hard_stop_reason,
            "hard_stop_sig": self.hard_stop_sig,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClarificationState":
        """Deserialize from dict."""
        state = cls(
            job_id=data.get("job_id", ""),
            current_round=data.get("current_round", 0),
            total_questions_asked=data.get("total_questions_asked", 0),
            asked_sigs=set(data.get("asked_sigs", [])),
            rounds_by_sig=data.get("rounds_by_sig", {}),
            ready_for_pipeline=data.get("ready_for_pipeline", False),
            user_confirmed=data.get("user_confirmed", False),
            confirmed_at=data.get("confirmed_at"),
            confirmed_spec_version=data.get("confirmed_spec_version"),
            confirmed_spec_hash=data.get("confirmed_spec_hash"),
            hard_stopped=data.get("hard_stopped", False),
            hard_stop_reason=data.get("hard_stop_reason"),
            hard_stop_sig=data.get("hard_stop_sig"),
        )
        
        # Reconstruct rounds
        for r_data in data.get("rounds", []):
            state.rounds.append(ClarificationRound(
                round_number=r_data.get("round_number", 0),
                spec_version=r_data.get("spec_version", 0),
                questions_asked=r_data.get("questions_asked", []),
                question_sigs=r_data.get("question_sigs", []),
                asked_at=r_data.get("asked_at", ""),
                user_response=r_data.get("user_response"),
                responded_at=r_data.get("responded_at"),
            ))
        
        return state


# =============================================================================
# File I/O
# =============================================================================

def _get_clarification_path(job_artifact_root: str, job_id: str) -> Path:
    """Get path to clarification_state.json."""
    return Path(job_artifact_root) / "jobs" / job_id / GOVERNANCE_DIR / CLARIFICATION_FILE


def load_clarification_state(
    job_artifact_root: str,
    job_id: str,
) -> ClarificationState:
    """Load clarification state from file, or create new if not exists."""
    path = _get_clarification_path(job_artifact_root, job_id)
    
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ClarificationState.from_dict(data)
        except Exception as e:
            logger.warning(f"[clarification] Failed to load state: {e}")
    
    return ClarificationState(job_id=job_id)


def save_clarification_state(
    job_artifact_root: str,
    job_id: str,
    state: ClarificationState,
) -> Path:
    """Save clarification state to file."""
    path = _get_clarification_path(job_artifact_root, job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2)
    
    logger.debug(f"[clarification] Saved state to {path}")
    return path


# =============================================================================
# Spec Summary for Confirmation Gate
# =============================================================================

def format_spec_summary_markdown(
    spec_payload: Dict[str, Any],
    spec_file_path: str,
    clarification_state: Optional[ClarificationState] = None,
) -> str:
    """Format spec as markdown summary for confirmation gate.
    
    Shows key sections for quick sanity check, plus link to full file.
    """
    lines = []
    
    # Header
    lines.append("## 📋 Spec Ready for Critical Pipeline")
    lines.append("")
    
    # Identity
    spec_id = spec_payload.get("spec_id", "?")[:8]
    spec_version = spec_payload.get("spec_version", "?")
    spec_hash = spec_payload.get("spec_hash", spec_payload.get("_spec_hash", "?"))[:12]
    job_id = spec_payload.get("job_id", "?")
    created_at = spec_payload.get("created_at", "?")
    created_by = spec_payload.get("created_by_model", "?")
    
    lines.append(f"**Job ID:** `{job_id}`")
    lines.append(f"**Spec ID:** `{spec_id}...` | **Version:** `{spec_version}` | **Hash:** `{spec_hash}...`")
    lines.append(f"**Created:** `{created_at}` by `{created_by}`")
    lines.append("")
    
    # Goal
    goal = spec_payload.get("goal", "(no goal)")
    lines.append(f"### Goal")
    lines.append(f"> {goal[:500]}{'...' if len(goal) > 500 else ''}")
    lines.append("")
    
    # Requirements summary
    reqs = spec_payload.get("requirements", {})
    must = reqs.get("must", [])
    should = reqs.get("should", [])
    
    if must or should:
        lines.append("### Requirements")
        if must:
            lines.append(f"**Must ({len(must)}):**")
            for r in must[:5]:
                lines.append(f"- {r[:100]}{'...' if len(r) > 100 else ''}")
            if len(must) > 5:
                lines.append(f"- _...and {len(must) - 5} more_")
        if should:
            lines.append(f"**Should ({len(should)}):**")
            for r in should[:3]:
                lines.append(f"- {r[:100]}{'...' if len(r) > 100 else ''}")
            if len(should) > 3:
                lines.append(f"- _...and {len(should) - 3} more_")
        lines.append("")
    
    # Constraints
    constraints = spec_payload.get("constraints", {})
    if constraints:
        lines.append("### Constraints")
        for k, v in list(constraints.items())[:5]:
            lines.append(f"- **{k}:** `{v}`")
        lines.append("")
    
    # Acceptance tests
    tests = spec_payload.get("acceptance_tests", [])
    if tests:
        lines.append(f"### Acceptance Tests ({len(tests)})")
        for t in tests[:3]:
            lines.append(f"- {t[:80]}{'...' if len(t) > 80 else ''}")
        if len(tests) > 3:
            lines.append(f"- _...and {len(tests) - 3} more_")
        lines.append("")
    
    # Clarification history
    if clarification_state and clarification_state.current_round > 0:
        lines.append("### Clarification History")
        lines.append(f"- **Rounds:** {clarification_state.current_round}")
        lines.append(f"- **Questions asked:** {clarification_state.total_questions_asked}")
        lines.append("")
    
    # File link
    lines.append("### Full Spec")
    lines.append(f"📁 `{spec_file_path}`")
    lines.append("")
    
    # Confirmation prompt
    lines.append("---")
    lines.append("**Ready to run the Critical Pipeline with this spec?**")
    lines.append("")
    lines.append("Reply **Yes** to proceed, or **No** to cancel.")
    
    return "\n".join(lines)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "ClarificationDecision",
    "HoleType",
    # Functions
    "compute_question_signature",
    "compute_question_signature_simple",
    "load_clarification_state",
    "save_clarification_state",
    "format_spec_summary_markdown",
    # Classes
    "ClarificationState",
    "ClarificationRound",
    # Constants
    "MAX_ROUNDS_PER_HOLE",
]
