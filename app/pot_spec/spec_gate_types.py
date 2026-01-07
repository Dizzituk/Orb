# FILE: app/pot_spec/spec_gate_types.py
"""
Spec Gate v2 - Types and Constants

Contains:
- Result dataclass
- Blocking validation config
- Placeholder patterns and detection
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


# ---------------------------------------------------------------------------
# BLOCKING VALIDATION CONFIG (v2.1)
# ---------------------------------------------------------------------------

BLOCKING_FIELDS = {
    "steps": {"min_count": 1, "description": "execution steps"},
    "outputs": {"min_count": 1, "description": "output artifacts"},
    "acceptance_criteria": {"min_count": 1, "description": "verification criteria"},
}

PLACEHOLDER_PATTERNS = [
    r"^\s*\(?\s*not\s+specified\s*\)?\s*$",
    r"^\s*\(?\s*tbd\s*\)?\s*$",
    r"^\s*\(?\s*to\s+be\s+determined\s*\)?\s*$",
    r"^\s*\(?\s*n/?a\s*\)?\s*$",
    r"^\s*\(?\s*unknown\s*\)?\s*$",
    r"^\s*\(?\s*unspecified\s*\)?\s*$",
    r"^\s*\.\.\.\s*$",
    r"^\s*-\s*$",
    r"^\s*$",
]

PLACEHOLDER_RE = re.compile("|".join(PLACEHOLDER_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SpecGateResult:
    """Result from Spec Gate v2 processing."""
    ready_for_pipeline: bool = False
    open_questions: List[str] = field(default_factory=list)
    spot_markdown: Optional[str] = None
    db_persisted: bool = False
    spec_id: Optional[str] = None
    spec_hash: Optional[str] = None
    spec_version: Optional[int] = None
    hard_stopped: bool = False
    hard_stop_reason: Optional[str] = None
    notes: Optional[str] = None
    blocking_issues: List[str] = field(default_factory=list)
    validation_status: str = "pending"  # pending, needs_clarification, validated, blocked


# ---------------------------------------------------------------------------
# Placeholder detection
# ---------------------------------------------------------------------------

def is_placeholder(value: Any) -> bool:
    """Check if a value is a placeholder that shouldn't count as real content."""
    if value is None:
        return True
    if isinstance(value, str):
        return bool(PLACEHOLDER_RE.match(value.strip()))
    if isinstance(value, dict):
        name = value.get("name", "")
        if is_placeholder(name):
            return True
    return False


def count_real_items(items: List[Any]) -> int:
    """Count items that are not placeholders."""
    if not items or not isinstance(items, list):
        return 0
    return sum(1 for item in items if not is_placeholder(item))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_blocking_fields(
    outputs: List[dict],
    steps: List[str],
    acceptance: List[str],
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate that blocking fields have real content.
    
    Returns:
        (is_valid, blocking_issues, clarification_questions)
    """
    blocking_issues: List[str] = []
    questions: List[str] = []
    
    real_outputs = count_real_items(outputs)
    if real_outputs < BLOCKING_FIELDS["outputs"]["min_count"]:
        blocking_issues.append(f"BLOCKING: No valid outputs specified (found {real_outputs} real items)")
        questions.append("What exact output artifacts should exist when done? (file names + locations)")
    
    real_steps = count_real_items(steps)
    if real_steps < BLOCKING_FIELDS["steps"]["min_count"]:
        blocking_issues.append(f"BLOCKING: No valid steps specified (found {real_steps} real items)")
        questions.append("What exact steps should the system take, in order?")
    
    real_acceptance = count_real_items(acceptance)
    if real_acceptance < BLOCKING_FIELDS["acceptance_criteria"]["min_count"]:
        blocking_issues.append(f"BLOCKING: No valid acceptance criteria (found {real_acceptance} real items)")
        questions.append("How should we verify success? (exact file content, path, etc.)")
    
    return len(blocking_issues) == 0, blocking_issues, questions


def derive_required_questions(
    outputs: List[dict], 
    steps: List[str], 
    acceptance: List[str]
) -> List[str]:
    """Legacy function - derive questions for missing fields."""
    q: List[str] = []
    if not outputs:
        q.append("What exact output artifacts should exist when the job is done (file/folder names + locations)?")
    if not steps:
        q.append("What exact steps should the system take, in order? (keep it short)")
    if not acceptance:
        q.append("How should we verify success? (e.g., exact file content, exact path, overwrite behaviour)")
    return q


__all__ = [
    "BLOCKING_FIELDS",
    "PLACEHOLDER_RE",
    "SpecGateResult",
    "is_placeholder",
    "count_real_items",
    "validate_blocking_fields",
    "derive_required_questions",
]