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
from enum import Enum
from typing import Any, List, Optional, Tuple


# ---------------------------------------------------------------------------
# OUTPUT MODE ENUM (v1.13 - MICRO_FILE_TASK output mode)
# ---------------------------------------------------------------------------

class OutputMode(str, Enum):
    """
    v1.13: Output mode for MICRO_FILE_TASK jobs (sandbox file discovery jobs).
    
    Determines where the reply should be written:
    - APPEND_IN_PLACE: Write reply into the same file (under the question)
    - SEPARATE_REPLY_FILE: Write to a separate reply.txt file
    - CHAT_ONLY: No file output, reply in chat only
    
    Detection is based on user intent keywords:
    - APPEND_IN_PLACE: "write under", "append", "add below", "put reply underneath", 
                       "beneath the question", "write it in the file", "write reply in"
    - SEPARATE_REPLY_FILE: "save to reply.txt", "write to a new file", "create a reply file",
                           "separate file", "save as"
    - CHAT_ONLY: "just answer here", "don't change the file", "reply here in chat only",
                 "chat only", "don't write"
    
    Default: If user intent implies writing into the file → APPEND_IN_PLACE
    Safe default (no write indicators): CHAT_ONLY
    """
    APPEND_IN_PLACE = "append_in_place"
    SEPARATE_REPLY_FILE = "separate_reply_file"
    CHAT_ONLY = "chat_only"


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
    """
    Result from Spec Gate v2 processing.
    
    v2.2 (2026-01-21): Added grounding_data field for Critical Pipeline job classification.
    This enables micro vs architecture routing by persisting sandbox discovery results.
    """
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
    
    # v2.2: Grounding data for Critical Pipeline job classification
    # Contains sandbox resolution fields needed for micro vs architecture routing
    grounding_data: Optional[dict] = None


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
    "OutputMode",  # v1.13: MICRO_FILE_TASK output mode
    "is_placeholder",
    "count_real_items",
    "validate_blocking_fields",
    "derive_required_questions",
]