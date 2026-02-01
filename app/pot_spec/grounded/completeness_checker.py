# FILE: app/pot_spec/grounded/completeness_checker.py
"""
Spec Completeness Checker for SpecGate

This module provides logic to determine if a spec is complete enough
to proceed without more questions.

Responsibilities:
- Check if spec meets minimum requirements for valid POT spec
- Enable early exit when no blocking questions remain
- Prevent "question-hunting" by allowing graceful completion

Key Features:
- v1.4: Early exit logic for complete specs

Used by:
- spec_runner.py for completion status determination

Version: v2.0 (2026-02-01) - Extracted from spec_generation.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .spec_models import GroundedPOTSpec, GroundedQuestion


__all__ = [
    "_is_spec_complete_enough",
]


def _is_spec_complete_enough(
    spec: GroundedPOTSpec,
    intent: Dict[str, Any],
    blocking_questions: List[GroundedQuestion],
) -> Tuple[bool, str]:
    """
    v1.4: Check if spec is complete enough to proceed without more questions.
    
    This prevents "question-hunting" by allowing early exit when:
    - No blocking questions remain
    - Enough information exists to build a valid POT spec
    
    Args:
        spec: GroundedPOTSpec to check
        intent: Parsed Weaver intent
        blocking_questions: List of blocking questions remaining
        
    Returns:
        Tuple of (is_complete: bool, reason_string: str)
    """
    # If there are blocking questions, spec is not complete
    if blocking_questions:
        return False, f"{len(blocking_questions)} blocking question(s) remain"
    
    # Check minimum requirements for a valid POT spec
    checks = []
    
    # 1. Goal must be defined
    if not spec.goal or spec.goal.strip() == "":
        checks.append("goal is missing")
    
    if checks:
        return False, f"Missing: {', '.join(checks)}"
    
    return True, "Spec is complete enough - no blocking questions remain"
