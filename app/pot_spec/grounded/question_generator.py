# FILE: app/pot_spec/grounded/question_generator.py
"""
Question Generator for SpecGate

This module handles generation of grounded questions for specs,
focusing on genuine unknowns and decision forks.

Responsibilities:
- Generate questions ONLY for genuine unknowns
- Extract decision forks from detected domains
- Apply v1.2 contract rules (bounded A/B/C questions only)
- Populate assumptions as safe defaults

Key Features:
- v1.2: Decision forks replace lazy questions
- v1.5: Fork extraction populates assumptions every round
- Max 7 questions total, only high-impact product decisions

Used by:
- spec_runner.py for question generation

Version: v2.0 (2026-02-01) - Extracted from spec_generation.py
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, TYPE_CHECKING

from .spec_models import (
    QuestionCategory,
    GroundedQuestion,
    GroundedPOTSpec,
    MAX_QUESTIONS,
)
from .domain_detection import (
    detect_domains,
    extract_decision_forks,
)
from .step_derivation import (
    _derive_steps_from_domain,
    _derive_tests_from_domain,
)

if TYPE_CHECKING:
    from ..evidence_collector import EvidenceBundle

logger = logging.getLogger(__name__)


__all__ = [
    "generate_grounded_questions",
]


def generate_grounded_questions(
    spec: GroundedPOTSpec,
    intent: Dict[str, Any],
    evidence: "EvidenceBundle",
    round_number: int,
) -> List[GroundedQuestion]:
    """
    Generate questions ONLY for genuine unknowns.
    
    v1.2 CONTRACT:
    - Round 1: Ask bounded decision forks (A/B/C) only
    - Round 2+: Steps/tests are DERIVED from fork answers (not asked for)
    - Never ask "tell me the steps" or "tell me the acceptance criteria"
    - Max 7 questions total, only high-impact product decisions
    
    Rules:
    - Only ask when NOT derivable from evidence
    - Only ask high-impact questions (wrong answer = rework)
    - Preference/product decisions only (not engineering facts)
    
    Args:
        spec: GroundedPOTSpec to populate with questions and assumptions
        intent: Parsed Weaver intent
        evidence: EvidenceBundle with loaded evidence sources
        round_number: Current question round (1 = initial, 2+ = after answers)
        
    Returns:
        List of GroundedQuestion objects (max MAX_QUESTIONS)
    """
    questions = []
    
    # Get Weaver text for domain detection and fork extraction
    weaver_text = intent.get("raw_text", "") or ""
    
    # =================================================================
    # v1.5: ALWAYS extract forks to populate assumptions (every round)
    # =================================================================
    detected_domains = detect_domains(weaver_text)
    fork_questions = []
    fork_assumptions = []
    
    if detected_domains:
        fork_questions, fork_assumptions = extract_decision_forks(
            weaver_text=weaver_text,
            detected_domains=detected_domains,
            max_questions=MAX_QUESTIONS,
        )
        # Always populate assumptions (even on Round 2+)
        spec.assumptions.extend(fork_assumptions)
        
        logger.info(
            "[question_generator] v1.5: Detected domains %s, blocking questions=%d, assumptions=%d (round=%d)",
            detected_domains, len(fork_questions), len(fork_assumptions), round_number
        )
    
    # =================================================================
    # ROUND 2+: Derive steps/tests from fork answers, don't ask more
    # =================================================================
    if round_number >= 2:
        # Only ask critical questions if there's a genuine blocker
        if not spec.goal or spec.goal.strip() == "":
            questions.append(GroundedQuestion(
                question="What is the primary goal/objective of this job?",
                category=QuestionCategory.MISSING_PRODUCT_DECISION,
                why_it_matters="Without a clear goal, the spec cannot be grounded",
                evidence_found="No goal found in Weaver output",
            ))
        
        # Derive steps/tests if missing
        if not spec.proposed_steps:
            spec.proposed_steps = _derive_steps_from_domain(intent, spec)
        if not spec.acceptance_tests or all('(To be determined)' in str(t) for t in spec.acceptance_tests):
            spec.acceptance_tests = _derive_tests_from_domain(intent, spec)
        
        return questions[:MAX_QUESTIONS]
    
    # =================================================================
    # ROUND 1: Ask bounded decision forks (A/B/C) - NOT lazy questions
    # =================================================================
    
    # Check for missing goal (this is a critical blocker)
    if not spec.goal or spec.goal.strip() == "":
        questions.append(GroundedQuestion(
            question="What is the primary goal/objective of this job?",
            category=QuestionCategory.MISSING_PRODUCT_DECISION,
            why_it_matters="Without a clear goal, the spec cannot be grounded",
            evidence_found="No goal found in Weaver output",
        ))
    
    # v1.5: Fork questions were already extracted above
    if fork_questions:
        questions.extend(fork_questions)
    
    # Check for ambiguous paths
    if spec.what_missing:
        missing_count = len(spec.what_missing)
        if missing_count > 0 and len(questions) < MAX_QUESTIONS:
            questions.append(GroundedQuestion(
                question=f"These paths were mentioned but not found in evidence: {', '.join(spec.what_missing[:3])}. Should they be created, or are the paths incorrect?",
                category=QuestionCategory.AMBIGUOUS_EVIDENCE,
                why_it_matters="Need to know if files should be created vs paths are wrong",
                evidence_found=f"Searched architecture map and codebase report - {missing_count} path(s) not found",
                options=["Create new files at these paths", "Paths may be incorrect - suggest alternatives"],
            ))
    
    # Check for safety constraints if touching critical paths
    critical_paths = ['stream_router', 'overwatcher', 'translation', 'routing']
    touches_critical = any(
        any(crit in fact.description.lower() for crit in critical_paths)
        for fact in spec.confirmed_components
    )
    if touches_critical and not any('sandbox' in c.lower() for c in spec.constraints_from_intent):
        if len(questions) < MAX_QUESTIONS:
            questions.append(GroundedQuestion(
                question="This job touches critical routing/pipeline code. Should changes be tested in SANDBOX first before MAIN repo?",
                category=QuestionCategory.SAFETY_CONSTRAINT,
                why_it_matters="Touching critical code without sandbox testing risks breaking the system",
                evidence_found="Detected critical paths in scope",
                options=["Sandbox first, then MAIN", "MAIN repo directly (I'll verify manually)"],
            ))
    
    return questions[:MAX_QUESTIONS]
