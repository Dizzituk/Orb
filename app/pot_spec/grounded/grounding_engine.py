# FILE: app/pot_spec/grounded/grounding_engine.py
"""
Grounding Engine for SpecGate

This module provides the core grounding logic that validates Weaver intent
against repository evidence.

Responsibilities:
- Ground mentioned paths/modules against evidence
- Verify what exists vs what doesn't in the codebase
- Identify constraints from repo patterns
- Track evidence completeness and gaps
- Detect refactor candidates from codebase report

Key Features:
- v1.25: Evidence-First architecture
- Micro-task detection for clutter reduction
- Arch map inference for related modules

Used by:
- spec_runner.py for intent grounding

Version: v2.0 (2026-02-01) - Extracted from spec_generation.py
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, TYPE_CHECKING

from .spec_models import (
    GroundedFact,
    GroundedPOTSpec,
)
from .text_helpers import _extract_paths_from_text, _extract_keywords

if TYPE_CHECKING:
    from ..evidence_collector import EvidenceBundle

logger = logging.getLogger(__name__)

# Evidence collector imports
try:
    from ..evidence_collector import (
        find_in_evidence,
        verify_path_exists,
    )
    _EVIDENCE_AVAILABLE = True
except ImportError as e:
    logger.warning("[grounding_engine] evidence_collector not available: %s", e)
    _EVIDENCE_AVAILABLE = False
    find_in_evidence = None
    verify_path_exists = None


__all__ = [
    "ground_intent_with_evidence",
]


def ground_intent_with_evidence(
    intent: Dict[str, Any],
    evidence: "EvidenceBundle",
    is_micro_task: bool = False,
) -> GroundedPOTSpec:
    """
    Ground Weaver intent against repo evidence.
    
    This is the core grounding logic:
    1. Look for mentioned paths/modules in evidence
    2. Verify what exists vs what doesn't
    3. Identify constraints from repo patterns
    4. Generate questions ONLY for true unknowns
    
    Args:
        intent: Parsed Weaver intent from parse_weaver_intent()
        evidence: EvidenceBundle with loaded evidence sources
        is_micro_task: If True, skip arch map inference (clutter reduction)
        
    Returns:
        GroundedPOTSpec with grounded facts and gaps
    """
    spec = GroundedPOTSpec(
        goal=intent.get("goal", ""),
        evidence_bundle=evidence,
    )
    
    # Track evidence completeness
    spec.evidence_complete = True
    spec.evidence_gaps = []
    
    # Check if codebase report was loaded
    has_codebase_report = False
    has_arch_map = False
    if evidence:
        for source in evidence.sources:
            if source.source_type == "codebase_report":
                if source.found:
                    has_codebase_report = True
                elif source.error:
                    spec.evidence_gaps.append(f"Codebase report: {source.error}")
                    spec.evidence_complete = False
            if source.source_type == "architecture_map":
                if source.found:
                    has_arch_map = True
                elif source.error:
                    spec.evidence_gaps.append(f"Architecture map: {source.error}")
                    spec.evidence_complete = False
    
    # Extract any paths mentioned in intent
    mentioned_paths = _extract_paths_from_text(intent.get("raw_text", ""))
    mentioned_paths.extend(_extract_paths_from_text(intent.get("goal", "")))
    
    # Add location if specified
    if intent.get("location"):
        mentioned_paths.append(intent["location"])
    
    # Ground each mentioned path
    for path in set(mentioned_paths):
        if evidence and _EVIDENCE_AVAILABLE and verify_path_exists:
            exists, source = verify_path_exists(evidence, path)
            if exists:
                spec.confirmed_components.append(GroundedFact(
                    description=f"Path `{path}` exists",
                    source=source or "evidence",
                    path=path,
                    confidence="confirmed",
                ))
                spec.what_exists.append(f"`{path}`")
            else:
                spec.what_missing.append(f"`{path}` (not found in evidence)")
    
    # Extract constraints from intent
    if intent.get("constraints"):
        spec.constraints_from_intent.extend(intent["constraints"])
    if intent.get("scope_constraints"):
        spec.constraints_from_intent.extend(intent["scope_constraints"])
    
    # Extract scope
    if intent.get("scope_in"):
        spec.in_scope.extend(intent["scope_in"])
    if intent.get("scope_out"):
        spec.out_of_scope.extend(intent["scope_out"])
    
    # Try to find relevant patterns in evidence
    # Skip arch map inference for micro tasks (clutter reduction)
    if evidence and evidence.arch_map_content and not is_micro_task and _EVIDENCE_AVAILABLE and find_in_evidence:
        # Look for related modules
        goal_keywords = _extract_keywords(intent.get("goal", ""))
        for keyword in goal_keywords[:5]:  # Top 5 keywords
            matches = find_in_evidence(evidence, rf"\b{re.escape(keyword)}\b", "architecture_map")
            if matches:
                spec.confirmed_components.append(GroundedFact(
                    description=f"Related content found for '{keyword}' in architecture map",
                    source="architecture_map",
                    confidence="inferred",
                ))
    
    # Copy steps/outputs from Weaver if available
    if intent.get("weaver_steps"):
        spec.proposed_steps = intent["weaver_steps"]
    if intent.get("weaver_acceptance"):
        spec.acceptance_tests = intent["weaver_acceptance"]
    
    # Detect refactor candidates from codebase report
    if evidence and evidence.codebase_report_content and _EVIDENCE_AVAILABLE and find_in_evidence:
        # Look for bloat warnings
        bloat_matches = find_in_evidence(
            evidence,
            r"(size_critical|size_high|lines_critical|lines_high)",
            "codebase_report"
        )
        if bloat_matches:
            spec.refactor_flags.append(
                "Codebase report indicates large/complex files - consider refactoring"
            )
    
    return spec
