# FILE: app/llm/pipeline/critique_parts/blocker_filtering.py
"""Block 5: Blocker Filtering (v1.2) - only approved blocker types can block."""

import logging
from typing import List, Tuple

from app.llm.pipeline.critique_schemas import (
    CritiqueIssue,
    APPROVED_ARCHITECTURE_BLOCKER_TYPES,
)

logger = logging.getLogger(__name__)


def filter_blocking_issues(
    issues: List[CritiqueIssue],
    require_evidence: bool = True,
) -> Tuple[List[CritiqueIssue], List[CritiqueIssue]]:
    """
    Filter blocking issues to only approved blocker types.
    
    v1.2: Implements the "Critique Contract" rules:
    1. Only approved blocker types can block (security, correctness, spec_mismatch, etc.)
    2. Blocking issues MUST have BOTH spec_ref AND arch_ref (evidence requirement)
    3. If category unknown BUT has perfect evidence AND description mentions drift/hallucination → allow
    
    Returns:
        (real_blocking, downgraded_to_non_blocking)
    """
    real_blocking: List[CritiqueIssue] = []
    downgraded: List[CritiqueIssue] = []
    
    drift_keywords = [
        "contradict", "contradiction", "drift", "hallucination", "invented",
        "does not exist", "doesn't exist", "mismatch", "violate", "violation",
        "missing required", "spec says", "spec requires",
    ]
    
    for issue in issues:
        category = (issue.category or "").lower().replace(" ", "_").replace("-", "_")
        has_spec_ref = bool(issue.spec_ref and issue.spec_ref.strip())
        has_arch_ref = bool(issue.arch_ref and issue.arch_ref.strip())
        has_full_evidence = has_spec_ref and has_arch_ref
        is_approved_category = category in APPROVED_ARCHITECTURE_BLOCKER_TYPES
        description_lower = (issue.description or "").lower()
        has_drift_keywords = any(kw in description_lower for kw in drift_keywords)
        
        should_block = False
        reason = ""
        
        if is_approved_category:
            if require_evidence and not has_full_evidence:
                reason = f"category={category} approved, but missing evidence (spec_ref={has_spec_ref}, arch_ref={has_arch_ref})"
            else:
                should_block = True
                reason = f"category={category} approved, evidence present"
        else:
            if has_full_evidence and has_drift_keywords:
                should_block = True
                reason = f"category={category} unknown but has drift keywords and full evidence"
            else:
                reason = f"category={category} not approved, no drift keywords or missing evidence"
        
        if should_block:
            real_blocking.append(issue)
            logger.debug("[critique] KEPT blocker %s: %s", issue.id, reason)
        else:
            issue.severity = "non_blocking"
            downgraded.append(issue)
            logger.info("[critique] DOWNGRADED issue %s to non_blocking: %s", issue.id, reason)
    
    if downgraded:
        print(f"[DEBUG] [critique] Filtered blockers: kept={len(real_blocking)}, downgraded={len(downgraded)}")
        logger.info("[critique] Blocker filtering: %d kept, %d downgraded", len(real_blocking), len(downgraded))
    
    return real_blocking, downgraded
