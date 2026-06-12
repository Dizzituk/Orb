# FILE: app/transparency/matcher.py
# Purpose: CorrectionMatcher — queries relevant past corrections during pipeline runs.
# Called-by: app.debug.feedback
# Depends-on: app.transparency.corrections, app.transparency.schemas
# Last-renovated: 2026-06-11
"""
CorrectionMatcher — queries relevant past corrections during pipeline runs.

Called by pipeline stages before making decisions to check if there
are relevant user corrections that should influence the current decision.

Matching strategy:
1. Stage name filter (exact match)
2. Keyword overlap scoring
3. Severity weighting (broke_things > wrong_output > note)
4. Recency bias (newer corrections weighted higher)

v1.0 (2026-02): Initial implementation — keyword matching only
Future: Add embedding similarity for semantic matching
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from app.transparency.corrections import CorrectionStore, extract_keywords
from app.transparency.schemas import CorrectionMatch, UserCorrection

logger = logging.getLogger(__name__)


# =============================================================================
# SEVERITY WEIGHTS
# =============================================================================

_SEVERITY_WEIGHTS = {
    "broke_things": 3.0,
    "wrong_output": 2.0,
    "note": 1.0,
}

# Maximum age in days for corrections to be considered relevant
_MAX_AGE_DAYS = 90

# Minimum relevance score to include in results
_MIN_RELEVANCE = 0.15


# =============================================================================
# CORRECTION MATCHER
# =============================================================================

class CorrectionMatcher:
    """
    Matches past corrections to current pipeline context.

    Usage:
        matcher = CorrectionMatcher()
        matches = matcher.find_relevant(
            stage_name="specgate",
            context_keywords=["auth", "middleware", "router"],
            max_results=5,
        )

        for match in matches:
            # Inject into LLM prompt
            prompt += match.to_prompt_injection()
    """

    def find_relevant(
        self,
        stage_name: str,
        context_keywords: Optional[List[str]] = None,
        context_text: str = "",
        max_results: int = 5,
    ) -> List[CorrectionMatch]:
        """
        Find corrections relevant to the current pipeline context.

        Args:
            stage_name: Current pipeline stage (e.g. "specgate")
            context_keywords: Keywords from the current task context
            context_text: Full text context (keywords extracted if not provided)
            max_results: Maximum corrections to return

        Returns:
            List of CorrectionMatch objects sorted by relevance (highest first)
        """
        # Extract keywords from text if not provided
        if not context_keywords and context_text:
            context_keywords = extract_keywords(context_text)

        if not context_keywords:
            return []

        # Get all corrections for this stage
        corrections = CorrectionStore.get_corrections_by_stage(
            stage_name=stage_name,
            limit=100,
        )

        if not corrections:
            return []

        # Score each correction
        scored: List[CorrectionMatch] = []
        now = datetime.now(timezone.utc)

        for correction in corrections:
            score = self._score_correction(
                correction, context_keywords, now
            )
            if score >= _MIN_RELEVANCE:
                scored.append(CorrectionMatch(
                    correction=correction,
                    relevance_score=score,
                    original_context=correction.user_comment[:200],
                ))

        # Sort by relevance (highest first) and limit
        scored.sort(key=lambda m: m.relevance_score, reverse=True)
        return scored[:max_results]

    def _score_correction(
        self,
        correction: UserCorrection,
        context_keywords: List[str],
        now: datetime,
    ) -> float:
        """
        Score a correction's relevance to the current context.

        Scoring factors:
        1. Keyword overlap (primary signal)
        2. Severity weight (broke_things is more important)
        3. Recency (newer corrections more relevant)
        """
        if not correction.context_keywords:
            return 0.0

        # 1. Keyword overlap — Jaccard-like score
        correction_kw_set = set(correction.context_keywords)
        context_kw_set = set(context_keywords)

        overlap = len(correction_kw_set & context_kw_set)
        union = len(correction_kw_set | context_kw_set)

        if union == 0:
            return 0.0

        keyword_score = overlap / union

        # 2. Severity weight
        severity_weight = _SEVERITY_WEIGHTS.get(correction.severity, 1.0)
        severity_factor = severity_weight / 3.0  # Normalise to 0-1 range

        # 3. Recency — linear decay over _MAX_AGE_DAYS
        recency_factor = 1.0
        if correction.created_at:
            try:
                created = datetime.fromisoformat(correction.created_at)
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_days = (now - created).days
                if age_days > _MAX_AGE_DAYS:
                    recency_factor = 0.1  # Very old but not zero
                else:
                    recency_factor = 1.0 - (age_days / _MAX_AGE_DAYS) * 0.5
            except (ValueError, TypeError):
                recency_factor = 0.5

        # Combine: keyword overlap is primary (60%), severity (25%), recency (15%)
        score = (
            keyword_score * 0.60
            + severity_factor * 0.25
            + recency_factor * 0.15
        )

        return round(score, 4)

    def format_for_prompt(
        self,
        matches: List[CorrectionMatch],
        max_chars: int = 1000,
    ) -> str:
        """
        Format matched corrections for injection into an LLM prompt.

        Returns a string block that can be appended to the system prompt
        or injected into the task context.
        """
        if not matches:
            return ""

        lines = ["RELEVANT PAST CORRECTIONS (from user feedback):"]
        total_chars = len(lines[0])

        for match in matches:
            line = match.to_prompt_injection()
            if total_chars + len(line) > max_chars:
                break
            lines.append(f"- {line}")
            total_chars += len(line)

        lines.append("Apply these corrections where relevant to the current task.")
        return "\n".join(lines)


# Module-level singleton for convenience
_matcher = CorrectionMatcher()


def find_relevant_corrections(
    stage_name: str,
    context_keywords: Optional[List[str]] = None,
    context_text: str = "",
    max_results: int = 5,
) -> List[CorrectionMatch]:
    """Module-level convenience function."""
    return _matcher.find_relevant(
        stage_name=stage_name,
        context_keywords=context_keywords,
        context_text=context_text,
        max_results=max_results,
    )


def format_corrections_for_prompt(
    stage_name: str,
    context_keywords: Optional[List[str]] = None,
    context_text: str = "",
    max_results: int = 5,
    max_chars: int = 1000,
) -> str:
    """Find and format corrections ready for prompt injection."""
    matches = find_relevant_corrections(
        stage_name=stage_name,
        context_keywords=context_keywords,
        context_text=context_text,
        max_results=max_results,
    )
    return _matcher.format_for_prompt(matches, max_chars=max_chars)


__all__ = [
    "CorrectionMatcher",
    "find_relevant_corrections",
    "format_corrections_for_prompt",
]
