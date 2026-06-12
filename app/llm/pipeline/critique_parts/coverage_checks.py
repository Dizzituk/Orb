# FILE: app/llm/pipeline/critique_parts/coverage_checks.py
# Purpose: Deterministic Critique — Acceptance Criteria Coverage.
# Called-by: app.llm.pipeline.critique_parts.deterministic_verdict
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Deterministic Critique — Acceptance Criteria Coverage.

Check 2: Acceptance criteria coverage
    Each acceptance criterion from the spec maps to at least one
    section in the architecture. Uses keyword/phrase matching
    against section headers and body content.

Zero LLM calls. Pure keyword matching.

v1.0 (2026-02-27): Initial implementation — Stage 1 of deterministic
verification migration.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

COVERAGE_CHECKS_BUILD_ID = "2026-02-27-v1.0-acceptance-criteria-coverage"


# =========================================================================
# KEYWORD EXTRACTION
# =========================================================================

# Words too common to be meaningful for matching
_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "it", "its",
    "that", "this", "these", "those", "and", "or", "but", "if", "then",
    "than", "when", "where", "which", "who", "what", "how", "not", "no",
    "all", "each", "every", "any", "both", "some", "such", "only", "own",
    "same", "so", "very", "too", "also", "just", "as", "at", "by", "for",
    "from", "in", "into", "of", "on", "to", "up", "with", "about", "out",
    "over", "after", "before", "between", "under", "until", "through",
    "file", "files", "code", "function", "functions", "class", "module",
    "implement", "implementation", "create", "add", "new", "existing",
    "ensure", "verify", "check", "test", "return", "returns", "use",
    "used", "using", "include", "includes", "support", "supports",
}


def _extract_keywords(text: str) -> Set[str]:
    """
    Extract meaningful keywords from a text string.

    Strips stop words, normalises to lowercase, returns unique keywords.
    """
    # Extract word tokens
    words = re.findall(r'[a-zA-Z_]\w+', text.lower())
    # Filter stop words and very short words
    return {w for w in words if w not in _STOP_WORDS and len(w) > 2}


def _extract_technical_phrases(text: str) -> Set[str]:
    """
    Extract multi-word technical phrases that should match as units.

    Looks for patterns like:
    - snake_case identifiers
    - CamelCase identifiers
    - Quoted strings
    - dotted paths (e.g., app.orchestrator.cohesion)
    """
    phrases: Set[str] = set()

    # snake_case identifiers
    for m in re.finditer(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b', text):
        phrases.add(m.group(1).lower())

    # CamelCase identifiers
    for m in re.finditer(r'\b([A-Z][a-zA-Z0-9]+(?:[A-Z][a-z]+)+)\b', text):
        phrases.add(m.group(1).lower())

    # Backtick-quoted identifiers
    for m in re.finditer(r'`(\w[\w.]+)`', text):
        phrases.add(m.group(1).lower())

    return phrases


# =========================================================================
# CHECK 2: Acceptance Criteria Coverage
# =========================================================================

def check_acceptance_criteria_coverage(
    arch_content: str,
    spec_json: Optional[str] = None,
    spec_markdown: Optional[str] = None,
    segment_spec: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Check that each acceptance criterion is addressed in the architecture.

    Each criterion must have at least MATCH_THRESHOLD of its keywords
    present in the architecture, OR at least one of its technical
    phrases found.

    Args:
        arch_content: Architecture markdown document
        spec_json: Optional JSON string of the spec
        spec_markdown: Optional spec markdown
        segment_spec: Optional segment spec dict

    Returns:
        List of issue dicts
    """
    issues: List[Dict[str, Any]] = []

    # Collect acceptance criteria
    criteria: List[str] = []

    if segment_spec and segment_spec.get("acceptance_criteria"):
        criteria = segment_spec["acceptance_criteria"]
    elif spec_json:
        try:
            spec = json.loads(spec_json)
            if isinstance(spec, dict):
                criteria = spec.get("acceptance_criteria", [])
        except (json.JSONDecodeError, TypeError):
            pass

    if not criteria:
        # Also try extracting from spec markdown
        if spec_markdown:
            # Look for acceptance criteria section
            ac_match = re.search(
                r'(?:^|\n)#+\s*(?:Acceptance\s+Criteria|AC)\s*\n(.*?)(?:\n#+|\n---|\Z)',
                spec_markdown, re.DOTALL | re.IGNORECASE
            )
            if ac_match:
                section = ac_match.group(1)
                # Extract list items
                for m in re.finditer(r'[-*]\s+(.+)', section):
                    criteria.append(m.group(1).strip())

    if not criteria:
        return issues

    # Prepare architecture for matching
    arch_lower = arch_content.lower()
    arch_keywords = _extract_keywords(arch_content)
    arch_phrases = _extract_technical_phrases(arch_content)

    MATCH_THRESHOLD = 0.5  # At least 50% of keywords must appear

    for idx, criterion in enumerate(criteria):
        criterion_text = criterion.strip()
        if not criterion_text:
            continue

        # Extract keywords and phrases from this criterion
        crit_keywords = _extract_keywords(criterion_text)
        crit_phrases = _extract_technical_phrases(criterion_text)

        # Check for phrase matches first (stronger signal)
        phrase_matched = any(p in arch_phrases or p in arch_lower for p in crit_phrases)
        if phrase_matched:
            continue  # Criterion is covered

        # Check keyword overlap
        if crit_keywords:
            matched = crit_keywords & arch_keywords
            coverage = len(matched) / len(crit_keywords)

            if coverage < MATCH_THRESHOLD:
                unmatched = sorted(crit_keywords - matched)
                issues.append({
                    "rule_id": "DET-COVERAGE-GAP",
                    "severity": "warning",
                    "file": "",
                    "spec_ref": f"acceptance_criteria[{idx}]",
                    "arch_ref": "Architecture sections",
                    "description": (
                        f"Acceptance criterion '{criterion_text[:100]}' "
                        f"has low coverage in architecture ({coverage:.0%}). "
                        f"Missing keywords: {', '.join(unmatched[:5])}"
                    ),
                    "suggested_fix": (
                        f"Ensure the architecture addresses this criterion. "
                        f"Add a section or implementation detail covering: "
                        f"{criterion_text[:80]}"
                    ),
                })

    if issues:
        logger.info(
            "[det_critique] Coverage check: %d criteria with low coverage",
            len(issues),
        )

    return issues


__all__ = [
    "check_acceptance_criteria_coverage",
    "COVERAGE_CHECKS_BUILD_ID",
]
