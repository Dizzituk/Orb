# FILE: app/llm/weaver_rules_inject.py
"""
Weaver Rules Injection — formats pre-classified items for the Weaver prompt.

Takes the output of weaver_rules_engine.classify_conversation() and
produces a structured block that's injected into the Weaver's user
prompt. The LLM then verifies/adjusts rather than classifying from scratch.

v1.0 (2026-03-01): Initial implementation.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from app.llm.weaver_rules_engine import (
    CAT_AMBIGUOUS,
    CAT_CONSTRAINT,
    CAT_DESIGN_PREF,
    CAT_ESTABLISHED_FACT,
    CAT_QUESTION_USER,
    CAT_REQ_FUNCTIONAL,
    CAT_REQ_TECHNICAL,
    CAT_SPECGATE,
    ClassifiedItem,
)

logger = logging.getLogger(__name__)

# Max items per category to inject (avoid prompt bloat)
_MAX_ITEMS_PER_CATEGORY = 12


def format_preclassified_block(
    classified: Dict[str, List[ClassifiedItem]],
) -> str:
    """Format pre-classified items as a prompt injection block.

    Produces a structured section that tells the LLM which items
    have already been sorted and which remain ambiguous.

    Args:
        classified: Output from classify_conversation().

    Returns:
        Formatted string for prompt injection. Empty string if
        nothing was confidently classified.
    """
    total_classified = sum(
        len(items) for cat, items in classified.items()
        if cat != CAT_AMBIGUOUS
    )

    if total_classified == 0:
        return ""

    total_ambiguous = len(classified.get(CAT_AMBIGUOUS, []))

    parts = [
        "─── PRE-CLASSIFIED ITEMS (deterministic rules engine) ───",
        "",
        f"The rules engine has pre-sorted {total_classified} items below.",
        f"{total_ambiguous} items remain ambiguous and need your classification.",
        "Review the pre-sorted items for accuracy. Move any misclassified items.",
        "Then classify the ambiguous items into the appropriate sections.",
        "",
    ]

    # Functional requirements
    _add_category(parts, classified, CAT_REQ_FUNCTIONAL,
                  "Key requirements (functional)")

    # Technical requirements
    _add_category(parts, classified, CAT_REQ_TECHNICAL,
                  "Key requirements (technical)")

    # Design preferences
    _add_category(parts, classified, CAT_DESIGN_PREF,
                  "Design preferences")

    # Constraints
    _add_category(parts, classified, CAT_CONSTRAINT,
                  "Constraints")

    # Established facts (from assistant)
    _add_category(parts, classified, CAT_ESTABLISHED_FACT,
                  "Established facts (from codebase analysis)")

    # SpecGate directives
    _add_category(parts, classified, CAT_SPECGATE,
                  "SpecGate must resolve")

    # Questions for user (rare)
    _add_category(parts, classified, CAT_QUESTION_USER,
                  "Possible questions for user (verify — most should be SpecGate)")

    # Ambiguous items
    if total_ambiguous > 0:
        parts.append("### UNCLASSIFIED (your job to sort):")
        amb = classified[CAT_AMBIGUOUS][:_MAX_ITEMS_PER_CATEGORY]
        for item in amb:
            role_tag = "[H]" if item.source_role == "human" else "[A]"
            parts.append(f"  {role_tag} {item.text[:120]}")
        if len(classified[CAT_AMBIGUOUS]) > _MAX_ITEMS_PER_CATEGORY:
            parts.append(
                f"  ... and {len(classified[CAT_AMBIGUOUS]) - _MAX_ITEMS_PER_CATEGORY} more"
            )
        parts.append("")

    parts.append("─── END PRE-CLASSIFIED ───")

    result = "\n".join(parts)
    logger.info(
        "[weaver_rules_inject] Injecting %d pre-classified items (%d chars)",
        total_classified, len(result),
    )
    return result


def _add_category(
    parts: List[str],
    classified: Dict[str, List[ClassifiedItem]],
    category: str,
    label: str,
) -> None:
    """Add a category section if it has items."""
    items = classified.get(category, [])
    if not items:
        return

    parts.append(f"### {label}:")
    for item in items[:_MAX_ITEMS_PER_CATEGORY]:
        conf = f" ({item.confidence:.0%})" if item.confidence < 0.80 else ""
        parts.append(f"  - {item.text[:150]}{conf}")
    if len(items) > _MAX_ITEMS_PER_CATEGORY:
        parts.append(f"  ... and {len(items) - _MAX_ITEMS_PER_CATEGORY} more")
    parts.append("")
