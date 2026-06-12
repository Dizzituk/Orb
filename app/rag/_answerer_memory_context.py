# FILE: app/rag/_answerer_memory_context.py
# Purpose: Memory context builder for RAG queries.
# Called-by: app.rag.answerer
# Depends-on: app.memory.router
# Last-renovated: 2026-06-11
"""
Memory context builder for RAG queries.

Queries the MemoryRouter across all five domain stores, groups results
by domain, and formats them as labelled sections for LLM injection.

Token budget: 3,000 tokens (~12,000 chars) by default, leaving room
for the 8,000-token architecture chunk budget.

Fail-safe: if the MemoryRouter is unavailable or returns nothing,
returns an empty string. Memory failures never block RAG queries.

v10.0 (2026-02-23): Job 10C — memory-grounded evidence assembly.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Domain display order — most useful for codebase questions first
_DOMAIN_ORDER = [
    "architecture",
    "preferences",
    "knowledge",
    "context",
    "confidence",
]

# Domain labels for the formatted output
_DOMAIN_LABELS = {
    "architecture": "ARCHITECTURE KNOWLEDGE",
    "preferences": "USER PREFERENCES",
    "knowledge": "DOMAIN KNOWLEDGE",
    "context": "SESSION CONTEXT",
    "confidence": "CONFIDENCE DATA",
}


def build_memory_context(
    question: str,
    project_id: str = "astra-core",
    token_budget: int = 3000,
) -> str:
    """
    Query memory router and format results for LLM context injection.

    Args:
        question:     The user's codebase question.
        project_id:   Project scope for the memory query.
        token_budget: Maximum tokens (~4 chars each) for the output.

    Returns:
        Formatted string with domain-grouped memory results.
        Empty string if no results or router unavailable.
    """
    try:
        from app.memory.router import memory_router

        results = memory_router.query(
            text=question,
            project_id=project_id,
            limit=15,
            min_relevance=0.2,
        )

        if not results:
            logger.debug("[rag:memory_context] No memory results for: %s", question[:50])
            return ""

        return _format_memory_results(results, token_budget)

    except Exception as e:
        logger.warning("[rag:memory_context] Memory query failed: %s", e)
        return ""


def _format_memory_results(
    results: list,
    token_budget: int,
) -> str:
    """
    Group results by domain and format as labelled sections.

    Respects token budget — stops adding content once budget is reached.
    Rough token estimate: 4 chars ≈ 1 token (same heuristic used in
    _build_context_from_chunks in answerer.py).
    """
    char_budget = token_budget * 4

    # Group by domain
    grouped: dict[str, list] = {}
    for r in results:
        domain = getattr(r, "domain", "unknown")
        if domain not in grouped:
            grouped[domain] = []
        grouped[domain].append(r)

    # Build formatted sections in priority order
    sections = []
    chars_used = 0

    for domain in _DOMAIN_ORDER:
        if domain not in grouped:
            continue

        label = _DOMAIN_LABELS.get(domain, domain.upper())
        entries = grouped[domain]

        section_lines = [f"[{label}]"]
        for entry in entries:
            content = getattr(entry, "content", "")[:200]
            line = f"  - {content}"

            # Check budget before adding
            line_chars = len(line) + 1  # +1 for newline
            if chars_used + line_chars > char_budget:
                break

            section_lines.append(line)
            chars_used += line_chars

        section_lines.append(f"[/{label}]")
        sections.append("\n".join(section_lines))

        if chars_used >= char_budget:
            break

    if not sections:
        return ""

    formatted = "\n\n".join(sections)

    # Count domains that contributed
    domain_count = len([d for d in _DOMAIN_ORDER if d in grouped])
    result_count = len(results)
    tokens_est = chars_used // 4

    logger.info(
        "[rag:memory_context] %d results across %d domains (budget: %d/%d tokens)",
        result_count, domain_count, tokens_est, token_budget,
    )

    return formatted
