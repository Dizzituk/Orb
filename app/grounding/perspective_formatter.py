# FILE: app/grounding/perspective_formatter.py
"""
Perspective Formatter for the Grounded Intelligence System.

Structures multi-perspective search results into a clear evidence
format that the LLM can use to build a balanced response.

Groups sources by the perspective they represent, tags each with
credibility info, and produces a structured [BALANCED EVIDENCE]
block for injection into the LLM context.

v1.0 (2026-03): Initial implementation.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional
from urllib.parse import urlparse

from app.grounding.perspective_engine import Perspective

logger = logging.getLogger(__name__)


# =========================================================================
# Source with perspective tag
# =========================================================================

@dataclass
class PerspectiveSource:
    """A search result tagged with the perspective it represents."""
    title: str
    url: str
    snippet: str = ""
    credibility_label: str = "unknown"
    source_type: str = "unknown"
    bias_note: str = ""
    perspective_label: str = ""      # Which perspective this source supports
    perspective_index: int = -1      # 0-indexed perspective assignment
    page_text: str = ""              # Fetched page text (may be empty)


# =========================================================================
# Perspective assignment heuristics
# =========================================================================

# Keywords that hint at perspective alignment
_LEFT_PROGRESSIVE_SIGNALS = {
    "progressive", "social justice", "inequality", "equity",
    "public investment", "regulation needed", "workers rights",
    "guardian", "huffpost", "vox", "msnbc",
}
_RIGHT_CONSERVATIVE_SIGNALS = {
    "conservative", "free market", "deregulation", "liberty",
    "small government", "traditional", "heritage foundation",
    "fox news", "daily mail", "telegraph", "national review",
}
_CENTRIST_SIGNALS = {
    "moderate", "centrist", "pragmatic", "bipartisan",
    "evidence based", "compromise", "balanced approach",
    "bbc", "reuters", "associated press", "economist",
}


def _score_perspective_alignment(
    text: str,
    perspective_index: int,
    total_perspectives: int,
) -> float:
    """
    Score how well a source's text aligns with a given perspective.
    Returns 0.0-1.0.

    This is a rough heuristic — it checks for signal words associated
    with different viewpoints. Not perfect, but better than random.
    """
    lower = text.lower()
    if total_perspectives < 2:
        return 0.5

    # For a 3-perspective setup (left/right/centre), check signals
    if perspective_index == 0:
        signals = _LEFT_PROGRESSIVE_SIGNALS
    elif perspective_index == 1:
        signals = _RIGHT_CONSERVATIVE_SIGNALS
    elif perspective_index == 2:
        signals = _CENTRIST_SIGNALS
    else:
        return 0.3  # Unknown perspective

    hits = sum(1 for s in signals if s in lower)
    return min(1.0, hits * 0.2)


def assign_perspectives(
    sources: list[dict],
    perspectives: list[Perspective],
) -> list[PerspectiveSource]:
    """
    Assign each source to the perspective it best represents.

    Strategy:
    - If sources came from perspective-targeted queries, use query order
    - Otherwise, use keyword heuristics to guess alignment
    - Sources that don't clearly align get labelled "neutral/unaligned"

    Args:
        sources: List of source dicts from web search.
        perspectives: List of Perspective objects from debate analysis.

    Returns:
        List of PerspectiveSource with perspective assignments.
    """
    if not perspectives:
        return [
            PerspectiveSource(
                title=s.get("title", ""),
                url=s.get("url", ""),
                snippet=s.get("snippet", ""),
                credibility_label=s.get("credibility_label", "unknown"),
                source_type=s.get("source_type", "unknown"),
                bias_note=s.get("bias_note", ""),
                perspective_label="unassigned",
                perspective_index=-1,
            )
            for s in sources
        ]

    tagged: list[PerspectiveSource] = []
    n_perspectives = len(perspectives)

    for i, src in enumerate(sources):
        text_to_check = f"{src.get('title', '')} {src.get('snippet', '')}".lower()

        # Score against each perspective
        best_idx = -1
        best_score = 0.0
        for pidx in range(n_perspectives):
            score = _score_perspective_alignment(text_to_check, pidx, n_perspectives)
            if score > best_score:
                best_score = score
                best_idx = pidx

        # If no clear match, use round-robin based on query order
        # (sources from perspective-targeted queries tend to come in groups)
        if best_score < 0.2:
            # Distribute based on position: neutral query results first,
            # then perspective-targeted results in order
            group_size = max(1, len(sources) // (n_perspectives + 1))
            if i < group_size:
                best_idx = n_perspectives - 1  # Last perspective (usually centrist/neutral)
            else:
                adjusted = i - group_size
                best_idx = min(adjusted // group_size, n_perspectives - 1)

        p_label = perspectives[best_idx].label if 0 <= best_idx < n_perspectives else "unassigned"

        tagged.append(PerspectiveSource(
            title=src.get("title", ""),
            url=src.get("url", ""),
            snippet=src.get("snippet", ""),
            credibility_label=src.get("credibility_label", "unknown"),
            source_type=src.get("source_type", "unknown"),
            bias_note=src.get("bias_note", ""),
            perspective_label=p_label,
            perspective_index=best_idx,
        ))

    return tagged


# =========================================================================
# Evidence block formatting
# =========================================================================

def format_balanced_evidence(
    tagged_sources: list[PerspectiveSource],
    perspectives: list[Perspective],
) -> str:
    """
    Format perspective-tagged sources into a [BALANCED EVIDENCE] block.

    Groups sources by perspective for clarity.
    """
    if not tagged_sources:
        return "[BALANCED EVIDENCE]\nNo sources found for any perspective.\n"

    lines = ["[BALANCED EVIDENCE]", ""]

    # Group by perspective
    by_perspective: dict[str, list[PerspectiveSource]] = {}
    for src in tagged_sources:
        key = src.perspective_label or "unassigned"
        by_perspective.setdefault(key, []).append(src)

    # Output in perspective order
    source_num = 1
    for p in perspectives:
        group = by_perspective.get(p.label, [])
        if not group:
            continue

        lines.append(f"--- {p.label.upper()} ---")
        lines.append(f"({p.description})")
        lines.append("")

        for src in group:
            cred = src.credibility_label.upper()
            line = f"[{source_num}] [{cred}] ({src.source_type}) {src.title}"
            if src.url:
                line += f"\n    URL: {src.url}"
            if src.snippet:
                line += f"\n    Snippet: {src.snippet}"
            if src.bias_note:
                line += f"\n    ⚠ Bias note: {src.bias_note}"
            if src.page_text:
                truncated = src.page_text[:1500]
                if len(src.page_text) > 1500:
                    truncated += "... [truncated]"
                line += f"\n    PageText: {truncated}"
            lines.append(line)
            lines.append("")
            source_num += 1

    # Unassigned sources
    unassigned = by_perspective.get("unassigned", [])
    if unassigned:
        lines.append("--- NEUTRAL / UNALIGNED SOURCES ---")
        lines.append("")
        for src in unassigned:
            cred = src.credibility_label.upper()
            line = f"[{source_num}] [{cred}] ({src.source_type}) {src.title}"
            if src.url:
                line += f"\n    URL: {src.url}"
            if src.snippet:
                line += f"\n    Snippet: {src.snippet}"
            lines.append(line)
            lines.append("")
            source_num += 1

    return "\n".join(lines)


__all__ = [
    "PerspectiveSource",
    "assign_perspectives",
    "format_balanced_evidence",
]
