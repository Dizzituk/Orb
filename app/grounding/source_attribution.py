# FILE: app/grounding/source_attribution.py
# Purpose: Source Attribution for the Grounded Intelligence System.
# Called-by: app.grounding.grounding_gate
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Source Attribution for the Grounded Intelligence System.

Formats source citations naturally for conversational responses.
No academic footnotes — just clear, natural references like:
  "According to a Reuters report from today..."
  "Based on Trading Economics data updated this morning..."

Also provides the system prompt injection that instructs the LLM
to ground its response in the provided sources.

v1.0 (2026-03): Initial implementation.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# =========================================================================
# Grounding system prompt blocks
# =========================================================================

GROUNDING_SYSTEM_PROMPT = """
## GROUNDED RESPONSE MODE — ACTIVE

A web search has been performed to ground your response in current evidence.
The search results are provided below under [GROUNDING EVIDENCE].

**MANDATORY RULES:**
1. Base your response ONLY on the provided sources. Do NOT fill gaps with
   your training data for factual claims.
2. Where sources conflict, state BOTH positions and note which source
   appears more authoritative.
3. If the sources are insufficient to answer the question fully, say so
   clearly: "I found limited evidence on this — here's what I have..."
4. Include natural source attribution in your response. Examples:
   - "According to a Financial Times piece from today..."
   - "Reuters is reporting that..."
   - "Data from Trading Economics shows..."
5. If a source has low credibility (COMMUNITY or UNKNOWN tier), flag it:
   "One source suggests X, though it's from a less established outlet."
6. NEVER present ungrounded claims with confidence. If you're not sure,
   say you're not sure.
7. End your response with a brief confidence note if the evidence is thin:
   "Note: I only found [N] sources on this, so treat this as preliminary."

**YOUR CORE PRINCIPLE:** It is always better to say "I don't have solid
data on this" than to confidently state something that might be wrong.
""".strip()


CONTESTED_SYSTEM_PROMPT = """
## BALANCED PERSPECTIVE MODE — ACTIVE

This topic has been identified as contested or opinion-heavy.
Multiple perspective searches have been performed. Results are below.

**MANDATORY RULES:**
1. Present MULTIPLE viewpoints fairly — do not pick a side by default.
2. Structure your response as:
   a) Perspective A — strongest arguments and evidence
   b) Perspective B — strongest arguments and evidence
   c) Where they agree / where they genuinely diverge
   d) What the empirical evidence supports (if applicable)
3. Use natural attribution: "Those on the left argue..." / "Conservative
   analysts point to..." / "Economists on both sides agree that..."
4. Do NOT tell the user what to think. Your job is to INFORM, not persuade.
5. If one side has significantly stronger empirical evidence, you may note
   this factually, but still present the opposing view's strongest case.
6. End with: "This is a topic where reasonable people disagree — these are
   the strongest arguments I found on each side."
""".strip()


CONFIDENCE_DISCLAIMER_THIN = (
    "\n\n*Note: I found limited sources on this. "
    "Treat this as a starting point rather than a definitive answer.*"
)

CONFIDENCE_DISCLAIMER_STALE = (
    "\n\n*Note: The most recent source I found on this is {age} old. "
    "More current data may be available.*"
)


# =========================================================================
# Evidence formatting
# =========================================================================

def format_evidence_block(
    sources: list[dict],
    fetched_texts: Optional[list[str]] = None,
) -> str:
    """
    Format search results into a [GROUNDING EVIDENCE] block for injection
    into the LLM context.
    
    Each source dict should have: title, url, snippet, and optionally
    credibility_label, source_type, bias_note.
    """
    if not sources:
        return "[GROUNDING EVIDENCE]\nNo sources found.\n"
    
    lines = ["[GROUNDING EVIDENCE]", ""]
    for i, src in enumerate(sources, start=1):
        title = src.get("title", "Untitled")
        url = src.get("url", "")
        snippet = src.get("snippet", "")
        cred = src.get("credibility_label", "unknown").upper()
        stype = src.get("source_type", "unknown")
        bias = src.get("bias_note", "")
        
        line = f"[{i}] [{cred}] ({stype}) {title}"
        if url:
            line += f"\n    URL: {url}"
        if snippet:
            line += f"\n    Snippet: {snippet}"
        if bias:
            line += f"\n    ⚠ Bias note: {bias}"
        
        # Include fetched page text if available
        if fetched_texts and i - 1 < len(fetched_texts):
            page_text = fetched_texts[i - 1]
            if page_text:
                # Cap at 2000 chars per source
                truncated = page_text[:2000]
                if len(page_text) > 2000:
                    truncated += "... [truncated]"
                line += f"\n    PageText: {truncated}"
        
        lines.append(line)
        lines.append("")  # Blank line between sources
    
    return "\n".join(lines)


def build_confidence_note(source_count: int, oldest_source_age_hours: float = 0) -> str:
    """
    Build a confidence disclaimer if evidence is thin or stale.
    Returns empty string if evidence is adequate.
    """
    if source_count == 0:
        return "\n\n*I couldn't find any current sources on this topic.*"
    
    if source_count <= 1:
        return CONFIDENCE_DISCLAIMER_THIN
    
    if oldest_source_age_hours > 168:  # > 7 days
        if oldest_source_age_hours > 720:  # > 30 days
            age_str = f"{int(oldest_source_age_hours / 720)} months"
        elif oldest_source_age_hours > 168:
            age_str = f"{int(oldest_source_age_hours / 168)} weeks"
        else:
            age_str = f"{int(oldest_source_age_hours / 24)} days"
        return CONFIDENCE_DISCLAIMER_STALE.format(age=age_str)
    
    return ""


__all__ = [
    "GROUNDING_SYSTEM_PROMPT",
    "CONTESTED_SYSTEM_PROMPT",
    "CONFIDENCE_DISCLAIMER_THIN",
    "CONFIDENCE_DISCLAIMER_STALE",
    "format_evidence_block",
    "build_confidence_note",
]
