# FILE: app/grounding/perspective_prompts.py
"""
Enhanced system prompts for balanced perspective responses.

Generates context-aware system prompt injections based on the detected
debate type. Each debate type gets a tailored prompt that instructs
the LLM on how to present perspectives appropriately.

v1.0 (2026-03): Initial implementation.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.grounding.perspective_engine import (
    DebateType,
    DebateAnalysis,
    Perspective,
)

logger = logging.getLogger(__name__)


# =========================================================================
# Base prompt (applies to all contested topics)
# =========================================================================

_BASE_BALANCED_PROMPT = """
## BALANCED PERSPECTIVE MODE — ACTIVE

This topic has been identified as contested or opinion-heavy.
Multiple perspective searches have been performed to ensure you have
evidence from different viewpoints.

**CORE PRINCIPLE:** Your job is to INFORM, never to PERSUADE.
Present each perspective at its strongest, then let the user decide.

**MANDATORY RULES:**
1. Present each perspective fairly — find the strongest version of
   each argument, not a strawman.
2. Use natural attribution for each viewpoint (phrases provided below).
3. Do NOT tell the user what to think or which side is "correct."
4. Where empirical evidence clearly supports one position, you may note
   this factually — but still present the opposing view's best case.
5. Identify where perspectives AGREE (common ground often exists).
6. End by acknowledging this is a topic where reasonable people disagree.
""".strip()


# =========================================================================
# Debate-type-specific prompt additions
# =========================================================================

_TYPE_SPECIFIC_PROMPTS: dict[DebateType, str] = {
    DebateType.POLITICAL_LR: """
**POLITICAL TOPIC GUIDELINES:**
- Avoid loaded language from either side ("woke mob", "fascist", etc.)
- Present policy positions in terms of stated goals and trade-offs
- Note where polling data shows public opinion distribution
- Distinguish between the ideological argument and the practical evidence
- Remember: most people hold views from BOTH sides on different issues
""",

    DebateType.ECONOMIC_POLICY: """
**ECONOMIC POLICY GUIDELINES:**
- Distinguish between theoretical economic arguments and empirical evidence
- Note where economic data exists to test claims (and where it doesn't)
- Present trade-offs honestly: most economic policies have winners and losers
- Avoid implying one economic school is universally "correct"
- If there IS a strong empirical consensus, state it clearly
""",

    DebateType.AI_ETHICS: """
**AI ETHICS TOPIC GUIDELINES:**
- This field moves fast — note the recency of arguments and evidence
- Distinguish between near-term concrete risks and long-term speculative ones
- Present both safety-cautious and innovation-optimistic views fairly
- Note that many AI researchers hold nuanced positions between extremes
- Avoid catastrophising OR dismissing legitimate safety concerns
""",

    DebateType.SOCIAL_CULTURAL: """
**SOCIAL/CULTURAL TOPIC GUIDELINES:**
- Present values-based arguments with respect for both positions
- Distinguish between moral/ethical arguments and empirical claims
- Where data exists (e.g. crime stats, health outcomes), cite it neutrally
- Avoid language that codes one position as "modern" and the other as "backward"
- Note that cultural values vary significantly across regions and communities
""",

    DebateType.SCIENTIFIC_CONTESTED: """
**SCIENTIFIC TOPIC GUIDELINES:**
- CLEARLY distinguish between scientific consensus and public debate
- If scientific consensus exists (e.g. climate change), state it plainly
- Still present the strongest version of dissenting scientific views
- Separate legitimate minority scientific positions from misinformation
- Note the difference between "scientists disagree" and "the public disagrees"
""",

    DebateType.GEOPOLITICAL: """
**GEOPOLITICAL TOPIC GUIDELINES:**
- Present perspectives from MULTIPLE geopolitical positions, not just Western media
- Note that news coverage differs significantly depending on the source country
- Distinguish between stated motivations and strategic interests
- Avoid framing one side as inherently virtuous and the other as villainous
- Use realist/interest-based analysis alongside values-based arguments
""",

    DebateType.TECH_DEBATE: """
**TECHNOLOGY DEBATE GUIDELINES:**
- Focus on concrete trade-offs rather than hype or FUD
- Where benchmarks or data exist, reference them
- Note the difference between short-term and long-term arguments
- Present both "this is the future" and "this is overhyped" positions fairly
""",

    DebateType.GENERAL: """
**GENERAL DEBATE GUIDELINES:**
- Identify the strongest version of each side's argument
- Look for common ground between positions
- Where evidence exists, present it; where it doesn't, say so
""",
}


# =========================================================================
# Response structure template
# =========================================================================

_RESPONSE_STRUCTURE = """
**RESPONSE STRUCTURE:**
Present your response in this order:

1. **Brief context** — one or two sentences setting up what this debate is about
2. **{p1_label}** — present the strongest arguments and evidence
3. **{p2_label}** — present the strongest arguments and evidence
{p3_section}4. **Common ground** — where these perspectives agree (if applicable)
5. **Where the evidence lands** — what empirical data supports (if applicable)
6. **Closing** — "This is a topic where reasonable people disagree.
   These are the strongest arguments I found on each side."

**ATTRIBUTION PHRASES TO USE:**
{attribution_block}
"""


def _build_attribution_block(perspectives: list[Perspective]) -> str:
    """Build the attribution phrases section for the prompt."""
    lines = []
    for p in perspectives:
        if p.attribution_phrases:
            examples = " / ".join(f'"{phrase}"' for phrase in p.attribution_phrases[:2])
            lines.append(f"- For {p.label}: {examples}")
    return "\n".join(lines)


def _build_response_structure(perspectives: list[Perspective]) -> str:
    """Build the response structure section with perspective labels."""
    if len(perspectives) < 2:
        return ""

    p1 = perspectives[0].label
    p2 = perspectives[1].label
    p3_section = ""
    if len(perspectives) >= 3:
        p3 = perspectives[2].label
        p3_section = f"3b. **{p3}** — the middle ground or pragmatic position\n"

    attribution = _build_attribution_block(perspectives)

    return _RESPONSE_STRUCTURE.format(
        p1_label=p1,
        p2_label=p2,
        p3_section=p3_section,
        attribution_block=attribution,
    )


# =========================================================================
# Main prompt builder
# =========================================================================

def build_balanced_prompt(analysis: DebateAnalysis) -> str:
    """
    Build a complete system prompt injection for a contested topic.

    Combines:
    - Base balanced prompt (applies to all)
    - Debate-type-specific guidelines
    - Response structure with perspective labels
    - Attribution phrases

    Args:
        analysis: DebateAnalysis from perspective_engine.analyse_debate()

    Returns:
        Complete system prompt injection string.
    """
    parts = [_BASE_BALANCED_PROMPT]

    # Add type-specific guidelines
    type_prompt = _TYPE_SPECIFIC_PROMPTS.get(
        analysis.debate_type,
        _TYPE_SPECIFIC_PROMPTS[DebateType.GENERAL],
    )
    parts.append(type_prompt.strip())

    # Add response structure
    if analysis.perspectives:
        structure = _build_response_structure(analysis.perspectives)
        if structure:
            parts.append(structure.strip())

    return "\n\n".join(parts)


__all__ = [
    "build_balanced_prompt",
]
