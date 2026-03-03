# FILE: app/grounding/perspective_engine.py
"""
Balanced Perspective Engine for the Grounded Intelligence System.

When a topic is classified as CONTESTED, this engine:
1. Detects what kind of debate it is (political, economic, tech, social, etc.)
2. Identifies the likely "sides" or perspectives
3. Generates targeted search queries per perspective to surface the
   strongest arguments from each viewpoint
4. Performs multi-perspective web searches
5. Tags each source with the perspective it represents

The goal: ASTRA never picks a side. It presents the strongest case
from each perspective and lets the user form their own view.

v1.0 (2026-03): Initial implementation.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


# =========================================================================
# Debate types and perspective definitions
# =========================================================================

class DebateType(str, Enum):
    """Categories of contested topics."""
    POLITICAL_LR = "political_left_right"      # Left vs right political spectrum
    ECONOMIC_POLICY = "economic_policy"          # Different economic schools
    TECH_DEBATE = "tech_debate"                  # Tech industry disagreements
    SOCIAL_CULTURAL = "social_cultural"          # Social/cultural values debates
    SCIENTIFIC_CONTESTED = "scientific_contested" # Scientific topics with public disagreement
    GEOPOLITICAL = "geopolitical"                # International relations / foreign policy
    AI_ETHICS = "ai_ethics"                      # AI-specific ethics and safety debates
    GENERAL = "general"                          # Catch-all


@dataclass(frozen=True)
class Perspective:
    """A single perspective/viewpoint in a debate."""
    label: str             # Short label: "Progressive view", "Market-based approach"
    description: str       # One-line description of the viewpoint
    search_suffixes: list  # Query suffixes to surface this perspective
    attribution_phrases: list  # How to attribute this view in prose


@dataclass
class DebateAnalysis:
    """Analysis of what kind of debate a topic represents."""
    debate_type: DebateType
    topic_core: str                          # The core topic stripped of framing
    perspectives: List[Perspective] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    confidence: float = 0.0


# =========================================================================
# Perspective templates per debate type
# =========================================================================

_PERSPECTIVE_TEMPLATES: dict[DebateType, list[Perspective]] = {
    DebateType.POLITICAL_LR: [
        Perspective(
            label="Progressive / left-leaning view",
            description="Emphasis on social equity, government intervention, collective solutions",
            search_suffixes=[
                "progressive argument",
                "left wing perspective",
                "social justice case for",
            ],
            attribution_phrases=[
                "From a progressive standpoint",
                "Those on the left argue",
                "Left-leaning analysts point to",
            ],
        ),
        Perspective(
            label="Conservative / right-leaning view",
            description="Emphasis on individual liberty, free markets, traditional values",
            search_suffixes=[
                "conservative argument",
                "right wing perspective",
                "free market case against",
            ],
            attribution_phrases=[
                "From a conservative standpoint",
                "Those on the right argue",
                "Right-leaning commentators counter",
            ],
        ),
        Perspective(
            label="Centrist / pragmatic view",
            description="Emphasis on evidence-based policy, compromise, practical outcomes",
            search_suffixes=[
                "centrist analysis",
                "pragmatic middle ground",
                "evidence based policy",
            ],
            attribution_phrases=[
                "Centrist analysts suggest",
                "A more moderate position holds",
                "Pragmatists on both sides note",
            ],
        ),
    ],

    DebateType.ECONOMIC_POLICY: [
        Perspective(
            label="Free-market / supply-side view",
            description="Lower taxes, deregulation, private sector solutions",
            search_suffixes=[
                "free market argument",
                "supply side economics case",
                "deregulation benefits",
            ],
            attribution_phrases=[
                "Free-market economists argue",
                "Supply-side proponents point to",
                "Market-oriented analysts suggest",
            ],
        ),
        Perspective(
            label="Interventionist / demand-side view",
            description="Government spending, regulation, public investment",
            search_suffixes=[
                "keynesian argument",
                "government intervention case",
                "public investment benefits",
            ],
            attribution_phrases=[
                "Keynesian economists counter",
                "Interventionist analysts argue",
                "Those favouring public investment point to",
            ],
        ),
        Perspective(
            label="Empirical consensus (where it exists)",
            description="What the data actually shows regardless of ideology",
            search_suffixes=[
                "economic evidence data",
                "empirical research shows",
                "economists agree",
            ],
            attribution_phrases=[
                "The empirical evidence suggests",
                "Data from multiple studies indicates",
                "Most economists across the spectrum agree",
            ],
        ),
    ],

    DebateType.AI_ETHICS: [
        Perspective(
            label="AI safety / cautious view",
            description="Prioritise alignment, regulation, existential risk mitigation",
            search_suffixes=[
                "AI safety risks argument",
                "AI regulation case",
                "alignment research importance",
            ],
            attribution_phrases=[
                "AI safety researchers argue",
                "Those in the cautious camp point to",
                "Safety-focused commentators warn",
            ],
        ),
        Perspective(
            label="AI acceleration / optimistic view",
            description="Prioritise innovation speed, open access, transformative benefits",
            search_suffixes=[
                "AI progress benefits argument",
                "effective accelerationism",
                "AI innovation case",
            ],
            attribution_phrases=[
                "AI optimists counter",
                "Acceleration proponents argue",
                "Those favouring rapid development point to",
            ],
        ),
        Perspective(
            label="Governance / balanced regulation view",
            description="Targeted regulation without stifling innovation",
            search_suffixes=[
                "AI governance balanced approach",
                "targeted AI regulation",
                "AI policy framework",
            ],
            attribution_phrases=[
                "Policy researchers suggest a middle path",
                "Governance-focused analysts propose",
                "Those seeking balanced regulation argue",
            ],
        ),
    ],

    DebateType.TECH_DEBATE: [
        Perspective(
            label="Pro / in favour",
            description="Arguments supporting the technology or approach",
            search_suffixes=[
                "benefits advantages argument",
                "case for why better",
                "supporters say",
            ],
            attribution_phrases=[
                "Proponents argue",
                "Supporters of this approach point to",
                "Those in favour highlight",
            ],
        ),
        Perspective(
            label="Against / critical view",
            description="Arguments against or critiquing the technology or approach",
            search_suffixes=[
                "problems criticism argument against",
                "case against drawbacks",
                "critics say",
            ],
            attribution_phrases=[
                "Critics counter",
                "Sceptics point to",
                "Those against this approach highlight",
            ],
        ),
    ],

    DebateType.SOCIAL_CULTURAL: [
        Perspective(
            label="Progressive / reform view",
            description="Emphasis on changing existing norms and structures",
            search_suffixes=[
                "progressive case for reform",
                "why change needed",
                "social justice perspective",
            ],
            attribution_phrases=[
                "Progressive voices argue",
                "Reform advocates point to",
                "Those pushing for change highlight",
            ],
        ),
        Perspective(
            label="Traditional / conservative view",
            description="Emphasis on preserving existing norms and caution about change",
            search_suffixes=[
                "traditional values argument",
                "conservative case against",
                "concerns about changing",
            ],
            attribution_phrases=[
                "Traditionalist voices counter",
                "Those cautious about change argue",
                "Conservative commentators point to",
            ],
        ),
    ],

    DebateType.GEOPOLITICAL: [
        Perspective(
            label="Western / NATO-aligned view",
            description="Perspective from Western governments and aligned media",
            search_suffixes=[
                "western analysis perspective",
                "NATO allied view",
                "US European perspective",
            ],
            attribution_phrases=[
                "Western analysts argue",
                "NATO-aligned commentators point to",
                "The Western perspective holds",
            ],
        ),
        Perspective(
            label="Alternative / non-Western view",
            description="Perspective from non-aligned or opposing nations",
            search_suffixes=[
                "non western perspective analysis",
                "global south view",
                "alternative geopolitical analysis",
            ],
            attribution_phrases=[
                "Non-Western analysts counter",
                "From an alternative geopolitical lens",
                "Commentators outside the Western bloc argue",
            ],
        ),
        Perspective(
            label="Realist / neutral analysis",
            description="Interest-based analysis without ideological framing",
            search_suffixes=[
                "realist geopolitical analysis",
                "neutral strategic assessment",
                "interests based analysis",
            ],
            attribution_phrases=[
                "Realist analysts observe",
                "From a strategic-interest perspective",
                "Neutral observers note",
            ],
        ),
    ],

    DebateType.SCIENTIFIC_CONTESTED: [
        Perspective(
            label="Scientific consensus view",
            description="What the majority of peer-reviewed research supports",
            search_suffixes=[
                "scientific consensus evidence",
                "peer reviewed research shows",
                "scientific community position",
            ],
            attribution_phrases=[
                "The scientific consensus holds",
                "Peer-reviewed research indicates",
                "The majority of researchers agree",
            ],
        ),
        Perspective(
            label="Dissenting / minority scientific view",
            description="Legitimate minority positions or emerging counter-evidence",
            search_suffixes=[
                "scientific criticism dissent",
                "counter evidence challenges",
                "minority scientific view",
            ],
            attribution_phrases=[
                "A minority of researchers argue",
                "Dissenting scientists point to",
                "Some emerging evidence challenges this, suggesting",
            ],
        ),
    ],
}

# General fallback uses simple pro/con/middle
_PERSPECTIVE_TEMPLATES[DebateType.GENERAL] = [
    Perspective(
        label="Arguments in favour",
        description="The strongest case for",
        search_suffixes=["arguments for benefits case", "supporters say advantages"],
        attribution_phrases=["Supporters argue", "Those in favour point to"],
    ),
    Perspective(
        label="Arguments against",
        description="The strongest case against",
        search_suffixes=["arguments against criticism case", "critics say drawbacks"],
        attribution_phrases=["Critics counter", "Those against argue"],
    ),
    Perspective(
        label="Middle ground",
        description="Where both sides find common ground or the evidence is mixed",
        search_suffixes=["balanced analysis evidence both sides", "nuanced view"],
        attribution_phrases=["A balanced view suggests", "Both sides tend to agree"],
    ),
]


# =========================================================================
# Debate type detection
# =========================================================================

_DEBATE_DETECTORS: dict[DebateType, list[re.Pattern]] = {
    DebateType.POLITICAL_LR: [
        re.compile(r"\b(?:left|right)\s+(?:wing|leaning)", re.I),
        re.compile(r"\b(?:liberal|conservative|progressive|populist|socialist)\b", re.I),
        re.compile(r"\b(?:democrat|republican|labour|tory|tories)\b", re.I),
        re.compile(r"\b(?:woke|anti[-\s]?woke)\b", re.I),
    ],
    DebateType.ECONOMIC_POLICY: [
        re.compile(r"\b(?:ubi|universal\s+basic\s+income|austerity)\b", re.I),
        re.compile(r"\b(?:wealth\s+(?:tax|inequality|gap)|redistribution)\b", re.I),
        re.compile(r"\b(?:privatis|nationalis|deregulat|subsid)", re.I),
        re.compile(r"\b(?:minimum\s+wage|trade\s+(?:deal|war|tariff))\b", re.I),
        re.compile(r"\b(?:welfare|tax\s+(?:cut|rise|increase|reform)|spending\s+cut)\b", re.I),
    ],
    DebateType.AI_ETHICS: [
        re.compile(r"\b(?:ai\s+(?:safety|risk|alignment|regulation|sentien|consciousness))\b", re.I),
        re.compile(r"\b(?:regulat\w*\s+(?:ai|artificial\s+intelligence))", re.I),
        re.compile(r"\b(?:agi\s+(?:timeline|risk|doom))\b", re.I),
        re.compile(r"\b(?:open\s+source\s+vs|ai\s+(?:ethics|bias|fairness))\b", re.I),
        re.compile(r"\b(?:effective\s+accelerat|e/acc|doomer)\b", re.I),
        re.compile(r"\bai\b.*\b(?:danger|threat|existential|destroy|replace\s+(?:jobs|humans))", re.I),
    ],
    DebateType.SOCIAL_CULTURAL: [
        re.compile(r"\b(?:immigration(?:\s+(?:policy|debate|crisis))?)\b", re.I),
        re.compile(r"\b(?:gun\s+(?:control|rights)|abortion|death\s+penalty)\b", re.I),
        re.compile(r"\b(?:free\s+speech|censorship|cancel\s+culture)\b", re.I),
        re.compile(r"\b(?:gender\s+(?:identity|debate)|transgender)\b", re.I),
    ],
    DebateType.SCIENTIFIC_CONTESTED: [
        re.compile(r"\b(?:climate\s+(?:change|crisis|hoax)|net\s+zero)\b", re.I),
        re.compile(r"\b(?:vaccine|vaccination)s?\s*(?:debate|safety|mandate|safe|danger|risk)?\b", re.I),
        re.compile(r"\b(?:nuclear\s+(?:energy|power)\s+(?:safe|danger))\b", re.I),
    ],
    DebateType.GEOPOLITICAL: [
        re.compile(r"\b(?:war\s+in|conflict\s+in|invasion\s+of)\b", re.I),
        re.compile(r"\b(?:nato|brics|sanctions?\s+(?:on|against))\b", re.I),
        re.compile(r"\b(?:china|russia|iran|israel|palestine)\b.*\b(?:should|right|wrong|fault)\b", re.I),
    ],
    DebateType.TECH_DEBATE: [
        re.compile(r"\b(?:open\s+source|closed\s+source)\b", re.I),
        re.compile(r"\bis\s+\S+\s+better\s+than\s+\S+\b", re.I),
        re.compile(r"\b(?:blockchain|web3|crypto)\s+(?:scam|future|dead|useful)\b", re.I),
    ],
}


def _detect_debate_type(text: str) -> DebateType:
    """Detect what kind of debate a contested topic represents."""
    best_type = DebateType.GENERAL
    best_hits = 0

    for dtype, patterns in _DEBATE_DETECTORS.items():
        hits = sum(1 for rx in patterns if rx.search(text))
        if hits > best_hits:
            best_hits = hits
            best_type = dtype

    return best_type


def _extract_topic_core(text: str) -> str:
    """Strip conversational framing to get the core topic."""
    cleaned = re.sub(
        r"^(?:what\s+do\s+(?:you|people)\s+think\s+about\s+|"
        r"is\s+it\s+(?:right|wrong|fair|ethical)\s+(?:to|that)\s+|"
        r"should\s+(?:we|they|the\s+government)\s+|"
        r"tell\s+me\s+about\s+(?:the\s+)?(?:debate|controversy)\s+(?:over|about)\s+|"
        r"what(?:'s|\s+is)\s+(?:your|the)\s+(?:view|opinion|take)\s+on\s+)",
        "", text.strip(), flags=re.IGNORECASE,
    ).strip().rstrip("?!.")
    return cleaned if len(cleaned) >= 3 else text.strip()


# =========================================================================
# Main analysis function
# =========================================================================

def analyse_debate(text: str) -> DebateAnalysis:
    """
    Analyse a contested topic to determine the debate type, perspectives,
    and generate targeted search queries for each perspective.

    Args:
        text: The user's message (already classified as CONTESTED).

    Returns:
        DebateAnalysis with perspectives and search queries.
    """
    debate_type = _detect_debate_type(text)
    topic_core = _extract_topic_core(text)
    perspectives = _PERSPECTIVE_TEMPLATES.get(debate_type, _PERSPECTIVE_TEMPLATES[DebateType.GENERAL])

    # Generate search queries: one neutral + one per perspective
    queries = [topic_core]  # Neutral query first
    for p in perspectives:
        # Pick the first suffix for each perspective
        if p.search_suffixes:
            q = f"{topic_core} {p.search_suffixes[0]}"
            queries.append(q.strip())

    logger.info(
        "[perspective_engine] Debate analysis: type=%s, core='%s', "
        "perspectives=%d, queries=%d",
        debate_type.value, topic_core[:50], len(perspectives), len(queries),
    )

    return DebateAnalysis(
        debate_type=debate_type,
        topic_core=topic_core,
        perspectives=list(perspectives),
        search_queries=queries,
        confidence=0.8 if debate_type != DebateType.GENERAL else 0.6,
    )


__all__ = [
    "DebateType",
    "Perspective",
    "DebateAnalysis",
    "analyse_debate",
]
