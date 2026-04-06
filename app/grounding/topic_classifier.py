# FILE: app/grounding/topic_classifier.py
"""
Topic Classifier for the Grounded Intelligence System.

Categorises inbound messages into one of three buckets:
  - PERSONAL:   Conversational, about the user, preferences, schedules,
                personal context. No grounding required.
  - CLAIMS:     Factual, current-state, data-dependent. MUST be grounded
                via web search before ASTRA responds.
  - CONTESTED:  Opinion-heavy, politically/socially divisive. Requires
                balanced multi-perspective search (Part Two).

Classification is rule-based for high-confidence cases (keywords, patterns)
with a lightweight heuristic scorer for ambiguous inputs.

v1.0 (2026-03): Initial implementation.
v1.1 (2026-03-10): Personal signals win when they outnumber claims signals.
    Brainstorming/feature-description messages no longer trigger web searches.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


# =========================================================================
# Topic categories
# =========================================================================

class TopicCategory(str, Enum):
    PERSONAL = "personal"
    CLAIMS = "claims"
    CONTESTED = "contested"


@dataclass(frozen=True)
class ClassificationResult:
    """Result of topic classification."""
    category: TopicCategory
    confidence: float                    # 0.0 - 1.0
    matched_signals: List[str] = field(default_factory=list)
    suggested_search_queries: List[str] = field(default_factory=list)
    domain_hint: str = ""                # e.g. "finance", "ai_tech", "geopolitics"


# =========================================================================
# Signal dictionaries
# =========================================================================

# Keywords/phrases that strongly indicate PERSONAL (no grounding needed)
_PERSONAL_SIGNALS = {
    # Self-referential
    r"\bmy\s+(keto|diet|workout|routine|schedule|route|van|deliveries?)\b",
    r"\bhow(?:'s| is)\s+my\b",
    r"\bremind\s+me\b",
    r"\bwhat\s+(?:do|should)\s+i\b",
    r"\bhow\s+(?:am|was)\s+i\b",
    # ASTRA system queries
    r"\bpipeline\b",
    r"\bweaver\b",
    r"\bspec\s*gate\b",
    r"\boverwatcher\b",
    r"\bsandbox\b",
    r"\bcodebase\b",
    r"\bcode\s+base\b",
    r"\barchitecture\s+map\b",
    r"\bembedding\b",
    # Casual greetings / meta
    r"^(?:hi|hello|hey|morning|afternoon|evening|cheers|thanks?|ta)\b",
    r"^(?:how\s+are\s+you|what(?:'s| is)\s+up)\b",
    # v1.1: Brainstorming / feature description / app planning
    r"\b(?:i(?:'ve| have)\s+been\s+thinking\s+about\s+(?:building|making|creating))\b",
    r"\b(?:i\s+want\s+to\s+(?:build|make|create|design))\b",
    r"\b(?:let(?:'s| us)\s+(?:build|make|create|design|plan|map\s+out))\b",
    r"\b(?:companion\s+app|phone\s+app|mobile\s+app|web\s+app|desktop\s+app)\b",
    r"\b(?:voice\s+input|text\s+back|dashboard\s+showing)\b",
    r"\b(?:feature|mvp|prototype|wireframe|mockup|sketch\s+out)\b",
    r"\b(?:nothing\s+crazy|simple\s+(?:app|tool|page))\b",
    r"\b(?:what\s+(?:do\s+you|can\s+you)\s+think\s+about)\b",
    r"\b(?:back\s+at\s+base|on\s+the\s+road|out\s+doing\s+deliveries)\b",
}

# Keywords that strongly indicate CLAIMS (grounding required)
_CLAIMS_SIGNALS = {
    # Temporal / currency markers
    r"\b(?:current(?:ly)?|latest|today(?:'s)?|recent(?:ly)?|right\s+now)\b",
    r"\b(?:this\s+(?:week|month|quarter|year))\b",
    r"\b(?:yesterday|last\s+(?:week|month|night))\b",
    # Domain: finance / markets
    # v1.1: Removed bare "share" — false positive on "population share", "share this" etc.
    r"\b(?:market\s+(?:cap|crash|rally|correction)|stock\s+(?:market|price|exchange))\b",
    r"\b(?:crypto|bitcoin|ethereum|tao|trading)\b",
    r"\b(?:gdp|inflation|interest\s+rate|unemployment|recession|tariff)\b",
    r"\b(?:ftse|s&p|nasdaq|dow\s+jones|nikkei)\b",
    r"\b(?:exchange\s+rate|forex|currency)\b",
    # Domain: AI / tech
    r"\b(?:gpt[-\s]?\d|claude[-\s]?\d|gemini[-\s]?\d|llama[-\s]?\d|mistral)\b",
    r"\b(?:openai|anthropic|google\s+(?:ai|deepmind)|meta\s+ai)\b",
    r"\b(?:new\s+model|model\s+release|benchmark|leaderboard)\b",
    # Domain: geopolitics / world affairs
    r"\b(?:war\s+in|conflict\s+in|sanctions?\s+(?:on|against))\b",
    r"\b(?:election|vote|referendum|summit|treaty|ceasefire)\b",
    r"\b(?:nato|un|eu\s+(?:policy|decision)|g7|g20|brics)\b",
    # Domain: news / events
    r"\b(?:breaking\s+news|headline|announced|signed|passed\s+(?:a|the)\s+(?:bill|law))\b",
    r"\b(?:happened|happening|unfolding|developing\s+story)\b",
    # Explicit factual queries
    r"\bwho\s+is\s+(?:the\s+)?(?:current|new)\b",
    r"\bwhat\s+(?:is|are)\s+the\s+(?:latest|current|new)\b",
    r"\bhow\s+much\s+(?:is|does|are)\b",
    r"\bwhat\s+happened\s+(?:with|to|in|at)\b",
}

# Keywords that indicate CONTESTED (balanced perspective needed)
_CONTESTED_SIGNALS = {
    # Political framing
    r"\b(?:left\s+(?:wing|leaning)|right\s+(?:wing|leaning)|liberal|conservative)\b",
    r"\b(?:woke|anti[-\s]?woke|progressive|populist|nationalist)\b",
    r"\b(?:should\s+(?:we|they|the\s+government))\b",
    r"\b(?:is\s+it\s+(?:right|wrong|fair|ethical|moral)\s+(?:to|that))\b",
    # Social / cultural debate
    r"\b(?:immigration\s+(?:policy|debate|crisis))\b",
    r"\b(?:gun\s+(?:control|rights)|abortion|death\s+penalty)\b",
    r"\b(?:climate\s+(?:change|crisis|hoax)|net\s+zero)\b",
    r"\b(?:free\s+speech|censorship|cancel\s+culture)\b",
    # Economic ideology
    r"\b(?:universal\s+basic\s+income|ubi|austerity|socialism|capitalism)\b",
    r"\b(?:wealth\s+(?:tax|inequality|gap|redistribution))\b",
    r"\b(?:privatisation|nationalisation|deregulation)\b",
    # AI-specific debates
    r"\b(?:ai\s+(?:risk|safety|alignment|regulation|consciousness|sentien))\b",
    r"\b(?:agi\s+(?:timeline|risk|doom))\b",
    r"\b(?:open\s+source\s+vs|closed\s+source)\b",
    # Comparative / opinion-seeking
    r"\b(?:what\s+do\s+(?:you|people)\s+think\s+about)\b",
    r"\b(?:is\s+\S+\s+better\s+than\s+\S+\s+(?:for|at|in))\b",
    r"\b(?:debate|controversy|contentious|divisive|polarising)\b",
}

# Domain detection (for freshness policy lookup)
_DOMAIN_PATTERNS = {
    "finance":      [r"\b(?:markets?|stock|crypto|trading|gdp|inflation|tariffs?|ftse|s&p|forex|economy)\b"],
    "ai_tech":      [r"\b(?:gpt|claude|gemini|llama|mistral|openai|anthropic|model|benchmark|ai\b)"],
    "geopolitics":  [r"\b(?:war|conflict|sanctions?|election|nato|un\b|eu\b|summit|treaty)\b"],
    "uk_affairs":   [r"\b(?:uk|britain|british|parliament|nhs|boe|bank\s+of\s+england)\b"],
    "health":       [r"\b(?:study\s+(?:shows?|finds?)|clinical\s+trial|vaccine|treatment|diagnosis)\b"],
    "surf_weather": [r"\b(?:surf|swell|wave|tide|forecast|bodyboard)\b"],
}


# Compile all patterns once at import time
_PERSONAL_RX = [re.compile(p, re.IGNORECASE) for p in _PERSONAL_SIGNALS]
_CLAIMS_RX = [re.compile(p, re.IGNORECASE) for p in _CLAIMS_SIGNALS]
_CONTESTED_RX = [re.compile(p, re.IGNORECASE) for p in _CONTESTED_SIGNALS]
_DOMAIN_RX = {
    domain: [re.compile(p, re.IGNORECASE) for p in patterns]
    for domain, patterns in _DOMAIN_PATTERNS.items()
}


# =========================================================================
# Classification logic
# =========================================================================

def _count_matches(text: str, patterns: list) -> list[str]:
    """Return list of pattern names that matched."""
    matched = []
    for rx in patterns:
        if rx.search(text):
            matched.append(rx.pattern[:60])
    return matched


def _detect_domain(text: str) -> str:
    """Return the best-matching domain hint, or empty string."""
    best_domain = ""
    best_count = 0
    for domain, patterns in _DOMAIN_RX.items():
        hits = sum(1 for rx in patterns if rx.search(text))
        if hits > best_count:
            best_count = hits
            best_domain = domain
    return best_domain


def _extract_search_queries(text: str, category: TopicCategory) -> list[str]:
    """Generate suggested search queries from the user's message.

    v1.1: Skip extraction for long messages (>150 words) — these are
    instructions/prompts, not factual questions.  For shorter messages,
    extract the first sentence rather than the last 6 words to avoid
    grabbing irrelevant tail content (e.g. footer text, design notes).
    """
    cleaned = re.sub(
        r"^(?:can\s+you\s+|could\s+you\s+|please\s+|i\s+want\s+to\s+know\s+|"
        r"tell\s+me\s+(?:about\s+)?|what(?:'s| is)\s+|explain\s+)",
        "", text.strip(), flags=re.IGNORECASE,
    ).strip().rstrip("?!.")

    if not cleaned or len(cleaned) < 3:
        return []

    words = cleaned.split()

    # Long messages are instructions/prompts, not search-worthy questions.
    # Grounding gate should not try to web-search a 500-word design brief.
    if len(words) > 150:
        return []

    # For moderate-length messages, extract the first sentence as the
    # primary query — it's far more likely to contain the actual question
    # than the tail of the message.
    first_sentence_match = re.match(r'^(.+?[.?!])\s', cleaned)
    if first_sentence_match and len(first_sentence_match.group(1).split()) >= 4:
        first_sentence = first_sentence_match.group(1).rstrip(".")
        queries = [first_sentence[:200]]
    else:
        queries = [cleaned[:200]]

    # Add a shorter version if the query is still long
    if len(words) > 8:
        short = " ".join(words[:6])  # First 6 words, not last 6
        if short != queries[0]:
            queries.append(short)

    return queries[:3]


def classify_topic(text: str) -> ClassificationResult:
    """
    Classify a user message into PERSONAL, CLAIMS, or CONTESTED.

    Priority order:
    1. PERSONAL wins when personal signals >= claims signals (brainstorming guard)
    2. CONTESTED checked second (claims + opinion framing = contested)
    3. CLAIMS checked third (factual grounding needed)
    4. Default: PERSONAL (assume conversational if nothing matches)
    """
    if not text or not text.strip():
        return ClassificationResult(
            category=TopicCategory.PERSONAL,
            confidence=1.0,
            matched_signals=["empty_input"],
        )
    text_clean = text.strip()

    # v1.1: Long messages (>150 words) are instructions/prompts/design briefs,
    # not factual questions that need grounding.  Skip classification entirely
    # to avoid false-positive claims detection on embedded data/statistics.
    if len(text_clean.split()) > 150:
        return ClassificationResult(
            category=TopicCategory.PERSONAL,
            confidence=0.85,
            matched_signals=["long_message_instruction"],
            domain_hint="",
        )
    # Count signal matches
    personal_hits = _count_matches(text_clean, _PERSONAL_RX)
    claims_hits = _count_matches(text_clean, _CLAIMS_RX)
    contested_hits = _count_matches(text_clean, _CONTESTED_RX)

    p_count = len(personal_hits)
    c_count = len(claims_hits)
    x_count = len(contested_hits)

    domain = _detect_domain(text_clean)

    # 1. Personal wins when personal signals outnumber or equal claims signals.
    #    This prevents brainstorming messages from triggering web searches
    #    just because they contain an incidental claims word like "happening".
    if p_count > 0 and p_count >= c_count and x_count == 0:
        return ClassificationResult(
            category=TopicCategory.PERSONAL,
            confidence=min(1.0, 0.7 + (p_count * 0.1)),
            matched_signals=personal_hits,
            domain_hint=domain,
        )

    # 2. Contested signals present -> CONTESTED (supersedes claims)
    if x_count > 0:
        queries = _extract_search_queries(text_clean, TopicCategory.CONTESTED)
        confidence = min(1.0, 0.6 + (x_count * 0.15))
        if c_count > 0:
            confidence = min(1.0, confidence + 0.1)
        return ClassificationResult(
            category=TopicCategory.CONTESTED,
            confidence=confidence,
            matched_signals=contested_hits + claims_hits,
            suggested_search_queries=queries,
            domain_hint=domain,
        )

    # 3. Claims signals present -> CLAIMS (grounding required)
    if c_count > 0:
        queries = _extract_search_queries(text_clean, TopicCategory.CLAIMS)
        confidence = min(1.0, 0.6 + (c_count * 0.12))
        return ClassificationResult(
            category=TopicCategory.CLAIMS,
            confidence=confidence,
            matched_signals=claims_hits,
            suggested_search_queries=queries,
            domain_hint=domain,
        )

    # 4. Heuristic: question about real-world state with no explicit signals
    _QUESTION_ABOUT_WORLD = re.compile(
        r"(?:what|how|who|where|when|why|is|are|has|have|did|does|will)\b.{10,}",
        re.IGNORECASE,
    )
    if _QUESTION_ABOUT_WORLD.match(text_clean):
        is_about_self = bool(re.search(r"\b(?:my|i|me|our|astra|pipeline)\b", text_clean, re.IGNORECASE))
        if not is_about_self and len(text_clean.split()) >= 5:
            queries = _extract_search_queries(text_clean, TopicCategory.CLAIMS)
            return ClassificationResult(
                category=TopicCategory.CLAIMS,
                confidence=0.5,
                matched_signals=["heuristic_world_question"],
                suggested_search_queries=queries,
                domain_hint=domain,
            )

    # 5. Default: personal/conversational
    return ClassificationResult(
        category=TopicCategory.PERSONAL,
        confidence=0.6,
        matched_signals=["default_personal"],
        domain_hint=domain,
    )


__all__ = [
    "TopicCategory",
    "ClassificationResult",
    "classify_topic",
]
