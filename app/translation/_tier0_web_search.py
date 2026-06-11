# FILE: app/translation/_tier0_web_search.py
"""
Tier 0 rule: Web search intent detection.

Detects natural language requests to search the web and routes
to the WEB_SEARCH intent. Read-only, no confirmation required.

Patterns:
  - "search the web for X"
  - "you need to search the web for X"
  - "go search the web for X"
  - "look up X"
  - "get me the latest on X"
  - "what's the latest on the web about X"
  - "search for X online"
  - "find out about X"
  - "google X" / "brave search X"

v2.1 (2026-02): Initial implementation.
"""
from __future__ import annotations

import re
from typing import Optional

from app.translation.schemas import CanonicalIntent


class Tier0RuleResult:
    """Minimal result compatible with tier0_rules.Tier0RuleResult."""

    def __init__(
        self,
        matched: bool,
        intent: Optional[CanonicalIntent] = None,
        confidence: float = 1.0,
        rule_name: Optional[str] = None,
        reason: Optional[str] = None,
        extracted_query: Optional[str] = None,
    ):
        self.matched = matched
        self.intent = intent
        self.confidence = confidence
        self.rule_name = rule_name
        self.reason = reason
        self.extracted_query = extracted_query


# =========================================================================
# Conversational prefix stripper
# =========================================================================
# Handles: "you need to ...", "can you ...", "go ...", "please ...", etc.

_CONVERSATIONAL_PREFIX = re.compile(
    r"^(?:"
    r"(?:you\s+need\s+to\s+)"
    r"|(?:can\s+you\s+)"
    r"|(?:could\s+you\s+)"
    r"|(?:go\s+(?:and\s+)?)"
    r"|(?:please\s+)"
    r"|(?:i\s+need\s+you\s+to\s+)"
    r"|(?:i\s+want\s+you\s+to\s+)"
    r")",
    re.IGNORECASE,
)


def _strip_conversational_prefix(text: str) -> str:
    """Remove polite/conversational prefixes."""
    return _CONVERSATIONAL_PREFIX.sub("", text).strip()


# =========================================================================
# Explicit web search patterns (high confidence)
# =========================================================================
# Each tuple: (compiled regex, pattern name, query capture group index)

_EXPLICIT_PATTERNS = [
    # "search the web for X" / "search online for X" / "web search X"
    (re.compile(
        r"^(?:search\s+(?:the\s+)?(?:web|internet|online)\s+(?:for\s+)?)"
        r"(.+)$",
        re.IGNORECASE,
    ), "search_web_for", 1),

    # "web search: X"
    (re.compile(
        r"^web\s+search:?\s+(.+)$",
        re.IGNORECASE,
    ), "web_search_prefix", 1),

    # "look up X" / "look up X online"
    (re.compile(
        r"^look\s+up\s+(.+?)(?:\s+online|\s+on\s+the\s+(?:web|internet))?$",
        re.IGNORECASE,
    ), "look_up", 1),

    # "google X" / "brave search X"
    (re.compile(
        r"^(?:google|brave\s+search)\s+(.+)$",
        re.IGNORECASE,
    ), "colloquial_search", 1),

    # "search for X online" / "search for X on the web"
    (re.compile(
        r"^search\s+for\s+(.+?)\s+(?:online|on\s+the\s+(?:web|internet))$",
        re.IGNORECASE,
    ), "search_for_online", 1),

    # "find out about X" / "find information on X"
    (re.compile(
        r"^find\s+(?:out\s+about|information\s+(?:on|about))\s+(.+)$",
        re.IGNORECASE,
    ), "find_out_about", 1),

    # "get me the latest on X" / "get the latest X"
    (re.compile(
        r"^get\s+(?:me\s+)?the\s+latest\s+(?:on\s+|about\s+|news\s+(?:on|about)\s+)?"
        r"(.+)$",
        re.IGNORECASE,
    ), "get_latest", 1),

    # "what's the latest on X" / "what's the latest on the web about X"
    (re.compile(
        r"^what(?:'s|\s+is)\s+the\s+latest"
        r"(?:\s+(?:on|about|from)\s+(?:the\s+)?(?:web|internet)\s+(?:on|about|for))?"
        r"\s+(?:on|about|with|for|regarding)?\s*"
        r"(.+)$",
        re.IGNORECASE,
    ), "whats_latest", 1),

    # "research X" / "research X for me"
    (re.compile(
        r"^research\s+(.+?)(?:\s+for\s+me)?$",
        re.IGNORECASE,
    ), "research", 1),

    # "check online for X" / "check the web for X"
    (re.compile(
        r"^check\s+(?:the\s+)?(?:web|internet|online)\s+(?:for\s+)?"
        r"(.+)$",
        re.IGNORECASE,
    ), "check_online", 1),

    # "have a look online for X"
    (re.compile(
        r"^have\s+a\s+look\s+(?:online|on\s+the\s+(?:web|internet))\s+(?:for\s+)?"
        r"(.+)$",
        re.IGNORECASE,
    ), "have_a_look", 1),
]


# =========================================================================
# Contextual web search patterns (medium confidence)
# These need the query to NOT match codebase-related keywords.
# =========================================================================

_CONTEXTUAL_PATTERNS = [
    # "what's the current price of X" / "what is the current status of X"
    (re.compile(
        r"^what(?:'s|\s+is)\s+the\s+current\s+(.+)$",
        re.IGNORECASE,
    ), "whats_current", 1),

    # "how much does X cost" / "how much is X"
    (re.compile(
        r"^how\s+much\s+(?:does|is|are)\s+(.+?)(?:\s+cost)?$",
        re.IGNORECASE,
    ), "how_much", 1),

    # "is X still Y" - implies checking current state
    (re.compile(
        r"^is\s+(.+\s+still\s+.+)$",
        re.IGNORECASE,
    ), "is_still", 1),
]


# Keywords that indicate the user is asking about the codebase, not the web.
# Single-word guards use whole-word matching; multi-word guards use substring.
_CODEBASE_GUARD_WORDS_SINGLE = {
    # Backend / pipeline terms
    "codebase", "pipeline", "specgate", "overwatcher", "weaver",
    "sandbox", "module", "function", "endpoint",
    "router", "handler", "fixture", "refactor",
    # Frontend / build terms (v2.3)
    "component", "components", "stylesheet", "jsx", "tsx",
    "frontend", "backend", "vite", "react", "typescript",
    # Build intent terms (v2.3)
    "build", "create", "implement", "dummy", "tab",
    "screenshot", "screenshots", "spacing", "styling",
}
_CODEBASE_GUARD_WORDS_MULTI = {
    "code base", "spec gate",
    # Frontend / UI phrases (v2.3)
    "front end", "front-end", "the code", "search the code",
    "the style", "the styles",
    # Local-file-context phrases (v2.4): suppress web search when the
    # user is clearly referring to a local artefact / running file, not
    # asking the system to look something up online. These fire only on
    # contextual/fallback/staleness — explicit "search the web for X"
    # patterns still pass through.
    "the html", "the dashboard", "the spreadsheet",
    "is already live", "is live", "already exists",
    "go in there", "go into the", "open the",
    "look at it", "look at the", "look in the",
    "add what i", "update what i", "change what i",
    "the work folder", "this folder", "this file",
}


def _has_codebase_keywords(text: str) -> bool:
    """Check if text contains codebase-related keywords.
    
    Uses whole-word matching for single words to avoid substring
    false positives (e.g. 'test' matching inside 'latest').
    """
    lower = text.lower()
    words = set(re.findall(r'[a-z]+', lower))
    if words & _CODEBASE_GUARD_WORDS_SINGLE:
        return True
    return any(mg in lower for mg in _CODEBASE_GUARD_WORDS_MULTI)



# =========================================================================
# DEEP_RESEARCH patterns — thorough/iterative research
# =========================================================================

_DEEP_RESEARCH_PATTERNS = [
    # "research X thoroughly" / "research X in depth"
    (re.compile(
        r"^research\s+(.+?)\s+(?:thoroughly|in\s+depth|in\s+detail|deeply|comprehensively)$",
        re.IGNORECASE,
    ), "research_thoroughly", 1),

    # "deep dive into X" / "deep search X"
    (re.compile(
        r"^deep\s+(?:dive\s+(?:into|on)|search\s+(?:for\s+)?|research\s+(?:into\s+)?)\s*(.+)$",
        re.IGNORECASE,
    ), "deep_dive", 1),

    # "do a deep search on X" / "do deep research on X"
    (re.compile(
        r"^do\s+(?:a\s+)?deep\s+(?:search|research|dive)\s+(?:on|into|about|for)\s+(.+)$",
        re.IGNORECASE,
    ), "do_deep_search", 1),

    # "thoroughly research X" / "thoroughly look into X"
    (re.compile(
        r"^thoroughly\s+(?:research|look\s+into|investigate|search\s+for)\s+(.+)$",
        re.IGNORECASE,
    ), "thoroughly_research", 1),

    # "investigate X" / "investigate X for me"
    (re.compile(
        r"^investigate\s+(.+?)(?:\s+for\s+me)?$",
        re.IGNORECASE,
    ), "investigate", 1),

    # "I need a full breakdown of X" / "give me a full breakdown of X"
    (re.compile(
        r"^(?:i\s+need|give\s+me)\s+(?:a\s+)?(?:full|complete|comprehensive|detailed)\s+"
        r"(?:breakdown|analysis|overview|report|summary)\s+(?:of|on|about)\s+(.+)$",
        re.IGNORECASE,
    ), "full_breakdown", 1),

    # "dig into X" / "dig deeper into X"
    (re.compile(
        r"^dig\s+(?:deeper?\s+)?into\s+(.+)$",
        re.IGNORECASE,
    ), "dig_into", 1),
]


# =========================================================================
# Keyword-based query extraction (v2.2)
# =========================================================================
# When structural regex patterns fail to match natural language web search
# requests, this fallback extracts a search query by:
# 1. Checking for co-occurrence of web-context + action keywords
# 2. Stripping out noise words (go, online, and, find, the, for, me, etc.)
# 3. Returning the remaining content as the search query

_WEB_NOISE_WORDS = {
    "go", "online", "and", "the", "web", "internet", "for", "me",
    "find", "search", "look", "up", "check", "get", "fetch",
    "show", "pull", "grab", "browse", "discover", "research",
    "please", "can", "you", "could", "would", "need", "want",
    "to", "i", "a", "some", "about", "on", "in", "at", "of",
    "what", "is", "are", "there", "whats", "what's",
}

_KEYWORD_WEB_CONTEXT = {
    "online", "web", "internet", "latest", "news", "today",
    "recent", "current", "trending", "happening", "updates",
    "headlines", "google", "brave",
}

_KEYWORD_ACTION_VERBS = {
    "search", "find", "look", "check", "get", "fetch",
    "show", "pull", "grab", "browse", "research",
    "investigate", "discover", "happening",
}


def _has_lifestyle_diary_keywords(text: str) -> bool:
    """v3.1 (2026-06-11): personal food-diary commands must never be
    swallowed by the web-search fallbacks. 'Copy yesterday's food into
    today', 'log my lunch', 'I'm eating the same as yesterday' were
    matching the keyword/staleness fallbacks and routing to WEB_SEARCH
    instead of the lifestyle domain. Requires BOTH a diary action or
    first-person eating cue AND a food noun, so genuine searches like
    'best food in Lisbon' still pass through."""
    t = text.lower()
    if not re.search(
        r"\b(food|meals?|nutrition|diet|diary|macros?|calories|breakfast|lunch|dinner|snack)\b", t
    ):
        return False
    action = re.search(
        r"\b(log|logged|copy|copied|duplicate|repeat|mirror|carry|track|record)\b", t
    )
    first_person = re.search(
        r"\b(i\s+(ate|had|eat)|i'?m\s+(eating|having)|what\s+(did|have)\s+i\s+(eat|eaten|ate)"
        r"|my\s+(food|meals?|diet|diary|macros|calories))\b", t
    )
    return bool(action or first_person)


def _extract_query_by_keywords(text: str) -> Optional[str]:
    """Extract a web search query using keyword co-occurrence.
    
    Returns the cleaned query string if web-search intent is detected,
    None otherwise. Intent requires both a web-context keyword AND
    an action verb to be present.
    """
    lower = text.lower().strip().rstrip("?!.")
    words = re.findall(r"[a-z']+", lower)
    word_set = set(words)
    
    has_context = bool(word_set & _KEYWORD_WEB_CONTEXT)
    has_action = bool(word_set & _KEYWORD_ACTION_VERBS)
    
    # v2.3: "research" alone implies external lookup — no context keyword needed
    _STRONG_ACTION_VERBS = {"research"}
    has_strong_action = bool(word_set & _STRONG_ACTION_VERBS) and len(word_set) >= 4
    
    if not ((has_context and has_action) or has_strong_action):
        return None
    # Strip noise words, keep meaningful content
    # Also keep words that are in context keywords if they're topical
    # (e.g. "news" is both a context keyword and a valid query term)
    topical_context = {"news", "latest", "updates", "headlines", "trending", "today"}
    meaningful = []
    for w in words:
        if w in _WEB_NOISE_WORDS and w not in topical_context:
            continue
        meaningful.append(w)
    
    query = " ".join(meaningful).strip()
    
    # Must have at least 2 chars of actual query content
    if len(query) < 2:
        return None
    
    return query



# =========================================================================
# Staleness detector (v3.0) — auto-trigger web search for stale topics
# =========================================================================
# Topics that change faster than training data can keep up with.
# If the user asks about any of these, ASTRA should verify online
# rather than confidently answering from potentially outdated knowledge.

# Current events, conflicts, politics — always changing
_STALENESS_CURRENT_EVENTS = {
    "war", "conflict", "invasion", "ceasefire", "bombing", "strike",
    "election", "voted", "referendum", "coup",
    "president", "prime minister", "chancellor", "leader",
    "sanctions", "tariff", "tariffs", "trade war", "embargo",
    "crisis", "emergency", "disaster", "earthquake", "hurricane",
    "protest", "riot", "demonstration",
}

# Technology and AI — moves fast
_STALENESS_TECH = {
    "gpt", "claude", "gemini", "llama", "mistral", "deepseek",
    "openai", "anthropic", "google ai", "meta ai",
    "model release", "new model", "benchmark", "evaluation",
    "api", "sdk", "framework", "library", "release",
    "gpu", "nvidia", "amd", "rtx", "cuda",
    "robotics", "self-driving", "autonomous",
}

# Finance and markets — real-time
_STALENESS_FINANCE = {
    "stock", "share price", "market", "bitcoin", "crypto",
    "interest rate", "inflation", "recession",
    "earnings", "quarterly", "ipo",
}

# Named entities that imply current state questions
_STALENESS_CURRENT_STATE = {
    "iran", "ukraine", "russia", "gaza", "israel", "taiwan",
    "china", "north korea", "syria",
    "nhs", "hmrc", "dwp",
}

# Single-word temporal signals — checked against tokenised words
# (whole-word match) to avoid substring collisions like "now" inside
# "know" or "current" inside "currents". Common conversational words
# ("now", "still", "yet", "update", "developments") removed because
# they fire on benign tutoring/discussion language.
_STALENESS_TEMPORAL_SINGLE = {
    "latest", "recent", "current", "today", "currently",
    "anymore", "trending", "headlines",
    "2026", "2025",
}

# Multi-word temporal phrases — substring-matched since they are
# deliberate phrasings unlikely to collide with normal language.
_STALENESS_TEMPORAL_MULTI = {
    "this week", "this month", "this year",
    "right now", "at the moment",
    "what happened", "what is happening", "what's happening",
}

# Statement / explanation markers — when the message is framed as
# the user's own reasoning, hypothesis, restatement, or answer, it
# is NOT a search request. Used to suppress staleness firing on
# tutoring and discussion language like "well, I think the answer
# is...". Order matters: prefixes are anchored at start of message.
_STALENESS_STATEMENT_PREFIXES = (
    "well,", "well ",
    "i think", "i believe", "i guess", "i reckon",
    "in my opinion",
    "the answer", "my answer",
    "so we", "so the",
    "actually,",
    "because", "the reason",
)

_STALENESS_STATEMENT_PHRASES = (
    "the answer is", "the answer would",
    "that's how", "that is how",
    "that's because", "thats because",
    "the reason is",
    "i would say",
    "it depends",
)


def _check_staleness(text: str) -> Optional[str]:
    """Check if the topic is likely to have stale training data.

    Returns a reason string if the topic is stale, None otherwise.
    The reason is used in logging to explain why auto-search triggered.

    v3.0 (2026-02): Initial implementation. Checks against known fast-moving
    topic categories. This is intentionally broad — better to search
    unnecessarily than to confidently give outdated information.

    v3.1 (2026-05-01): Tightened to stop false positives in tutoring and
    discussion contexts. Three guards added:

    1. Length cap of 30 words (separate from the explicit-search cap of
       50). Staleness signals on long messages are almost always part of
       an explanation, not a search request.
    2. Statement-marker guard. Messages framed as the user's own
       reasoning ("well, I think...", "the answer is...", "so we...")
       are not search requests, even if they contain "news" or "today".
    3. Whole-word match for single-word temporal signals via token-set
       intersection — "now" no longer fires inside "know", "current"
       no longer fires inside "currents". Conversational noise words
       ("now", "still", "yet", "update", "developments") removed from
       the temporal list entirely.
    """
    lower = text.lower()
    words = set(re.findall(r"[a-z']+", lower))

    # Length guard: staleness fires on short queries, not long explanations.
    if len(text.split()) > 30:
        return None

    # Statement guard: don't fire on the user's own reasoning / answers.
    stripped = lower.lstrip()
    if any(stripped.startswith(p) for p in _STALENESS_STATEMENT_PREFIXES):
        return None
    if any(p in lower for p in _STALENESS_STATEMENT_PHRASES):
        return None

    # Check temporal signals first — strongest indicator.
    # Multi-word phrases substring-matched, single words whole-word matched.
    for signal in _STALENESS_TEMPORAL_MULTI:
        if signal in lower:
            return f"temporal_signal:{signal}"
    matched_single = words & _STALENESS_TEMPORAL_SINGLE
    if matched_single:
        return f"temporal_signal:{sorted(matched_single)[0]}"

    # Check current events keywords
    for kw in _STALENESS_CURRENT_EVENTS:
        if kw in lower:
            return f"current_events:{kw}"

    # Check named entities that are fast-moving situations
    for entity in _STALENESS_CURRENT_STATE:
        if entity in lower:
            # Only trigger if the question seems to be about current state,
            # not historical facts. "When did Iran become a republic" = no.
            # "What's happening in Iran" = yes.
            _historical = {"when did", "when was", "history of", "in 1979",
                           "in the 1980s", "founded", "established"}
            if not any(h in lower for h in _historical):
                return f"current_state:{entity}"

    # Check tech/AI topics — but only if it seems like a capability
    # or status question, not a general concept question
    for kw in _STALENESS_TECH:
        if kw in lower:
            # "What is a GPU" = conceptual, no search needed
            # "What GPU should I buy" = current, search needed
            # "What can GPT-5 do" = capability, search needed
            _conceptual = {"what is a ", "what are ", "explain ", "define ",
                           "how does a ", "concept of "}
            _overrides = {"new ", "latest ", "current ", "feature", "update",
                          "release", "capable", "benchmark", "can "}
            is_conceptual = any(lower.startswith(c) for c in _conceptual)
            has_override = any(o in lower for o in _overrides)
            if not is_conceptual or has_override:
                return f"tech_stale:{kw}"

    # Check finance
    for kw in _STALENESS_FINANCE:
        if kw in lower:
            return f"finance:{kw}"

    return None

def check_deep_research_trigger(text: str) -> Tier0RuleResult:
    """Check if the user wants deep/iterative research."""
    text_stripped = text.strip()
    if not text_stripped or len(text_stripped) < 8:
        return Tier0RuleResult(matched=False)

    cleaned = _strip_conversational_prefix(text_stripped)

    for candidate in [cleaned, text_stripped]:
        if not candidate:
            continue
        for pattern, name, group_idx in _DEEP_RESEARCH_PATTERNS:
            m = pattern.match(candidate)
            if m:
                query = m.group(group_idx).strip()
                if query:
                    return Tier0RuleResult(
                        matched=True,
                        intent=CanonicalIntent.DEEP_RESEARCH,
                        confidence=1.0,
                        rule_name=f"deep_research_{name}",
                        reason=f"Deep research detected: {name}",
                        extracted_query=query,
                    )

    return Tier0RuleResult(matched=False)


def check_web_search_trigger(text: str) -> Tier0RuleResult:
    """
    Check if the user message is a web search request.

    Strips conversational prefixes first, then matches against
    explicit and contextual patterns.

    v2.3: Added length guard — messages over 50 words are never web
    searches. Long natural language descriptions of build tasks were
    being misclassified because they incidentally contained words like
    "search" and "online".
    """
    text_stripped = text.strip()
    if not text_stripped or len(text_stripped) < 5:
        return Tier0RuleResult(matched=False)

    # v2.4: Negation guard. If the user explicitly says this is NOT a
    # web search, do not match. Catches the rejection loop where the
    # phrase "not a web search" itself contained keywords ("web",
    # "search") that triggered the keyword fallback.
    _lower = text_stripped.lower()
    _negation_starts = ("no,", "no ", "nope", "not a ", "not an ", "this is not", "this isn't", "this ain't")
    _negation_about_search = ("not a web search", "not a search", "isn't a web search", "isn't a search", "ain't a search")
    if any(_lower.startswith(p) for p in _negation_starts) and any(p in _lower for p in _negation_about_search):
        return Tier0RuleResult(matched=False)
    # Also catch plain "no, this is not a web search" / "no this isn't a search"
    if any(p in _lower for p in _negation_about_search):
        return Tier0RuleResult(matched=False)

    # v2.3: Length guard — web search queries are short.
    # A 50+ word message is a build description or conversation, not a search.
    # v2.3.1: Raised to 100 for messages with explicit "research" keyword,
    # since voice-to-text input is verbose but "research X" is unambiguous.
    word_count = len(text_stripped.split())
    _has_research = "research" in text_stripped.lower().split()
    _word_limit = 100 if _has_research else 50
    if word_count > _word_limit:
        return Tier0RuleResult(matched=False)

    # Strip conversational prefixes:
    # "you need to search the web for X" -> "search the web for X"
    # "go search the web for X" -> "search the web for X"
    cleaned = _strip_conversational_prefix(text_stripped)

    # Try both the original and cleaned versions
    for candidate in [cleaned, text_stripped]:
        if not candidate:
            continue

        # 1) Explicit patterns - high confidence
        for pattern, name, group_idx in _EXPLICIT_PATTERNS:
            m = pattern.match(candidate)
            if m:
                query = m.group(group_idx).strip()
                if query:
                    # v2.3.1: The "research" pattern is ambiguous — it could
                    # target codebase or the web. Guard with codebase keywords.
                    if name == "research" and _has_codebase_keywords(query):
                        continue
                    return Tier0RuleResult(
                        matched=True,
                        intent=CanonicalIntent.WEB_SEARCH,
                        confidence=1.0,
                        rule_name=f"web_search_{name}",
                        reason=f"Web search detected: {name}",
                        extracted_query=query,
                    )

    # 2) Contextual patterns - only if no codebase keywords
    if not _has_codebase_keywords(text_stripped) and not _has_lifestyle_diary_keywords(text_stripped):
        for candidate in [cleaned, text_stripped]:
            if not candidate:
                continue
            for pattern, name, group_idx in _CONTEXTUAL_PATTERNS:
                m = pattern.match(candidate)
                if m:
                    query = m.group(group_idx).strip()
                    if query:
                        return Tier0RuleResult(
                            matched=True,
                            intent=CanonicalIntent.WEB_SEARCH,
                            confidence=0.85,
                            rule_name=f"web_search_{name}",
                            reason=f"Web search (contextual): {name}",
                            extracted_query=query,
                        )

    # 3) Keyword-based fallback (v2.2)
    # Catches natural phrasings like "go online and find AI news today"
    # that don't match any structural regex. Uses keyword co-occurrence
    # and extracts the query by stripping web/action words.
    if not _has_codebase_keywords(text_stripped) and not _has_lifestyle_diary_keywords(text_stripped):
        query = _extract_query_by_keywords(text_stripped)
        if query:
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.WEB_SEARCH,
                confidence=0.80,
                rule_name="web_search_keyword_fallback",
                reason="Web search (keyword co-occurrence fallback)",
                extracted_query=query,
            )

    # 4) STALENESS DETECTOR (v3.0)
    # If the topic is likely to have changed since training data cutoff,
    # auto-trigger web search even without explicit search keywords.
    # The user should never have to say "go online" — ASTRA should know
    # when its knowledge is likely stale and verify automatically.
    if not _has_codebase_keywords(text_stripped) and not _has_lifestyle_diary_keywords(text_stripped):
        staleness = _check_staleness(text_stripped)
        if staleness:
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.WEB_SEARCH,
                confidence=0.75,
                rule_name="web_search_staleness_auto",
                reason=f"Auto web search (stale topic: {staleness})",
                extracted_query=text_stripped,
            )

    return Tier0RuleResult(matched=False)

