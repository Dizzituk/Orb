# FILE: app/llm/routing/rag_fallback.py
# Purpose: RAG fallback detection for architecture/codebase queries.
# Called-by: app.llm.routing, app.llm.routing.chat_routing
# Depends-on: app.llm.routing.conversational_gate
# Last-renovated: 2026-07-01
"""
RAG fallback detection for architecture/codebase queries.

v1.0 (2026-01-20): Extracted from stream_router.py (v4.12 RAG fallback).

This module provides:
- High-precision patterns for detecting architecture questions
- `is_architecture_query()` - Check if message should trigger RAG fallback

Used when translation layer fails or returns None to catch plain-English
codebase questions that would otherwise fall through to chat mode.
"""

from __future__ import annotations

import re
from typing import List, Set

from .conversational_gate import is_conversational_turn

# =============================================================================
# ARCHITECTURE QUERY PATTERNS (v4.12)
# =============================================================================

# High-precision patterns for architecture/codebase questions
# These catch queries when translation layer fails/returns None
_ARCHITECTURE_QUERY_PATTERNS: List[re.Pattern] = [
    # "Where is X" patterns (broad - catches most arch questions)
    re.compile(
        r"^[Ww]here\s+is\s+(?:the\s+)?(?:main\s+)?(?:\w+\s+){0,6}"
        r"(?:entrypoint|entry\s*point|router|stream|handler|function|class|module|file|config|constant|routing|implementation|trigger|pipeline|gate)[s]?"
    ),
    
    # "Show me where X" patterns
    re.compile(
        r"^[Ss]how\s+(?:me\s+)?where\s+.+"
        r"(?:is\s+)?(?:implemented|defined|located|triggered|called|used|loaded|handled|routed|processed)"
    ),
    
    # "Find where X" patterns
    re.compile(
        r"^[Ff]ind\s+(?:where|the\s+file\s+where)\s+.+"
        r"(?:is\s+)?(?:implemented|defined|located|triggered|called|used|loaded|handled|routed|processed|routes)"
    ),
    
    # "Find the file where X" patterns
    re.compile(
        r"^[Ff]ind\s+(?:the\s+)?(?:file|module|class|function)\s+(?:where|that)\s+.+"
    ),
    
    # "Find/List/Show call sites/callers of X" patterns
    re.compile(
        r"^(?:[Ff]ind|[Ll]ist|[Ss]how)\s+(?:the\s+)?(?:call\s*sites?|callers?)\s+(?:of|for)\s+.+"
    ),
    
    # "Who calls X" patterns
    re.compile(r"^[Ww]ho\s+calls\s+.+"),
    
    # Codebase-specific questions with known ASTRA terms
    re.compile(
        r"^(?:[Ww]here|[Hh]ow|[Ww]hat)\s+.+"
        r"(?:[Ss]pec\s*[Gg]ate|[Oo]verwatcher|[Ww]eaver|[Cc]ritical\s*[Pp]ipeline|"
        r"[Ss]tream\s*[Rr]outer|[Tt]ranslation\s*[Ll]ayer|[Rr][Aa][Gg]|[Ee]mbedding)"
        r"\s*.+[?.!]?$"
    ),
]

# =========================================================================
# NATURAL CODEBASE QUESTION PATTERNS (v10.0 — Job 10A)
# =========================================================================
# Broader patterns that catch natural questions. Require a guard keyword
# to prevent false positives on general knowledge questions.

_NATURAL_CODEBASE_PATTERNS: List[re.Pattern] = [
    re.compile(r"^[Hh]ow\s+many\s+(?:times?\s+)?(?:does|do|is|are)\s+.+", re.IGNORECASE),
    re.compile(
        r"^[Hh]ow\s+(?:hard|difficult|easy|complex)\s+would\s+it\s+be\s+to\s+"
        r"(?:refactor|rename|change|replace|update|migrate|remove|extract|split|merge|consolidate)\s+.+",
        re.IGNORECASE,
    ),
    re.compile(r"^[Ww]hat\s+does?\s+(?:the\s+)?(?:\w+\s+){0,3}(?:do|handle|manage|control|process|route)\s*\??", re.IGNORECASE),
    re.compile(r"^[Hh]ow\s+does\s+(?:the\s+)?(?:\w+\s+){0,3}(?:work|function|operate|run)\s*\??", re.IGNORECASE),
    re.compile(r"^[Tt]ell\s+me\s+about\s+(?:the\s+)?.+", re.IGNORECASE),
    re.compile(r"^[Ee]xplain\s+(?:the\s+)?(?:how\s+)?.+", re.IGNORECASE),
    re.compile(r"^[Ww]hat\s+happens\s+(?:when|if)\s+.+", re.IGNORECASE),
    re.compile(r"^[Ww]here\s+(?:does|do|is|are)\s+.+", re.IGNORECASE),
    re.compile(r"^[Ww]hich\s+(?:files?|modules?|functions?|classes?|components?)\s+.+", re.IGNORECASE),
    # v2026-06-14: feature/data questions ("what data does the chat window have",
    # "what does the finance panel show", "what's in the sidebar"). These rarely
    # contain a guard keyword, so they don't fire is_architecture_query on their
    # own — but they make needs_llm_codebase_check() True so the LLM safety-net
    # adjudicates (and they route directly when a guard keyword IS present).
    re.compile(r"^[Ww]hat\s+(?:data|info|information|fields?|columns?|state|content|settings?)\s+(?:does|do|is|are|can)\s+.+", re.IGNORECASE),
    re.compile(r"^[Ww]hat\s+does?\s+(?:the\s+)?(?:\w+\s+){0,4}(?:show|display|contain|store|hold|track|keep|have)\b", re.IGNORECASE),
    re.compile(r"^[Ww]hat(?:'s| is)\s+(?:in|stored\s+in|inside|kept\s+in)\s+(?:the\s+)?.+", re.IGNORECASE),
]

# Guard keywords — must contain at least one for natural patterns to fire.
# Mirrors CODEBASE_GUARD_KEYWORDS in _tier0_codebase_questions.py.
_CODEBASE_GUARD_KEYWORDS: Set[str] = {
    "weaver", "specgate", "spec gate", "overwatcher", "sandbox",
    "translation", "pipeline", "implementer",
    "memory", "rag", "confidence", "preference", "context",
    "knowledge", "embedding", "store", "domain store",
    "codebase", "module", "file", "function", "class", "method",
    "import", "dependency", "endpoint", "handler", "router",
    "provider", "registry", "stream", "hook", "tier",
    "refactor", "rename", "migrate", "extract", "decompose",
    "architecture", "subsystem", "component",
    "database", "table", "sqlite", "orb.db",
}


def _has_guard_keyword(text: str) -> bool:
    """Check if text contains at least one codebase guard keyword."""
    text_lower = text.lower()
    for kw in _CODEBASE_GUARD_KEYWORDS:
        if " " in kw:
            if kw in text_lower:
                return True
        else:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                return True
    return False


# Command prefix pattern - never route explicit commands to RAG fallback
_COMMAND_PREFIX_PATTERN = re.compile(r"^[Aa]stra[,:]?\s*command:\s*", re.IGNORECASE)


def is_architecture_query(message: str) -> bool:
    """
    Check if a message is a plain-English architecture/codebase question.
    
    Used as a fallback when translation layer fails or returns None.
    
    Two-stage matching:
        1. High-precision patterns (original) — always fire
        2. Natural language patterns (Job 10A) — require guard keyword
    
    Args:
        message: User message to check
    
    Returns:
        True if message matches an architecture query pattern
        False for explicit command prefixes (e.g. "Astra, command: ...")
        False for conversational turns (talking ABOUT the codebase —
        conversational_gate, 2026-07-01 — e.g. "I'm thinking about tidying
        the memory" must stay in chat, not divert to the RAG answerer)
    """
    text = message.strip()
    
    # Never trigger for explicit command prefix
    if _COMMAND_PREFIX_PATTERN.match(text):
        return False
    
    # Stage 1: High-precision patterns (no guard needed). These shapes are
    # already command-like lookups ("where is X implemented", "who calls Y"),
    # so they deliberately BYPASS the conversational gate — a trailing musing
    # clause must not strip a locate question of its grounding (2026-07-01).
    for pattern in _ARCHITECTURE_QUERY_PATTERNS:
        if pattern.match(text):
            return True
    
    # Conversational gate (2026-07-01): musings/discussion about the repo,
    # memory, architecture etc. are chat, not RAG queries. Guards only the
    # broad natural shapes below; a genuine imperative anywhere in the
    # message ("map the repo", "refactor X") also passes through.
    if is_conversational_turn(text):
        return False
    
    # Stage 2: Natural patterns (guard keyword required)
    for pattern in _NATURAL_CODEBASE_PATTERNS:
        if pattern.match(text):
            if _has_guard_keyword(text):
                return True
            # Pattern matched but no guard keyword — don't trigger
            break
    
    return False


# =========================================================================
# LLM SAFETY-NET GATE (v2026-06-14)
# =========================================================================
# DELIBERATELY NARROWER than _NATURAL_CODEBASE_PATTERNS. The LLM safety-net
# pays a real (blocking) model round-trip per fire, so the gate must not match
# everyday questions. Adversarial review found the broad shapes ("tell me about
# X", "explain X", bare "how does X work") fired on ~85% of ordinary chat. These
# shapes are anchored on "the/your/its/our <feature>" — the way feature/UI
# questions are actually phrased ("how does THE finance panel work", "what data
# does THE chat window have") — which excludes general-knowledge questions
# ("how does a mortgage work", "how does photosynthesis work", "tell me about
# the Roman Empire") that use a different article or none.
_LLM_NET_PATTERNS: List[re.Pattern] = [
    re.compile(
        r"^[Hh]ow\s+(?:does|do|did)\s+(?:the|your|its|our)\s+(?:\w+\s+){0,4}"
        r"(?:works?|function|functions?|operate|run|behave|flow)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^[Ww]hat\s+does?\s+(?:the|your|its|our)\s+(?:\w+\s+){0,4}"
        r"(?:do|does|handle|manage|control|process|route|show|display|contain|"
        r"store|hold|track|keep|have|return)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^[Ww]hat\s+(?:data|info|information|fields?|columns?|state|content|settings?)\s+"
        r"(?:does|do|is|are|can)\s+(?:the|your|its|our)\s+.+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^[Ww]hat(?:'s| is)\s+(?:in|stored\s+in|inside|kept\s+in)\s+(?:the|your)\s+.+",
        re.IGNORECASE,
    ),
    re.compile(
        r"^[Ww]hich\s+(?:files?|modules?|functions?|classes?|components?|tables?|"
        r"endpoints?|handlers?)\s+.+",
        re.IGNORECASE,
    ),
]


def needs_llm_codebase_check(message: str) -> bool:
    """
    True when a message LOOKS like an ASTRA feature/data question (matches a
    narrow, "the <feature>"-anchored shape) but lacks a guard keyword — so
    is_architecture_query() declined it and it fell to plain chat.

    These are the ONLY messages worth paying for the LLM safety-net
    (codebase_intent_llm.is_codebase_question_llm). The pattern set is kept
    deliberately tight (see _LLM_NET_PATTERNS) so general-knowledge questions and
    small talk never trigger the model call.

    Returns False for explicit command prefixes and for messages already caught
    by the high-precision patterns (those route via is_architecture_query).
    """
    text = (message or "").strip()
    if not text or _COMMAND_PREFIX_PATTERN.match(text):
        return False

    # Conversational gate (2026-07-01): don't pay the LLM round-trip for a
    # turn that is talk ABOUT a feature/topic rather than a question needing
    # a grounded answer.
    if is_conversational_turn(text):
        return False

    # High-precision patterns already route deterministically — no LLM needed.
    for pattern in _ARCHITECTURE_QUERY_PATTERNS:
        if pattern.match(text):
            return False

    # Narrow feature-shape match WITHOUT a guard keyword → candidate for the net.
    for pattern in _LLM_NET_PATTERNS:
        if pattern.match(text):
            return not _has_guard_keyword(text)

    return False


def get_pattern_count() -> int:
    """Get the number of architecture query patterns (for testing)."""
    return len(_ARCHITECTURE_QUERY_PATTERNS)


__all__ = [
    "is_architecture_query",
    "needs_llm_codebase_check",
    "get_pattern_count",
]
