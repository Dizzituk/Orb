# FILE: app/translation/_tier0_codebase_questions.py
"""
Tier 0: RAG codebase question detection.

Extracted from tier0_rules.py (v10.0) to consolidate all RAG-related
Tier 0 patterns in one place and keep the parent file under 30KB.

Contains:
    1. check_rag_codebase_query()     — Existing rigid patterns (explicit commands
                                         + structured architecture questions)
    2. check_natural_codebase_question() — NEW Job 10A natural language patterns
                                           with guard keyword filtering

Both route to RAG_CODEBASE_QUERY intent. The rigid patterns fire first
in the tier0_classify() chain (confidence 0.95–1.0). Natural patterns
fire second as a catch-all (confidence 0.92).

v10.0 (2026-02-23): Extracted from tier0_rules.py + added natural patterns (Job 10A)
"""
from __future__ import annotations

import re
from typing import List, Set

from .schemas import CanonicalIntent


# Import Tier0RuleResult from the parent — avoids circular import
# since tier0_rules.py imports us, not the other way around.
from .tier0_rules import Tier0RuleResult


# =============================================================================
# GUARD KEYWORD SET (Job 10A)
# =============================================================================
# Natural language questions must contain at least one of these terms
# to route as RAG_CODEBASE_QUERY. Prevents false positives on general
# knowledge questions like "how does photosynthesis work?"

CODEBASE_GUARD_KEYWORDS: Set[str] = {
    # ASTRA component names
    "weaver", "specgate", "spec gate", "overwatcher", "sandbox",
    "translation", "pipeline", "implementer",
    # Memory system
    "memory", "rag", "confidence", "preference", "context",
    "knowledge", "embedding", "store", "domain store",
    # Code structure
    "codebase", "module", "file", "function", "class", "method",
    "import", "dependency", "endpoint", "handler", "router",
    "provider", "registry", "stream", "hook", "tier",
    # Development actions
    "refactor", "rename", "migrate", "extract", "decompose",
    # Architecture
    "architecture", "subsystem", "component",
    # Database
    "database", "table", "sqlite", "orb.db",
}


# =============================================================================
# EXISTING RIGID PATTERNS (extracted from tier0_rules.py)
# =============================================================================

def check_rag_codebase_query(text: str) -> Tier0RuleResult:
    """
    Existing handler for RAG codebase queries.

    Searches indexed codebase and answers questions.
    v1.5: Fixed patterns to match trailing punctuation (?, !)
    v1.4: Expanded patterns to catch architecture/codebase questions.
    v1.3: Added RAG query support.
    v10.0: Extracted to _tier0_codebase_questions.py
    """
    text_stripped = text.strip()
    text_lower = text_stripped.lower()

    # Remove trailing punctuation for pattern matching
    text_clean = text_lower.rstrip('?.!')

    # Index commands
    index_patterns = [
        r"^index\s+(?:the\s+)?(?:architecture|codebase|rag)$",
        r"^run\s+rag\s+index$",
    ]

    for pattern in index_patterns:
        if re.match(pattern, text_clean):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.RAG_CODEBASE_QUERY,
                confidence=1.0,
                rule_name="rag_index",
                reason="RAG index command detected",
            )

    # Explicit search queries (highest confidence)
    explicit_search_patterns = [
        r"^search\s+(?:the\s+)?codebase:\s*.+",
        r"^ask\s+about\s+(?:the\s+)?codebase:\s*.+",
        r"^codebase\s+(?:search|query):\s*.+",
        r"^in\s+(?:the|this)\s+codebase,?\s+.+",
    ]

    for pattern in explicit_search_patterns:
        if re.match(pattern, text_lower):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.RAG_CODEBASE_QUERY,
                confidence=1.0,
                rule_name="rag_explicit_search",
                reason="RAG codebase search detected",
            )

    # Architecture-related questions (auto-route to RAG)
    architecture_question_patterns = [
        # Entry points / main flow
        r"^what\s+(?:are|is)\s+(?:the\s+)?(?:main\s+)?entry\s*points?$",
        r"^where\s+(?:are|is)\s+(?:the\s+)?(?:main\s+)?entry\s*points?$",
        r"^show\s+(?:me\s+)?(?:the\s+)?entry\s*points?$",
        r"^(?:list|find)\s+(?:the\s+)?entry\s*points?$",

        # Bottlenecks / performance
        r"^what\s+(?:are|is)\s+(?:the\s+)?(?:potential\s+)?bottlenecks?$",
        r"^where\s+(?:are|is)\s+(?:the\s+)?bottlenecks?$",
        r"^(?:find|identify|show)\s+(?:the\s+)?bottlenecks?$",
        r"^bottlenecks$",

        # Coupling / dependencies
        r"^(?:which|what)\s+modules?\s+(?:are|is)\s+(?:tightly\s+)?coupled",
        r"^where\s+(?:is|are)\s+(?:the\s+)?tight\s+coupling",
        r"^show\s+(?:me\s+)?(?:the\s+)?dependencies",

        # Where should X live / feature placement
        r"^where\s+should\s+(?:a\s+)?(?:new\s+)?(?:feature|functionality|code|module|component)\s+.+\s+(?:live|go)",
        r"^where\s+should\s+I\s+(?:put|add|place|implement)\s+.+",
        r"^where\s+does\s+.+\s+(?:belong|go|fit)",

        # What functions/classes handle X
        r"^what\s+(?:function|class|method|module|file)s?\s+(?:handle|process|manage|control)s?\s+.+",
        r"^(?:which|what)\s+(?:function|class|method|module|file)s?\s+(?:are\s+)?responsible\s+for\s+.+",
        r"^(?:which|what)\s+(?:function|class|method|module|file)s?\s+(?:do|does)\s+.+",

        # Where is X implemented/defined/located
        r"^where\s+is\s+.+\s+(?:implemented|defined|located|declared|found)",
        r"^find\s+(?:the\s+)?(?:implementation|definition|location)\s+of\s+.+",
        r"^(?:locate|find)\s+.+\s+(?:in\s+the\s+codebase|code)",

        # How does X work / flow
        r"^how\s+does\s+(?:the\s+)?(?:routing|streaming|pipeline|job|auth|memory|rag)\s+(?:work|function|flow)",
        r"^explain\s+(?:the\s+)?(?:routing|streaming|pipeline|job|auth|memory|rag)\s+(?:flow|system|architecture)",

        # Optimization / improvement
        r"^how\s+(?:would|should|could)\s+(?:you|I|we)\s+optimize\s+.+",
        r"^(?:suggest|recommend)\s+(?:optimizations?|improvements?)\s+(?:for|to)\s+.+",

        # Code structure questions
        r"^what\s+is\s+the\s+(?:purpose|role|responsibility)\s+of\s+.+",
        r"^what\s+does\s+(?:the\s+)?(?:file|module|class|function)\s+.+\s+do",
        r"^describe\s+(?:the\s+)?(?:file|module|class|function|component)\s+.+",

        # List/show components
        r"^(?:list|show|display)\s+(?:all\s+)?(?:the\s+)?(?:modules?|components?|services?|handlers?|routers?)$",
        r"^what\s+(?:modules?|components?|services?)\s+(?:exist|are\s+there)",
    ]

    for pattern in architecture_question_patterns:
        if re.match(pattern, text_clean):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.RAG_CODEBASE_QUERY,
                confidence=0.95,
                rule_name="rag_architecture_question",
                reason=f"Architecture question detected: '{text_clean}'",
            )

    return Tier0RuleResult(matched=False)


# =============================================================================
# NEW NATURAL LANGUAGE PATTERNS (Job 10A)
# =============================================================================

_NATURAL_CODEBASE_PATTERNS: List[re.Pattern] = [
    # "How many X in the codebase/system/code" patterns
    re.compile(
        r"^[Hh]ow\s+many\s+(?:times?\s+)?(?:does|do|is|are)\s+.+",
        re.IGNORECASE,
    ),

    # "How hard/difficult/easy would it be to X" about code
    re.compile(
        r"^[Hh]ow\s+(?:hard|difficult|easy|complex)\s+would\s+it\s+be\s+to\s+"
        r"(?:refactor|rename|change|replace|update|migrate|remove|extract|split|merge|consolidate)\s+.+",
        re.IGNORECASE,
    ),

    # "What does X do/handle/manage" about system components
    re.compile(
        r"^[Ww]hat\s+does?\s+(?:the\s+)?(?:\w+\s+){0,3}"
        r"(?:do|handle|manage|control|process|route)\s*\??",
        re.IGNORECASE,
    ),

    # "How does X work" about system
    re.compile(
        r"^[Hh]ow\s+does\s+(?:the\s+)?(?:\w+\s+){0,3}"
        r"(?:work|function|operate|run)\s*\??",
        re.IGNORECASE,
    ),

    # "Tell me about X" where X contains known ASTRA terms
    re.compile(
        r"^[Tt]ell\s+me\s+about\s+(?:the\s+)?.+",
        re.IGNORECASE,
    ),

    # "Explain X" where X is a system concept
    re.compile(
        r"^[Ee]xplain\s+(?:the\s+)?(?:how\s+)?.+",
        re.IGNORECASE,
    ),

    # "Can you show/tell me how X" patterns
    re.compile(
        r"^[Cc]an\s+you\s+(?:show|tell|explain)\s+.+",
        re.IGNORECASE,
    ),

    # "What happens when X" about system behaviour
    re.compile(
        r"^[Ww]hat\s+happens\s+(?:when|if)\s+.+",
        re.IGNORECASE,
    ),

    # "Where does X happen/live/go" about code location
    re.compile(
        r"^[Ww]here\s+(?:does|do|is|are)\s+.+",
        re.IGNORECASE,
    ),

    # "Which files/modules/functions X" about code inventory
    re.compile(
        r"^[Ww]hich\s+(?:files?|modules?|functions?|classes?|components?)\s+.+",
        re.IGNORECASE,
    ),
]


def _message_has_guard_keyword(text: str) -> bool:
    """
    Check if message contains at least one codebase guard keyword.

    Uses case-insensitive substring matching. Multi-word keywords
    (e.g. "spec gate", "domain store") are checked as substrings.
    Single-word keywords use word-boundary matching to prevent
    partial matches (e.g. "file" shouldn't match "profile").
    """
    text_lower = text.lower()
    for kw in CODEBASE_GUARD_KEYWORDS:
        if " " in kw:
            # Multi-word: substring match
            if kw in text_lower:
                return True
        else:
            # Single word: word boundary match
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                return True
    return False


def check_natural_codebase_question(text: str) -> Tier0RuleResult:
    """
    Catch natural language codebase questions that miss the rigid patterns.

    Two-stage filter:
        1. Pattern match — does the question LOOK like a codebase question?
        2. Guard keyword — does it contain at least one ASTRA/code term?

    If pattern matches but no guard keyword → fall through to Tier 1.
    This prevents "how does photosynthesis work?" from hitting RAG.

    Confidence: 0.92 (below rigid architecture patterns at 0.95,
    above obvious chat at 0.95 which uses a different intent).

    v10.0 (2026-02-23): Job 10A — broadened codebase question detection.
    """
    text_stripped = text.strip()

    # Step 1: Pattern match
    matched_pattern = False
    for pattern in _NATURAL_CODEBASE_PATTERNS:
        if pattern.match(text_stripped):
            matched_pattern = True
            break

    if not matched_pattern:
        return Tier0RuleResult(matched=False)

    # Step 2: Guard keyword check
    if not _message_has_guard_keyword(text_stripped):
        # Pattern matched but no ASTRA/code keyword — let Tier 1 decide
        return Tier0RuleResult(
            matched=False,
            reason="Natural pattern matched but no guard keyword — deferring to Tier 1",
        )

    return Tier0RuleResult(
        matched=True,
        intent=CanonicalIntent.RAG_CODEBASE_QUERY,
        confidence=0.92,
        rule_name="rag_natural_codebase_question",
        reason=f"Natural codebase question detected: '{text_stripped[:60]}'",
    )
