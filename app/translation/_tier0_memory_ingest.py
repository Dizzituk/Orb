# FILE: app/translation/_tier0_memory_ingest.py
"""
Tier 0 rule: Memory ingest and store intent detection.

Two intents:
  MEMORY_INGEST — Bulk data ingestion (files, exports, uploads)
    - "ingest this into your memory"
    - "upload this to memory"
    - "process this data"
    - "import my GPT data"

  MEMORY_STORE — Remember a specific fact from conversation
    - "remember this"
    - "remember that I..."
    - "save this to memory"
    - "put this in your memory"
    - "don't forget that..."
    - "note that..."

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
# Conversational prefix stripper (shared logic)
# =========================================================================

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


def _strip_prefix(text: str) -> str:
    return _CONVERSATIONAL_PREFIX.sub("", text).strip()


# =========================================================================
# MEMORY_INGEST patterns — bulk data / file upload
# =========================================================================

_INGEST_PATTERNS = [
    # "ingest this into your memory" / "ingest this data"
    (re.compile(
        r"^ingest\s+(?:this|that|the)\s+(?:data\s+)?(?:into\s+(?:your\s+)?memory)?",
        re.IGNORECASE,
    ), "ingest_this"),

    # "upload this to memory" / "upload this into memory"
    (re.compile(
        r"^upload\s+(?:this|that|the\s+\w+)\s+(?:to|into)\s+(?:your\s+)?memory",
        re.IGNORECASE,
    ), "upload_to_memory"),

    # "import my GPT data" / "import my chatgpt export" / "import this data"
    (re.compile(
        r"^import\s+(?:my\s+)?(?:gpt|chatgpt|openai)?\s*(?:data|export|conversations?|history)",
        re.IGNORECASE,
    ), "import_data"),

    # "process this data" / "process this file into memory"
    (re.compile(
        r"^process\s+(?:this|that|the)\s+(?:data|file|export)\s*(?:into\s+(?:your\s+)?memory)?",
        re.IGNORECASE,
    ), "process_data"),

    # "load this into memory" / "load this data"
    (re.compile(
        r"^load\s+(?:this|that|the)\s+(?:data\s+|file\s+|export\s+)?(?:into\s+(?:your\s+)?memory)?",
        re.IGNORECASE,
    ), "load_into_memory"),

    # "add this to your knowledge" / "add this to your memory"
    (re.compile(
        r"^add\s+(?:this|that|the\s+\w+)\s+(?:to|into)\s+(?:your\s+)?(?:memory|knowledge)",
        re.IGNORECASE,
    ), "add_to_memory"),

    # "store this data" / "store this in memory"
    (re.compile(
        r"^store\s+(?:this|that|the)\s+(?:data\s+|file\s+)?(?:in(?:to)?\s+(?:your\s+)?memory)?",
        re.IGNORECASE,
    ), "store_data"),

    # "learn from this" / "learn from this data"
    (re.compile(
        r"^learn\s+(?:from\s+)?(?:this|that|the)\s*(?:data|file|export|conversations?)?",
        re.IGNORECASE,
    ), "learn_from"),
]


# =========================================================================
# MEMORY_STORE patterns — remember a specific fact
# =========================================================================

_STORE_PATTERNS = [
    # "remember this" / "remember that I..." / "remember that my..."
    (re.compile(
        r"^remember\s+(?:that\s+|this\s*$)",
        re.IGNORECASE,
    ), "remember", None),

    # "remember I am X" / "remember my name is X"
    (re.compile(
        r"^remember\s+(?:that\s+)?(.+)$",
        re.IGNORECASE,
    ), "remember_fact", 1),

    # "save this to memory" / "save this to your memory"
    (re.compile(
        r"^save\s+(?:this|that)\s+(?:to|in(?:to)?)\s+(?:your\s+)?memory",
        re.IGNORECASE,
    ), "save_to_memory", None),

    # "put this in your memory" / "put this into your memory"
    (re.compile(
        r"^put\s+(?:this|that)\s+(?:in(?:to)?)\s+(?:your\s+)?memory",
        re.IGNORECASE,
    ), "put_in_memory", None),

    # "don't forget that..." / "dont forget..."
    (re.compile(
        r"^don'?t\s+forget\s+(?:that\s+)?(.+)$",
        re.IGNORECASE,
    ), "dont_forget", 1),

    # "note that..." / "make a note that..."
    (re.compile(
        r"^(?:make\s+a\s+)?note\s+(?:that\s+)?(.+)$",
        re.IGNORECASE,
    ), "note_that", 1),

    # "keep in mind that..." / "bear in mind that..."
    (re.compile(
        r"^(?:keep|bear)\s+in\s+mind\s+(?:that\s+)?(.+)$",
        re.IGNORECASE,
    ), "keep_in_mind", 1),

    # "from now on..." / "going forward..." (preference-like)
    (re.compile(
        r"^(?:from\s+now\s+on|going\s+forward)\s*,?\s*(.+)$",
        re.IGNORECASE,
    ), "from_now_on", 1),

    # "I prefer X" / "I always want X" / "I like X"
    (re.compile(
        r"^i\s+(?:prefer|always\s+want|like|need|want)\s+(.+)$",
        re.IGNORECASE,
    ), "i_prefer", 1),

    # "never X" / "always X" (standalone commands)
    (re.compile(
        r"^(?:always|never)\s+(.+)$",
        re.IGNORECASE,
    ), "always_never", 1),
]


def check_memory_trigger(text: str) -> Tier0RuleResult:
    """
    Check if the user message is a memory ingest or store request.

    Returns:
        MEMORY_INGEST for bulk data/file ingestion.
        MEMORY_STORE for remembering specific facts/preferences.
    """
    text_stripped = text.strip()
    if not text_stripped or len(text_stripped) < 4:
        return Tier0RuleResult(matched=False)

    cleaned = _strip_prefix(text_stripped)

    # 1) Check INGEST patterns first (bulk data)
    for candidate in [cleaned, text_stripped]:
        if not candidate:
            continue
        for pattern, name in _INGEST_PATTERNS:
            if pattern.match(candidate):
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.MEMORY_INGEST,
                    confidence=1.0,
                    rule_name=f"memory_ingest_{name}",
                    reason=f"Memory ingest detected: {name}",
                )

    # 2) Check STORE patterns (remember specific facts)
    for candidate in [cleaned, text_stripped]:
        if not candidate:
            continue
        for pattern, name, group_idx in _STORE_PATTERNS:
            m = pattern.match(candidate)
            if m:
                extracted = None
                if group_idx is not None:
                    try:
                        extracted = m.group(group_idx).strip()
                    except (IndexError, AttributeError):
                        pass
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.MEMORY_STORE,
                    confidence=0.95,
                    rule_name=f"memory_store_{name}",
                    reason=f"Memory store detected: {name}",
                    extracted_query=extracted,
                )

    return Tier0RuleResult(matched=False)
