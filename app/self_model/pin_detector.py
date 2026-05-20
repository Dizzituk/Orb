# FILE: app/self_model/pin_detector.py
"""
Detect user commands that mean "pin this to memory".

Matches phrasings like:
  - "remember that X"
  - "save this to memory: X"
  - "don't forget X"
  - "make a note that X"
  - "keep in mind X"
  - "store this in memory: X"

Rejects look-alikes:
  - "remember when we..." (reminiscence, not pin)
  - "do you remember ..." (question to assistant)
  - "forget X" preceded by "don't" (that's a pin, not forget)
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Phrases that invalidate a "remember" match because they are not commands
# to store something — they are questions or reminiscences.
_REMEMBER_REJECTIONS = [
    re.compile(r"\bdo\s+you\s+remember\b", re.IGNORECASE),
    re.compile(r"\bcan\s+you\s+remember\b", re.IGNORECASE),
    re.compile(r"\bremember\s+(?:when|who|what|where|why|how|the\s+time|that\s+time|back\s+when)\b", re.IGNORECASE),
    re.compile(r"\bi\s+remember\b", re.IGNORECASE),
]

# Pin patterns: group 1 = content to store.
_PIN_PATTERNS = [
    re.compile(
        r"(?:^|[.!?\s])(?:astra[,\s]+)?(?:please\s+)?remember\s+(?:that\s+|this\s*[:\-]\s*)?(.{6,400})",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?:please\s+)?save\s+(?:this\s+)?(?:to\s+)?(?:your\s+)?(?:memory|memories)\s*[:\-]?\s*(.{6,400})",
        re.IGNORECASE | re.DOTALL,
    ),
    # "don't forget X" → PIN, not forget. Must come before the forget pattern.
    re.compile(
        r"(?:please\s+)?(?:do\s+not|don(?:'t|['\s]?t))\s+forget\s+(?:that\s+|this\s*[:\-]?\s*)?(.{6,400})",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?:please\s+)?make\s+a\s+(?:mental\s+)?note\s+(?:that\s+|of\s+|this\s*[:\-]?\s*)?(.{6,400})",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?:please\s+)?keep\s+in\s+mind\s+(?:that\s+|this\s*[:\-]?\s*)?(.{6,400})",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?:please\s+)?store\s+(?:this|it|that)\s+(?:in\s+(?:your\s+)?(?:memory|memories))?\s*[:\-]?\s*(.{6,400})",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?:please\s+)?note\s+(?:to\s+self\s+|down\s+)?(?:that\s+|this\s*[:\-]?\s*)(.{6,400})",
        re.IGNORECASE | re.DOTALL,
    ),
]


# Forget patterns — must NOT fire on "don't forget" (handled above).
# Require "forget" not be preceded by "don't", "do not", "never", "can't".
_FORGET_PATTERNS = [
    re.compile(
        r"(?<!don['’]t\s)(?<!do\s+not\s)(?<!never\s)(?:^|[.!?\s])(?:please\s+)?(?:astra[,\s]+)?forget\s+(?:about\s+|that\s+)?(.{3,200})",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?:please\s+)?(?:remove|delete)\s+(?:from\s+)?(?:memory|memories|your\s+memory)(?:\s+that)?\s*[:\-]?\s*(.{3,200})",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?:^|[.!?\s])i\s+no\s+longer\s+(.{3,200})",
        re.IGNORECASE | re.DOTALL,
    ),
]


def _strip_trailing_noise(s: str) -> str:
    s = s.strip().rstrip(".!?;:,")
    s = re.sub(r"[\s,]+(?:please|ok|okay|thanks|thank\s+you|cheers)\s*\.?\s*$", "", s, flags=re.IGNORECASE)
    return s.strip()


def _is_reminiscence_or_question(msg: str) -> bool:
    """True if the 'remember' usage is a question or reminiscence, not a pin command."""
    for p in _REMEMBER_REJECTIONS:
        if p.search(msg):
            return True
    return False


def detect_pin_command(message: str) -> Optional[Dict[str, Any]]:
    if not message or len(message.strip()) < 8:
        return None

    msg = message.strip()

    # Pin patterns first — we want "don't forget" to be detected as pin,
    # not as forget.
    for i, p in enumerate(_PIN_PATTERNS):
        m = p.search(msg)
        if not m:
            continue
        # Pattern 0 is the "remember" pattern — reject reminiscence/questions
        if i == 0 and _is_reminiscence_or_question(msg):
            continue
        content = _strip_trailing_noise(m.group(1))
        if content and len(content) >= 6:
            return {"action": "pin", "content": content, "matched_on": p.pattern[:40]}

    # Forget patterns
    for p in _FORGET_PATTERNS:
        m = p.search(msg)
        if m:
            content = _strip_trailing_noise(m.group(1))
            if content and len(content) >= 3:
                return {"action": "forget", "content": content, "matched_on": p.pattern[:40]}

    return None