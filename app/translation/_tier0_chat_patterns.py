# FILE: app/translation/_tier0_chat_patterns.py
# Purpose: Tier 0 user-chat pattern library + matcher (chat short-circuiting).
# Called-by: app.translation._tier0_filesystem (shim)
# Depends-on: (stdlib re only)
# Last-renovated: 2026-06-21
"""
Tier 0 user-chat pattern library.

Split out of _tier0_filesystem.py (BATCH 4) verbatim. Seed patterns for casual
user chat, used for additional chat short-circuiting. Legacy TAZISH_* aliases preserved.
"""
from __future__ import annotations
import re


USER_CHAT_PATTERNS = [
    # Exploratory questions about the system
    r"tell me (?:about|more about)",
    r"what (?:is|are) your",
    r"describe your",
    r"explain (?:your|the|how)",
    r"show me (?:your|the|how)",
    r"how does (?:your|the)",
    r"what does (?:your|the|this|that|it)",
    
    # Casual conversation
    r"^(?:hi|hello|hey|yo|sup)",
    r"^(?:thanks|thank you|cheers)",
    r"^(?:ok|okay|sure|got it|understood)",
    r"^(?:hmm|huh|interesting|cool|nice)",
    
    # Questions about capabilities
    r"can you (?:tell|show|explain)",
    r"do you (?:have|know|understand)",
    r"are you (?:able|capable)",
]


_COMPILED_USER_CHAT = [re.compile(p, re.IGNORECASE) for p in USER_CHAT_PATTERNS]


def is_user_chat_pattern(text: str) -> bool:
    """
    Check if text matches known user chat patterns.
    Used for additional chat short-circuiting.
    """
    for pattern in _COMPILED_USER_CHAT:
        if pattern.search(text):
            return True
    return False


TAZISH_CHAT_PATTERNS = USER_CHAT_PATTERNS


_COMPILED_TAZISH_CHAT = _COMPILED_USER_CHAT


is_tazish_chat = is_user_chat_pattern
