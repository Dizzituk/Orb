# FILE: app/self_model/fragments/classifier.py
"""
Lightweight signal-type and claim-type classification.

Design: pure rule-based (no LLM call on hot path). A small set of
phrase patterns routes the message into a SignalType + ClaimType.

When nothing matches, defaults to (PASSIVE, UNKNOWN).
"""
from __future__ import annotations

import re
from typing import Tuple

from app.self_model.fragments.models import SignalType, ClaimType

# Correction markers -- user telling Astra it's not following a directive
# Includes voice-to-text friendly patterns (STT from AstraBridge)
_CORRECTION_PATTERNS = [
    re.compile(r"\bi(?:'ve|\s+have)?\s+(?:told|asked|said)\s+(?:you|astra)\b", re.IGNORECASE),
    re.compile(r"\byou(?:'re|\s+are)\s+(?:not|still\s+not)\s+(?:doing|following|listening)\b", re.IGNORECASE),
    re.compile(r"\bstop\s+(?:doing|saying)\b", re.IGNORECASE),
    re.compile(r"\bi\s+keep\s+(?:telling|asking)\b", re.IGNORECASE),
    re.compile(r"\bhow\s+many\s+times\b", re.IGNORECASE),
    # Voice-to-text additions
    re.compile(r"\bthat(?:'s|\s+is)\s+(?:not\s+right|wrong|incorrect|not\s+what)\b", re.IGNORECASE),
    re.compile(r"\byou\s+(?:got|have)\s+(?:that|it|this)\s+wrong\b", re.IGNORECASE),
    re.compile(r"\byou\s+need\s+to\s+(?:update|change|fix|correct|remember)\b", re.IGNORECASE),
    re.compile(r"\b(?:don't|do\s+not)\s+forget\b", re.IGNORECASE),
    re.compile(r"\bmake\s+sure\s+you\s+(?:remember|don't\s+forget|update)\b", re.IGNORECASE),
    re.compile(r"\bno\s+(?:that's|that\s+is)\s+wrong\b", re.IGNORECASE),
    re.compile(r"\bi\s+(?:already|just)\s+(?:told|said|explained|mentioned)\b", re.IGNORECASE),
]

# Declaration markers -- explicit "I want / I like / I prefer / I'm into"
# Includes voice-to-text friendly patterns
_DECLARATION_PATTERNS = [
    re.compile(r"\bi\s+(?:really\s+|genuinely\s+|truly\s+)?(?:want|would\s+like|prefer|like|love|hate|dislike|enjoy)\b", re.IGNORECASE),
    re.compile(r"\bi(?:'m|\s+am)\s+(?:really\s+|very\s+|genuinely\s+|quite\s+)?(?:into|interested\s+in|passionate\s+about|obsessed\s+with|keen\s+on|a\s+fan\s+of)\b", re.IGNORECASE),
    re.compile(r"\bmy\s+(?:preference|style|approach)\s+is\b", re.IGNORECASE),
    re.compile(r"\b(?:please|always|never)\s+(?:do|avoid|use|be|include|exclude|write|say)\b", re.IGNORECASE),
    re.compile(r"\bi\s+(?:think|believe|feel|reckon)\b", re.IGNORECASE),
    # Voice-to-text additions
    re.compile(r"\bremember\s+(?:that|this)\b", re.IGNORECASE),
    re.compile(r"\bthe\s+(?:important|key|main)\s+(?:thing|point|bit)\s+is\b", re.IGNORECASE),
    re.compile(r"\bi\s+(?:need|require|demand)\b", re.IGNORECASE),
    re.compile(r"\bfor\s+me\s+(?:the|it's|what)\b", re.IGNORECASE),
]

# Behaviour markers -- claims about how Astra should act
_BEHAVIOUR_PATTERNS = [
    re.compile(r"\b(?:be|act|behave|respond|talk|write|reply)\s+(?:more|less)\b", re.IGNORECASE),
    re.compile(r"\bdon't\s+(?:be|act|use|say|write)\b", re.IGNORECASE),
    re.compile(r"\bi\s+want\s+you\s+to\b", re.IGNORECASE),
    re.compile(r"\byou\s+should\s+(?:always|never|just|only)\b", re.IGNORECASE),
    re.compile(r"\bpush\s+back\b", re.IGNORECASE),
    re.compile(r"\bchallenge\s+me\b", re.IGNORECASE),
    re.compile(r"\b(?:always|never)\s+use\b", re.IGNORECASE),
    # Voice-to-text additions
    re.compile(r"\bfrom\s+now\s+on\b", re.IGNORECASE),
    re.compile(r"\bgoing\s+forward\b", re.IGNORECASE),
    re.compile(r"\bi\s+(?:don't|do\s+not)\s+want\s+(?:you|astra)\s+to\b", re.IGNORECASE),
]

# Preference markers -- stated preferences about tools/formats/workflows
_PREFERENCE_PATTERNS = [
    re.compile(r"\bmy\s+preferred\b", re.IGNORECASE),
    re.compile(r"\bi\s+prefer\s+(?:using|to\s+use)\b", re.IGNORECASE),
    re.compile(r"\bi\s+always\s+use\b", re.IGNORECASE),
    re.compile(r"\bi\s+don't\s+use\b", re.IGNORECASE),
    # Commitment language -- active plans vs idle wishes
    re.compile(r"\bi(?:'m|\s+am)\s+(?:working\s+(?:on|toward)|building|pursuing|planning)\b", re.IGNORECASE),
    re.compile(r"\bi(?:'ve|\s+have)\s+(?:decided|committed|started|begun|applied)\b", re.IGNORECASE),
    re.compile(r"\bthe\s+plan\s+is\b", re.IGNORECASE),
]

def classify(message: str) -> Tuple[SignalType, ClaimType]:
    """
    Return (signal_type, claim_type) for a user message.
    """
    if not message or len(message.strip()) < 3:
        return SignalType.PASSIVE, ClaimType.UNKNOWN

    msg = message.strip()

    # Signal detection (correction > declaration > passive)
    signal = SignalType.PASSIVE
    for p in _CORRECTION_PATTERNS:
        if p.search(msg):
            signal = SignalType.CORRECTION
            break
    if signal == SignalType.PASSIVE:
        for p in _DECLARATION_PATTERNS:
            if p.search(msg):
                signal = SignalType.DECLARATION
                break

    # Claim type detection
    claim = ClaimType.UNKNOWN
    for p in _BEHAVIOUR_PATTERNS:
        if p.search(msg):
            claim = ClaimType.BEHAVIOUR
            break
    if claim == ClaimType.UNKNOWN:
        for p in _PREFERENCE_PATTERNS:
            if p.search(msg):
                claim = ClaimType.PREFERENCE
                break
    if claim == ClaimType.UNKNOWN and signal == SignalType.DECLARATION:
        # Declarations with "I'm into X", "I love Y" are interests by default
        claim = ClaimType.INTEREST

    return signal, claim