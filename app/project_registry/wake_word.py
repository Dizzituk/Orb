# FILE: app/project_registry/wake_word.py
# Purpose: Wake word detection and intent classification for natural language commands.
# Called-by: app.translation.translator
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Wake word detection and intent classification for natural language commands.

Detects when the user addresses ASTRA by name and classifies whether
the utterance is a command, a question, or just conversational.

The classification feeds into the confidence gating system — every
detected command intent gets a confidence score, and user yes/no
feedback adjusts future scoring.

v1.0 (2026-03): Initial implementation.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ── Wake word patterns ──────────────────────────────────────────────────

# Match "Astra" at the start of a message, optionally followed by comma/colon
_WAKE_WORD_PATTERN = re.compile(
    r"^\s*[Aa][Ss][Tt][Rr][Aa]\s*[,:\s]\s*",
)

# Match "Astra" anywhere in the message (for detecting references)
_WAKE_WORD_ANYWHERE = re.compile(
    r"\b[Aa][Ss][Tt][Rr][Aa]\b",
)


class UtteranceType(str, Enum):
    """Classification of what the user is doing when they say ASTRA's name."""
    COMMAND = "command"          # Direct instruction: "Astra, scan the repo"
    QUESTION = "question"       # Asking for info: "Astra, how's the pipeline?"
    SUGGESTION = "suggestion"   # Soft intent: "Astra, we should probably..."
    REFERENCE = "reference"     # Talking about ASTRA: "I think Astra should..."
    NONE = "none"               # No wake word detected


@dataclass
class WakeWordResult:
    """Result of wake word detection and intent classification."""
    type: UtteranceType
    addressed: bool         # Was ASTRA addressed directly (name at start)?
    mentioned: bool         # Was ASTRA mentioned at all?
    confidence: float       # 0.0-1.0 confidence this is a command
    payload: str            # Message text after the wake word (if addressed)
    raw_message: str        # Original message


# ── Imperative verb detection ───────────────────────────────────────────

# Verbs that strongly indicate a direct command
_IMPERATIVE_VERBS = {
    "scan", "build", "run", "create", "delete", "remove", "update",
    "refresh", "generate", "deploy", "start", "stop", "restart",
    "install", "configure", "setup", "fix", "repair", "clear",
    "reset", "sync", "push", "pull", "fetch", "check", "verify",
    "test", "analyse", "analyze", "optimise", "optimize", "refactor",
    "migrate", "backup", "restore", "index", "embed", "ingest",
    "register", "deregister", "list", "show",
}

# Phrases that indicate a question, not a command
_QUESTION_STARTERS = {
    "what", "how", "why", "when", "where", "who", "which",
    "is", "are", "was", "were", "do", "does", "did",
    "can", "could", "would", "should", "will",
    "have", "has", "had",
}

# Phrases that indicate a soft suggestion
_SUGGESTION_MARKERS = {
    "we should", "we could", "we might", "maybe we",
    "perhaps we", "it might be", "it would be",
    "i think we", "i reckon we", "how about we",
    "shall we", "want to", "let's think about",
    "at some point", "eventually", "when we get a chance",
    "probably should", "might want to",
}


# ── Classification ──────────────────────────────────────────────────────

def classify(message: str) -> WakeWordResult:
    """
    Detect wake word and classify the utterance type.

    Returns a WakeWordResult with the classification and confidence.
    """
    raw = message.strip()
    if not raw:
        return WakeWordResult(
            type=UtteranceType.NONE, addressed=False, mentioned=False,
            confidence=0.0, payload="", raw_message=raw,
        )

    mentioned = bool(_WAKE_WORD_ANYWHERE.search(raw))
    wake_match = _WAKE_WORD_PATTERN.match(raw)
    addressed = bool(wake_match)

    if not mentioned:
        return WakeWordResult(
            type=UtteranceType.NONE, addressed=False, mentioned=False,
            confidence=0.0, payload=raw, raw_message=raw,
        )

    # Extract payload (text after wake word)
    if addressed and wake_match:
        payload = raw[wake_match.end():].strip()
    else:
        payload = raw

    if not payload:
        return WakeWordResult(
            type=UtteranceType.REFERENCE, addressed=addressed, mentioned=True,
            confidence=0.0, payload="", raw_message=raw,
        )

    payload_lower = payload.lower()

    # Check for suggestion markers first (before command detection)
    for marker in _SUGGESTION_MARKERS:
        if marker in payload_lower:
            return WakeWordResult(
                type=UtteranceType.SUGGESTION,
                addressed=addressed, mentioned=True,
                confidence=0.2,
                payload=payload, raw_message=raw,
            )

    # Check if it starts with a question word
    first_word = payload_lower.split()[0] if payload_lower.split() else ""
    if first_word in _QUESTION_STARTERS:
        return WakeWordResult(
            type=UtteranceType.QUESTION,
            addressed=addressed, mentioned=True,
            confidence=0.1,
            payload=payload, raw_message=raw,
        )

    # Check for imperative verbs (command indicators)
    if first_word in _IMPERATIVE_VERBS:
        confidence = 0.85 if addressed else 0.4
        return WakeWordResult(
            type=UtteranceType.COMMAND,
            addressed=addressed, mentioned=True,
            confidence=confidence,
            payload=payload, raw_message=raw,
        )

    # Addressed but no clear command pattern — moderate confidence
    if addressed:
        return WakeWordResult(
            type=UtteranceType.COMMAND,
            addressed=addressed, mentioned=True,
            confidence=0.5,
            payload=payload, raw_message=raw,
        )

    # Mentioned but not addressed — likely a reference
    return WakeWordResult(
        type=UtteranceType.REFERENCE,
        addressed=False, mentioned=True,
        confidence=0.1,
        payload=payload, raw_message=raw,
    )