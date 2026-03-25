# FILE: app/memory/realtime_fact_capture.py
"""
Real-time biographical fact capture from conversation.

Detects when the user makes explicit personal statements or corrections
in chat (e.g. "I live in Redruth now", "my email is X", "I work at Y")
and immediately writes them to ASTRA's permanent preference system.

This is the FAST path — no LLM call, pure pattern matching. It catches
the most common biographical corrections in real-time. The slower
conversation knowledge extractor (knowledge_extractor.py) picks up
subtler facts when sessions are archived.

Called from: app/memory/integration.py → after_user_message()
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Biographical patterns ───────────────────────────────────────
# Each tuple: (compiled regex, preference_key, permanence)
# Group 1 in the regex is the captured value.

_PATTERNS = [
    # Location
    (re.compile(
        r"(?:i\s+live\s+in|i(?:'m|\s+am)\s+(?:based|living|located)\s+in|"
        r"i\s+moved\s+to|i(?:'ve|\s+have)\s+moved\s+to|"
        r"my\s+(?:home|address|location)\s+is)\s+(.{3,60})",
        re.IGNORECASE,
    ), "biographical:current_location", "current_state"),

    # Full name (legal name)
    (re.compile(
        r"(?:my\s+name\s+is|my\s+full\s+name\s+is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        re.IGNORECASE,
    ), "biographical:full_name", "permanent"),

    # Preferred name / nickname
    (re.compile(
        r"(?:call\s+me|you\s+can\s+call\s+me|i\s+(?:go|prefer)\s+(?:by|to\s+be\s+called)|i(?:'m|\s+am)\s+called)\s+(\w+(?:\s+\w+)?)",
        re.IGNORECASE,
    ), "biographical:preferred_name", "current_state"),

    # Job / occupation
    (re.compile(
        r"(?:i\s+work\s+(?:at|for|as)|my\s+job\s+is|i(?:'m|\s+am)\s+a\s+)\s*(.{3,80})",
        re.IGNORECASE,
    ), "biographical:current_occupation", "current_state"),

    # Age / DOB
    (re.compile(
        r"i(?:'m|\s+am)\s+(\d{2,3})\s+(?:years?\s+old|yrs?\s+old)",
        re.IGNORECASE,
    ), "biographical:age", "current_state"),

    # Email
    (re.compile(
        r"my\s+email\s+(?:is|address\s+is)\s+([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)",
        re.IGNORECASE,
    ), "biographical:email", "permanent"),

    # Phone
    (re.compile(
        r"my\s+(?:phone|number|mobile)\s+(?:is|number\s+is)\s+([\d\s\+\-()]{8,20})",
        re.IGNORECASE,
    ), "biographical:phone_number", "current_state"),
]

# Map permanence strings to PreferenceStrength values
_STRENGTH_MAP = {
    "permanent": "hard_rule",
    "current_state": "default",
    "soft": "soft",
}


def capture_biographical_facts(
    message: str,
    db=None,
) -> int:
    """
    Scan a user message for biographical facts and write them
    to ASTRA's permanent preference system immediately.

    Returns the number of facts captured.
    """
    if not message or len(message) < 5:
        return 0

    if db is None:
        return 0

    captured = 0
    msg = message.strip()

    for pattern, key_suffix, permanence in _PATTERNS:
        match = pattern.search(msg)
        if not match:
            continue

        value = match.group(1).strip().rstrip(".,!?;:")
        if not value or len(value) < 2:
            continue

        pref_key = f"doc_extract:{key_suffix}"

        try:
            from app.astra_memory.preference_service import create_preference
            from app.astra_memory.preference_models import (
                PreferenceRecord,
                PreferenceStrength,
                RecordStatus,
                SignalType,
            )
            from app.astra_memory.confidence_scoring import (
                append_preference_evidence,
            )

            strength = PreferenceStrength(_STRENGTH_MAP.get(permanence, "default"))

            # Check if this key already exists
            existing = (
                db.query(PreferenceRecord)
                .filter(PreferenceRecord.preference_key == pref_key)
                .first()
            )

            if existing:
                if existing.preference_value == value:
                    # Same value — reinforce
                    append_preference_evidence(
                        db=db,
                        preference_key=pref_key,
                        signal_type=SignalType.EXPLICIT,
                        context_pointer="conversation_correction",
                        details={"action": "reinforce_from_chat", "message": msg[:100]},
                    )
                    logger.info("[realtime_capture] Reinforced: %s", pref_key)
                else:
                    # Different value — user is correcting a fact
                    old_value = existing.preference_value
                    existing.status = RecordStatus.SUPERSEDED
                    db.commit()

                    create_preference(
                        db=db,
                        preference_key=pref_key,
                        preference_value=value,
                        strength=strength,
                        source="conversation_correction",
                        namespace="user_personal",
                        context_pointer="conversation_correction",
                    )
                    logger.info(
                        "[realtime_capture] CORRECTED: %s — '%s' → '%s'",
                        pref_key, str(old_value)[:40], value[:40],
                    )
            else:
                # New fact from conversation
                create_preference(
                    db=db,
                    preference_key=pref_key,
                    preference_value=value,
                    strength=strength,
                    source="conversation_capture",
                    namespace="user_personal",
                    context_pointer="conversation_capture",
                )
                logger.info("[realtime_capture] NEW: %s = %s", pref_key, value[:60])

            captured += 1

        except Exception as e:
            logger.warning("[realtime_capture] Failed to write %s: %s", pref_key, e)

    return captured
