# FILE: app/memory/domains/preference_capture.py
"""
Preference capture channels.

Three mechanisms for learning user preferences:

1. Explicit — User directly states a preference ("I prefer X", "always do Y")
2. Correction — User corrects ASTRA output ("no, use tabs not spaces")
3. Inferred — Pattern detected over multiple interactions

These functions are HOOKS — they process detection events and delegate
to app/astra_memory/preference_service.py for storage. They do not
modify the Weaver or translation layer directly. Wiring into those
systems is a separate integration step.

Usage:
    from app.memory.domains.preference_capture import (
        capture_explicit,
        capture_correction,
        capture_inferred,
    )

    # When user says "I always want 4-space indentation"
    capture_explicit("development", "indent_spaces", 4, context="msg:123")

    # When user corrects output format
    capture_correction("content", "date_format", "DD/MM/YYYY", context="msg:456")

    # When pattern analysis detects repeated behavior
    capture_inferred("development", "prefer_snake_case", True, count=5)
"""

import logging
from typing import Any, Optional

from app.db import get_db_session
from app.astra_memory.preference_models import (
    PreferenceRecord,
    PreferenceStrength,
)
from app.astra_memory.preference_service import (
    create_preference,
    update_preference_value,
    learn_from_behavior,
    get_preference,
)
from app.memory.domains.preference_registry import is_valid_domain

logger = logging.getLogger(__name__)


# =========================================================================
# Channel 1: Explicit capture
# =========================================================================

def capture_explicit(
    domain: str,
    key: str,
    value: Any,
    category: Optional[str] = None,
    context: Optional[str] = None,
    is_hard_rule: bool = False,
) -> Optional[int]:
    """
    Capture an explicitly stated preference.

    Called when the user directly states a preference:
      "I prefer X"
      "Always use Y"
      "Never do Z"

    High confidence, immediate store.

    Args:
        domain: Preference namespace (development, content, etc.)
        key: Preference key.
        value: Preference value.
        category: Component scope (maps to applies_to).
        context: Context pointer for audit trail (e.g. "msg:123").
        is_hard_rule: True if user used absolute language ("always", "never").

    Returns:
        Preference record ID, or None on failure.
    """
    if not is_valid_domain(domain):
        logger.warning(
            "[capture_explicit] Unknown domain '%s', using 'general'",
            domain,
        )
        domain = "general"

    strength = (
        PreferenceStrength.HARD_RULE if is_hard_rule
        else PreferenceStrength.DEFAULT
    )

    db = get_db_session()
    try:
        pref = create_preference(
            db=db,
            preference_key=key,
            preference_value=value,
            strength=strength,
            source="user_declared",
            applies_to=category,
            namespace=domain,
            context_pointer=context,
        )
        logger.info(
            "[capture_explicit] Stored: %s.%s = %s (strength=%s)",
            domain, key, value, strength.value,
        )
        return pref.id
    except Exception as e:
        logger.error("[capture_explicit] Failed: %s", e)
        return None
    finally:
        db.close()


# =========================================================================
# Channel 2: Correction capture
# =========================================================================

def capture_correction(
    domain: str,
    key: str,
    corrected_value: Any,
    category: Optional[str] = None,
    context: Optional[str] = None,
) -> Optional[int]:
    """
    Capture a preference inferred from a correction.

    Called when the user corrects ASTRA's output, implying a preference:
      "No, use tabs not spaces"
      "That should be DD/MM/YYYY"
      "I wanted British English"

    Medium confidence. If the preference already exists with a different
    value, this is treated as a contradiction (handled by the confidence
    system). If new, creates with DEFAULT strength.

    Args:
        domain: Preference namespace.
        key: Preference key.
        corrected_value: The value the user corrected to.
        category: Component scope.
        context: Context pointer for audit trail.

    Returns:
        Preference record ID, or None on failure.
    """
    if not is_valid_domain(domain):
        domain = "general"

    db = get_db_session()
    try:
        existing = get_preference(db, key)

        if existing:
            # Update with explicit flag (correction = user statement)
            pref = update_preference_value(
                db=db,
                preference_key=key,
                new_value=corrected_value,
                is_explicit=True,
                context_pointer=context,
            )
            if pref:
                logger.info(
                    "[capture_correction] Updated: %s.%s → %s",
                    domain, key, corrected_value,
                )
                return pref.id
        else:
            # Create new from correction
            pref = create_preference(
                db=db,
                preference_key=key,
                preference_value=corrected_value,
                strength=PreferenceStrength.DEFAULT,
                source="correction",
                applies_to=category,
                namespace=domain,
                context_pointer=context,
            )
            logger.info(
                "[capture_correction] Created: %s.%s = %s",
                domain, key, corrected_value,
            )
            return pref.id

        return None
    except Exception as e:
        logger.error("[capture_correction] Failed: %s", e)
        return None
    finally:
        db.close()


# =========================================================================
# Channel 3: Inferred capture
# =========================================================================

def capture_inferred(
    domain: str,
    key: str,
    observed_value: Any,
    count: int = 1,
    category: Optional[str] = None,
    context: Optional[str] = None,
) -> Optional[int]:
    """
    Capture a preference inferred from repeated behavior.

    Called when pattern analysis detects that the user consistently
    makes the same choice. Low confidence — the preference service
    applies bad-learning prevention (requires evidence_count >= 2
    or one explicit instruction).

    Args:
        domain: Preference namespace.
        key: Preference key.
        observed_value: The value observed in user behavior.
        count: How many times this pattern has been observed.
        category: Component scope.
        context: Context pointer for audit trail.

    Returns:
        Preference record ID, or None on failure.
    """
    if not is_valid_domain(domain):
        domain = "general"

    is_repeated = count >= 2

    db = get_db_session()
    try:
        pref = learn_from_behavior(
            db=db,
            preference_key=key,
            observed_value=observed_value,
            context_pointer=context,
            is_repeated=is_repeated,
        )

        if pref:
            # Ensure namespace is set correctly
            if pref.namespace != domain:
                pref.namespace = domain
                if category and not pref.applies_to:
                    pref.applies_to = category
                db.commit()

            logger.info(
                "[capture_inferred] %s: %s.%s = %s (count=%d, repeated=%s)",
                "Reinforced" if pref.evidence_count > 1 else "Observed",
                domain, key, observed_value, count, is_repeated,
            )
            return pref.id

        return None
    except Exception as e:
        logger.error("[capture_inferred] Failed: %s", e)
        return None
    finally:
        db.close()


# =========================================================================
# Utility: Detect preference language
# =========================================================================

# Patterns that indicate explicit preference statements.
# Used by downstream detection logic (e.g. Weaver hooks) to
# identify when a user message contains a preference declaration.

EXPLICIT_TRIGGERS = (
    "i prefer",
    "i always want",
    "always use",
    "never use",
    "i like",
    "i don't like",
    "i dont like",
    "make sure to",
    "make sure you",
    "from now on",
    "going forward",
    "remember that i",
    "remember i",
)

HARD_RULE_TRIGGERS = (
    "always",
    "never",
    "must",
    "don't ever",
    "dont ever",
    "absolutely",
    "non-negotiable",
    "no exceptions",
)


def detect_preference_intent(message: str) -> dict:
    """
    Detect if a user message contains preference-setting language.

    Returns a dict with:
        has_preference: bool
        is_hard_rule: bool
        triggers_found: list of matched trigger phrases

    This is a lightweight detection helper. Full NLU parsing of
    the preference key/value is left to the caller (or an LLM pass).
    """
    lower = message.lower()
    found_explicit = [t for t in EXPLICIT_TRIGGERS if t in lower]
    found_hard = [t for t in HARD_RULE_TRIGGERS if t in lower]

    return {
        "has_preference": len(found_explicit) > 0,
        "is_hard_rule": len(found_hard) > 0,
        "triggers_found": found_explicit + found_hard,
    }
