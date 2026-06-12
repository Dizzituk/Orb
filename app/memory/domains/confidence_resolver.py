# FILE: app/memory/domains/confidence_resolver.py
# Purpose: Confidence response resolver (Spec Section 5.3).
# Called-by: app.memory.domains
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Confidence response resolver (Spec Section 5.3).

Takes a confidence match result and determines the appropriate
user-facing behaviour based on the 4-band threshold model:

    Band            Range       Behaviour
    ─────────────   ─────────   ──────────────────────────────────
    auto_execute    0.9–1.0     Execute immediately, no interruption
    notify          0.7–0.89    Execute with notification (can cancel)
    confirm         0.4–0.69    Ask for confirmation before executing
    clarify         0.0–0.39    Ask for clarification (ambiguous)

Risk-adjusted overrides (Section 5.4):
    Destructive intents ALWAYS require explicit confirmation
    regardless of confidence band — they never auto-execute.

Usage:
    from app.memory.domains.confidence_resolver import resolve_action

    match = confidence_store.match("send it to the pipeline")
    action = resolve_action(match)

    if action.band == "auto_execute":
        execute(action.intent)
    elif action.band == "notify":
        execute(action.intent)
        notify_user(action.message)
    elif action.band == "confirm":
        ask_confirmation(action.message, action.intent)
    elif action.band == "clarify":
        ask_clarification(action.message)
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# =========================================================================
# Band boundaries
# =========================================================================

BAND_AUTO_EXECUTE = 0.9
BAND_NOTIFY = 0.7
BAND_CONFIRM = 0.4
# Below BAND_CONFIRM = clarify

# Intents that NEVER auto-execute (locked to confirm band minimum)
ALWAYS_CONFIRM_INTENTS = {
    "DELETE_FILE", "PURGE_QUARANTINE", "GIT_PUSH",
    "WIPE_DOMAIN", "RESET_STATE",
    "DROP_TABLE", "CLEAR_DATABASE", "FORCE_PUSH",
}


# =========================================================================
# Action result
# =========================================================================

@dataclass
class ConfidenceAction:
    """
    The resolved action based on confidence scoring.

    This is what the UI/translation layer uses to decide how
    to present the intent to the user.
    """
    band: str               # "auto_execute", "notify", "confirm", "clarify"
    intent: Optional[str]   # The matched intent (None if clarify)
    confidence: float       # Effective confidence score
    message: str            # User-facing message
    can_execute: bool       # Whether execution is permitted
    needs_response: bool    # Whether we need user input before proceeding
    phrase_pattern: str     # The matched phrase pattern
    risk_override: bool     # True if risk rules forced a band change

    @property
    def is_destructive(self) -> bool:
        return self.intent and self.intent.upper() in ALWAYS_CONFIRM_INTENTS


# =========================================================================
# Resolver
# =========================================================================

def resolve_action(
    match: Optional[dict],
    original_phrase: Optional[str] = None,
) -> ConfidenceAction:
    """
    Resolve a confidence match into a user-facing action.

    Args:
        match: Result from ConfidenceStore.match(), or None if no match.
        original_phrase: The original user input (for clarify messages).

    Returns:
        ConfidenceAction with band, message, and execution flags.
    """
    # No match at all — clarify
    if match is None:
        return ConfidenceAction(
            band="clarify",
            intent=None,
            confidence=0.0,
            message=_clarify_message(original_phrase),
            can_execute=False,
            needs_response=True,
            phrase_pattern="",
            risk_override=False,
        )

    intent = match["intent"]
    confidence = match["confidence"]
    phrase = match.get("phrase_pattern", "")
    risk_override = False

    # Determine base band from confidence
    band = _confidence_to_band(confidence)

    # Risk override: destructive intents are ALWAYS confirm minimum
    if intent and intent.upper() in ALWAYS_CONFIRM_INTENTS:
        if band in ("auto_execute", "notify"):
            band = "confirm"
            risk_override = True

    # Build appropriate message
    message = _build_message(band, intent, confidence, phrase, risk_override)

    # Execution flags
    can_execute = band in ("auto_execute", "notify")
    needs_response = band in ("confirm", "clarify")

    action = ConfidenceAction(
        band=band,
        intent=intent,
        confidence=confidence,
        message=message,
        can_execute=can_execute,
        needs_response=needs_response,
        phrase_pattern=phrase,
        risk_override=risk_override,
    )

    logger.debug(
        "[resolver] %s → band=%s, conf=%.3f, exec=%s, risk=%s",
        phrase[:40], band, confidence, can_execute, risk_override,
    )

    return action


def resolve_no_match(original_phrase: str) -> ConfidenceAction:
    """Convenience: resolve when there's no match at all."""
    return resolve_action(None, original_phrase)


# =========================================================================
# Band classification
# =========================================================================

def _confidence_to_band(confidence: float) -> str:
    """Map a confidence score to its behaviour band."""
    if confidence >= BAND_AUTO_EXECUTE:
        return "auto_execute"
    if confidence >= BAND_NOTIFY:
        return "notify"
    if confidence >= BAND_CONFIRM:
        return "confirm"
    return "clarify"


def get_band_for_confidence(confidence: float) -> dict:
    """
    Get band info for a confidence score.

    Returns dict with band name, boundaries, and description.
    Useful for debugging and UI display.
    """
    band = _confidence_to_band(confidence)
    return {
        "band": band,
        "confidence": confidence,
        "boundaries": {
            "auto_execute": f">= {BAND_AUTO_EXECUTE}",
            "notify": f"{BAND_NOTIFY} – {BAND_AUTO_EXECUTE - 0.01}",
            "confirm": f"{BAND_CONFIRM} – {BAND_NOTIFY - 0.01}",
            "clarify": f"< {BAND_CONFIRM}",
        },
        "description": {
            "auto_execute": "Execute immediately, no interruption",
            "notify": "Execute with notification (can cancel)",
            "confirm": "Ask for confirmation before executing",
            "clarify": "Ask for clarification (ambiguous)",
        }[band],
    }


# =========================================================================
# Message builders
# =========================================================================

def _build_message(
    band: str,
    intent: str,
    confidence: float,
    phrase: str,
    risk_override: bool,
) -> str:
    """Build the user-facing message for each band."""
    intent_display = _intent_to_display(intent)

    if band == "auto_execute":
        return f"Running: {intent_display}"

    if band == "notify":
        return f"{intent_display}... (cancel if this isn't right)"

    if band == "confirm":
        if risk_override:
            return (
                f"This is a destructive action. "
                f"Did you mean: {intent_display}?"
            )
        return f"Did you mean: {intent_display}?"

    # clarify
    return _clarify_message(phrase)


def _clarify_message(phrase: Optional[str] = None) -> str:
    """Build a clarification request message."""
    if phrase:
        return (
            f"I'm not sure what you'd like me to do with "
            f"\"{phrase[:60]}\". Could you clarify?"
        )
    return "I'm not sure what you'd like me to do. Could you clarify?"


def _intent_to_display(intent: str) -> str:
    """Convert an intent enum to a readable action description."""
    display_map = {
        "RUN_PIPELINE": "run the pipeline",
        "REFACTOR": "start a refactor",
        "SCAN_CODEBASE": "scan the codebase",
        "CREATE_ARCHITECTURE_MAP": "create the architecture map",
        "BUILD_SPEC": "build a specification",
        "DELETE_FILE": "delete a file",
        "PURGE_QUARANTINE": "purge quarantined entries",
        "SHOW_STATUS": "show system status",
        "LIST_FILES": "list files",
        "SHOW_ARCHITECTURE": "show architecture info",
        "SHOW_LOGS": "show logs",
        "EXECUTE_SANDBOX": "execute in sandbox",
        "CHAT_ONLY": "chat",
        "SEND_TO_CONTACT": "send to a contact",
    }
    return display_map.get(intent, intent.lower().replace("_", " "))
