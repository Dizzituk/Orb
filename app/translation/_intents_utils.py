from .schemas import CanonicalIntent, IntentDefinition
from typing import List, Optional


def get_intent_definition(intent: CanonicalIntent) -> IntentDefinition:
    """Get the definition for a canonical intent."""
    from .intents import INTENT_DEFINITIONS
    return INTENT_DEFINITIONS[intent]

def get_all_command_intents() -> List[CanonicalIntent]:
    """Get all intents that are actual commands (not chat or feedback)."""
    return [
        intent for intent in CanonicalIntent 
        if intent not in (CanonicalIntent.CHAT_ONLY, CanonicalIntent.USER_BEHAVIOR_FEEDBACK)
    ]

def get_high_stakes_intents() -> List[CanonicalIntent]:
    """Get all intents that require confirmation."""
    from .intents import INTENT_DEFINITIONS
    return [
        intent for intent, defn in INTENT_DEFINITIONS.items()
        if defn.requires_confirmation
    ]

def get_spec_gate_flow_intents() -> List[CanonicalIntent]:
    """Get intents related to the Spec Gate flow."""
    return [
        CanonicalIntent.WEAVER_BUILD_SPEC,
        CanonicalIntent.SEND_TO_SPEC_GATE,
        CanonicalIntent.RUN_PIPELINE,  # v5.4: unified
        CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB,  # v5.4: deprecated alias
    ]

def get_overwatcher_intents() -> List[CanonicalIntent]:
    """Get intents related to Overwatcher."""
    return [
        CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES,
    ]

def get_intent_by_trigger_phrase(phrase: str) -> Optional[CanonicalIntent]:
    """
    Exact match lookup for trigger phrases.
    Returns None if no exact match found.
    """
    from .intents import INTENT_DEFINITIONS
    for intent, defn in INTENT_DEFINITIONS.items():
        if phrase in defn.trigger_phrases:
            return intent
    return None
