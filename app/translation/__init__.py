# FILE: app/translation/__init__.py
"""
ASTRA Translation Layer

Converts "Tazish" (natural language) into canonical intents with safety gates.
This layer governs all chat-driven interaction with ASTRA, Sandbox Zombie Self,
and the Critical Pipeline.

v1.1 (2026-01): Added Spec Gate flow intents (WEAVER_BUILD_SPEC, SEND_TO_SPEC_GATE)

Usage:
    from app.translation import translate_message, TranslationResult
    
    result = await translate_message("Create architecture map")
    if result.should_execute:
        # Execute the command
        execute_intent(result.resolved_intent, result.extracted_context)
    else:
        # Chat response
        handle_chat(result.original_text)

Spec Gate Flow:
    1. User rambles (captured by conversation)
    2. User says "How does that look all together?" -> WEAVER_BUILD_SPEC
    3. Weaver builds candidate spec, reads back to user
    4. User says "Send to Spec Gate" -> SEND_TO_SPEC_GATE
    5. Spec Gate validates, returns approved or questions
    6. User says "Run critical pipeline" -> RUN_CRITICAL_PIPELINE_FOR_JOB (requires confirmation)

Key Invariants:
- Chat mode MUST NOT trigger any backend action
- Intent classification NEVER requires a frontier model
- High-stakes operations ALWAYS require explicit confirmation
- Default to safety (if uncertain -> chat)
- Questions and meta-discussion NEVER trigger expensive pipelines
"""
from __future__ import annotations

# Schemas
from .schemas import (
    TranslationMode,
    CanonicalIntent,
    LatencyTier,
    TranslationResult,
    FeedbackEvent,
    GateResult,
    DirectiveGateResult,
    ContextGateResult,
    ConfirmationGateResult,
    PhraseCacheEntry,
    IntentDefinition,
)

# Mode classification
from .modes import (
    classify_mode,
    classify_mode_with_ui,
    UIContext,
)

# Intents
from .intents import (
    INTENT_DEFINITIONS,
    get_intent_definition,
    get_all_command_intents,
    get_high_stakes_intents,
    get_spec_gate_flow_intents,
)

# Gates
from .gates import (
    check_directive_gate,
    check_context_gate,
    check_confirmation_gate,
    is_obvious_chat,
    ConfirmationState,
)

# Tier 0 rules
from .tier0_rules import (
    tier0_classify,
    is_user_chat_pattern,
    is_tazish_chat,  # Legacy alias
    Tier0RuleResult,
    check_weaver_trigger,
    check_spec_gate_trigger,
    check_critical_pipeline_trigger,
)

# Tier 1 classifier
from .tier1_classifier import (
    Tier1Classifier,
    MockTier1Classifier,
    CONFIDENCE_THRESHOLD,
)

# Phrase cache
from .phrase_cache import (
    PhraseCache,
    get_phrase_cache,
    normalize_phrase,
    extract_pattern,
)

# Feedback
from .feedback import (
    FeedbackLogger,
    get_feedback_logger,
    parse_feedback_message,
    RulePromoter,
)

# Main translator
from .translator import (
    Translator,
    get_translator,
    translate_message,
    translate_message_sync,
)


__all__ = [
    # Schemas
    "TranslationMode",
    "CanonicalIntent",
    "LatencyTier",
    "TranslationResult",
    "FeedbackEvent",
    "GateResult",
    "DirectiveGateResult",
    "ContextGateResult",
    "ConfirmationGateResult",
    "PhraseCacheEntry",
    "IntentDefinition",
    
    # Mode classification
    "classify_mode",
    "classify_mode_with_ui",
    "UIContext",
    
    # Intents
    "INTENT_DEFINITIONS",
    "get_intent_definition",
    "get_all_command_intents",
    "get_high_stakes_intents",
    "get_spec_gate_flow_intents",
    
    # Gates
    "check_directive_gate",
    "check_context_gate",
    "check_confirmation_gate",
    "is_obvious_chat",
    "ConfirmationState",
    
    # Tier 0 rules
    "tier0_classify",
    "is_user_chat_pattern",
    "is_tazish_chat",  # Legacy alias
    "Tier0RuleResult",
    "check_weaver_trigger",
    "check_spec_gate_trigger",
    "check_critical_pipeline_trigger",
    
    # Tier 1 classifier
    "Tier1Classifier",
    "MockTier1Classifier",
    "CONFIDENCE_THRESHOLD",
    
    # Phrase cache
    "PhraseCache",
    "get_phrase_cache",
    "normalize_phrase",
    "extract_pattern",
    
    # Feedback
    "FeedbackLogger",
    "get_feedback_logger",
    "parse_feedback_message",
    "RulePromoter",
    
    # Main translator
    "Translator",
    "get_translator",
    "translate_message",
    "translate_message_sync",
]
