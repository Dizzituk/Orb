# FILE: app/translation/translator.py
"""
Main Translation Layer coordinator for ASTRA.
Converts "Tazish" (natural language) into canonical intents with safety gates.

Pipeline:
1. Mode Classification (Chat/Command-Capable/Feedback)
2. Tier 0 Rules (regex/string - no LLM)
3. Phrase Cache Lookup
4. Tier 1 Classifier (GPT-5 mini - only if needed)
5. Directive vs Story Gate
6. Context Gate
7. Confirmation Gate (for high-stakes)

INVARIANT: Intent classification NEVER requires a frontier model.
"""
from __future__ import annotations
import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path

from .schemas import (
    TranslationMode,
    CanonicalIntent,
    LatencyTier,
    TranslationResult,
    GateResult,
)
from .modes import classify_mode_with_ui, UIContext
from .intents import get_intent_definition, get_all_command_intents
from .gates import (
    check_directive_gate,
    check_context_gate,
    check_confirmation_gate,
    extract_context_from_text,
    is_obvious_chat,
    ConfirmationState,
)
from .tier0_rules import tier0_classify, is_user_chat_pattern
from .tier1_classifier import Tier1Classifier, CONFIDENCE_THRESHOLD
from .phrase_cache import get_phrase_cache, PhraseCache
from .feedback import get_feedback_logger, FeedbackLogger, parse_feedback_message

logger = logging.getLogger(__name__)


# =============================================================================
# TRANSLATOR
# =============================================================================

class Translator:
    """
    Main translation layer for ASTRA.
    Converts user messages to canonical intents with safety gates.
    """
    
    def __init__(
        self,
        user_id: str,
        cache_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        llm_client=None,
    ):
        """
        Initialize translator for a user.
        
        Args:
            user_id: User identifier
            cache_dir: Directory for phrase cache persistence
            log_dir: Directory for feedback logs
            llm_client: Optional LLM client for Tier 1 classification
        """
        self.user_id = user_id
        self._phrase_cache = get_phrase_cache(user_id, cache_dir)
        self._feedback_logger = get_feedback_logger(user_id, log_dir)
        self._tier1_classifier = Tier1Classifier(llm_client)
        self._confirmation_state = ConfirmationState()
    
    async def translate(
        self,
        text: str,
        ui_context: Optional[UIContext] = None,
        conversation_id: Optional[str] = None,
    ) -> TranslationResult:
        """
        Translate a user message to a canonical intent.
        
        Args:
            text: The user's message
            ui_context: Current UI context (job config, sandbox control, etc.)
            conversation_id: ID for confirmation state tracking
            
        Returns:
            TranslationResult with resolved intent and gate results
        """
        result = TranslationResult(
            original_text=text,
            mode=TranslationMode.CHAT,
            resolved_intent=CanonicalIntent.CHAT_ONLY,
            latency_tier=LatencyTier.TIER_0_RULES,
            should_execute=False,
        )
        
        # Step 1: Mode Classification
        mode, wake_phrase, remaining_text = classify_mode_with_ui(text, ui_context)
        result.mode = mode
        result.wake_phrase_detected = wake_phrase
        
        logger.debug(f"Mode: {mode.value}, wake_phrase: {wake_phrase}")
        
        # Step 2: Handle based on mode
        if mode == TranslationMode.CHAT:
            # Chat mode - no commands possible
            result.resolved_intent = CanonicalIntent.CHAT_ONLY
            result.should_execute = False
            result.execution_blocked_reason = "Chat mode - no commands"
            return result
        
        if mode == TranslationMode.FEEDBACK:
            # Feedback mode - log feedback, no commands
            result.resolved_intent = CanonicalIntent.USER_BEHAVIOR_FEEDBACK
            result.should_execute = False
            await self._handle_feedback(remaining_text, conversation_id)
            return result
        
        # Step 3: Command-Capable mode - go through intent resolution
        return await self._resolve_command_intent(
            text=remaining_text,
            result=result,
            ui_context=ui_context,
            conversation_id=conversation_id,
        )
    
    async def _resolve_command_intent(
        self,
        text: str,
        result: TranslationResult,
        ui_context: Optional[UIContext],
        conversation_id: Optional[str],
    ) -> TranslationResult:
        """
        Resolve intent for a command-capable message.
        Goes through Tier 0 -> Cache -> Tier 1 -> Gates
        """
        # Step 3a: Tier 0 Rules (run FIRST to catch known commands)
        tier0_result = tier0_classify(text)
        if tier0_result.matched:
            result.resolved_intent = tier0_result.intent
            result.intent_confidence = tier0_result.confidence
            result.latency_tier = LatencyTier.TIER_0_RULES
            logger.debug(f"Tier 0 match: {tier0_result.rule_name} -> {tier0_result.intent}")
            
            if tier0_result.intent == CanonicalIntent.CHAT_ONLY:
                result.should_execute = False
                return result
            
            # Continue to gates for commands
            return await self._apply_gates(text, result, ui_context, conversation_id)
        # Step 3b: Directive vs Story Gate (only for non-tier0 matches)
        directive_result = check_directive_gate(text)
        result.directive_gate = directive_result
        
        if not directive_result.passed:
            # Not a directive - treat as chat
            result.resolved_intent = CanonicalIntent.CHAT_ONLY
            result.latency_tier = LatencyTier.TIER_0_RULES
            result.should_execute = False
            result.execution_blocked_reason = f"Directive gate: {directive_result.reason}"
            logger.debug(f"Blocked by directive gate: {directive_result.detected_pattern}")
            return result
        
        # Step 3c: Phrase Cache Lookup
        cache_lookup = self._phrase_cache.lookup(text)
        if cache_lookup is not None:
            intent, entry = cache_lookup
            result.resolved_intent = intent
            result.intent_confidence = entry.confidence
            result.latency_tier = LatencyTier.TIER_0_RULES  # Cache is Tier 0
            result.from_phrase_cache = True
            result.cache_pattern_matched = entry.pattern
            logger.debug(f"Cache hit: {entry.pattern} -> {intent}")
            
            if intent == CanonicalIntent.CHAT_ONLY:
                result.should_execute = False
                return result
            
            # Continue to gates for commands
            return await self._apply_gates(text, result, ui_context, conversation_id)
        
        # Step 3d: Tier 1 Classifier
        classifier_response = await self._tier1_classifier.classify(
            text=text,
            candidate_intents=get_all_command_intents() + [CanonicalIntent.CHAT_ONLY],
        )
        
        result.latency_tier = LatencyTier.TIER_1_CLASSIFIER
        result.intent_confidence = classifier_response.confidence
        
        if classifier_response.confidence < CONFIDENCE_THRESHOLD:
            # Low confidence - default to chat
            result.resolved_intent = CanonicalIntent.CHAT_ONLY
            result.should_execute = False
            result.execution_blocked_reason = (
                f"Low classifier confidence ({classifier_response.confidence:.2f} < {CONFIDENCE_THRESHOLD})"
            )
            logger.debug(f"Low confidence classification, defaulting to chat")
            return result
        
        result.resolved_intent = classifier_response.intent
        
        # Auto-cache high-confidence classifications
        self._phrase_cache.add_from_tier1(
            text=text,
            intent=classifier_response.intent,
            confidence=classifier_response.confidence,
        )
        
        if classifier_response.intent == CanonicalIntent.CHAT_ONLY:
            result.should_execute = False
            return result
        
        # Continue to gates for commands
        return await self._apply_gates(text, result, ui_context, conversation_id)
    
    async def _apply_gates(
        self,
        text: str,
        result: TranslationResult,
        ui_context: Optional[UIContext],
        conversation_id: Optional[str],
    ) -> TranslationResult:
        """
        Apply context and confirmation gates to a resolved command intent.
        """
        intent = result.resolved_intent
        
        # Extract context from text and UI
        extracted = extract_context_from_text(text, intent)
        if ui_context:
            extracted.update(ui_context.to_dict())
        result.extracted_context = extracted
        
        # Step 4: Context Gate
        context_result = check_context_gate(intent, extracted)
        result.context_gate = context_result
        
        if not context_result.passed:
            result.should_execute = False
            result.execution_blocked_reason = f"Context gate: {context_result.reason}"
            logger.debug(f"Blocked by context gate: {context_result.missing_context}")
            return result
        
        # Step 5: Confirmation Gate
        confirmation_result = check_confirmation_gate(
            intent=intent,
            context=extracted,
            confirmation_state=self._confirmation_state,
            confirmation_id=conversation_id,
        )
        result.confirmation_gate = confirmation_result
        
        if confirmation_result.requires_confirmation and not confirmation_result.passed:
            # Request confirmation
            if conversation_id:
                self._confirmation_state.request_confirmation(
                    confirmation_id=conversation_id,
                    intent=intent,
                    context=extracted,
                )
            result.should_execute = False
            result.execution_blocked_reason = "Awaiting confirmation"
            return result
        
        # All gates passed - ready for execution
        result.should_execute = True
        logger.info(f"Intent resolved and approved: {intent.value}")
        return result
    
    async def check_confirmation_response(
        self,
        response: str,
        conversation_id: str,
    ) -> Tuple[bool, Optional[CanonicalIntent], Optional[Dict[str, Any]]]:
        """
        Check if a response confirms a pending high-stakes operation.
        
        Returns:
            (confirmed, intent, context) if confirmed
            (False, None, None) if not confirmed
        """
        return self._confirmation_state.check_confirmation(conversation_id, response)
    
    async def _handle_feedback(
        self,
        feedback_text: str,
        conversation_id: Optional[str],
    ) -> None:
        """
        Handle a feedback message.
        """
        # Parse the feedback
        parsed = parse_feedback_message(feedback_text)
        
        if parsed is None:
            logger.warning(f"Could not parse feedback: {feedback_text}")
            return
        
        feedback_type = parsed.get("feedback_type")
        expected_intent = parsed.get("expected_intent")
        original_message = parsed.get("original_message")
        
        if feedback_type == "false_positive":
            self._feedback_logger.log_false_positive(
                original_text=original_message or feedback_text,
                resolved_intent=expected_intent or CanonicalIntent.CHAT_ONLY,
            )
        elif feedback_type == "false_negative" and expected_intent:
            self._feedback_logger.log_false_negative(
                original_text=original_message or feedback_text,
                expected_intent=expected_intent,
            )
    
    def translate_sync(
        self,
        text: str,
        ui_context: Optional[UIContext] = None,
        conversation_id: Optional[str] = None,
    ) -> TranslationResult:
        """
        Synchronous wrapper for translate().
        Use only when async is not available.
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.translate(text, ui_context, conversation_id)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.translate(text, ui_context, conversation_id)
                )
        except RuntimeError:
            return asyncio.run(
                self.translate(text, ui_context, conversation_id)
            )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_translator_instances: Dict[str, Translator] = {}


def get_translator(
    user_id: str,
    cache_dir: Optional[Path] = None,
    log_dir: Optional[Path] = None,
) -> Translator:
    """Get or create a translator for a user."""
    if user_id not in _translator_instances:
        _translator_instances[user_id] = Translator(user_id, cache_dir, log_dir)
    return _translator_instances[user_id]


async def translate_message(
    text: str,
    user_id: str = "default",
    ui_context: Optional[UIContext] = None,
    conversation_id: Optional[str] = None,
) -> TranslationResult:
    """
    Convenience function to translate a message.
    
    Args:
        text: User message
        user_id: User identifier
        ui_context: UI context
        conversation_id: Conversation ID for confirmation tracking
        
    Returns:
        TranslationResult
    """
    translator = get_translator(user_id)
    return await translator.translate(text, ui_context, conversation_id)


def translate_message_sync(
    text: str,
    user_id: str = "default",
    ui_context: Optional[UIContext] = None,
    conversation_id: Optional[str] = None,
) -> TranslationResult:
    """Synchronous version of translate_message()."""
    translator = get_translator(user_id)
    return translator.translate_sync(text, ui_context, conversation_id)
