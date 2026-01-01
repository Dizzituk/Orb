# FILE: app/translation/feedback.py
"""
Feedback logging and rule promotion for ASTRA Translation Layer.

Handles:
- Misfire logging ("that should have been a command", "that should NOT have been a command")
- Structured feedback events
- Rule promotion from feedback to Tier 0

IMPORTANT: Behavior tuning occurs ONLY in sandbox.
This module logs feedback for later processing.
"""
from __future__ import annotations
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
from .schemas import (
    CanonicalIntent,
    FeedbackEvent,
    TranslationResult,
)
from .phrase_cache import PhraseCache, get_phrase_cache

logger = logging.getLogger(__name__)

# =============================================================================
# FEEDBACK PATTERNS
# =============================================================================

# Patterns that indicate feedback about misfires
FEEDBACK_PATTERNS = {
    "false_negative": [
        # "That should have been a command"
        r"(?:that|this|it)\s+should\s+have\s+been\s+(?:a\s+)?command",
        r"should\s+(?:have\s+)?trigger(?:ed)?",
        r"should\s+(?:have\s+)?execut(?:e|ed)",
        r"missed\s+(?:the\s+)?command",
        r"didn't\s+(?:catch|recognize|detect)",
    ],
    "false_positive": [
        # "That should NOT have been a command"
        r"(?:that|this|it)\s+should\s+(?:not|n't)\s+have\s+been\s+(?:a\s+)?command",
        r"should\s+(?:not|n't)\s+(?:have\s+)?trigger(?:ed)?",
        r"shouldn't\s+(?:have\s+)?execut(?:e|ed)",
        r"misfire",
        r"false\s+(?:positive|alarm)",
        r"was\s+(?:just\s+)?(?:chat|talking|asking)",
        r"didn't\s+mean\s+to\s+trigger",
    ],
}


# =============================================================================
# FEEDBACK LOGGER
# =============================================================================

class FeedbackLogger:
    """
    Logs feedback events for behavior tuning.
    Events are stored in a log file for sandbox processing.
    """
    
    def __init__(
        self,
        user_id: str,
        log_dir: Optional[Path] = None,
        phrase_cache: Optional[PhraseCache] = None,
    ):
        """
        Initialize feedback logger.
        
        Args:
            user_id: User identifier
            log_dir: Directory for feedback logs
            phrase_cache: Optional phrase cache to update
        """
        self.user_id = user_id
        self.log_dir = log_dir
        self._phrase_cache = phrase_cache or get_phrase_cache(user_id)
        self._session_feedback: List[FeedbackEvent] = []
    
    def log_feedback(
        self,
        original_text: str,
        resolved_intent: Optional[CanonicalIntent],
        expected_intent: CanonicalIntent,
        feedback_type: str,
        user_correction: Optional[str] = None,
        translation_result: Optional[TranslationResult] = None,
    ) -> FeedbackEvent:
        """
        Log a feedback event.
        
        Args:
            original_text: The original user message
            resolved_intent: What the system resolved it to
            expected_intent: What it should have been
            feedback_type: "false_positive" or "false_negative"
            user_correction: Optional user's correction text
            translation_result: Full translation result if available
            
        Returns:
            The logged FeedbackEvent
        """
        event = FeedbackEvent(
            original_text=original_text,
            resolved_intent=resolved_intent,
            expected_intent=expected_intent,
            feedback_type=feedback_type,
            user_correction=user_correction,
            translation_result=translation_result,
        )
        
        self._session_feedback.append(event)
        self._persist_event(event)
        
        # Update phrase cache based on feedback
        self._update_phrase_cache(event)
        
        logger.info(
            f"Feedback logged: {feedback_type} - "
            f"'{original_text[:50]}...' -> expected {expected_intent.value}"
        )
        
        return event
    
    def log_false_positive(
        self,
        original_text: str,
        resolved_intent: CanonicalIntent,
        translation_result: Optional[TranslationResult] = None,
    ) -> FeedbackEvent:
        """
        Log a false positive (triggered when it shouldn't have).
        The expected intent is CHAT_ONLY.
        """
        return self.log_feedback(
            original_text=original_text,
            resolved_intent=resolved_intent,
            expected_intent=CanonicalIntent.CHAT_ONLY,
            feedback_type="false_positive",
            translation_result=translation_result,
        )
    
    def log_false_negative(
        self,
        original_text: str,
        expected_intent: CanonicalIntent,
        user_correction: Optional[str] = None,
        translation_result: Optional[TranslationResult] = None,
    ) -> FeedbackEvent:
        """
        Log a false negative (didn't trigger when it should have).
        """
        return self.log_feedback(
            original_text=original_text,
            resolved_intent=CanonicalIntent.CHAT_ONLY,
            expected_intent=expected_intent,
            feedback_type="false_negative",
            user_correction=user_correction,
            translation_result=translation_result,
        )
    
    def _update_phrase_cache(self, event: FeedbackEvent) -> None:
        """Update phrase cache based on feedback."""
        if event.feedback_type == "false_positive":
            # This was incorrectly triggered - add to cache as CHAT_ONLY
            self._phrase_cache.add(
                text=event.original_text,
                intent=CanonicalIntent.CHAT_ONLY,
                confidence=1.0,
                source="feedback_false_positive",
            )
        elif event.feedback_type == "false_negative":
            # This should have been a command - add to cache
            self._phrase_cache.add(
                text=event.original_text,
                intent=event.expected_intent,
                confidence=1.0,
                source="feedback_false_negative",
            )
    
    def _persist_event(self, event: FeedbackEvent) -> None:
        """Persist feedback event to log file."""
        if self.log_dir is None:
            return
        
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.log_dir / f"feedback_{self.user_id}.jsonl"
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(event.model_dump(mode='json'), default=str) + '\n')
        except Exception as e:
            logger.warning(f"Failed to persist feedback event: {e}")
    
    def get_session_feedback(self) -> List[FeedbackEvent]:
        """Get all feedback from this session."""
        return self._session_feedback.copy()
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics about feedback."""
        total = len(self._session_feedback)
        false_positives = sum(1 for e in self._session_feedback if e.feedback_type == "false_positive")
        false_negatives = sum(1 for e in self._session_feedback if e.feedback_type == "false_negative")
        
        return {
            "total_feedback": total,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "false_positive_rate": false_positives / total if total > 0 else 0,
            "false_negative_rate": false_negatives / total if total > 0 else 0,
        }


# =============================================================================
# FEEDBACK PARSER
# =============================================================================

def parse_feedback_message(
    feedback_text: str,
    previous_message: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Parse a feedback message to extract feedback type and intent.
    
    Args:
        feedback_text: The feedback message (after "Astra, feedback:")
        previous_message: The message being corrected
        
    Returns:
        Dict with 'feedback_type' and optionally 'expected_intent',
        or None if not parseable
    """
    import re
    
    feedback_lower = feedback_text.lower()
    
    # Check for false negative patterns
    for pattern in FEEDBACK_PATTERNS["false_negative"]:
        if re.search(pattern, feedback_lower, re.IGNORECASE):
            # Try to extract what intent it should have been
            expected_intent = _extract_expected_intent(feedback_text)
            return {
                "feedback_type": "false_negative",
                "expected_intent": expected_intent,
                "original_message": previous_message,
            }
    
    # Check for false positive patterns
    for pattern in FEEDBACK_PATTERNS["false_positive"]:
        if re.search(pattern, feedback_lower, re.IGNORECASE):
            return {
                "feedback_type": "false_positive",
                "expected_intent": CanonicalIntent.CHAT_ONLY,
                "original_message": previous_message,
            }
    
    return None


def _extract_expected_intent(feedback_text: str) -> Optional[CanonicalIntent]:
    """
    Try to extract the expected intent from feedback text.
    E.g., "that should have been CREATE ARCHITECTURE MAP"
    """
    from .intents import get_intent_by_trigger_phrase, INTENT_DEFINITIONS
    
    feedback_upper = feedback_text.upper()
    
    # Check for explicit intent mentions
    for intent, defn in INTENT_DEFINITIONS.items():
        if intent.value in feedback_upper:
            return intent
        for phrase in defn.trigger_phrases:
            if phrase.upper() in feedback_upper:
                return intent
    
    return None


# =============================================================================
# RULE PROMOTION
# =============================================================================

class RulePromoter:
    """
    Promotes frequently-used phrase cache entries to Tier 0 rules.
    
    This runs in SANDBOX ONLY. The main system just logs feedback.
    """
    
    def __init__(self, phrase_cache: PhraseCache):
        self.phrase_cache = phrase_cache
        self._promoted_rules: List[Dict[str, Any]] = []
    
    def check_promotions(self) -> List[Dict[str, Any]]:
        """
        Check for cache entries that should be promoted to Tier 0.
        
        Returns:
            List of promotion candidates with their suggested rules
        """
        candidates = self.phrase_cache.get_promotion_candidates()
        promotions = []
        
        for entry in candidates:
            promotion = {
                "pattern": entry.pattern,
                "intent": entry.intent.value,
                "hit_count": entry.hit_count,
                "confidence": entry.confidence,
                "source": entry.source,
                "suggested_rule": self._suggest_rule(entry),
            }
            promotions.append(promotion)
        
        return promotions
    
    def _suggest_rule(self, entry) -> str:
        """
        Suggest a Tier 0 rule for the given cache entry.
        """
        pattern = entry.pattern
        intent = entry.intent.value
        
        # Generate regex pattern
        regex_pattern = self._pattern_to_regex(pattern)
        
        return f'''
# Auto-promoted from phrase cache (hits: {entry.hit_count})
# Original pattern: "{pattern}"
("{regex_pattern}", CanonicalIntent.{intent}),
'''
    
    def _pattern_to_regex(self, pattern: str) -> str:
        """Convert a normalized pattern to a regex."""
        # Escape special regex characters
        import re
        escaped = re.escape(pattern)
        
        # Convert placeholders back to regex groups
        escaped = escaped.replace(r'\{uuid\}', r'[a-f0-9\-]{36}')
        escaped = escaped.replace(r'\{id\}', r'[a-f0-9]{8,}')
        escaped = escaped.replace(r'\{job\}', r'job[_\-]?\d+')
        escaped = escaped.replace(r'\{target\}', r'[\w\-]+')
        escaped = escaped.replace(r'\{sandbox\}', r'sandbox[_\-]?\d*')
        
        return f"^{escaped}$"
    
    def apply_promotion(self, pattern: str) -> None:
        """
        Mark a pattern as promoted.
        The actual rule addition happens in sandbox processing.
        """
        self.phrase_cache.mark_promoted(pattern)
        logger.info(f"Promoted pattern to Tier 0: {pattern}")


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

_logger_instances: Dict[str, FeedbackLogger] = {}


def get_feedback_logger(
    user_id: str,
    log_dir: Optional[Path] = None,
) -> FeedbackLogger:
    """Get or create a feedback logger for a user."""
    if user_id not in _logger_instances:
        _logger_instances[user_id] = FeedbackLogger(user_id, log_dir)
    return _logger_instances[user_id]
