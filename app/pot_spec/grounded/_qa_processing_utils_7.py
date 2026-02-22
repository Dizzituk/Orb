from __future__ import annotations
import logging
import re
from typing import Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


DEEP_ANALYSIS_TOKENS_PER_QUESTION = 400

DEEP_ANALYSIS_BASE_TOKENS = 2000  # v1.30: Increased for full doc analysis

STANDARD_QA_BASE_TOKENS = 1000  # v1.30: Increased for full doc analysis

def detect_simple_instruction(content: str) -> Optional[str]:
    """
    Detect simple instructions like "Say X" that don't need LLM.
    
    Returns the simple reply if detected, None otherwise.
    """
    if not content:
        return None
    
    content_lower = content.lower().strip()
    
    # "Say X" / "Reply X" / "Answer X" patterns
    simple_patterns = [
        r'^(?:just\s+)?say\s+["\']?([^"\']+)["\']?[.!?]?$',
        r'^(?:just\s+)?reply\s+(?:with\s+)?["\']?([^"\']+)["\']?[.!?]?$',
        r'^(?:your\s+)?answer\s+(?:is\s+)?["\']?([^"\']+)["\']?[.!?]?$',
        r'^(?:respond\s+(?:with\s+)?)["\']?([^"\']+)["\']?[.!?]?$',
    ]
    
    for pattern in simple_patterns:
        match = re.match(pattern, content_lower)
        if match:
            reply = match.group(1).strip()
            if reply:
                return reply
    
    # "Say OK" / "Reply OK" without quotes
    if re.match(r'^(?:just\s+)?(?:say|reply|respond)\s+(ok|okay|yes|no|done|understood)[.!?]?$', content_lower):
        match = re.search(r'\b(ok|okay|yes|no|done|understood)\b', content_lower)
        if match:
            return match.group(1).capitalize()
    
    return None

def _needs_deep_analysis(user_text: str) -> bool:
    """
    v1.28: Detect if user wants deep Q&A analysis beyond simple fill-in.
    
    Deep analysis includes:
    - Checking if existing answers are correct
    - Identifying trick questions
    - Fixing wrong answers
    
    Args:
        user_text: The user's request text (from Weaver or direct input)
        
    Returns:
        True if user wants correctness checking, False for simple fill-in
    """
    if not user_text:
        return False
    
    text_lower = user_text.lower()
    
    # Triggers that indicate user wants correctness checking
    deep_analysis_triggers = [
        # Fix/correct existing answers
        "fix wrong",
        "fix incorrect",
        "fix the wrong",
        "fix any wrong",
        "correct wrong",
        "correct incorrect",
        "correct the wrong",
        "correct the answers",
        "correct any wrong",
        
        # Check/verify existing answers
        "check the answers",
        "check answers",
        "verify answers",
        "verify the answers",
        "review answers",
        "review the answers",
        
        # Trick questions
        "identify trick",
        "spot the trick",
        "find trick",
        "trick question",
        "watch for trick",
        "look for trick",
        
        # Answers might be wrong
        "some are wrong",
        "might be wrong",
        "may be wrong",
        "may be incorrect",
        "could be wrong",
        "answers wrong",
        "wrong answers",
        "incorrect answers",
        "answered wrong",
        
        # General deep analysis signals
        "all of the above",
        "check all",
        "verify all",
        "fix all",
        "correct all",
        
        # v1.30: Additional triggers for fill-in scenarios
        "fill in the missing",
        "fill in missing",
        "fill the missing",
        "missing ones",
        "missing answers",
    ]
    
    for trigger in deep_analysis_triggers:
        if trigger in text_lower:
            logger.info(
                "[qa_processing] v1.30 _needs_deep_analysis: TRIGGERED by '%s'",
                trigger
            )
            return True
    
    return False

MULTI_FILE_SYNTHESIS_MIN_TOKENS = 2000

MULTI_FILE_SYNTHESIS_MAX_TOKENS = 16000

MULTI_FILE_SYNTHESIS_TOKENS_PER_FILE = 500
