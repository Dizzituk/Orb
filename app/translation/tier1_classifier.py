# FILE: app/translation/tier1_classifier.py
"""
Tier 1: Lightweight LLM classifier (GPT-5 mini).
Used only when Tier 0 rules are ambiguous or don't match.

INVARIANT: Intent classification must NEVER require a frontier model.
Tier 2 (Opus/GPT-5.2/Gemini 3 Pro) is NEVER used for routing decisions.

v1.1 (2026-01): Fixed import error - get_lightweight_client doesn't exist
  - Added LightweightLLMClient wrapper class
  - Uses providers.registry.llm_call for actual calls
  - Graceful fallback if registry unavailable
"""
from __future__ import annotations
import json
import logging
from typing import Optional, List, Dict, Any
from .schemas import (
    CanonicalIntent,
    ClassifierRequest,
    ClassifierResponse,
    LatencyTier,
)
from .intents import get_all_command_intents, INTENT_DEFINITIONS

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Confidence threshold for accepting classifier result
CONFIDENCE_THRESHOLD = 0.7

# Model to use for classification - MUST be lightweight
CLASSIFIER_MODEL = "gpt-5-mini"  # Never use frontier models here

# Maximum tokens for classifier response
MAX_CLASSIFIER_TOKENS = 100


# =============================================================================
# CLASSIFIER PROMPT
# =============================================================================

CLASSIFIER_SYSTEM_PROMPT = """You are a message classifier for the ASTRA system.
Your job is to determine if a user message is a COMMAND or just CHAT.

CRITICAL RULES:
1. Questions are ALWAYS chat, never commands
2. Past tense descriptions are ALWAYS chat, never commands
3. Future planning is ALWAYS chat, never commands
4. Meta-discussion about commands is ALWAYS chat, never commands
5. Only TRUE IMPERATIVES can be commands

CANONICAL INTENTS (commands):
{intent_list}

If the message doesn't clearly match one of these intents, classify as CHAT_ONLY.

When in doubt, ALWAYS choose CHAT_ONLY. Safety over convenience.

Respond with JSON only:
{{"intent": "<INTENT_NAME>", "confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}"""


def _build_intent_list() -> str:
    """Build the intent list for the classifier prompt."""
    lines = []
    for intent in get_all_command_intents():
        defn = INTENT_DEFINITIONS[intent]
        triggers = ", ".join(f'"{p}"' for p in defn.trigger_phrases[:3])
        lines.append(f"- {intent.value}: {defn.description}")
        if triggers:
            lines.append(f"  Triggers: {triggers}")
    return "\n".join(lines)


# =============================================================================
# LIGHTWEIGHT LLM CLIENT (v1.1)
# =============================================================================

class LightweightLLMClient:
    """
    Simple LLM client wrapper for lightweight classification tasks.
    
    v1.1: Created to replace missing get_lightweight_client import.
    Uses providers.registry.llm_call for actual API calls.
    """
    
    def __init__(self):
        self._registry_available = False
        self._llm_call = None
        
        # Try to import the registry
        try:
            from app.providers.registry import llm_call as registry_llm_call
            self._llm_call = registry_llm_call
            self._registry_available = True
            logger.debug("[LightweightLLMClient] Registry loaded successfully")
        except ImportError as e:
            logger.warning(f"[LightweightLLMClient] Registry import failed: {e}")
    
    async def call_async(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 100,
        temperature: float = 0.0,
    ) -> str:
        """
        Make an async LLM call.
        
        Returns the content string from the response.
        """
        if not self._registry_available or self._llm_call is None:
            logger.warning("[LightweightLLMClient] Registry unavailable, returning empty")
            return '{"intent": "CHAT_ONLY", "confidence": 0.5, "reasoning": "LLM unavailable"}'
        
        try:
            # Determine provider from model name
            provider = "openai"  # Default to OpenAI for lightweight models
            if "claude" in model.lower() or "anthropic" in model.lower():
                provider = "anthropic"
            elif "gemini" in model.lower():
                provider = "google"
            
            # Make the call
            result = await self._llm_call(
                provider_id=provider,
                model_id=model,
                messages=messages,
            )
            
            # Extract content from result
            if hasattr(result, 'content'):
                return result.content
            elif isinstance(result, dict) and 'content' in result:
                return result['content']
            else:
                return str(result)
                
        except Exception as e:
            logger.error(f"[LightweightLLMClient] Call failed: {e}")
            return f'{{"intent": "CHAT_ONLY", "confidence": 0.5, "reasoning": "LLM call failed: {e}"}}'


def get_lightweight_client() -> LightweightLLMClient:
    """
    Get a lightweight LLM client for classification tasks.
    
    v1.1: Added this function to fix the import error.
    """
    return LightweightLLMClient()


# =============================================================================
# CLASSIFIER CLIENT
# =============================================================================

class Tier1Classifier:
    """
    Lightweight LLM classifier for intent classification.
    
    Usage:
        classifier = Tier1Classifier(llm_client)
        result = await classifier.classify(text, candidates)
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize classifier.
        
        Args:
            llm_client: Optional LLM client. If None, uses default lightweight client.
        """
        self._llm_client = llm_client
        self._system_prompt = CLASSIFIER_SYSTEM_PROMPT.format(
            intent_list=_build_intent_list()
        )
    
    async def classify(
        self,
        text: str,
        candidate_intents: Optional[List[CanonicalIntent]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ClassifierResponse:
        """
        Classify a message using the lightweight LLM.
        
        Args:
            text: The user message to classify
            candidate_intents: Optional list of candidate intents to consider
            context: Optional additional context
            
        Returns:
            ClassifierResponse with intent and confidence
        """
        if candidate_intents is None:
            candidate_intents = list(CanonicalIntent)
        
        try:
            response = await self._call_classifier(text, candidate_intents, context)
            return response
        except Exception as e:
            logger.error(f"Tier 1 classification failed: {e}")
            # Default to chat on error
            return ClassifierResponse(
                intent=CanonicalIntent.CHAT_ONLY,
                confidence=0.5,
                reasoning=f"Classification failed, defaulting to chat: {e}",
            )
    
    async def _call_classifier(
        self,
        text: str,
        candidates: List[CanonicalIntent],
        context: Optional[Dict[str, Any]],
    ) -> ClassifierResponse:
        """Internal method to call the LLM classifier."""
        
        # Build user prompt
        user_prompt = f"Classify this message:\n\n\"{text}\""
        if context:
            user_prompt += f"\n\nContext: {json.dumps(context)}"
        
        # Get LLM client
        client = self._get_llm_client()
        
        # Call LLM
        response = await client.call_async(
            model=CLASSIFIER_MODEL,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=MAX_CLASSIFIER_TOKENS,
            temperature=0.0,  # Deterministic
        )
        
        # Parse response
        return self._parse_response(response)
    
    def _get_llm_client(self):
        """Get the LLM client to use."""
        if self._llm_client is not None:
            return self._llm_client
        
        # v1.1: Use local lightweight client instead of broken import
        return get_lightweight_client()
    
    def _parse_response(self, response: str) -> ClassifierResponse:
        """Parse the LLM response into a ClassifierResponse."""
        try:
            # Try to parse as JSON
            data = json.loads(response.strip())
            intent_str = data.get("intent", "CHAT_ONLY")
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "")
            
            # Map to CanonicalIntent
            try:
                intent = CanonicalIntent(intent_str)
            except ValueError:
                intent = CanonicalIntent.CHAT_ONLY
                confidence = 0.5
                reasoning = f"Unknown intent '{intent_str}', defaulting to chat"
            
            return ClassifierResponse(
                intent=intent,
                confidence=confidence,
                reasoning=reasoning,
            )
        except json.JSONDecodeError:
            # If not JSON, default to chat
            return ClassifierResponse(
                intent=CanonicalIntent.CHAT_ONLY,
                confidence=0.5,
                reasoning="Failed to parse classifier response",
            )


# =============================================================================
# SYNC WRAPPER
# =============================================================================

def classify_sync(
    text: str,
    candidate_intents: Optional[List[CanonicalIntent]] = None,
    context: Optional[Dict[str, Any]] = None,
    llm_client=None,
) -> ClassifierResponse:
    """
    Synchronous wrapper for classification.
    Use only when async is not available.
    """
    import asyncio
    
    classifier = Tier1Classifier(llm_client)
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context, create a new loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    classifier.classify(text, candidate_intents, context)
                )
                return future.result()
        else:
            return loop.run_until_complete(
                classifier.classify(text, candidate_intents, context)
            )
    except RuntimeError:
        # No event loop
        return asyncio.run(
            classifier.classify(text, candidate_intents, context)
        )


# =============================================================================
# STUB FOR TESTING
# =============================================================================

class MockTier1Classifier:
    """
    Mock classifier for testing without LLM calls.
    """
    
    def __init__(self, default_intent: CanonicalIntent = CanonicalIntent.CHAT_ONLY):
        self.default_intent = default_intent
        self.call_count = 0
        self.last_text = None
    
    async def classify(
        self,
        text: str,
        candidate_intents: Optional[List[CanonicalIntent]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ClassifierResponse:
        self.call_count += 1
        self.last_text = text
        
        return ClassifierResponse(
            intent=self.default_intent,
            confidence=0.8,
            reasoning="Mock classification",
        )
