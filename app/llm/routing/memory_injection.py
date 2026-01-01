# FILE: app/llm/routing/memory_injection.py
"""
Memory Context Injection for LLM Routing

Injects relevant memory context into LLM calls based on:
1. Intent depth classification (D0-D4)
2. Applicable preferences for the job type
3. Hot index retrieval results

This module is called during envelope synthesis to add memory
context to the system prompt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Import memory system (with fallback if not available)
try:
    from app.astra_memory import (
        classify_intent_depth,
        retrieve_for_query,
        get_applicable_preferences,
        get_preference_value,
        IntentDepth,
        RetrievalResult,
        PreferenceRecord,
    )
    from app.astra_memory.confidence_config import get_config
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    logger.warning("[memory_injection] ASTRA memory system not available")


@dataclass
class MemoryContext:
    """Memory context to inject into LLM call."""
    depth: str
    preferences_text: str
    facts_text: str
    token_estimate: int
    preferences_applied: List[str]
    records_retrieved: int
    
    def is_empty(self) -> bool:
        """Check if there's anything to inject."""
        return not self.preferences_text and not self.facts_text
    
    def format_for_system_prompt(self) -> str:
        """Format memory context for system prompt injection."""
        if self.is_empty():
            return ""
        
        parts = []
        
        if self.preferences_text:
            parts.append(f"<user_preferences>\n{self.preferences_text}\n</user_preferences>")
        
        if self.facts_text:
            parts.append(f"<memory_context>\n{self.facts_text}\n</memory_context>")
        
        return "\n\n".join(parts)


def _extract_user_message_text(messages: List[Dict[str, Any]]) -> str:
    """Extract text from the last user message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Multimodal: extract text parts
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                return " ".join(text_parts)
    return ""


def _format_preferences(preferences: List[Any]) -> str:
    """Format preferences as readable text."""
    if not preferences:
        return ""
    
    lines = []
    for pref in preferences:
        key = pref.preference_key
        value = pref.preference_value
        strength = pref.strength.value if hasattr(pref.strength, 'value') else str(pref.strength)
        
        # Format based on value type
        if isinstance(value, bool):
            value_str = "enabled" if value else "disabled"
        elif isinstance(value, dict):
            value_str = ", ".join(f"{k}={v}" for k, v in value.items())
        else:
            value_str = str(value)
        
        # Add strength indicator for hard rules
        if strength == "hard_rule":
            lines.append(f"• {key}: {value_str} [REQUIRED]")
        else:
            lines.append(f"• {key}: {value_str}")
    
    return "\n".join(lines)


def _format_retrieved_records(result: Any) -> str:
    """Format retrieved records as readable text."""
    if not result or not result.records:
        return ""
    
    lines = []
    for record in result.records:
        title = record.title
        content = record.content
        
        # Truncate long content
        if len(content) > 200:
            content = content[:200] + "..."
        
        lines.append(f"• [{record.record_type}] {title}: {content}")
    
    return "\n".join(lines)


def build_memory_context(
    db: Session,
    messages: List[Dict[str, Any]],
    job_type: Optional[str] = None,
    component: str = "llm_router",
) -> MemoryContext:
    """
    Build memory context for injection.
    
    Args:
        db: Database session
        messages: The conversation messages
        job_type: Optional job type for preference filtering
        component: Component name for preference scoping
        
    Returns:
        MemoryContext with formatted text ready for injection
    """
    if not MEMORY_AVAILABLE:
        return MemoryContext(
            depth="unavailable",
            preferences_text="",
            facts_text="",
            token_estimate=0,
            preferences_applied=[],
            records_retrieved=0,
        )
    
    # Extract user message for depth classification
    user_message = _extract_user_message_text(messages)
    
    # Classify intent depth
    depth = classify_intent_depth(user_message)
    
    # D0: No memory at all
    if depth == IntentDepth.D0:
        return MemoryContext(
            depth="D0",
            preferences_text="",
            facts_text="",
            token_estimate=0,
            preferences_applied=[],
            records_retrieved=0,
        )
    
    # Get applicable preferences
    preferences = []
    preferences_applied = []
    try:
        # Get preferences for this component
        prefs = get_applicable_preferences(db, component)
        
        # Also get preferences for "all" and the specific job type
        if job_type:
            job_prefs = get_applicable_preferences(db, job_type)
            prefs.extend(job_prefs)
        
        # Deduplicate by key
        seen_keys = set()
        for pref in prefs:
            if pref.preference_key not in seen_keys:
                preferences.append(pref)
                preferences_applied.append(pref.preference_key)
                seen_keys.add(pref.preference_key)
                
    except Exception as e:
        logger.warning(f"[memory_injection] Failed to get preferences: {e}")
    
    preferences_text = _format_preferences(preferences)
    
    # Retrieve facts based on depth
    facts_text = ""
    records_retrieved = 0
    
    if depth != IntentDepth.D0:
        try:
            # Extract tags/entities from message for better retrieval
            # (Simple implementation - could be enhanced with NER)
            query_tags = None
            query_entities = None
            
            result = retrieve_for_query(
                db=db,
                user_message=user_message,
                query_tags=query_tags,
                query_entities=query_entities,
                depth_override=depth,
            )
            
            facts_text = _format_retrieved_records(result)
            records_retrieved = result.records_expanded
            
        except Exception as e:
            logger.warning(f"[memory_injection] Failed to retrieve facts: {e}")
    
    # Estimate tokens (rough: 4 chars per token)
    total_text = preferences_text + facts_text
    token_estimate = len(total_text) // 4
    
    return MemoryContext(
        depth=depth.value,
        preferences_text=preferences_text,
        facts_text=facts_text,
        token_estimate=token_estimate,
        preferences_applied=preferences_applied,
        records_retrieved=records_retrieved,
    )


def inject_memory_into_system_prompt(
    system_prompt: Optional[str],
    memory_context: MemoryContext,
) -> str:
    """
    Inject memory context into system prompt.
    
    Memory is prepended to the system prompt so it's available
    as context for the entire conversation.
    """
    if memory_context.is_empty():
        return system_prompt or ""
    
    memory_block = memory_context.format_for_system_prompt()
    
    if system_prompt:
        return f"{memory_block}\n\n{system_prompt}"
    else:
        return memory_block


def get_memory_injection_stats(memory_context: MemoryContext) -> Dict[str, Any]:
    """Get stats about memory injection for logging/debugging."""
    return {
        "depth": memory_context.depth,
        "token_estimate": memory_context.token_estimate,
        "preferences_applied": memory_context.preferences_applied,
        "records_retrieved": memory_context.records_retrieved,
        "is_empty": memory_context.is_empty(),
    }


__all__ = [
    "MemoryContext",
    "build_memory_context",
    "inject_memory_into_system_prompt",
    "get_memory_injection_stats",
    "MEMORY_AVAILABLE",
]
