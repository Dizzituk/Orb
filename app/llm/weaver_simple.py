# FILE: app/llm/weaver_simple.py
r"""
Simple Weaver - Text organizer for ASTRA.

This module implements the LOCKED Weaver behaviour specification:
- Purpose: Convert human rambling into a structured job outline
- NOT a full spec builder - just a text organizer
- No system access, no DB, no file inspection
- Stateless, tool-free, isolated

v1.2 (2026-01-19): Fixed LLM call signature (system_prompt is first arg).
v1.1 (2026-01-19): Fixed LLM routing to use ENV-based stage_models.
v1.0 (2026-01-19): Initial implementation per locked behaviour spec.
"""
from __future__ import annotations

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# WEAVER SYSTEM PROMPT (LOCKED BEHAVIOUR)
# =============================================================================

WEAVER_SYSTEM_PROMPT = """You are Weaver, a text organizer.

Your ONLY job: Take the human's rambling and restructure it into a clear, readable document.

## What You DO:
- Group related ideas together
- Rephrase for clarity and structure ONLY (meaning stays exactly the same)
- Preserve ambiguities and contradictions (do NOT resolve them)
- Write down explicit implications if clearly stated (e.g., "this must be secure")

## What You DO NOT DO:
- No adding detail
- No removing ambiguity  
- No resolving contradictions
- No inferring intent
- No inventing implications
- No technical feasibility checking
- No system/file/architecture awareness
- No suggesting improvements
- No optimisation

## Output Format:
Produce a structured job description document. Structure adapts to the content.
Possible sections (use only what's relevant):
- What is being built or changed
- Intended outcome
- Platform/environment (only if mentioned)
- Constraints (only if explicitly stated)
- End goal / success picture
- Priority notes
- Unresolved ambiguities (preserved, not resolved)

## Critical Rule:
If the human didn't say it, it doesn't appear in your output.

## Exception - Single Question Allowed:
You may ask ONE clarifying question ONLY if:
- The core goal is completely missing, AND
- It is impossible to infer ANY goal from the input

Otherwise, produce the job description even if vague.
"""


def _get_weaver_config() -> Tuple[str, str]:
    """
    Get provider and model for Weaver using ENV-based stage_models.
    
    Uses centralized configuration from app.llm.stage_models which reads:
      - WEAVER_PROVIDER (e.g., "openai", "anthropic", "google")
      - WEAVER_MODEL (e.g., "gpt-4.1-mini", "gemini-2.0-flash")
    
    Falls back to defaults if not set.
    """
    try:
        from app.llm.stage_models import get_weaver_config as _get_cfg
        cfg = _get_cfg()
        logger.debug("[weaver_simple] Using stage_models config: provider=%s, model=%s", cfg.provider, cfg.model)
        return cfg.provider, cfg.model
    except ImportError:
        # Fallback if stage_models not available
        import os
        provider = os.getenv("WEAVER_PROVIDER", "openai")
        model = os.getenv("WEAVER_MODEL", "gpt-4.1-mini")
        logger.warning("[weaver_simple] stage_models not available, using env directly: provider=%s, model=%s", provider, model)
        return provider, model


def _call_llm(system_prompt: str, user_message: str, provider: str, model: str) -> str:
    """
    Make an LLM call using the appropriate provider client.
    
    Uses the unified clients module which routes through the provider registry.
    This ensures consistent behaviour regardless of provider.
    
    IMPORTANT: All client functions have signature:
        call_xxx(system_prompt, messages, temperature=0.7, model=None)
    """
    messages = [{"role": "user", "content": user_message}]
    
    provider_lower = provider.lower()
    
    try:
        if provider_lower in ("openai", "openai-compatible"):
            from app.llm.clients import call_openai
            # Signature: call_openai(system_prompt, messages, temperature=0.7, model=None)
            content, usage = call_openai(system_prompt, messages, 0.3, model)
            logger.info("[weaver_simple] OpenAI call complete: %d tokens", usage.get("total_tokens", 0))
            return content
            
        elif provider_lower in ("anthropic", "claude"):
            from app.llm.clients import call_anthropic
            # Signature: call_anthropic(system_prompt, messages, temperature=0.7, model=None)
            content, usage = call_anthropic(system_prompt, messages, 0.3, model)
            logger.info("[weaver_simple] Anthropic call complete: %d tokens", usage.get("total_tokens", 0))
            return content
            
        elif provider_lower in ("google", "gemini"):
            from app.llm.clients import call_google
            # Signature: call_google(system_prompt, messages, temperature=0.7, model=None)
            content, usage = call_google(system_prompt, messages, 0.3, model)
            logger.info("[weaver_simple] Google call complete: %d tokens", usage.get("total_tokens", 0))
            return content
            
        else:
            # Unknown provider - try OpenAI as fallback
            logger.warning("[weaver_simple] Unknown provider '%s', trying OpenAI fallback", provider)
            from app.llm.clients import call_openai
            content, usage = call_openai(system_prompt, messages, 0.3, model)
            return content
            
    except Exception as e:
        logger.exception("[weaver_simple] LLM call failed")
        raise


def _is_goal_missing(ramble_text: str) -> bool:
    """
    Check if the ramble has no discernible goal at all.
    This is the ONLY case where Weaver may ask a question.
    """
    # Very short or empty
    if len(ramble_text.strip()) < 20:
        return True
    
    # Contains only vague phrases with no concrete nouns
    vague_only_patterns = [
        "make it better",
        "fix it",
        "improve things",
        "do something",
        "help me",
        "you know",
        "just do it",
    ]
    lower = ramble_text.lower()
    
    # If the ENTIRE content is just vague phrases
    stripped = lower
    for pattern in vague_only_patterns:
        stripped = stripped.replace(pattern, "")
    
    # If after removing vague phrases, nothing substantial remains
    remaining = stripped.strip()
    if len(remaining) < 10:
        return True
    
    return False


def weave(ramble_text: str) -> str:
    """
    Transform rambling text into a structured job description.
    
    This is a SYNCHRONOUS function that calls an LLM to organize text.
    Uses ENV-based provider/model configuration.
    
    Args:
        ramble_text: The raw, unstructured human input to organize
        
    Returns:
        A structured job description document (plain text)
    """
    if not ramble_text or not ramble_text.strip():
        return "❓ **Clarification needed:**\n\nWhat would you like me to help you with?"
    
    # Check for completely missing goal
    if _is_goal_missing(ramble_text):
        return "❓ **Clarification needed:**\n\nI couldn't identify a clear goal. What specifically would you like to accomplish?"
    
    provider, model = _get_weaver_config()
    logger.info("[weaver_simple] Starting weave with provider=%s, model=%s", provider, model)
    
    try:
        user_message = f"Organize this ramble into a job description:\n\n{ramble_text}\n\nRemember: Only include what was actually said. Preserve any ambiguities or contradictions."
        
        result = _call_llm(
            system_prompt=WEAVER_SYSTEM_PROMPT,
            user_message=user_message,
            provider=provider,
            model=model,
        )
        
        return result
            
    except Exception as e:
        logger.exception("[weaver_simple] LLM call failed")
        return f"❌ **Weaver Error:**\n\nFailed to organize the text: {str(e)}"


async def weave_async(ramble_text: str) -> str:
    """
    Async version of weave() for streaming contexts.
    Wraps the synchronous call in an executor.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, weave, ramble_text)


def _format_messages_as_ramble(messages: list[dict]) -> str:
    """Format conversation messages as a ramble text block."""
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if not content:
            continue
        # Skip system messages and Orb control messages
        if role == "system":
            continue
        # Skip Weaver/command outputs
        lower_content = content.lower()
        if any(marker in lower_content for marker in [
            "🧵 weaving", "📋 spec", "astra, command:", "astra command:",
            "shall i send", "say yes to proceed", "⚠️ weak spots",
        ]):
            continue
        
        speaker = "Human" if role == "user" else "Assistant"
        lines.append(f"[{speaker}]: {content}")
    
    return "\n\n".join(lines)


def weave_from_messages(messages: list[dict]) -> str:
    """
    Convenience function to weave from a list of conversation messages.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
        
    Returns:
        A structured job description document
    """
    ramble_text = _format_messages_as_ramble(messages)
    return weave(ramble_text)


# =============================================================================
# TEST HARNESS
# =============================================================================

def _test_weave():
    """Simple test harness for Weaver."""
    
    test_cases = [
        # Test 1: Normal ramble
        {
            "name": "Normal ramble",
            "input": "I want an app that runs on Android, it needs to be secure, and it should track my deliveries...",
            "expected_contains": ["Android", "secure"],
            "expected_not_contains": ["React", "Firebase"],  # Should not invent tech
        },
        # Test 2: Contradiction preserved
        {
            "name": "Contradiction preserved",
            "input": "It must be offline only... but also it should sync live to the cloud...",
            "expected_contains": ["offline", "cloud"],  # Both should appear
            "expected_not_contains": [],
        },
        # Test 3: No goal - question allowed
        {
            "name": "No goal - question allowed",
            "input": "Just make it better",
            "expected_contains": ["Clarification", "?"],
            "expected_not_contains": [],
        },
    ]
    
    print("=" * 60)
    print("WEAVER SIMPLE - TEST HARNESS")
    print("=" * 60)
    
    for i, tc in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {tc['name']} ---")
        print(f"Input: {tc['input'][:50]}...")
        
        result = weave(tc["input"])
        print(f"Output:\n{result}\n")
        
        # Check expected content
        for expected in tc["expected_contains"]:
            if expected.lower() in result.lower():
                print(f"  ✅ Contains '{expected}'")
            else:
                print(f"  ❌ Missing '{expected}'")
        
        for not_expected in tc["expected_not_contains"]:
            if not_expected.lower() not in result.lower():
                print(f"  ✅ Does not contain '{not_expected}'")
            else:
                print(f"  ⚠️  Unexpectedly contains '{not_expected}'")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    _test_weave()
