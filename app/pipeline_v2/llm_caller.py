# FILE: app/pipeline_v2/llm_caller.py
# Purpose: LLM Caller — unified interface for calling any model from pipeline stages.
# Called-by: app.intelligent_memory.extraction, app.pipeline_v2.bvl.cross_model_diagnostic, app.pipeline_v2.bvl.tier2_test_generator, app.pipeline_v2.checkout (+4 more)
# Depends-on: app.llm.streaming
# Last-renovated: 2026-06-11
"""
LLM Caller — unified interface for calling any model from pipeline stages.

One function: call_llm(). Takes provider, model, messages, max_tokens.
Returns the text response. Handles retries internally.

No provider-specific logic leaks into the pipeline stages.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def call_llm(
    provider: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 16000,
    temperature: float = 0.0,
    timeout_seconds: int = 300,
) -> str:
    """Call an LLM and return the text response.

    Args:
        provider: "openai", "anthropic", or "google"
        model: Model identifier (e.g. "gpt-5.4", "claude-opus-4-6")
        system_prompt: System message.
        user_prompt: User message.
        max_tokens: Maximum output tokens.
        temperature: Sampling temperature (0.0 = deterministic).
        timeout_seconds: Request timeout.

    Returns:
        Response text as string.

    Raises:
        RuntimeError: If the LLM call fails after retries.
    """
    try:
        from app.llm.streaming import call_llm_text
    except ImportError as e:
        raise RuntimeError(f"LLM streaming module not available: {e}")

    logger.info("[llm_caller] Calling %s/%s (max_tokens=%d)", provider, model, max_tokens)

    try:
        result = await call_llm_text(
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )
        logger.info("[llm_caller] Response: %d chars from %s/%s", len(result), provider, model)
        return result
    except Exception as e:
        logger.error("[llm_caller] %s/%s failed: %s", provider, model, e)
        raise RuntimeError(f"LLM call failed ({provider}/{model}): {e}")


async def call_llm_with_messages(
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 16000,
    temperature: float = 0.0,
) -> str:
    """Call an LLM with a full message list (for multi-turn conversations).

    Messages format: [{"role": "system"|"user"|"assistant", "content": "..."}]
    """
    system_prompt = ""
    user_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            system_prompt = msg.get("content", "")
        else:
            user_messages.append(msg)

    user_prompt = ""
    if user_messages:
        user_prompt = user_messages[-1].get("content", "")

    return await call_llm(
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
