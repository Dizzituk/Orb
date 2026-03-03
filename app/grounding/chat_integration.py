# FILE: app/grounding/chat_integration.py
"""
Integration layer for wiring the Grounding Gate into chat routing.

Provides a synchronous-compatible wrapper that can be called from
handle_chat_mode() to augment the system prompt with grounding
evidence when the user's message is claims-dependent or contested.

Since the grounding gate is async (web searches) but chat_routing
calls are sync, this module provides run_grounding_sync() which
uses asyncio to bridge the gap.

v1.0 (2026-03): Initial implementation.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _is_grounding_enabled() -> bool:
    """Quick check — avoids importing the full gate if disabled."""
    val = os.getenv("GROUNDING_GATE_ENABLED", "true").strip().lower()
    return val in ("true", "1", "yes")


def run_grounding_sync(
    message: str,
    system_prompt: str,
    context: Optional[dict] = None,
) -> tuple[str, dict]:
    """
    Synchronous wrapper for the grounding gate.
    
    Classifies the message and, if grounding is needed, runs web
    searches and augments the system prompt with evidence.
    
    Args:
        message: User's raw message text.
        system_prompt: Current system prompt (will be augmented).
        context: Optional context dict for search.
    
    Returns:
        Tuple of (augmented_system_prompt, grounding_metadata).
        If no grounding is needed, returns the original prompt unchanged.
        grounding_metadata contains: grounding_applied, category, source_count,
        confidence, domain_hint for logging/tracing.
    """
    if not _is_grounding_enabled():
        return system_prompt, {"grounding_applied": False, "reason": "disabled"}
    
    try:
        from app.grounding.grounding_gate import evaluate_grounding
    except ImportError as e:
        logger.warning("[grounding_integration] Gate not available: %s", e)
        return system_prompt, {"grounding_applied": False, "reason": "import_error"}
    
    # Run the async gate in a sync context
    try:
        # Try to get the running event loop
        try:
            loop = asyncio.get_running_loop()
            # We're already in an async context — can't use run_until_complete.
            # Create a task and return. This shouldn't normally happen since
            # handle_chat_mode is sync, but handle it gracefully.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = loop.run_in_executor(
                    pool,
                    lambda: asyncio.run(evaluate_grounding(message, context=context)),
                )
                # This won't work synchronously — fall back to topic-only classification
                logger.warning("[grounding_integration] Already in async context, falling back to classification only")
                from app.grounding.topic_classifier import classify_topic
                classification = classify_topic(message)
                if classification.category.value == "personal":
                    return system_prompt, {
                        "grounding_applied": False,
                        "category": classification.category.value,
                        "reason": "personal_topic",
                    }
                # Can't do async search from here — inject a search-needed hint
                from app.grounding.source_attribution import GROUNDING_SYSTEM_PROMPT
                hint = (
                    "\n\n## GROUNDING NOTICE\n"
                    "This query appears to be claims-dependent but grounding search "
                    "could not be performed in this context. Caveat any factual claims "
                    "and suggest the user ask ASTRA to search for the latest data.\n"
                )
                return system_prompt + hint, {
                    "grounding_applied": False,
                    "category": classification.category.value,
                    "reason": "async_context_limitation",
                }
        except RuntimeError:
            # No running event loop — we can use asyncio.run()
            result = asyncio.run(evaluate_grounding(message, context=context))
    except Exception as e:
        logger.error("[grounding_integration] Gate evaluation failed: %s", e)
        return system_prompt, {"grounding_applied": False, "reason": f"error: {e}"}
    
    # Build metadata
    metadata = {
        "grounding_applied": result.grounding_applied,
        "category": result.category.value,
        "confidence": result.confidence,
        "domain_hint": result.domain_hint,
        "source_count": result.source_count,
        "search_queries": result.search_queries_used,
        "signals": result.classification_signals[:5],
    }
    
    if not result.grounding_applied:
        return system_prompt, metadata
    
    # Augment the system prompt with grounding injection
    augmented = system_prompt + result.system_prompt_injection
    
    # Add confidence note if evidence is thin
    if result.confidence_note:
        augmented += f"\n\n[CONFIDENCE NOTE FOR LLM]: {result.confidence_note}"
    
    logger.info(
        "[grounding_integration] Prompt augmented: category=%s, sources=%d, domain=%s",
        result.category.value, result.source_count, result.domain_hint,
    )
    
    return augmented, metadata


async def run_grounding_async(
    message: str,
    system_prompt: str,
    context: Optional[dict] = None,
) -> tuple[str, dict]:
    """
    Async version — for use in async streaming handlers.
    
    Same interface as run_grounding_sync but properly awaits the gate.
    """
    if not _is_grounding_enabled():
        return system_prompt, {"grounding_applied": False, "reason": "disabled"}
    
    try:
        from app.grounding.grounding_gate import evaluate_grounding
    except ImportError as e:
        logger.warning("[grounding_integration] Gate not available: %s", e)
        return system_prompt, {"grounding_applied": False, "reason": "import_error"}
    
    try:
        result = await evaluate_grounding(message, context=context)
    except Exception as e:
        logger.error("[grounding_integration] Gate evaluation failed: %s", e)
        return system_prompt, {"grounding_applied": False, "reason": f"error: {e}"}
    
    metadata = {
        "grounding_applied": result.grounding_applied,
        "category": result.category.value,
        "confidence": result.confidence,
        "domain_hint": result.domain_hint,
        "source_count": result.source_count,
        "search_queries": result.search_queries_used,
        "signals": result.classification_signals[:5],
    }
    
    if not result.grounding_applied:
        return system_prompt, metadata
    
    augmented = system_prompt + result.system_prompt_injection
    
    if result.confidence_note:
        augmented += f"\n\n[CONFIDENCE NOTE FOR LLM]: {result.confidence_note}"
    
    logger.info(
        "[grounding_integration] Prompt augmented (async): category=%s, sources=%d",
        result.category.value, result.source_count,
    )
    
    return augmented, metadata


__all__ = [
    "run_grounding_sync",
    "run_grounding_async",
]
