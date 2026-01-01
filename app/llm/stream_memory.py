# FILE: app/llm/stream_memory.py
"""
Memory injection helper for stream_router.

Provides a simple function to inject ASTRA memory context
into system prompts for streaming endpoints.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Try to import memory system
try:
    from app.llm.routing.memory_injection import (
        build_memory_context,
        inject_memory_into_system_prompt,
        MEMORY_AVAILABLE,
    )
    _MEMORY_INJECTION_AVAILABLE = MEMORY_AVAILABLE
except ImportError:
    _MEMORY_INJECTION_AVAILABLE = False
    logger.warning("[stream_memory] Memory injection not available")


def inject_memory_for_stream(
    db: Session,
    messages: List[Dict[str, Any]],
    system_prompt: str,
    job_type: Optional[str] = None,
) -> str:
    """
    Inject memory context into system prompt for streaming.
    
    Args:
        db: Database session
        messages: Conversation messages
        system_prompt: Current system prompt
        job_type: Optional job type for preference filtering
        
    Returns:
        System prompt with memory injected (or original if unavailable)
    """
    if not _MEMORY_INJECTION_AVAILABLE:
        return system_prompt
    
    try:
        memory_context = build_memory_context(
            db=db,
            messages=messages,
            job_type=job_type,
            component="stream_router",
        )
        
        if not memory_context.is_empty():
            logger.info(
                f"[stream_memory] Injected: depth={memory_context.depth} "
                f"prefs={len(memory_context.preferences_applied)} "
                f"records={memory_context.records_retrieved}"
            )
            return inject_memory_into_system_prompt(system_prompt, memory_context)
        
        return system_prompt
        
    except Exception as e:
        logger.warning(f"[stream_memory] Injection failed: {e}")
        return system_prompt


__all__ = ["inject_memory_for_stream", "_MEMORY_INJECTION_AVAILABLE"]
