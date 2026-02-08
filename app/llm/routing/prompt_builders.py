# FILE: app/llm/routing/prompt_builders.py
"""
System prompt and message builders for stream routing.

v1.0 (2026-01-20): Extracted from stream_router.py for modularity.
v1.1 (2026-01-20): Added large-output truncation for command outputs.

This module provides:
- `build_system_prompt()` - Constructs system prompt with capability layer
- `build_messages()` - Constructs message list from history + current message
"""

from __future__ import annotations

import logging
from typing import List, Optional, Any, Dict

from sqlalchemy.orm import Session

from app.memory import service as memory_service

from .handler_registry import (
    _CAPABILITIES_AVAILABLE,
    get_capability_context,
)

logger = logging.getLogger(__name__)


# v5.0 (2026-02-04): CONVERSATIONAL MODE GUIDELINES
# The baseline/chat LLM must behave as a conversational assistant, NOT
# a code generator. It should clarify, ask questions, and build context.
# The downstream pipeline (Weaver  SpecGate  CriticalPipeline) handles
# the actual implementation work.
_CONVERSATIONAL_GUIDELINES = """

## YOUR ROLE IN THE PIPELINE

You are the **conversational front-end** of a multi-stage development pipeline.
Your job is to UNDERSTAND what the user wants through natural dialogue.
You are NOT responsible for implementation - that happens in later pipeline stages.

## CRITICAL BEHAVIOUR RULES

1. **DO NOT write code or implementation files** unless the user explicitly asks
   you to write specific code right now. Your role is conversation, not generation.
2. **Ask clarifying questions** when the request is ambiguous or underspecified.
   Build understanding through dialogue before anything gets built.
3. **Keep responses focused and concise** - a few paragraphs maximum.
   Do not dump walls of text, architecture docs, or full file contents.
4. **Summarise your understanding** back to the user. Confirm what you think
   they want before the pipeline starts building it.
5. **Flag potential concerns** naturally: scope, complexity, ambiguity.
   But do it conversationally, not as a checklist.

## WHAT TO DO INSTEAD OF WRITING CODE

- Acknowledge the request
- Ask about unclear aspects (target platform, integration points, preferences)
- Confirm scope ("So you want X that does Y, right?")
- Mention any obvious considerations ("This will need a backend endpoint too")
- Let the user know the pipeline will handle the implementation

## EXAMPLES

GOOD: "Got it - you want push-to-talk voice input for the desktop app. A couple
of quick questions: should this use a cloud speech-to-text service like OpenAI
Whisper, or do you want it fully local? And where in the UI should the button go?"

BAD: [generating 500 lines of React components, Python endpoints, and config files]
"""


def build_system_prompt(project: Any, full_context: str) -> str:
    """
    Build system prompt with project context and ASTRA capability layer.
    
    v4.9: Injects capability layer at the top of every system prompt.
    v5.0: Adds conversational guidelines to prevent code dumping.
    
    Args:
        project: Project ORM object with name and description
        full_context: Pre-built context string (semantic, documents, etc.)
    
    Returns:
        Complete system prompt string
    """
    # Start with ASTRA capability layer
    capability_layer = ""
    if _CAPABILITIES_AVAILABLE and get_capability_context:
        try:
            capability_layer = get_capability_context()
        except Exception as e:
            print(f"[CAPABILITY_INJECTION] Error getting capability context: {e}")
    
    # Build project context
    system_prompt = f"Project: {project.name}."
    if project.description:
        system_prompt += f" {project.description}"
    if full_context:
        system_prompt += f"\n\nYou have access to the following context:\n\n{full_context}"
    
    # v5.0: Add conversational guidelines
    system_prompt += _CONVERSATIONAL_GUIDELINES
    
    # Combine: capabilities first, then project context
    if capability_layer:
        return f"{capability_layer}\n\n{system_prompt}"
    return system_prompt


def build_messages(
    message: str,
    project_id: int,
    db: Session,
    include_history: bool = True,
    history_limit: int = 20,
) -> List[Dict[str, str]]:
    """
    Build message list from history + current message.
    
    v1.1: Added large-output truncation for command outputs.
    Large assistant messages (>10k chars) are truncated to prevent
    context exhaustion, except for the most recent K=2 large messages
    which are kept in full.
    
    Args:
        message: Current user message
        project_id: Project ID for history lookup
        db: Database session
        include_history: Whether to include conversation history
        history_limit: Max number of history messages to include
    
    Returns:
        List of message dicts with role and content
    """
    LARGE_THRESHOLD = 10_000
    KEEP_RECENT_FULL = 2
    TRUNCATE_HEAD = 8_000
    TRUNCATE_TAIL = 1_000
    
    messages_list = []
    
    if include_history:
        try:
            history = memory_service.list_messages(db, project_id, limit=history_limit)
            
            # Identify large assistant messages
            large_assistant_msgs = [
                msg for msg in history 
                if msg.role == "assistant" and len(msg.content) > LARGE_THRESHOLD
            ]
            
            # Most recent K large messages get full content
            keep_full_ids = {
                msg.id for msg in large_assistant_msgs[-KEEP_RECENT_FULL:]
            }
            
            # Convert to LLM format with truncation
            for msg in history:
                content = msg.content
                
                # Apply truncation if needed
                if (msg.role == "assistant" and 
                    len(content) > LARGE_THRESHOLD and 
                    msg.id not in keep_full_ids):
                    
                    head = content[:TRUNCATE_HEAD]
                    tail = content[-TRUNCATE_TAIL:]
                    marker = (
                        "\n\n[...TRUNCATED: Large command output. "
                        "Ask to retrieve specific sections if needed...]\n\n"
                    )
                    content = head + marker + tail
                
                messages_list.append({
                    "role": msg.role,
                    "content": content
                })
        except Exception as e:
            logger.warning(f"[prompt_builders] Failed to load history: {e}")
    
    messages_list.append({"role": "user", "content": message})
    return messages_list


def build_full_context(
    db: Session,
    project_id: int,
    message: str,
    use_semantic_search: bool = True,
) -> str:
    """
    Build full context string from multiple sources.
    
    Args:
        db: Database session
        project_id: Project ID
        message: Current message (for semantic search)
        use_semantic_search: Whether to include semantic search results
    
    Returns:
        Combined context string
    """
    # Import here to avoid circular imports
    from app.llm.stream_utils import (
        build_context_block,
        build_document_context,
        get_semantic_context,
    )
    
    context_block = build_context_block(db, project_id)
    semantic_context = get_semantic_context(db, project_id, message) if use_semantic_search else ""
    doc_context = build_document_context(db, project_id)
    
    full_context = ""
    if context_block:
        full_context += context_block + "\n\n"
    if semantic_context:
        full_context += semantic_context + "\n\n"
    if doc_context:
        full_context += "=== UPLOADED DOCUMENTS ===" + doc_context
    
    return full_context


__all__ = [
    "build_system_prompt",
    "build_messages",
    "build_full_context",
]