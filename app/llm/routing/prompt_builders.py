# FILE: app/llm/routing/prompt_builders.py
"""
System prompt and message builders for stream routing.

v1.0 (2026-01-20): Extracted from stream_router.py for modularity.
v1.1 (2026-01-20): Added large-output truncation for command outputs.
v1.2 (2026-02-04): Added architecture map injection for section-based context retrieval.

This module provides:
- `build_system_prompt()` - Constructs system prompt with capability layer
- `build_messages()` - Constructs message list from history + current message
- `inject_architecture_sections()` - Injects relevant architecture sections into message list
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Any, Dict

from sqlalchemy.orm import Session

from app.memory import service as memory_service

from .handler_registry import (
    _CAPABILITIES_AVAILABLE,
    get_capability_context,
)

logger = logging.getLogger(__name__)


# Default path to architecture map file
_ARCHITECTURE_MAP_PATH = r'D:\Orb\.architecture\ARCHITECTURE_MAP.md'


# v5.0 (2026-02-04): CONVERSATIONAL MODE GUIDELINES
# The baseline/chat LLM must behave as a conversational assistant, NOT
# a code generator. It should clarify, ask questions, and build context.
# The downstream pipeline (Weaver → SpecGate → CriticalPipeline) handles
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


def _load_architecture_map(map_path: Optional[str] = None) -> Optional[str]:
    """
    Load architecture map file content.
    
    Args:
        map_path: Optional custom path. If None, uses _ARCHITECTURE_MAP_PATH.
    
    Returns:
        File content as string, or None if file not found/error.
    """
    path = Path(map_path or _ARCHITECTURE_MAP_PATH)
    
    if not path.exists():
        logger.debug(f"[arch_inject] Architecture map not found: {path}")
        return None
    
    try:
        content = path.read_text(encoding='utf-8')
        logger.debug(f"[arch_inject] Loaded architecture map: {len(content)} chars")
        return content
    except Exception as e:
        logger.warning(f"[arch_inject] Failed to read architecture map: {e}")
        return None


def _extract_section_by_title(content: str, title: str) -> Optional[str]:
    """
    Extract a section from markdown content by exact title match.
    
    Handles both ATX-style (#) and Setext-style (===) headers.
    Section ends at next header of same or higher level, or EOF.
    
    Args:
        content: Full markdown content
        title: Exact section title to find (case-insensitive)
    
    Returns:
        Section content including header, or None if not found.
    """
    lines = content.split('\n')
    title_lower = title.lower().strip()
    
    section_start = None
    section_level = None
    
    # Find section start
    for i, line in enumerate(lines):
        # ATX-style: # Title
        atx_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if atx_match:
            level = len(atx_match.group(1))
            header_text = atx_match.group(2).strip().lower()
            if header_text == title_lower:
                section_start = i
                section_level = level
                break
        
        # Setext-style: Title\n===
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if re.match(r'^=+$', next_line):
                if line.strip().lower() == title_lower:
                    section_start = i
                    section_level = 1
                    break
            elif re.match(r'^-+$', next_line):
                if line.strip().lower() == title_lower:
                    section_start = i
                    section_level = 2
                    break
    
    if section_start is None:
        return None
    
    # Find section end
    section_end = len(lines)
    for i in range(section_start + 1, len(lines)):
        line = lines[i]
        
        # ATX-style header
        atx_match = re.match(r'^(#{1,6})\s+', line)
        if atx_match:
            level = len(atx_match.group(1))
            if level <= section_level:
                section_end = i
                break
        
        # Setext-style header
        if i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if re.match(r'^[=-]+$', next_line):
                # Setext = level 1, - = level 2
                setext_level = 1 if '=' in next_line else 2
                if setext_level <= section_level:
                    section_end = i
                    break
    
    section_content = '\n'.join(lines[section_start:section_end])
    return section_content.strip()


def _detect_section_references(message: str) -> List[str]:
    """
    Detect section reference patterns in user message.
    
    Patterns supported:
    - "what does [Section Title] say"
    - "check the [Another Section] section"
    - "according to [Foo Bar]"
    - Bracket notation alone: [Section Name]
    
    Args:
        message: User message text
    
    Returns:
        List of detected section titles (deduplicated, case-preserved)
    """
    patterns = [
        r'\[([^\]]+)\]\s+(?:say|section|mention|describe|explain)',
        r'(?:what|how|check|see|read|according to|from)\s+(?:the\s+)?\[([^\]]+)\]',
        r'\[([^\]]+)\]',
    ]
    
    found = []
    for pattern in patterns:
        matches = re.finditer(pattern, message, re.IGNORECASE)
        for match in matches:
            title = match.group(1).strip()
            if title and title not in found:
                found.append(title)
    
    return found


def inject_architecture_sections(
    messages: List[Dict[str, str]],
    map_path: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Inject relevant architecture sections before the final user message.
    
    Scans the last user message for section references like [Section Title].
    If found, extracts matching sections from ARCHITECTURE_MAP.md and injects
    them as system messages immediately before the user message.
    
    Args:
        messages: Original message list (will be copied, not modified)
        map_path: Optional custom architecture map path
    
    Returns:
        New message list with injected sections, or original if no refs found
    """
    if not messages:
        return messages
    
    # Work on a copy
    messages = messages.copy()
    
    # Find last user message
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get('role') == 'user':
            last_user_idx = i
            break
    
    if last_user_idx is None:
        return messages
    
    user_message = messages[last_user_idx]['content']
    
    # Detect section references
    section_refs = _detect_section_references(user_message)
    if not section_refs:
        logger.debug("[arch_inject] No section references detected")
        return messages
    
    logger.info(f"[arch_inject] Detected section references: {section_refs}")
    
    # Load architecture map
    arch_content = _load_architecture_map(map_path)
    if not arch_content:
        logger.warning("[arch_inject] Architecture map not available, skipping injection")
        return messages
    
    # Extract sections
    injected_sections = []
    for title in section_refs:
        section = _extract_section_by_title(arch_content, title)
        if section:
            logger.info(f"[arch_inject] Extracted section '{title}': {len(section)} chars")
            injected_sections.append({
                'title': title,
                'content': section
            })
        else:
            logger.warning(f"[arch_inject] Section '{title}' not found in architecture map")
    
    if not injected_sections:
        logger.info("[arch_inject] No matching sections found in architecture map")
        return messages
    
    # Build injection message
    injection_parts = [
        "=== ARCHITECTURE CONTEXT (INJECTED) ===",
        "",
        "The following sections from the architecture map are relevant to your query:",
        ""
    ]
    
    for i, sec in enumerate(injected_sections, 1):
        injection_parts.append(f"--- Section {i}: {sec['title']} ---")
        injection_parts.append("")
        injection_parts.append(sec['content'])
        injection_parts.append("")
    
    injection_parts.append("=== END ARCHITECTURE CONTEXT ===")
    injection_content = '\n'.join(injection_parts)
    
    # Insert before last user message
    injection_msg = {
        'role': 'system',
        'content': injection_content
    }
    
    messages.insert(last_user_idx, injection_msg)
    
    logger.info(f"[arch_inject] Injected {len(injected_sections)} section(s) ({len(injection_content)} chars)")
    
    return messages


__all__ = [
    "build_system_prompt",
    "build_messages",
    "build_full_context",
    "inject_architecture_sections",
]