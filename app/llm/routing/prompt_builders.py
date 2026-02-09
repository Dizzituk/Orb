# FILE: app/llm/routing/prompt_builders.py
"""
System prompt and message builders for stream routing.

v1.0 (2026-01-20): Extracted from stream_router.py for modularity.
v1.1 (2026-01-20): Added large-output truncation for command outputs.
v1.2 (2026-02-09): Added scan-aware context injection — TOC replacement + section retrieval.

This module provides:
- `build_system_prompt()` - Constructs system prompt with capability layer
- `build_messages()` - Constructs message list from history + current message
  (now scan-aware: replaces breadcrumbs with TOC, injects sections on demand)
"""

from __future__ import annotations

import logging
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


# Architecture map file path (host filesystem, read-only)
ARCHMAP_PATH = r'D:\Orb\.architecture\ARCHITECTURE_MAP.md'


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


# =============================================================================
# SCAN-AWARE ARCHITECTURE MAP HELPERS
# =============================================================================

def _try_read_archmap() -> Optional[str]:
    """Read architecture map file. Returns content or None on any error."""
    path = Path(ARCHMAP_PATH)
    try:
        if not path.exists():
            return None
        return path.read_text(encoding='utf-8', errors='ignore')
    except (IOError, OSError) as e:
        logger.debug(f"[arch_inject] Could not read architecture map: {e}")
        return None


def _parse_archmap_sections(text: str) -> List[Dict[str, Any]]:
    """
    Parse architecture map sections using ## <number>. pattern.
    Returns list of dicts with: number, title, start, end.
    """
    pattern = r'(?m)^##\s+(\d+)\.\s*(.*?)\s*$'
    matches = list(re.finditer(pattern, text))
    
    sections = []
    for i, match in enumerate(matches):
        number = int(match.group(1))
        title = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        
        sections.append({
            'number': number,
            'title': title,
            'start': start,
            'end': end,
        })
    
    return sections


def _build_toc(sections: List[Dict], target_chars: int = 500) -> str:
    """Build compact TOC from sections, truncating to target_chars."""
    if not sections:
        return "[architecture_scan] Scan completed (no sections parsed)"
    
    parts = [f"{s['number']}. {s['title']}" for s in sections]
    toc = "Architecture scan available — sections: " + ", ".join(parts)
    toc += "\n\nAsk about any section by number or name (e.g. 'tell me about section 12')."
    
    if len(toc) > target_chars:
        # Truncate to fit, keep whole section entries
        truncated = toc[:target_chars - 4]
        last_comma = truncated.rfind(',')
        if last_comma > 0:
            toc = truncated[:last_comma] + ", …\n\nAsk about any section by number or name."
        else:
            toc = truncated + " …"
    
    return toc


def _detect_requested_sections(user_text: str, sections: List[Dict]) -> List[Dict]:
    """
    Detect which sections user is referencing.
    
    Supports:
    - Explicit: "section 12", "section 3"
    - Numbered: "tell me about 12.", "what's in 25"
    - Title keywords: "observations", "dependency graph", "patterns"
    
    Returns list of matching section dicts (may be empty).
    """
    matches = []
    
    # 1. Explicit "section N" references
    explicit_pattern = r'(?i)\bsection\s+(\d+)\b'
    for match in re.finditer(explicit_pattern, user_text):
        section_num = int(match.group(1))
        for sec in sections:
            if sec['number'] == section_num and sec not in matches:
                matches.append(sec)
                break
    
    # 2. Standalone number references like "tell me about 12" or "25."
    number_pattern = r'(?i)(?:about|regarding|explain|describe|show|what.s in)\s+(\d+)\.?\b'
    for match in re.finditer(number_pattern, user_text):
        section_num = int(match.group(1))
        for sec in sections:
            if sec['number'] == section_num and sec not in matches:
                matches.append(sec)
                break
    
    # 3. Title keyword matching (only if no explicit matches yet)
    if not matches:
        user_lower = user_text.lower()
        for sec in sections:
            title_lower = sec['title'].lower()
            # Match if significant words (>3 chars) from title appear in user text
            title_words = [w for w in title_lower.split() if len(w) > 3]
            if title_words and any(word in user_lower for word in title_words):
                matches.append(sec)
    
    return matches


# =============================================================================
# CORE PROMPT BUILDERS
# =============================================================================

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
    v1.2: Scan-aware context injection:
      - Replaces [architecture_scan] breadcrumbs with lightweight TOC
      - Injects relevant section content when user references a section
    
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
    SECTION_SOFT_CAP = 3_500
    
    messages_list = []
    has_scan_breadcrumb = False
    
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
                
                # v1.2: Detect architecture scan breadcrumb and replace with TOC
                if (msg.role == "assistant" and 
                    isinstance(content, str) and 
                    content.startswith("[architecture_scan]")):
                    
                    has_scan_breadcrumb = True
                    arch_text = _try_read_archmap()
                    if arch_text:
                        sections = _parse_archmap_sections(arch_text)
                        if sections:
                            content = _build_toc(sections)
                            logger.info(f"[arch_inject] Replaced breadcrumb with TOC ({len(sections)} sections)")
                        else:
                            logger.debug("[arch_inject] Architecture map found but no sections parsed")
                    else:
                        logger.debug(f"[arch_inject] Architecture map not readable, keeping breadcrumb")
                
                # Apply truncation if needed (TOC is small, won't trigger this)
                elif (msg.role == "assistant" and 
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
    
    # v1.2: Inject section content if user references a section
    if has_scan_breadcrumb:
        arch_text = _try_read_archmap()
        if arch_text:
            sections = _parse_archmap_sections(arch_text)
            if sections:
                requested = _detect_requested_sections(message, sections)
                if requested:
                    # Inject first matched section only (prevent context bloat)
                    sec = requested[0]
                    section_content = arch_text[sec['start']:sec['end']]
                    
                    # Soft cap at 3500 chars
                    if len(section_content) > SECTION_SOFT_CAP:
                        section_content = section_content[:3000] + "\n\n[...SECTION TRUNCATED...]\n"
                    
                    injection = {
                        "role": "system",
                        "content": (
                            f"Architecture scan context — Section {sec['number']}: {sec['title']}\n\n"
                            f"{section_content}"
                        )
                    }
                    messages_list.append(injection)
                    logger.info(
                        f"[arch_inject] Injected section {sec['number']} "
                        f"({len(section_content)} chars) for current request"
                    )
    
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
