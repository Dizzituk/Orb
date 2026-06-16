# FILE: app/llm/routing/prompt_builders.py
# Purpose: System prompt and message builders for stream routing.
# Called-by: app.bridge.capability_layer, app.endpoints.chat_attachments, app.llm.routing.chat_routing, app.llm.routing.image_chat_routing
# Depends-on: app.core_principles, app.llm.astra_filesystem_block, app.llm.context, app.llm.routing._prompt_blocks (+12 more)
# Last-renovated: 2026-06-11
"""
System prompt and message builders for stream routing.

v1.0 (2026-01-20): Extracted from stream_router.py for modularity.
v1.1 (2026-01-20): Added large-output truncation for command outputs.
v1.2 (2026-02-09): Added scan-aware context injection — TOC replacement + section retrieval.
v1.3 (2026-05-01): Extracted CONVERSATIONAL_GUIDELINES and IMAGE_GEN_MARKER_INSTRUCTIONS
    into sibling module ._prompt_blocks to bring this file under the 30KB hard limit.
    Same module also rewrites CONVERSATIONAL_GUIDELINES to fix the "Yes —" /
    "Yeah —" / "Got it —" opener tic and add explicit register guidance.

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

from app.core_principles import get_principles_block as _get_principles
from .handler_registry import (
    _CAPABILITIES_AVAILABLE,
    get_capability_context,
)

# Prompt string blocks live in a sibling module to keep this file under
# the 30KB hard limit. CONVERSATIONAL_GUIDELINES is injected into every
# chat-mode system prompt; IMAGE_GEN_MARKER_INSTRUCTIONS is injected only
# when the chat router detected image-gen intent.
from ._prompt_blocks import (
    CONVERSATIONAL_GUIDELINES as _CONVERSATIONAL_GUIDELINES,
    IMAGE_GEN_MARKER_INSTRUCTIONS as _IMAGE_GEN_MARKER_INSTRUCTIONS,
    LOCAL_FILES_DISCIPLINE as _LOCAL_FILES_DISCIPLINE,
)
from ._prompt_blocks_action import ACT_ON_TASKS as _ACT_ON_TASKS
# v17.0 (2026-05-20): Always inject current date/time/timezone so the LLM
# knows when "now" is. Wired into every chat-mode system prompt.
from app.llm.context import get_system_context as _get_system_context


logger = logging.getLogger(__name__)


# Architecture map file path (host filesystem, read-only)
ARCHMAP_PATH = r'D:\Orb\.architecture\ARCHITECTURE_MAP.md'


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

# =============================================================================
# UI CONTEXT DESCRIPTIONS — Injected into system prompt so the LLM knows
# which tab/view the user is currently looking at.
# =============================================================================

_UI_CONTEXT_DESCRIPTIONS: dict[str, str] = {
    "investments": (
        "The user is currently viewing the INVESTMENTS dashboard. "
        "This tab shows their live portfolio positions, allocation chart, growth history, "
        "and curated news feed. The data is RIGHT THERE on their screen. "
        "CRITICAL: If they ask about investments, ANSWER DIRECTLY using portfolio data "
        "provided below. Do NOT ask them to paste holdings, confirm formats, choose "
        "metrics, or pick time windows. Just answer with the data you have. "
        "If data is missing from context, say so briefly and offer to check."
    ),
    "accounts": (
        "The user is currently viewing the ACCOUNTS / FINANCE tab. "
        "This shows bank transactions, spending categories, tax summaries, "
        "and QuickBooks integration. Answer finance questions directly."
    ),
    "content": (
        "The user is currently viewing the CONTENT tab. "
        "This is the content creation pipeline — style references, video creation, "
        "image creation, blog creation, and content output/publishing. "
        "They may be inside a specific content project workspace."
    ),
    "debug": (
        "The user is currently viewing the DEBUG tab. "
        "This provides direct conversational access to the codebase, logs, "
        "pipeline state, and diagnostic tools. Respond technically."
    ),
    "project_builds": (
        "The user is currently viewing the PROJECT BUILDS tab. "
        "This is where ASTRA's build pipeline runs — Weaver, SpecGate, "
        "Critical Pipeline, Overwatcher, Implementer. They may be looking "
        "at build progress, specs, or architecture for a specific project."
    ),
    "health_fitness": (
        "The user is currently viewing the HEALTH & FITNESS tab. "
        "This covers workout programming, nutrition, and progress tracking."
    ),
    "social_media": (
        "The user is currently viewing the SOCIAL MEDIA tab. "
        "This covers scheduling, analytics, cross-platform posting."
    ),
    "website": (
        "The user is currently viewing the WEBSITE tab. "
        "This covers the client-facing website builder and CMS."
    ),
    "education": (
        "The user is currently viewing the EDUCATION tab. "
        "This covers learning paths, course tracking, skill development."
    ),
    "settings": (
        "The user is currently viewing the SETTINGS page. "
        "They may have questions about API keys, voice settings, or configuration."
    ),
}


def _build_ui_context_block(ui_context: Any) -> str:
    """Build a system prompt section describing the user's current UI context."""
    if ui_context is None:
        return ""

    view_type = getattr(ui_context, 'view_type', None)
    job_type = getattr(ui_context, 'job_type', None)
    label = getattr(ui_context, 'label', None)

    if view_type == 'job' and job_type:
        desc = _UI_CONTEXT_DESCRIPTIONS.get(job_type, "")
        block = f"\n\n## CURRENT UI CONTEXT\nThe user is in the '{label or job_type}' tab."
        if desc:
            block += f"\n{desc}"
        block += (
            "\nRespond with awareness of what they can see on screen. "
            "Reference data and features available in this tab directly — "
            "do not ask the user to provide information that the tab already shows."
        )
        return block
    elif view_type == 'settings':
        desc = _UI_CONTEXT_DESCRIPTIONS.get('settings', '')
        return f"\n\n## CURRENT UI CONTEXT\n{desc}"
    else:
        return ""


def build_system_prompt(
    project: Any,
    full_context: str,
    ui_context: Any = None,
    image_intent: bool = False,
) -> str:
    """
    Build system prompt with project context and ASTRA capability layer.
    
    v4.9: Injects capability layer at the top of every system prompt.
    v5.0: Adds conversational guidelines to prevent code dumping.
    v6.0: Injects UI context from Universal Chat Panel.
    v16.0 (2026-05-01): Adds image-gen marker instructions when image_intent
         is set. Replaces the old Gemini synth bypass — chat LLM now writes
         the gpt-image-2 prompt itself, with full conversation context.
    
    Args:
        project: Project ORM object with name and description
        full_context: Pre-built context string (semantic, documents, etc.)
        ui_context: Optional UIContext from chat panel (which tab user is viewing)
        image_intent: If True, inject image-gen marker instructions so the
            chat LLM emits [IMAGE_PROMPT]: <prompt> for direct dispatch.
    
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
    
    # v1.1: Inject core engineering principles (file size, modularity, etc.)
    system_prompt += '\n\n' + _get_principles()

    # v5.0: Add conversational guidelines
    system_prompt += _CONVERSATIONAL_GUIDELINES

    # v18.0 (2026-05-24): Local files discipline — tells the LLM where the
    # user's files actually live (OneDrive paths) and to list_dir before
    # writing. Mirrors the image-turn path discipline so text-only
    # follow-ups (e.g. "add X to the dashboard") behave consistently.
    system_prompt += _LOCAL_FILES_DISCIPLINE

    # v20.0 (2026-05-30): Bias-to-action override. Appended AFTER the scoping
    # guidance so it wins for tool-backed tasks (stop re-asking answered
    # questions / pausing mid-batch). See _prompt_blocks_action.
    system_prompt += _ACT_ON_TASKS

    # v19.0 (2026-05-24): Working-set context block.  Injects the list
    # of files this project's conversation has touched, with cached
    # contents inlined for the most-recently-used ones.  Cross-model
    # handoff: GPT-mini and Gemini both see the same set of files
    # without needing tool calls to rediscover them.
    try:
        from app.memory.working_set import build_context_block as _ws_block
        _ws_project_id = getattr(project, "id", None)
        if _ws_project_id:
            _block = _ws_block(_ws_project_id)
            if _block:
                system_prompt += _block
                print(f"[PROMPT_BUILDER] Working-set block injected ({len(_block)} chars)")
    except Exception as _ws_err:
        print(f"[PROMPT_BUILDER] Working-set inject failed (non-fatal): {_ws_err}")
    
    # v6.0: Inject UI context from Universal Chat Panel
    ui_block = _build_ui_context_block(ui_context)
    if ui_block:
        system_prompt += ui_block

    # v16.0: Inject image marker instructions when an image was requested.
    # This is what tells the chat LLM to emit [IMAGE_PROMPT]: ... so the
    # stream wrapper can fire it directly at gpt-image-2.
    if image_intent:
        system_prompt += _IMAGE_GEN_MARKER_INSTRUCTIONS
        print("[PROMPT_BUILDER] Image-gen marker instructions injected")
    
    # v17.0: Prepend current date/time/timezone before capabilities so the
    # LLM always knows when "now" is. ASTRA's inbuilt calendar/clock.
    datetime_header = "## SYSTEM CONTEXT\n" + _get_system_context()

    # Combine: datetime first, then capabilities, then project context
    if capability_layer:
        return f"{datetime_header}\n\n{capability_layer}\n\n{system_prompt}"
    return f"{datetime_header}\n\n{system_prompt}"


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
    
    # v10.0: Reduce raw history when a conversation summary is available.
    # Summary provides the broader context; raw messages provide recency.
    SUMMARY_HISTORY_LIMIT = 8  # When summary exists, only load last N messages
    
    messages_list = []
    has_scan_breadcrumb = False
    
    # v10.0: Check for conversation summary and inject it
    summary_block = ""
    try:
        from app.memory.summary_injection import build_summary_context
        summary_block = build_summary_context(db, project_id)
    except Exception as e:
        logger.debug("[prompt_builders] Summary injection skipped: %s", e)
    
    if summary_block:
        # Summary exists — it's injected via build_full_context() into the
        # system prompt (not here as a message, because Anthropic and Gemini
        # strip role="system" from message lists). We still reduce raw history
        # since the summary provides the older context.
        effective_limit = min(history_limit, SUMMARY_HISTORY_LIMIT)
        logger.info(
            "[prompt_builders] Summary available, reducing history from %d to %d",
            history_limit, effective_limit,
        )
    else:
        effective_limit = history_limit
    
    if include_history:
        try:
            history = memory_service.list_messages(db, project_id, limit=effective_limit)
            
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
    
    # v0.14.0 DISABLED (Phase 7 cleanup — 2026-04-14).
    # The old SQL PreferenceRecord biographical injection has been replaced by
    # the Tier 1 identity store block (see v15 below). The old data is still in
    # astra_preferences; garbage rows have been marked SUPERSEDED, but this
    # injector was double-injecting on top of [USER IDENTITY] so it is disabled.

    # v15 (Phase 2): Tier 1 identity block — hard facts, always injected.
    # Loaded from data/self_model/identity.json. No confidence scoring,
    # no retrieval, always present when set.
    try:
        from app.self_model.identity_format import build_identity_block
        _id_block = build_identity_block()
        if _id_block:
            full_context += "\n\n" + _id_block
            print(f"[CONTEXT] Identity block injected ({len(_id_block)} chars)")
    except Exception as _id_err:
        print(f"[CONTEXT] Identity injection failed: {_id_err}")

    # v14.1 (Phase 1): Self-model facts injected via filter.
    # Noise reduction — only HIGH-confidence project/learning, MEDIUM+
    # preferences/patterns, and ALL biographical/philosophy facts pass.
    # Set env SELF_MODEL_INJECT_ALL=1 to bypass filter for debugging.
    try:
        from app.self_model.injection import build_self_model_block
        _sm_block = build_self_model_block()
        if _sm_block:
            full_context += "\n\n" + _sm_block
            print(f"[CONTEXT] Self-model block injected ({len(_sm_block)} chars)")
        else:
            print("[CONTEXT] Self-model block empty after filter")
    except Exception as _sm_err:
        print(f"[CONTEXT] Self-model injection failed: {_sm_err}")

    # v19 (2026-06-14): Dynamic capability manifest — scan-derived self-knowledge,
    # including the architecture subsystem list from CREATE ARCHITECTURE MAP.
    # The manifest hot row is otherwise only reachable via the envelope/memory
    # path (synthesize_envelope_from_task → build_memory_context → retrieve_for_query),
    # which the STREAMING chat path does NOT use — so "what are you / what can you
    # do / tell me about yourself" in normal chat never saw it. Gate on the
    # identity_capability topic tag so it loads only for self/identity questions
    # (the manifest is ~12KB — never inject it on every turn).
    try:
        from app.astra_memory.topic_tagger import extract_tags as _self_tags
        if "identity_capability" in (_self_tags(message) or []):
            from app.self_model.capability_manifest import read_manifest
            _manifest = read_manifest()
            if _manifest:
                full_context += (
                    "\n\n## ASTRA SELF-KNOWLEDGE (live capability manifest)\n"
                    + _manifest
                )
                print(f"[CONTEXT] Capability manifest injected ({len(_manifest)} chars)")
    except Exception as _man_err:
        print(f"[CONTEXT] Capability manifest injection failed: {_man_err}")

    # v17 (2026-04-25): ASTRA filesystem facts — always-on knowledge of where
    # generated outputs live on disk. Stops the agent wasting tool calls
    # searching for files that a previous turn produced (see image_output_dir.py).
    try:
        from app.llm.astra_filesystem_block import build_filesystem_block
        _fs_block = build_filesystem_block()
        if _fs_block:
            full_context += "\n\n" + _fs_block
            print(f"[CONTEXT] Filesystem block injected ({len(_fs_block)} chars)")
    except Exception as _fs_err:
        print(f"[CONTEXT] Filesystem block injection failed: {_fs_err}")

    # v16 (Phase 7): Long-term behavioural / interest themes.
    # Only HOT themes (sustained evidence, diversified) appear here.
    # Cold themes surface via a separate surfacing path, not this block.
    try:
        from app.self_model.fragments.injection import build_fragments_block
        _fg_block = build_fragments_block()
        if _fg_block:
            full_context += "\n\n" + _fg_block
            print(f"[CONTEXT] Learned-patterns block injected ({len(_fg_block)} chars)")
    except Exception as _fg_err:
        print(f"[CONTEXT] Learned-patterns injection failed: {_fg_err}")

    # Job 0 (2026-06-15): the single memory front door (MEMORY_MAP.md §2).
    # Replaces the two separate injections that used to live here (the
    # MemoryRouter keyword block + the rolling conversation summary). The
    # front door assembles preferences + semantic facts + repo +
    # nudges/sentinel/proposals + conversation summary + within-conversation
    # recall + MemoryRouter behind the ONE D0-D4 depth gate, de-duped so a
    # fact surfaces at most once. The desktop-streaming and phone chat paths
    # reach memory context ONLY through here.
    try:
        from app.llm.routing.memory_injection import build_memory_block
        memory_block = build_memory_block(
            db, user_message=message, project_id=project_id,
        )
        if memory_block:
            full_context += "\n\n" + memory_block
    except Exception as _mem_err:
        print(f"[CONTEXT] Memory front door failed (non-fatal): {_mem_err}")

    # v18 (2026-05-01): Recent classifier-decision context.
    # When the translation layer fires a confirmation gate or auto-executes
    # an intent, the rule_name and reason are recorded by
    # app/translation/recent_decisions.py. Surfacing them here lets the
    # chat LLM answer "why did you think that was a web search?" accurately
    # instead of confabulating. Without this block the chat LLM has no
    # visibility into the classifier's reasoning.
    try:
        from app.translation.recent_decisions import build_decisions_block
        _cd_block = build_decisions_block(str(project_id))
        if _cd_block:
            full_context += "\n\n" + _cd_block
            print(
                f"[CONTEXT] Classifier decisions block injected "
                f"({len(_cd_block)} chars)"
            )
    except Exception as _cd_err:
        print(f"[CONTEXT] Classifier decisions injection failed: {_cd_err}")

    return full_context


__all__ = [
    "build_system_prompt",
    "build_messages",
    "build_full_context",
]
