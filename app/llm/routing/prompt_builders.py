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

from app.core_principles import get_principles_block as _get_principles
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
2. **ONE QUESTION AT A TIME.** This is non-negotiable. When you need to clarify
   something, ask ONE question, wait for the answer, then ask the next one.
   Never dump a numbered list of 5-10 questions. The user communicates via voice
   while driving — they cannot process or remember a batch of questions.
   Build understanding incrementally: ask, listen, absorb, ask the next thing.
   If you have multiple things to clarify, pick the MOST IMPORTANT one first.
3. **Keep responses focused and concise** - a few paragraphs maximum.
   Do not dump walls of text, architecture docs, or full file contents.
4. **Summarise your understanding** back to the user. Confirm what you think
   they want before the pipeline starts building it.
5. **Flag potential concerns** naturally: scope, complexity, ambiguity.
   But do it conversationally, not as a checklist.
6. **No numbered lists of options unless asked.** Present information
   conversationally. Instead of "Option A: ... Option B: ... Option C: ..."
   just say what you think is best and why, then ask if they agree.

## WHAT TO DO INSTEAD OF WRITING CODE

- Acknowledge the request
- Ask about unclear aspects (target platform, integration points, preferences)
- Confirm scope ("So you want X that does Y, right?")
- Mention any obvious considerations ("This will need a backend endpoint too")
- Let the user know the pipeline will handle the implementation

## CODEBASE CONTEXT

If your context includes a [CODEBASE CONTEXT] block, source files have been
pre-loaded for you from the sandbox. You already have the code. Do NOT:
- Call execute_command, shell commands, or generate tool_call JSON to explore files
- Say "let me look at the codebase" or "give me a moment to dig"
Instead, reference the loaded files directly. Cite patterns, variable names,
component structures, and CSS tokens from the code you can already see.

## EXAMPLES

GOOD: "Got it - a companion app that connects to Astra from your phone. First
thing I need to know: are you thinking Android only, or do you want iOS as well?"
[wait for answer, then ask next question]

GOOD: "Makes sense. So Android it is. Next question - when you are out on the
road, should the phone connect directly to your desktop, or go through a cloud
relay so it works even when your PC is behind a firewall?"

BAD: "Here are my 9 questions: 1. Platform? 2. Connectivity model? 3. Real-time
vs batch? 4. Speech recognition? 5. Notifications? 6. Authentication? 7. Dashboard
contents? 8. Privacy? 9. Offline behaviour?"

BAD: [generating 500 lines of React components, Python endpoints, and config files]

## FILE GENERATION

EXCEPTION to the "no code" rule: When the user explicitly asks you to CREATE
a file (HTML page, document, etc.) — for example "create me an HTML file",
"build this as a webpage", "make that into a downloadable file" — you SHOULD
generate the complete file content in a fenced code block.

Rules for file generation:
- Output the COMPLETE file in a single `html (or appropriate language) code block
- Make it self-contained (inline CSS/JS, no external dependencies except CDN fonts)
- Do NOT ask where to save it — the system handles file extraction automatically
- Do NOT ask for confirmation before generating — if they asked for a file, make it
- Be creative and thorough — this is your chance to show what you can build
- The system will automatically extract HTML from your response, save it to disk,
  and present it to the user as a clickable file they can open in their browser
- Keep your surrounding message SHORT — a brief sentence or two before the code block
  explaining what you built, then the code block, then done. The user does NOT want to
  scroll through 200 lines of HTML in chat — they will open the file via the download
  card that appears automatically. Do not describe the code after the block either.

## VISUAL CONTENT IN HTML PAGES

When creating HTML pages (websites, blog posts, interactive articles), you are ENCOURAGED
to include rich visual content to make pages more engaging and immersive:

### Images
- Use CSS art, SVG graphics, and CSS gradients for decorative visuals (these are self-contained)
- For concept illustrations and hero images, create detailed SVG inline graphics
- Use CSS animations and @keyframes for animated visual elements
- For placeholder images where a real photo would go, use solid gradient backgrounds
  with descriptive overlay text explaining what the image would show

### Animated Elements (GIF-like effects)
- Use CSS animations (@keyframes) to create animated visuals: pulsing orbs, flowing
  gradients, particle effects, rotating elements, scroll-triggered reveals
- Use SVG animations (animateTransform, animate) for animated diagrams and illustrations
- Create "living" page elements: animated backgrounds, breathing glows, flowing lines,
  orbiting nodes, progress animations
- These CSS/SVG animations serve the same purpose as GIFs but are self-contained,
  resolution-independent, and much smaller in file size

### Interactive Visualisations
- Use JavaScript + Canvas or SVG for interactive charts, timelines, and diagrams
- Add scroll-triggered animations that reveal content as the user reads
- Include toggle buttons that switch between different data views
- Create hover effects that reveal additional information

### Design Philosophy
- Every section of a blog or webpage should have VISUAL INTEREST — not just text
- Alternate between text sections and visual/interactive sections
- Use the page's colour palette consistently across all generated visuals
- Animated elements should be subtle and purposeful, not distracting
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


def build_system_prompt(project: Any, full_context: str, ui_context: Any = None) -> str:
    """
    Build system prompt with project context and ASTRA capability layer.
    
    v4.9: Injects capability layer at the top of every system prompt.
    v5.0: Adds conversational guidelines to prevent code dumping.
    v6.0: Injects UI context from Universal Chat Panel.
    
    Args:
        project: Project ORM object with name and description
        full_context: Pre-built context string (semantic, documents, etc.)
        ui_context: Optional UIContext from chat panel (which tab user is viewing)
    
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
    
    # v6.0: Inject UI context from Universal Chat Panel
    ui_block = _build_ui_context_block(ui_context)
    if ui_block:
        system_prompt += ui_block
    
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
    
    # v0.14.0: Always inject biographical preferences into context.
    # These are permanent facts about the user (name, location, job, etc.)
    # that should be available to every model in every session.
    try:
        from app.db import get_db_session as _get_bio_db
        from app.astra_memory.preference_models import PreferenceRecord, RecordStatus
        _bio_db = _get_bio_db()
        try:
            _bio_prefs = (
                _bio_db.query(PreferenceRecord)
                .filter(
                    PreferenceRecord.preference_key.like("doc_extract:biographical:%"),
                    PreferenceRecord.status == RecordStatus.ACTIVE,
                )
                .all()
            )
            if _bio_prefs:
                _bio_lines = ["[USER PROFILE]"]
                for _bp in _bio_prefs:
                    _key_short = _bp.preference_key.replace("doc_extract:biographical:", "")
                    _bio_lines.append(f"  {_key_short}: {_bp.preference_value}")
                _bio_lines.append("[/USER PROFILE]")
                full_context += "\n\n" + "\n".join(_bio_lines)
                print(f"[CONTEXT] User profile injected: {len(_bio_prefs)} biographical facts")
            else:
                print("[CONTEXT] No biographical prefs found in DB")
        finally:
            _bio_db.close()
    except Exception as _bio_err:
        print(f"[CONTEXT] Biographical injection failed: {_bio_err}")

    # v5.4: RAG memory injection from unified memory router
    try:
        from app.memory.integration import inject_memory_context
        memory_ctx = inject_memory_context(
            query=message,
            project_id=str(project_id),
            limit=10,
        )
        if memory_ctx:
            full_context += "\n\n" + memory_ctx
    except Exception:
        pass  # Non-fatal — memory system may not be initialised yet
    
    # v10.0: Conversation summary injection into system context.
    # Injected here (not in build_messages) because Anthropic and Gemini
    # strip role="system" from the message list. The system prompt is the
    # one place all three providers reliably receive context.
    try:
        from app.memory.summary_injection import build_summary_context
        summary_ctx = build_summary_context(db, project_id)
        if summary_ctx:
            full_context += "\n\n" + summary_ctx
    except Exception:
        pass  # Non-fatal

    return full_context


__all__ = [
    "build_system_prompt",
    "build_messages",
    "build_full_context",
]
