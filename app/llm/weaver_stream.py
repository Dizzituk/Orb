# FILE: app/llm/weaver_stream.py
r"""
Weaver Stream Handler for ASTRA - SIMPLIFIED VERSION

v4.0.0 (2026-02-04): LLM-GENERATED QUESTIONS - Remove hardcoded game-design questions
- CRITICAL FIX: Removed SHALLOW_QUESTIONS dict (Tetris-era hardcoded questions)
- Removed DESIGN_JOB_INDICATORS and _is_design_job() - triggered on every feature request
- Removed SHALLOW_QUESTION_KEYWORDS and _get_shallow_questions()
- Removed questions_context injection into LLM prompt
- LLM now generates its own contextual questions based on actual gaps in user requirements
- System prompt rewritten: no game-specific examples, domain-agnostic question guidance
- Removed _detect_filled_slots(), _reconcile_filled_slots(), _add_known_requirements_section()
- Removed SLOT_AMBIGUITY_PATTERNS and SLOT_QUESTION_PATTERNS (hardcoded slot model)
- GPT-5.2 is intelligent enough to identify missing information without hardcoded menus
- Fixes: voice-to-text request no longer gets "Dark mode or light mode?" and "Arcade-style?" 

v3.10.0 (2026-02-04): REFACTOR DETECTION FIX - Pattern-based, not keyword-based
- CRITICAL FIX: "astra" was hardcoded as a refactor indicator, causing EVERY
  message mentioning the app name to be classified as a REFACTOR_TASK
- Removed all app-name-specific indicators: "astra", "orb to astra", "branding"
- Removed overly-generic indicators: "front-end ui", "frontend ui", "across", "everywhere"
- Replaced keyword matching with REFACTOR_ACTION_PATTERNS (regex-based)
- Refactor detection now requires actual rename/replace ACTION + SCOPE context
  e.g. "rename X to Y", "replace all X with Y", "find and replace across codebase"
- Prevents false positives: "Add voice-to-text to ASTRA" no longer triggers refactor mode

v3.9.0 (2026-02-01): VISION CONTEXT FLOW FIX
- CRITICAL FIX: Vision analysis from Gemini now flows through to SpecGate
- Added _is_vision_context() to detect assistant messages containing image analysis
- Changed message filter: USER messages + assistant messages with vision context
- Vision context includes: screenshot descriptions, UI element analysis, visual descriptions
- This allows SpecGate's classifier to know which matches are USER-VISIBLE UI elements
- Refactor tasks now get vision context for intelligent classification

v3.8.0 (2026-02-01): REFACTOR TASK MODE - Separate handling for rename/refactor operations
- CRITICAL FIX: Refactor tasks now bypass design job logic entirely
- Added _is_refactor_task() check that takes precedence over _is_design_job()
- Added REFACTOR_TASK_SYSTEM_PROMPT - no design questions, focused on search/replace scope
- Added "questions not needed" detection - respects user dismissal of questions
- Refactor tasks output: what, scope, search/replace terms, constraints only
- No more "Dark mode or light mode?" questions on text rename tasks
- Questions section says "none" unless user explicitly left something unclear

v3.7.0 (2026-02-01): REFACTOR INDICATOR FIX - Codebase-wide renames never micro-tasks
- FIXED: "Orb to Astra" rename was falsely classified as MICRO_FILE_TASK
- Added REFACTOR_INDICATORS list: rename, rebrand, refactor, astra, front-end ui, etc.
- Refactor check runs FIRST in _is_micro_file_task() before file indicators
- These operations need full pipeline (Weaver→SpecGate→CriticalPipeline→Implementer)
- Version marker now shows v3.7.0 in logs for verification

v3.6.1 (2026-01-30): CRITICAL FIX - Context-aware micro-task detection
- FIXED: "create a file" was falsely triggering "build verb + non-micro"
- FIXED: "on my system" was falsely matching "system" as software system
- Removed "system" and "platform" from NON_MICRO_INDICATORS (they're location context, not software)
- Added explicit file creation patterns: "create a file", "make a file", etc.
- File indicators now take priority over build verb detection
- Strengthened MICRO_TASK_SYSTEM_PROMPT to forbid ALL discovery questions
- Questions section now always says "none" except DELETE/MOVE blockers
- SpecGate handles all file discovery - Weaver should never ask about paths/locations/extensions

v3.6.0 (2026-01-23): TIGHTEN WEAVER - Blocker-Only Questions + Micro-Task Classifier
- Added MICRO_FILE_TASK classification for simple file operations (read/write/find)
- Micro tasks skip all unnecessary questions (OS, platform, desktop, exact filename)
- Added silent typo normalization (deck top -> desktop, floder -> folder, etc.)
- Blocker-only question logic: only ask when execution would truly fail
- read+write is NOT a conflict (normal reply flow)
- Only delete/move with unclear destination triggers blocker question
- Micro tasks use minimal 10-20 line output format
- Prevents over-questioning on simple file jobs

v3.5.2 (2026-01-22): SLOT RECONCILIATION PATTERN FIX
- CRITICAL FIX: Patterns now match BOTH "unspecified" AND "not specified" (LLM variance)
- Added "visual theme" patterns for look_feel detection
- Added section header detection for lines containing "unresolved ambiguities" (not just startswith)
- Enhanced logging: Shows which patterns matched/didn't match
- Debug output when no matches found to help troubleshooting
- All slot patterns now use (not\s+specified|unspecified|unclear) for consistency

v3.5.1 (2026-01-22): SLOT RECONCILIATION FIX (Question Regression)
- CRITICAL FIX: Answered questions are now removed from Unresolved/Questions sections
- Added _detect_filled_slots() - deterministic slot extraction from user messages
- Added _reconcile_filled_slots() - removes answered slots from output
- Added _add_known_requirements_section() - shows filled slots explicitly
- Slot reconciliation is DETERMINISTIC post-processing (doesn't rely on LLM compliance)
- Fixed: "Android, Dark mode, centered" now properly removes those ambiguities/questions

v3.5.0 (2026-01-22): WEAVER HARDENING + SCOPE BOUNDARY FIX
- Bug 1: Core goal detection now includes creative/project targets (game, prototype, demo, etc.)
- Bug 2: Meta-chat extraction - separates pipeline control language from product requirements
- Bug 3: Deduplication - prevents same sentence appearing in multiple sections
- Bug 4: Added execution_mode field to output schema (backward compatible)
- Bug 5: Scope boundary enforcement - Weaver stays shallow, no technical design
- Weaver now ALWAYS outputs structured outline (never conversational "need clarity" responses)
- Questions limited to 3-5 shallow framing questions max
- No framework/architecture/algorithm questions allowed

v3.4.2 (2026-01-20): DESIGN PREFERENCE HYGIENE
- Added _enforce_design_pref_hygiene() post-processor
- Design preferences section now only contains visual/UI prefs (color, layout, style)
- Functional requirements (calculations, sync, tracking, profit/pay/fuel) are filtered out
- Prevents requirement duplication across sections during UPDATE merges
- Stricter logic: ambiguous lines in Design prefs are now removed (not kept)

v3.4.1 (2026-01-20): INTENT PATTERN RECOGNITION
- Core goal detection now recognizes "I want/I need/I'd like" patterns
- Prevents false negatives on "I want a delivery tracker app" style messages
- Intent patterns require CONCRETE targets (not "something/it/thing")
- Added CONCRETE_TARGETS list for safer intent pattern matching

v3.4 (2026-01-20): HASH-BASED DELTA + PROMPT FIX
- Bug A fix: Replace index-slicing with hash-based message deduplication
- Bug B fix: Rewrite UPDATE prompt to prevent scaffold leakage
- Added _hash_message() for stable message hashing
- Added _sanitize_weaver_output() for post-processing

v3.3 (2026-01-20): ASSISTANT HALLUCINATION FIX
- CREATE mode now filters to USER messages only (same as UPDATE mode)
- Prevents Gemini/chat hallucinations from being woven into the spec
- Core goal check now only examines what the USER said, not assistant responses

v3.2 (2026-01-20): PERSISTENT PREFS + INCREMENTAL WEAVING
- Design prefs persist across weave runs (sticky prefs)
- Weave checkpoint tracks where last weave ended
- Subsequent weaves only process NEW messages (incremental/update mode)
- Questions are only asked if prefs not already confirmed

LOCKED WEAVER BEHAVIOUR (v4.0):
- Purpose: Convert human rambling into a structured job outline
- NOT a full spec builder - just a text organizer
- Reads messages to get input (the ramble)
- Does NOT persist to specs table
- Does NOT build JSON specs
- Does NOT resolve ambiguities or contradictions
- ALWAYS outputs structured outline (never conversational responses)
- LLM generates its own contextual questions based on actual gaps (no hardcoded question menus)
- NEVER asks technical questions (frameworks, algorithms, architecture)

WEAVER DECISION TREE (v3.5):
1) Gather ALL messages
   - If no messages -> stream "No conversation to weave" -> STOP
2) Extract meta-mode phrases (no code, just planning, etc.)
3) Load confirmed design prefs + woven hashes
4) Compute new user messages using hash-based dedup
   - If UPDATE mode and no new messages -> "Nothing new" -> STOP
5) Weave (ALWAYS - no core goal check blocks, just list ambiguities)
   - UPDATE mode: pass previous output + new messages to LLM
   - CREATE mode: pass all messages to LLM
   - Include execution_mode if extracted
   - Apply deduplication post-check
6) Save hashes + checkpoint + confirmed prefs
7) Sanitize output (strip any prompt leakage)
8) Stream result -> DONE
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator, Dict, List, Optional, Any, Set, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Imports with graceful fallbacks
# ---------------------------------------------------------------------------

try:
    from app.llm.audit_logger import RoutingTrace
except ImportError:
    RoutingTrace = None

# Memory service for reading conversation
try:
    from app.memory import service as memory_service
    _MEMORY_AVAILABLE = True
except ImportError:
    memory_service = None
    _MEMORY_AVAILABLE = False

# Flow state for storing output and design question state
try:
    from app.llm.spec_flow_state import (
        start_weaver_flow,
        SpecFlowStage,
        set_weaver_design_questions,
        get_weaver_design_state,
        clear_weaver_design_questions,
        get_active_flow,
        # v1.2: Persistent prefs and checkpoints
        save_confirmed_design_prefs,
        get_confirmed_design_prefs,
        save_weave_checkpoint,
        get_weave_checkpoint,
        # v1.3: Hash-based delta tracking
        save_woven_user_hashes,
        get_woven_user_hashes,
    )
    _FLOW_STATE_AVAILABLE = True
except ImportError:
    start_weaver_flow = None
    SpecFlowStage = None
    set_weaver_design_questions = None
    get_weaver_design_state = None
    clear_weaver_design_questions = None
    get_active_flow = None
    save_confirmed_design_prefs = None
    get_confirmed_design_prefs = None
    save_weave_checkpoint = None
    get_weave_checkpoint = None
    save_woven_user_hashes = None
    get_woven_user_hashes = None
    _FLOW_STATE_AVAILABLE = False

# Simple weaver function
try:
    from app.llm.weaver_simple import weave, WEAVER_SYSTEM_PROMPT, _format_messages_as_ramble
    _SIMPLE_WEAVER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"[weaver_stream] weaver_simple not available: {e}")
    _SIMPLE_WEAVER_AVAILABLE = False
    weave = None

# Import streaming functions for all providers
try:
    from app.llm.streaming import stream_openai, stream_anthropic, stream_gemini
    _STREAMING_AVAILABLE = True
except ImportError:
    try:
        from .streaming import stream_openai, stream_anthropic, stream_gemini
        _STREAMING_AVAILABLE = True
    except ImportError:
        stream_openai = None
        stream_anthropic = None
        stream_gemini = None
        _STREAMING_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _serialize_sse(data: Dict[str, Any]) -> bytes:
    """Serialize dict to SSE format."""
    return f"data: {json.dumps(data)}\n\n".encode("utf-8")


def _get_weaver_config() -> tuple[str, str]:
    """Get provider and model for weaver from environment."""
    provider = os.getenv("WEAVER_PROVIDER", "openai")
    model = os.getenv("WEAVER_MODEL", "gpt-4.1-mini")
    return provider, model


def _get_streaming_function(provider: str):
    """Get the appropriate streaming function for the provider."""
    provider_lower = provider.lower()
    if provider_lower in ("openai", "openai-compatible"):
        return stream_openai
    elif provider_lower in ("anthropic", "claude"):
        return stream_anthropic
    elif provider_lower in ("google", "gemini"):
        return stream_gemini
    else:
        logger.warning("[WEAVER] Unknown provider '%s', defaulting to OpenAI", provider)
        return stream_openai


def _is_control_message(role: str, content: str) -> bool:
    """Check if message is a control/system message to skip."""
    c = (content or "").strip()
    rl = (role or "").strip().lower()
    
    if not c:
        return True
    
    if rl == "system":
        return True
    
    # Skip command triggers
    if rl == "user":
        lc = c.lower()
        if any(lc.startswith(prefix) for prefix in [
            "astra, command:", "astra command:", "astra, cmd:", "orb, command:",
            "how does that look all together",
        ]):
            return True
    
    # Skip Weaver/Orb output messages
    if rl in ("assistant", "orb"):
        markers = (
            "🧵 weaving", "📋 spec", "📋 job description",
            "shall i send", "say yes to proceed", "⚠️ weak spots",
            "ready for spec gate", "provenance",
            "🎨 design preferences", "design preferences needed",
            "🎨 got it", "🎨 perfect",
        )
        lc = c.lower()
        if any(m in lc for m in markers):
            return True
    
    return False


def _gather_ramble_messages(db: Session, project_id: int, max_messages: int = 50) -> List[Dict[str, Any]]:
    """
    Gather recent conversation messages as the ramble input.
    
    This is the ONLY DB access Weaver does - reading its input.
    """
    if not _MEMORY_AVAILABLE or not memory_service:
        return []
    
    try:
        messages_raw = memory_service.list_messages(db, project_id, limit=max_messages)
        messages_raw = list(reversed(messages_raw))  # Chronological order
        
        messages: List[Dict[str, Any]] = []
        for msg in messages_raw:
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", "") or ""
            
            if _is_control_message(role, content):
                continue
            
            messages.append({
                "role": role,
                "content": content,
            })
        
        return messages
    except Exception as e:
        logger.error("[WEAVER] Failed to gather messages: %s", e)
        return []


def _format_ramble(messages: List[Dict[str, Any]]) -> str:
    """Format messages into a ramble text block."""
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if not content:
            continue
        speaker = "Human" if role == "user" else "Assistant"
        lines.append(f"[{speaker}]: {content}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Message Hashing (v3.4) - For durable delta tracking
# ---------------------------------------------------------------------------

def _hash_message(msg: Dict[str, Any]) -> str:
    """
    Create a stable hash for a message.
    
    Uses role + normalized content to create a short hash.
    This allows us to track which messages have been woven,
    regardless of message ordering or count drift.
    """
    role = msg.get("role", "").strip().lower()
    content = msg.get("content", "").strip()
    # Normalize whitespace for stability
    content = " ".join(content.split())
    raw = f"{role}:{content}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _hash_messages(messages: List[Dict[str, Any]]) -> Set[str]:
    """Hash a list of messages, returning a set of hashes."""
    return {_hash_message(m) for m in messages}


# ---------------------------------------------------------------------------
# Vision Context Detection (v3.9.0) - Preserve valuable assistant analysis
# ---------------------------------------------------------------------------

# Patterns that indicate an assistant message contains vision/image analysis
# These messages should NOT be filtered out even though they're from assistant
VISION_CONTEXT_PATTERNS = [
    # Screenshot/image descriptions
    r"screenshot",
    r"image shows",
    r"i can see",
    r"i see a",
    r"the image",
    r"in the picture",
    r"looking at the",
    # UI element descriptions (from vision analysis)
    r"title bar",
    r"window title",
    r"menu bar",
    r"status bar",
    r"status indicator",
    r"toolbar",
    r"heading.*says",
    r"button.*labeled",
    r"text.*reads",
    r"displays.*text",
    r"shows.*logo",
    r"cyan.*text",
    r"blue.*text",
    r"icon.*shows",
    # Visual descriptions
    r"ui shows",
    r"ui elements",
    r"visible.*elements",
    r"display shows",
    r"interface shows",
    r"window shows",
    r"window contains",
    # Color/appearance descriptions
    r"dark\s*(?:theme|mode|background)",
    r"light\s*(?:theme|mode|background)",
    r"colored.*(?:text|background|border)",
    # Position descriptions
    r"top.*(?:left|right|corner)",
    r"bottom.*(?:left|right|corner)",
    r"center of",
    r"sidebar",
    # Action analysis phrases
    r"appears to be",
    r"looks like",
    r"seems to show",
]


def _is_vision_context(content: str) -> bool:
    """
    Detect if an assistant message contains vision/image analysis.
    
    v3.9.0: Vision analysis from Gemini should NOT be filtered out.
    This context is valuable for downstream stages (SpecGate classifier)
    to understand which matches are USER-VISIBLE UI elements.
    
    Returns True if the message likely contains vision analysis.
    """
    if not content:
        return False
    
    content_lower = content.lower()
    
    # Check for vision context patterns
    for pattern in VISION_CONTEXT_PATTERNS:
        if re.search(pattern, content_lower, re.IGNORECASE):
            print(f"[WEAVER] v3.9 Vision context detected (pattern: {pattern[:30]}...)")
            return True
    
    return False


def _extract_vision_context(messages: List[Dict[str, Any]]) -> str:
    """
    Extract vision context from assistant messages for refactor tasks.
    
    v3.9.0: Returns a string describing UI elements that were identified
    from screenshot analysis. This context is passed to SpecGate.
    """
    vision_parts = []
    
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if _is_vision_context(content):
                # Extract relevant portions (first 1000 chars to avoid bloat)
                vision_parts.append(content[:1000])
    
    if vision_parts:
        return "\n\n".join(vision_parts)
    return ""


# ---------------------------------------------------------------------------
# Meta-Mode Extraction (v3.5.0 - Bug 2 fix)
# Separates pipeline control language from product requirements
# ---------------------------------------------------------------------------

META_MODE_PATTERNS = [
    r"just\s+talk\s+about\s+it",
    r"no\s+code",
    r"don'?t\s+build\s+it\s+yet",
    r"just\s+planning",
    r"only\s+discuss",
    r"ask\s+me\s+questions\s+first",
    r"before\s+coding",
    r"don'?t\s+assume\s+too\s+much",
    r"discussion\s+only",
    r"no\s+implementation",
    r"planning\s+phase",
    r"just\s+the\s+idea",
    r"for\s+now",
    r"at\s+the\s+moment",
]


def _extract_meta_mode(messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Extract meta/mode phrases from user messages (v3.5.0 - Bug 2 fix).
    
    Pipeline control language like "no code", "just talk about it" should NOT
    end up in the product spec. They are execution constraints.
    
    Returns:
        Tuple of (filtered_messages, extracted_modes)
    """
    filtered_messages = []
    extracted_modes = []
    
    for msg in messages:
        content = msg.get("content", "")
        role = msg.get("role", "user")
        
        if role == "user" and content:
            original_content = content
            for pattern in META_MODE_PATTERNS:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    matched_text = match.group(0)
                    if matched_text not in extracted_modes:
                        extracted_modes.append(matched_text)
                    # Remove the meta phrase from content
                    content = re.sub(pattern, "", content, flags=re.IGNORECASE)
            
            # Clean up any trailing punctuation artifacts
            content = re.sub(r'\s*[,.]\s*[,.]\s*', '. ', content)
            content = re.sub(r'\s+', ' ', content).strip()
            content = re.sub(r'^[,.]\s*', '', content)
            content = re.sub(r'\s*[,.]$', '', content)
            
            if content != original_content:
                print(f"[WEAVER] Extracted meta-mode phrases: {extracted_modes}")
            
            msg = {**msg, "content": content}
        
        # Only keep non-empty messages
        if msg.get("content", "").strip():
            filtered_messages.append(msg)
    
    return filtered_messages, extracted_modes


def _format_execution_mode(extracted_modes: List[str]) -> str:
    """
    Format extracted meta-mode phrases into a clean execution mode string.
    """
    if not extracted_modes:
        return ""
    
    # Normalize and deduplicate
    normalized = []
    for mode in extracted_modes:
        mode_lower = mode.lower().strip()
        if "no code" in mode_lower or "don't build" in mode_lower:
            if "No coding yet" not in normalized:
                normalized.append("No coding yet")
        elif "talk about" in mode_lower or "discuss" in mode_lower:
            if "Discussion only" not in normalized:
                normalized.append("Discussion only")
        elif "planning" in mode_lower:
            if "Planning phase" not in normalized:
                normalized.append("Planning phase")
        elif "questions first" in mode_lower or "don't assume" in mode_lower:
            if "Clarification needed first" not in normalized:
                normalized.append("Clarification needed first")
        else:
            # Keep as-is if no normalization rule
            cap_mode = mode.strip().capitalize()
            if cap_mode not in normalized:
                normalized.append(cap_mode)
    
    return ", ".join(normalized) if normalized else ""


# ---------------------------------------------------------------------------
# Output Sanitization (v3.4) - Strip prompt leakage
# ---------------------------------------------------------------------------

# Patterns that indicate prompt scaffold leaked into output
LEAKAGE_PATTERNS = [
    r"^EXISTING JOB DESCRIPTION:\s*",
    r"^---\s*$",
    r"^NEW USER REQUIREMENTS.*?:\s*",
    r"^NEW USER MESSAGE.*?:\s*",
    r"^PREVIOUS SPEC:\s*",
    r"^UPDATED JOB DESCRIPTION:\s*",
]

# Patterns that ARE valid in Design preferences (visual/UI only)
DESIGN_PREF_WHITELIST_PATTERNS = [
    r"\bcolor\b", r"\bcolour\b", r"\bcolors\b", r"\bcolours\b",
    r"\bdark\s*mode\b", r"\blight\s*mode\b", r"\btheme\b", r"\bpalette\b",
    r"\bbrand\b",  # "brand colors"
    r"\blayout\b", r"\bsidebar\b", r"\btop\s*nav\b", r"\bcentered\b", r"\bgrid\b",
    r"\bstyle\b", r"\bminimal\b", r"\bmodern\b", r"\bplayful\b", r"\bclean\b",
    r"\bbig\s*buttons?\b", r"\bno\s*clutter\b", r"\bdead[\s-]*simple\b",
    r"\bui\s*(elements?|feel)\b", r"\bvisual\b", r"\baesthetic\b",
    r"\bsimple\b", r"\bfast\b", r"\bsleek\b", r"\belegant\b",
]

# Patterns that should NOT be in Design preferences (functional requirements)
DESIGN_PREF_BLACKLIST_PATTERNS = [
    # Calculations & logic
    r"\bcalculat", r"\bcompute\b", r"\bformula\b", r"\baverag",
    r"\bper\s*(day|week|parcel)\b", r"\b/\s*day\b", r"\bdaily\s+cost\b",
    # Data handling
    r"\bsync\b", r"\bexport\b", r"\bimport\b", r"\bapi\b",
    r"\btrack\b", r"\brecord\b", r"\blog\b", r"\binput\b",
    r"\bextract", r"\bformat\b", r"\bpars",  # parse, parsing
    # Screenshot / OCR
    r"\bocr\b", r"\bscreenshot", r"\bphoto\b", r"\bimage\b",
    r"\bdetect", r"\brecogni",  # detect, detection, recognize, recognition
    # Business / financial
    r"\bprofit\b", r"\bpay\b", r"\bfuel\b", r"\bwear\b", r"\bcost\b",
    r"\bparcel\b", r"\bdelivery\b", r"\bdeliveries\b", r"\bearning",
    # UI elements that are functional, not visual
    r"\bhistory\s*(list|row|screen)\b", r"\bshow\s*(gross|net|costs?)\b",
    r"\bimport\s*button\b", r"\bstart\s*day\b", r"\bfinish\s*day\b",
    # Workflow / method preferences
    r"\bworkflow\b", r"\bmethod\b", r"\bhandling\b", r"\bmodel\b",
    r"\bpriority\b", r"\bpriorities\b", r"\bprefer\b",
    r"\bauto\b", r"\bautomat",  # auto, automatic, automatically
    # Integration / system
    r"\bastra\b", r"\bweekly\s*breakdown\b",
    r"\bno\s*(manual|export|chart)\b",  # functional constraints
    # Misc functional terms
    r"\bvs\.?\b", r"\bversus\b",  # "X vs Y" is a functional choice
    r"\brequire", r"\bshould\b", r"\bmust\b",  # requirement language
]


def _sanitize_weaver_output(output: str) -> str:
    """
    Sanitize weaver output to remove any prompt scaffold leakage.
    
    If the LLM accidentally echoes parts of the prompt template,
    this function strips them out.
    """
    lines = output.split("\n")
    cleaned_lines = []
    skip_until_content = False
    
    for line in lines:
        line_stripped = line.strip()
        
        # Check for leakage patterns
        is_leakage = False
        for pattern in LEAKAGE_PATTERNS:
            if re.match(pattern, line_stripped, re.IGNORECASE):
                is_leakage = True
                # If we hit a scaffold header, skip until we see real content
                if "EXISTING" in line_stripped.upper() or "PREVIOUS" in line_stripped.upper():
                    skip_until_content = True
                break
        
        if is_leakage:
            continue
        
        # Skip separator lines when in skip mode
        if skip_until_content and line_stripped == "---":
            continue
        
        # Once we see real content, stop skipping
        if skip_until_content and line_stripped and line_stripped != "---":
            skip_until_content = False
        
        cleaned_lines.append(line)
    
    result = "\n".join(cleaned_lines).strip()
    
    # Log if we cleaned anything
    if result != output.strip():
        print("[WEAVER] Sanitized output - removed prompt leakage")
    
    return result


def _enforce_design_pref_hygiene(output: str) -> str:
    """
    Enforce section hygiene: Design preferences should only contain visual/UI prefs.
    
    v3.4.2: Removes functional requirements that were incorrectly bucketed into 
    Design preferences. This prevents duplication across sections during UPDATE merges.
    
    Keeps: color, layout, style, "big buttons", "no clutter", "dead simple"
    Removes: calculations, sync rules, tracking, OCR, profit/pay/fuel, history contents
    """
    lines = output.split("\n")
    result_lines = []
    in_design_section = False
    removed_count = 0
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Detect section headers (various formats)
        is_design_header = any([
            line_lower.startswith("design preferences"),
            line_lower.startswith("**design preferences"),
            line_lower.startswith("## design preferences"),
            line_lower.startswith("### design preferences"),
        ])
        
        # Detect other section headers (to know when we've left design prefs)
        is_other_header = any([
            line_lower.startswith("constraints"),
            line_lower.startswith("**constraints"),
            line_lower.startswith("platform"),
            line_lower.startswith("**platform"),
            line_lower.startswith("priority"),
            line_lower.startswith("**priority"),
            line_lower.startswith("unresolved"),
            line_lower.startswith("**unresolved"),
            line_lower.startswith("intended outcome"),
            line_lower.startswith("**intended outcome"),
            line_lower.startswith("what is being"),
            line_lower.startswith("**what is being"),
            line_lower.startswith("execution mode"),
            line_lower.startswith("**execution mode"),
        ])
        
        if is_design_header:
            in_design_section = True
            result_lines.append(line)
            continue
        
        if is_other_header and in_design_section:
            in_design_section = False
        
        if in_design_section and line_lower:
            # Skip structural/formatting lines - keep them
            if line_lower in ["---", "***", "___"] or line_lower.startswith("(if "):
                result_lines.append(line)
                continue
            
            # Check if this line belongs in design preferences
            has_whitelist = any(re.search(p, line_lower) for p in DESIGN_PREF_WHITELIST_PATTERNS)
            has_blacklist = any(re.search(p, line_lower) for p in DESIGN_PREF_BLACKLIST_PATTERNS)
            
            # Keep line if it's a valid preference line (starts with Color:/Layout:/Style:)
            is_valid_pref_line = any([
                line_lower.startswith("color"),
                line_lower.startswith("- color"),
                line_lower.startswith("* color"),
                line_lower.startswith("layout"),
                line_lower.startswith("- layout"),
                line_lower.startswith("* layout"),
                line_lower.startswith("style"),
                line_lower.startswith("- style"),
                line_lower.startswith("* style"),
                line_lower.startswith("ui element"),
                line_lower.startswith("- ui element"),
                line_lower.startswith("* ui element"),
            ])
            
            # v3.4.2 logic: Be strict about what stays in Design preferences
            # - Valid pref line (Color:/Layout:/Style:) -> KEEP
            # - Whitelist match without blacklist -> KEEP  
            # - Blacklist match -> REMOVE
            # - Neither (ambiguous) -> REMOVE (stricter than before)
            if is_valid_pref_line:
                result_lines.append(line)
            elif has_whitelist and not has_blacklist:
                result_lines.append(line)
            elif has_blacklist:
                # Skip this line - it's a functional requirement
                removed_count += 1
                preview = line.strip()[:60]
                print(f"[WEAVER] Removed from Design prefs (functional): {preview}...")
            else:
                # Ambiguous line in Design prefs - remove it (be strict)
                # This catches things that don't match whitelist visual patterns
                removed_count += 1
                preview = line.strip()[:60]
                print(f"[WEAVER] Removed from Design prefs (ambiguous): {preview}...")
        else:
            result_lines.append(line)
    
    if removed_count > 0:
        print(f"[WEAVER] Design pref hygiene: removed {removed_count} functional requirement(s)")
    
    return "\n".join(result_lines)


def _enforce_deduplication(output: str) -> str:
    """
    Enforce deduplication: Same sentence should not appear in multiple sections (Bug 3 fix).
    
    Specifically checks if "What is being built" and "Intended outcome" are identical
    or near-identical, and rewrites Outcome if so.
    """
    lines = output.split("\n")
    
    # Extract What and Outcome values
    what_value = ""
    outcome_value = ""
    what_line_idx = -1
    outcome_line_idx = -1
    
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        
        # Look for "What is being built" section
        if line_lower.startswith("what is being built") or line_lower.startswith("**what is being built"):
            # Value is on same line after colon, or next line
            if ":" in line:
                what_value = line.split(":", 1)[1].strip()
                what_line_idx = i
            elif i + 1 < len(lines):
                what_value = lines[i + 1].strip()
                what_line_idx = i + 1
        
        # Look for "Intended outcome" section
        elif line_lower.startswith("intended outcome") or line_lower.startswith("**intended outcome"):
            if ":" in line:
                outcome_value = line.split(":", 1)[1].strip()
                outcome_line_idx = i
            elif i + 1 < len(lines):
                outcome_value = lines[i + 1].strip()
                outcome_line_idx = i + 1
    
    # Check for duplication
    if what_value and outcome_value:
        # Normalize for comparison (lowercase, strip punctuation)
        what_normalized = re.sub(r'[^\w\s]', '', what_value.lower()).strip()
        outcome_normalized = re.sub(r'[^\w\s]', '', outcome_value.lower()).strip()
        
        # Check if identical or very similar
        is_duplicate = (
            what_normalized == outcome_normalized or
            what_normalized in outcome_normalized or
            outcome_normalized in what_normalized
        )
        
        if is_duplicate and outcome_line_idx >= 0:
            print(f"[WEAVER] Deduplication: What and Outcome were identical/similar")
            # Rewrite outcome to be different
            # Simple heuristic: prepend "Functional" or "Working" + add "implementation"
            if ":" in lines[outcome_line_idx]:
                prefix = lines[outcome_line_idx].split(":")[0] + ":"
                lines[outcome_line_idx] = f"{prefix} Functional {what_value.lower()} implementation"
            else:
                lines[outcome_line_idx] = f"Functional {what_value.lower()} implementation"
            print(f"[WEAVER] Rewrote Outcome to: {lines[outcome_line_idx]}")
    
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core Goal Detection (2-Factor Heuristic) - v3.5.0
# ---------------------------------------------------------------------------

# Action verbs that indicate intent
CORE_GOAL_VERBS = [
    "build", "create", "make", "add", "remove", "delete", "fix", "change",
    "update", "modify", "refactor", "implement", "write", "reply", "respond",
    "design", "develop", "integrate", "connect", "migrate", "convert", "generate",
    "set up", "setup", "configure", "install", "deploy", "test", "check",
    "analyze", "review", "edit", "improve", "optimize", "clean", "organize",
]

# Intent patterns that express desire/need (must be paired with a target)
# These are NOT action verbs but indicate the user wants something done
INTENT_GOAL_PATTERNS = [
    r"\bi\s+want\b",
    r"\bi\s+need\b",
    r"\bi'm\s+trying\b",
    r"\bi\s+am\s+trying\b",
    r"\bi'd\s+like\b",
    r"\bi\s+would\s+like\b",
]

# Target/directive words that indicate what the action applies to
# v3.5.0: Added creative/project targets (game, prototype, demo, etc.) - Bug 1 fix
CORE_GOAL_TARGETS = [
    # Technical targets
    "app", "application", "website", "page", "component", "feature", "function",
    "api", "endpoint", "service", "database", "table", "file", "folder", "code",
    "script", "module", "class", "method", "button", "form", "ui", "interface",
    "dashboard", "panel", "modal", "menu", "navbar", "sidebar", "widget",
    # Content targets
    "message", "email", "reply", "response", "document", "report", "spec",
    "text", "content", "data", "config", "settings", "theme", "style", "layout",
    # Project targets (v3.4.1)
    "tracker", "tool", "integration", "screen", "overlay", "plan", "flow",
    "logger", "monitor", "viewer", "editor", "builder", "generator",
    # Creative/project targets (v3.5.0 - Bug 1 fix)
    "game", "prototype", "demo", "simulator", "visualizer", "calculator",
    "timer", "clock", "todo", "calendar", "planner", "clone", "replica",
    # Abstract targets (only valid with action verbs, not intent patterns)
    "it", "this", "that", "one", "something", "thing", "system", "process",
]

# Concrete targets - subset that counts for intent patterns
# These are specific enough that "I want a <target>" is a clear goal
# v3.5.0: Added creative/project targets (game, prototype, demo, etc.) - Bug 1 fix
CONCRETE_TARGETS = [
    "app", "application", "website", "page", "component", "feature", "function",
    "api", "endpoint", "service", "database", "table", "file", "folder", "code",
    "script", "module", "class", "method", "button", "form", "ui", "interface",
    "dashboard", "panel", "modal", "menu", "navbar", "sidebar", "widget",
    "message", "email", "reply", "response", "document", "report", "spec",
    "tracker", "tool", "integration", "screen", "overlay", "plan", "flow",
    "logger", "monitor", "viewer", "editor", "builder", "generator",
    # Creative/project targets (v3.5.0 - Bug 1 fix)
    "game", "prototype", "demo", "simulator", "visualizer", "calculator",
    "timer", "clock", "todo", "calendar", "planner", "clone", "replica",
]

# Negation patterns that invalidate a goal
NEGATION_PATTERNS = [
    r"\bdon'?t\s+",
    r"\bdo\s+not\s+",
    r"\bnever\s+",
    r"\bno\s+need\s+to\s+",
    r"\bwithout\s+",
    r"\bavoid\s+",
    r"\bskip\s+",
]


def _has_core_goal(ramble_text: str) -> bool:
    """
    Check if ramble has a clear action/goal using 2-factor heuristic.
    
    v3.5.0: Now recognizes creative targets like "game", "prototype", "demo".
    
    Logic:
    - PASS if: (action_verb + any_target) OR (intent_pattern + concrete_target)
    - FAIL if: negated OR no valid pattern found
    
    NOTE: In v3.5.0, even if this returns False, Weaver should STILL produce
    a structured outline with ambiguities listed. This function is now used
    only for logging/debugging, not for blocking weave.
    """
    text_lower = ramble_text.lower()
    
    # --- Check for ACTION VERB + TARGET ---
    has_action_verb = False
    for verb in CORE_GOAL_VERBS:
        verb_pattern = rf"\b{re.escape(verb)}\b"
        verb_match = re.search(verb_pattern, text_lower)
        
        if not verb_match:
            continue
        
        verb_pos = verb_match.start()
        prefix_start = max(0, verb_pos - 20)
        prefix = text_lower[prefix_start:verb_pos]
        
        is_negated = any(re.search(neg, prefix) for neg in NEGATION_PATTERNS)
        
        if not is_negated:
            has_action_verb = True
            break
    
    if has_action_verb:
        # Action verb found - check for ANY target (including abstract ones)
        for target in CORE_GOAL_TARGETS:
            target_pattern = rf"\b{re.escape(target)}\b"
            if re.search(target_pattern, text_lower):
                print("[WEAVER] Core goal detected (action verb + target)")
                return True
    
    # --- Check for INTENT PATTERN + CONCRETE TARGET ---
    has_intent_pattern = False
    intent_negated = False
    
    for pattern in INTENT_GOAL_PATTERNS:
        intent_match = re.search(pattern, text_lower)
        if intent_match:
            # Check for negation before the intent pattern
            intent_pos = intent_match.start()
            prefix_start = max(0, intent_pos - 20)
            prefix = text_lower[prefix_start:intent_pos]
            
            if any(re.search(neg, prefix) for neg in NEGATION_PATTERNS):
                intent_negated = True
                continue
            
            has_intent_pattern = True
            break
    
    if has_intent_pattern and not intent_negated:
        # Intent pattern found - check for CONCRETE targets only
        # (prevents "I want something" from passing)
        for target in CONCRETE_TARGETS:
            target_pattern = rf"\b{re.escape(target)}\b"
            if re.search(target_pattern, text_lower):
                print(f"[WEAVER] Core goal detected (intent pattern + concrete target: '{target}')")
                return True
    
    # No valid goal pattern found
    if has_action_verb:
        print("[WEAVER] Action verb found but no target")
    elif has_intent_pattern:
        print("[WEAVER] Intent pattern found but no concrete target")
    else:
        print("[WEAVER] No action verb or intent pattern found")
    
    return False


# ---------------------------------------------------------------------------
# Design Job Detection — REMOVED in v4.0.0
# ---------------------------------------------------------------------------
# v4.0.0: _is_design_job() and DESIGN_JOB_INDICATORS removed.
# The old design job detector triggered on ANY mention of "app", "ui",
# "interface", "game", etc. — which meant EVERY ASTRA feature request
# was classified as a "design job" and got hardcoded game-focused questions.
# The LLM now generates its own contextual questions without gating.

# ---------------------------------------------------------------------------
# Shallow Question Generation — REMOVED in v4.0.0
# ---------------------------------------------------------------------------
# v4.0.0: SHALLOW_QUESTIONS, SHALLOW_QUESTION_KEYWORDS, and
# _get_shallow_questions() removed. These were hardcoded game-design
# questions ("Arcade-style or minimal?", "Keyboard or touch or controller?",
# "Centered vs sidebar HUD?") that were injected into the LLM prompt
# regardless of context. A voice-to-text feature request would get asked
# about game controllers because the keyword "app" appeared in the text.
#
# The LLM (GPT-5.2) now generates its own questions based on what's
# actually unclear in the user's requirements. This uses the model's
# reasoning capability rather than a fixed menu of 6 questions.


# v4.0.0: _detect_filled_slots() REMOVED
# Was hardcoded to detect game-design slots (platform, look_feel, controls, scope, layout).
# Now that the LLM generates its own contextual questions, slot detection is unnecessary.
# The LLM reads the user's requirements directly and knows what's been answered.


# ---------------------------------------------------------------------------
# Slot Reconciliation - v4.0.0: REMOVED
# The entire slot-based reconciliation system has been removed.
# It was built around 6 hardcoded game-design slots (platform, look_feel,
# controls, scope, layout) and is incompatible with LLM-generated questions.
# The LLM now handles question generation contextually, reading the user's
# actual requirements to determine what's been answered.
# ---------------------------------------------------------------------------

# v4.0.0: SLOT_AMBIGUITY_PATTERNS REMOVED (was hardcoded game-design slots)
# v4.0.0: SLOT_QUESTION_PATTERNS REMOVED (was hardcoded game-design slots)
_SLOT_RECONCILIATION_REMOVED = True  # Marker for grep/search




# ---------------------------------------------------------------------------
# Micro-Task Classification (v3.6.0)
# Detect simple file operations that need no questions
# ---------------------------------------------------------------------------

# Indicators FOR micro-task (simple file operations)
MICRO_FILE_INDICATORS = [
    "desktop", "folder", "file", "txt", "read", "open", "find",
    "answer", "reply", "write", "document", "documents", "message",
    "look at", "check", "locate", "search",
]

# Indicators AGAINST micro-task (only when paired with build verbs)
# v3.6.1: REMOVED "system" - "on my system" is file context, not software system
# v3.6.1: REMOVED "platform" - "on desktop" is a location, not software platform
NON_MICRO_INDICATORS = [
    "app", "application", "website", "page", "component", "feature",
    "game", "dashboard", "ui", "interface", "api", "endpoint", "service",
    "database", "design", "develop", "implement",
    "prototype", "demo", "refactor", "restructure", "migrate",
]

# v3.7: REFACTOR/RENAME operations should NEVER be micro-tasks
# v3.10: Moved REFACTOR_INDICATORS below (cleaned up, app-name entries removed)
# v3.10: Added REFACTOR_ACTION_PATTERNS for context-aware detection

# v3.10: Refactor detection patterns — context-aware, not keyword-only
# These patterns detect ACTUAL rename/refactor intent, not just keyword presence.
# Each pattern requires a refactor ACTION + a SCOPE or TARGET indicator.
REFACTOR_ACTION_PATTERNS = [
    # "rename X to Y" / "rename all X to Y"
    r"\brename\b.{1,40}\bto\b",
    # "rebrand from X to Y" / "rebrand X as Y"
    r"\brebrand\b",
    # "refactor" + scope indicator (codebase, all files, across, everywhere)
    r"\brefactor\b.{0,30}\b(across|codebase|all\s+files|everywhere|project)\b",
    # "replace all X with Y" / "replace X with Y in all files"
    r"\breplace\s+all\b",
    r"\breplace\b.{1,40}\b(across|everywhere|all\s+files|codebase|all\s+occurrences)\b",
    # "change all X to Y" / "change X to Y across"
    r"\bchange\s+all\b.{1,40}\bto\b",
    r"\bchange\b.{1,40}\bto\b.{1,40}\b(across|everywhere|all\s+files|codebase)\b",
    # "search and replace" / "find and replace" + scope
    r"\b(search|find)\s+and\s+replace\b",
    # Explicit codebase-wide rename language
    r"\ball\s+occurrences\b.{0,30}\b(of|rename|replace|change)\b",
    r"\b(rename|replace|change)\b.{0,30}\ball\s+occurrences\b",
]

# v3.8: Patterns that indicate user dismissed/answered questions
QUESTIONS_DISMISSED_PATTERNS = [
    r"questions?\s+(are\s+)?not\s+(really\s+)?needed",
    r"don'?t\s+need\s+(to\s+)?(ask|answer)\s+(those\s+)?questions?",
    r"reply\s+to\s+(your\s+)?questions?\s+(are\s+)?not\s+needed",
    r"no\s+need\s+(for|to\s+ask)\s+questions?",
    r"skip\s+(the\s+)?questions?",
    r"ignore\s+(the\s+)?questions?",
    r"questions?\s+aren'?t\s+(really\s+)?relevant",
]

# v3.8: Refactor task system prompt - NO design questions, focused on search/replace
# v3.10: Legacy list kept ONLY for _is_micro_file_task guard.
# Stripped of app-name-specific and overly-generic entries.
# The real refactor detection now uses REFACTOR_ACTION_PATTERNS above.
REFACTOR_INDICATORS = [
    "rename", "rebrand", "refactor", "replace all", "change all",
    "all occurrences", "codebase",
    # NOTE: "astra", "orb to astra", "branding", "front-end ui",
    # "across", "everywhere" were REMOVED in v3.10 — too generic,
    # caused false positives on any message mentioning the app name.
]

REFACTOR_TASK_SYSTEM_PROMPT = """You are Weaver for REFACTOR/RENAME TASKS.

Your job: Produce a FOCUSED job outline for text replacement / rename operations.

## CRITICAL RULES FOR REFACTOR TASKS:
1. NO design questions (dark mode, light mode, controls, layout, etc.) - IRRELEVANT
2. NO UI/UX questions - this is a TEXT REPLACEMENT task
3. NO platform questions - the pipeline knows the platform
4. Focus ONLY on: what to search, what to replace, where to search
5. Questions section should say "none" - the pipeline handles discovery

## WHAT TO EXTRACT:
- Search term: What text/string to find (e.g., "Orb")
- Replace term: What to replace it with (e.g., "Astra")
- Scope: Where to search (folder path, file types)
- Constraints: What NOT to change (e.g., no logos, text-only)

## OUTPUT FORMAT:

What is being built: [Short description] (refactor task)
Intent: Rename/replace "[SEARCH]" with "[REPLACE]" in [SCOPE]
Execution type: REFACTOR_TASK
Search term: [exact text to find]
Replace term: [exact text to replace with]
Scope: [folder/path to search]
Constraints:
- [constraint 1]
- [constraint 2]
Questions: none

## EXAMPLE:

Input: "Change the front-end UI so it's called Astra instead of Orb. Look in Orb Desktop on D drive. Text only, no logos."

Output:
What is being built: Text rebrand from Orb to Astra (refactor task)
Intent: Rename all occurrences of "Orb" to "Astra" in D:\\Orb Desktop front-end files
Execution type: REFACTOR_TASK
Search term: Orb (case-preserving: Orb→Astra, ORB→ASTRA, orb→astra)
Replace term: Astra
Scope: D:\\Orb Desktop (front-end UI files)
Constraints:
- Text-only changes (no logos or icons)
- Case-preserving replacement
- Front-end UI files only
Questions: none

CRITICAL:
- Keep output under 20 lines
- Questions section MUST say "none" - design questions are NEVER relevant for refactor tasks
- The Implementer will handle file discovery and show matches for confirmation
- DO NOT ask about colors, themes, controls, layout, scope preferences, etc."""


def _is_refactor_task(text: str) -> bool:
    """
    Detect refactor/rename operations that need special handling.
    
    v3.10: Now uses PATTERN-BASED detection instead of keyword matching.
    Requires actual rename/replace ACTION + SCOPE/TARGET context.
    
    This prevents false positives like:
    - "Add voice-to-text to the ASTRA desktop app" (mentions app name)
    - "Update the branding page" (mentions branding as a feature)
    - "Improve the front-end UI" (mentions UI as a feature target)
    
    Only triggers on actual refactor language like:
    - "Rename Orb to Astra across the codebase"
    - "Replace all occurrences of X with Y"
    - "Rebrand from Orb to Astra"
    - "Find and replace in all files"
    
    Returns True if a refactor ACTION PATTERN matches.
    """
    text_lower = text.lower()
    
    for pattern in REFACTOR_ACTION_PATTERNS:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            matched_text = match.group(0).strip()
            print(f"[WEAVER] v3.10 REFACTOR_TASK detected (pattern: '{matched_text}')")
            return True
    
    print(f"[WEAVER] v3.10 NOT refactor task (no action patterns matched)")
    return False


def _user_dismissed_questions(text: str) -> bool:
    """
    Detect if user explicitly dismissed or said questions aren't needed (v3.8.0).
    
    Patterns like:
    - "The reply to your questions are actually not needed"
    - "Questions aren't relevant"
    - "Don't need to ask those questions"
    
    Returns True if user dismissed questions.
    """
    text_lower = text.lower()
    
    for pattern in QUESTIONS_DISMISSED_PATTERNS:
        if re.search(pattern, text_lower):
            print(f"[WEAVER] v3.8 User dismissed questions (pattern matched)")
            return True
    
    return False

# Build verbs that make NON_MICRO_INDICATORS decisive
BUILD_VERBS = [
    "build", "create", "make", "develop", "implement", "prototype",
]

# Silent typo normalizations (v3.6.0)
# Uses word boundaries to avoid substring collisions
TYPO_NORMALIZATIONS = [
    (r"\bdeck\s*top\b", "desktop"),
    (r"\bdecktop\b", "desktop"),
    (r"\bdekstop\b", "desktop"),
    (r"\bdestop\b", "desktop"),
    (r"\bdesctop\b", "desktop"),
    (r"\bdocumets\b", "documents"),
    (r"\bdocments\b", "documents"),
    (r"\bfloder\b", "folder"),
    (r"\bfodler\b", "folder"),
    (r"\bfild\b", "file"),
    (r"\bflie\b", "file"),
    (r"\bmesage\b", "message"),
    (r"\bmessge\b", "message"),
    (r"\bmesssage\b", "message"),
    (r"\banser\b", "answer"),
    (r"\banwser\b", "answer"),
    (r"\brepley\b", "reply"),
    (r"\brelpy\b", "reply"),
    (r"\bwirte\b", "write"),
    (r"\bwrtie\b", "write"),
]

# Micro-task system prompt (v3.6.1 - STRICTER, no unnecessary questions)
MICRO_TASK_SYSTEM_PROMPT = """You are Weaver for MICRO FILE TASKS.

Your job: Produce a SHORT, minimal job outline (10-20 lines max) for simple file operations.

## ABSOLUTE RULES FOR MICRO FILE TASKS:
1. NO questions about OS/platform - it's always Windows
2. NO questions about desktop location - there's only one accessible
3. NO questions about file extensions - the system will search
4. NO questions about paths - the system will find them
5. NO questions about file format - default is plain text (.txt)
6. NO questions about overwriting - default is overwrite if exists
7. NO questions about exact filenames - the system searches
8. Questions section should say "none" unless execution would truly FAIL

## THE ONLY BLOCKING QUESTIONS (rare):
- DELETE operations need confirmation ("Should I really delete X?")
- MOVE without destination needs clarification ("Where to?")
- NOTHING ELSE is a blocker

## OUTPUT FORMAT (keep it short!):

What is being built: [Short description] (micro file task)
Intent: [One line - what to find and what to do with it]
Execution type: MICRO_FILE_TASK
Planned steps:
- [Step 1: Locate]
- [Step 2: Action]
- [Step 3: Output/Return]
Questions: none

That's the entire output. No ambiguities section. No extra sections.

## EXAMPLES:

Input: "Find test1, test2, test3, test4 on my system, read them, create reply file on desktop, write a reply"
Output:
What is being built: Multi-file reader with reply synthesis (micro file task)
Intent: Find test1-test4 anywhere on system, read content, create Desktop/reply.txt with response
Execution type: MICRO_FILE_TASK
Planned steps:
- System-wide search for test1, test2, test3, test4
- Read all found files
- Synthesize content into a reply
- Create Desktop/reply.txt with synthesized response
Questions: none

CRITICAL: 
- Keep output under 15 lines
- Questions section must say "none" unless DELETE or unclear MOVE destination
- SpecGate handles all file discovery - don't ask about paths/filenames/locations"""


def _normalize_typos(text: str) -> str:
    """
    Silently normalize common typos without flagging them (v3.6.0).
    
    Uses word boundaries to avoid substring collisions.
    """
    result = text
    normalized_any = False
    
    for pattern, correction in TYPO_NORMALIZATIONS:
        if re.search(pattern, result, re.IGNORECASE):
            result = re.sub(pattern, correction, result, flags=re.IGNORECASE)
            normalized_any = True
    
    if normalized_any:
        print(f"[WEAVER] Normalized typos in input")
    
    return result


# v3.11: Feature component indicators - multi-component requests are NEVER micro
# If 3+ of these appear in a request, it's a substantial feature, not a file task
FEATURE_COMPONENT_INDICATORS = [
    "audio", "microphone", "recording", "capture",
    "transcription", "speech", "voice", "stt", "whisper",
    "button", "widget", "component", "panel",
    "endpoint", "api", "route", "handler",
    "provider", "service", "integration",
    "config", "settings", "environment",
    "stream", "websocket", "real-time", "realtime",
    "authentication", "auth", "permission",
    "notification", "alert", "feedback",
    "database", "storage", "persistence",
]


def _is_micro_file_task(text: str) -> bool:
    """
    Detect simple file operations that need no questions (v3.6.0).
    v3.6.1: Context-aware detection - "create a file" is micro, "create an app" is not.
    v3.7.0: Refactor/rename operations are NEVER micro-tasks.
    v3.11.0: NON_MICRO indicators now checked EVEN when file indicators are present.
             Multi-component feature detection prevents substantial features from
             being classified as micro tasks.
    
    Logic:
    - If REFACTOR_INDICATOR present → NOT micro (codebase-wide operation)
    - If 3+ FEATURE_COMPONENT_INDICATORS present → NOT micro (substantial feature)
    - If NON_MICRO indicator present → NOT micro (even if file indicators match)
    - If BUILD_VERB + NON_MICRO → NOT micro (it's a build job)
    - If "create" + "file" → IS micro (simple file creation)
    - If any MICRO_FILE_INDICATOR present (without non-micro) → IS micro
    - Otherwise → NOT micro
    
    CRITICAL: Context matters!
    - "create a file" → micro task
    - "create an app" → NOT micro task
    - "build a game" → NOT micro task
    - "find file on my system" → micro task
    - "rename Orb to Astra" → NOT micro task (refactor!)
    - "add voice-to-text to the desktop app" → NOT micro task (feature!)
    - "add push-to-talk with audio capture and transcription" → NOT micro task (multi-component!)
    """
    text_lower = text.lower()
    
    # v3.7.0: FIRST check for refactor/rename indicators - these are NEVER micro
    for indicator in REFACTOR_INDICATORS:
        if indicator in text_lower:
            print(f"[WEAVER] v3.7 NOT micro-task (refactor indicator: '{indicator}')")
            return False
    
    # v3.11.0: Check for multi-component feature requests
    # If 3+ distinct feature components are mentioned, this is a substantial feature
    component_matches = [ind for ind in FEATURE_COMPONENT_INDICATORS if ind in text_lower]
    if len(component_matches) >= 3:
        print(f"[WEAVER] v3.11 NOT micro-task (multi-component feature: {component_matches[:5]}...)")
        return False
    
    # v3.11.0: Check NON_MICRO indicators EARLY - these override file indicators
    # "desktop app", "desktop application", "desktop feature" are NOT file tasks
    has_non_micro = any(ind in text_lower for ind in NON_MICRO_INDICATORS)
    if has_non_micro:
        # Find which non-micro indicator matched for logging
        matched_non_micro = [ind for ind in NON_MICRO_INDICATORS if ind in text_lower]
        print(f"[WEAVER] v3.11 NOT micro-task (non-micro indicators present: {matched_non_micro})")
        return False
    
    # v3.6.1: Check for explicit file creation context
    # "create a file", "create new file", "make a file" are MICRO tasks
    file_creation_patterns = [
        r"create\s+(?:a\s+)?(?:new\s+)?file",
        r"make\s+(?:a\s+)?(?:new\s+)?file",
        r"write\s+(?:a\s+)?(?:new\s+)?file",
        r"create\s+(?:a\s+)?(?:text|txt|reply|response)\s+file",
    ]
    for pattern in file_creation_patterns:
        if re.search(pattern, text_lower):
            print("[WEAVER] Classified as MICRO_FILE_TASK (file creation pattern)")
            return True
    
    # Check for file operation indicators
    has_file_indicator = any(ind in text_lower for ind in MICRO_FILE_INDICATORS)
    
    # v3.6.1: Check for BUILD VERB (even without non-micro, build verbs suggest non-micro)
    has_build_verb = any(v in text_lower for v in BUILD_VERBS)
    if has_build_verb:
        print(f"[WEAVER] NOT micro-task (build verb present without file context)")
        return False
    
    # If file indicators present and no non-micro/build overrides, it's a micro task
    if has_file_indicator:
        print("[WEAVER] Classified as MICRO_FILE_TASK")
        return True
    
    return False


def _get_blocking_questions(text: str, is_micro_task: bool) -> List[str]:
    """
    Only return questions that would BLOCK execution (v3.6.0).
    
    For micro tasks:
    - read + write is NOT a conflict (normal output flow)
    - delete IS a blocker (dangerous, must confirm)
    - move/copy without destination IS a blocker
    
    For non-micro tasks, returns empty (uses existing shallow question logic).
    """
    if not is_micro_task:
        return []  # Non-micro tasks use existing shallow question logic
    
    text_lower = text.lower()
    questions = []
    
    # Check for ACTUALLY conflicting/dangerous actions
    has_delete = any(w in text_lower for w in ["delete", "remove", "erase"])
    has_move = any(w in text_lower for w in ["move", "copy", "transfer"])
    
    # Blocker: delete is mentioned (dangerous, must confirm)
    if has_delete:
        questions.append("You mentioned deleting - should I delete the file, or just read it?")
    
    # Blocker: move/copy with unclear destination
    if has_move:
        # Check if destination is specified
        has_destination = any(w in text_lower for w in ["to ", "into ", "destination"])
        if not has_destination:
            questions.append("Where should I move/copy the file to?")
    
    # NON-blockers (do NOT ask):
    # - read + write (normal output flow)
    # - read + answer/reply (normal response flow)
    # - OS/platform (sandbox handles it)
    # - Which desktop (only one accessible)
    # - Exact filename (search and pick)
    # - Multiple files (use default selection rules)
    
    return questions


# v4.0.0: _reconcile_filled_slots() REMOVED — was built around hardcoded slots
# v4.0.0: _add_known_requirements_section() REMOVED — was built around hardcoded slots


# ---------------------------------------------------------------------------
# Main Stream Generator - v3.5.0 with all bug fixes
# ---------------------------------------------------------------------------

async def generate_weaver_stream(
    *,
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[Any] = None,
    conversation_id: str,
    is_continuation: bool = False,
    captured_answers: Optional[Dict[str, str]] = None,
) -> AsyncIterator[bytes]:
    """
    Weaver handler - v3.5.0 with FULL BUG FIXES.
    
    v3.5.0 CHANGES:
    - Bug 1: Core goal detection includes creative targets (game, prototype, etc.)
    - Bug 2: Meta-chat extraction (no code, just planning) goes to execution_mode
    - Bug 3: Deduplication post-check for What/Outcome
    - Bug 4: execution_mode field in output
    - Bug 5: Scope boundary - Weaver stays shallow, always outputs structure
    
    CRITICAL BEHAVIOR CHANGE (v3.5.0):
    - Weaver NEVER responds conversationally ("I need clarity")
    - Weaver ALWAYS outputs structured job outline
    - If ambiguous, lists ambiguities + asks 3-5 shallow questions
    - No framework/architecture/algorithm questions
    """
    print(f"[WEAVER] Starting weaver v4.0.0 for project_id={project_id}")
    logger.info("[WEAVER] Starting weaver v4.0.0 for project_id=%s", project_id)
    
    provider, model = _get_weaver_config()
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    if not _STREAMING_AVAILABLE:
        error_msg = "Streaming providers not available - check imports"
        yield _serialize_sse({"type": "token", "content": f"X {error_msg}"})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})
        return
    
    if not _MEMORY_AVAILABLE:
        error_msg = "Memory service not available - cannot read conversation"
        yield _serialize_sse({"type": "token", "content": f"X {error_msg}"})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})
        return
    
    stream_fn = _get_streaming_function(provider)
    if stream_fn is None:
        error_msg = f"Streaming function not available for provider: {provider}"
        yield _serialize_sse({"type": "token", "content": f"X {error_msg}"})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})
        return
    
    try:
        # =====================================================================
        # STEP 1: Gather ALL messages
        # =====================================================================
        
        all_messages = _gather_ramble_messages(db, project_id)
        
        if not all_messages:
            no_messages_msg = (
                "**No conversation to weave**\n\n"
                "I don't see any recent messages to organize into a job description.\n\n"
                "Share what you want to build or change, then say "
                "`how does that look all together` again."
            )
            yield _serialize_sse({"type": "token", "content": no_messages_msg})
            yield _serialize_sse({"type": "done", "provider": provider, "model": model})
            return
        
        total_message_count = len(all_messages)
        print(f"[WEAVER] Gathered {total_message_count} total messages")
        
        # =====================================================================
        # STEP 2: Extract meta-mode phrases (Bug 2 fix)
        # =====================================================================
        
        filtered_messages, extracted_modes = _extract_meta_mode(all_messages)
        execution_mode = _format_execution_mode(extracted_modes)
        
        if execution_mode:
            print(f"[WEAVER] Extracted execution_mode: {execution_mode}")
        
        # =====================================================================
        # STEP 2b: Apply typo normalization (v3.6.0)
        # Must happen BEFORE Step 4 builds ramble_text so classification sees
        # normalized text (e.g., "deck top" → "desktop")
        # =====================================================================
        
        for i, msg in enumerate(filtered_messages):
            if msg.get("role") == "user" and msg.get("content"):
                normalized_content = _normalize_typos(msg["content"])
                if normalized_content != msg["content"]:
                    filtered_messages[i] = {**msg, "content": normalized_content}
        
        # =====================================================================
        # STEP 3: Load confirmed prefs + woven hashes + checkpoint
        # =====================================================================
        
        confirmed_prefs = {}
        if _FLOW_STATE_AVAILABLE and get_confirmed_design_prefs:
            confirmed_prefs = get_confirmed_design_prefs(project_id)
            if confirmed_prefs:
                print(f"[WEAVER] Loaded confirmed prefs: {confirmed_prefs}")
        
        # Merge with any newly captured answers
        if captured_answers:
            confirmed_prefs.update(captured_answers)
            if _FLOW_STATE_AVAILABLE and save_confirmed_design_prefs:
                save_confirmed_design_prefs(project_id, captured_answers)
        
        # Load woven hashes for delta detection
        woven_hashes: Set[str] = set()
        if _FLOW_STATE_AVAILABLE and get_woven_user_hashes:
            woven_hashes = get_woven_user_hashes(project_id)
            if woven_hashes:
                print(f"[WEAVER] Loaded {len(woven_hashes)} woven user hashes")
        
        # Load checkpoint for previous output
        checkpoint = None
        if _FLOW_STATE_AVAILABLE and get_weave_checkpoint:
            checkpoint = get_weave_checkpoint(project_id)
            if checkpoint:
                print(f"[WEAVER] Loaded checkpoint: {checkpoint['message_count']} messages")
        
        # =====================================================================
        # STEP 4: Compute new messages using HASH-BASED dedup
        # v3.9.0: Now includes assistant messages with vision context
        # =====================================================================
        
        # v3.9.0: Extract vision context BEFORE filtering
        # This preserves Gemini vision analysis for SpecGate
        vision_context = _extract_vision_context(filtered_messages)
        if vision_context:
            print(f"[WEAVER] v3.9 Extracted {len(vision_context)} chars of vision context")
        
        # Filter to USER messages + assistant messages with vision context
        # v3.9.0: Changed from USER-only to include valuable vision analysis
        relevant_messages = [
            m for m in filtered_messages 
            if m.get("role") == "user" or 
               (m.get("role") == "assistant" and _is_vision_context(m.get("content", "")))
        ]
        
        # For hashing, we still only track USER messages (to determine what's new)
        user_messages_only = [m for m in filtered_messages if m.get("role") == "user"]
        print(f"[WEAVER] Filtered to {len(relevant_messages)} relevant messages ({len(user_messages_only)} USER + vision context) (from {total_message_count} total)")
        
        # Compute hashes for current user messages
        current_user_hashes = _hash_messages(user_messages_only)
        
        # Determine which messages are NEW (not in woven_hashes)
        new_user_hashes = current_user_hashes - woven_hashes
        
        # Get the actual new messages (those whose hash is in new_user_hashes)
        new_user_messages = [
            m for m in user_messages_only
            if _hash_message(m) in new_user_hashes
        ]
        
        # Determine mode: UPDATE if we have previous output AND woven hashes
        is_update_mode = bool(woven_hashes) and checkpoint is not None and checkpoint.get("last_output")
        
        if is_update_mode:
            if not new_user_messages:
                no_new_msg = (
                    "**Nothing new to weave**\n\n"
                    "I don't see any new requirements from you since the last weave.\n\n"
                    "Add more details to your conversation, then say "
                    "`how does that look all together` again."
                )
                yield _serialize_sse({"type": "token", "content": no_new_msg})
                yield _serialize_sse({"type": "done", "provider": provider, "model": model})
                return
            
            print(f"[WEAVER] UPDATE mode: {len(new_user_messages)} new USER messages (hash-based detection)")
        else:
            print("[WEAVER] CREATE mode: first weave for this project")
        
        # Format ramble text from RELEVANT messages (USER + vision context)
        # v3.9.0: Now includes vision analysis for context
        ramble_text = _format_ramble(relevant_messages)
        
        # =====================================================================
        # STEP 5: Core goal check (FOR LOGGING ONLY - v3.5.0)
        # In v3.5.0, Weaver ALWAYS proceeds to weave, even if core goal unclear
        # =====================================================================
        
        has_clear_goal = _has_core_goal(ramble_text)
        if not has_clear_goal:
            print("[WEAVER] Core goal unclear - will list as ambiguity in output")
        
        # =====================================================================
        # STEP 5b: Micro-task classification (v3.6.0)
        # Detect simple file operations that should skip unnecessary questions
        # =====================================================================
        
        is_micro_task = _is_micro_file_task(ramble_text)
        is_refactor_task = _is_refactor_task(ramble_text)
        questions_dismissed = _user_dismissed_questions(ramble_text)
        
        if is_micro_task:
            print("[WEAVER] MICRO_FILE_TASK mode - minimal output, no unnecessary questions")
        if is_refactor_task:
            print("[WEAVER] v3.8 REFACTOR_TASK mode - no design questions")
        if questions_dismissed:
            print("[WEAVER] v3.8 User dismissed questions - skipping shallow questions")
        
        # =====================================================================
        # STEP 6: Get blocking questions for micro tasks (v4.0.0 simplified)
        # v4.0.0: Removed _is_design_job() and hardcoded shallow questions.
        # The LLM now generates its own contextual questions in the system prompt.
        # Only micro tasks still get deterministic blocker questions.
        # =====================================================================
        
        blocking_questions = []
        
        if is_micro_task:
            # Micro tasks: blocker-only questions (delete confirmation, move destination)
            blocking_questions = _get_blocking_questions(ramble_text, is_micro_task=True)
            if blocking_questions:
                print(f"[WEAVER] Micro-task has {len(blocking_questions)} blocking question(s)")
        elif is_refactor_task or questions_dismissed:
            # v3.8.0: Refactor tasks and dismissed questions skip question generation
            print("[WEAVER] v3.8 Skipping questions (refactor task or user dismissed)")
        else:
            # v4.0.0: Normal feature requests - LLM generates its own questions
            # No hardcoded question injection. The system prompt instructs the LLM
            # to identify genuine gaps and ask contextually relevant questions.
            print("[WEAVER] v4.0 LLM will generate contextual questions (no hardcoded injection)")
        
        # Clear any lingering question state
        if _FLOW_STATE_AVAILABLE and clear_weaver_design_questions:
            clear_weaver_design_questions(project_id)
        
        # =====================================================================
        # STEP 7: Weave - CREATE or UPDATE mode
        # v3.5.0: ALWAYS produces structured output, never conversational
        # =====================================================================
        
        # Build prefs context
        prefs_context = ""
        if confirmed_prefs:
            prefs_lines = [f"- {k.title()}: {v}" for k, v in confirmed_prefs.items()]
            prefs_context = "\n\nUser's confirmed design preferences:\n" + "\n".join(prefs_lines)
        
        # Build execution mode context
        exec_mode_context = ""
        if execution_mode:
            exec_mode_context = f"\n\nExecution mode (extracted from meta-phrases): {execution_mode}"
        
        # v4.0.0: No questions_context injection. The LLM generates its own
        # contextual questions based on actual gaps in the user's requirements.
        
        if is_micro_task:
            # =================================================================
            # MICRO-TASK MODE (v3.6.0) - Simple file operations, minimal output
            # =================================================================
            print(f"[WEAVER] MICRO-TASK mode: using minimal prompt for file operation")
            
            start_message = f"**Quick task detected...**\n\n"
            yield _serialize_sse({"type": "token", "content": start_message})
            
            # Build blocking questions context if any
            blocker_context = ""
            if blocking_questions:
                blocker_context = "\n\nBLOCKING QUESTIONS (must include in output):\n" + "\n".join(f"- {q}" for q in blocking_questions)
            
            system_prompt = MICRO_TASK_SYSTEM_PROMPT
            user_prompt = f"""User request:

{ramble_text}{blocker_context}

Produce the minimal job outline:"""
        
        elif is_refactor_task:
            # =================================================================
            # REFACTOR-TASK MODE (v3.8.0) - Text replacement, no design questions
            # =================================================================
            print(f"[WEAVER] REFACTOR-TASK mode: using refactor prompt")
            
            start_message = f"**Refactor/rename task detected...**\n\n"
            yield _serialize_sse({"type": "token", "content": start_message})
            
            system_prompt = REFACTOR_TASK_SYSTEM_PROMPT
            user_prompt = f"""User request:

{ramble_text}

Produce the refactor job outline:"""
        
        elif is_update_mode:
            # UPDATE MODE - Merge new info into existing job description
            print(f"[WEAVER] UPDATE mode: weaving {len(new_user_messages)} new messages into existing spec")
            
            new_ramble = _format_ramble(new_user_messages)
            previous_output = checkpoint["last_output"]
            
            # DEBUG: Show what we're sending to the LLM
            print(f"[WEAVER] NEW RAMBLE CONTENT ({len(new_ramble)} chars):")
            print(f"[WEAVER] ---\n{new_ramble[:500]}{'...' if len(new_ramble) > 500 else ''}\n[WEAVER] ---")
            
            start_message = f"**Updating your job description...**\n\nIncorporating {len(new_user_messages)} new requirement(s) from you.\n\n"
            yield _serialize_sse({"type": "token", "content": start_message})
            
            # v3.5.0: UPDATED PROMPT with execution_mode and shallow questions
            system_prompt = """You are Weaver, a text organizer that UPDATES existing job descriptions.

Your task: Take an existing job description and ADD all new requirements from the user's latest messages.

CRITICAL RULES:
1. READ the new user text carefully - extract EVERY feature/requirement mentioned
2. ADD each feature as a clear bullet point in the appropriate section
3. Create new sections if needed (e.g., "Quality of Life Features", "Calculations")
4. DO NOT summarize multiple features into one line - list them separately
5. KEEP all existing content from the previous spec
6. DO NOT include any meta-commentary or headers like "Updated spec:" or "Here is the updated version:"
7. If "Execution mode" is provided, include it as a section
8. "What is being built" must be a SHORT NOUN PHRASE (not a sentence)
9. "Intended outcome" must be DIFFERENT wording from "What is being built" (Bug 3 - no duplication)

OUTPUT FORMAT:
- Output ONLY the complete updated job description
- Start directly with the content (e.g., "What is being built or changed")
- Include "Execution mode" section if provided
- Do NOT include any preamble or explanation
- Do NOT echo any part of these instructions
- Do NOT ask technical questions (frameworks, algorithms, architecture)"""

            user_prompt = f"""Previous job description:

{previous_output}

New requirements from user (extract and add EVERY feature):

{new_ramble}
{prefs_context}{exec_mode_context}

Output the complete updated job description with all new features added:"""

        else:
            # CREATE MODE - First weave
            print(f"[WEAVER] CREATE mode: weaving {total_message_count} messages")
            
            start_message = f"**Organizing your thoughts...**\n\nAnalyzing {total_message_count} messages to create a job description.\n\n"
            yield _serialize_sse({"type": "token", "content": start_message})
            
            # v4.0.0: LLM-GENERATED QUESTIONS - domain-agnostic, contextual
            system_prompt = """You are Weaver, a SHALLOW text organizer.

Your ONLY job: Take the human's rambling and restructure it into a minimal, stable job outline.

## What You DO:
- Extract the core goal as a SHORT NOUN PHRASE (not a full sentence)
- Summarize intent into "What is being built" and "Intended outcome" (DIFFERENT wording, no duplication)
- Faithfully list ALL requirements, constraints, and specifications the user provided
- List unresolved ambiguities at high level
- Generate up to 3-5 contextual clarifying questions about GENUINE GAPS (see rules below)
- Include execution mode if extracted from meta-phrases

## What You DO NOT DO (CRITICAL - SCOPE BOUNDARY):
- NO framework/library choices (don't suggest specific libraries or tools)
- NO file structure discussion
- NO algorithm or data structure talk
- NO architecture proposals
- NO implementation plans
- NO technical questions (those belong to later pipeline stages)
- NO resolving ambiguities yourself
- NO inventing requirements the user didn't state

## QUESTION GENERATION RULES (v4.1 - CRITICAL):
Zero questions is the PREFERRED and DEFAULT outcome. You generate questions ONLY when there
is a genuine gap that would make the requirement AMBIGUOUS TO BUILD.

Do NOT manufacture questions to appear thorough. Do NOT ask questions to fill a quota.
If the user gave clear, comprehensive requirements: output "Questions: none" and move on.

Rules:
1. DEFAULT TO ZERO QUESTIONS. Only ask if you genuinely cannot determine what to build.
2. READ the user's requirements carefully first. Do NOT ask about things they already specified.
3. Questions must be HIGH-LEVEL framing questions, never technical implementation questions.
4. Absolute maximum: 3 questions. But 0 is almost always correct for detailed requests.
5. Each question must address a GENUINE GAP - something the user didn't cover that would affect
   what gets built (not how it gets built).
6. Before writing ANY question, ask yourself: "Would the downstream pipeline be blocked without
   this answer?" If no, don't ask it.
7. NEVER ask these if the user already specified them (check carefully!):
   - Platform (if they said "desktop app" or "Windows" - that's answered)
   - Controls (if they described input methods - that's answered)
   - Scope (if they defined phases or boundaries - that's answered)
   - Architecture (if they described backend/frontend structure - that's answered)
   - Technology choices (if they named specific tools/libraries - that's answered)
8. If the user provided a detailed, well-structured request with explicit requirements,
   constraints, and phase boundaries, you MUST output "Questions: none".

ANTI-PATTERNS (never do these):
- Asking 3-5 questions on every request regardless of completeness
- Rephrasing stated requirements as questions ("You mentioned X, did you mean X?")
- Asking about preferences the user clearly stated
- Asking about things the downstream pipeline will handle (file paths, exact APIs, etc.)

BAD questions (generic, context-blind):
- "Dark mode or light mode?" (when user is asking for a backend service)
- "Keyboard or touch?" (when user specified keyboard shortcuts)
- "Bare minimum or extras?" (when user defined explicit Phase 1 boundaries)

GOOD questions (contextual, gap-filling — but ONLY if genuinely needed):
- "What latency target for transcription?" (voice feature, not specified)
- "Should wake word detection run continuously or only when app is focused?" (genuine ambiguity)
- "Target OS(es) beyond Windows?" (user said desktop but didn't clarify OS scope)

## Output Format:
Produce a MINIMAL structured job outline with these sections:
- **What is being built**: Short noun phrase (e.g., "Voice-to-text input system")
- **Intended outcome**: Different wording (e.g., "Local speech transcription integrated into desktop app")
- **Execution mode**: Only if extracted (e.g., "Discussion only, no code yet")
- **Key requirements**: Bullet list of what the user explicitly asked for
- **Design preferences**: Only if specified (visual/UI preferences only)
- **Constraints**: Only if explicitly stated by the user
- **Unresolved ambiguities**: Things genuinely unclear from the user's description
- **Questions**: Usually "none" — only include if a genuine gap would block building

## DEDUPLICATION RULE:
"What is being built" and "Intended outcome" must use DIFFERENT words.
BAD: What: "Voice input feature" / Outcome: "Voice input feature"
GOOD: What: "Voice-to-text input system" / Outcome: "Local speech transcription for desktop app"

## Critical Rules:
1. If the human didn't say it, it doesn't appear in your output.
2. If the human DID say it, it MUST appear in your output (don't drop requirements).
3. You are a TEXT ORGANIZER, not a solution designer.
4. Preserve the user's terminology and domain language."""

            user_prompt = f"""Organize this conversation into a job description:

{ramble_text}{prefs_context}{exec_mode_context}

Remember:
- Include ALL requirements the user stated (don't drop anything)
- Preserve any ambiguities (list them, don't resolve them)
- Keep What and Outcome DIFFERENT (no duplication)
- Only ask questions about GENUINE GAPS you identified (may be zero if requirements are comprehensive)
- NO technical questions (frameworks, algorithms, architecture)
- Preserve the user's domain terminology"""
        
        # Stream from LLM
        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response_chunks: List[str] = []
        
        async for chunk in stream_fn(messages=llm_messages, model=model):
            content = None
            if isinstance(chunk, dict):
                content = chunk.get("text") or chunk.get("content")
                if chunk.get("type") == "metadata":
                    continue
            elif hasattr(chunk, "choices") and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    content = delta.content
            if content:
                response_chunks.append(content)
                yield _serialize_sse({"type": "token", "content": content})
        
        # =====================================================================
        # STEP 8: Post-process and save
        # =====================================================================
        
        raw_output = "".join(response_chunks).strip()
        
        # Sanitize output to remove any prompt leakage
        sanitized_output = _sanitize_weaver_output(raw_output)
        
        # Enforce design preference hygiene (v3.4.2)
        hygiene_output = _enforce_design_pref_hygiene(sanitized_output)
        
        # Enforce deduplication (v3.5.0 - Bug 3 fix)
        dedup_output = _enforce_deduplication(hygiene_output)
        
        # v4.0.0: Slot detection/reconciliation REMOVED.
        # The LLM generates its own contextual questions and reads the user's
        # requirements directly — no need for hardcoded slot post-processing.
        job_description = dedup_output
        
        weaver_output_id = f"weaver-{uuid.uuid4().hex[:12]}"
        
        # Save woven user hashes (accumulate - don't replace)
        if _FLOW_STATE_AVAILABLE and save_woven_user_hashes:
            save_woven_user_hashes(project_id, current_user_hashes)
        
        # Save weave checkpoint
        if _FLOW_STATE_AVAILABLE and save_weave_checkpoint:
            save_weave_checkpoint(project_id, total_message_count, job_description)
        
        # Save confirmed prefs (they persist)
        if _FLOW_STATE_AVAILABLE and save_confirmed_design_prefs and confirmed_prefs:
            save_confirmed_design_prefs(project_id, confirmed_prefs)
        
        # Store in flow state for Spec Gate
        # v3.9.1: Now passing vision_context for intelligent UI classification
        if _FLOW_STATE_AVAILABLE and start_weaver_flow:
            try:
                start_weaver_flow(
                    project_id=project_id,
                    weaver_spec_id=weaver_output_id,
                    weaver_job_description=job_description,
                    vision_context=vision_context,  # v3.9.1: Pass vision context to flow state
                )
                if vision_context:
                    print(f"[WEAVER] v3.9.1 Vision context stored in flow state ({len(vision_context)} chars)")
            except Exception as e:
                logger.warning("[WEAVER] Failed to store in flow state: %s", e)
        
        # =====================================================================
        # Completion message
        # =====================================================================
        
        mode_indicator = "updated" if is_update_mode else "ready"
        completion_message = f"""

---

**Job description {mode_indicator}** (`{weaver_output_id}`)

This is a structured outline of what you described. Review it above.

**Next step:** Say **'Send to Spec Gate'** to validate and build a full specification."""

        yield _serialize_sse({"type": "token", "content": completion_message})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})
        
    except Exception as e:
        logger.exception("[WEAVER] Error during streaming")
        error_message = f"\n\nWeaver error: {str(e)}"
        yield _serialize_sse({"type": "token", "content": error_message})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})


# ---------------------------------------------------------------------------
# LEGACY COMPATIBILITY
# ---------------------------------------------------------------------------

__all__ = ["generate_weaver_stream"]
