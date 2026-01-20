# FILE: app/llm/weaver_stream.py
"""
Weaver Stream Handler for ASTRA - SIMPLIFIED VERSION

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
  - Index slicing was brittle and extracted wrong messages
  - Hash-based tracking guarantees correct delta detection
- Bug B fix: Rewrite UPDATE prompt to prevent scaffold leakage
  - Removed "EXISTING JOB DESCRIPTION:" markers from prompt
  - Added output sanitization to strip any leakage
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

v3.1 (2026-01-20): DESIGN QUESTION CAPABILITY
- Added design question detection for UI/app jobs
- Added core goal detection (2-factor heuristic)
- Decision tree: No messages → Core goal check → Design questions → Weave

v3.0 (2026-01-19): LOCKED BEHAVIOUR SPEC IMPLEMENTATION
- Weaver is now a SIMPLE TEXT ORGANIZER
- No spec building, no JSON parsing, no DB persistence
- Reads conversation (ramble) and organizes it into a job description
- Stores output in flow state for Spec Gate to build spec from

LOCKED WEAVER BEHAVIOUR:
- Purpose: Convert human rambling into a structured job outline
- NOT a full spec builder - just a text organizer
- Reads messages to get input (the ramble)
- Does NOT persist to specs table
- Does NOT build JSON specs
- Does NOT resolve ambiguities or contradictions
- May ask ONE question ONLY if core goal is completely missing

WEAVER DECISION TREE (v3.4):
1) Gather ALL messages
   - If no messages → stream "No conversation to weave" → STOP
2) Load confirmed design prefs + woven hashes
3) Compute new user messages using hash-based dedup
   - If UPDATE mode and no new messages → "Nothing new" → STOP
4) Core goal check (only on first weave)
   - If core goal missing/unclear → ask ONLY 1 core-goal question → STOP
5) Design input check (only if prefs not confirmed)
   - If UI/app job + prefs not confirmed → ask design questions → STOP
   - When answered, save as confirmed_design_prefs
6) Weave
   - UPDATE mode: pass previous output + new messages to LLM
   - CREATE mode: pass all messages to LLM
   - Save hashes of woven messages
   - Sanitize output (strip any prompt leakage)
   - Stream result → DONE
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import AsyncIterator, Dict, List, Optional, Any, Set

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
            # - Valid pref line (Color:/Layout:/Style:) → KEEP
            # - Whitelist match without blacklist → KEEP  
            # - Blacklist match → REMOVE
            # - Neither (ambiguous) → REMOVE (stricter than before)
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


# ---------------------------------------------------------------------------
# Core Goal Detection (2-Factor Heuristic) - v3.4.1
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
# v3.4.1: Added more concrete targets for intent patterns
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
    # Abstract targets (only valid with action verbs, not intent patterns)
    "it", "this", "that", "one", "something", "thing", "system", "process",
]

# Concrete targets - subset that counts for intent patterns
# These are specific enough that "I want a <target>" is a clear goal
CONCRETE_TARGETS = [
    "app", "application", "website", "page", "component", "feature", "function",
    "api", "endpoint", "service", "database", "table", "file", "folder", "code",
    "script", "module", "class", "method", "button", "form", "ui", "interface",
    "dashboard", "panel", "modal", "menu", "navbar", "sidebar", "widget",
    "message", "email", "reply", "response", "document", "report", "spec",
    "tracker", "tool", "integration", "screen", "overlay", "plan", "flow",
    "logger", "monitor", "viewer", "editor", "builder", "generator",
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
    
    v3.4.1: Now recognizes intent patterns like "I want/I need" when paired
    with a concrete target. This prevents false negatives on messages like
    "I want a delivery day tracker app."
    
    Logic:
    - PASS if: (action_verb + any_target) OR (intent_pattern + concrete_target)
    - FAIL if: negated OR no valid pattern found
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
# Design Job Detection
# ---------------------------------------------------------------------------

DESIGN_JOB_INDICATORS = [
    "app", "ui", "interface", "page", "screen", "dashboard",
    "website", "form", "button", "layout", "component", "modal",
    "sidebar", "navbar", "menu", "panel", "widget", "view",
    "frontend", "front-end", "front end", "web app", "webapp",
]


def _is_design_job(ramble_text: str) -> bool:
    """Check if this job involves UI/design decisions."""
    text_lower = ramble_text.lower()
    for indicator in DESIGN_JOB_INDICATORS:
        if indicator in text_lower:
            print(f"[WEAVER] Design job detected (indicator: '{indicator}')")
            return True
    return False


# ---------------------------------------------------------------------------
# Design Question Generation - DYNAMIC (LLM-based)
# ---------------------------------------------------------------------------

# Keywords used to detect if user already specified preferences
# (Used for quick keyword matching before asking questions)
COLOR_KEYWORDS = [
    "color", "colour", "dark mode", "light mode", "theme", "palette",
    "black", "white", "blue", "red", "green", "gray", "grey",
    "brand color", "brand colours", "colors", "colours",
]

STYLE_KEYWORDS = [
    "style", "minimal", "minimalist", "modern", "playful", "professional",
    "clean", "flat", "material", "glassmorphism", "neumorphism",
    "simple", "dead simple", "big buttons", "sleek", "elegant",
    "corporate", "casual", "fun", "serious", "formal", "informal",
]

LAYOUT_KEYWORDS = [
    "layout", "sidebar", "side bar", "top nav", "topnav", "centered", "grid", 
    "cards", "tabs", "split", "single column", "two column", "three column", 
    "responsive", "mobile first", "desktop first", "full width", "fixed width",
]

# Fallback questions (used when dynamic generation fails or for keyword capture)
DESIGN_QUESTIONS = {
    "color": "What color scheme would you like? (e.g., dark mode, light mode, brand colors)",
    "style": "Any particular visual style? (e.g., minimal, modern, playful, clean)",
    "layout": "Do you have a preferred layout? (e.g., sidebar, top nav, centered, grid)",
}

# Rules that guide what questions might be needed for UI/design jobs
# The LLM uses these to decide what to ask (if anything)
DESIGN_QUESTION_RULES = """
For UI/app jobs, the following preferences help produce a better spec:

1. COLOR SCHEME - How should the app look color-wise?
   - Examples: dark mode, light mode, brand colors, specific colors
   - Only ask if not mentioned or implied in the conversation

2. VISUAL STYLE - What's the overall aesthetic?
   - Examples: minimal, modern, playful, clean, professional, fun
   - Only ask if not mentioned or implied in the conversation

3. LAYOUT STRUCTURE - How should the main navigation work?
   - Examples: sidebar, top nav, centered, grid, tabs
   - Only ask if not mentioned or implied in the conversation

RULES:
- Only generate questions for preferences that are TRULY MISSING
- If the user has given ANY indication of preference (even indirect), don't ask
- Return EMPTY if all preferences can be inferred or aren't relevant
- Keep questions short and conversational
- Include examples in parentheses
"""


async def _generate_dynamic_design_questions(
    ramble_text: str,
    confirmed_prefs: Dict[str, str],
    previous_output: Optional[str],
    stream_fn,
    model: str,
) -> Dict[str, str]:
    """
    Use LLM to dynamically decide what design questions to ask (if any).
    
    Returns dict of question_type -> question_text, or empty dict if no questions needed.
    """
    # Build context about what we already know
    known_prefs = []
    if confirmed_prefs:
        known_prefs.extend([f"- {k}: {v}" for k, v in confirmed_prefs.items()])
    
    # Also check previous output for prefs
    if previous_output:
        extracted = _extract_prefs_from_output(previous_output)
        for k, v in extracted.items():
            if k not in confirmed_prefs:
                known_prefs.append(f"- {k}: {v} (from previous spec)")
    
    known_section = "\n".join(known_prefs) if known_prefs else "None yet"
    
    system_prompt = f"""You analyze conversations to decide what design questions to ask.

{DESIGN_QUESTION_RULES}

Already known preferences:
{known_section}

Your task: Analyze the conversation and return ONLY questions for preferences that are genuinely missing.

RESPONSE FORMAT (JSON only, no markdown):
{{
  "questions": [
    {{"type": "color", "question": "What color scheme would you like? (e.g., dark mode, light mode)"}},
    {{"type": "style", "question": "Any particular visual style? (e.g., minimal, modern)"}}
  ]
}}

Or if no questions needed:
{{
  "questions": []
}}"""

    user_prompt = f"""Conversation to analyze:

{ramble_text}

Based on this conversation, what design questions (if any) should be asked? Return JSON only."""
    
    # Quick LLM call to get questions
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    response_text = ""
    try:
        async for chunk in stream_fn(messages=messages, model=model):
            content = None
            if isinstance(chunk, dict):
                content = chunk.get("text") or chunk.get("content")
            elif hasattr(chunk, "choices") and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    content = delta.content
            if content:
                response_text += content
        
        # Parse JSON response
        # Clean up markdown if present
        clean_text = response_text.strip()
        if clean_text.startswith("```"):
            clean_text = re.sub(r"```json?\n?", "", clean_text)
            clean_text = re.sub(r"```\n?$", "", clean_text)
        
        data = json.loads(clean_text)
        questions = data.get("questions", [])
        
        # Convert to dict format
        result = {}
        for q in questions:
            q_type = q.get("type", "")
            q_text = q.get("question", "")
            if q_type and q_text:
                result[q_type] = q_text
        
        print(f"[WEAVER] LLM generated {len(result)} design questions: {list(result.keys())}")
        return result
        
    except Exception as e:
        logger.warning(f"[WEAVER] Failed to generate dynamic questions: {e}")
        print(f"[WEAVER] Dynamic question generation failed: {e}")
        # Fall back to no questions (proceed to weave)
        return {}


def _get_missing_design_prefs(
    messages: List[Dict[str, Any]],
    confirmed_prefs: Dict[str, str],
    previous_output: Optional[str] = None,
) -> Dict[str, str]:
    """
    Check which design prefs are missing.
    
    Checks:
    1. confirmed_prefs from flow state
    2. Previous job description output (parses "Design preferences" section)
    3. User messages for keywords
    
    Returns dict of question_type → question_text for missing prefs.
    """
    questions = {}
    
    # First, try to extract prefs from previous job description
    # This is the "intelligent" approach - read our own output
    extracted_from_output = {}
    if previous_output:
        extracted_from_output = _extract_prefs_from_output(previous_output)
        if extracted_from_output:
            print(f"[WEAVER] Extracted prefs from previous output: {extracted_from_output}")
    
    # Merge all sources: output prefs → confirmed prefs (confirmed wins)
    all_prefs = {**extracted_from_output, **confirmed_prefs}
    
    # Extract only USER content for preference checking
    user_text = " ".join(
        msg.get("content", "").lower()
        for msg in messages
        if msg.get("role") == "user"
    )
    
    # Also consider all known prefs
    prefs_text = " ".join(all_prefs.values()).lower() if all_prefs else ""
    check_text = user_text + " " + prefs_text
    
    print(f"[WEAVER] Checking design prefs (confirmed: {list(all_prefs.keys())})")
    
    # Color - already have it?
    if "color" in all_prefs:
        print(f"[WEAVER]   → Color: confirmed ({all_prefs['color']})")
    elif any(kw in check_text for kw in COLOR_KEYWORDS):
        print("[WEAVER]   → Color: found in text")
    else:
        questions["color"] = DESIGN_QUESTIONS["color"]
        print("[WEAVER]   → Color: MISSING")
    
    # Style - already have it?
    if "style" in all_prefs:
        print(f"[WEAVER]   → Style: confirmed ({all_prefs['style']})")
    elif any(kw in check_text for kw in STYLE_KEYWORDS):
        print("[WEAVER]   → Style: found in text")
    else:
        questions["style"] = DESIGN_QUESTIONS["style"]
        print("[WEAVER]   → Style: MISSING")
    
    # Layout - already have it?
    if "layout" in all_prefs:
        print(f"[WEAVER]   → Layout: confirmed ({all_prefs['layout']})")
    elif any(kw in check_text for kw in LAYOUT_KEYWORDS):
        print("[WEAVER]   → Layout: found in text")
    else:
        questions["layout"] = DESIGN_QUESTIONS["layout"]
        print("[WEAVER]   → Layout: MISSING")
    
    return questions


def _extract_prefs_from_output(job_description: str) -> Dict[str, str]:
    """
    Extract design preferences from a previous job description.
    
    Looks for patterns like:
    - "Color: dark mode" or "Color scheme: dark"
    - "Style: minimal"
    - "Layout: sidebar"
    
    This is the "intelligent" approach - reading our own output.
    """
    prefs = {}
    text = job_description.lower()
    
    # Look for Design preferences section
    # Pattern: "Color: X" or "Color scheme: X"
    color_match = re.search(r'color(?:\s+scheme)?\s*[:\-]\s*([^\n,]+)', text)
    if color_match:
        prefs["color"] = color_match.group(1).strip()
    
    # Pattern: "Style: X" or "Visual style: X"
    style_match = re.search(r'(?:visual\s+)?style\s*[:\-]\s*([^\n,]+)', text)
    if style_match:
        prefs["style"] = style_match.group(1).strip()
    
    # Pattern: "Layout: X"
    layout_match = re.search(r'layout\s*[:\-]\s*([^\n,]+)', text)
    if layout_match:
        prefs["layout"] = layout_match.group(1).strip()
    
    return prefs


# ---------------------------------------------------------------------------
# Main Stream Generator - v3.4 with Hash-Based Delta + Prompt Fix
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
    Weaver handler - v3.4 with HASH-BASED DELTA + PROMPT FIX.
    
    DECISION TREE (v3.4):
    1) Gather ALL messages
    2) Load confirmed design prefs + woven hashes
    3) Compute new user messages using hash-based dedup
    4) Core goal check (only on first weave)
    5) Design input check (only if prefs not confirmed)
    6) Weave (UPDATE mode uses previous output + new messages)
    7) Save hashes + checkpoint + confirmed prefs
    8) Sanitize output (strip any prompt leakage)
    """
    print(f"[WEAVER] Starting weaver for project_id={project_id}")
    logger.info("[WEAVER] Starting weaver for project_id=%s", project_id)
    
    provider, model = _get_weaver_config()
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    if not _STREAMING_AVAILABLE:
        error_msg = "Streaming providers not available - check imports"
        yield _serialize_sse({"type": "token", "content": f"❌ {error_msg}"})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})
        return
    
    if not _MEMORY_AVAILABLE:
        error_msg = "Memory service not available - cannot read conversation"
        yield _serialize_sse({"type": "token", "content": f"❌ {error_msg}"})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})
        return
    
    stream_fn = _get_streaming_function(provider)
    if stream_fn is None:
        error_msg = f"Streaming function not available for provider: {provider}"
        yield _serialize_sse({"type": "token", "content": f"❌ {error_msg}"})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})
        return
    
    try:
        # =====================================================================
        # STEP 1: Gather ALL messages
        # =====================================================================
        
        all_messages = _gather_ramble_messages(db, project_id)
        
        if not all_messages:
            no_messages_msg = (
                "🧵 **No conversation to weave**\n\n"
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
        # STEP 2: Load confirmed prefs + woven hashes + checkpoint
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
        # STEP 3: Compute new messages using HASH-BASED dedup
        # =====================================================================
        
        # Filter to USER messages only (prevents assistant pollution)
        user_messages_only = [m for m in all_messages if m.get("role") == "user"]
        print(f"[WEAVER] Filtered to {len(user_messages_only)} USER messages (from {total_message_count} total)")
        
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
                    "🧵 **Nothing new to weave**\n\n"
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
        
        # Format ramble text from USER messages
        ramble_text = _format_ramble(user_messages_only)
        
        # =====================================================================
        # STEP 4: Core goal check (only on first weave / CREATE mode)
        # =====================================================================
        
        if not is_update_mode and not _has_core_goal(ramble_text):
            print("[WEAVER] Core goal missing - asking clarification question")
            
            core_goal_question = (
                "🧵 **I need a bit more clarity**\n\n"
                "I can see some conversation, but I'm not sure what you want me to do.\n\n"
                "**What's the main goal?**\n"
                "For example: *Build an app*, *Fix this bug*, *Reply to the message*, "
                "*Create a dashboard*, etc.\n\n"
                "Once you've clarified, say `how does that look all together` again."
            )
            yield _serialize_sse({"type": "token", "content": core_goal_question})
            yield _serialize_sse({"type": "done", "provider": provider, "model": model})
            return
        
        # =====================================================================
        # STEP 5: Design input check (only if prefs not already confirmed)
        # =====================================================================
        
        if _is_design_job(ramble_text):
            # Get previous output for intelligent pref extraction
            previous_output = checkpoint.get("last_output") if checkpoint else None
            missing_prefs = _get_missing_design_prefs(all_messages, confirmed_prefs, previous_output)
            
            if missing_prefs and not is_continuation:
                print(f"[WEAVER] Design job - asking {len(missing_prefs)} design questions")
                
                # Store questions in flow state for auto-capture
                if _FLOW_STATE_AVAILABLE and set_weaver_design_questions:
                    set_weaver_design_questions(project_id, missing_prefs)
                
                questions_formatted = "\n".join(f"• {q}" for q in missing_prefs.values())
                
                design_questions_msg = (
                    "🎨 **Design preferences needed**\n\n"
                    "This looks like a UI/app job. Before I organize your thoughts, "
                    "I have a few quick questions about the look and feel:\n\n"
                    f"{questions_formatted}\n\n"
                    "Just answer in your next message and I'll continue automatically."
                )
                yield _serialize_sse({"type": "token", "content": design_questions_msg})
                yield _serialize_sse({"type": "done", "provider": provider, "model": model})
                return
            
            elif missing_prefs and is_continuation:
                # Still missing some prefs after answering - ask remaining
                questions_formatted = "\n".join(f"• {q}" for q in missing_prefs.values())
                captured_list = ", ".join(confirmed_prefs.values()) if confirmed_prefs else "none yet"
                
                partial_msg = (
                    f"🎨 **Got it!** I've noted: {captured_list}\n\n"
                    f"Still need to know:\n\n"
                    f"{questions_formatted}\n\n"
                    f"Answer in your next message."
                )
                
                if _FLOW_STATE_AVAILABLE and set_weaver_design_questions:
                    set_weaver_design_questions(project_id, missing_prefs)
                
                yield _serialize_sse({"type": "token", "content": partial_msg})
                yield _serialize_sse({"type": "done", "provider": provider, "model": model})
                return
        
        # Clear any lingering question state
        if _FLOW_STATE_AVAILABLE and clear_weaver_design_questions:
            clear_weaver_design_questions(project_id)
        
        # =====================================================================
        # STEP 6: Weave - CREATE or UPDATE mode
        # =====================================================================
        
        # Build prefs context
        prefs_context = ""
        if confirmed_prefs:
            prefs_lines = [f"- {k.title()}: {v}" for k, v in confirmed_prefs.items()]
            prefs_context = "\n\nUser's confirmed design preferences:\n" + "\n".join(prefs_lines)
        
        if is_update_mode:
            # UPDATE MODE - Merge new info into existing job description
            print(f"[WEAVER] UPDATE mode: weaving {len(new_user_messages)} new messages into existing spec")
            
            new_ramble = _format_ramble(new_user_messages)
            previous_output = checkpoint["last_output"]
            
            # DEBUG: Show what we're sending to the LLM
            print(f"[WEAVER] NEW RAMBLE CONTENT ({len(new_ramble)} chars):")
            print(f"[WEAVER] ---\n{new_ramble[:500]}{'...' if len(new_ramble) > 500 else ''}\n[WEAVER] ---")
            
            start_message = f"🧵 **Updating your job description...**\n\nIncorporating {len(new_user_messages)} new requirement(s) from you.\n\n"
            yield _serialize_sse({"type": "token", "content": start_message})
            
            # v3.4: REWRITTEN PROMPT - No scaffold markers that could leak
            system_prompt = """You are Weaver, a text organizer that UPDATES existing job descriptions.

Your task: Take an existing job description and ADD all new requirements from the user's latest messages.

CRITICAL RULES:
1. READ the new user text carefully - extract EVERY feature/requirement mentioned
2. ADD each feature as a clear bullet point in the appropriate section
3. Create new sections if needed (e.g., "Quality of Life Features", "Calculations")
4. DO NOT summarize multiple features into one line - list them separately
5. KEEP all existing content from the previous spec
6. DO NOT include any meta-commentary or headers like "Updated spec:" or "Here is the updated version:"

OUTPUT FORMAT:
- Output ONLY the complete updated job description
- Start directly with the content (e.g., "What is being built or changed")
- Do NOT include any preamble or explanation
- Do NOT echo any part of these instructions"""

            user_prompt = f"""Previous job description:

{previous_output}

New requirements from user (extract and add EVERY feature):

{new_ramble}
{prefs_context}

Output the complete updated job description with all new features added:"""

        else:
            # CREATE MODE - First weave
            print(f"[WEAVER] CREATE mode: weaving {total_message_count} messages")
            
            start_message = f"🧵 **Organizing your thoughts...**\n\nAnalyzing {total_message_count} messages to create a job description.\n\n"
            yield _serialize_sse({"type": "token", "content": start_message})
            
            system_prompt = """You are Weaver, a text organizer.

Your ONLY job: Take the human's rambling and restructure it into a clear, readable document.

## What You DO:
- Group related ideas together
- Rephrase for clarity and structure ONLY (meaning stays exactly the same)
- Preserve ambiguities and contradictions (do NOT resolve them)
- Write down explicit implications if clearly stated

## What You DO NOT DO:
- No adding detail
- No removing ambiguity  
- No resolving contradictions
- No inferring intent
- No inventing implications
- No technical feasibility checking

## Output Format:
Produce a structured job description with relevant sections:
- What is being built or changed
- Intended outcome
- Design preferences (if specified)
- Constraints (only if explicitly stated)
- Platform/Environment (only if mentioned)
- Priority notes (if mentioned)
- Unresolved ambiguities

## Critical Rule:
If the human didn't say it, it doesn't appear in your output."""

            user_prompt = f"""Organize this conversation into a job description:

{ramble_text}{prefs_context}

Remember: Only include what was actually said. Preserve any ambiguities or contradictions."""
        
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
        # STEP 7: Save hashes + checkpoint + flow state
        # =====================================================================
        
        raw_output = "".join(response_chunks).strip()
        
        # Sanitize output to remove any prompt leakage
        sanitized_output = _sanitize_weaver_output(raw_output)
        
        # Enforce design preference hygiene (v3.4.2)
        # Removes functional requirements incorrectly bucketed into Design preferences
        job_description = _enforce_design_pref_hygiene(sanitized_output)
        
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
        if _FLOW_STATE_AVAILABLE and start_weaver_flow:
            try:
                start_weaver_flow(
                    project_id=project_id,
                    weaver_spec_id=weaver_output_id,
                    weaver_job_description=job_description,
                )
            except Exception as e:
                logger.warning("[WEAVER] Failed to store in flow state: %s", e)
        
        # =====================================================================
        # Completion message
        # =====================================================================
        
        mode_indicator = "updated" if is_update_mode else "ready"
        completion_message = f"""

---

📋 **Job description {mode_indicator}** (`{weaver_output_id}`)

This is a structured outline of what you described. Review it above.

**Next step:** Say **'Send to Spec Gate'** to validate and build a full specification."""

        yield _serialize_sse({"type": "token", "content": completion_message})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})
        
    except Exception as e:
        logger.exception("[WEAVER] Error during streaming")
        error_message = f"\n\n❌ Weaver error: {str(e)}"
        yield _serialize_sse({"type": "token", "content": error_message})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})


# ---------------------------------------------------------------------------
# LEGACY COMPATIBILITY
# ---------------------------------------------------------------------------

__all__ = ["generate_weaver_stream"]
