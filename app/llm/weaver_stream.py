# FILE: app/llm/weaver_stream.py
r"""
Weaver Stream Handler for ASTRA - SIMPLIFIED VERSION

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

LOCKED WEAVER BEHAVIOUR (v3.5):
- Purpose: Convert human rambling into a structured job outline
- NOT a full spec builder - just a text organizer
- Reads messages to get input (the ramble)
- Does NOT persist to specs table
- Does NOT build JSON specs
- Does NOT resolve ambiguities or contradictions
- ALWAYS outputs structured outline (never conversational responses)
- May ask up to 3-5 SHALLOW framing questions (platform, look/feel, controls, scope, layout)
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
# Design Job Detection
# ---------------------------------------------------------------------------

DESIGN_JOB_INDICATORS = [
    "app", "ui", "interface", "page", "screen", "dashboard",
    "website", "form", "button", "layout", "component", "modal",
    "sidebar", "navbar", "menu", "panel", "widget", "view",
    "frontend", "front-end", "front end", "web app", "webapp",
    "game",  # v3.5.0: Games also need design questions
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
# Shallow Question Generation (v3.5.0 - Bug 5 fix)
# Weaver only asks 3-5 HIGH LEVEL questions, never technical
# ---------------------------------------------------------------------------

# v3.5.0: Shallow framing questions only (Bug 5 - Scope Boundary Enforcement)
# These are the ONLY types of questions Weaver is allowed to ask
SHALLOW_QUESTIONS = {
    "platform": "What platform do you want this on? (web / Android / desktop / iOS)",
    "look_feel": "Dark mode or light mode?",
    "style": "Minimal or arcade-style / fancy?",
    "controls": "What controls? (keyboard / touch / controller)",
    "scope": "Bare minimum playable first, or add some extras?",
    "layout": "Any preference on layout? (centered vs sidebar HUD)",
}

# Keywords that indicate a question type is already answered
SHALLOW_QUESTION_KEYWORDS = {
    "platform": ["web", "android", "ios", "desktop", "mobile", "browser", "windows", "mac", "linux"],
    "look_feel": ["dark mode", "light mode", "dark", "light", "bright", "night"],
    "style": ["minimal", "minimalist", "arcade", "fancy", "retro", "classic", "modern", "playful", "simple"],
    "controls": ["keyboard", "touch", "controller", "mouse", "wasd", "arrows", "swipe", "gamepad"],
    "scope": ["minimal", "basic", "extras", "features", "bare", "simple", "full", "complete"],
    "layout": ["centered", "sidebar", "hud", "split", "fullscreen"],
}


def _get_shallow_questions(ramble_text: str) -> Dict[str, str]:
    """
    Get shallow framing questions that haven't been answered yet.
    
    v3.5.0: Weaver is limited to 3-5 shallow questions maximum.
    These are high-level framing questions only, never technical.
    
    ALLOWED: platform, look/feel, controls, scope, layout
    NOT ALLOWED: frameworks, algorithms, architecture, data structures
    """
    text_lower = ramble_text.lower()
    questions = {}
    
    for q_type, keywords in SHALLOW_QUESTION_KEYWORDS.items():
        if not any(kw in text_lower for kw in keywords):
            questions[q_type] = SHALLOW_QUESTIONS[q_type]
    
    # Limit to 5 questions max
    return dict(list(questions.items())[:5])


def _detect_filled_slots(ramble_text: str) -> Dict[str, str]:
    """
    Detect which shallow question slots have been answered in user messages.
    
    v3.5.1: Deterministic slot detection for reconciliation.
    Returns a dict of slot_name -> detected_value for filled slots.
    """
    text_lower = ramble_text.lower()
    filled_slots = {}
    
    # Platform detection
    platform_patterns = [
        (r"\bandroid\b", "Android"),
        (r"\bios\b", "iOS"),
        (r"\biphone\b", "iOS"),
        (r"\bipad\b", "iOS"),
        (r"\bweb\b", "Web"),
        (r"\bbrowser\b", "Web"),
        (r"\bdesktop\b", "Desktop"),
        (r"\bwindows\b", "Windows"),
        (r"\bmac\b", "Mac"),
        (r"\blinux\b", "Linux"),
        (r"\bmobile\b", "Mobile"),
    ]
    for pattern, value in platform_patterns:
        if re.search(pattern, text_lower):
            filled_slots["platform"] = value
            break
    
    # Color mode / theme detection
    if re.search(r"\bdark\s*mode\b|\bdark\s+theme\b|\bdark\b", text_lower):
        filled_slots["look_feel"] = "Dark mode"
    elif re.search(r"\blight\s*mode\b|\blight\s+theme\b|\blight\b|\bbright\b", text_lower):
        filled_slots["look_feel"] = "Light mode"
    
    # Controls detection
    controls_patterns = [
        (r"\btouch\b|\btap\b|\bswipe\b", "Touch"),
        (r"\bkeyboard\b|\bwasd\b|\barrow\s*keys?\b", "Keyboard"),
        (r"\bcontroller\b|\bgamepad\b|\bjoystick\b", "Controller"),
        (r"\bmouse\b|\bclick\b", "Mouse"),
    ]
    for pattern, value in controls_patterns:
        if re.search(pattern, text_lower):
            filled_slots["controls"] = value
            break
    
    # Scope detection
    if re.search(r"\bbare\s*minimum\b|\bminimal\b|\bbasic\b|\bsimple\b|\bjust\s+the\s+basics?\b", text_lower):
        filled_slots["scope"] = "Bare minimum / basic"
    elif re.search(r"\bextras?\b|\bfull\b|\bcomplete\b|\bfeatures?\b|\badvanced\b", text_lower):
        filled_slots["scope"] = "With extras / features"
    
    # Layout detection
    if re.search(r"\bcentered\b|\bcenter\b|\bfull\s*screen\b|\bfullscreen\b", text_lower):
        filled_slots["layout"] = "Centered / fullscreen"
    elif re.search(r"\bsidebar\b|\bhud\b|\bsplit\b", text_lower):
        filled_slots["layout"] = "Sidebar / HUD"
    
    return filled_slots


# ---------------------------------------------------------------------------
# Slot Reconciliation (v3.5.1 - Question Regression Fix)
# Deterministically removes filled slots from Unresolved/Questions sections
# ---------------------------------------------------------------------------

# v3.5.2: Patterns for detecting slot-related lines in output sections
# CRITICAL: Must match BOTH "unspecified" AND "not specified" variations
SLOT_AMBIGUITY_PATTERNS = {
    "platform": [
        r"platform\s*(is\s+)?(not\s+specified|unspecified|unclear)",
        r"target\s+platform\s*(is\s+)?(not\s+specified|unspecified|unclear)",
        r"platform.*(not\s+specified|unspecified|unclear)",
        r"which\s+platform",
        r"web.*android.*desktop.*ios.*(not\s+specified|unspecified)",
    ],
    "look_feel": [
        r"color\s*mode\s*(is\s+)?(not\s+specified|unspecified|unclear)",
        r"visual\s+theme.*(not\s+specified|unspecified|unclear)",
        r"dark\s*(vs|or|/|versus)\s*light.*(not\s+specified|unspecified|unclear)",
        r"dark.*light.*(not\s+specified|unspecified|unclear)",
        r"theme\s*(is\s+)?(not\s+specified|unspecified|unclear)",
        r"(dark|light)\s*mode.*(not\s+specified|unspecified|unclear)",
    ],
    "controls": [
        r"control\s*(method\s*)?(is\s+)?(not\s+specified|unspecified|unclear)",
        r"controls?\s*(is\s+|are\s+)?(not\s+specified|unspecified|unclear)",
        r"keyboard.*touch.*controller.*(not\s+specified|unspecified|unclear)",
        r"input\s*(method\s*)?(is\s+)?(not\s+specified|unspecified|unclear)",
        r"exact\s+control.*unspecified",
        r"key\s*mappings?.*(not\s+specified|unspecified|unclear)",
    ],
    "scope": [
        r"scope\s*(is\s+)?(not\s+specified|unspecified|unclear)",
        r"scope.*level\s*(is\s+)?(not\s+specified|unspecified|unclear)",
        r"bare\s*minimum.*extras?.*(not\s+specified|unspecified|unclear)",
        r"minimum\s*(vs|or|/)\s*extras?.*(not\s+specified|unspecified|unclear)",
    ],
    "layout": [
        r"layout\s*(preference\s*)?(is\s+)?(not\s+specified|unspecified|unclear)",
        r"layout.*(not\s+specified|unspecified|unclear)",
        r"hud\s*placement.*(not\s+specified|unspecified|unclear)",
        r"centered.*sidebar.*hud.*(not\s+specified|unspecified|unclear)",
    ],
}

# v3.5.2: Question patterns - must match actual LLM output variations
SLOT_QUESTION_PATTERNS = {
    "platform": [
        r"what\s+platform",
        r"which\s+platform",
        r"web.*android.*desktop.*ios",
        r"platform.*\?",
        r"target\s+platform",
    ],
    "look_feel": [
        r"dark\s*(mode)?\s*(or|vs|/)\s*light\s*(mode)?",
        r"color\s*(mode|scheme)\s*\?",
        r"theme\s*\?",
        r"visual\s+theme",
        r"dark.*light.*\?",
    ],
    "controls": [
        r"what\s+controls",
        r"keyboard.*touch.*controller",
        r"controls?\s*\?",
        r"input\s*(method)?\s*\?",
        r"control\s+method",
        r"key\s*mappings?",
    ],
    "scope": [
        r"bare\s*minimum.*extras?",
        r"minimum.*playable.*extras?",
        r"scope\s*\?",
        r"basic.*or.*full",
        r"minimal.*or.*complete",
    ],
    "layout": [
        r"layout\s*\?",
        r"centered.*sidebar",
        r"preference\s*on\s*layout",
        r"hud\s*placement",
        r"sidebar.*hud",
    ],
}


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
NON_MICRO_INDICATORS = [
    "app", "application", "website", "page", "component", "feature",
    "game", "dashboard", "ui", "interface", "api", "endpoint", "service",
    "database", "system", "platform", "design", "develop", "implement",
    "prototype", "demo", "all files", "every file", "entire", "batch",
    "multiple files", "refactor", "restructure", "migrate",
]

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

# Micro-task system prompt (v3.6.0)
MICRO_TASK_SYSTEM_PROMPT = """You are Weaver for MICRO FILE TASKS.

Your job: Produce a SHORT, minimal job outline (10-20 lines max) for simple file operations.

## RULES FOR MICRO FILE TASKS:
1. NO questions about OS, platform, desktop, or exact filenames
2. NO questions about paths - the system will search and find
3. Output is MINIMAL - just enough for downstream to execute
4. Steps are simple and direct

## OUTPUT FORMAT (keep it short!):

What is being built: [Short description] (micro file task)
Intent: [One line - what to find and what to do with it]
Execution type: MICRO_FILE_TASK
Planned steps:
- [Step 1: Locate]
- [Step 2: Action]
- [Step 3: Output/Return]
Questions: [only if true blockers exist, otherwise "none"]

That's the entire output. No more.

## EXAMPLES:

Input: "Go to desktop, find folder test, read txt, answer question"
Output:
What is being built: Desktop question responder (micro file task)
Intent: Find Desktop/test/*.txt, read question, produce reply
Execution type: MICRO_FILE_TASK
Planned steps:
- Locate Desktop/test folder
- Read .txt file (prefer test.txt, else first .txt found)
- Generate reply from file content
- Return reply in chat
Questions: none

CRITICAL: Keep output under 20 lines. No framework/architecture discussion."""


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


def _is_micro_file_task(text: str) -> bool:
    """
    Detect simple file operations that need no questions (v3.6.0).
    
    Logic:
    - If BUILD_VERB + NON_MICRO_INDICATOR → NOT micro (it's a build job)
    - If any MICRO_FILE_INDICATOR present → IS micro
    - Otherwise → NOT micro
    
    CRITICAL: NON_MICRO only decisive when paired with a build verb!
    This prevents false negatives like "Open the app folder on desktop".
    """
    text_lower = text.lower()
    
    # Check for BUILD VERB + NON_MICRO combo (NOT micro)
    has_build_verb = any(v in text_lower for v in BUILD_VERBS)
    has_non_micro = any(ind in text_lower for ind in NON_MICRO_INDICATORS)
    
    if has_build_verb and has_non_micro:
        print(f"[WEAVER] NOT micro-task (build verb + non-micro indicator)")
        return False
    
    # Check for file operation indicators
    has_file_indicator = any(ind in text_lower for ind in MICRO_FILE_INDICATORS)
    
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


def _reconcile_filled_slots(output: str, filled_slots: Dict[str, str]) -> str:
    """
    Deterministically remove filled slots from Unresolved ambiguities and Questions.
    
    v3.5.1: This is the key fix for the question regression bug.
    v3.5.2: Fixed pattern matching - now matches "not specified" in addition to "unspecified"
    
    When user answers questions (e.g., "Android, Dark mode, centered"), subsequent
    Weaver runs must remove those slots from:
    - "Unresolved ambiguities" section
    - "Questions" section
    
    This is DETERMINISTIC post-processing, not relying on LLM compliance.
    """
    if not filled_slots:
        print("[WEAVER] Reconciliation: No filled slots detected, skipping")
        return output
    
    print(f"[WEAVER] Reconciliation: Processing {len(filled_slots)} filled slots: {list(filled_slots.keys())}")
    
    lines = output.split("\n")
    result_lines = []
    in_ambiguities_section = False
    in_questions_section = False
    removed_ambiguities = 0
    removed_questions = 0
    
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        
        # Detect section headers (various formats the LLM might output)
        is_ambiguity_header = any([
            line_lower.startswith("unresolved ambiguit"),
            line_lower.startswith("**unresolved ambiguit"),
            line_lower.startswith("## unresolved"),
            line_lower.startswith("### unresolved"),
            # Handle with or without colon
            "unresolved ambiguities" in line_lower,
        ])
        
        is_question_header = any([
            line_lower.startswith("questions"),
            line_lower.startswith("**questions"),
            line_lower.startswith("## questions"),
            line_lower.startswith("### questions"),
            line_lower == "questions" or line_lower == "questions:",
        ])
        
        # Detect other section headers (to know when we've left the section)
        is_other_header = any([
            line_lower.startswith("what is being"),
            line_lower.startswith("**what is being"),
            line_lower.startswith("intended outcome"),
            line_lower.startswith("**intended outcome"),
            line_lower.startswith("design preferences"),
            line_lower.startswith("**design preferences"),
            line_lower.startswith("constraints"),
            line_lower.startswith("**constraints"),
            line_lower.startswith("execution mode"),
            line_lower.startswith("**execution mode"),
            line_lower.startswith("new requirements"),
            line_lower.startswith("**new requirements"),
            line_lower.startswith("known requirements"),
            line_lower.startswith("**known requirements"),
            line_lower.startswith("job description"),
            line_lower.startswith("**job description"),
            line_lower.startswith("next step"),
            line_lower.startswith("**next step"),
        ])
        
        # Track which section we're in
        if is_ambiguity_header:
            in_ambiguities_section = True
            in_questions_section = False
            print(f"[WEAVER] Reconciliation: Entered AMBIGUITIES section at line {i}")
            result_lines.append(line)
            continue
        elif is_question_header:
            in_questions_section = True
            in_ambiguities_section = False
            print(f"[WEAVER] Reconciliation: Entered QUESTIONS section at line {i}")
            result_lines.append(line)
            continue
        elif is_other_header and (in_ambiguities_section or in_questions_section):
            if in_ambiguities_section:
                print(f"[WEAVER] Reconciliation: Left AMBIGUITIES section at line {i}")
            if in_questions_section:
                print(f"[WEAVER] Reconciliation: Left QUESTIONS section at line {i}")
            in_ambiguities_section = False
            in_questions_section = False
        
        # Check if this line should be removed (filled slot)
        should_remove = False
        matched_slot = None
        matched_pattern = None
        
        if in_ambiguities_section and line_lower and not line_lower.startswith("#"):
            # Check if this line mentions a filled slot
            for slot_name in filled_slots:
                patterns = SLOT_AMBIGUITY_PATTERNS.get(slot_name, [])
                for pattern in patterns:
                    if re.search(pattern, line_lower, re.IGNORECASE):
                        should_remove = True
                        matched_slot = slot_name
                        matched_pattern = pattern
                        removed_ambiguities += 1
                        break
                if should_remove:
                    break
            
            if should_remove:
                print(f"[WEAVER] REMOVED ambiguity ({matched_slot}): '{line.strip()[:60]}' [matched: {matched_pattern}]")
        
        elif in_questions_section and line_lower and not line_lower.startswith("#"):
            # Check if this line is a question about a filled slot
            for slot_name in filled_slots:
                patterns = SLOT_QUESTION_PATTERNS.get(slot_name, [])
                for pattern in patterns:
                    if re.search(pattern, line_lower, re.IGNORECASE):
                        should_remove = True
                        matched_slot = slot_name
                        matched_pattern = pattern
                        removed_questions += 1
                        break
                if should_remove:
                    break
            
            if should_remove:
                print(f"[WEAVER] REMOVED question ({matched_slot}): '{line.strip()[:60]}' [matched: {matched_pattern}]")
        
        if not should_remove:
            result_lines.append(line)
    
    total_removed = removed_ambiguities + removed_questions
    if total_removed > 0:
        print(f"[WEAVER] Slot reconciliation COMPLETE: removed {removed_ambiguities} ambiguities, {removed_questions} questions")
    else:
        print(f"[WEAVER] Slot reconciliation: NO MATCHES FOUND (check patterns vs actual output)")
        # Debug: Show what's in the sections we scanned
        if filled_slots:
            print(f"[WEAVER] Debug: Filled slots were: {filled_slots}")
    
    return "\n".join(result_lines)


def _add_known_requirements_section(output: str, filled_slots: Dict[str, str]) -> str:
    """
    Add a "Known requirements" section showing filled slots at the top.
    
    v3.5.1: Makes it explicit what has been answered.
    """
    if not filled_slots:
        return output
    
    # Build known requirements section
    known_lines = ["\n**Known requirements:**"]
    
    slot_display_names = {
        "platform": "Target platform",
        "look_feel": "Color mode",
        "controls": "Controls",
        "scope": "Scope",
        "layout": "Layout",
    }
    
    for slot_name, value in filled_slots.items():
        display_name = slot_display_names.get(slot_name, slot_name.replace("_", " ").title())
        known_lines.append(f"- {display_name}: {value}")
    
    known_section = "\n".join(known_lines)
    
    # Find where to insert (after "Intended outcome" or "Design preferences")
    lines = output.split("\n")
    insert_idx = -1
    
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        # Insert after design preferences, intended outcome, or what is being built
        if any([
            line_lower.startswith("intended outcome"),
            line_lower.startswith("**intended outcome"),
            line_lower.startswith("design preferences"),
            line_lower.startswith("**design preferences"),
        ]):
            # Look for the end of this section (next blank line or header)
            for j in range(i + 1, len(lines)):
                if not lines[j].strip() or lines[j].strip().startswith("**"):
                    insert_idx = j
                    break
            if insert_idx == -1:
                insert_idx = i + 1
            break
    
    if insert_idx == -1:
        # Fallback: insert after first line
        insert_idx = 1
    
    # Check if "Known requirements" already exists
    output_lower = output.lower()
    if "known requirements" in output_lower:
        return output  # Already has it
    
    lines.insert(insert_idx, known_section)
    return "\n".join(lines)


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
    print(f"[WEAVER] Starting weaver v3.5.0 for project_id={project_id}")
    logger.info("[WEAVER] Starting weaver v3.5.0 for project_id=%s", project_id)
    
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
        # =====================================================================
        
        # Filter to USER messages only (prevents assistant pollution)
        user_messages_only = [m for m in filtered_messages if m.get("role") == "user"]
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
        
        # Format ramble text from USER messages (with meta-mode removed)
        ramble_text = _format_ramble(user_messages_only)
        
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
        if is_micro_task:
            print("[WEAVER] MICRO_FILE_TASK mode - minimal output, no unnecessary questions")
        
        # =====================================================================
        # STEP 6: Get questions based on task type (v3.6.0 modified)
        # - Micro tasks: blocker-only questions
        # - Design jobs: existing shallow question logic
        # =====================================================================
        
        shallow_questions = {}
        blocking_questions = []
        
        if is_micro_task:
            # Micro tasks: blocker-only questions
            blocking_questions = _get_blocking_questions(ramble_text, is_micro_task=True)
            if blocking_questions:
                print(f"[WEAVER] Micro-task has {len(blocking_questions)} blocking question(s)")
        elif _is_design_job(ramble_text):
            # Design jobs: existing shallow question logic
            shallow_questions = _get_shallow_questions(ramble_text)
            print(f"[WEAVER] Generated {len(shallow_questions)} shallow questions")
        
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
        
        # Build shallow questions context
        questions_context = ""
        if shallow_questions:
            q_list = "\n".join([f"- {q}" for q in shallow_questions.values()])
            questions_context = f"\n\nShallow questions to include in output:\n{q_list}"
        
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
            
            # v3.5.0: COMPLETELY REWRITTEN PROMPT with scope boundaries (Bug 5)
            system_prompt = """You are Weaver, a SHALLOW text organizer.

Your ONLY job: Take the human's rambling and restructure it into a minimal, stable job outline.

## What You DO:
- Extract the core goal as a SHORT NOUN PHRASE (not a full sentence)
- Summarize intent into "What is being built" and "Intended outcome" (DIFFERENT wording, no duplication)
- List unresolved ambiguities at high level
- Include up to 3-5 SHALLOW framing questions if provided
- Include execution mode if extracted from meta-phrases

## What You DO NOT DO (CRITICAL - SCOPE BOUNDARY):
- NO framework/library choices (don't suggest Pygame, React, etc.)
- NO file structure discussion
- NO algorithm talk (collision detection, rotations, data structures)
- NO architecture proposals
- NO implementation plans
- NO technical questions
- NO resolving ambiguities yourself

## Output Format:
Produce a MINIMAL structured job outline with these sections:
- **What is being built**: Short noun phrase (e.g., "Classic Tetris game")
- **Intended outcome**: Different wording (e.g., "Playable Tetris implementation")
- **Execution mode**: Only if extracted (e.g., "Discussion only, no code yet")
- **Design preferences**: Only if specified
- **Constraints**: Only if explicitly stated
- **Unresolved ambiguities**: List what's unclear (platform, style, controls, etc.)
- **Questions**: 3-5 shallow framing questions ONLY (platform, look/feel, controls, scope, layout)

## DEDUPLICATION RULE (Bug 3):
"What is being built" and "Intended outcome" must use DIFFERENT words.
BAD: What: "A Tetris game" / Outcome: "A Tetris game"
GOOD: What: "Classic Tetris game" / Outcome: "Playable Tetris implementation"

## SHALLOW QUESTIONS ONLY (Bug 5):
Questions must be HIGH-LEVEL framing only:
ALLOWED: "Web or desktop?", "Dark mode or light?", "Keyboard or touch?"
NOT ALLOWED: "Use Pygame or Arcade?", "How should rotation work?", "What data structure for blocks?"

## Critical Rule:
If the human didn't say it, it doesn't appear in your output.
You are a TEXT ORGANIZER, not a solution designer."""

            user_prompt = f"""Organize this conversation into a job description:

{ramble_text}{prefs_context}{exec_mode_context}{questions_context}

Remember: 
- Only include what was actually said
- Preserve any ambiguities (list them, don't resolve them)
- Keep What and Outcome DIFFERENT (no duplication)
- Only ask shallow framing questions (platform, look/feel, controls, scope, layout)
- NO technical questions (frameworks, algorithms, architecture)"""
        
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
        
        # v3.5.1: Detect filled slots from ALL user messages
        filled_slots = _detect_filled_slots(ramble_text)
        if filled_slots:
            print(f"[WEAVER] Detected filled slots: {filled_slots}")
        
        # v3.5.1: Reconcile slots - remove answered questions/ambiguities (CRITICAL FIX)
        reconciled_output = _reconcile_filled_slots(dedup_output, filled_slots)
        
        # v3.5.1: Add "Known requirements" section for filled slots
        job_description = _add_known_requirements_section(reconciled_output, filled_slots)
        
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
