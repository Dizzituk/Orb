# FILE: app/llm/weaver_stream.py
r"""
Weaver Stream Handler for ASTRA - SIMPLIFIED VERSION

v4.2.0 (2026-02-06): SPECGATE DIRECTIVE HANDOFF - Stop asking user code-answerable questions
- CRITICAL CHANGE: Weaver now splits gaps into two categories:
  1. "Questions for user" — ONLY subjective/preference gaps (visual style, UX feel, naming,
     business priorities) that cannot be determined from code
  2. "SpecGate must resolve" — Implementation gaps that SpecGate can answer by scanning
     the codebase (endpoint patterns, error conventions, response formats, existing APIs)
- Weaver NEVER asks the user about things the code can tell it
- SpecGate directives are explicit instructions telling SpecGate what to look for
- This eliminates the 3-round question ping-pong that blocked pipeline flow
- Updated CREATE mode, UPDATE mode, and LOCKED BEHAVIOUR docs

v4.1.0 (2026-02-06): AUTO-REWEAVE RACE CONDITION FIX
- CRITICAL FIX: User replies during auto-reweave were invisible to hash-based dedup
- Root cause: stream_router fires Weaver UPDATE before user message is persisted to DB
- _gather_ramble_messages() reads DB, sees only old messages, hashes match → "Nothing new"
- Fix: Added pending_user_message parameter to generate_weaver_stream()
- stream_router passes req.message directly so Weaver sees it regardless of DB timing
- Hash-based dedup prevents double-counting if message IS already in DB
- Version bump to v4.1.0 in all log markers

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

LOCKED WEAVER BEHAVIOUR (v4.2):
- Purpose: Convert human rambling into a structured job outline
- NOT a full spec builder - just a text organizer
- Reads messages to get input (the ramble)
- Does NOT persist to specs table
- Does NOT build JSON specs
- Does NOT resolve ambiguities or contradictions
- ALWAYS outputs structured outline (never conversational responses)
- TWO types of gap handling:
  1. "Questions for user" - ONLY subjective/preference gaps (colours, UX feel, naming)
  2. "SpecGate must resolve" - Code-answerable gaps delegated to SpecGate
- NEVER asks the user about implementation patterns, conventions, or anything the code can answer
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
from app.llm._weaver_stream_utils import _SLOT_RECONCILIATION_REMOVED, _enforce_deduplication, _format_execution_mode, _format_ramble, _get_blocking_questions, _get_weaver_config, _is_control_message, _serialize_sse
from app.llm._weaver_stream_utils import BUILD_VERBS, INTENT_GOAL_PATTERNS, LEAKAGE_PATTERNS, MICRO_FILE_INDICATORS, NEGATION_PATTERNS, NON_MICRO_INDICATORS, REFACTOR_INDICATORS, _hash_messages
from app.llm._weaver_stream_utils import CORE_GOAL_VERBS, FEATURE_COMPONENT_INDICATORS, META_MODE_PATTERNS, QUESTIONS_DISMISSED_PATTERNS, _get_streaming_function, _hash_message, _normalize_typos, _sanitize_weaver_output
from app.llm._weaver_stream_utils import DESIGN_PREF_WHITELIST_PATTERNS, REFACTOR_ACTION_PATTERNS, TYPO_NORMALIZATIONS, _extract_meta_mode, _extract_vision_context, _gather_ramble_messages, _is_micro_file_task, _user_dismissed_questions
from app.llm._weaver_stream_utils import CONCRETE_TARGETS, DESIGN_PREF_BLACKLIST_PATTERNS, MICRO_TASK_SYSTEM_PROMPT, REFACTOR_TASK_SYSTEM_PROMPT, VISION_CONTEXT_PATTERNS, _enforce_design_pref_hygiene, _is_refactor_task, _is_vision_context
from app.llm._weaver_stream_utils import CORE_GOAL_TARGETS, _has_core_goal

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
    from app.memory import schemas as memory_schemas
    _MEMORY_AVAILABLE = True
except ImportError:
    memory_service = None
    memory_schemas = None
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


# ---------------------------------------------------------------------------
# Message Hashing (v3.4) - For durable delta tracking
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Vision Context Detection (v3.9.0) - Preserve valuable assistant analysis
# ---------------------------------------------------------------------------

# Patterns that indicate an assistant message contains vision/image analysis
# These messages should NOT be filtered out even though they're from assistant


# ---------------------------------------------------------------------------
# Meta-Mode Extraction (v3.5.0 - Bug 2 fix)
# Separates pipeline control language from product requirements
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Output Sanitization (v3.4) - Strip prompt leakage
# ---------------------------------------------------------------------------

# Patterns that indicate prompt scaffold leaked into output

# Patterns that ARE valid in Design preferences (visual/UI only)

# Patterns that should NOT be in Design preferences (functional requirements)


# ---------------------------------------------------------------------------
# Core Goal Detection (2-Factor Heuristic) - v3.5.0
# ---------------------------------------------------------------------------

# Action verbs that indicate intent

# Intent patterns that express desire/need (must be paired with a target)
# These are NOT action verbs but indicate the user wants something done

# Target/directive words that indicate what the action applies to
# v3.5.0: Added creative/project targets (game, prototype, demo, etc.) - Bug 1 fix

# Concrete targets - subset that counts for intent patterns
# These are specific enough that "I want a <target>" is a clear goal
# v3.5.0: Added creative/project targets (game, prototype, demo, etc.) - Bug 1 fix

# Negation patterns that invalidate a goal


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




# ---------------------------------------------------------------------------
# Micro-Task Classification (v3.6.0)
# Detect simple file operations that need no questions
# ---------------------------------------------------------------------------

# Indicators FOR micro-task (simple file operations)

# Indicators AGAINST micro-task (only when paired with build verbs)
# v3.6.1: REMOVED "system" - "on my system" is file context, not software system
# v3.6.1: REMOVED "platform" - "on desktop" is a location, not software platform

# v3.7: REFACTOR/RENAME operations should NEVER be micro-tasks
# v3.10: Moved REFACTOR_INDICATORS below (cleaned up, app-name entries removed)
# v3.10: Added REFACTOR_ACTION_PATTERNS for context-aware detection

# v3.10: Refactor detection patterns — context-aware, not keyword-only
# These patterns detect ACTUAL rename/refactor intent, not just keyword presence.
# Each pattern requires a refactor ACTION + a SCOPE or TARGET indicator.

# v3.8: Patterns that indicate user dismissed/answered questions

# v3.8: Refactor task system prompt - NO design questions, focused on search/replace
# v3.10: Legacy list kept ONLY for _is_micro_file_task guard.
# Stripped of app-name-specific and overly-generic entries.
# The real refactor detection now uses REFACTOR_ACTION_PATTERNS above.

# Build verbs that make NON_MICRO_INDICATORS decisive

# Silent typo normalizations (v3.6.0)
# Uses word boundaries to avoid substring collisions

# Micro-task system prompt (v3.6.1 - STRICTER, no unnecessary questions)


# v3.11: Feature component indicators - multi-component requests are NEVER micro
# If 3+ of these appear in a request, it's a substantial feature, not a file task


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
    pending_user_message: Optional[str] = None,
) -> AsyncIterator[bytes]:
    """
    Weaver handler - v4.1.0 with FULL BUG FIXES.
    
    v4.1.0 CHANGES:
    - CRITICAL FIX: Added pending_user_message parameter for auto-reweave race condition
    - When stream_router auto-routes to Weaver UPDATE, the user's reply may not yet
      be persisted to the DB. pending_user_message is injected directly into the
      message list so hash-based dedup sees it as new content, preventing the
      false "Nothing new to weave" response.
    
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
    print(f"[WEAVER] Starting weaver v4.2.0 for project_id={project_id}")
    logger.info("[WEAVER] Starting weaver v4.2.0 for project_id=%s", project_id)
    
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
        
        # =============================================================
        # v4.1.0: INJECT PENDING USER MESSAGE (auto-reweave race fix)
        # When stream_router auto-routes to Weaver UPDATE, the user's
        # latest message may not yet be persisted to the DB (the SSE
        # handler fires before message persistence completes). We
        # inject it directly so hash-based dedup sees it as new.
        # Dedup by hash ensures no double-counting if it IS in DB.
        # =============================================================
        if pending_user_message and pending_user_message.strip():
            pending_msg = {"role": "user", "content": pending_user_message.strip()}
            # Only inject if not already present (hash check prevents duplicates)
            pending_hash = _hash_message(pending_msg)
            existing_hashes = _hash_messages(all_messages)
            if pending_hash not in existing_hashes:
                all_messages.append(pending_msg)
                print(f"[WEAVER] v4.1.0 Injected pending_user_message ({len(pending_user_message)} chars, hash={pending_hash})")
            else:
                print(f"[WEAVER] v4.1.0 pending_user_message already in DB (hash={pending_hash}), skipping injection")
        
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
        
        # =====================================================================
        # v5.5 PHASE 4C: Progressive Memory — compact long conversations
        # =====================================================================
        _compaction_applied = False
        if len(relevant_messages) >= 15:
            try:
                from app.llm.weaver_memory import compact_conversation
                _compaction_result = await compact_conversation(relevant_messages)
                if _compaction_result.was_compacted:
                    ramble_text = _compaction_result.format_for_weaver()
                    _compaction_applied = True
                    logger.info(
                        "[WEAVER] v5.5 Progressive memory: %d messages compacted "
                        "(%d distilled → %d chars, %d verbatim)",
                        _compaction_result.total_messages,
                        _compaction_result.compacted_count,
                        len(_compaction_result.distilled_summary),
                        _compaction_result.preserved_count,
                    )
                    print(
                        f"[WEAVER] v5.5 COMPACTION: {_compaction_result.compacted_count} old → "
                        f"{len(_compaction_result.distilled_summary)} char summary, "
                        f"{_compaction_result.preserved_count} recent kept verbatim"
                    )
                else:
                    logger.debug("[WEAVER] v5.5 Compaction skipped: %s", _compaction_result.skip_reason)
            except (ImportError, Exception) as _compact_err:
                logger.debug("[WEAVER] v5.5 Progressive memory unavailable: %s", _compact_err)

        # Format ramble text from RELEVANT messages (USER + vision context)
        # v3.9.0: Now includes vision analysis for context
        if not _compaction_applied:
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
            
            # v5.5 PHASE 4C: Compact new messages if there are many
            _update_compacted = False
            if len(new_user_messages) >= 15:
                try:
                    from app.llm.weaver_memory import compact_conversation
                    _update_compact = await compact_conversation(new_user_messages)
                    if _update_compact.was_compacted:
                        new_ramble = _update_compact.format_for_weaver()
                        _update_compacted = True
                        print(f"[WEAVER] v5.5 UPDATE compaction: {_update_compact.compacted_count} distilled + {_update_compact.preserved_count} verbatim")
                except (ImportError, Exception):
                    pass
            if not _update_compacted:
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
10. If the previous spec has a "SpecGate must resolve" section, KEEP it and add new directives if needed
11. NEVER add code-answerable questions to "Questions for user" - those go in "SpecGate must resolve"

OUTPUT FORMAT:
- Output ONLY the complete updated job description
- Start directly with the content (e.g., "What is being built or changed")
- Include "Execution mode" section if provided
- Preserve "SpecGate must resolve" section (add new directives from new requirements)
- "Questions for user" should ONLY contain subjective/preference gaps (visual, UX, naming)
- Do NOT include any preamble or explanation
- Do NOT echo any part of these instructions"""

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
            
            # v4.2.0: SPECGATE DIRECTIVE HANDOFF - Two-tier gap handling
            system_prompt = """You are Weaver, a SHALLOW text organizer.

Your ONLY job: Take the human's rambling and restructure it into a minimal, stable job outline.

## What You DO:
- Extract the core goal as a SHORT NOUN PHRASE (not a full sentence)
- Summarize intent into "What is being built" and "Intended outcome" (DIFFERENT wording, no duplication)
- Faithfully list ALL requirements, constraints, and specifications the user provided
- List unresolved ambiguities at high level
- Classify any gaps into TWO categories (see GAP HANDLING below)
- Include execution mode if extracted from meta-phrases

## What You DO NOT DO (CRITICAL - SCOPE BOUNDARY):
- NO framework/library choices (don't suggest specific libraries or tools)
- NO file structure discussion
- NO algorithm or data structure talk
- NO architecture proposals
- NO implementation plans
- NO resolving ambiguities yourself
- NO inventing requirements the user didn't state
- NEVER ask the user about implementation patterns, conventions, or technical details

## GAP HANDLING (v4.2 - CRITICAL NEW BEHAVIOUR):

When you identify gaps in the requirements, you MUST classify each gap into exactly one of
two categories. Getting this classification right is your most important job.

### Category 1: "Questions for user" — ASK THE HUMAN
ONLY for subjective decisions that NO amount of code scanning could answer:
- Visual/aesthetic preferences (colour scheme, theme, visual style)
- UX feel and interaction preferences ("should it feel snappy or smooth?")
- Naming/branding choices (what to call things in the UI)
- Business logic priorities (which feature matters more, what tradeoffs to make)
- Target audience or persona preferences
- Emotional/tonal qualities ("playful vs professional")

THESE ARE RARE. Most requests have zero questions for the user. Default to NONE.
Maximum: 2 questions. If you can't limit to 2, you're asking about the wrong things.

### Category 2: "SpecGate must resolve" — DELEGATE TO THE PIPELINE
For ANY gap that could be answered by reading the existing codebase:
- Endpoint conventions (input format, response shape, error patterns)
- Existing service APIs and how to integrate with them
- File structure and where new code should go
- Database schema patterns and existing models
- Authentication/authorization patterns
- Configuration conventions (env vars, config files)
- Testing patterns and conventions
- Import paths and module organization
- Any "how does the existing system do X?" question

These become explicit directives telling SpecGate what to investigate.
Write them as actionable investigation tasks, e.g.:
- "Determine the endpoint input format convention by examining app/endpoints/"
- "Identify the error response pattern used across existing FastAPI routers"
- "Find how existing services are registered in main.py"

### THE GOLDEN RULE:
If the answer COULD exist somewhere in the codebase → SpecGate must resolve.
If the answer can ONLY come from the human's brain → Questions for user.
When in doubt → SpecGate must resolve. The pipeline is smarter than you think.

## Output Format:
Produce a MINIMAL structured job outline with these sections:
- **What is being built**: Short noun phrase (e.g., "Voice-to-text input system")
- **Intended outcome**: Different wording (e.g., "Local speech transcription integrated into desktop app")
- **Execution mode**: Only if extracted (e.g., "Discussion only, no code yet")
- **Key requirements**: Bullet list of what the user explicitly asked for
- **Design preferences**: Only if specified (visual/UI preferences only)
- **Constraints**: Only if explicitly stated by the user
- **Unresolved ambiguities**: Things genuinely unclear from the user's description
- **SpecGate must resolve**: Directives for SpecGate to investigate by scanning the codebase
  (this section is EXPECTED to have items — most implementation gaps belong here)
- **Questions for user**: ONLY subjective/preference gaps. Usually "none".
  (if you have items here, each MUST be something no code can answer)

## DEDUPLICATION RULE:
"What is being built" and "Intended outcome" must use DIFFERENT words.
BAD: What: "Voice input feature" / Outcome: "Voice input feature"
GOOD: What: "Voice-to-text input system" / Outcome: "Local speech transcription for desktop app"

## EXAMPLES OF CORRECT GAP CLASSIFICATION:

User says: "Add voice-to-text to the ASTRA desktop app using faster-whisper"

SpecGate must resolve:
- Determine the endpoint input format convention (multipart? raw body?) by examining existing endpoints in app/endpoints/
- Identify the standard response model pattern (Pydantic models, JSON shape) from existing routers
- Determine the error handling convention (HTTPException patterns, status codes) across the codebase
- Find how new FastAPI routers are registered in main.py
- Check if an audio processing dependency (PyAV/FFmpeg) is already in requirements

Questions for user: none
(The user specified the tool, the feature, and the platform. Everything else is code-answerable.)

---

User says: "Build me a dashboard"

SpecGate must resolve:
- Identify existing frontend component patterns and framework
- Determine the data sources available for dashboard widgets

Questions for user:
- What information should the dashboard show? (Only the user knows what they want to see)

## Critical Rules:
1. If the human didn't say it, it doesn't appear in your output.
2. If the human DID say it, it MUST appear in your output (don't drop requirements).
3. You are a TEXT ORGANIZER, not a solution designer.
4. Preserve the user's terminology and domain language.
5. NEVER put a code-answerable question in "Questions for user" — that's SpecGate's job."""

            user_prompt = f"""Organize this conversation into a job description:

{ramble_text}{prefs_context}{exec_mode_context}

Remember:
- Include ALL requirements the user stated (don't drop anything)
- Preserve any ambiguities (list them, don't resolve them)
- Keep What and Outcome DIFFERENT (no duplication)
- Code-answerable gaps go in "SpecGate must resolve" (NOT questions for user)
- Only put genuinely subjective/preference questions in "Questions for user"
- When in doubt, it's a SpecGate directive, not a user question
- Preserve the user's domain terminology"""
        
        # v3.0: Inject user memory into Weaver system prompt
        try:
            from app.experience.user_memory import get_user_context_for_conversation
            _user_conv_ctx = get_user_context_for_conversation(
                db, query=user_prompt[:300], project_id=project_id,
            )
            if _user_conv_ctx:
                system_prompt += f"\n\n{_user_conv_ctx}"
        except Exception:
            pass

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
        # Persist to message history for cross-model context continuity
        # =====================================================================
        if _MEMORY_AVAILABLE and memory_service and memory_schemas:
            try:
                memory_service.create_message(
                    db,
                    memory_schemas.MessageCreate(
                        project_id=project_id,
                        role="assistant",
                        content=job_description,
                        provider=provider,
                        model=model,
                    ),
                )
            except Exception as e:
                logger.warning("[WEAVER] Failed to save message to history: %s", e)

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
