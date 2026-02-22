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
from app.llm._weaver_stream_utils import generate_weaver_stream

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


# ---------------------------------------------------------------------------
# LEGACY COMPATIBILITY
# ---------------------------------------------------------------------------

__all__ = ["generate_weaver_stream"]
