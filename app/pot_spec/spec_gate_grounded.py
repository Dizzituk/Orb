# FILE: app/pot_spec/spec_gate_grounded.py
"""
SpecGate Contract v1 - Grounded POT Spec Builder

PURPOSE (non-negotiable):
SpecGate turns Weaver output (intent) into a grounded, implementable Point-of-Truth (POT) spec.
It exists to:
- Stop drift
- Remove ambiguity
- Anchor work in repo reality
- Ask only the questions that truly require the human

CORE DECISION RULE: Look first. Ask second. Never guess.

RUNTIME: STRICTLY READ-ONLY
- No filesystem writes (no artifacts, no files)
- No DB writes (even persistence tables)
- Output must be returned/streamed only

QUESTION RULES:
- Max 3-7 questions total, only high-impact
- Only ask when:
  1. Not derivable from evidence (code, structure, patterns, docs)
  2. High-impact (wrong answer causes rewrite / wasted days / wrong UX)
  3. A user preference / product decision (not an engineering fact)

EVIDENCE PRIORITY:
1. Latest architecture map
2. Latest codebase report
3. read/head/lines/find
4. arch_query fallback
5. Ask user (only if still unresolved)

v1.0 (2026-01): Initial Contract v1 implementation
v1.1 (2026-01): Fixed question generation + status logic (Contract v1 compliance)
v1.2 (2026-01): Decision forks replace lazy questions (Contract v1.2 compliance)
              - Round 1 asks bounded A/B/C product decisions, not "tell me steps/tests"
              - Round 2 derives steps/tests from domain + answered forks
              - Added domain detection and fork question bank
v1.3 (2026-01): Content-aware sandbox discovery wired (JOB 2 complete)
              - Sandbox jobs now auto-discover input file by content, not filename
              - Output path locked to same folder as input (reply.txt)
              - Ambiguity triggers question (same-type files with close scores)
              - Progressive reads: snippet for classification, full read for winner
v1.4 (2026-01): Question discipline upgrade (blocker-only gating)
              - Questions only asked if they change architecture/data model/acceptance
              - Non-blocking forks get safe defaults + recorded as assumptions
              - Early exit when spec is complete (no question-hunting)
              - Added GroundedAssumption dataclass for tracking defaults
v1.5 (2026-01): Decision tracking + conditional steps/tests
              - Added spec.decisions dict for resolved blocking forks
              - Assumptions populate every round (not just Round 1)
              - Steps/tests conditional on decisions (no "if selected" when decided)
              - OCR test references "Successfully Completed Parcels" correctly
              - Added "Resolved Decisions" section to markdown output
v1.6 (2026-01): Sandbox discovery fixes (MUST NOT silently skip)
              - Added "sandbox_file" domain detection for file tasks
              - Added explicit logging for sandbox discovery flow
              - BLOCKING: If sandbox hints detected but tools unavailable, return error
              - BLOCKING: If discovery runs but finds no file, return error
              - BLOCKING: If discovery returns empty or raises exception, return error
              - Added sandbox-specific steps and tests for sandbox_file domain
              - Added sandbox_discovery_status and sandbox_skip_reason tracking
              - Never silently validate a generic spec for sandbox tasks
v1.7 (2026-01): Read-only reply output fix
              - SpecGate is STRICTLY READ-ONLY: never claims to write files
              - Removed "Write reply to..." and "Verify output created" from steps/tests
              - Added sandbox_generated_reply field to include reply IN the SPoT output
              - Added "Reply (Read-Only)" section to markdown output
              - Changed constraint wording: "Planned output path (for later stages)" not "must be written"
              - Reply is dynamically generated from file content, not hardcoded
v1.8 (2026-01): LLM-powered intelligent reply generation + stopword parsing fix
              - _generate_reply_from_content() now uses LLM (via llm_call) for intelligent answers
              - Actually ANSWERS questions in files instead of generic acknowledgements
              - Explains code when file contains code snippets
              - Fallback to heuristics if LLM unavailable
              - Uses provider_id/model_id passed to run_spec_gate_grounded()
              - CRITICAL FIX: _extract_sandbox_hints() stopword filtering
                  - Added comprehensive SUBFOLDER_STOPWORDS set (on, in, at, to, of, for, etc.)
                  - Prevents "desktop/on" from ever being generated
                  - Improved pattern order (most specific first)
                  - Strips meta-instructions ("reply ok", "say ok when you understand")
                  - Better logging for debugging hint extraction
"""

from __future__ import annotations

import logging
import re
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# =============================================================================
# IMPORTS
# =============================================================================

# Evidence collector (primary)
try:
    from .evidence_collector import (
        EvidenceBundle,
        EvidenceSource,
        load_evidence,
        add_file_read_to_bundle,
        add_search_to_bundle,
        find_in_evidence,
        verify_path_exists,
        refuse_write_operation,
        WRITE_REFUSED_ERROR,
    )
    _EVIDENCE_AVAILABLE = True
except ImportError as e:
    logger.warning("[spec_gate_grounded] evidence_collector not available: %s", e)
    _EVIDENCE_AVAILABLE = False

# Types from spec_gate_types
try:
    from .spec_gate_types import SpecGateResult
except ImportError:
    # Define minimal result type
    @dataclass
    class SpecGateResult:
        ready_for_pipeline: bool = False
        open_questions: List[str] = field(default_factory=list)
        spot_markdown: Optional[str] = None
        db_persisted: bool = False
        spec_id: Optional[str] = None
        spec_hash: Optional[str] = None
        spec_version: Optional[int] = None
        hard_stopped: bool = False
        hard_stop_reason: Optional[str] = None
        notes: Optional[str] = None
        blocking_issues: List[str] = field(default_factory=list)
        validation_status: str = "pending"

# Sandbox inspection (read-only discovery) - v1.3
try:
    from app.llm.local_tools.zobie.sandbox_inspector import (
        run_sandbox_discovery_chain,
        file_exists_in_sandbox,
        read_sandbox_file,
        SANDBOX_ROOTS,
    )
    _SANDBOX_INSPECTOR_AVAILABLE = True
except ImportError as e:
    logger.warning("[spec_gate_grounded] sandbox_inspector not available: %s", e)
    _SANDBOX_INSPECTOR_AVAILABLE = False
    run_sandbox_discovery_chain = None

# LLM call for intelligent reply generation - v1.8
try:
    from app.providers.registry import llm_call, LlmCallResult
    _LLM_CALL_AVAILABLE = True
except ImportError as e:
    logger.warning("[spec_gate_grounded] llm_call not available: %s", e)
    _LLM_CALL_AVAILABLE = False
    llm_call = None


# =============================================================================
# CONSTANTS
# =============================================================================

# Question budget
MIN_QUESTIONS = 0
MAX_QUESTIONS = 7

# Question categories (allowed)
class QuestionCategory(str, Enum):
    PREFERENCE = "preference"           # UI style, tone, naming preference
    MISSING_PRODUCT_DECISION = "product_decision"  # New workflow, manual vs auto
    AMBIGUOUS_EVIDENCE = "ambiguous"    # Map says X, code says Y
    SAFETY_CONSTRAINT = "safety"        # Sandbox vs main, backwards compat
    DECISION_FORK = "decision_fork"     # v1.2: Bounded A/B/C product decision


# =============================================================================
# DECISION FORK SYSTEM (v1.2 - Contract v1.2 Compliance)
# =============================================================================
# 
# SpecGate asks ONLY bounded product decision forks (A/B/C choices).
# It does NOT ask "tell me the steps" or "tell me the acceptance criteria".
# Those are SpecGate's job to DERIVE after forks are answered.
#

# Domain detection keywords (case-insensitive)
DOMAIN_KEYWORDS = {
    "mobile_app": [
        "mobile app", "phone app", "android", "ios", "iphone",
        "offline-first", "offline first", "sync", "ocr", "screenshot",
        "voice", "push-to-talk", "push to talk", "wake word", "wakeword",
        "encryption", "encrypted", "trusted wi-fi", "trusted wifi",
        "in-van", "in van", "delivery", "parcels", "shift",
    ],
    # v1.6: Sandbox file tasks (read file, reply, find by discovery)
    "sandbox_file": [
        "sandbox", "sandbox desktop", "sandbox task",
        "find the file", "find by discovery", "discovery",
        "read the file", "read the question", "read the message",
        "reply", "include the reply", "include reply",
        "desktop folder", "folder called", "folder named",
        "text file", ".txt", "test.txt",
    ],
    # Future domains can be added here (e.g., "web_app", "cli_tool", "api_service")
}

# Fork question bank - templates for bounded A/B/C questions
# Each fork has: question, why_it_matters, options (A/B/C)
# Evidence is dynamically populated from Weaver text
#
# v1.4 BLOCKING CRITERIA:
# A question is BLOCKING only if the answer changes:
#   - Architecture/platform choice
#   - Required tooling (OCR vs manual)
#   - Data model / persistence approach  
#   - Acceptance criteria / definition of done
#   - Integration boundaries required for v1 correctness
#
# Non-blocking forks get safe defaults and are recorded as assumptions.
#
MOBILE_APP_FORK_BANK = [
    {
        "id": "platform_v1",
        "question": "Platform for v1 release?",
        "why_it_matters": "Determines SDK choice, build tooling, and timeline. iOS adds ~40% development time.",
        "options": ["Android-only first", "Android + iOS from day 1"],
        "triggers": ["android", "ios", "iphone", "mobile", "phone"],
        "blocking": True,  # Changes architecture fundamentally
        "default_value": None,
        "default_reason": None,
    },
    {
        "id": "offline_storage",
        "question": "Offline data storage approach?",
        "why_it_matters": "Affects data model complexity, encryption implementation, and sync conflict resolution.",
        "options": [
            "Room/SQLite + SQLCipher (structured, queryable)",
            "Encrypted file store (JSON + crypto, simpler but less queryable)",
        ],
        "triggers": ["offline", "encryption", "encrypted", "storage", "local"],
        "blocking": False,  # v1.4: Can default to simpler option for v1
        "default_value": "Local encrypted storage (SQLite + encryption for v1)",
        "default_reason": "Standard v1 approach; can upgrade to SQLCipher later if needed",
    },
    {
        "id": "input_mode_v1",
        "question": "Primary input mode for v1?",
        "why_it_matters": "Voice requires speech-to-text integration and error handling. Manual is simpler but slower in-van.",
        "options": [
            "Push-to-talk voice + manual fallback",
            "Voice-only (no manual entry)",
            "Manual-only (voice deferred to v2)",
        ],
        "triggers": ["voice", "push-to-talk", "push to talk", "manual", "input", "talk"],
        "blocking": True,  # Changes tooling (speech-to-text integration)
        "default_value": None,
        "default_reason": None,
    },
    {
        "id": "ocr_scope_v1",
        "question": "Screenshot OCR scope for v1?",
        "why_it_matters": "Multiple formats require more OCR training/templates. Single format is faster to ship.",
        "options": [
            "Finish Tour screenshot only",
            "Multiple screen formats (Finish Tour + route summary + others)",
        ],
        "triggers": ["ocr", "screenshot", "parse", "finish tour", "scan"],
        "blocking": True,  # Changes tooling (OCR templates needed)
        "default_value": None,
        "default_reason": None,
    },
    {
        "id": "sync_behaviour",
        "question": "Data sync behaviour?",
        "why_it_matters": "Auto-sync needs background service and battery optimization. Manual is simpler but requires user action.",
        "options": [
            "Manual sync only (user taps 'Sync now')",
            "Auto-sync on trusted Wi-Fi",
            "Both (manual + optional auto on trusted Wi-Fi)",
        ],
        "triggers": ["sync", "wi-fi", "wifi", "upload", "background"],
        "blocking": False,  # v1.4: NOT blocking - v1 is local-only by default
        "default_value": "Local-only for v1 (no cloud sync)",
        "default_reason": "v1 focus is local capture; sync deferred or via export",
    },
    {
        "id": "sync_target",
        "question": "Sync target for v1?",
        "why_it_matters": "Live endpoint requires server setup and auth. File export is portable but no real-time.",
        "options": [
            "Private ASTRA endpoint over LAN/VPN",
            "Export/import file only (no live endpoint yet)",
        ],
        "triggers": ["astra", "endpoint", "server", "export", "private", "lan", "vpn"],
        "blocking": False,  # v1.4: NOT blocking - can be export-only placeholder
        "default_value": "Export file for ASTRA integration (no live endpoint for v1)",
        "default_reason": "ASTRA integration via file export/import; live endpoint deferred",
    },
    {
        "id": "knockon_tracking",
        "question": "Knock-on day tracking method?",
        "why_it_matters": "Inferred rules need pattern detection logic. Manual toggle is explicit but requires user discipline.",
        "options": [
            "Manual toggle per day ('Today is a knock-on day: yes/no')",
            "Inferred from Tue/Thu pattern (auto-detected)",
        ],
        "triggers": ["knock-on", "knockon", "knock on", "tuesday", "thursday", "reschedule"],
        "blocking": False,  # v1.4: NOT blocking - manual toggle is safe default
        "default_value": "Manual toggle per day",
        "default_reason": "Simpler for v1; pattern detection can be added later",
    },
    {
        "id": "pay_variation",
        "question": "Pay-per-parcel variation handling?",
        "why_it_matters": "Daily override needs UI for quick rate changes. Fixed default is simpler but less accurate.",
        "options": [
            "Default rate with quick voice/tap override ('Pay is 2.00 today')",
            "Forced daily confirmation before shift start",
        ],
        "triggers": ["pay", "rate", "parcel", "1.85", "2.00", "variation", "override"],
        "blocking": False,  # v1.4: NOT blocking - default rate is safe
        "default_value": "Default rate (£1.85) with optional override in settings",
        "default_reason": "Use provided rate as default; override in settings if needed",
    },
]


def detect_domains(text: str) -> List[str]:
    """
    Detect which domains are mentioned in the text.
    Returns list of domain keys (e.g., ["mobile_app"]).
    """
    if not text:
        return []
    
    text_lower = text.lower()
    detected = []
    
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                detected.append(domain)
                break  # One match is enough for this domain
    
    return detected


def extract_unresolved_ambiguities(weaver_text: str) -> List[str]:
    """
    Extract the "Unresolved ambiguities" section from Weaver output.
    Returns list of ambiguity strings.
    """
    if not weaver_text:
        return []
    
    ambiguities = []
    in_section = False
    
    for line in weaver_text.split("\n"):
        line_stripped = line.strip()
        line_lower = line_stripped.lower()
        
        # Detect section start
        if "unresolved ambigu" in line_lower:
            in_section = True
            continue
        
        # Detect section end (next header or empty after content)
        if in_section:
            if line_stripped.startswith("#") or line_stripped.startswith("**") and not line_stripped.startswith("**-"):
                # New section started
                break
            if line_stripped.startswith("-") or line_stripped.startswith("*"):
                # Bullet point - extract content
                content = line_stripped.lstrip("-*").strip()
                if content:
                    ambiguities.append(content)
            elif line_stripped and not line_stripped.startswith("#"):
                # Non-bullet content in section
                ambiguities.append(line_stripped)
    
    return ambiguities


def extract_decision_forks(
    weaver_text: str,
    detected_domains: List[str],
    max_questions: int = 7,
) -> Tuple[List[GroundedQuestion], List[GroundedAssumption]]:
    """
    Extract bounded decision fork questions from Weaver text.
    
    v1.2: This replaces the lazy "tell me steps/tests" questions.
    Only asks for genuine product decisions that SpecGate cannot derive.
    
    v1.4: BLOCKER-ONLY GATING
    - Only returns questions where blocking=True
    - Non-blocking forks get safe defaults and are returned as assumptions
    - This prevents "question-hunting" (asking for the sake of asking)
    
    Args:
        weaver_text: Full Weaver job description text
        detected_domains: List of detected domain keys
        max_questions: Maximum questions to return (default 7)
        
    Returns:
        Tuple of (blocking_questions, assumptions)
    """
    if not weaver_text:
        return [], []
    
    questions: List[GroundedQuestion] = []
    assumptions: List[GroundedAssumption] = []
    text_lower = weaver_text.lower()
    
    # Get unresolved ambiguities for evidence citation
    ambiguities = extract_unresolved_ambiguities(weaver_text)
    
    # Process mobile app domain
    if "mobile_app" in detected_domains:
        for fork in MOBILE_APP_FORK_BANK:
            # Check if any trigger keywords are present
            triggered = any(trigger in text_lower for trigger in fork["triggers"])
            
            if not triggered:
                continue
            
            # v1.4: Check if this fork is blocking or can be defaulted
            is_blocking = fork.get("blocking", True)  # Default to blocking if not specified
            
            if is_blocking:
                # This question changes architecture/tooling/acceptance - MUST ASK
                evidence = "Detected from Weaver intent"
                for amb in ambiguities:
                    amb_lower = amb.lower()
                    if any(trigger in amb_lower for trigger in fork["triggers"]):
                        evidence = f"Weaver ambiguity: '{amb[:100]}...'" if len(amb) > 100 else f"Weaver ambiguity: '{amb}'"
                        break
                
                questions.append(GroundedQuestion(
                    question=fork["question"],
                    category=QuestionCategory.DECISION_FORK,
                    why_it_matters=fork["why_it_matters"],
                    evidence_found=evidence,
                    options=fork["options"],
                ))
                
                logger.info(
                    "[spec_gate_grounded] v1.4 BLOCKING question: %s",
                    fork["question"]
                )
            else:
                # v1.4: Non-blocking - apply safe default and record as assumption
                default_value = fork.get("default_value")
                default_reason = fork.get("default_reason")
                
                if default_value and default_reason:
                    assumptions.append(GroundedAssumption(
                        topic=fork["id"],
                        assumed_value=default_value,
                        reason=default_reason,
                        can_override=True,
                    ))
                    
                    logger.info(
                        "[spec_gate_grounded] v1.4 ASSUMED (not blocking): %s -> %s",
                        fork["id"], default_value
                    )
            
            if len(questions) >= max_questions:
                break
    
    # Future: Add other domain fork banks here
    
    return questions[:max_questions], assumptions


# =============================================================================
# SANDBOX DISCOVERY HELPERS (v1.3)
# =============================================================================

def _extract_sandbox_hints(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract anchor and subfolder from Weaver/user text.
    
    v1.8: Fixed stopword filtering - "on", "in", etc. must not become subfolder names.
    
    Returns:
        (anchor, subfolder) or (None, None) if no sandbox hints found
        
    Examples:
        "Desktop folder called test" → ("desktop", "test")
        "file on the desktop" → ("desktop", None)
        "Documents/reports" → ("documents", "reports")
        "text file in my test folder on the desktop" → ("desktop", "test")
    """
    if not text:
        return None, None
    
    text_lower = text.lower()
    
    # v1.8: Comprehensive stopword list - these must NEVER be captured as subfolder names
    SUBFOLDER_STOPWORDS = {
        # Prepositions
        "on", "in", "at", "to", "of", "for", "from", "with", "by",
        # Articles/determiners
        "the", "a", "an", "my", "your", "this", "that", "it",
        # Generic words
        "folder", "file", "files", "directory", "dir",
        # Conversational
        "ok", "okay", "yes", "no", "please", "thanks",
    }
    
    # Detect anchor
    anchor = None
    if "desktop" in text_lower:
        anchor = "desktop"
    elif "documents" in text_lower or "document" in text_lower:
        anchor = "documents"
    
    if not anchor:
        return None, None
    
    # v1.8: Strip meta-instructions before extracting subfolder
    # These contaminate intent parsing: "reply ok", "say ok when you understand", etc.
    meta_patterns = [
        r'reply\s+(?:with\s+)?ok\b[^.]*',
        r'say\s+(?:with\s+)?ok\b[^.]*',
        r'just\s+say\s+ok\b[^.]*',
        r'when\s+you\s+understand[^.]*',
    ]
    cleaned_text = text_lower
    for meta_pat in meta_patterns:
        cleaned_text = re.sub(meta_pat, '', cleaned_text)
    
    # Extract subfolder - v1.8: improved patterns and order
    # The patterns are ordered from most specific to least specific
    patterns = [
        # v1.8 FIX: "folder on the desktop called X" - words between folder and called
        r'folder\s+(?:on|in)\s+(?:the\s+)?(?:desktop|documents)\s+(?:called|named)\s+["\']?(\w+)["\']?',
        # "desktop folder called X" or "documents folder called X"
        r'(?:desktop|documents)\s+folder\s+(?:called|named)\s+["\']?(\w+)["\']?',
        # Most specific: "folder called/named X" (adjacent)
        r'folder\s+(?:called|named)\s+["\']?(\w+)["\']?',
        # "in X folder" or "in my X folder" (X must come before "folder")
        r'in\s+(?:my\s+|the\s+|a\s+)?([\w]+)\s+folder',
        # "X folder on/in the desktop" (folder name before "folder on")
        r'([\w]+)\s+folder\s+(?:on|in)\s+(?:the\s+)?(?:desktop|documents)',
        # "my X folder" pattern (possessive)
        r'my\s+([\w]+)\s+folder',
        # "called X" or "named X" (anywhere in text, last resort)
        r'(?:called|named)\s+["\']?(\w+)["\']?',
        # Path separators: "desktop/X" or "desktop\X"
        r'desktop[/\\\\]+(\w+)',
        r'documents[/\\\\]+(\w+)',
    ]
    
    subfolder = None
    for pattern in patterns:
        match = re.search(pattern, cleaned_text)
        if match:
            candidate = match.group(1)
            # v1.8: Strict stopword filtering
            if candidate not in SUBFOLDER_STOPWORDS and len(candidate) > 1:
                subfolder = candidate
                logger.info(
                    "[spec_gate_grounded] v1.8 Extracted subfolder '%s' using pattern: %s",
                    subfolder, pattern
                )
                break
            else:
                logger.debug(
                    "[spec_gate_grounded] v1.8 Rejected stopword candidate '%s' from pattern: %s",
                    candidate, pattern
                )
    
    # v1.8: Extra validation - log warning if we couldn't extract a subfolder
    # but the text mentions folder names
    if not subfolder:
        # Check for folder mentions we might have missed
        folder_mention = re.search(r'([\w]+)\s+folder', cleaned_text)
        if folder_mention:
            candidate = folder_mention.group(1)
            if candidate not in SUBFOLDER_STOPWORDS and len(candidate) > 1:
                subfolder = candidate
                logger.info(
                    "[spec_gate_grounded] v1.8 Fallback extracted subfolder '%s' from 'X folder' pattern",
                    subfolder
                )
    
    # v1.8: Final validation - NEVER return empty string as subfolder
    if subfolder is not None:
        subfolder = subfolder.strip()
        if not subfolder or subfolder in SUBFOLDER_STOPWORDS:
            logger.warning(
                "[spec_gate_grounded] v1.8 Discarding invalid subfolder: '%s'",
                subfolder
            )
            subfolder = None
    
    logger.info(
        "[spec_gate_grounded] v1.8 _extract_sandbox_hints result: anchor='%s', subfolder='%s'",
        anchor, subfolder
    )
    
    return anchor, subfolder


def _classify_job_size(weaver_output: str) -> str:
    """
    Classify job size for evidence loading decisions.
    
    Returns: "tiny", "normal", or "critical"
    """
    if not weaver_output:
        return "normal"
    
    text = weaver_output.lower()
    word_count = len(text.split())
    
    # Tiny indicators (simple sandbox file tasks)
    tiny_indicators = [
        "reply to", "read the message", "write a reply",
        "find the file", "simple test", "message file",
        "respond to", "answer the"
    ]
    if any(w in text for w in tiny_indicators) and word_count < 100:
        return "tiny"
    
    # Critical indicators (complex/risky tasks)
    critical_words = [
        "refactor", "security", "encrypt", "migration",
        "schema", "all files", "entire codebase", "complete rewrite",
        "breaking change", "backwards compat"
    ]
    if any(w in text for w in critical_words) or word_count > 500:
        return "critical"
    
    return "normal"


# Evidence loading config by job size
EVIDENCE_CONFIG = {
    "tiny": {
        "include_arch_map": False,
        "include_codebase_report": False,
    },
    "normal": {
        "include_arch_map": True,
        "include_codebase_report": True,
        "arch_map_max_lines": 300,
        "codebase_report_max_lines": 200,
    },
    "critical": {
        "include_arch_map": True,
        "include_codebase_report": True,
        "arch_map_max_lines": 500,
        "codebase_report_max_lines": 300,
    },
}


# =============================================================================
# REPLY GENERATION (v1.8 - Read-Only, LLM-Powered)
# =============================================================================

async def _generate_reply_from_content(
    content: str,
    content_type: Optional[str] = None,
    provider_id: str = "openai",
    model_id: str = "gpt-5-mini",
) -> str:
    """
    v1.8: Generate an intelligent reply based on the file content using LLM.
    
    This uses LLM intelligence to actually ANSWER questions in the file.
    The reply is included IN the SPoT output (SpecGate is read-only).
    
    Args:
        content: The full text content of the input file
        content_type: Classification type (MESSAGE, CODE, etc.) if known
        provider_id: LLM provider to use
        model_id: LLM model to use
        
    Returns:
        Generated reply text (actual answer, not placeholder)
    """
    if not content:
        return "(No content to reply to - file was empty)"
    
    content = content.strip()
    
    # Check for simple instructions first (no LLM needed)
    simple_reply = _detect_simple_instruction(content)
    if simple_reply:
        logger.info("[spec_gate_grounded] v1.8 Simple instruction detected, returning: %s", simple_reply[:50])
        return simple_reply
    
    # Use LLM for intelligent reply
    if _LLM_CALL_AVAILABLE and llm_call:
        try:
            # Build prompt for the LLM
            system_prompt = """You are a helpful assistant answering questions found in files.
Your job is to provide clear, concise, accurate answers.
If the file contains code, explain what it does.
If the file contains a question, answer it directly.
Keep your response brief and to the point (1-3 sentences for simple questions).
Do NOT include any preamble like "The answer is" - just give the answer directly."""

            user_prompt = f"""The following content was found in a file. Please provide an appropriate response:

---
{content}
---

Your response:"""

            logger.info(
                "[spec_gate_grounded] v1.8 Making LLM call for intelligent reply (provider=%s, model=%s)",
                provider_id, model_id
            )
            
            result = await llm_call(
                provider_id=provider_id,
                model_id=model_id,
                messages=[{"role": "user", "content": user_prompt}],
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=500,
                timeout_seconds=30,
            )
            
            if result.is_success() and result.content:
                reply = result.content.strip()
                logger.info(
                    "[spec_gate_grounded] v1.8 LLM reply generated successfully: %s",
                    reply[:100] if reply else "(empty)"
                )
                return reply
            else:
                logger.warning(
                    "[spec_gate_grounded] v1.8 LLM call failed: status=%s, error=%s",
                    result.status, result.error_message
                )
                # Fall through to fallback
                
        except Exception as e:
            logger.warning("[spec_gate_grounded] v1.8 LLM reply generation exception: %s", e)
            # Fall through to fallback
    else:
        logger.warning("[spec_gate_grounded] v1.8 LLM call not available, using fallback")
    
    # Fallback: use simple heuristics (v1.7 behavior)
    return _generate_reply_fallback(content, content_type)


def _generate_reply_fallback(content: str, content_type: Optional[str] = None) -> str:
    """
    v1.8: Fallback reply generation when LLM is unavailable.
    
    Uses simple heuristics - kept for backward compatibility and error recovery.
    """
    if not content:
        return "(No content to reply to - file was empty)"
    
    content = content.strip()
    content_lower = content.lower()
    
    # Simple question detection
    is_question = (
        content.endswith("?") or
        content_lower.startswith(("what", "who", "where", "when", "why", "how", "is ", "are ", "do ", "does ", "can ", "could", "would", "should"))
    )
    
    # Simple greeting/instruction detection
    is_greeting = content_lower.startswith(("hello", "hi ", "hi,", "hey", "greetings", "say "))
    
    if is_greeting:
        if "say " in content_lower:
            parts = content.split(" ", 1)
            if len(parts) > 1:
                return parts[1].strip()
        return "Hello!"
    
    if is_question:
        if "hello" in content_lower or "hi" in content_lower:
            return "Hello! How can I help you?"
        elif "name" in content_lower:
            return "I am SpecGate, the specification builder component of ASTRA."
        elif "time" in content_lower or "date" in content_lower:
            return f"The current time is {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}."
        else:
            return f"(LLM unavailable) Question detected: '{content[:80]}{'...' if len(content) > 80 else ''}'"
    
    # Default acknowledgement
    if len(content) < 50:
        return content
    else:
        return f"(LLM unavailable) Content received ({len(content)} characters). Manual review required."


def _detect_simple_instruction(content: str) -> Optional[str]:
    """
    v1.8: Detect if content is a simple instruction like "Say Hello".
    
    Returns the expected response, or None if not a simple instruction.
    """
    if not content:
        return None
    
    content = content.strip()
    content_lower = content.lower()
    
    # Pattern: "Say X" -> return "X"
    if content_lower.startswith("say "):
        response = content[4:].strip()
        # Capitalize first letter if it's a single word
        if " " not in response and response:
            response = response.capitalize()
        return response
    
    # Pattern: "Reply with X" or "Respond with X"
    for prefix in ["reply with ", "respond with ", "answer with "]:
        if content_lower.startswith(prefix):
            return content[len(prefix):].strip()
    
    return None


# =============================================================================
# SPEC COMPLETENESS CHECK (v1.4 - Early Exit)
# =============================================================================

def _is_spec_complete_enough(
    spec: "GroundedPOTSpec",
    intent: Dict[str, Any],
    blocking_questions: List["GroundedQuestion"],
) -> Tuple[bool, str]:
    """
    v1.4: Check if spec is complete enough to proceed without more questions.
    
    This prevents "question-hunting" by allowing early exit when:
    - No blocking questions remain
    - Enough information exists to build a valid POT spec
    
    Returns:
        (is_complete, reason_string)
    """
    # If there are blocking questions, spec is not complete
    if blocking_questions:
        return False, f"{len(blocking_questions)} blocking question(s) remain"
    
    # Check minimum requirements for a valid POT spec
    checks = []
    
    # 1. Goal must be defined
    if not spec.goal or spec.goal.strip() == "":
        checks.append("goal is missing")
    
    # 2. For mobile app domain, check domain-specific requirements
    weaver_text = intent.get("raw_text", "") or ""
    detected_domains = detect_domains(weaver_text)
    
    if "mobile_app" in detected_domains:
        # Mobile app needs at minimum:
        # - Goal (checked above)
        # - Input mode (can be assumed or asked - if asked, it's blocking)
        # - Output format (daily summary is default)
        # - Storage (local-only is default)
        # All these can be safely defaulted if not blocking
        pass  # Defaults are handled by assumptions
    
    # 3. Check if we have enough structure
    # Note: steps/tests are derived, not required from user
    
    if checks:
        return False, f"Missing: {', '.join(checks)}"
    
    return True, "Spec is complete enough - no blocking questions remain"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GroundedFact:
    """A fact grounded in repo evidence."""
    description: str
    source: str  # Which evidence confirmed this
    path: Optional[str] = None
    confidence: str = "confirmed"  # confirmed, inferred, unverified


@dataclass
class GroundedAssumption:
    """
    v1.4: A safe default applied instead of asking a non-blocking question.
    
    These are recorded in the spec so the user can override if needed,
    but they don't block spec completion.
    """
    topic: str              # e.g., "sync_behaviour", "pay_variation"
    assumed_value: str      # The default we applied
    reason: str             # Why this is safe for v1
    can_override: bool = True  # User can change this later


@dataclass
class GroundedQuestion:
    """A high-impact question that requires human input."""
    question: str
    category: QuestionCategory
    why_it_matters: str
    evidence_found: str  # What SpecGate found so far
    options: Optional[List[str]] = None  # A/B options if applicable

    def format(self) -> str:
        """Format question for POT spec output."""
        lines = [f"**Q:** {self.question}"]
        lines.append(f"  - *Why it matters:* {self.why_it_matters}")
        lines.append(f"  - *Evidence found:* {self.evidence_found}")
        if self.options:
            lines.append(f"  - *Options:* " + " / ".join(f"({chr(65+i)}) {opt}" for i, opt in enumerate(self.options)))
        return "\n".join(lines)


@dataclass
class GroundedPOTSpec:
    """Point-of-Truth Spec grounded in repo evidence."""
    # Core
    goal: str
    
    # Grounded reality
    confirmed_components: List[GroundedFact] = field(default_factory=list)
    what_exists: List[str] = field(default_factory=list)
    what_missing: List[str] = field(default_factory=list)
    
    # Scope
    in_scope: List[str] = field(default_factory=list)
    out_of_scope: List[str] = field(default_factory=list)
    
    # Constraints
    constraints_from_intent: List[str] = field(default_factory=list)
    constraints_from_repo: List[str] = field(default_factory=list)
    
    # Evidence
    evidence_bundle: Optional[EvidenceBundle] = None
    
    # Plan
    proposed_steps: List[str] = field(default_factory=list)
    acceptance_tests: List[str] = field(default_factory=list)
    
    # Risks
    risks: List[Dict[str, str]] = field(default_factory=list)
    refactor_flags: List[str] = field(default_factory=list)
    
    # Questions (human decisions only)
    open_questions: List[GroundedQuestion] = field(default_factory=list)
    
    # v1.4: Assumptions (safe defaults applied instead of asking)
    assumptions: List[GroundedAssumption] = field(default_factory=list)
    
    # v1.5: Resolved decisions (explicit answers to blocking forks)
    decisions: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    spec_id: Optional[str] = None
    spec_hash: Optional[str] = None
    spec_version: int = 1
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Validation
    is_complete: bool = False
    blocking_issues: List[str] = field(default_factory=list)
    
    # Evidence completeness tracking (v1.1)
    evidence_complete: bool = True
    evidence_gaps: List[str] = field(default_factory=list)
    
    # Sandbox resolution (v1.3 - for sandbox file jobs)
    sandbox_input_path: Optional[str] = None       # Full path to input file in sandbox
    sandbox_output_path: Optional[str] = None      # Full path for output (same folder as input)
    sandbox_folder_path: Optional[str] = None      # Folder containing input
    sandbox_anchor: Optional[str] = None           # "desktop", "documents", etc.
    sandbox_subfolder: Optional[str] = None        # Subfolder name if specified
    sandbox_selected_type: Optional[str] = None    # MESSAGE, CODE, etc.
    sandbox_selection_confidence: float = 0.0      # Classification confidence
    sandbox_input_excerpt: Optional[str] = None    # First ~500 chars of input
    sandbox_input_full_content: Optional[str] = None  # v1.7: Full content of input file
    sandbox_generated_reply: Optional[str] = None  # v1.7: Generated reply (read-only, included in SPoT)
    sandbox_discovery_used: bool = False           # True if sandbox discovery was run
    sandbox_ambiguity: Optional[str] = None        # Ambiguity reason if any
    sandbox_discovery_status: Optional[str] = None # v1.6: not_attempted, attempted, success, no_match, error
    sandbox_skip_reason: Optional[str] = None      # v1.6: Why discovery was skipped


# =============================================================================
# POT SPEC TEMPLATE BUILDER
# =============================================================================

def build_pot_spec_markdown(spec: GroundedPOTSpec) -> str:
    """
    Build POT spec markdown in the required template format.
    
    Template:
    - Goal
    - Current Reality (Grounded Facts)
    - Scope (in/out)
    - Constraints (from Weaver + discovered)
    - Evidence Used
    - Proposed Step Plan (small, testable steps only)
    - Acceptance Tests
    - Risks + Mitigations
    - Refactor Flags (recommendations only)
    - Open Questions (human decisions only)
    """
    lines = []
    
    # Title
    lines.append("# Point-of-Truth Specification")
    lines.append("")
    
    # Goal
    lines.append("## Goal")
    lines.append(spec.goal or "(Not specified)")
    lines.append("")
    
    # Current Reality
    lines.append("## Current Reality (Grounded Facts)")
    lines.append("")
    
    # Sandbox Resolution (if sandbox job) - v1.3
    # v1.7: Updated wording - SpecGate is READ-ONLY
    if spec.sandbox_discovery_used and spec.sandbox_input_path:
        lines.append("### Sandbox File Resolution")
        lines.append(f"- **Input file:** `{spec.sandbox_input_path}`")
        lines.append(f"- **Planned output path (for later stages):** `{spec.sandbox_output_path}`")
        lines.append(f"- **Content type:** {spec.sandbox_selected_type}")
        lines.append(f"- **Selection confidence:** {spec.sandbox_selection_confidence:.2f}")
        if spec.sandbox_input_excerpt:
            excerpt_lines = spec.sandbox_input_excerpt.split('\n')[:5]
            lines.append("")
            lines.append("**Input excerpt:**")
            lines.append("```")
            for el in excerpt_lines:
                lines.append(el)
            lines.append("```")
        lines.append("")
        
        # v1.7: Add Reply (Read-Only) section - this is the key addition
        if spec.sandbox_generated_reply:
            lines.append("### 📝 Reply (Read-Only)")
            lines.append("")
            lines.append("*This reply was generated by SpecGate based on the file content.*")
            lines.append("*SpecGate is READ-ONLY and does not write files. Later pipeline stages may write this to the planned output path.*")
            lines.append("")
            lines.append("```")
            lines.append(spec.sandbox_generated_reply)
            lines.append("```")
            lines.append("")
    elif spec.sandbox_discovery_status and spec.sandbox_discovery_status != "not_attempted":
        # v1.6: Show sandbox discovery status even when failed
        lines.append("### ⚠️ Sandbox Discovery Status")
        lines.append(f"- **Status:** {spec.sandbox_discovery_status}")
        if spec.sandbox_skip_reason:
            lines.append(f"- **Reason:** {spec.sandbox_skip_reason}")
        lines.append("")
    
    if spec.confirmed_components:
        lines.append("### Confirmed Components/Files/Modules")
        for fact in spec.confirmed_components:
            conf = f" [{fact.confidence}]" if fact.confidence != "confirmed" else ""
            src = f" (source: {fact.source})" if fact.source else ""
            lines.append(f"- {fact.description}{conf}{src}")
        lines.append("")
    
    if spec.what_exists:
        lines.append("### What Exists Now")
        for item in spec.what_exists:
            lines.append(f"- {item}")
        lines.append("")
    
    if spec.what_missing:
        lines.append("### What Doesn't Exist (Gaps)")
        for item in spec.what_missing:
            lines.append(f"- {item}")
        lines.append("")
    
    # Scope
    lines.append("## Scope")
    lines.append("")
    lines.append("### In Scope")
    if spec.in_scope:
        for item in spec.in_scope:
            lines.append(f"- {item}")
    else:
        lines.append("- (To be determined)")
    lines.append("")
    
    lines.append("### Out of Scope")
    if spec.out_of_scope:
        for item in spec.out_of_scope:
            lines.append(f"- {item}")
    else:
        lines.append("- (None explicitly specified)")
    lines.append("")
    
    # Constraints
    lines.append("## Constraints")
    lines.append("")
    
    lines.append("### From Weaver Intent")
    if spec.constraints_from_intent:
        for c in spec.constraints_from_intent:
            lines.append(f"- {c}")
    else:
        lines.append("- (None specified)")
    lines.append("")
    
    lines.append("### Discovered from Repo")
    if spec.constraints_from_repo:
        for c in spec.constraints_from_repo:
            lines.append(f"- {c}")
    else:
        lines.append("- (None discovered)")
    lines.append("")
    
    # Evidence Used
    lines.append("## Evidence Used")
    lines.append("")
    if spec.evidence_bundle:
        for source in spec.evidence_bundle.sources:
            if source.found or source.error:
                lines.append(f"- {source.to_evidence_line()}")
    else:
        lines.append("- (No evidence collected)")
    lines.append("")
    
    # Evidence Gaps Warning (v1.1)
    if spec.evidence_gaps:
        lines.append("### ⚠️ Evidence Gaps")
        lines.append("*The following evidence sources were unavailable, limiting grounding confidence:*")
        lines.append("")
        for gap in spec.evidence_gaps:
            lines.append(f"- {gap}")
        lines.append("")
    
    # Proposed Step Plan
    lines.append("## Proposed Step Plan")
    lines.append("*(Small, testable steps only)*")
    lines.append("")
    if spec.proposed_steps:
        for i, step in enumerate(spec.proposed_steps, 1):
            lines.append(f"{i}. {step}")
    else:
        lines.append("1. (Steps to be determined after questions resolved)")
    lines.append("")
    
    # Acceptance Tests
    lines.append("## Acceptance Tests")
    lines.append("")
    if spec.acceptance_tests:
        for test in spec.acceptance_tests:
            lines.append(f"- [ ] {test}")
    else:
        lines.append("- [ ] (To be determined)")
    lines.append("")
    
    # Risks + Mitigations
    lines.append("## Risks + Mitigations")
    lines.append("")
    if spec.risks:
        lines.append("| Risk | Mitigation |")
        lines.append("|------|------------|")
        for risk in spec.risks:
            lines.append(f"| {risk.get('risk', 'N/A')} | {risk.get('mitigation', 'N/A')} |")
    else:
        lines.append("| Risk | Mitigation |")
        lines.append("|------|------------|")
        lines.append("| (None identified) | - |")
    lines.append("")
    
    # Refactor Flags
    lines.append("## Refactor Flags (Recommendations Only)")
    lines.append("")
    if spec.refactor_flags:
        for flag in spec.refactor_flags:
            lines.append(f"- ⚠️ {flag}")
    else:
        lines.append("- (None)")
    lines.append("")
    
    # Open Questions - ALWAYS show if present (even on Round 3 finalization)
    lines.append("## Open Questions (Human Decisions Only)")
    lines.append("")
    if spec.open_questions:
        # If finalized with questions, mark as UNRESOLVED (no guessing)
        if spec.is_complete and spec.spec_version >= 3:
            lines.append("⚠️ **FINALIZED WITH UNRESOLVED QUESTIONS** - These were NOT guessed or filled in:")
            lines.append("")
        for i, q in enumerate(spec.open_questions, 1):
            lines.append(f"### Question {i}")
            if spec.is_complete and spec.spec_version >= 3:
                lines.append("**Status:** ❓ UNRESOLVED (no guess - human decision required)")
            lines.append(q.format())
            lines.append("")
    else:
        # v1.1 FIX: Only claim "all grounded" if evidence is truly complete
        if spec.evidence_complete and not spec.evidence_gaps:
            lines.append("✅ No blocking questions - all information grounded from evidence.")
        else:
            lines.append("⚠️ No questions generated, but evidence was incomplete (see Evidence Gaps above).")
    lines.append("")
    
    # v1.5: Resolved Decisions (explicit answers to blocking forks)
    if spec.decisions:
        lines.append("## Resolved Decisions")
        lines.append("")
        lines.append("*These were explicitly answered by the user.*")
        lines.append("")
        for key, value in spec.decisions.items():
            # Format key nicely (platform_v1 -> Platform v1)
            nice_key = key.replace("_", " ").title()
            lines.append(f"- **{nice_key}:** {value}")
        lines.append("")
    
    # v1.4: Assumptions (safe defaults applied instead of asking)
    if spec.assumptions:
        lines.append("## Assumptions (v1 Safe Defaults)")
        lines.append("")
        lines.append("*These were applied automatically instead of asking non-blocking questions.*")
        lines.append("*Override in spec if needed.*")
        lines.append("")
        for assumption in spec.assumptions:
            lines.append(f"- **{assumption.topic}:** {assumption.assumed_value}")
            lines.append(f"  - *Reason:* {assumption.reason}")
        lines.append("")
    
    # Blocking Issues
    if spec.blocking_issues:
        lines.append("---")
        lines.append("## ⛔ Blocking Issues")
        lines.append("")
        for issue in spec.blocking_issues:
            lines.append(f"- {issue}")
        lines.append("")
    
    # Unresolved Items Summary (for Round 3 finalization)
    has_unresolved = (
        spec.open_questions or
        not spec.proposed_steps or
        not spec.acceptance_tests or
        "(To be determined)" in str(spec.in_scope)
    )
    if spec.is_complete and has_unresolved:
        lines.append("---")
        lines.append("## ⚠️ Unresolved / Unknown (No Guess)")
        lines.append("")
        lines.append("*The following items remain unresolved. SpecGate did NOT fill these with assumptions:*")
        lines.append("")
        if spec.open_questions:
            lines.append(f"- **{len(spec.open_questions)} unanswered question(s)** - see above")
        if not spec.proposed_steps:
            lines.append("- **Steps:** Not specified (requires human input)")
        if not spec.acceptance_tests or all('(To be determined)' in str(t) for t in spec.acceptance_tests):
            lines.append("- **Acceptance tests:** Not specified (requires human input)")
        lines.append("")
    
    # Metadata
    lines.append("---")
    lines.append("## Metadata")
    lines.append(f"- **Spec ID:** `{spec.spec_id or 'N/A'}`")
    lines.append(f"- **Spec Hash:** `{spec.spec_hash[:16] if spec.spec_hash else 'N/A'}...`")
    lines.append(f"- **Version:** {spec.spec_version}")
    lines.append(f"- **Generated:** {spec.generated_at.isoformat()}")
    # v1.1 FIX: Status reflects true completeness
    lines.append(f"- **Status:** {'Complete' if spec.is_complete else 'Awaiting answers'}")
    
    return "\n".join(lines)


# =============================================================================
# WEAVER INTENT PARSER
# =============================================================================

def parse_weaver_intent(constraints_hint: Optional[Dict]) -> Dict[str, Any]:
    """
    Parse Weaver output to extract intent components.
    
    Handles both:
    - v3.0 simple text (weaver_job_description_text)
    - v2.x full spec JSON (weaver_spec_json)
    """
    if not constraints_hint:
        logger.warning("[spec_gate_grounded] parse_weaver_intent: constraints_hint is empty/None")
        return {}
    
    result = {}
    
    # v1.6: Log what keys are available
    logger.info(
        "[spec_gate_grounded] parse_weaver_intent: constraints_hint keys=%s",
        list(constraints_hint.keys())
    )
    
    # v3.0: Simple Weaver text
    job_desc_text = constraints_hint.get("weaver_job_description_text")
    if job_desc_text:
        result["raw_text"] = job_desc_text
        result["source"] = "weaver_simple"
        logger.info(
            "[spec_gate_grounded] parse_weaver_intent: set raw_text from weaver_job_description_text (%d chars)",
            len(job_desc_text)
        )
        
        # Extract goal from text
        lines = job_desc_text.strip().split("\n")
        if lines:
            # First non-empty line is usually the goal
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    result["goal"] = line
                    break
        
        # Look for constraints/scope markers
        result["constraints"] = []
        result["scope_in"] = []
        result["scope_out"] = []
        
        for line in lines:
            line_lower = line.lower().strip()
            if "constraint" in line_lower or "must" in line_lower or "require" in line_lower:
                result["constraints"].append(line.strip())
            if "in scope" in line_lower or "should" in line_lower:
                result["scope_in"].append(line.strip())
            if "out of scope" in line_lower or "should not" in line_lower or "don't" in line_lower:
                result["scope_out"].append(line.strip())
    else:
        logger.warning("[spec_gate_grounded] parse_weaver_intent: weaver_job_description_text not found in constraints_hint")
    
    # v2.x: Full spec JSON
    weaver_spec = constraints_hint.get("weaver_spec_json")
    if isinstance(weaver_spec, dict):
        result["source"] = weaver_spec.get("source", "weaver_spec")
        result["goal"] = (
            weaver_spec.get("objective") or
            weaver_spec.get("title") or
            weaver_spec.get("job_description", "")[:200]
        )
        
        # Extract metadata
        metadata = weaver_spec.get("metadata", {}) or {}
        result["content_verbatim"] = (
            metadata.get("content_verbatim") or
            weaver_spec.get("content_verbatim")
        )
        result["location"] = (
            metadata.get("location") or
            weaver_spec.get("location")
        )
        result["scope_constraints"] = (
            metadata.get("scope_constraints") or
            weaver_spec.get("scope_constraints", [])
        )
        
        # Steps and outputs
        result["weaver_steps"] = weaver_spec.get("steps", [])
        result["weaver_outputs"] = weaver_spec.get("outputs", [])
        result["weaver_acceptance"] = weaver_spec.get("acceptance_criteria", [])
    
    return result


# =============================================================================
# GROUNDING ENGINE
# =============================================================================

def ground_intent_with_evidence(
    intent: Dict[str, Any],
    evidence: EvidenceBundle,
) -> GroundedPOTSpec:
    """
    Ground Weaver intent against repo evidence.
    
    This is the core grounding logic:
    1. Look for mentioned paths/modules in evidence
    2. Verify what exists vs what doesn't
    3. Identify constraints from repo patterns
    4. Generate questions ONLY for true unknowns
    """
    spec = GroundedPOTSpec(
        goal=intent.get("goal", ""),
        evidence_bundle=evidence,
    )
    
    # v1.1: Track evidence completeness
    spec.evidence_complete = True
    spec.evidence_gaps = []
    
    # Check if codebase report was loaded
    has_codebase_report = False
    has_arch_map = False
    for source in evidence.sources:
        if source.source_type == "codebase_report":
            if source.found:
                has_codebase_report = True
            elif source.error:
                spec.evidence_gaps.append(f"Codebase report: {source.error}")
                spec.evidence_complete = False
        if source.source_type == "architecture_map":
            if source.found:
                has_arch_map = True
            elif source.error:
                spec.evidence_gaps.append(f"Architecture map: {source.error}")
                spec.evidence_complete = False
    
    # Extract any paths mentioned in intent
    mentioned_paths = _extract_paths_from_text(intent.get("raw_text", ""))
    mentioned_paths.extend(_extract_paths_from_text(intent.get("goal", "")))
    
    # Add location if specified
    if intent.get("location"):
        mentioned_paths.append(intent["location"])
    
    # Ground each mentioned path
    for path in set(mentioned_paths):
        exists, source = verify_path_exists(evidence, path)
        if exists:
            spec.confirmed_components.append(GroundedFact(
                description=f"Path `{path}` exists",
                source=source or "evidence",
                path=path,
                confidence="confirmed",
            ))
            spec.what_exists.append(f"`{path}`")
        else:
            spec.what_missing.append(f"`{path}` (not found in evidence)")
    
    # Extract constraints from intent
    if intent.get("constraints"):
        spec.constraints_from_intent.extend(intent["constraints"])
    if intent.get("scope_constraints"):
        spec.constraints_from_intent.extend(intent["scope_constraints"])
    
    # Extract scope
    if intent.get("scope_in"):
        spec.in_scope.extend(intent["scope_in"])
    if intent.get("scope_out"):
        spec.out_of_scope.extend(intent["scope_out"])
    
    # Try to find relevant patterns in evidence
    if evidence.arch_map_content:
        # Look for related modules
        goal_keywords = _extract_keywords(intent.get("goal", ""))
        for keyword in goal_keywords[:5]:  # Top 5 keywords
            matches = find_in_evidence(evidence, rf"\b{re.escape(keyword)}\b", "architecture_map")
            if matches:
                spec.confirmed_components.append(GroundedFact(
                    description=f"Related content found for '{keyword}' in architecture map",
                    source="architecture_map",
                    confidence="inferred",
                ))
    
    # Copy steps/outputs from Weaver if available
    if intent.get("weaver_steps"):
        spec.proposed_steps = intent["weaver_steps"]
    if intent.get("weaver_acceptance"):
        spec.acceptance_tests = intent["weaver_acceptance"]
    
    # Detect refactor candidates from codebase report
    if evidence.codebase_report_content:
        # Look for bloat warnings
        bloat_matches = find_in_evidence(
            evidence,
            r"(size_critical|size_high|lines_critical|lines_high)",
            "codebase_report"
        )
        if bloat_matches:
            spec.refactor_flags.append(
                "Codebase report indicates large/complex files - consider refactoring"
            )
    
    return spec


def _extract_paths_from_text(text: str) -> List[str]:
    """Extract file/directory paths from text."""
    if not text:
        return []
    
    patterns = [
        r'`([^`]+\.(?:py|ts|tsx|js|jsx|json|md|yaml|yml))`',  # backtick paths
        r'[\'"]([^\'"]+\.(?:py|ts|tsx|js|jsx|json|md|yaml|yml))[\'"]',  # quoted paths
        r'(?:^|\s)(app/[^\s]+)',  # app/ paths
        r'(?:^|\s)(src/[^\s]+)',  # src/ paths
        r'(?:^|\s)(tests/[^\s]+)',  # tests/ paths
    ]
    
    paths = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        paths.extend(matches)
    
    return paths


def _extract_keywords(text: str) -> List[str]:
    """Extract meaningful keywords from text."""
    if not text:
        return []
    
    # Remove common words
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'this', 'that', 'these', 'those', 'it', 'its', 'i', 'you', 'we',
        'they', 'he', 'she', 'what', 'which', 'who', 'whom', 'when', 'where',
        'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())
    
    # Filter and score by length (longer = more meaningful)
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    
    # Dedupe while preserving order
    seen = set()
    result = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            result.append(kw)
    
    return result


# =============================================================================
# QUESTION GENERATOR (v1.2 - Decision Forks, Not Lazy Questions)
# =============================================================================

def generate_grounded_questions(
    spec: GroundedPOTSpec,
    intent: Dict[str, Any],
    evidence: EvidenceBundle,
    round_number: int,
) -> List[GroundedQuestion]:
    """
    Generate questions ONLY for genuine unknowns.
    
    v1.2 CONTRACT:
    - Round 1: Ask bounded decision forks (A/B/C) only
    - Round 2+: Steps/tests are DERIVED from fork answers (not asked for)
    - Never ask "tell me the steps" or "tell me the acceptance criteria"
    - Max 7 questions total, only high-impact product decisions
    
    Rules:
    - Only ask when NOT derivable from evidence
    - Only ask high-impact questions (wrong answer = rework)
    - Preference/product decisions only (not engineering facts)
    """
    questions = []
    
    # Get Weaver text for domain detection and fork extraction
    weaver_text = intent.get("raw_text", "") or ""
    
    # =================================================================
    # v1.5: ALWAYS extract forks to populate assumptions (every round)
    # =================================================================
    detected_domains = detect_domains(weaver_text)
    fork_questions = []
    fork_assumptions = []
    
    if detected_domains:
        fork_questions, fork_assumptions = extract_decision_forks(
            weaver_text=weaver_text,
            detected_domains=detected_domains,
            max_questions=MAX_QUESTIONS,
        )
        # Always populate assumptions (even on Round 2+)
        spec.assumptions.extend(fork_assumptions)
        
        logger.info(
            "[spec_gate_grounded] v1.5: Detected domains %s, blocking questions=%d, assumptions=%d (round=%d)",
            detected_domains, len(fork_questions), len(fork_assumptions), round_number
        )
    
    # =================================================================
    # ROUND 2+: Derive steps/tests from fork answers, don't ask more
    # =================================================================
    if round_number >= 2:
        # v1.2: In Round 2+, we DERIVE steps/tests from answered forks.
        # We do NOT ask the user to write them for us.
        # If steps/tests are still missing, they will be generated here.
        
        # Only ask critical questions if there's a genuine blocker
        # that can't be derived (e.g., truly missing goal)
        if not spec.goal or spec.goal.strip() == "":
            questions.append(GroundedQuestion(
                question="What is the primary goal/objective of this job?",
                category=QuestionCategory.MISSING_PRODUCT_DECISION,
                why_it_matters="Without a clear goal, the spec cannot be grounded",
                evidence_found="No goal found in Weaver output",
            ))
        
        # v1.2: Steps and tests are SpecGate's job to derive, NOT the user's
        # If we reach Round 2 without them, derive from the domain + forks
        if not spec.proposed_steps:
            spec.proposed_steps = _derive_steps_from_domain(intent, spec)
        if not spec.acceptance_tests or all('(To be determined)' in str(t) for t in spec.acceptance_tests):
            spec.acceptance_tests = _derive_tests_from_domain(intent, spec)
        
        return questions[:MAX_QUESTIONS]
    
    # =================================================================
    # ROUND 1: Ask bounded decision forks (A/B/C) - NOT lazy questions
    # =================================================================
    
    # Check for missing goal (this is a critical blocker, not a lazy question)
    if not spec.goal or spec.goal.strip() == "":
        questions.append(GroundedQuestion(
            question="What is the primary goal/objective of this job?",
            category=QuestionCategory.MISSING_PRODUCT_DECISION,
            why_it_matters="Without a clear goal, the spec cannot be grounded",
            evidence_found="No goal found in Weaver output",
        ))
    
    # v1.5: Fork questions were already extracted above, now add them to questions list
    if fork_questions:
        questions.extend(fork_questions)
    
    # v1.2 REMOVED: Do NOT ask lazy "steps/tests" questions
    # These are SpecGate's job to DERIVE after forks are answered:
    #   - "What are the key implementation steps for this work?" ← REMOVED
    #   - "What acceptance criteria should verify this work is complete?" ← REMOVED
    
    # Check for ambiguous paths (mentioned but not found) - this is still valid
    if spec.what_missing:
        missing_count = len(spec.what_missing)
        if missing_count > 0 and len(questions) < MAX_QUESTIONS:
            questions.append(GroundedQuestion(
                question=f"These paths were mentioned but not found in evidence: {', '.join(spec.what_missing[:3])}. Should they be created, or are the paths incorrect?",
                category=QuestionCategory.AMBIGUOUS_EVIDENCE,
                why_it_matters="Need to know if files should be created vs paths are wrong",
                evidence_found=f"Searched architecture map and codebase report - {missing_count} path(s) not found",
                options=["Create new files at these paths", "Paths may be incorrect - suggest alternatives"],
            ))
    
    # Check for safety constraints if touching critical paths
    critical_paths = ['stream_router', 'overwatcher', 'translation', 'routing']
    touches_critical = any(
        any(crit in fact.description.lower() for crit in critical_paths)
        for fact in spec.confirmed_components
    )
    if touches_critical and not any('sandbox' in c.lower() for c in spec.constraints_from_intent):
        if len(questions) < MAX_QUESTIONS:
            questions.append(GroundedQuestion(
                question="This job touches critical routing/pipeline code. Should changes be tested in SANDBOX first before MAIN repo?",
                category=QuestionCategory.SAFETY_CONSTRAINT,
                why_it_matters="Touching critical code without sandbox testing risks breaking the system",
                evidence_found="Detected critical paths in scope",
                options=["Sandbox first, then MAIN", "MAIN repo directly (I'll verify manually)"],
            ))
    
    # Cap at MAX_QUESTIONS
    return questions[:MAX_QUESTIONS]


def _derive_steps_from_domain(intent: Dict[str, Any], spec: GroundedPOTSpec) -> List[str]:
    """
    v1.5: Derive implementation steps from domain + resolved decisions + assumptions.
    
    This is SpecGate's job, NOT the user's. Once product decisions are made,
    the steps can be derived automatically.
    
    Rules:
    - Check spec.decisions first (explicit user answers)
    - Check spec.assumptions second (safe defaults)
    - Only include steps that match resolved decisions
    - No "(if selected)" wording when already decided
    """
    weaver_text = intent.get("raw_text", "") or ""
    detected_domains = detect_domains(weaver_text)
    
    # v1.6: Diagnostic logging
    logger.info(
        "[spec_gate_grounded] _derive_steps_from_domain: raw_text_len=%d, detected_domains=%s",
        len(weaver_text), detected_domains
    )
    if not detected_domains:
        logger.warning(
            "[spec_gate_grounded] _derive_steps_from_domain: No domains detected! raw_text preview: %s",
            weaver_text[:200] if weaver_text else "(empty)"
        )
    
    # Build lookup of resolved values: decisions override assumptions
    resolved = {}
    for assumption in spec.assumptions:
        resolved[assumption.topic] = assumption.assumed_value
    for key, value in spec.decisions.items():
        resolved[key] = value
    
    steps = []
    
    # v1.6: Sandbox file domain - specific steps for file discovery/read/reply tasks
    # v1.7: UPDATED - SpecGate is READ-ONLY, never claims to write files
    if "sandbox_file" in detected_domains:
        # Check if sandbox discovery was used
        if spec.sandbox_discovery_used and spec.sandbox_input_path:
            steps = [
                f"Read input file from sandbox: `{spec.sandbox_input_path}`",
                "Parse and understand the question/content in the file",
                "Generate reply based on file content (included in SPoT output)",
            ]
            # v1.7: Add note about planned output (for later stages, not SpecGate)
            if spec.sandbox_output_path:
                steps.append(f"[For later stages] Planned output path: `{spec.sandbox_output_path}`")
        else:
            # Discovery didn't run or failed - generic sandbox steps
            steps = [
                "Discover target file in sandbox (Desktop or Documents)",
                "Read and parse the file content",
                "Generate reply based on file content (included in SPoT output)",
                "[For later stages] Planned output path: reply.txt in same folder as input",
            ]
        return steps
    
    if "mobile_app" in detected_domains:
        # Mobile app domain - conditional implementation steps
        
        # 1. Platform setup (always needed)
        platform = resolved.get("platform_v1", "")
        if "android" in platform.lower():
            steps.append("Set up Android project (Android Studio, Gradle)")
        elif "ios" in platform.lower() or "both" in platform.lower():
            steps.append("Set up mobile project structure (Android + iOS)")
        else:
            steps.append("Set up mobile project structure")
        
        # 2. Storage (always needed for mobile app)
        steps.append("Implement local encrypted data storage layer")
        
        # 3. Core UI (always needed)
        steps.append("Build core UI screens (shift start/stop, daily summary)")
        
        # 4. Input mode - based on decision
        input_mode = resolved.get("input_mode_v1", "")
        if "voice" in input_mode.lower() and "manual" in input_mode.lower():
            steps.append("Implement push-to-talk voice input with manual fallback")
        elif "voice" in input_mode.lower():
            steps.append("Implement voice input handler")
        elif "manual" in input_mode.lower() or "screenshot" in input_mode.lower():
            steps.append("Implement manual input forms (screenshot import + manual entry)")
        # If not selected (deferred), don't add step
        
        # 5. OCR - based on decision
        ocr_scope = resolved.get("ocr_scope_v1", "")
        if ocr_scope:
            if "finish tour" in ocr_scope.lower() or "completed parcels" in ocr_scope.lower():
                steps.append("Implement screenshot OCR parser for Finish Tour screen (Successfully Completed Parcels)")
            elif "multiple" in ocr_scope.lower():
                steps.append("Implement screenshot OCR parser for multiple screen formats")
        # If not selected, don't add OCR step
        
        # 6. Calculations (always needed)
        steps.append("Implement pay/cost/net calculations (parcel rate, fuel, wear & tear)")
        
        # 7. Weekly summary (always needed)
        steps.append("Implement end-of-week summary calculations")
        
        # 8. Sync - based on decision/assumption
        sync_behaviour = resolved.get("sync_behaviour", "")
        sync_target = resolved.get("sync_target", "")
        if "local-only" in sync_behaviour.lower() or "local only" in sync_behaviour.lower():
            # No sync step needed - local only
            pass
        elif "export" in sync_target.lower() or "file" in sync_target.lower():
            steps.append("Add export functionality for ASTRA integration (file-based)")
        elif "endpoint" in sync_target.lower() or "live" in sync_target.lower():
            steps.append("Build sync mechanism and ASTRA integration endpoint")
        # Default: no sync step if not specified (local-only assumption)
        
        # 9. Testing (always needed)
        steps.append("Integration testing")
        
        # 10. Security (always needed)
        steps.append("Security audit (encryption, data handling)")
        
    else:
        # Generic steps for unknown domains
        steps = [
            "Analyze requirements and create technical design",
            "Set up project structure and dependencies",
            "Implement core functionality",
            "Add error handling and edge cases",
            "Write tests and documentation",
            "Integration testing",
            "Security review",
        ]
    
    return steps


def _derive_tests_from_domain(intent: Dict[str, Any], spec: GroundedPOTSpec) -> List[str]:
    """
    v1.5: Derive acceptance tests from domain + resolved decisions + assumptions.
    
    This is SpecGate's job, NOT the user's. Once product decisions are made,
    the acceptance criteria can be derived automatically.
    
    Rules:
    - Check spec.decisions first (explicit user answers)
    - Check spec.assumptions second (safe defaults)
    - Only include tests that match resolved decisions
    - No "(if enabled)" wording when already decided
    - OCR test MUST reference "Successfully Completed Parcels" (not "stop count")
    """
    weaver_text = intent.get("raw_text", "") or ""
    detected_domains = detect_domains(weaver_text)
    
    # Build lookup of resolved values: decisions override assumptions
    resolved = {}
    for assumption in spec.assumptions:
        resolved[assumption.topic] = assumption.assumed_value
    for key, value in spec.decisions.items():
        resolved[key] = value
    
    tests = []
    
    # v1.6: Sandbox file domain - specific tests for file discovery/read/reply tasks
    # v1.7: UPDATED - SpecGate is READ-ONLY, tests verify reading not writing
    if "sandbox_file" in detected_domains:
        # Check if sandbox discovery was used
        if spec.sandbox_discovery_used and spec.sandbox_input_path:
            tests = [
                f"Input file `{spec.sandbox_input_path}` was found and read successfully",
                "File content was correctly parsed and understood",
                "Reply was generated based on file content",
                "Reply is included in SPoT output (read-only stage)",
            ]
            # Add excerpt verification if available
            if spec.sandbox_input_excerpt:
                tests.insert(1, f"Input content type identified: {spec.sandbox_selected_type or 'detected'}")
            # v1.7: Note planned output path (for later stages to verify)
            if spec.sandbox_output_path:
                tests.append(f"[For later stages] Planned output path recorded: `{spec.sandbox_output_path}`")
        else:
            # Discovery didn't run or failed - generic sandbox tests
            tests = [
                "Target file was discovered in sandbox",
                "File content was read and parsed correctly",
                "Reply was generated based on file content",
                "Reply is included in SPoT output (read-only stage)",
                "[For later stages] Planned output path: reply.txt",
            ]
        return tests
    
    if "mobile_app" in detected_domains:
        # Mobile app domain - conditional acceptance tests
        
        # 1. App startup (always needed)
        tests.append("App starts and displays main screen within 2 seconds")
        
        # 2. Shift logging (always needed)
        tests.append("Shift start/stop logs timestamp correctly to local storage")
        
        # 3. Input mode tests - based on decision
        input_mode = resolved.get("input_mode_v1", "")
        if "voice" in input_mode.lower():
            tests.append("Voice input correctly transcribes test phrases")
        if "manual" in input_mode.lower() or "screenshot" in input_mode.lower():
            tests.append("Manual entry form accepts and validates input correctly")
        # If voice deferred, no voice test
        
        # 4. OCR test - based on decision, with CORRECT target field
        ocr_scope = resolved.get("ocr_scope_v1", "")
        if ocr_scope:
            # BUG FIX: Must reference "Successfully Completed Parcels", NOT "stop count"
            if "finish tour" in ocr_scope.lower() or "completed parcels" in ocr_scope.lower():
                tests.append("Screenshot OCR extracts 'Successfully Completed Parcels' from test Finish Tour screenshot")
            elif "multiple" in ocr_scope.lower():
                tests.append("Screenshot OCR extracts parcel counts from multiple screen format test images")
        # If OCR not selected, no OCR test
        
        # 5. Data persistence (always needed)
        tests.append("Data persists across app restart (encrypted storage verified)")
        
        # 6. Sync test - based on decision/assumption
        sync_behaviour = resolved.get("sync_behaviour", "")
        sync_target = resolved.get("sync_target", "")
        if "local-only" in sync_behaviour.lower() or "local only" in sync_behaviour.lower():
            # No sync test - local only
            pass
        elif "export" in sync_target.lower() or "file" in sync_target.lower():
            tests.append("Export functionality produces valid file for ASTRA import")
        elif "endpoint" in sync_target.lower() or "live" in sync_target.lower():
            tests.append("Sync successfully transfers data to ASTRA endpoint")
        # Default: no sync test if local-only assumed
        
        # 7. Calculations (always needed)
        tests.append("Pay calculation correctly computes gross from parcel count (rate × parcels)")
        tests.append("Net profit calculation correctly subtracts fuel and wear & tear")
        
        # 8. Weekly summary (always needed)
        tests.append("End-of-week summary shows correct totals for parcels and pay")
        
        # 9. Offline (always needed for mobile app)
        tests.append("App functions fully offline (no network required for core features)")
        
        # 10. Security (always needed)
        tests.append("No sensitive data exposed in logs or debug output")
        
    else:
        # Generic tests for unknown domains
        tests = [
            "Core functionality works as specified",
            "Error handling covers expected failure modes",
            "Performance meets requirements",
            "Security review passes",
            "Documentation is complete and accurate",
        ]
    
    return tests


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def run_spec_gate_grounded(
    db: Session,
    job_id: str,
    user_intent: str,
    provider_id: str,
    model_id: str,
    project_id: int,
    constraints_hint: Optional[Dict] = None,
    spec_version: int = 1,
    user_answers: Optional[Dict[str, str]] = None,
) -> SpecGateResult:
    """
    Run SpecGate Contract v1 - Grounded POT Spec Builder.
    
    RUNTIME IS READ-ONLY:
    - No filesystem writes
    - No DB writes
    - Output/stream only
    
    Args:
        db: Database session (NOT USED for writes)
        job_id: Job identifier
        user_intent: User's raw intent text
        provider_id: LLM provider (for metadata only)
        model_id: LLM model (for metadata only)
        project_id: Project ID
        constraints_hint: Weaver output and other hints
        spec_version: Round number (1 = initial, 2+ = after answers)
        user_answers: User's answers to previous questions
        
    Returns:
        SpecGateResult with POT spec or questions
    """
    try:
        round_n = max(1, min(3, int(spec_version or 1)))
        
        logger.info(
            "[spec_gate_grounded] Starting round %d for job %s (project %d)",
            round_n, job_id, project_id
        )
        
        # =================================================================
        # STEP 1: Load Evidence (read-only)
        # =================================================================
        
        if not _EVIDENCE_AVAILABLE:
            return SpecGateResult(
                ready_for_pipeline=False,
                hard_stopped=True,
                hard_stop_reason="Evidence collector not available",
                validation_status="error",
            )
        
        evidence = load_evidence(
            include_arch_map=True,
            include_codebase_report=True,
            arch_map_max_lines=500,
            codebase_report_max_lines=300,
        )
        
        logger.info(
            "[spec_gate_grounded] Loaded evidence: %d sources, %d errors",
            len(evidence.sources),
            len(evidence.errors),
        )
        
        # =================================================================
        # STEP 1.5: Sandbox Discovery (if sandbox job detected)
        # v1.6: Added explicit logging for why discovery is skipped
        # =================================================================
        
        sandbox_discovery_result = None
        sandbox_discovery_status = "not_attempted"  # Track what happened
        sandbox_skip_reason = None
        
        weaver_job_text = (constraints_hint or {}).get('weaver_job_description_text', '')
        combined_text = f"{user_intent or ''} {weaver_job_text}"
        anchor, subfolder = _extract_sandbox_hints(combined_text)
        
        # v1.6: Diagnostic logging for sandbox discovery debugging
        logger.info(
            "[spec_gate_grounded] v1.6 Sandbox hint extraction: anchor=%s, subfolder=%s, text_len=%d, weaver_len=%d",
            anchor, subfolder, len(combined_text), len(weaver_job_text)
        )
        logger.info(
            "[spec_gate_grounded] v1.6 combined_text preview (first 300 chars): %s",
            combined_text[:300] if combined_text else "(empty)"
        )
        
        # v1.6: Check if sandbox keywords are present but anchor wasn't detected
        text_lower = combined_text.lower()
        sandbox_keywords_found = [kw for kw in DOMAIN_KEYWORDS.get("sandbox_file", []) if kw in text_lower]
        if sandbox_keywords_found and not anchor:
            logger.warning(
                "[spec_gate_grounded] v1.6 BUG: sandbox keywords found %s but anchor=%s - check _extract_sandbox_hints",
                sandbox_keywords_found[:5], anchor
            )
        
        # v1.6: Explicit logging for each condition
        if not anchor:
            sandbox_skip_reason = "No sandbox anchor detected (no 'desktop' or 'documents' in text)"
            logger.info("[spec_gate_grounded] Sandbox discovery skipped: %s", sandbox_skip_reason)
        elif not _SANDBOX_INSPECTOR_AVAILABLE:
            sandbox_skip_reason = "sandbox_inspector module not available (import failed)"
            logger.warning("[spec_gate_grounded] Sandbox discovery skipped: %s", sandbox_skip_reason)
            
            # v1.6: BLOCKING - if sandbox hints detected but tools unavailable, return error
            # Do NOT silently validate a generic spec
            return SpecGateResult(
                ready_for_pipeline=False,
                open_questions=["Sandbox discovery tools unavailable. Cannot locate file in sandbox."],
                spec_version=round_n,
                validation_status="blocked",
                blocking_issues=[sandbox_skip_reason],
                notes="Sandbox task detected but sandbox_inspector module failed to import",
            )
        elif not run_sandbox_discovery_chain:
            sandbox_skip_reason = "run_sandbox_discovery_chain function not available"
            logger.warning("[spec_gate_grounded] Sandbox discovery skipped: %s", sandbox_skip_reason)
            
            # v1.6: BLOCKING - if sandbox hints detected but tools unavailable, return error
            return SpecGateResult(
                ready_for_pipeline=False,
                open_questions=["Sandbox discovery function unavailable. Cannot locate file in sandbox."],
                spec_version=round_n,
                validation_status="blocked",
                blocking_issues=[sandbox_skip_reason],
                notes="Sandbox task detected but run_sandbox_discovery_chain not available",
            )
        else:
            # All conditions met - run discovery
            logger.info("[spec_gate_grounded] Running sandbox discovery: anchor=%s, subfolder=%s", anchor, subfolder)
            sandbox_discovery_status = "attempted"
            
            try:
                sandbox_discovery_result = run_sandbox_discovery_chain(
                    anchor=anchor,
                    subfolder=subfolder,
                    job_intent=combined_text,
                )
                
                if sandbox_discovery_result:
                    if sandbox_discovery_result.get("selected_file"):
                        sandbox_discovery_status = "success"
                        logger.info(
                            "[spec_gate_grounded] Sandbox discovery SUCCESS: selected=%s",
                            sandbox_discovery_result["selected_file"].get("path", "unknown")
                        )
                    elif sandbox_discovery_result.get("ambiguous"):
                        sandbox_discovery_status = "ambiguous"
                        logger.info(
                            "[spec_gate_grounded] Sandbox discovery AMBIGUOUS: candidates=%s",
                            sandbox_discovery_result.get("ambiguous_candidates", [])
                        )
                    else:
                        sandbox_discovery_status = "no_match"
                        sandbox_skip_reason = "Discovery ran but found no matching file"
                        logger.warning(
                            "[spec_gate_grounded] Sandbox discovery NO MATCH: found=%s, files=%s",
                            sandbox_discovery_result.get("found"),
                            sandbox_discovery_result.get("files", [])
                        )
                        
                        # v1.6: BLOCKING - if sandbox task but no file found, return error
                        return SpecGateResult(
                            ready_for_pipeline=False,
                            open_questions=[f"Could not find target file in sandbox {anchor}/{subfolder or ''}. Please check the folder exists and contains the file."],
                            spec_version=round_n,
                            validation_status="blocked",
                            blocking_issues=[sandbox_skip_reason],
                            notes=f"Sandbox discovery found={sandbox_discovery_result.get('found')}, files={sandbox_discovery_result.get('files', [])}",
                        )
                else:
                    sandbox_discovery_status = "empty_result"
                    sandbox_skip_reason = "Discovery returned None/empty"
                    logger.warning("[spec_gate_grounded] Sandbox discovery returned empty result")
                    
                    # v1.6: BLOCKING - if sandbox task but discovery failed, return error
                    return SpecGateResult(
                        ready_for_pipeline=False,
                        open_questions=[f"Sandbox discovery returned no results for {anchor}/{subfolder or ''}. Please verify the sandbox folder exists."],
                        spec_version=round_n,
                        validation_status="blocked",
                        blocking_issues=[sandbox_skip_reason],
                        notes="Sandbox discovery returned None or empty result",
                    )
                    
            except Exception as e:
                sandbox_discovery_status = "error"
                sandbox_skip_reason = f"Discovery raised exception: {e}"
                logger.exception("[spec_gate_grounded] Sandbox discovery exception: %s", e)
                
                # v1.6: BLOCKING - if sandbox task but discovery crashed, return error
                return SpecGateResult(
                    ready_for_pipeline=False,
                    open_questions=[f"Sandbox discovery failed with error: {e}"],
                    spec_version=round_n,
                    validation_status="blocked",
                    blocking_issues=[sandbox_skip_reason],
                    notes=f"Sandbox discovery exception for {anchor}/{subfolder or ''}",
                )
            
            # Handle ambiguity - return early with question
            if sandbox_discovery_result and sandbox_discovery_result.get("ambiguous") and sandbox_discovery_result.get("question"):
                if not sandbox_discovery_result.get("selected_file"):
                    return SpecGateResult(
                        ready_for_pipeline=False,
                        open_questions=[sandbox_discovery_result["question"]],
                        spec_version=round_n,
                        validation_status="needs_clarification",
                        notes=f"Sandbox ambiguity: {sandbox_discovery_result.get('ambiguous_candidates', [])}",
                    )
        
        # =================================================================
        # STEP 2: Parse Weaver Intent
        # =================================================================
        
        intent = parse_weaver_intent(constraints_hint or {})
        
        # Include user's raw text if provided
        if user_intent and user_intent.strip():
            # Strip "Astra, command:" prefix
            clean_intent = re.sub(
                r'^(?:astra[,:]?\s*)?(?:command[:\s]+)?(?:critical\s+)?(?:architecture\s*)?',
                '',
                user_intent,
                flags=re.IGNORECASE
            ).strip()
            if clean_intent:
                intent["user_text"] = clean_intent
                if not intent.get("goal"):
                    intent["goal"] = clean_intent
        
        # =================================================================
        # STEP 3: Ground Intent with Evidence
        # =================================================================
        
        spec = ground_intent_with_evidence(intent, evidence)
        
        # =================================================================
        # STEP 3.5: Populate sandbox resolution into spec
        # =================================================================
        
        if sandbox_discovery_result and sandbox_discovery_result.get("selected_file"):
            import os
            selected = sandbox_discovery_result["selected_file"]
            folder_path = sandbox_discovery_result["path"]
            output_path = os.path.join(folder_path, "reply.txt")
            
            spec.sandbox_discovery_used = True
            spec.sandbox_anchor = anchor
            spec.sandbox_subfolder = subfolder
            spec.sandbox_folder_path = folder_path
            spec.sandbox_input_path = selected["path"]
            spec.sandbox_output_path = output_path
            spec.sandbox_selected_type = selected["content_type"]
            spec.sandbox_selection_confidence = selected.get("confidence", 0.0)
            
            content = selected.get("content", "")
            if content:
                spec.sandbox_input_excerpt = content[:500] + ("..." if len(content) > 500 else "")
            
            spec.what_exists.append(f"Sandbox input: `{selected['path']}` ({selected['content_type']})")
            spec.confirmed_components.append(GroundedFact(
                description=f"Selected sandbox file: {selected['name']}",
                source="sandbox_inspector",
                path=selected["path"],
                confidence="confirmed",
            ))
            # v1.7: Read-only wording - we don't write files, just record planned output for later stages
            spec.constraints_from_repo.append(f"Planned output path (for later stages): `{output_path}`")
            
            # v1.8: Store full content and generate intelligent reply using LLM
            full_content = selected.get("content", "")
            if full_content:
                spec.sandbox_input_full_content = full_content
                # Generate reply using LLM intelligence (v1.8)
                spec.sandbox_generated_reply = await _generate_reply_from_content(
                    full_content, 
                    selected.get("content_type"),
                    provider_id=provider_id,
                    model_id=model_id,
                )
                logger.info(
                    "[spec_gate_grounded] v1.8 Generated LLM reply for sandbox content: %s",
                    spec.sandbox_generated_reply[:100] if spec.sandbox_generated_reply else "(empty)"
                )
            
            logger.info(
                "[spec_gate_grounded] Sandbox resolved: input=%s, output=%s, type=%s",
                selected["path"], output_path, selected["content_type"]
            )
        
        # v1.6: Track sandbox discovery status even if no file was selected
        spec.sandbox_discovery_status = sandbox_discovery_status
        spec.sandbox_skip_reason = sandbox_skip_reason
        
        # v1.6: If sandbox was detected but discovery failed, add warning
        if anchor and sandbox_discovery_status not in ("success", "not_attempted"):
            warning_msg = f"Sandbox file task detected but discovery {sandbox_discovery_status}"
            if sandbox_skip_reason:
                warning_msg += f": {sandbox_skip_reason}"
            spec.evidence_gaps.append(warning_msg)
            logger.warning("[spec_gate_grounded] %s", warning_msg)
        
        # v1.6: CRITICAL FIX - If sandbox_file domain detected but discovery wasn't used,
        # something is wrong (either anchor extraction failed or discovery was skipped).
        # This catches the case where domain keywords are found but anchor wasn't.
        weaver_text_for_domain = intent.get("raw_text", "") or ""
        detected_domains_check = detect_domains(weaver_text_for_domain)
        if "sandbox_file" in detected_domains_check and not spec.sandbox_discovery_used:
            logger.warning(
                "[spec_gate_grounded] v1.6 MISMATCH: sandbox_file domain detected but discovery not used! "
                "anchor=%s, weaver_text_len=%d",
                anchor, len(weaver_text_for_domain)
            )
            if not anchor:
                # Domain detected but anchor extraction failed - this is a bug
                spec.evidence_gaps.append(
                    "INTERNAL: sandbox_file domain detected but anchor extraction failed - check _extract_sandbox_hints()"
                )
        
        # =================================================================
        # STEP 4: Apply User Answers (if round 2+)
        # =================================================================
        
        if user_answers and round_n >= 2:
            # v1.5: Parse answers into spec.decisions for blocking forks
            for key, answer in user_answers.items():
                key_lower = key.lower()
                answer_lower = answer.lower() if answer else ""
                
                # Map common answer patterns to decision keys
                if "platform" in key_lower or "android" in answer_lower or "ios" in answer_lower:
                    spec.decisions["platform_v1"] = answer
                elif "input" in key_lower or "voice" in answer_lower or "screenshot" in answer_lower or "manual" in answer_lower:
                    spec.decisions["input_mode_v1"] = answer
                elif "ocr" in key_lower or "completed parcels" in answer_lower or "finish tour" in answer_lower:
                    spec.decisions["ocr_scope_v1"] = answer
                elif "sync" in key_lower:
                    if "target" in key_lower or "endpoint" in answer_lower or "export" in answer_lower:
                        spec.decisions["sync_target"] = answer
                    else:
                        spec.decisions["sync_behaviour"] = answer
                # Legacy handling for other answer types
                elif "scope" in key_lower:
                    spec.out_of_scope.append(answer)
                elif "step" in key_lower:
                    spec.proposed_steps.append(answer)
                elif "path" in key_lower or "file" in key_lower:
                    spec.what_exists.append(f"User confirmed: {answer}")
            
            logger.info(
                "[spec_gate_grounded] v1.5: Parsed user_answers into decisions: %s",
                spec.decisions
            )
        
        # =================================================================
        # STEP 5: Generate Questions (if needed)
        # =================================================================
        
        questions = generate_grounded_questions(spec, intent, evidence, round_n)
        spec.open_questions = questions
        
        # =================================================================
        # STEP 6: Determine Completion Status (v1.4: Early Exit)
        # =================================================================
        
        # v1.4: Check if spec is complete enough for early exit
        is_complete_enough, completion_reason = _is_spec_complete_enough(
            spec, intent, questions
        )
        
        logger.info(
            "[spec_gate_grounded] v1.4 Completion check: complete_enough=%s, reason='%s'",
            is_complete_enough, completion_reason
        )
        
        # Round 3 always finalizes (even with gaps)
        # IMPORTANT: We do NOT fill gaps - we just mark them as unresolved
        if round_n >= 3:
            spec.is_complete = True
            if questions:
                spec.blocking_issues.append(
                    f"Finalized with {len(questions)} unanswered question(s) - NOT guessed"
                )
            # Questions are preserved in spec.open_questions for the markdown output
        elif is_complete_enough:
            # v1.4: EARLY EXIT - spec is complete, no more rounds needed
            spec.is_complete = True
            
            # Derive steps/tests now since we're completing early
            if not spec.proposed_steps:
                spec.proposed_steps = _derive_steps_from_domain(intent, spec)
            if not spec.acceptance_tests or all('(To be determined)' in str(t) for t in spec.acceptance_tests):
                spec.acceptance_tests = _derive_tests_from_domain(intent, spec)
            
            logger.info(
                "[spec_gate_grounded] v1.4 EARLY EXIT: %s (round %d)",
                completion_reason, round_n
            )
        else:
            # v1.1 FIX: is_complete only when:
            # - No questions AND
            # - Steps exist AND
            # - Acceptance tests exist (and aren't placeholders)
            has_real_steps = bool(spec.proposed_steps)
            has_real_tests = (
                bool(spec.acceptance_tests) and
                not all('(To be determined)' in str(t) for t in spec.acceptance_tests)
            )
            
            spec.is_complete = (
                len(questions) == 0 and
                has_real_steps and
                has_real_tests
            )
            
            logger.info(
                "[spec_gate_grounded] Completion check: questions=%d, steps=%s, tests=%s -> complete=%s",
                len(questions), has_real_steps, has_real_tests, spec.is_complete
            )
        
        # =================================================================
        # STEP 7: Generate IDs and Hash (no writes!)
        # =================================================================
        
        import uuid
        spec.spec_id = f"sg-{uuid.uuid4().hex[:12]}"
        spec.spec_version = round_n
        
        # Compute hash from spec content
        hash_content = json.dumps({
            "goal": spec.goal,
            "in_scope": spec.in_scope,
            "out_of_scope": spec.out_of_scope,
            "steps": spec.proposed_steps,
            "version": round_n,
        }, sort_keys=True)
        spec.spec_hash = hashlib.sha256(hash_content.encode()).hexdigest()
        
        # =================================================================
        # STEP 8: Build POT Spec Markdown
        # =================================================================
        
        spot_md = build_pot_spec_markdown(spec)
        
        # =================================================================
        # STEP 9: Return Result (NO DB/FILE WRITES)
        # =================================================================
        
        validation_status = "validated" if spec.is_complete else "needs_clarification"
        if spec.blocking_issues:
            validation_status = "validated_with_issues" if spec.is_complete else "blocked"
        
        # Include questions in output - even if complete (Round 3), so they're visible as unresolved
        open_q_text = [q.question for q in spec.open_questions]
        
        logger.info(
            "[spec_gate_grounded] Result: complete=%s, questions=%d, round=%d",
            spec.is_complete, len(open_q_text), round_n
        )
        
        return SpecGateResult(
            ready_for_pipeline=spec.is_complete,
            # Always return questions so they're visible (especially on Round 3)
            open_questions=open_q_text,
            spot_markdown=spot_md if spec.is_complete else None,
            db_persisted=False,  # NEVER persist - read-only runtime
            spec_id=spec.spec_id,
            spec_hash=spec.spec_hash,
            spec_version=round_n,
            notes=(
                f"Evidence sources: {len(evidence.sources)}; "
                f"arch_query_used: {evidence.arch_query_used}; "
                f"evidence_complete: {spec.evidence_complete}"
            ),
            blocking_issues=[str(i) for i in spec.blocking_issues],
            validation_status=validation_status,
        )
        
    except Exception as e:
        logger.exception("[spec_gate_grounded] HARD STOP: %s", e)
        return SpecGateResult(
            ready_for_pipeline=False,
            hard_stopped=True,
            hard_stop_reason=str(e),
            spec_version=int(spec_version) if isinstance(spec_version, int) else None,
            validation_status="error",
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "run_spec_gate_grounded",
    "GroundedPOTSpec",
    "GroundedQuestion",
    "GroundedFact",
    "GroundedAssumption",  # v1.4
    "QuestionCategory",
    "build_pot_spec_markdown",
    "load_evidence",
    "WRITE_REFUSED_ERROR",
    # v1.2 additions
    "detect_domains",
    "extract_decision_forks",
    "extract_unresolved_ambiguities",
    "DOMAIN_KEYWORDS",
    "MOBILE_APP_FORK_BANK",
]
