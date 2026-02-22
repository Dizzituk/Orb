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
v1.10 (2026-01-22): GREENFIELD BUILD FIX (CREATE_NEW vs MODIFY_EXISTING)
              - CRITICAL FIX: "Target platform: Desktop" no longer triggers sandbox discovery
              - Added greenfield_build domain detection for CREATE_NEW jobs
              - Stricter sandbox_file domain patterns - require explicit file context
              - _extract_sandbox_hints() now distinguishes platform vs file location
              - Jobs like "build Tetris for desktop" skip sandbox discovery entirely
              - Only triggers sandbox discovery when clear file operations detected
              - Platform context patterns: "for desktop", "desktop app", "Target platform: Desktop"
              - File context patterns: "desktop folder", "file on desktop", "sandbox desktop"
v1.11 (2026-01-22): TECH STACK ANCHORING (upstream capture)
              - CRITICAL FIX: implementation_stack field is now populated from Weaver/conversation
              - Added _detect_implementation_stack() to extract tech stack from messages
              - Detects user explicit statements: "use Python", "I want Pygame", "let's use React"
              - Detects assistant proposals + user confirmations (sets stack_locked=True)
              - Added ImplementationStack import from schemas.py
              - Added implementation_stack to GroundedPOTSpec dataclass
              - Wired into grounding_data for Critical Pipeline job classification
              - Added "Implementation Stack" section to markdown output
              - See CRITICAL_PIPELINE_FAILURE_REPORT.md and tech_stack_anchoring_handover.md
v1.12 (2026-01-22): DOMAIN DRIFT FIX (Tetris generates courier app steps)
              - CRITICAL FIX: Domain detection was matching on QUESTION text, not job description
              - Example: Weaver asks "What platform? (web/Android/iOS)" - Android/iOS triggered mobile_app domain
              - Added _extract_job_description_only() to strip Questions/Unresolved ambiguities sections
              - detect_domains() now excludes question text by default (exclude_questions=True)
              - Added "game" domain with keywords: tetris, snake, game, playfield, score, etc.
              - Game domain takes PRIORITY over mobile_app in step/test derivation
              - Added game-specific steps and tests for game-building jobs
              - Prevents courier shift logging app steps/tests being generated for Tetris jobs
              - See specgate_domain_drift_fix.md for full details
v1.13 (2026-01-23): MICRO_FILE_TASK output mode + reduced clutter
              - Added OutputMode enum (APPEND_IN_PLACE, SEPARATE_REPLY_FILE, CHAT_ONLY)
              - Added _detect_output_mode() to determine where reply should go based on user intent
              - OutputMode detection keywords:
                  - CHAT_ONLY: "just answer here", "don't change the file", "chat only"
                  - SEPARATE_REPLY_FILE: "save to reply.txt", "create a reply file"
                  - APPEND_IN_PLACE: "write under", "append", "add below", "beneath the question"
              - Output path now varies based on detected mode:
                  - APPEND_IN_PLACE: output_path = input_path (same file)
                  - SEPARATE_REPLY_FILE: output_path = reply.txt
                  - CHAT_ONLY: output_path = None
              - Added sandbox_output_mode and sandbox_insertion_format to GroundedPOTSpec
              - Steps/tests vary based on output_mode (append vs write vs chat-only)
              - Reduced clutter in markdown output for micro tasks:
                  - Removed "Selection confidence" (not useful for micro tasks)
                  - Only show content_type if not "unknown"
              - Added sandbox_output_mode and sandbox_insertion_format to grounding_data
v1.14 (2026-01-24): ANCHOR EXTRACTION REGRESSION FIX
              - CRITICAL FIX: _extract_sandbox_hints() was too strict in v1.10
              - v1.10 required file_context_patterns to match TWICE (is_file_context + anchor extraction)
              - This broke prompts like "On the desktop, there is a folder called Test"
              - Fix: If sandbox_file domain detected but anchor=None, fall back to Desktop discovery
              - Added fallback_anchor_for_sandbox_domain() helper
              - Anchor extraction now tries:
                1. Strict file_context_patterns (v1.10 behavior)
                2. Fallback: "desktop" or "documents" mention + file operation keywords
              - Ensures prompts like "read the file in Test folder on desktop" still work
              - Fixes regression where spec was "Complete" but had no grounded file data
v1.15 (2026-01-25): SANDBOX_FILE DOMAIN OVERRIDE FIX
              - CRITICAL FIX: _extract_sandbox_hints() v1.10 platform context check was too aggressive
              - When Weaver summarizes "On the desktop, there is a folder called Test" to
                "Target platform: Desktop" + "read the file", v1.10 would return (None, None)
              - Symptom: sandbox_file keywords detected but anchor=None (logged as BUG)
              - Fix: If sandbox_file domain keywords detected ("read the file", "test.txt", etc.),
                force is_file_context=True even when platform context patterns match
              - New logic flow:
                1. Check for sandbox_file domain keywords FIRST
                2. If found, override the platform context early-return
                3. Force anchor extraction using fallback patterns
              - Added has_sandbox_file_signals detection at function start
              - Added basic_file_indicators fallback for is_file_context override
              - Ensures micro file tasks work even when Weaver output includes "Target platform: Desktop"
v1.16 (2026-01-25): Q&A FILE INTELLIGENCE + OUTPUT MODE SAFETY FIX
              - Added _analyze_qa_file() to detect Question/Answer structure in files
              - Added _process_question_block() helper for parsing Q&A blocks
              - _generate_reply_from_content() now detects answered vs unanswered questions
              - Only generates answers for UNANSWERED questions
              - LLM prompt explicitly instructs to EXPLAIN code, not just output result
              - Prevents "0 1 2" bug where LLM outputs raw code execution
              - Increased excerpt display from 5 to 15 lines in build_pot_spec_markdown()
              - CRITICAL FIX: _detect_output_mode() now uses simple string matching FIRST
              - Simple phrases like "do not change" match within longer sentences
              - Catches "Do not change anything that is in that file" → CHAT_ONLY
              - Safety phrases checked BEFORE append/write triggers (absolute override)
              - Added "question in chat" regex to catch "answer the question in chat"
v1.17 (2026-01-25): REWRITE_IN_PLACE TRIGGER EXPANSION
              - Added explicit "fill in" triggers to REWRITE_IN_PLACE (not APPEND_IN_PLACE):
                  - "fill in the missing", "fill the missing", "fill in the answer"
                  - "fill blank", "fill empty", "populate answers", "complete answers"
                  - "under answer:", "into the file under", "preserve everything else"
              - REWRITE_IN_PLACE = intelligent Q&A block-aware insertion
              - APPEND_IN_PLACE = simple append at end of file
              - Example prompt: "Fill the missing answers into the file under Answer: headings"
                triggers REWRITE_IN_PLACE (reads file, finds blank Answer blocks, inserts)
v1.18 (2026-01-25): SCAN_ONLY JOB TYPE SUPPORT
              - Added "scan_only" domain to DOMAIN_KEYWORDS for read-only filesystem scans
              - Added scan_only detection to classify_job_kind() (priority before greenfield_build)
              - SCAN_ONLY jobs: No sandbox_input_path/output_path required, CHAT_ONLY output
              - Keywords: scan the, scan all, find all occurrences, search for, enumerate, etc.
              - Example: "Scan D:\\Orb for references to orb/Orb/ORB" triggers scan_only
              - Grounding_data passes job_kind=scan_only to Critical Pipeline
              - Critical Pipeline v2.5 handles SCAN_ONLY path with scan_quickcheck()
v1.19 (2026-01-25): SCAN_ONLY PARAMETER EXTRACTION (Complete Feature)
              - CRITICAL FIX: SpecGate now extracts scan parameters for SCAN_ONLY jobs
              - Added _extract_scan_params() to parse user intent into scan fields
              - Extracts scan_roots from drive letters ("D drive") and explicit paths ("D:\\Orb")
              - Extracts scan_terms from slash-separated variants (Orb/ORB/orb), quoted strings, descriptive patterns
              - Extracts scan_targets from keywords ("names" for filenames, "contents" for code)
              - Detects scan_case_mode from explicit keywords or multiple case variants
              - Added scan fields to GroundedPOTSpec dataclass
              - Added scan fields to grounding_data for Critical Pipeline:
                  - scan_roots, scan_terms, scan_targets, scan_case_mode, scan_exclusions
                  - output_mode="chat_only", write_policy="read_only"
              - Critical Pipeline scan_quickcheck() will now pass for valid scan prompts
              - Example: "Scan D: for Orb/ORB/orb" produces:
                  scan_roots=["D:\\"], scan_terms=["Orb", "ORB", "orb"], scan_targets=["names", "contents"]
v1.21 (2026-01-25): SCAN_ONLY SECURITY FIX (CRITICAL - HOST PC PROTECTION)
              - CRITICAL SECURITY FIX: SCAN_ONLY was incorrectly allowing bare drive roots (D:\\, C:\\)
              - Host PC filesystem should NEVER be scanned - only sandbox workspace paths allowed
              - Added SAFE_DEFAULT_SCAN_ROOTS: ["workspace", "project", "src", "D:\\Sandbox", etc.]
              - Added FORBIDDEN_SCAN_ROOTS: ["C:\\", "D:\\", "E:\\", "/", "/home", etc.]
              - Added _validate_scan_roots() to sanitize and reject dangerous roots
              - Root validation flow:
                1. If detected roots are in FORBIDDEN_SCAN_ROOTS → use safe defaults
                2. If detected roots are workspace-relative → use as-is
                3. If no roots detected → use safe defaults
              - Explicit paths like "D:\\Orb" are allowed (specific project, not bare drive)
              - Logging added: "[SECURITY] Blocked dangerous scan roots: ..."
              - This prevents accidental host PC scans from user prompts like "scan D: for..."
v1.25 (2026-01-25): MODULARIZATION COMPLETE
              - Refactored monolithic 5500-line file into 10 focused modules
              - This file now imports from app.pot_spec.grounded subpackage
              - All functionality preserved, all version history maintained
              - Module structure:
                  - spec_models.py: Data structures, enums, constants
                  - domain_detection.py: Domain keywords, fork extraction
                  - job_classification.py: Job kind determination
                  - scan_operations.py: CRITICAL SECURITY - scan validation
                  - sandbox_discovery.py: Sandbox hints, output mode
                  - tech_stack_detection.py: Tech stack anchoring
                  - qa_processing.py: Q&A analysis, LLM replies
                  - evidence_gathering.py: v1.25 Evidence-First
                  - spec_generation.py: Main entry point, spec building
                  - __init__.py: Package exports

============================================================
MODULARIZATION NOTE (v1.25):
This file is now a thin re-export layer over the grounded subpackage.
All implementation lives in app/pot_spec/grounded/*.py modules.
External imports should continue to work via this file or can
import directly from app.pot_spec.grounded for specific modules.
============================================================
"""

# =============================================================================
# IMPORTS FROM MODULARIZED SUBPACKAGE
# =============================================================================

# Re-export all public APIs from the grounded subpackage
from app.pot_spec.grounded import (
    # === Data Structures (spec_models.py) ===
    GroundedPOTSpec,
    GroundedQuestion,
    GroundedFact,
    GroundedAssumption,
    QuestionCategory,
    
    # === Domain Detection (domain_detection.py) ===
    DOMAIN_KEYWORDS,
    MOBILE_APP_FORK_BANK,
    detect_domains,
    extract_decision_forks,
    extract_unresolved_ambiguities,
    
    # === Job Classification (job_classification.py) ===
    EVIDENCE_CONFIG,
    classify_job_kind,
    classify_job_size,
    
    # === Scan Operations - CRITICAL SECURITY (scan_operations.py) ===
    SAFE_DEFAULT_SCAN_ROOTS,
    FORBIDDEN_SCAN_ROOTS,
    validate_scan_roots,
    extract_scan_params,
    
    # === Sandbox Discovery (sandbox_discovery.py) ===
    OutputMode,
    extract_sandbox_hints,
    detect_output_mode,
    extract_replacement_text,
    
    # === Tech Stack Detection (tech_stack_detection.py) ===
    STACK_DETECTION_PATTERNS,
    detect_implementation_stack,
    
    # === Q&A Processing (qa_processing.py) ===
    analyze_qa_file,
    generate_reply_from_content,
    detect_simple_instruction,
    
    # === Evidence Gathering (evidence_gathering.py) ===
    FileEvidence,
    EvidencePackage,
    gather_filesystem_evidence,
    resolve_path_enhanced,
    
    # === Spec Generation - Main Entry Point (spec_generation.py) ===
    parse_weaver_intent,
    ground_intent_with_evidence,
    generate_grounded_questions,
    build_pot_spec_markdown,
    run_spec_gate_grounded,  # MAIN ASYNC ENTRY POINT
)

# =============================================================================
# VERSION DIAGNOSTIC (for debugging import issues)
# =============================================================================

def _version_check():
    """Quick diagnostic to verify module loading."""
    import sys
    grounded_modules = [k for k in sys.modules.keys() if 'grounded' in k]
    return {
        'version': 'v1.25 (modularized)',
        'loaded_modules': grounded_modules,
        'main_entry': 'run_spec_gate_grounded',
    }

# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

# For any code that imports specific internal functions, provide aliases
# These point to the same functions in the subpackage

# Alias for legacy internal function names (underscore-prefixed)
_detect_domains = detect_domains
_extract_sandbox_hints = extract_sandbox_hints
_detect_output_mode = detect_output_mode
_extract_scan_params = extract_scan_params
_validate_scan_roots = validate_scan_roots
_detect_implementation_stack = detect_implementation_stack
_analyze_qa_file = analyze_qa_file
_generate_reply_from_content = generate_reply_from_content

# =============================================================================
# MODULE-LEVEL EXPORTS
# =============================================================================

__all__ = [
    # Data structures
    'GroundedPOTSpec',
    'GroundedQuestion', 
    'GroundedFact',
    'GroundedAssumption',
    'QuestionCategory',
    'OutputMode',
    'FileEvidence',
    'EvidencePackage',
    
    # Constants
    'DOMAIN_KEYWORDS',
    'MOBILE_APP_FORK_BANK',
    'EVIDENCE_CONFIG',
    'SAFE_DEFAULT_SCAN_ROOTS',
    'FORBIDDEN_SCAN_ROOTS',
    'STACK_DETECTION_PATTERNS',
    
    # Core functions
    'detect_domains',
    'extract_decision_forks',
    'extract_unresolved_ambiguities',
    'classify_job_kind',
    'classify_job_size',
    'validate_scan_roots',
    'extract_scan_params',
    'extract_sandbox_hints',
    'detect_output_mode',
    'extract_replacement_text',
    'detect_implementation_stack',
    'analyze_qa_file',
    'generate_reply_from_content',
    'detect_simple_instruction',
    'gather_filesystem_evidence',
    'resolve_path_enhanced',
    'parse_weaver_intent',
    'ground_intent_with_evidence',
    'generate_grounded_questions',
    'build_pot_spec_markdown',
    'run_spec_gate_grounded',
]
