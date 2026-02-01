# FILE: app/pot_spec/grounded/spec_runner.py
"""
SpecGate Main Entry Point Runner

This module provides the main async entry point for running SpecGate
Contract v1 - Grounded POT Spec Builder.

Responsibilities:
- Orchestrate the 9-step spec generation process
- Load evidence and run sandbox discovery
- Detect multi-file operations and multi-target reads
- Parse Weaver intent and ground against evidence
- Generate questions and determine completion status
- Build final POT spec markdown

Key Features:
- v1.25: Evidence-First architecture
- v1.33: Multi-file operations wiring
- v1.34: Multi-target file read support
- v1.39: Multi-target read handling fixes
- v2.0: Intelligent refactor classification

RUNTIME IS READ-ONLY:
- No filesystem writes
- No DB writes
- Output/stream only

Used by:
- External callers needing SpecGate functionality

Version: v2.0 (2026-02-01) - Extracted from spec_generation.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Build ID for verification
# v2.4 (2026-02-01): Fixed regex to match paths with spaces
#   - "D:\Orb Desktop" was being truncated to "D:\Orb" because \w doesn't include spaces
#   - Now uses [^delimiter] pattern to capture full paths
# v2.6 (2026-02-01): VISION CONTEXT FLOW FIX
#   - Added _extract_vision_context_from_messages() for extracting Gemini vision analysis
#   - Vision context is now passed to _build_multi_file_operation() for intelligent classification
#   - Classifier now knows which matches are USER-VISIBLE UI elements
# v2.7 (2026-02-01): VISION CONTEXT FROM FLOW STATE (COMPLETE FIX)
#   - Now checks constraints_hint["vision_context"] directly from flow state
#   - Flow state vision context is passed from Weaver → SpecFlowState → spec_gate_stream → constraints_hint
#   - Prefers direct vision_context over extracting from messages
#   - This completes the Weaver → SpecGate vision context data flow
SPEC_RUNNER_BUILD_ID = "2026-02-01-v2.7-vision-context-complete"
print(f"[SPEC_RUNNER_LOADED] BUILD_ID={SPEC_RUNNER_BUILD_ID}")
logger.info(f"[spec_runner] Module loaded: BUILD_ID={SPEC_RUNNER_BUILD_ID}")

# =============================================================================
# IMPORTS FROM SIBLING MODULES
# =============================================================================

from .spec_models import (
    GroundedFact,
    FileTarget,
    GroundedPOTSpec,
)

from .domain_detection import detect_domains

from .job_classification import classify_job_kind

from .scan_operations import (
    DEFAULT_SCAN_EXCLUSIONS,
    extract_scan_params,
)

from .sandbox_discovery import (
    OutputMode,
    extract_sandbox_hints,
    detect_output_mode,
    extract_replacement_text,
)

from .tech_stack_detection import detect_implementation_stack

from .qa_processing import (
    generate_reply_from_content,
    analyze_qa_file,
)

from .evidence_gathering import (
    gather_filesystem_evidence,
    format_evidence_for_prompt,
    format_multi_target_reply,
    sandbox_read_file,
)

from .multi_file_detection import (
    _detect_multi_file_intent,
    _build_multi_file_operation,
)

from .text_helpers import _extract_paths_from_text

from .weaver_parser import parse_weaver_intent
from .grounding_engine import ground_intent_with_evidence
from .question_generator import generate_grounded_questions
from .step_derivation import _derive_steps_from_domain, _derive_tests_from_domain
from .completeness_checker import _is_spec_complete_enough
from .markdown_builder import build_pot_spec_markdown

# =============================================================================
# EXTERNAL IMPORTS (with fallbacks)
# =============================================================================

# Evidence collector
try:
    from ..evidence_collector import (
        EvidenceBundle,
        load_evidence,
    )
    _EVIDENCE_AVAILABLE = True
except ImportError as e:
    logger.warning("[spec_runner] evidence_collector not available: %s", e)
    _EVIDENCE_AVAILABLE = False
    EvidenceBundle = None
    load_evidence = None

# SpecGateResult type
try:
    from ..spec_gate_types import SpecGateResult
except ImportError:
    from dataclasses import dataclass, field
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
        grounding_data: Optional[Dict] = None

# LLM call function
try:
    from app.providers.registry import llm_call
    _LLM_CALL_AVAILABLE = True
except ImportError as e:
    logger.warning("[spec_runner] llm_call not available: %s", e)
    _LLM_CALL_AVAILABLE = False
    llm_call = None

# Sandbox inspection
try:
    from app.llm.local_tools.zobie.sandbox_inspector import (
        run_sandbox_discovery_chain,
    )
    _SANDBOX_INSPECTOR_AVAILABLE = True
except ImportError:
    _SANDBOX_INSPECTOR_AVAILABLE = False
    run_sandbox_discovery_chain = None


__all__ = [
    "run_spec_gate_grounded",
]


# =============================================================================
# VISION CONTEXT EXTRACTION (v2.6)
# =============================================================================

# Patterns that indicate an assistant message contains vision/image analysis
# Copied from weaver_stream.py to ensure consistency
VISION_CONTEXT_PATTERNS = [
    # Screenshot/image descriptions
    r"screenshot", r"image shows", r"i can see", r"i see a",
    r"the image", r"in the picture", r"looking at the",
    # UI element descriptions (from vision analysis)
    r"title bar", r"window title", r"menu bar", r"status bar",
    r"status indicator", r"toolbar", r"heading.*says",
    r"button.*labeled", r"text.*reads", r"displays.*text",
    r"shows.*logo", r"cyan.*text", r"blue.*text", r"icon.*shows",
    # Visual descriptions
    r"ui shows", r"ui elements", r"visible.*elements",
    r"display shows", r"interface shows", r"window shows", r"window contains",
    # Color/appearance descriptions
    r"dark\s*(?:theme|mode|background)", r"light\s*(?:theme|mode|background)",
    r"colored.*(?:text|background|border)",
    # Position descriptions
    r"top.*(?:left|right|corner)", r"bottom.*(?:left|right|corner)",
    r"center of", r"sidebar",
    # Action analysis phrases
    r"appears to be", r"looks like", r"seems to show",
]


def _is_vision_context(content: str) -> bool:
    """
    Detect if an assistant message contains vision/image analysis.
    
    v2.6: Vision analysis from Gemini should be identified for downstream use.
    This context is valuable for SpecGate classifier to understand which
    matches are USER-VISIBLE UI elements.
    
    Returns True if the message likely contains vision analysis.
    """
    if not content:
        return False
    
    content_lower = content.lower()
    
    # Check for vision context patterns
    for pattern in VISION_CONTEXT_PATTERNS:
        if re.search(pattern, content_lower, re.IGNORECASE):
            return True
    
    return False


def _extract_vision_context_from_messages(messages: List[Dict[str, Any]]) -> str:
    """
    Extract vision context from conversation messages.
    
    v2.6: Looks for assistant messages containing Gemini vision analysis
    (descriptions of screenshots, UI elements, visual content).
    
    Args:
        messages: List of conversation messages (role + content)
        
    Returns:
        Concatenated vision context string, or empty string if none found.
    """
    vision_parts = []
    
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if _is_vision_context(content):
                # Extract relevant portions (first 1000 chars to avoid bloat)
                vision_parts.append(content[:1000])
                logger.info(
                    "[spec_runner] v2.6 Found vision context in assistant message (%d chars)",
                    len(content[:1000])
                )
    
    if vision_parts:
        combined = "\n\n".join(vision_parts)
        print(f"[spec_runner] v2.6 VISION CONTEXT extracted: {len(combined)} chars from {len(vision_parts)} messages")
        return combined
    
    return ""


# =============================================================================
# v2.2: WINDOWS PATH EXTRACTION HELPER
# =============================================================================

def _extract_windows_project_paths(text: str) -> List[str]:
    """
    v2.2: Extract Windows-style project paths from natural language text.
    
    Handles patterns like:
    - "D:\\orb-desktop" or "D:/orb-desktop"
    - "D drive, Orb Desktop" -> "D:\\orb-desktop"
    - "in the D drive" + "orb-desktop" folder name
    - "Orb Desktop" (standalone) -> "D:\\orb-desktop" (if D drive mentioned)
    
    Returns:
        List of potential project paths (best-effort extraction)
    """
    if not text:
        return []
    
    paths = []
    text_lower = text.lower()
    
    # Pattern 1: Explicit Windows paths (D:\folder or D:/folder)
    # v2.5: More conservative - stop at common terminators like "front", "files", etc.
    # This prevents matching "D:\Orb Desktop front-end files" instead of "D:\Orb Desktop"
    explicit_path_matches = re.findall(
        r'([A-Za-z]:[\\/][A-Za-z][A-Za-z0-9_\s\-]+?)(?=\s+(?:front|files|folder|directory|project|UI|code|source|—)|[,;()\[\]{}"\n]|$)',
        text
    )
    # Clean up: strip trailing whitespace and common punctuation
    for p in explicit_path_matches:
        cleaned = p.rstrip(' \t')
        if cleaned and len(cleaned) > 3:  # At least "D:\x"
            paths.append(cleaned)
            logger.info("[spec_runner] v2.5 Extracted explicit path: '%s'", cleaned)
            print(f"[spec_runner] v2.5 PATH EXTRACTED: '{cleaned}'")
            
            # v2.5: Also add hyphenated-lowercase variation
            # "D:\Orb Desktop" -> "D:\orb-desktop"
            if ' ' in cleaned and len(cleaned) > 3:
                drive_part = cleaned[:3]  # "D:\\"
                folder_part = cleaned[3:]  # "Orb Desktop"
                hyphenated = folder_part.replace(' ', '-').lower()
                variation = drive_part + hyphenated
                if variation not in paths:
                    paths.append(variation)
                    logger.info("[spec_runner] v2.5 Added hyphenated variation: '%s'", variation)
                    print(f"[spec_runner] v2.5 PATH VARIATION: '{variation}'")
    
    # Pattern 2: "X drive" detection
    drive_match = re.search(r'\b([A-Za-z])\s+drive\b', text, re.IGNORECASE)
    drive_letter = drive_match.group(1).upper() if drive_match else None
    
    # Pattern 3: Known project name patterns (case-insensitive)
    # These are common project folder naming conventions
    # v2.5: Add BOTH spaced AND hyphenated versions
    known_project_patterns = [
        (r'\borb[\s-]*desktop\b', ['Orb Desktop', 'orb-desktop']),
        (r'\bastra[\s-]*desktop\b', ['Astra Desktop', 'astra-desktop']),
        (r'\borb[\s-]*app\b', ['Orb App', 'orb-app']),
        (r'\bastra[\s-]*app\b', ['Astra App', 'astra-app']),
    ]
    
    for pattern, folder_names in known_project_patterns:
        if re.search(pattern, text_lower):
            if drive_letter:
                for folder_name in folder_names:
                    constructed = f"{drive_letter}:\\{folder_name}"
                    if constructed not in paths:
                        paths.append(constructed)
                        logger.info("[spec_runner] v2.5 Inferred project path: %s", constructed)
                        print(f"[spec_runner] v2.5 PATH INFERRED: '{constructed}'")
    
    # Pattern 4: Generic folder references near drive
    if drive_letter:
        folder_patterns = [
            r'drive[,\s]+([A-Za-z][A-Za-z0-9_-]+(?:\s+[A-Za-z][A-Za-z0-9_-]+)?)',  # "D drive, Orb Desktop"
            r'([A-Za-z][A-Za-z0-9_-]+(?:-[A-Za-z][A-Za-z0-9_-]+)?)\s+(?:folder|directory|project)',  # "orb-desktop folder"
            r'(?:find|in|at|called)\s+([A-Za-z][A-Za-z0-9_-]+(?:[\s-][A-Za-z][A-Za-z0-9_-]+)?)',  # "find Orb Desktop"
        ]
        
        for pattern in folder_patterns:
            folder_match = re.search(pattern, text, re.IGNORECASE)
            if folder_match:
                folder_name = folder_match.group(1).strip()
                # v2.3: Try BOTH versions - with space and with hyphen
                # Some folders use spaces ("Orb Desktop"), some use hyphens ("orb-desktop")
                # Skip if it's a stopword
                if folder_name.lower() not in ('the', 'a', 'an', 'to', 'in', 'on', 'it'):
                    # Version 1: Preserve original capitalization and spacing
                    constructed_path_original = f"{drive_letter}:\\{folder_name}"
                    if constructed_path_original not in paths:
                        paths.append(constructed_path_original)
                    
                    # Version 2: Also try hyphenated lowercase version
                    folder_name_hyphenated = folder_name.replace(' ', '-').lower()
                    constructed_path_hyphen = f"{drive_letter}:\\{folder_name_hyphenated}"
                    if constructed_path_hyphen not in paths and constructed_path_hyphen.lower() != constructed_path_original.lower():
                        paths.append(constructed_path_hyphen)
    
    # Pattern 5: Hyphenated project names anywhere in text
    project_folder_matches = re.findall(
        r'\b([A-Za-z][A-Za-z0-9]*-[A-Za-z][A-Za-z0-9_-]*)\b',
        text
    )
    for folder in project_folder_matches:
        folder_lower = folder.lower()
        # Filter out common non-project patterns
        if folder_lower not in ('case-insensitive', 'case-sensitive', 'read-only', 'front-end', 'back-end'):
            if drive_letter:
                constructed = f"{drive_letter}:\\{folder_lower}"
                if constructed not in paths:
                    paths.append(constructed)
    
    # Deduplicate while preserving order and normalizing case
    seen = set()
    unique_paths = []
    for p in paths:
        p_normalized = p.lower().replace('/', '\\')
        if p_normalized not in seen:
            seen.add(p_normalized)
            unique_paths.append(p)
    
    logger.info("[spec_runner] v2.2 Extracted project paths: %s", unique_paths)
    return unique_paths


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
            "[spec_runner] Starting round %d for job %s (project %d)",
            round_n, job_id, project_id
        )
        
        # =================================================================
        # STEP 1: Load Evidence (read-only)
        # =================================================================
        
        if not _EVIDENCE_AVAILABLE or not load_evidence:
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
            "[spec_runner] Loaded evidence: %d sources, %d errors",
            len(evidence.sources),
            len(evidence.errors),
        )
        
        # =================================================================
        # STEP 1.5: Sandbox Discovery (if sandbox job detected)
        # =================================================================
        
        sandbox_discovery_result = None
        sandbox_discovery_status = "not_attempted"
        sandbox_skip_reason = None
        
        weaver_job_text = (constraints_hint or {}).get('weaver_job_description_text', '')
        combined_text = f"{user_intent or ''} {weaver_job_text}"
        anchor, subfolder = extract_sandbox_hints(combined_text)
        
        logger.info(
            "[spec_runner] v1.6 Sandbox hint extraction: anchor=%s, subfolder=%s, text_len=%d",
            anchor, subfolder, len(combined_text)
        )
        
        # =================================================================
        # STEP 1.6: EVIDENCE-FIRST FILESYSTEM VALIDATION (v1.25)
        # =================================================================
        
        fs_evidence = None
        fs_evidence_block = ""
        
        if anchor or "sandbox_file" in detect_domains(combined_text):
            logger.info(
                "[spec_runner] v1.25 EVIDENCE-FIRST: Gathering filesystem evidence..."
            )
            
            # Convert RAG evidence sources to dicts for rag_hints
            rag_hints = None
            if evidence and evidence.sources:
                rag_hints = [
                    {
                        "source_type": src.source_type,
                        "filename": src.filename,
                        "path": src.path,
                        "found": src.found,
                        "error": src.error,
                    }
                    for src in evidence.sources
                ]
            
            fs_evidence = gather_filesystem_evidence(
                combined_text=combined_text,
                anchor=anchor,
                subfolder=subfolder,
                rag_hints=rag_hints,
            )
            
            if fs_evidence:
                logger.info(
                    "[spec_runner] v1.25 Evidence gathered: %s",
                    fs_evidence.to_summary()
                )
                
                fs_evidence_block = format_evidence_for_prompt(fs_evidence)
                
                if fs_evidence.has_valid_targets():
                    primary = fs_evidence.get_primary_target()
                    logger.info(
                        "[spec_runner] v1.25 PRIMARY TARGET FOUND: %s (exists=%s, readable=%s)",
                        primary.resolved_path, primary.exists, primary.readable
                    )
                else:
                    if fs_evidence.validation_errors:
                        logger.warning(
                            "[spec_runner] v1.25 Evidence validation errors: %s",
                            fs_evidence.validation_errors
                        )
        
        # v1.25: SHORT-CIRCUIT - Use evidence to pre-populate sandbox discovery result
        evidence_based_sandbox_result = None
        
        if fs_evidence and fs_evidence.has_valid_targets():
            primary = fs_evidence.get_primary_target()
            if primary and primary.resolved_path and primary.exists and primary.readable:
                logger.info(
                    "[spec_runner] v1.25 Using evidence-first path resolution: %s",
                    primary.resolved_path
                )
                evidence_based_sandbox_result = {
                    "selected_file": {
                        "path": primary.resolved_path,
                        "name": os.path.basename(primary.resolved_path),
                        "content_type": primary.detected_structure or "plain_text",
                        "content": primary.content_preview,
                        "confidence": 1.0,
                    },
                    "path": os.path.dirname(primary.resolved_path),
                    "ambiguous": False,
                    "evidence_source": "v1.25_evidence_first",
                }
                
                if primary.resolved_path:
                    success, full_content = sandbox_read_file(primary.resolved_path, max_chars=50000)
                    if success and full_content:
                        evidence_based_sandbox_result["selected_file"]["content"] = full_content
                        logger.info(
                            "[spec_runner] v1.26 Read full content via sandbox: %d chars",
                            len(full_content)
                        )
                    else:
                        logger.warning(
                            "[spec_runner] v1.26 Could not read full content from sandbox: %s",
                            primary.resolved_path
                        )
        
        # v1.10: GREENFIELD BUILD CHECK
        detected_domains = detect_domains(combined_text)
        is_greenfield = "greenfield_build" in detected_domains
        
        if is_greenfield:
            logger.info("[spec_runner] v1.10 GREENFIELD BUILD detected - skipping sandbox discovery")
            anchor = None
            sandbox_skip_reason = "Greenfield build detected (CREATE_NEW job type)"
        
        # =================================================================
        # STEP 1.7: Multi-File Operation Detection (v1.33 / v2.2 context-aware)
        # =================================================================
        
        multi_file_op = None
        
        # v2.2: Extract project paths for context-aware inference
        extracted_project_paths = _extract_windows_project_paths(combined_text)
        logger.info(
            "[spec_runner] v2.2 Extracted project paths for multi-file context: %s",
            extracted_project_paths
        )
        
        # v2.2: Extract vision results from constraints_hint if available
        # (Gemini Vision analysis results may be passed from upstream)
        vision_results = None
        if constraints_hint:
            vision_results = constraints_hint.get('vision_analysis') or constraints_hint.get('vision_results')
            if vision_results:
                logger.info(
                    "[spec_runner] v2.2 Vision results available for multi-file context: %s",
                    list(vision_results.keys()) if isinstance(vision_results, dict) else 'present'
                )
        
        # v2.6: Extract conversation messages early for vision context extraction
        conversation_messages = []
        if constraints_hint:
            conversation_messages = constraints_hint.get("messages", [])
            if not conversation_messages:
                conversation_messages = constraints_hint.get("conversation", [])
        
        # v2.7: Also check for direct vision_context key from flow state
        direct_vision_context = None
        if constraints_hint:
            direct_vision_context = constraints_hint.get("vision_context", "")
            if direct_vision_context:
                logger.info(
                    "[spec_runner] v2.7 Direct vision_context from constraints_hint: %d chars",
                    len(direct_vision_context)
                )
                print(f"[spec_runner] v2.7 DIRECT VISION CONTEXT from constraints_hint: {len(direct_vision_context)} chars")
        
        multi_file_meta = _detect_multi_file_intent(
            combined_text=combined_text,
            constraints_hint=constraints_hint,
            project_paths=extracted_project_paths,
            vision_results=vision_results,
        )
        
        if multi_file_meta and multi_file_meta.get("is_multi_file"):
            logger.info(
                "[spec_runner] v1.33 MULTI-FILE OPERATION detected: type=%s, pattern=%s",
                multi_file_meta.get("operation_type"),
                multi_file_meta.get("search_pattern"),
            )
            
            # v2.2: Determine file filter for UI-specific tasks
            ui_file_filter = None
            text_lower = combined_text.lower()
            if any(term in text_lower for term in ['front-end', 'frontend', 'ui ', ' ui', 'user interface', 'branding']):
                ui_file_filter = '*.tsx,*.jsx,*.html,*.css,*.ts,*.js,*.json'
                logger.info("[spec_runner] v2.2 UI-task detected, filtering to: %s", ui_file_filter)
            
            # v2.7: Extract vision context - prefer direct from constraints_hint, fallback to messages
            vision_context = ""
            if direct_vision_context:
                # v2.7: Use direct vision context from flow state (most reliable)
                vision_context = direct_vision_context
                logger.info(
                    "[spec_runner] v2.7 Using DIRECT vision context from flow state (%d chars)",
                    len(vision_context)
                )
                print(f"[spec_runner] v2.7 VISION CONTEXT from flow state: {len(vision_context)} chars")
            elif conversation_messages:
                # v2.6: Fallback to extracting from conversation messages
                vision_context = _extract_vision_context_from_messages(conversation_messages)
                if vision_context:
                    logger.info(
                        "[spec_runner] v2.6 Extracted vision context from messages (%d chars)",
                        len(vision_context)
                    )
            
            multi_file_op = await _build_multi_file_operation(
                operation_type=multi_file_meta.get("operation_type", "search"),
                search_pattern=multi_file_meta.get("search_pattern", ""),
                replacement_pattern=multi_file_meta.get("replacement_pattern", ""),
                file_filter=ui_file_filter or multi_file_meta.get("file_filter"),
                sandbox_client=None,
                job_description=weaver_job_text or combined_text,
                provider_id=provider_id,
                model_id=model_id,
                explicit_roots=extracted_project_paths if extracted_project_paths else None,
                vision_context=vision_context,  # v2.6: Pass vision context for intelligent classification
            )
            
            logger.info(
                "[spec_runner] v1.33 Multi-file discovery result: files=%d, occurrences=%d, error=%s",
                multi_file_op.total_files,
                multi_file_op.total_occurrences,
                multi_file_op.error_message,
            )
            
            # v1.33: Early return for refactor operations requiring confirmation
            if multi_file_op.operation_type == "refactor" and not multi_file_op.confirmed:
                preview_text = multi_file_op.file_preview if multi_file_op.file_preview else "(no preview available)"
                confirmation_question = (
                    f"⚠️ **Multi-File Refactor Confirmation Required**\n\n"
                    f"This operation will modify **{multi_file_op.total_files}** files "
                    f"({multi_file_op.total_occurrences} occurrences).\n\n"
                    f"**Pattern:** `{multi_file_op.search_pattern}`\n"
                    f"**Replace with:** `{multi_file_op.replacement_pattern}`\n\n"
                    f"**Preview:**\n```\n{preview_text}\n```\n\n"
                    f"Proceed with this refactor? (yes/no)"
                )
                
                logger.warning(
                    "[spec_runner] v1.33 REFACTOR CONFIRMATION REQUIRED: %d files, %d occurrences",
                    multi_file_op.total_files,
                    multi_file_op.total_occurrences,
                )
                
                return SpecGateResult(
                    ready_for_pipeline=False,
                    open_questions=[confirmation_question],
                    spec_version=round_n,
                    validation_status="needs_confirmation",
                    grounding_data={
                        "multi_file": multi_file_op.to_dict() if hasattr(multi_file_op, 'to_dict') else {
                            "is_multi_file": multi_file_op.is_multi_file,
                            "operation_type": multi_file_op.operation_type,
                            "search_pattern": multi_file_op.search_pattern,
                            "replacement_pattern": multi_file_op.replacement_pattern,
                            "total_files": multi_file_op.total_files,
                            "total_occurrences": multi_file_op.total_occurrences,
                            "requires_confirmation": multi_file_op.requires_confirmation,
                            "confirmed": multi_file_op.confirmed,
                        },
                    },
                    notes=f"Multi-file refactor pending confirmation: {multi_file_op.total_files} files",
                )
        
        # Sandbox discovery logic
        if not anchor:
            sandbox_skip_reason = "No sandbox anchor detected"
        elif not _SANDBOX_INSPECTOR_AVAILABLE:
            sandbox_skip_reason = "sandbox_inspector module not available"
            return SpecGateResult(
                ready_for_pipeline=False,
                open_questions=["Sandbox discovery tools unavailable."],
                spec_version=round_n,
                validation_status="blocked",
                blocking_issues=[sandbox_skip_reason],
            )
        elif not run_sandbox_discovery_chain:
            sandbox_skip_reason = "run_sandbox_discovery_chain function not available"
            return SpecGateResult(
                ready_for_pipeline=False,
                open_questions=["Sandbox discovery function unavailable."],
                spec_version=round_n,
                validation_status="blocked",
                blocking_issues=[sandbox_skip_reason],
            )
        elif evidence_based_sandbox_result:
            logger.info(
                "[spec_runner] v1.25 EVIDENCE-FIRST: Using pre-resolved sandbox result (skipping discovery chain)"
            )
            sandbox_discovery_result = evidence_based_sandbox_result
            sandbox_discovery_status = "evidence_first_success"
        else:
            logger.info("[spec_runner] Running sandbox discovery: anchor=%s, subfolder=%s", anchor, subfolder)
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
                    elif sandbox_discovery_result.get("ambiguous"):
                        sandbox_discovery_status = "ambiguous"
                    else:
                        sandbox_discovery_status = "no_match"
                        sandbox_skip_reason = "Discovery ran but found no matching file"
                        return SpecGateResult(
                            ready_for_pipeline=False,
                            open_questions=[f"Could not find target file in sandbox {anchor}/{subfolder or ''}."],
                            spec_version=round_n,
                            validation_status="blocked",
                            blocking_issues=[sandbox_skip_reason],
                        )
                else:
                    sandbox_discovery_status = "empty_result"
                    sandbox_skip_reason = "Discovery returned None/empty"
                    return SpecGateResult(
                        ready_for_pipeline=False,
                        open_questions=["Sandbox discovery returned no results."],
                        spec_version=round_n,
                        validation_status="blocked",
                        blocking_issues=[sandbox_skip_reason],
                    )
                    
            except Exception as e:
                sandbox_discovery_status = "error"
                sandbox_skip_reason = f"Discovery raised exception: {e}"
                logger.exception("[spec_runner] Sandbox discovery exception: %s", e)
                return SpecGateResult(
                    ready_for_pipeline=False,
                    open_questions=[f"Sandbox discovery failed with error: {e}"],
                    spec_version=round_n,
                    validation_status="blocked",
                    blocking_issues=[sandbox_skip_reason],
                )
            
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
        
        if user_intent and user_intent.strip():
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
        
        is_micro_task = "sandbox_file" in detected_domains
        spec = ground_intent_with_evidence(intent, evidence, is_micro_task=is_micro_task)
        
        # =================================================================
        # STEP 3.1: Multi-Target Read Handling (v1.34 - Level 2.5)
        # =================================================================
        
        if fs_evidence and hasattr(fs_evidence, 'is_multi_target') and fs_evidence.is_multi_target:
            logger.info(
                "[spec_runner] v1.34 MULTI-TARGET READ detected: %d targets",
                len(fs_evidence.target_files) if hasattr(fs_evidence, 'target_files') else 0
            )
            
            file_targets = []
            for fe in fs_evidence.target_files:
                target_info = fe.metadata.get("target_info", {}) if fe.metadata else {}
                file_target = FileTarget(
                    name=target_info.get("name", os.path.basename(fe.resolved_path or "")),
                    anchor=target_info.get("anchor"),
                    subfolder=target_info.get("subfolder"),
                    explicit_path=target_info.get("explicit_path"),
                    resolved_path=fe.resolved_path,
                    found=fe.exists and fe.readable,
                    content=fe.full_content if hasattr(fe, 'full_content') and fe.full_content else fe.content_preview,
                    error=None if fe.exists else f"Not found: {fe.original_reference}",
                )
                file_targets.append(file_target)
            
            spec.multi_target_files = file_targets
            spec.is_multi_target_read = True
            
            logger.info(
                "[spec_runner] v1.35 MULTI-TARGET REPLY: Checking if fs_evidence has multi_target_results attribute"
            )
            if hasattr(fs_evidence, 'multi_target_results'):
                logger.info(
                    "[spec_runner] v1.35 CALLING format_multi_target_reply for %d files",
                    len(file_targets)
                )
                multi_reply = await format_multi_target_reply(
                    fs_evidence,
                    provider_id=provider_id,
                    model_id=model_id,
                    llm_call_func=llm_call if _LLM_CALL_AVAILABLE else None,
                    user_request=combined_text,
                )
                logger.info(
                    "[spec_runner] v1.35 format_multi_target_reply returned: %s",
                    multi_reply[:200] if multi_reply else "(None)"
                )
                if multi_reply:
                    spec.sandbox_generated_reply = multi_reply
            else:
                logger.warning(
                    "[spec_runner] v1.35 fs_evidence does NOT have multi_target_results attribute! attrs=%s",
                    dir(fs_evidence)
                )
            
            multi_output_mode = detect_output_mode(combined_text)
            spec.sandbox_output_mode = multi_output_mode.value
            
            if multi_output_mode == OutputMode.SEPARATE_REPLY_FILE:
                first_valid = next((ft for ft in file_targets if ft.found and ft.resolved_path), None)
                if first_valid and first_valid.resolved_path:
                    folder_path = os.path.dirname(first_valid.resolved_path)
                    spec.sandbox_output_path = os.path.join(folder_path, "reply.txt")
                    spec.sandbox_folder_path = folder_path
                spec.constraints_from_repo.append(
                    f"Output mode: SEPARATE_REPLY_FILE (multi-target read: {len(file_targets)} files -> reply.txt)"
                )
                logger.info(
                    "[spec_runner] v1.39 Multi-target with SEPARATE_REPLY_FILE: output=%s",
                    spec.sandbox_output_path
                )
            else:
                spec.constraints_from_repo.append(
                    f"Output mode: {multi_output_mode.value.upper()} (multi-target read: {len(file_targets)} files)"
                )
            
            valid_count = sum(1 for ft in file_targets if ft.found)
            spec.what_exists.append(
                f"Multi-target read: {valid_count}/{len(file_targets)} files found"
            )
            
            mode_desc = {
                "overwrite_full": "overwrite file (destructive)",
                "append_in_place": "append in place",
                "rewrite_in_place": "rewrite in place (Q&A insertion)",
                "separate_reply_file": "write to reply.txt",
                "chat_only": "present in chat (no file modification)",
            }.get(spec.sandbox_output_mode, spec.sandbox_output_mode)
            spec.in_scope = [f"Read {valid_count} files → synthesize content → {mode_desc}"]
            
            logger.info(
                "[spec_runner] v1.39 Multi-target read populated: %d/%d files found, output_mode=%s",
                valid_count, len(file_targets), spec.sandbox_output_mode
            )
        
        # =================================================================
        # STEP 3.5: Populate sandbox resolution into spec
        # =================================================================
        
        if spec.is_multi_target_read:
            logger.info(
                "[spec_runner] v1.39 SKIPPING Step 3.5 (single-file sandbox resolution) - multi-target read already handled in Step 3.1"
            )
            spec.sandbox_discovery_used = True
            spec.sandbox_discovery_status = sandbox_discovery_status or "multi_target_read"
            if spec.multi_target_files:
                combined_excerpt_parts = []
                for ft in spec.multi_target_files:
                    if ft.found and ft.content:
                        name = ft.name or os.path.basename(ft.resolved_path or "")
                        preview = ft.content[:200] + "..." if len(ft.content) > 200 else ft.content
                        combined_excerpt_parts.append(f"--- {name} ---\n{preview}")
                if combined_excerpt_parts:
                    spec.sandbox_input_excerpt = "\n\n".join(combined_excerpt_parts)
                    logger.info(
                        "[spec_runner] v1.39 Built combined excerpt from %d files",
                        len(combined_excerpt_parts)
                    )
        elif sandbox_discovery_result and sandbox_discovery_result.get("selected_file"):
            selected = sandbox_discovery_result["selected_file"]
            folder_path = sandbox_discovery_result["path"]
            
            output_mode = detect_output_mode(combined_text)
            
            # v1.30: Q&A CONTEXT BIAS
            file_content = selected.get("content", "")
            if file_content and output_mode == OutputMode.CHAT_ONLY:
                qa_analysis = analyze_qa_file(file_content)
                
                if qa_analysis.get("is_qa_file") and qa_analysis.get("total_questions", 0) > 0:
                    text_lower = combined_text.lower()
                    answer_keywords = [
                        "answer the question", "answer these", "fill in", "fill the answer",
                        "answer where", "provide answer", "write the answer",
                        "complete the answer", "put the answer", "answer it", "answer them",
                    ]
                    has_answer_keywords = any(kw in text_lower for kw in answer_keywords)
                    
                    explicit_chat_only_signals = [
                        "do not modify", "don't modify", "don't change", "do not change",
                        "leave the file", "don't touch", "do not touch", "read only",
                        "read-only", "just tell me", "just show me", "don't write",
                        "do not write", "chat only",
                    ]
                    has_explicit_chat_only = any(sig in text_lower for sig in explicit_chat_only_signals)
                    
                    if has_answer_keywords and not has_explicit_chat_only:
                        logger.warning(
                            "[spec_runner] v1.30 AUTO-OVERRIDE: CHAT_ONLY -> REWRITE_IN_PLACE "
                            "(Q&A file detected with %d questions + answer keywords in user request)",
                            qa_analysis.get("total_questions", 0)
                        )
                        output_mode = OutputMode.REWRITE_IN_PLACE
            
            spec.sandbox_output_mode = output_mode.value
            
            if output_mode == OutputMode.OVERWRITE_FULL:
                output_path = selected["path"]
                spec.sandbox_insertion_format = None
            elif output_mode == OutputMode.REWRITE_IN_PLACE:
                output_path = selected["path"]
                spec.sandbox_insertion_format = "\n\nAnswer:\n{reply}\n"
            elif output_mode == OutputMode.APPEND_IN_PLACE:
                output_path = selected["path"]
                spec.sandbox_insertion_format = "\n\nAnswer:\n{reply}\n"
            elif output_mode == OutputMode.SEPARATE_REPLY_FILE:
                output_path = os.path.join(folder_path, "reply.txt")
                spec.sandbox_insertion_format = None
            else:
                output_path = None
                spec.sandbox_insertion_format = None
            
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
            
            if selected['content_type'].lower() != "unknown":
                spec.what_exists.append(f"Sandbox input: `{selected['path']}` ({selected['content_type']})")
            else:
                spec.what_exists.append(f"Sandbox input: `{selected['path']}`")
            spec.confirmed_components.append(GroundedFact(
                description=f"Selected sandbox file: {selected['name']}",
                source="sandbox_inspector",
                path=selected["path"],
                confidence="confirmed",
            ))
            
            if output_mode == OutputMode.OVERWRITE_FULL:
                spec.constraints_from_repo.append(f"Planned output mode: OVERWRITE_FULL (destructive write to `{output_path}`)")
            elif output_mode == OutputMode.REWRITE_IN_PLACE:
                spec.constraints_from_repo.append(f"Planned output mode: REWRITE_IN_PLACE (multi-question insert into `{output_path}`)")
            elif output_mode == OutputMode.APPEND_IN_PLACE:
                spec.constraints_from_repo.append(f"Planned output mode: APPEND_IN_PLACE (write into `{output_path}`)")
            elif output_mode == OutputMode.SEPARATE_REPLY_FILE:
                spec.constraints_from_repo.append(f"Planned output path (for later stages): `{output_path}`")
            else:
                spec.constraints_from_repo.append("Output mode: CHAT_ONLY (no file modification)")
            
            # v1.8/v1.24: Generate reply
            full_content = selected.get("content", "")
            if full_content:
                spec.sandbox_input_full_content = full_content
                
                if output_mode == OutputMode.OVERWRITE_FULL:
                    replacement_text = extract_replacement_text(combined_text)
                    if replacement_text:
                        spec.sandbox_generated_reply = replacement_text
                    else:
                        spec.sandbox_generated_reply = await generate_reply_from_content(
                            full_content, 
                            selected.get("content_type"),
                            provider_id=provider_id,
                            model_id=model_id,
                            llm_call_func=llm_call if _LLM_CALL_AVAILABLE else None,
                            output_mode="overwrite_full",
                        )
                else:
                    spec.sandbox_generated_reply = await generate_reply_from_content(
                        full_content, 
                        selected.get("content_type"),
                        provider_id=provider_id,
                        model_id=model_id,
                        llm_call_func=llm_call if _LLM_CALL_AVAILABLE else None,
                        output_mode=output_mode.value if output_mode else None,
                        user_request=combined_text,
                    )
                    logger.info(
                        "[spec_runner] v1.31 Generated LLM reply (user_request passed): %s",
                        spec.sandbox_generated_reply[:100] if spec.sandbox_generated_reply else "(empty)"
                    )
        
        spec.sandbox_discovery_status = sandbox_discovery_status
        spec.sandbox_skip_reason = sandbox_skip_reason
        
        if spec.sandbox_discovery_used and spec.sandbox_output_mode:
            mode_desc = {
                "overwrite_full": "overwrite file (destructive)",
                "append_in_place": "append in place",
                "rewrite_in_place": "rewrite in place (Q&A insertion)",
                "separate_reply_file": "write to reply.txt",
                "chat_only": "chat only (no file modification)",
            }.get(spec.sandbox_output_mode, spec.sandbox_output_mode)
            spec.in_scope = [f"Read file → generate reply → {mode_desc}"]
        
        # =================================================================
        # STEP 3.6: Detect Implementation Stack (v1.11)
        # =================================================================
        
        # NOTE: conversation_messages was already extracted earlier in STEP 1.7 (v2.6)
        # for vision context extraction. Using that same variable here.
        
        detected_stack = detect_implementation_stack(conversation_messages, weaver_job_text, intent)
        if detected_stack:
            spec.implementation_stack = detected_stack
            logger.info(
                "[spec_runner] v1.11 Detected implementation stack: %s/%s (locked=%s)",
                detected_stack.language,
                detected_stack.framework,
                detected_stack.stack_locked,
            )
            
            if detected_stack.stack_locked:
                lock_msg = f"⚠️ LOCKED STACK: {detected_stack.language}"
                if detected_stack.framework:
                    lock_msg += f" + {detected_stack.framework}"
                lock_msg += f" (source: {detected_stack.source})"
                spec.constraints_from_intent.append(lock_msg)
        
        # =================================================================
        # STEP 3.7: Extract SCAN_ONLY parameters (v1.19)
        # =================================================================
        
        if "scan_only" in detected_domains:
            scan_params = extract_scan_params(combined_text, intent)
            if scan_params:
                spec.scan_roots = scan_params.get("scan_roots", [])
                spec.scan_terms = scan_params.get("scan_terms", [])
                spec.scan_targets = scan_params.get("scan_targets", [])
                spec.scan_case_mode = scan_params.get("scan_case_mode", "case_insensitive")
                spec.scan_exclusions = scan_params.get("scan_exclusions", DEFAULT_SCAN_EXCLUSIONS)
                logger.info(
                    "[spec_runner] v1.19 SCAN_ONLY params: roots=%s, terms=%s",
                    spec.scan_roots, spec.scan_terms
                )
        
        # =================================================================
        # STEP 3.8: Multi-File Operation Population (v1.33)
        # =================================================================
        
        if multi_file_op and multi_file_op.is_multi_file:
            spec.multi_file = multi_file_op
            
            logger.info(
                "[spec_runner] v1.33 Populated spec.multi_file: type=%s, files=%d",
                multi_file_op.operation_type,
                multi_file_op.total_files,
            )
            
            if multi_file_op.total_files > 0:
                spec.confirmed_components.append(GroundedFact(
                    description=f"Multi-file {multi_file_op.operation_type}: {multi_file_op.total_files} files, {multi_file_op.total_occurrences} occurrences",
                    source="file_discovery",
                    confidence="confirmed",
                ))
            
            if multi_file_op.operation_type == "refactor":
                spec.constraints_from_repo.append(
                    f"⚠️ MULTI-FILE REFACTOR: {multi_file_op.total_files} files will be modified "
                    f"(pattern: `{multi_file_op.search_pattern}` → `{multi_file_op.replacement_pattern}`)"
                )
            
            if multi_file_op.operation_type == "search":
                spec.sandbox_output_mode = "chat_only"
                spec.constraints_from_repo.append(
                    f"Output mode: CHAT_ONLY (multi-file search results - {multi_file_op.total_occurrences} occurrences)"
                )
            
            if multi_file_op.operation_type == "search" and multi_file_op.file_preview:
                spec.sandbox_generated_reply = spec.get_multi_file_summary()
            
            if multi_file_op.operation_type == "search":
                spec.in_scope = [f"Search codebase for `{multi_file_op.search_pattern}` and report results"]
            elif multi_file_op.operation_type == "refactor":
                spec.in_scope = [
                    f"Replace `{multi_file_op.search_pattern}` with `{multi_file_op.replacement_pattern}` "
                    f"in {multi_file_op.total_files} files"
                ]
            
            if multi_file_op.error_message:
                spec.blocking_issues.append(f"Multi-file discovery error: {multi_file_op.error_message}")
        
        # =================================================================
        # STEP 4: Apply User Answers (if round 2+)
        # =================================================================
        
        if user_answers and round_n >= 2:
            for key, answer in user_answers.items():
                key_lower = key.lower()
                answer_lower = answer.lower() if answer else ""
                
                if "platform" in key_lower or "android" in answer_lower or "ios" in answer_lower:
                    spec.decisions["platform_v1"] = answer
                elif "input" in key_lower or "voice" in answer_lower or "screenshot" in answer_lower:
                    spec.decisions["input_mode_v1"] = answer
                elif "ocr" in key_lower or "completed parcels" in answer_lower:
                    spec.decisions["ocr_scope_v1"] = answer
                elif "sync" in key_lower:
                    if "target" in key_lower or "endpoint" in answer_lower:
                        spec.decisions["sync_target"] = answer
                    else:
                        spec.decisions["sync_behaviour"] = answer
                elif "scope" in key_lower:
                    spec.out_of_scope.append(answer)
                elif "step" in key_lower:
                    spec.proposed_steps.append(answer)
                elif "path" in key_lower or "file" in key_lower:
                    spec.what_exists.append(f"User confirmed: {answer}")
            
            logger.info(
                "[spec_runner] v1.5: Parsed user_answers into decisions: %s",
                spec.decisions
            )
        
        # =================================================================
        # STEP 5: Generate Questions (if needed)
        # =================================================================
        
        questions = generate_grounded_questions(spec, intent, evidence, round_n)
        spec.open_questions = questions
        
        # =================================================================
        # STEP 6: Determine Completion Status
        # =================================================================
        
        is_complete_enough, completion_reason = _is_spec_complete_enough(spec, intent, questions)
        
        logger.info(
            "[spec_runner] v1.4 Completion check: complete_enough=%s, reason='%s'",
            is_complete_enough, completion_reason
        )
        
        if round_n >= 3:
            spec.is_complete = True
            if questions:
                spec.blocking_issues.append(
                    f"Finalized with {len(questions)} unanswered question(s) - NOT guessed"
                )
        elif is_complete_enough:
            spec.is_complete = True
            if not spec.proposed_steps:
                spec.proposed_steps = _derive_steps_from_domain(intent, spec)
            if not spec.acceptance_tests or all('(To be determined)' in str(t) for t in spec.acceptance_tests):
                spec.acceptance_tests = _derive_tests_from_domain(intent, spec)
            logger.info("[spec_runner] v1.4 EARLY EXIT: %s (round %d)", completion_reason, round_n)
        else:
            has_real_steps = bool(spec.proposed_steps)
            has_real_tests = (
                bool(spec.acceptance_tests) and
                not all('(To be determined)' in str(t) for t in spec.acceptance_tests)
            )
            spec.is_complete = (len(questions) == 0 and has_real_steps and has_real_tests)
        
        # =================================================================
        # STEP 7: Generate IDs and Hash
        # =================================================================
        
        spec.spec_id = f"sg-{uuid.uuid4().hex[:12]}"
        spec.spec_version = round_n
        
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
        # STEP 9: Return Result
        # =================================================================
        
        validation_status = "validated" if spec.is_complete else "needs_clarification"
        if spec.blocking_issues:
            validation_status = "validated_with_issues" if spec.is_complete else "blocked"
        
        open_q_text = [q.question for q in spec.open_questions]
        
        # Job classification
        job_kind, job_kind_confidence, job_kind_reason = classify_job_kind(spec, intent)
        
        logger.info(
            "[spec_runner] v1.9 Job classification: kind=%s, confidence=%.2f",
            job_kind, job_kind_confidence
        )
        
        # Scan params extraction for scan_only jobs
        if job_kind == "scan_only":
            scan_params = extract_scan_params(combined_text, intent)
            if scan_params:
                spec.scan_roots = scan_params.get("scan_roots", [])
                spec.scan_terms = scan_params.get("scan_terms", [])
                spec.scan_targets = scan_params.get("scan_targets", [])
                spec.scan_case_mode = scan_params.get("scan_case_mode", "case_insensitive")
                spec.scan_exclusions = scan_params.get("scan_exclusions", [])
                
                spec.in_scope = [f"Scan {', '.join(spec.scan_roots)} for {', '.join(spec.scan_terms) if spec.scan_terms else 'specified patterns'}"]
                spec.constraints_from_intent.append("Output mode: CHAT_ONLY (read-only scan)")
                spec.constraints_from_intent.append(f"Write policy: READ_ONLY (scan operation)")
        
        # Build grounding_data
        grounding_data = {
            "job_kind": job_kind,
            "job_kind_confidence": job_kind_confidence,
            "job_kind_reason": job_kind_reason,
            "is_multi_target_read": spec.is_multi_target_read,
            "multi_target_files": [
                {
                    "name": ft.name,
                    "anchor": ft.anchor,
                    "subfolder": ft.subfolder,
                    "resolved_path": ft.resolved_path,
                    "found": ft.found,
                    "error": ft.error,
                    "content": ft.content if hasattr(ft, 'content') else "",
                    "path": ft.resolved_path,
                }
                for ft in (spec.multi_target_files or [])
            ],
            "multi_file": (
                multi_file_op.to_dict() if multi_file_op and hasattr(multi_file_op, 'to_dict') else
                {
                    "is_multi_file": multi_file_op.is_multi_file,
                    "operation_type": multi_file_op.operation_type,
                    "search_pattern": multi_file_op.search_pattern,
                    "replacement_pattern": multi_file_op.replacement_pattern,
                    "target_files": multi_file_op.target_files,
                    "total_files": multi_file_op.total_files,
                    "total_occurrences": multi_file_op.total_occurrences,
                    "file_preview": multi_file_op.file_preview,
                    "requires_confirmation": multi_file_op.requires_confirmation,
                    "confirmed": multi_file_op.confirmed,
                    "error_message": multi_file_op.error_message,
                } if multi_file_op else None
            ),
            "filesystem_evidence": {
                "task_type": fs_evidence.task_type if fs_evidence else None,
                "target_files_count": len(fs_evidence.target_files) if fs_evidence else 0,
                "has_valid_targets": fs_evidence.has_valid_targets() if fs_evidence else False,
                "primary_target": (
                    fs_evidence.get_primary_target().resolved_path
                    if fs_evidence and fs_evidence.get_primary_target()
                    else None
                ),
                "validation_errors": fs_evidence.validation_errors if fs_evidence else [],
                "warnings": fs_evidence.warnings if fs_evidence else [],
                "ground_truth_timestamp": fs_evidence.ground_truth_timestamp if fs_evidence else None,
            } if fs_evidence else None,
            "sandbox_input_path": spec.sandbox_input_path,
            "sandbox_output_path": spec.sandbox_output_path,
            "sandbox_generated_reply": spec.sandbox_generated_reply,
            "sandbox_discovery_used": spec.sandbox_discovery_used,
            "sandbox_input_excerpt": spec.sandbox_input_excerpt,
            "sandbox_selected_type": spec.sandbox_selected_type,
            "sandbox_folder_path": spec.sandbox_folder_path,
            "sandbox_discovery_status": spec.sandbox_discovery_status,
            "sandbox_output_mode": spec.sandbox_output_mode,
            "sandbox_insertion_format": spec.sandbox_insertion_format,
            "scan_roots": spec.scan_roots,
            "scan_terms": spec.scan_terms,
            "scan_targets": spec.scan_targets,
            "scan_case_mode": spec.scan_case_mode,
            "scan_exclusions": spec.scan_exclusions,
            "output_mode": "chat_only" if job_kind == "scan_only" else (spec.sandbox_output_mode or None),
            "write_policy": (
                "read_only" if job_kind == "scan_only" else
                "overwrite" if spec.sandbox_output_mode == "overwrite_full" else
                "append" if spec.sandbox_output_mode in ("append_in_place", "rewrite_in_place", "separate_reply_file") else
                None
            ),
            "implementation_stack": spec.implementation_stack.dict() if spec.implementation_stack else None,
            "goal": spec.goal,
            "what_exists": spec.what_exists,
            "what_missing": spec.what_missing,
            "constraints_from_repo": spec.constraints_from_repo,
            "constraints_from_intent": spec.constraints_from_intent,
            "proposed_steps": spec.proposed_steps,
            "acceptance_tests": spec.acceptance_tests,
        }
        
        logger.info(
            "[spec_runner] Result: complete=%s, questions=%d, round=%d",
            spec.is_complete, len(open_q_text), round_n
        )
        
        return SpecGateResult(
            ready_for_pipeline=spec.is_complete,
            open_questions=open_q_text,
            spot_markdown=spot_md if spec.is_complete else None,
            db_persisted=False,
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
            grounding_data=grounding_data,
        )
        
    except Exception as e:
        logger.exception("[spec_runner] HARD STOP: %s", e)
        return SpecGateResult(
            ready_for_pipeline=False,
            hard_stopped=True,
            hard_stop_reason=str(e),
            spec_version=int(spec_version) if isinstance(spec_version, int) else None,
            validation_status="error",
        )
