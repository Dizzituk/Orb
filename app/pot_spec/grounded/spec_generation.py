# FILE: app/pot_spec/grounded/spec_generation.py
"""
SpecGate Spec Generation Module (v1.35)

Core specification generation logic including:
- POT spec markdown builder
- Step/test derivation from domains
- Intent parsing and grounding
- Main async entry point
- v1.35: Multi-target file read - CRITICAL FIX for reply generation

v1.35 (2026-01-29): CRITICAL FIX - Multi-target reply generation (Level 2.5)
    - Fixed format_multi_target_reply() call to properly await async function
    - Now passes provider_id, model_id, llm_call_func parameters
    - Generates LLM replies for EACH file, not just first file
    - Previous version returned coroutine object instead of actual replies
v1.34 (2026-01-29): Multi-target file read support (Level 2.5)
    - Added import for FileTarget from spec_models
    - Added imports for extract_file_targets and is_multi_target_request from sandbox_discovery
    - Added import for format_multi_target_reply from evidence_gathering
    - Added STEP 1.6.5: Multi-target read detection and handling in run_spec_gate_grounded()
    - Convert FileEvidence to FileTarget dataclass instances
    - Populate spec.multi_target_files and spec.is_multi_target_read
    - Generate combined reply using format_multi_target_reply()
    - Added multi-target section to POT spec markdown with file status and combined content
v1.33 (2026-01-28): Multi-file operations wiring (Phase 4)
    - Fixed async/sync mismatch: _build_multi_file_operation() now async with await
    - Added STEP 1.7: Multi-file operation detection in run_spec_gate_grounded()
    - Added STEP 3.8: Multi-file spec population with confirmation flow
    - Added multi-file section to POT spec markdown output
    - Added multi_file to grounding_data for downstream stages
    - Early return for refactor operations requiring confirmation
v1.32 (2026-01-28): Multi-file operations (Level 3)
    - Added _detect_multi_file_intent() for multi-file pattern detection
    - Added _build_multi_file_operation() for file discovery integration
    - Integrated multi-file operation into run_spec_gate_grounded()
    - Added multi-file section to POT spec markdown output
v1.31 (2026-01-28): Fixed Q&A deep analysis and REWRITE_IN_PLACE consistency
    - Pass user_request to generate_reply_from_content() for deep analysis
    - Added REWRITE_IN_PLACE branch to _derive_steps_from_domain()
    - Added REWRITE_IN_PLACE branch to _derive_tests_from_domain()
v1.30 (2026-01-28): Q&A context bias for output mode detection
    - Added soft bias toward REWRITE_IN_PLACE when Q&A file + answer keywords detected
    - Automatic override from CHAT_ONLY with warning log for traceability
    - Explicit CHAT_ONLY signals still win (e.g., "do not modify the file")
v1.29 (2026-01-27): Fixed REWRITE_IN_PLACE constraint sync bug
    - Added missing REWRITE_IN_PLACE branch in constraints_from_repo logic
    - Now header and constraints show same output mode
v1.28 (2026-01-27): Fixed LLM import path
    - Changed from non-existent app.llm.llm_service to app.providers.registry
    - Resolves [SPECGATE_LLM_UNAVAILABLE] error for Q&A processing
v1.27 (2026-01-27): Pass llm_call_func to generate_reply_from_content
    - Now properly calls LLM for Q&A file processing
    - Added output_mode parameter to reply generation
v1.25: Evidence-First architecture - filesystem evidence gathered BEFORE LLM call

Version history preserved from spec_gate_grounded.py:
- v1.0 (2026-01): Initial Contract v1 implementation
- v1.1 (2026-01): Fixed question generation + status logic
- v1.2 (2026-01): Decision forks replace lazy questions
- v1.4 (2026-01): Question discipline upgrade (blocker-only gating)
- v1.5 (2026-01): Decision tracking + conditional steps/tests
- v1.7 (2026-01): Read-only reply output fix
- v1.25 (2026-01): Evidence-First architecture
"""

from __future__ import annotations

import logging
import os
import re
import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# =============================================================================
# v1.35 BUILD VERIFICATION
# =============================================================================
SPEC_GENERATION_BUILD_ID = "2026-01-30-v1.37-system-wide-scan"
print(f"[SPEC_GENERATION_LOADED] BUILD_ID={SPEC_GENERATION_BUILD_ID}")
logger.info(f"[spec_generation] Module loaded: BUILD_ID={SPEC_GENERATION_BUILD_ID}")

# =============================================================================
# IMPORTS FROM SIBLING MODULES
# =============================================================================

from .spec_models import (
    QuestionCategory,
    GroundedFact,
    GroundedAssumption,
    GroundedQuestion,
    MultiFileOperation,
    FileTarget,
    GroundedPOTSpec,
    MIN_QUESTIONS,
    MAX_QUESTIONS,
)

from .domain_detection import (
    DOMAIN_KEYWORDS,
    detect_domains,
    extract_decision_forks,
)

from .job_classification import (
    classify_job_kind,
    classify_job_size,
    EVIDENCE_CONFIG,
)

from .scan_operations import (
    DEFAULT_SCAN_EXCLUSIONS,
    extract_scan_params,
)

from .sandbox_discovery import (
    OutputMode,
    extract_sandbox_hints,
    detect_output_mode,
    extract_replacement_text,
    extract_file_targets,
    is_multi_target_request,
    is_system_wide_scan_request,  # v1.37: System-wide scan detection
)

from .tech_stack_detection import (
    detect_implementation_stack,
)

from .qa_processing import (
    generate_reply_from_content,
    analyze_qa_file,  # v1.30: For Q&A context bias
)

from .evidence_gathering import (
    gather_filesystem_evidence,
    format_evidence_for_prompt,
    format_multi_target_reply,
    sandbox_read_file,
)

# v1.32: File discovery for multi-file operations
from .file_discovery import (
    discover_files,
    discover_files_by_extension,
    DiscoveryResult,
    DEFAULT_ROOTS as DISCOVERY_DEFAULT_ROOTS,
)

# v1.32: Sandbox client for multi-file discovery
try:
    from app.overwatcher.sandbox_client import get_sandbox_client
    _SANDBOX_CLIENT_AVAILABLE = True
except ImportError as e:
    logger.warning("[spec_generation] sandbox_client not available: %s", e)
    _SANDBOX_CLIENT_AVAILABLE = False
    get_sandbox_client = None

# =============================================================================
# EXTERNAL IMPORTS (with fallbacks)
# =============================================================================

# Evidence collector
try:
    from ..evidence_collector import (
        EvidenceBundle,
        EvidenceSource,
        load_evidence,
        find_in_evidence,
        verify_path_exists,
        WRITE_REFUSED_ERROR,
    )
    _EVIDENCE_AVAILABLE = True
except ImportError as e:
    logger.warning("[spec_generation] evidence_collector not available: %s", e)
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

# ImplementationStack schema
try:
    from ..schemas import ImplementationStack
    _IMPL_STACK_AVAILABLE = True
except ImportError:
    _IMPL_STACK_AVAILABLE = False
    ImplementationStack = None

# LLM call function for Q&A processing (v1.28: Fixed import path)
try:
    from app.providers.registry import llm_call
    _LLM_CALL_AVAILABLE = True
except ImportError as e:
    logger.warning("[spec_generation] llm_call not available: %s", e)
    _LLM_CALL_AVAILABLE = False
    llm_call = None

# Sandbox inspection
try:
    from app.llm.local_tools.zobie.sandbox_inspector import (
        run_sandbox_discovery_chain,
        file_exists_in_sandbox,
        read_sandbox_file,
        SANDBOX_ROOTS,
    )
    _SANDBOX_INSPECTOR_AVAILABLE = True
except ImportError:
    _SANDBOX_INSPECTOR_AVAILABLE = False
    run_sandbox_discovery_chain = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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
# MULTI-FILE OPERATION HELPERS (v1.32 - Level 3)
# =============================================================================

def _detect_multi_file_intent(user_intent: str, constraints_hint: Optional[Dict]) -> Optional[Dict[str, Any]]:
    """
    Detect multi-file operation intent from user input or constraints.
    
    v1.32: Level 3 multi-file operations support.
    
    Args:
        user_intent: User's raw intent text
        constraints_hint: Weaver output and other hints (may contain multi_file_metadata)
    
    Returns:
        Dict with keys: operation_type, search_pattern, replacement_pattern
        or None if not a multi-file intent
    """
    # Check if constraints_hint already has multi-file metadata (from tier0)
    if constraints_hint:
        multi_file_meta = constraints_hint.get("multi_file_metadata")
        if multi_file_meta and multi_file_meta.get("is_multi_file"):
            logger.info(
                "[spec_generation] v1.32 Multi-file metadata found in constraints_hint: %s",
                multi_file_meta
            )
            return multi_file_meta
    
    # Fallback: Detect from user_intent text patterns
    if not user_intent:
        return None
    
    text_lower = user_intent.lower().strip()
    
    # Multi-file search patterns
    search_patterns = [
        (r"^find\s+all\s+(.+?)(?:\s+(?:in\s+)?(?:the\s+)?(?:codebase|repo|project|files?))?$", "search"),
        (r"^list\s+(?:all\s+)?files?\s+(?:containing|with)\s+(.+)$", "search"),
        (r"^search\s+(?:the\s+)?codebase\s+for\s+(.+)$", "search"),
        (r"^count\s+(?:all\s+)?(?:occurrences?|instances?)\s+of\s+(.+)$", "search"),
    ]
    
    # Multi-file refactor patterns
    refactor_patterns = [
        (r"^replace\s+(.+?)\s+with\s+(.+?)(?:\s+(?:everywhere|in\s+all\s+files?))?$", "refactor"),
        (r"^change\s+(?:all\s+)?(.+?)\s+to\s+(.+?)(?:\s+(?:everywhere|in\s+all\s+files?))?$", "refactor"),
        (r"^rename\s+(.+?)\s+to\s+(.+?)(?:\s+(?:everywhere|across\s+the\s+codebase))?$", "refactor"),
        (r"^(?:remove|delete)\s+(?:all\s+)?(.+?)(?:\s+from\s+(?:the\s+)?(?:codebase|all\s+files?))?$", "remove"),
    ]
    
    # Check for scope keywords (required for ambiguous patterns)
    scope_keywords = ["all", "every", "everywhere", "codebase", "repo", "project"]
    has_scope = any(kw in text_lower for kw in scope_keywords)
    
    if not has_scope:
        return None  # Not clearly a multi-file operation
    
    # Try search patterns
    for pattern, op_type in search_patterns:
        match = re.match(pattern, text_lower)
        if match:
            search_pattern = match.group(1).strip()
            logger.info(
                "[spec_generation] v1.32 Detected multi-file SEARCH from intent: pattern=%s",
                search_pattern
            )
            return {
                "is_multi_file": True,
                "operation_type": "search",
                "search_pattern": search_pattern,
                "replacement_pattern": "",
            }
    
    # Try refactor patterns
    for pattern, op_type in refactor_patterns:
        match = re.match(pattern, text_lower)
        if match:
            groups = match.groups()
            search_pattern = groups[0].strip() if len(groups) > 0 else ""
            replacement_pattern = groups[1].strip() if len(groups) > 1 and op_type != "remove" else ""
            
            logger.info(
                "[spec_generation] v1.32 Detected multi-file REFACTOR from intent: "
                "search=%s, replace=%s",
                search_pattern, replacement_pattern
            )
            return {
                "is_multi_file": True,
                "operation_type": "refactor",
                "search_pattern": search_pattern,
                "replacement_pattern": replacement_pattern,
            }
    
    return None


async def _build_multi_file_operation(
    operation_type: str,
    search_pattern: str,
    replacement_pattern: str = "",
    file_filter: Optional[str] = None,
    sandbox_client: Optional[Any] = None,
) -> MultiFileOperation:
    """
    Run file discovery and build MultiFileOperation for spec.
    
    v1.32: Level 3 multi-file operations support.
    v1.33: Fixed async/sync mismatch - now properly awaits discover_files().
    
    Args:
        operation_type: "search" or "refactor"
        search_pattern: Pattern to find
        replacement_pattern: Replacement text (empty for search/remove)
        file_filter: Optional extension filter (e.g., "*.py")
        sandbox_client: Optional pre-created sandbox client
    
    Returns:
        MultiFileOperation with discovery results (or error_message if failed)
    """
    logger.info(
        "[spec_generation] v1.33 Building multi-file operation: type=%s, pattern=%s",
        operation_type, search_pattern
    )
    
    # Check if sandbox client is available
    if not _SANDBOX_CLIENT_AVAILABLE or not get_sandbox_client:
        logger.warning("[spec_generation] v1.33 Sandbox client not available for multi-file discovery")
        return MultiFileOperation(
            is_multi_file=True,
            operation_type=operation_type,
            search_pattern=search_pattern,
            replacement_pattern=replacement_pattern,
            requires_confirmation=(operation_type == "refactor"),
            error_message="Sandbox client not available for file discovery",
        )
    
    try:
        # Get sandbox client (use provided or create new)
        client = sandbox_client or get_sandbox_client()
        
        # Run discovery (v1.33: properly await async call)
        result = await discover_files(
            search_pattern=search_pattern,
            sandbox_client=client,
            file_filter=file_filter,
            roots=DISCOVERY_DEFAULT_ROOTS,
        )
        
        # Build MultiFileOperation from result
        return MultiFileOperation(
            is_multi_file=True,
            operation_type=operation_type,
            search_pattern=search_pattern,
            replacement_pattern=replacement_pattern,
            target_files=[fm.path for fm in result.files],
            total_files=result.total_files,
            total_occurrences=result.total_occurrences,
            file_filter=file_filter,
            file_preview=result.get_file_preview(),
            discovery_truncated=result.truncated,
            discovery_duration_ms=result.duration_ms,
            roots_searched=result.roots_searched,
            requires_confirmation=(operation_type == "refactor"),
            confirmed=False,
            error_message=result.error_message if not result.success else None,
        )
        
    except Exception as e:
        logger.error("[spec_generation] v1.33 Multi-file discovery failed: %s", e)
        return MultiFileOperation(
            is_multi_file=True,
            operation_type=operation_type,
            search_pattern=search_pattern,
            replacement_pattern=replacement_pattern,
            requires_confirmation=(operation_type == "refactor"),
            error_message=f"Discovery failed: {str(e)}",
        )


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
        logger.warning("[spec_generation] parse_weaver_intent: constraints_hint is empty/None")
        return {}
    
    result = {}
    
    logger.info(
        "[spec_generation] parse_weaver_intent: constraints_hint keys=%s",
        list(constraints_hint.keys())
    )
    
    # v3.0: Simple Weaver text
    job_desc_text = constraints_hint.get("weaver_job_description_text")
    if job_desc_text:
        result["raw_text"] = job_desc_text
        result["source"] = "weaver_simple"
        logger.info(
            "[spec_generation] parse_weaver_intent: set raw_text from weaver_job_description_text (%d chars)",
            len(job_desc_text)
        )
        
        # v1.12: Extract goal from "What is being built" section, not just first line
        lines = job_desc_text.strip().split("\n")
        goal_found = False
        
        # First, try to find "What is being built" section
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            if "what is being built" in line_lower:
                # Check if goal is on same line (after colon/dash)
                if ":" in line or "-" in line:
                    parts = re.split(r'[:\-]', line, 1)
                    if len(parts) > 1 and parts[1].strip():
                        result["goal"] = parts[1].strip()
                        goal_found = True
                        logger.info(
                            "[spec_generation] v1.12 Extracted goal from 'What is being built' line: %s",
                            result["goal"][:100]
                        )
                        break
                # Otherwise, goal might be on next line
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.lower().startswith(("intended", "unresolved", "questions", "-", "*")):
                        result["goal"] = next_line.lstrip("- ").strip()
                        goal_found = True
                        logger.info(
                            "[spec_generation] v1.12 Extracted goal from line after 'What is being built': %s",
                            result["goal"][:100]
                        )
                        break
        
        # Fallback: first non-header line (old behavior)
        if not goal_found and lines:
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#") and not line.lower().startswith("what is being built"):
                    # Skip generic headers
                    if line.lower() not in ("job description", "job description from weaver"):
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
        logger.warning("[spec_generation] parse_weaver_intent: weaver_job_description_text not found in constraints_hint")
    
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
    evidence: "EvidenceBundle",
    is_micro_task: bool = False,
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
    
    # Track evidence completeness
    spec.evidence_complete = True
    spec.evidence_gaps = []
    
    # Check if codebase report was loaded
    has_codebase_report = False
    has_arch_map = False
    if evidence:
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
        if evidence and verify_path_exists:
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
    # Skip arch map inference for micro tasks (clutter reduction)
    if evidence and evidence.arch_map_content and not is_micro_task and find_in_evidence:
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
    if evidence and evidence.codebase_report_content and find_in_evidence:
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


# =============================================================================
# QUESTION GENERATOR (v1.2 - Decision Forks, Not Lazy Questions)
# =============================================================================

def generate_grounded_questions(
    spec: GroundedPOTSpec,
    intent: Dict[str, Any],
    evidence: "EvidenceBundle",
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
            "[spec_generation] v1.5: Detected domains %s, blocking questions=%d, assumptions=%d (round=%d)",
            detected_domains, len(fork_questions), len(fork_assumptions), round_number
        )
    
    # =================================================================
    # ROUND 2+: Derive steps/tests from fork answers, don't ask more
    # =================================================================
    if round_number >= 2:
        # Only ask critical questions if there's a genuine blocker
        if not spec.goal or spec.goal.strip() == "":
            questions.append(GroundedQuestion(
                question="What is the primary goal/objective of this job?",
                category=QuestionCategory.MISSING_PRODUCT_DECISION,
                why_it_matters="Without a clear goal, the spec cannot be grounded",
                evidence_found="No goal found in Weaver output",
            ))
        
        # Derive steps/tests if missing
        if not spec.proposed_steps:
            spec.proposed_steps = _derive_steps_from_domain(intent, spec)
        if not spec.acceptance_tests or all('(To be determined)' in str(t) for t in spec.acceptance_tests):
            spec.acceptance_tests = _derive_tests_from_domain(intent, spec)
        
        return questions[:MAX_QUESTIONS]
    
    # =================================================================
    # ROUND 1: Ask bounded decision forks (A/B/C) - NOT lazy questions
    # =================================================================
    
    # Check for missing goal (this is a critical blocker)
    if not spec.goal or spec.goal.strip() == "":
        questions.append(GroundedQuestion(
            question="What is the primary goal/objective of this job?",
            category=QuestionCategory.MISSING_PRODUCT_DECISION,
            why_it_matters="Without a clear goal, the spec cannot be grounded",
            evidence_found="No goal found in Weaver output",
        ))
    
    # v1.5: Fork questions were already extracted above
    if fork_questions:
        questions.extend(fork_questions)
    
    # Check for ambiguous paths
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
    
    return questions[:MAX_QUESTIONS]


# =============================================================================
# STEP/TEST DERIVATION (v1.5 - Conditional on Decisions)
# =============================================================================

def _derive_steps_from_domain(intent: Dict[str, Any], spec: GroundedPOTSpec) -> List[str]:
    """
    v1.5: Derive implementation steps from domain + resolved decisions + assumptions.
    
    This is SpecGate's job, NOT the user's. Once product decisions are made,
    the steps can be derived automatically.
    """
    weaver_text = intent.get("raw_text", "") or ""
    detected_domains = detect_domains(weaver_text)
    
    logger.info(
        "[spec_generation] _derive_steps_from_domain: raw_text_len=%d, detected_domains=%s",
        len(weaver_text), detected_domains
    )
    
    # Build lookup of resolved values: decisions override assumptions
    resolved = {}
    for assumption in spec.assumptions:
        resolved[assumption.topic] = assumption.assumed_value
    for key, value in spec.decisions.items():
        resolved[key] = value
    
    steps = []
    
    # v1.6: Sandbox file domain - specific steps for file discovery/read/reply tasks
    if "sandbox_file" in detected_domains:
        if spec.sandbox_discovery_used and spec.sandbox_input_path:
            steps = [
                f"Read input file from sandbox: `{spec.sandbox_input_path}`",
                "Parse and understand the question/content in the file",
                "Generate reply based on file content (included in SPoT output)",
            ]
            output_mode = spec.sandbox_output_mode
            # v1.31: Added REWRITE_IN_PLACE branch
            if output_mode == OutputMode.REWRITE_IN_PLACE.value:
                steps.append("Insert answers under each question in-place")
                steps.append(f"Verify file updated: `{spec.sandbox_input_path}`")
            elif output_mode == OutputMode.APPEND_IN_PLACE.value:
                steps.append("Append reply beneath question in same file")
                steps.append(f"Verify file updated: `{spec.sandbox_input_path}`")
            elif output_mode == OutputMode.SEPARATE_REPLY_FILE.value:
                steps.append(f"Write reply to: `{spec.sandbox_output_path}`")
                steps.append("Verify reply.txt file exists")
            else:  # CHAT_ONLY or None
                steps.append("[Chat only - no file modification]")
        else:
            steps = [
                "Discover target file in sandbox (Desktop or Documents)",
                "Read and parse the file content",
                "Generate reply based on file content (included in SPoT output)",
                "[For later stages] Output based on detected mode",
            ]
        return steps
    
    # v1.12: GAME DOMAIN - Takes priority over mobile_app
    if "game" in detected_domains:
        logger.info("[spec_generation] v1.12 GAME domain detected - using game steps")
        steps = [
            "Analyze game requirements and create technical design",
            "Set up project structure (HTML/CSS/JS or chosen framework)",
            "Implement game board/playfield rendering",
            "Implement game piece/entity logic",
            "Implement player input handling (keyboard/touch controls)",
            "Implement core game mechanics (movement, collision, scoring)",
            "Add game state management (start, pause, game over)",
            "Implement scoring and level progression",
            "Add visual polish (animations, transitions)",
            "Testing and bug fixes",
        ]
        return steps
    
    if "mobile_app" in detected_domains:
        # Mobile app domain - conditional implementation steps
        platform = resolved.get("platform_v1", "")
        if "android" in platform.lower():
            steps.append("Set up Android project (Android Studio, Gradle)")
        elif "ios" in platform.lower() or "both" in platform.lower():
            steps.append("Set up mobile project structure (Android + iOS)")
        else:
            steps.append("Set up mobile project structure")
        
        steps.append("Implement local encrypted data storage layer")
        steps.append("Build core UI screens (shift start/stop, daily summary)")
        
        input_mode = resolved.get("input_mode_v1", "")
        if "voice" in input_mode.lower() and "manual" in input_mode.lower():
            steps.append("Implement push-to-talk voice input with manual fallback")
        elif "voice" in input_mode.lower():
            steps.append("Implement voice input handler")
        elif "manual" in input_mode.lower() or "screenshot" in input_mode.lower():
            steps.append("Implement manual input forms (screenshot import + manual entry)")
        
        ocr_scope = resolved.get("ocr_scope_v1", "")
        if ocr_scope:
            if "finish tour" in ocr_scope.lower() or "completed parcels" in ocr_scope.lower():
                steps.append("Implement screenshot OCR parser for Finish Tour screen (Successfully Completed Parcels)")
            elif "multiple" in ocr_scope.lower():
                steps.append("Implement screenshot OCR parser for multiple screen formats")
        
        steps.append("Implement pay/cost/net calculations (parcel rate, fuel, wear & tear)")
        steps.append("Implement end-of-week summary calculations")
        
        sync_behaviour = resolved.get("sync_behaviour", "")
        sync_target = resolved.get("sync_target", "")
        if "local-only" not in sync_behaviour.lower() and "local only" not in sync_behaviour.lower():
            if "export" in sync_target.lower() or "file" in sync_target.lower():
                steps.append("Add export functionality for ASTRA integration (file-based)")
            elif "endpoint" in sync_target.lower() or "live" in sync_target.lower():
                steps.append("Build sync mechanism and ASTRA integration endpoint")
        
        steps.append("Integration testing")
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
    
    # v1.6: Sandbox file domain
    if "sandbox_file" in detected_domains:
        if spec.sandbox_discovery_used and spec.sandbox_input_path:
            tests = [
                f"Input file `{spec.sandbox_input_path}` was found and read successfully",
                "File content was correctly parsed and understood",
                "Reply was generated based on file content",
            ]
            if spec.sandbox_input_excerpt and spec.sandbox_selected_type and spec.sandbox_selected_type.lower() != "unknown":
                tests.insert(1, f"Input content type identified: {spec.sandbox_selected_type}")
            output_mode = spec.sandbox_output_mode
            # v1.31: Added REWRITE_IN_PLACE branch
            if output_mode == OutputMode.REWRITE_IN_PLACE.value:
                tests.append(f"Answers inserted under each question in `{spec.sandbox_input_path}`")
                tests.append("File contains both questions and answers in correct positions")
            elif output_mode == OutputMode.APPEND_IN_PLACE.value:
                tests.append(f"Reply appended to `{spec.sandbox_input_path}` beneath original question")
                tests.append("File contains both question and answer")
            elif output_mode == OutputMode.SEPARATE_REPLY_FILE.value:
                tests.append(f"Reply written to `{spec.sandbox_output_path}`")
                tests.append("reply.txt file exists and contains expected content")
            else:  # CHAT_ONLY or None
                tests.append("Reply presented in chat (no file modification)")
        else:
            tests = [
                "Target file was discovered in sandbox",
                "File content was read and parsed correctly",
                "Reply was generated based on file content",
                "Output delivered per detected mode (chat/file)",
            ]
        return tests
    
    # v1.12: GAME DOMAIN
    if "game" in detected_domains:
        logger.info("[spec_generation] v1.12 GAME domain detected - using game tests")
        tests = [
            "Game starts and displays initial state correctly",
            "Game board/playfield renders with correct dimensions",
            "Player input controls respond correctly (keyboard/touch)",
            "Game pieces/entities move and behave as expected",
            "Collision detection works correctly",
            "Scoring updates correctly on valid actions",
            "Game over condition triggers at correct time",
            "Level progression works (if applicable)",
            "Pause/resume functionality works",
            "Game is playable and fun to use",
        ]
        return tests
    
    if "mobile_app" in detected_domains:
        tests.append("App starts and displays main screen within 2 seconds")
        tests.append("Shift start/stop logs timestamp correctly to local storage")
        
        input_mode = resolved.get("input_mode_v1", "")
        if "voice" in input_mode.lower():
            tests.append("Voice input correctly transcribes test phrases")
        if "manual" in input_mode.lower() or "screenshot" in input_mode.lower():
            tests.append("Manual entry form accepts and validates input correctly")
        
        ocr_scope = resolved.get("ocr_scope_v1", "")
        if ocr_scope:
            if "finish tour" in ocr_scope.lower() or "completed parcels" in ocr_scope.lower():
                tests.append("Screenshot OCR extracts 'Successfully Completed Parcels' from test Finish Tour screenshot")
            elif "multiple" in ocr_scope.lower():
                tests.append("Screenshot OCR extracts parcel counts from multiple screen format test images")
        
        tests.append("Data persists across app restart (encrypted storage verified)")
        
        sync_behaviour = resolved.get("sync_behaviour", "")
        sync_target = resolved.get("sync_target", "")
        if "local-only" not in sync_behaviour.lower() and "local only" not in sync_behaviour.lower():
            if "export" in sync_target.lower() or "file" in sync_target.lower():
                tests.append("Export functionality produces valid file for ASTRA import")
            elif "endpoint" in sync_target.lower() or "live" in sync_target.lower():
                tests.append("Sync successfully transfers data to ASTRA endpoint")
        
        tests.append("Pay calculation correctly computes gross from parcel count (rate × parcels)")
        tests.append("Net profit calculation correctly subtracts fuel and wear & tear")
        tests.append("End-of-week summary shows correct totals for parcels and pay")
        tests.append("App functions fully offline (no network required for core features)")
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
# SPEC COMPLETENESS CHECK (v1.4 - Early Exit)
# =============================================================================

def _is_spec_complete_enough(
    spec: GroundedPOTSpec,
    intent: Dict[str, Any],
    blocking_questions: List[GroundedQuestion],
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
    
    if checks:
        return False, f"Missing: {', '.join(checks)}"
    
    return True, "Spec is complete enough - no blocking questions remain"


# =============================================================================
# POT SPEC TEMPLATE BUILDER
# =============================================================================

def build_pot_spec_markdown(spec: GroundedPOTSpec) -> str:
    """
    Build POT spec markdown in the required template format.
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
    
    # v1.19: SCAN_ONLY job parameters
    if spec.scan_roots or spec.scan_terms:
        lines.append("### 🔍 Scan Parameters (SCAN_ONLY Job)")
        lines.append("")
        if spec.scan_roots:
            lines.append(f"- **Scan roots:** {', '.join(f'`{r}`' for r in spec.scan_roots)}")
        if spec.scan_terms:
            lines.append(f"- **Search terms:** {', '.join(f'`{t}`' for t in spec.scan_terms)}")
        if spec.scan_targets:
            lines.append(f"- **Search targets:** {', '.join(spec.scan_targets)}")
        if spec.scan_case_mode:
            lines.append(f"- **Case mode:** {spec.scan_case_mode}")
        if spec.scan_exclusions:
            exclusions_display = spec.scan_exclusions[:5]
            if len(spec.scan_exclusions) > 5:
                exclusions_display.append(f"... and {len(spec.scan_exclusions) - 5} more")
            lines.append(f"- **Exclusions:** {', '.join(exclusions_display)}")
        lines.append("")
        lines.append("*Output: CHAT_ONLY (read-only scan, no file modification)*")
        lines.append("")
    
    # Sandbox Resolution
    if spec.sandbox_discovery_used and spec.sandbox_input_path:
        lines.append("### Sandbox File Resolution")
        lines.append(f"- **Input file:** `{spec.sandbox_input_path}`")
        
        output_mode = spec.sandbox_output_mode
        if output_mode == "rewrite_in_place":
            lines.append(f"- **Output mode:** REWRITE_IN_PLACE (multi-question insert)")
            lines.append(f"- **Output target:** `{spec.sandbox_input_path}`")
            if spec.sandbox_insertion_format:
                lines.append(f"- **Insertion format:** `{repr(spec.sandbox_insertion_format)}`")
        elif output_mode == "append_in_place":
            lines.append(f"- **Output mode:** APPEND_IN_PLACE (write into same file)")
            lines.append(f"- **Output target:** `{spec.sandbox_input_path}`")
            if spec.sandbox_insertion_format:
                lines.append(f"- **Insertion format:** `{repr(spec.sandbox_insertion_format)}`")
        elif output_mode == "separate_reply_file":
            lines.append(f"- **Output mode:** SEPARATE_REPLY_FILE")
            lines.append(f"- **Output target:** `{spec.sandbox_output_path}`")
        else:
            lines.append("- **Output mode:** CHAT_ONLY (no file modification)")
        
        if spec.sandbox_selected_type and spec.sandbox_selected_type.lower() != "unknown":
            lines.append(f"- **Content type:** {spec.sandbox_selected_type}")
        
        if spec.sandbox_input_excerpt:
            excerpt_lines = spec.sandbox_input_excerpt.split('\n')[:15]
            lines.append("")
            lines.append("**Input excerpt:**")
            lines.append("```")
            for el in excerpt_lines:
                lines.append(el)
            if len(spec.sandbox_input_excerpt.split('\n')) > 15:
                lines.append("... (truncated)")
            lines.append("```")
        lines.append("")
        
        if spec.sandbox_generated_reply:
            lines.append("### 📝 Reply (Read-Only)")
            lines.append("")
            lines.append("*This reply was generated by SpecGate based on the file content.*")
            if output_mode == "rewrite_in_place":
                lines.append("*Later stages will insert answers under each question in the input file.*")
            elif output_mode == "append_in_place":
                lines.append("*Later stages will append this reply to the input file.*")
            elif output_mode == "separate_reply_file":
                lines.append("*Later stages will write this to reply.txt.*")
            else:
                lines.append("*This reply will be shown in chat only (no file modification).*")
            lines.append("")
            lines.append("```")
            lines.append(spec.sandbox_generated_reply)
            lines.append("```")
            lines.append("")
    elif spec.sandbox_discovery_status and spec.sandbox_discovery_status != "not_attempted":
        lines.append("### ⚠️ Sandbox Discovery Status")
        lines.append(f"- **Status:** {spec.sandbox_discovery_status}")
        if spec.sandbox_skip_reason:
            lines.append(f"- **Reason:** {spec.sandbox_skip_reason}")
        lines.append("")
    
    # v1.34: Multi-Target Read Section (Level 2.5)
    if spec.is_multi_target_read and spec.multi_target_files:
        lines.append("### 📖 Multi-Target Read")
        lines.append("")
        
        valid_files = [ft for ft in spec.multi_target_files if ft.found]
        total_files = len(spec.multi_target_files)
        
        lines.append(f"- **Total targets:** {total_files}")
        lines.append(f"- **Files found:** {len(valid_files)}")
        
        if len(valid_files) < total_files:
            missing = [ft.name for ft in spec.multi_target_files if not ft.found]
            lines.append(f"- **Missing:** {', '.join(missing)}")
        
        lines.append("")
        lines.append("**Files:**")
        for ft in spec.multi_target_files:
            status = "✅" if ft.found else "❌"
            location = f" ({ft.anchor}" + (f"/{ft.subfolder}" if ft.subfolder else "") + ")" if ft.anchor else ""
            lines.append(f"- {status} `{ft.name}`{location}")
            if ft.resolved_path and ft.found:
                lines.append(f"  - Path: `{ft.resolved_path}`")
            if ft.error:
                lines.append(f"  - Error: {ft.error}")
        
        lines.append("")
        
        # Include combined content if available
        if spec.sandbox_generated_reply:
            lines.append("**Combined Content:**")
            lines.append("```")
            # Truncate if very long
            reply_lines = spec.sandbox_generated_reply.split('\n')[:100]
            for line in reply_lines:
                lines.append(line)
            if len(spec.sandbox_generated_reply.split('\n')) > 100:
                lines.append("... (truncated)")
            lines.append("```")
        
        lines.append("")
    
    # v1.33: Multi-File Operation Section
    if spec.multi_file and spec.multi_file.is_multi_file:
        lines.append("### 🔍 Multi-File Operation")
        mf = spec.multi_file
        
        lines.append(f"- **Type:** {mf.operation_type.upper()}")
        lines.append(f"- **Pattern:** `{mf.search_pattern}`")
        if mf.replacement_pattern:
            lines.append(f"- **Replace:** `{mf.replacement_pattern}`")
        lines.append(f"- **Files found:** {mf.total_files}")
        lines.append(f"- **Total occurrences:** {mf.total_occurrences}")
        
        if mf.roots_searched:
            lines.append(f"- **Roots searched:** {', '.join(mf.roots_searched[:3])}{'...' if len(mf.roots_searched) > 3 else ''}")
        
        if mf.discovery_duration_ms:
            lines.append(f"- **Discovery time:** {mf.discovery_duration_ms}ms")
        
        if mf.discovery_truncated:
            lines.append("- **Note:** Results truncated (too many matches)")
        
        if mf.requires_confirmation:
            status = "✅ Confirmed" if mf.confirmed else "⏳ Awaiting confirmation"
            lines.append(f"- **Confirmation:** {status}")
        
        if mf.error_message:
            lines.append(f"- **⚠️ Error:** {mf.error_message}")
        
        if mf.file_preview:
            lines.append("")
            lines.append("**File Preview:**")
            lines.append("```")
            # Truncate preview to avoid huge markdown
            preview_lines = mf.file_preview.split('\n')[:20]
            for line in preview_lines:
                lines.append(line)
            if len(mf.file_preview.split('\n')) > 20:
                lines.append("... (truncated)")
            lines.append("```")
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
    elif spec.sandbox_discovery_used and spec.sandbox_input_path:
        output_mode = spec.sandbox_output_mode
        if output_mode == "rewrite_in_place":
            lines.append("- Read file → parse questions → generate answers → insert under each question")
        elif output_mode == "append_in_place":
            lines.append("- Read file → generate reply → append in place")
        elif output_mode == "separate_reply_file":
            lines.append("- Read file → generate reply → write to reply.txt")
        else:
            lines.append("- Read file → generate reply → present in chat")
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
    
    # Open Questions
    lines.append("## Open Questions (Human Decisions Only)")
    lines.append("")
    if spec.open_questions:
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
        if spec.evidence_complete and not spec.evidence_gaps:
            lines.append("✅ No blocking questions - all information grounded from evidence.")
        else:
            lines.append("⚠️ No questions generated, but evidence was incomplete (see Evidence Gaps above).")
    lines.append("")
    
    # Resolved Decisions
    if spec.decisions:
        lines.append("## Resolved Decisions")
        lines.append("")
        lines.append("*These were explicitly answered by the user.*")
        lines.append("")
        for key, value in spec.decisions.items():
            nice_key = key.replace("_", " ").title()
            lines.append(f"- **{nice_key}:** {value}")
        lines.append("")
    
    # Implementation Stack
    if spec.implementation_stack:
        lines.append("## 🔧 Implementation Stack")
        lines.append("")
        if spec.implementation_stack.stack_locked:
            lines.append("⚠️ **STACK LOCKED** - Architecture MUST use this stack (user confirmed)")
        else:
            lines.append("*Stack detected from conversation - not explicitly locked*")
        lines.append("")
        if spec.implementation_stack.language:
            lines.append(f"- **Language:** {spec.implementation_stack.language}")
        if spec.implementation_stack.framework:
            lines.append(f"- **Framework:** {spec.implementation_stack.framework}")
        if spec.implementation_stack.runtime:
            lines.append(f"- **Runtime:** {spec.implementation_stack.runtime}")
        if spec.implementation_stack.source:
            lines.append(f"- **Source:** {spec.implementation_stack.source}")
        if spec.implementation_stack.notes:
            lines.append(f"- **Notes:** {spec.implementation_stack.notes}")
        lines.append("")
    
    # Assumptions
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
    
    # Unresolved Items Summary
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
    lines.append(f"- **Status:** {'Complete' if spec.is_complete else 'Awaiting answers'}")
    
    return "\n".join(lines)


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
            "[spec_generation] Starting round %d for job %s (project %d)",
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
            "[spec_generation] Loaded evidence: %d sources, %d errors",
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
            "[spec_generation] v1.6 Sandbox hint extraction: anchor=%s, subfolder=%s, text_len=%d",
            anchor, subfolder, len(combined_text)
        )
        
        # =================================================================
        # STEP 1.6: EVIDENCE-FIRST FILESYSTEM VALIDATION (v1.25)
        # =================================================================
        # Gather filesystem evidence BEFORE any LLM calls or sandbox discovery.
        # This grounds the spec in filesystem reality, not LLM guesses.
        
        fs_evidence = None
        fs_evidence_block = ""
        
        if anchor or "sandbox_file" in detect_domains(combined_text):
            logger.info(
                "[spec_generation] v1.25 EVIDENCE-FIRST: Gathering filesystem evidence..."
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
                    "[spec_generation] v1.25 Evidence gathered: %s",
                    fs_evidence.to_summary()
                )
                
                # Format evidence for any later LLM prompt injection
                fs_evidence_block = format_evidence_for_prompt(fs_evidence)
                
                # v1.25: If we have valid targets, we can short-circuit some discovery
                if fs_evidence.has_valid_targets():
                    primary = fs_evidence.get_primary_target()
                    logger.info(
                        "[spec_generation] v1.25 PRIMARY TARGET FOUND: %s (exists=%s, readable=%s)",
                        primary.resolved_path, primary.exists, primary.readable
                    )
                else:
                    if fs_evidence.validation_errors:
                        logger.warning(
                            "[spec_generation] v1.25 Evidence validation errors: %s",
                            fs_evidence.validation_errors
                        )
        
        # v1.25: SHORT-CIRCUIT - Use evidence to pre-populate sandbox discovery result
        # If we have a valid primary target, we can skip some discovery steps
        evidence_based_sandbox_result = None
        
        if fs_evidence and fs_evidence.has_valid_targets():
            primary = fs_evidence.get_primary_target()
            if primary and primary.resolved_path and primary.exists and primary.readable:
                logger.info(
                    "[spec_generation] v1.25 Using evidence-first path resolution: %s",
                    primary.resolved_path
                )
                # Build a sandbox_discovery_result-compatible dict from evidence
                evidence_based_sandbox_result = {
                    "selected_file": {
                        "path": primary.resolved_path,
                        "name": os.path.basename(primary.resolved_path),
                        "content_type": primary.detected_structure or "plain_text",
                        "content": primary.content_preview,
                        "confidence": 1.0,  # Evidence-first = high confidence
                    },
                    "path": os.path.dirname(primary.resolved_path),
                    "ambiguous": False,
                    "evidence_source": "v1.25_evidence_first",
                }
                
                # v1.26: Read full content via sandbox client (not local filesystem)
                if primary.resolved_path:
                    success, full_content = sandbox_read_file(primary.resolved_path, max_chars=50000)
                    if success and full_content:
                        evidence_based_sandbox_result["selected_file"]["content"] = full_content
                        logger.info(
                            "[spec_generation] v1.26 Read full content via sandbox: %d chars",
                            len(full_content)
                        )
                    else:
                        logger.warning(
                            "[spec_generation] v1.26 Could not read full content from sandbox: %s",
                            primary.resolved_path
                        )
        
        # v1.10: GREENFIELD BUILD CHECK
        detected_domains = detect_domains(combined_text)
        is_greenfield = "greenfield_build" in detected_domains
        
        if is_greenfield:
            logger.info("[spec_generation] v1.10 GREENFIELD BUILD detected - skipping sandbox discovery")
            anchor = None
            sandbox_skip_reason = "Greenfield build detected (CREATE_NEW job type)"
        
        # =================================================================
        # STEP 1.7: Multi-File Operation Detection (v1.33)
        # =================================================================
        # Detect multi-file intents (search/refactor across codebase) early.
        # This runs before sandbox discovery since multi-file ops are different.
        
        multi_file_op = None
        multi_file_meta = _detect_multi_file_intent(user_intent, constraints_hint)
        
        if multi_file_meta and multi_file_meta.get("is_multi_file"):
            logger.info(
                "[spec_generation] v1.33 MULTI-FILE OPERATION detected: type=%s, pattern=%s",
                multi_file_meta.get("operation_type"),
                multi_file_meta.get("search_pattern"),
            )
            
            # Build the MultiFileOperation (runs file discovery)
            multi_file_op = await _build_multi_file_operation(
                operation_type=multi_file_meta.get("operation_type", "search"),
                search_pattern=multi_file_meta.get("search_pattern", ""),
                replacement_pattern=multi_file_meta.get("replacement_pattern", ""),
                file_filter=multi_file_meta.get("file_filter"),
            )
            
            logger.info(
                "[spec_generation] v1.33 Multi-file discovery result: files=%d, occurrences=%d, error=%s",
                multi_file_op.total_files,
                multi_file_op.total_occurrences,
                multi_file_op.error_message,
            )
            
            # v1.33: Early return for refactor operations requiring confirmation
            if multi_file_op.operation_type == "refactor" and not multi_file_op.confirmed:
                # Build preview for confirmation question
                preview_text = multi_file_op.file_preview[:1000] if multi_file_op.file_preview else "(no preview available)"
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
                    "[spec_generation] v1.33 REFACTOR CONFIRMATION REQUIRED: %d files, %d occurrences",
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
            # v1.25: Use evidence-first result instead of discovery chain
            logger.info(
                "[spec_generation] v1.25 EVIDENCE-FIRST: Using pre-resolved sandbox result (skipping discovery chain)"
            )
            sandbox_discovery_result = evidence_based_sandbox_result
            sandbox_discovery_status = "evidence_first_success"
        else:
            logger.info("[spec_generation] Running sandbox discovery: anchor=%s, subfolder=%s", anchor, subfolder)
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
                logger.exception("[spec_generation] Sandbox discovery exception: %s", e)
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
        # If evidence gathering detected a multi-target read request,
        # populate spec with FileTarget objects and generate combined reply.
        
        if fs_evidence and hasattr(fs_evidence, 'is_multi_target') and fs_evidence.is_multi_target:
            logger.info(
                "[spec_generation] v1.34 MULTI-TARGET READ detected: %d targets",
                len(fs_evidence.target_files) if hasattr(fs_evidence, 'target_files') else 0
            )
            
            # Convert FileEvidence objects to FileTarget dataclass instances
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
            
            # Populate spec with multi-target data
            spec.multi_target_files = file_targets
            spec.is_multi_target_read = True
            
            # v1.35: Generate combined reply (FIXED - now awaits async function with parameters)
            logger.info(
                "[spec_generation] v1.35 MULTI-TARGET REPLY: Checking if fs_evidence has multi_target_results attribute"
            )
            if hasattr(fs_evidence, 'multi_target_results'):
                logger.info(
                    "[spec_generation] v1.35 CALLING format_multi_target_reply for %d files",
                    len(file_targets)
                )
                multi_reply = await format_multi_target_reply(
                    fs_evidence,
                    provider_id=provider_id,
                    model_id=model_id,
                    llm_call_func=llm_call if _LLM_CALL_AVAILABLE else None,
                    user_request=combined_text,  # v1.36: Pass user request for synthesis context
                )
                logger.info(
                    "[spec_generation] v1.35 format_multi_target_reply returned: %s",
                    multi_reply[:200] if multi_reply else "(None)"
                )
                if multi_reply:
                    spec.sandbox_generated_reply = multi_reply
            else:
                logger.warning(
                    "[spec_generation] v1.35 fs_evidence does NOT have multi_target_results attribute! attrs=%s",
                    dir(fs_evidence)
                )
            
            # Set output mode to CHAT_ONLY for multi-target reads
            spec.sandbox_output_mode = OutputMode.CHAT_ONLY.value
            spec.constraints_from_repo.append(
                f"Output mode: CHAT_ONLY (multi-target read: {len(file_targets)} files)"
            )
            
            # Add to what_exists
            valid_count = sum(1 for ft in file_targets if ft.found)
            spec.what_exists.append(
                f"Multi-target read: {valid_count}/{len(file_targets)} files found"
            )
            
            # Log success
            logger.info(
                "[spec_generation] v1.34 Multi-target read populated: %d/%d files found",
                valid_count, len(file_targets)
            )
        
        # =================================================================
        # STEP 3.5: Populate sandbox resolution into spec
        # =================================================================
        
        if sandbox_discovery_result and sandbox_discovery_result.get("selected_file"):
            selected = sandbox_discovery_result["selected_file"]
            folder_path = sandbox_discovery_result["path"]
            
            output_mode = detect_output_mode(combined_text)
            
            # =================================================================
            # v1.30: Q&A CONTEXT BIAS
            # If file is Q&A structure + user text has answer keywords + output_mode is CHAT_ONLY,
            # bias toward REWRITE_IN_PLACE (unless explicit CHAT_ONLY signals present)
            # =================================================================
            file_content = selected.get("content", "")
            if file_content and output_mode == OutputMode.CHAT_ONLY:
                # Check if file has Q&A structure
                qa_analysis = analyze_qa_file(file_content)
                
                if qa_analysis.get("is_qa_file") and qa_analysis.get("total_questions", 0) > 0:
                    # Check for answer-related keywords in user text
                    text_lower = combined_text.lower()
                    answer_keywords = [
                        "answer the question",
                        "answer these",
                        "fill in",
                        "fill the answer",
                        "answer where",
                        "provide answer",
                        "write the answer",
                        "complete the answer",
                        "put the answer",
                        "answer it",
                        "answer them",
                    ]
                    has_answer_keywords = any(kw in text_lower for kw in answer_keywords)
                    
                    # Check for explicit CHAT_ONLY signals (these should win)
                    explicit_chat_only_signals = [
                        "do not modify",
                        "don't modify",
                        "don't change",
                        "do not change",
                        "leave the file",
                        "don't touch",
                        "do not touch",
                        "read only",
                        "read-only",
                        "just tell me",
                        "just show me",
                        "don't write",
                        "do not write",
                        "chat only",
                    ]
                    has_explicit_chat_only = any(sig in text_lower for sig in explicit_chat_only_signals)
                    
                    if has_answer_keywords and not has_explicit_chat_only:
                        logger.warning(
                            "[spec_generation] v1.30 AUTO-OVERRIDE: CHAT_ONLY -> REWRITE_IN_PLACE "
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
            else:  # CHAT_ONLY
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
                    # v1.31: Pass user_request (combined_text) for deep analysis detection
                    spec.sandbox_generated_reply = await generate_reply_from_content(
                        full_content, 
                        selected.get("content_type"),
                        provider_id=provider_id,
                        model_id=model_id,
                        llm_call_func=llm_call if _LLM_CALL_AVAILABLE else None,
                        output_mode=output_mode.value if output_mode else None,
                        user_request=combined_text,  # v1.31: Enable deep Q&A analysis
                    )
                    logger.info(
                        "[spec_generation] v1.31 Generated LLM reply (user_request passed): %s",
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
        
        conversation_messages = []
        if constraints_hint:
            conversation_messages = constraints_hint.get("messages", [])
            if not conversation_messages:
                conversation_messages = constraints_hint.get("conversation", [])
        
        detected_stack = detect_implementation_stack(conversation_messages, weaver_job_text, intent)
        if detected_stack:
            spec.implementation_stack = detected_stack
            logger.info(
                "[spec_generation] v1.11 Detected implementation stack: %s/%s (locked=%s)",
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
                    "[spec_generation] v1.19 SCAN_ONLY params: roots=%s, terms=%s",
                    spec.scan_roots, spec.scan_terms
                )
        
        # =================================================================
        # STEP 3.8: Multi-File Operation Population (v1.33)
        # =================================================================
        # If multi-file operation was detected, populate the spec with it.
        
        if multi_file_op and multi_file_op.is_multi_file:
            spec.multi_file = multi_file_op
            
            logger.info(
                "[spec_generation] v1.33 Populated spec.multi_file: type=%s, files=%d",
                multi_file_op.operation_type,
                multi_file_op.total_files,
            )
            
            # Add to confirmed components if files were found
            if multi_file_op.total_files > 0:
                spec.confirmed_components.append(GroundedFact(
                    description=f"Multi-file {multi_file_op.operation_type}: {multi_file_op.total_files} files, {multi_file_op.total_occurrences} occurrences",
                    source="file_discovery",
                    confidence="confirmed",
                ))
            
            # Add constraints for refactor operations
            if multi_file_op.operation_type == "refactor":
                spec.constraints_from_repo.append(
                    f"⚠️ MULTI-FILE REFACTOR: {multi_file_op.total_files} files will be modified "
                    f"(pattern: `{multi_file_op.search_pattern}` → `{multi_file_op.replacement_pattern}`)"
                )
            
            # Set output mode for search operations (chat only)
            if multi_file_op.operation_type == "search":
                spec.sandbox_output_mode = "chat_only"
                spec.constraints_from_repo.append(
                    f"Output mode: CHAT_ONLY (multi-file search results - {multi_file_op.total_occurrences} occurrences)"
                )
            
            # Generate reply from file preview for search operations
            if multi_file_op.operation_type == "search" and multi_file_op.file_preview:
                spec.sandbox_generated_reply = spec.get_multi_file_summary()
            
            # Set scope based on multi-file operation
            if multi_file_op.operation_type == "search":
                spec.in_scope = [f"Search codebase for `{multi_file_op.search_pattern}` and report results"]
            elif multi_file_op.operation_type == "refactor":
                spec.in_scope = [
                    f"Replace `{multi_file_op.search_pattern}` with `{multi_file_op.replacement_pattern}` "
                    f"in {multi_file_op.total_files} files"
                ]
            
            # Handle discovery errors
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
                "[spec_generation] v1.5: Parsed user_answers into decisions: %s",
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
            "[spec_generation] v1.4 Completion check: complete_enough=%s, reason='%s'",
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
            logger.info("[spec_generation] v1.4 EARLY EXIT: %s (round %d)", completion_reason, round_n)
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
            "[spec_generation] v1.9 Job classification: kind=%s, confidence=%.2f",
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
            # v1.33: Multi-file operation data
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
            # v1.25: Evidence-First filesystem data
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
            "[spec_generation] Result: complete=%s, questions=%d, round=%d",
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
        logger.exception("[spec_generation] HARD STOP: %s", e)
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
    # Intent parsing
    "parse_weaver_intent",
    # Grounding
    "ground_intent_with_evidence",
    # Question generation
    "generate_grounded_questions",
    # Step/test derivation
    "_derive_steps_from_domain",
    "_derive_tests_from_domain",
    # Completeness check
    "_is_spec_complete_enough",
    # Markdown builder
    "build_pot_spec_markdown",
    # Main entry point
    "run_spec_gate_grounded",
    # Helpers
    "_extract_paths_from_text",
    "_extract_keywords",
]
