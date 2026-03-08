# FILE: app/pot_spec/grounded/multi_file_detection.py
"""
Multi-File Operation Detection for SpecGate

This module handles detection and building of multi-file operations,
including search/replace refactoring across codebases.

Responsibilities:
- Detect multi-file scope indicators in natural language
- Extract search and replacement terms from conversational text
- Build MultiFileOperation objects with LLM-powered classification
- Convert discovery results to RawMatch format for classification

Key Features (v2.1):
- STOPWORD VALIDATION: Never accepts "and", "the", etc. as search terms
- Spaced identifier normalization: "O-R-B" -> "ORB"
- Unicode quote normalization for pattern matching
- Flexible pattern extraction for GPT-5.2 Weaver output
- LLM-powered intelligent match classification
- 12-category bucket system for match analysis

Used by:
- spec_runner.py for multi-file operation detection

Version: v2.1 (2026-02-01) - Critical stopword validation fix
Version: v2.2 (2026-02-01) - Context-aware search term inference
Version: v2.4 (2026-02-01) - VISION CONTEXT FLOW FIX
    - Added vision_context parameter to _build_multi_file_operation()
    - Vision context is passed to build_refactor_plan() for intelligent classification
    - Classifier now knows which matches are USER-VISIBLE UI elements
    - When replacement is extracted but search term is missing:
      1. Infer from project path (orb-desktop -> Orb)
      2. Infer from text context ("the Orb system" -> Orb)
      3. Infer from vision analysis (if Gemini saw "ORB")
    - Added _infer_search_term_from_context()
    - Added _extract_replacement_only()
    - Fixed: "rename it to Astra" now works when context provides the source
Version: v2.3 (2026-02-01) - Enhanced scope and intent patterns for Orb→Astra rename
    - Added flexible drive references: "D drive", "on D:", "D:," formats
    - Added rename/refactor scope indicators in MULTI_FILE_SCOPE_INDICATORS
    - Added Weaver output patterns: "instances/branding", "to Astra", etc.
    - Fixed: "front-end UI" and "Orb Desktop" now detected as multi-file scope
    - Version markers now show v2.3 in logs for verification
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from app.pot_spec.grounded._multi_file_detection_utils_2 import MULTI_FILE_SCOPE_INDICATORS, STOPWORDS, UNICODE_QUOTES, _SCOPE_PATTERNS, _build_multi_file_operation, _convert_discovery_to_raw_matches, _detect_multi_file_intent, _has_multi_file_scope

logger = logging.getLogger(__name__)

# Import spec models
from .spec_models import MultiFileOperation

# Import file discovery
from .file_discovery import (
    discover_files,
    DiscoveryResult,
    DEFAULT_ROOTS as DISCOVERY_DEFAULT_ROOTS,
)

# Import refactor schemas and tools
from .refactor_schemas import (
    RefactorPlan,
    RawMatch,
)
from .refactor_classifier import build_refactor_plan
from .refactor_formatter import (
    format_human_readable,
    format_machine_readable,
    format_confirmation_message,
)

# v2.1: Rename policy for invariant-aware decisions
from .rename_policy import (
    build_rename_plan,
    RenamePlan,
    RenameDecision,
)

# Sandbox client for multi-file discovery
try:
    from app.sandbox.client import get_sandbox_client
    _SANDBOX_CLIENT_AVAILABLE = True
except ImportError as e:
    logger.warning("[multi_file_detection] sandbox_client not available: %s", e)
    _SANDBOX_CLIENT_AVAILABLE = False
    get_sandbox_client = None

# LLM call function
try:
    from app.providers.registry import llm_call
    _LLM_CALL_AVAILABLE = True
except ImportError as e:
    logger.warning("[multi_file_detection] llm_call not available: %s", e)
    _LLM_CALL_AVAILABLE = False
    llm_call = None


__all__ = [
    "MULTI_FILE_SCOPE_INDICATORS",
    "_has_multi_file_scope",
    "STOPWORDS",
    "_is_valid_term",
    "_normalize_spaced_identifier",
    "UNICODE_QUOTES",
    "_normalize_quotes",
    "_extract_search_and_replace_terms",
    "_extract_replacement_only",
    "_infer_search_term_from_context",
    "_detect_multi_file_intent",
    "_convert_discovery_to_raw_matches",
    "_build_multi_file_operation",
]


# =============================================================================
# MULTI-FILE SCOPE INDICATORS (v1.42 - Enhanced Natural Language Detection)
# =============================================================================


# =============================================================================
# STOPWORD AND VALIDATION (v2.1 - Critical safety gate)
# =============================================================================


def _is_valid_term(term: str) -> bool:
    """v2.1: Validate that an extracted term is a legitimate identifier."""
    if not term:
        return False
    term_lower = term.lower().strip()
    if term_lower in STOPWORDS:
        logger.warning("[multi_file_detection] v2.1 REJECTED stopword: '%s'", term)
        print(f"[multi_file_detection] v2.1 STOPWORD REJECTED: '{term}'")
        return False
    if len(term_lower) < 2:
        return False
    if not any(c.isalnum() for c in term):
        return False
    return True


def _normalize_spaced_identifier(text: str) -> str:
    """v2.1: Normalize spaced/hyphenated identifiers like 'O-R-B' to 'ORB'."""
    spaced_pattern = r'\b([A-Z])[-\s.]([A-Z])(?:[-\s.]([A-Z]))?(?:[-\s.]([A-Z]))?(?:[-\s.]([A-Z]))?(?:[-\s.]([A-Z]))?(?:[-\s.]([A-Z]))?(?:[-\s.]([A-Z]))?(?:[-\s.]([A-Z]))?(?:[-\s.]([A-Z]))?\b'
    def join_letters(match):
        letters = [g for g in match.groups() if g is not None]
        return ''.join(letters)
    return re.sub(spaced_pattern, join_letters, text)


def _normalize_quotes(text: str) -> str:
    """Normalize Unicode smart quotes to ASCII for reliable pattern matching."""
    if not text:
        return text
    replacements = {'"': '"', '"': '"', ''': "'", ''': "'"}
    for smart, ascii_q in replacements.items():
        text = text.replace(smart, ascii_q)
    return text


# =============================================================================
# SEARCH/REPLACE TERM EXTRACTION (v2.1 - With stopword validation)
# =============================================================================

def _extract_search_and_replace_terms(text: str) -> Optional[Dict[str, str]]:
    """Extract search pattern and replacement pattern from natural language text."""
    if not text:
        return None
    
    text_clean = _normalize_quotes(text.strip())
    text_normalized = _normalize_spaced_identifier(text_clean)
    
    print(f"[multi_file_detection] v2.1 _extract_search_and_replace_terms:")
    print(f"[multi_file_detection] v2.1   INPUT ({len(text_clean)} chars): {repr(text_clean[:200])}")
    
    def _validate_and_return(search: str, replace: str, pattern_name: str) -> Optional[Dict[str, str]]:
        search = search.strip().strip("'\"")
        replace = replace.strip().strip("'\"")
        if not _is_valid_term(search):
            print(f"[multi_file_detection] v2.1   {pattern_name}: search '{search}' FAILED validation")
            return None
        if not _is_valid_term(replace):
            print(f"[multi_file_detection] v2.1   {pattern_name}: replace '{replace}' FAILED validation")
            return None
        if search.lower() == replace.lower():
            return None
        print(f"[multi_file_detection] v2.1   {pattern_name} SUCCESS: '{search}' -> '{replace}'")
        return {"search_pattern": search, "replacement_pattern": replace}
    
    # Pattern groups for extraction
    # v2.3: Enhanced to match Weaver output formats including slash patterns
    weaver_patterns = [
        (r'references?\s+to\s+["\']?([A-Za-z][A-Za-z0-9_]+)["\']?.*?replace\s+(?:them\s+|it\s+)?(?:with|by)\s+["\']?([A-Za-z][A-Za-z0-9_]+)["\']?', 'weaver_references_to_replace'),
        (r'rename\s+["\']?([A-Za-z][A-Za-z0-9_]+)["\']?\s+to\s+["\']?([A-Za-z][A-Za-z0-9_]+)["\']?', 'weaver_rename_to'),
        (r'change\s+["\']?([A-Za-z][A-Za-z0-9_]+)["\']?\s+to\s+["\']?([A-Za-z][A-Za-z0-9_]+)["\']?', 'weaver_change_to'),
        # v2.2: "rename of X to Y" or "rename X UI to Y"
        (r'rename\s+(?:of\s+)?["\']?([A-Za-z][A-Za-z0-9_]+)["\']?\s+(?:UI\s+)?to\s+["\']?([A-Za-z][A-Za-z0-9_]+)["\']?', 'weaver_rename_of_to'),
        # v2.3: "instances/branding of 'X' with 'Y'" (the slash pattern!)
        (r'(?:instances\s*/\s*branding|instances|branding|occurrences|references)\s+of\s+["\']?([A-Za-z][A-Za-z0-9_]+)["\']?\s+(?:with|to)\s+["\']?([A-Za-z][A-Za-z0-9_]+)["\']?', 'weaver_instances_of_with'),
        # v2.3: "replace instances/branding of 'X' with 'Y'"
        (r'replace\s+(?:instances\s*/\s*branding|instances|branding)\s+of\s+["\']?([A-Za-z][A-Za-z0-9_]+)["\']?\s+with\s+["\']?([A-Za-z][A-Za-z0-9_]+)["\']?', 'weaver_replace_instances_of'),
        # v2.2: "from 'X' to 'Y'" (flexible quotes)
        (r'from\s+["\']?([A-Za-z][A-Za-z0-9_]+)["\']?\s+to\s+["\']?([A-Za-z][A-Za-z0-9_]+)["\']?', 'weaver_from_to'),
        # v2.3: "rename of X UI to Y"
        (r'rename\s+of\s+["\']?([A-Za-z][A-Za-z0-9_]+)["\']?\s+UI\s+to\s+["\']?([A-Za-z][A-Za-z0-9_]+)["\']?', 'weaver_rename_of_ui_to'),
    ]
    
    for pattern, name in weaver_patterns:
        match = re.search(pattern, text_normalized, re.IGNORECASE | re.DOTALL)
        if match:
            result = _validate_and_return(match.group(1), match.group(2), name)
            if result:
                return result
    
    from_to_patterns = [
        (r'from\s+["\']([A-Za-z][A-Za-z0-9_]*)["\']\s+to\s+["\']([A-Za-z][A-Za-z0-9_]*)["\']', 'from_quoted_to_quoted'),
        (r'from\s+([A-Z][A-Za-z0-9_]*)\s+to\s+([A-Z][A-Za-z0-9_]*)', 'from_upper_to_upper'),
        (r'(?:rename|change|replace)\s+from\s+["\']?([A-Za-z][A-Za-z0-9_]*)["\']?\s+to\s+["\']?([A-Za-z][A-Za-z0-9_]*)["\']?', 'verb_from_to'),
    ]
    
    for pattern, name in from_to_patterns:
        match = re.search(pattern, text_normalized, re.IGNORECASE | re.DOTALL)
        if match:
            result = _validate_and_return(match.group(1), match.group(2), name)
            if result:
                return result
    
    direct_patterns = [
        (r'replace\s+["\']?([A-Za-z][A-Za-z0-9_]+)["\']?\s+(?:with|by)\s+["\']?([A-Za-z][A-Za-z0-9_]+)["\']?', 'replace_with'),
        (r'rename\s+(?:all\s+)?["\']?([A-Za-z][A-Za-z0-9_]+)["\']?\s+to\s+["\']?([A-Za-z][A-Za-z0-9_]+)["\']?', 'rename_to'),
        (r'change\s+(?:all\s+)?["\']?([A-Za-z][A-Za-z0-9_]+)["\']?\s+to\s+["\']?([A-Za-z][A-Za-z0-9_]+)["\']?', 'change_to'),
    ]
    
    for pattern, name in direct_patterns:
        match = re.search(pattern, text_normalized, re.IGNORECASE)
        if match:
            result = _validate_and_return(match.group(1), match.group(2), name)
            if result:
                return result
    
    # Two-part extraction as fallback
    search_extractors = [
        r'references?\s+to\s+["\']?([A-Z][A-Za-z0-9_]+)["\']?',
        r'(?:the\s+)?name\s+["\']?([A-Z][A-Za-z0-9_]+)["\']?',
        r'occurrences?\s+of\s+["\']?([A-Z][A-Za-z0-9_]+)["\']?',
        r'(?:find|search|look\s+for)\s+["\']?([A-Z][A-Za-z0-9_]+)["\']?',
    ]
    replace_extractors = [
        r'replace\s+(?:them\s+|it\s+)?(?:with|by)\s+["\']?([A-Z][A-Za-z0-9_]+)["\']?',
        r'(?:change|rename)\s+(?:them\s+|it\s+)?to\s+["\']?([A-Z][A-Za-z0-9_]+)["\']?',
        r'to\s+(?:the\s+)?name\s+["\']?([A-Z][A-Za-z0-9_]+)["\']?',
        r'(?:with|to)\s+["\']?([A-Z][A-Za-z0-9_]+)["\']?\s*$',
    ]
    
    search_term = None
    replace_term = None
    
    for pattern in search_extractors:
        match = re.search(pattern, text_normalized, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip().strip("'\"")
            if _is_valid_term(candidate):
                search_term = candidate
                break
    
    for pattern in replace_extractors:
        match = re.search(pattern, text_normalized, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip().strip("'\"")
            if _is_valid_term(candidate):
                replace_term = candidate
                break
    
    if search_term and replace_term and search_term.lower() != replace_term.lower():
        print(f"[multi_file_detection] v2.1   two_part_extraction SUCCESS: '{search_term}' -> '{replace_term}'")
        return {"search_pattern": search_term, "replacement_pattern": replace_term}
    
    print(f"[multi_file_detection] v2.1 ALL EXTRACTION FAILED")
    print(f"[multi_file_detection] v2.1   search_term={search_term}, replace_term={replace_term}")
    return None


# =============================================================================
# v2.2: CONTEXT-AWARE SEARCH TERM INFERENCE
# =============================================================================

def _extract_replacement_only(text: str) -> Optional[str]:
    """
    v2.2: Extract just the replacement term when search term extraction fails.
    
    Handles patterns like:
    - "change it to Astra"
    - "rename to ASTRA"
    - "replace with Astra"
    """
    if not text:
        return None
    
    text_clean = _normalize_quotes(text.strip())
    text_normalized = _normalize_spaced_identifier(text_clean)
    
    replacement_only_patterns = [
        r'(?:change|rename|replace)\s+(?:it|them|that|this)\s+to\s+["\']?([A-Za-z][A-Za-z0-9_]+)["\']?',
        r'(?:change|rename|replace)\s+to\s+["\']?([A-Za-z][A-Za-z0-9_]+)["\']?',
        r'replace\s+(?:it\s+)?with\s+["\']?([A-Za-z][A-Za-z0-9_]+)["\']?',
        r'\bto\s+["\']?([A-Z][A-Za-z0-9_]+)["\']?\s*$',
    ]
    
    for pattern in replacement_only_patterns:
        match = re.search(pattern, text_normalized, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip().strip("'\"")
            if _is_valid_term(candidate):
                logger.info("[multi_file_detection] v2.2 Extracted replacement-only term: '%s'", candidate)
                print(f"[multi_file_detection] v2.2 REPLACEMENT-ONLY extracted: '{candidate}'")
                return candidate
    
    return None


def _infer_search_term_from_context(
    text: str,
    project_paths: Optional[List[str]] = None,
    vision_results: Optional[Dict[str, Any]] = None,
    constraints_hint: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    v2.2: Infer the search term from available context when direct extraction fails.
    
    Sources of inference (in priority order):
    1. Project path: "orb-desktop" -> "Orb"
    2. Text context: "the Orb system" -> "Orb"
    3. Vision analysis: If Gemini saw "ORB" in screenshot -> "Orb"
    4. Weaver job description context clues
    """
    candidates = []
    
    print(f"[multi_file_detection] v2.2 _infer_search_term_from_context:")
    print(f"[multi_file_detection] v2.2   project_paths={project_paths}")
    print(f"[multi_file_detection] v2.2   vision_results={'present' if vision_results else 'None'}")
    
    # SOURCE 1: Project path inference
    if project_paths:
        for path in project_paths:
            path_clean = path.rstrip('\\/').replace('/', '\\')
            parts = path_clean.split('\\')
            if parts:
                folder_name = parts[-1]
                core_patterns = [
                    r'^([A-Za-z][A-Za-z0-9]*)[-_]?(?:desktop|app|ui|web|frontend|client|service)?$',
                    r'^([A-Za-z][A-Za-z0-9]*)$',
                ]
                for pattern in core_patterns:
                    match = re.match(pattern, folder_name, re.IGNORECASE)
                    if match:
                        core_name = match.group(1)
                        inferred = core_name.capitalize()
                        if _is_valid_term(inferred):
                            candidates.append((inferred, f"project_path:{path}", 90))
                            print(f"[multi_file_detection] v2.2   PROJECT PATH inference: '{inferred}' from '{path}'")
                        break
    
    # SOURCE 2: Text context inference
    if text:
        text_clean = _normalize_quotes(text.strip())
        context_patterns = [
            r'\bthe\s+([A-Z][A-Za-z0-9]+)\s+(?:system|app|application|ui|desktop|project|codebase|repo)\b',
            r'\b([A-Z][A-Za-z0-9]+)\s+(?:Desktop|App|UI|System|Project)\b',
            r'\bUI\s+of\s+(?:the\s+)?([A-Z][A-Za-z0-9]+)\b',
            r'\b([A-Z][A-Za-z0-9]+)\s+(?:brand|branding|name|title)\b',
        ]
        for pattern in context_patterns:
            matches = re.findall(pattern, text_clean, re.IGNORECASE)
            for match_text in matches:
                if isinstance(match_text, tuple):
                    match_text = match_text[0]
                candidate = match_text.strip().capitalize()
                if _is_valid_term(candidate):
                    candidates.append((candidate, f"text_context:{pattern[:30]}", 80))
                    print(f"[multi_file_detection] v2.2   TEXT CONTEXT inference: '{candidate}'")
    
    # SOURCE 3: Vision analysis inference
    if vision_results:
        detected_text = vision_results.get("detected_text", [])
        description = vision_results.get("description", "")
        
        if isinstance(detected_text, list):
            for dt in detected_text:
                if isinstance(dt, str) and 2 <= len(dt) <= 20:
                    if re.match(r'^[A-Z][A-Za-z0-9]*$', dt) and _is_valid_term(dt):
                        candidates.append((dt, "vision:detected_text", 95))
                        print(f"[multi_file_detection] v2.2   VISION detected text: '{dt}'")
        
        if description:
            vision_text_patterns = [
                r'(?:shows?|displays?|says?|reads?)\s+["\']([A-Z][A-Za-z0-9]+)["\']',
                r'text\s+["\']([A-Z][A-Za-z0-9]+)["\']',
                r'title\s+(?:is\s+)?["\']([A-Z][A-Za-z0-9]+)["\']',
            ]
            for pattern in vision_text_patterns:
                matches = re.findall(pattern, description, re.IGNORECASE)
                for match_text in matches:
                    if _is_valid_term(match_text):
                        candidates.append((match_text, "vision:description", 85))
                        print(f"[multi_file_detection] v2.2   VISION description inference: '{match_text}'")
    
    # SOURCE 4: Constraints hint
    if constraints_hint:
        target_project = constraints_hint.get("target_project") or constraints_hint.get("project_name")
        if target_project and isinstance(target_project, str):
            candidate = target_project.split("-")[0].split("_")[0].capitalize()
            if _is_valid_term(candidate):
                candidates.append((candidate, "constraints:target_project", 70))
                print(f"[multi_file_detection] v2.2   CONSTRAINTS inference: '{candidate}'")
    
    if not candidates:
        print(f"[multi_file_detection] v2.2   NO CANDIDATES found for inference")
        return None
    
    candidates.sort(key=lambda x: x[2], reverse=True)
    best_term, best_source, best_score = candidates[0]
    logger.info("[multi_file_detection] v2.2 INFERRED search term '%s' from %s (score=%d)",
                best_term, best_source, best_score)
    print(f"[multi_file_detection] v2.2   SELECTED: '{best_term}' from {best_source} (score={best_score})")
    
    return best_term


# =============================================================================
# MULTI-FILE INTENT DETECTION (v2.2 - With context-aware fallback)
# =============================================================================


# =============================================================================
# DISCOVERY RESULT CONVERSION (v2.0)
# =============================================================================


# =============================================================================
# MULTI-FILE OPERATION BUILDER (v2.1)
# =============================================================================
