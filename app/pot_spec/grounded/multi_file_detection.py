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
    from app.overwatcher.sandbox_client import get_sandbox_client
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

MULTI_FILE_SCOPE_INDICATORS = [
    # Explicit scope keywords
    r"\b(?:all|every|everywhere|entire|whole)\b",
    r"\bcodebase\b",
    r"\brepo(?:sitory)?\b",
    r"\bproject\b",
    r"\bacross\s+(?:all\s+)?(?:files?|the\s+codebase)\b",
    r"\bthroughout\b",
    r"\bsystem[- ]?wide\b",
    r"\bproject[- ]?wide\b",
    # v2.3: Drive references - more flexible matching
    r"\b[A-Za-z]\s+drive\b",  # "D drive"
    r"\b[A-Za-z]:\s*(?:\\|/)",  # "D:\" or "D:/"
    r"\b[A-Za-z]:[,\s]",  # v2.3: "D:," or "D: " (Weaver output format)
    r"\bon\s+[A-Za-z]:\b",  # "on D:"
    r"\bover\s+(?:the\s+)?[A-Za-z]\s+drive\b",
    r"\bsearch\s+over\b",
    r"\bscan\s+(?:the\s+)?(?:entire|whole|all)\b",
    # File/folder scope
    r"\bfile\s+(?:names?|structures?)\b",
    r"\bfolder\s+names?\b",
    r"\bwithin\s+(?:the\s+)?(?:code|files?|folders?)\b",
    r"\bin\s+(?:all\s+)?(?:files?|folders?|the\s+code)\b",
    # v2.3: Rename/refactor scope indicators
    r"\brename\s+(?:from|to)\b",
    r"\breplace\s+(?:all|instances|occurrences)\b",
    r"\bchange\s+(?:the\s+)?(?:ui|front-?end|branding)\b",
    r"\borb\s+(?:desktop|system)\b",  # Specific project references
    r"\bfront-?end\s+ui\b",
]

_SCOPE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in MULTI_FILE_SCOPE_INDICATORS]


def _has_multi_file_scope(text: str) -> bool:
    """Check if text indicates multi-file scope."""
    if not text:
        return False
    for pattern in _SCOPE_PATTERNS:
        if pattern.search(text):
            return True
    return False


# =============================================================================
# STOPWORD AND VALIDATION (v2.1 - Critical safety gate)
# =============================================================================

STOPWORDS = frozenset({
    'and', 'or', 'the', 'to', 'a', 'an', 'in', 'on', 'at', 'for', 'of', 'with',
    'by', 'from', 'as', 'is', 'it', 'be', 'are', 'was', 'were', 'been', 'being',
    'all', 'any', 'every', 'each', 'some', 'that', 'this', 'these', 'those',
    'them', 'they', 'their', 'there', 'then', 'than',
    'find', 'search', 'look', 'replace', 'rename', 'change', 'update',
    'file', 'files', 'folder', 'folders', 'code', 'codebase',
    'extras', 'features', 'scope', 'task', 'micro', 'plan', 'audit',
    'safe', 'reference', 'references', 'occurrence', 'occurrences',
    'case', 'insensitive', 'sensitive',
})


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


UNICODE_QUOTES = '"\'""\'\''


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

def _detect_multi_file_intent(
    combined_text: str,
    constraints_hint: Optional[Dict] = None,
    project_paths: Optional[List[str]] = None,
    vision_results: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Detect multi-file operation intent from combined text or constraints.
    
    v2.2: Falls back to context-aware inference when direct extraction fails.
    """
    # Check if constraints_hint already has multi-file metadata
    if constraints_hint:
        multi_file_meta = constraints_hint.get("multi_file_metadata")
        if multi_file_meta and multi_file_meta.get("is_multi_file"):
            logger.info("[multi_file_detection] v2.1 Multi-file metadata found in constraints_hint: %s",
                        multi_file_meta)
            return multi_file_meta
    
    if not combined_text:
        return None
    
    text = combined_text.strip()
    text_lower = text.lower()
    
    if len(text_lower) < 15:
        return None
    
    # STEP 1: Check for scope indicators
    has_scope = _has_multi_file_scope(text)
    print(f"[multi_file_detection] v2.3 Multi-file scope check: has_scope={has_scope}")
    if not has_scope:
        return None
    
    # STEP 2: Check for refactor/rename intent keywords
    # v2.3: Fixed patterns to match actual Weaver output (lowercase matching)
    refactor_intent_patterns = [
        # Direct verb patterns
        r"\b(?:rename|replace|change|update)\s+(?:all|every|it|them|that|to|with|instances|branding|occurrences|references)",
        r"\b(?:want\s+to|need\s+to|should)\s+(?:rename|replace|change)",
        # Search + action patterns
        r"\bfind\s+(?:all\s+)?(?:references?|occurrences?|instances?)\b",
        r"\bsearch\s+(?:for|over)\b.*?(?:change|rename|replace)",
        # Safety patterns
        r"\bwithout\s+breaking\b",
        r"\brename\s+(?:plan\s+)?to\b",
        r"\bsafe\s+rename\b",
        # v2.3: Weaver output patterns (case-insensitive matching on lowercase text)
        r"\bfront-?end\s+rename\b",  # "Front-end rename"
        r"\brename\s+(?:of\s+)?[a-z]+\s+(?:ui\s+)?to\s+[a-z]",  # "rename of orb ui to astra"
        r"\binstances\s*/\s*branding\b",  # "instances/branding" (the slash!)
        r"\binstances\s+of\s",  # "instances of"
        r"\bbranding\s+of\s",  # "branding of"
        r"\b(?:of|from)\s+['\"]?[a-z]+['\"]?\s+(?:to|with)\s+['\"]?[a-z]+['\"]?",  # "of 'orb' with 'astra'"
        r"\bui\s+to\s+[a-z]",  # "UI to Astra"
        r"\bto\s+astra\b",  # explicit "to Astra"
        r"\borb\s+to\s+astra\b",  # explicit "Orb to Astra"
        r"\breplace\s+.*?\s+with\s",  # "replace ... with"
        r"\bchange\s+.*?\s+to\s+astra\b",  # "change ... to Astra"
        r"\bso\s+it'?s\s+called\b",  # "so it's called"
        r"\bcalled\s+astra\b",  # "called Astra"
    ]
    
    has_refactor_intent = False
    for pattern in refactor_intent_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
            has_refactor_intent = True
            break
    
    print(f"[multi_file_detection] v2.3 Refactor intent check: has_intent={has_refactor_intent}")
    
    if not has_refactor_intent:
        return None
    
    # STEP 3: Try direct extraction first
    terms = _extract_search_and_replace_terms(text)
    
    if terms:
        search_pattern = terms.get("search_pattern", "")
        replacement_pattern = terms.get("replacement_pattern", "")
        if search_pattern and replacement_pattern:
            logger.info("[multi_file_detection] v2.2 MULTI-FILE REFACTOR detected: '%s' -> '%s'",
                        search_pattern, replacement_pattern)
            return {
                "is_multi_file": True,
                "operation_type": "refactor",
                "search_pattern": search_pattern,
                "replacement_pattern": replacement_pattern,
            }
    
    # STEP 4: v2.2 FALLBACK - Try context-aware inference
    print("[multi_file_detection] v2.2 Direct extraction failed, trying context-aware inference...")
    
    replacement_pattern = _extract_replacement_only(text)
    if replacement_pattern:
        search_pattern = _infer_search_term_from_context(
            text=text,
            project_paths=project_paths,
            vision_results=vision_results,
            constraints_hint=constraints_hint,
        )
        
        if search_pattern:
            logger.info("[multi_file_detection] v2.2 CONTEXT-INFERRED REFACTOR: '%s' -> '%s'",
                        search_pattern, replacement_pattern)
            return {
                "is_multi_file": True,
                "operation_type": "refactor",
                "search_pattern": search_pattern,
                "replacement_pattern": replacement_pattern,
                "inferred": True,  # Flag that this was context-inferred
            }
    
    logger.warning("[multi_file_detection] v2.2 Has refactor intent + scope but extraction failed")
    return None


# =============================================================================
# DISCOVERY RESULT CONVERSION (v2.0)
# =============================================================================

def _convert_discovery_to_raw_matches(discovery_result: DiscoveryResult) -> List[RawMatch]:
    """v2.0: Convert file discovery results to RawMatch format for classification."""
    raw_matches = []
    for fm in discovery_result.files:
        for lm in fm.line_matches:
            raw_matches.append(RawMatch(
                file_path=fm.path,
                line_number=lm.line_number,
                line_content=lm.line_content,
                match_text=lm.line_content.strip() if lm.line_content else "",
            ))
    return raw_matches


# =============================================================================
# MULTI-FILE OPERATION BUILDER (v2.1)
# =============================================================================

async def _build_multi_file_operation(
    operation_type: str,
    search_pattern: str,
    replacement_pattern: str = "",
    file_filter: Optional[str] = None,
    sandbox_client: Optional[Any] = None,
    job_description: str = "",
    provider_id: str = "anthropic",
    model_id: str = "claude-sonnet-4-20250514",
    explicit_roots: Optional[List[str]] = None,
    vision_context: str = "",
) -> MultiFileOperation:
    """
    Run file discovery, classify matches, and build MultiFileOperation for spec.
    
    v2.2: Added explicit_roots parameter to scope discovery to specific projects.
    When explicit_roots is provided, ONLY those paths are searched (not DEFAULT_ROOTS).
    
    v2.4: Added vision_context parameter for intelligent UI classification.
    When vision context is present, the classifier can distinguish between
    user-visible UI text (title bars, headings) and internal code paths.
    """
    logger.info("[multi_file_detection] v2.4 Building multi-file operation: type=%s, pattern=%s, vision_context=%d chars",
                operation_type, search_pattern, len(vision_context))
    
    if vision_context:
        print(f"[multi_file_detection] v2.4 VISION CONTEXT available for refactor ({len(vision_context)} chars)")
    
    if not _is_valid_term(search_pattern):
        error_msg = f"REJECTED: search pattern '{search_pattern}' is a stopword or invalid"
        logger.error("[multi_file_detection] v2.1 %s", error_msg)
        return MultiFileOperation(
            is_multi_file=True,
            operation_type=operation_type,
            search_pattern=search_pattern,
            replacement_pattern=replacement_pattern,
            requires_confirmation=True,
            error_message=error_msg,
        )
    
    if not _SANDBOX_CLIENT_AVAILABLE or not get_sandbox_client:
        logger.warning("[multi_file_detection] v2.1 Sandbox client not available")
        return MultiFileOperation(
            is_multi_file=True,
            operation_type=operation_type,
            search_pattern=search_pattern,
            replacement_pattern=replacement_pattern,
            requires_confirmation=(operation_type == "refactor"),
            error_message="Sandbox client not available for file discovery",
        )
    
    try:
        client = sandbox_client or get_sandbox_client()
        
        # v2.2: Use explicit_roots if provided, otherwise fall back to defaults
        search_roots = explicit_roots if explicit_roots else DISCOVERY_DEFAULT_ROOTS
        logger.info("[multi_file_detection] v2.2 Discovery roots: %s (explicit=%s)",
                    search_roots, explicit_roots is not None)
        
        result = discover_files(
            search_pattern=search_pattern,
            sandbox_client=client,
            file_filter=file_filter,
            roots=search_roots,
        )
        
        if not result.success:
            return MultiFileOperation(
                is_multi_file=True,
                operation_type=operation_type,
                search_pattern=search_pattern,
                replacement_pattern=replacement_pattern,
                requires_confirmation=(operation_type == "refactor"),
                error_message=result.error_message,
            )
        
        raw_matches = _convert_discovery_to_raw_matches(result)
        
        refactor_plan: Optional[RefactorPlan] = None
        rename_plan: Optional[RenamePlan] = None
        classification_markdown = ""
        classification_json: Dict[str, Any] = {}
        confirmation_message = ""
        
        if operation_type == "refactor" and raw_matches:
            try:
                match_dicts = [
                    {'file_path': m.file_path, 'line_number': m.line_number, 'line_content': m.line_content}
                    for m in raw_matches
                ]
                
                rename_plan = build_rename_plan(
                    matches=match_dicts,
                    search_pattern=search_pattern,
                    replace_pattern=replacement_pattern,
                )
                
                classification_markdown = rename_plan.get_report()
                classification_json = {
                    'search_pattern': rename_plan.search_pattern,
                    'replace_pattern': rename_plan.replace_pattern,
                    'safe_count': rename_plan.safe_count,
                    'unsafe_count': rename_plan.unsafe_count,
                    'migration_count': rename_plan.migration_count,
                    'safe_files': list(set(m.file_path for m in rename_plan.safe_to_rename)),
                    'excluded_files': list(set(m.file_path for m in rename_plan.unsafe_excluded)),
                    'migration_files': list(set(m.file_path for m in rename_plan.migration_required)),
                    'required_checks': rename_plan.required_checks,
                }
                
                if rename_plan.unsafe_count > 0:
                    confirmation_message = (
                        f"⚠️ PARTIAL RENAME: {rename_plan.safe_count} items will be renamed, "
                        f"{rename_plan.unsafe_count} EXCLUDED (would break invariants), "
                        f"{rename_plan.migration_count} require migration."
                    )
                else:
                    confirmation_message = f"✅ FULL RENAME: All {rename_plan.safe_count} items can be safely renamed."
                    
            except Exception as policy_err:
                logger.error("[multi_file_detection] v2.1 Rename policy failed: %s", policy_err)
                try:
                    refactor_plan = await build_refactor_plan(
                        raw_matches=raw_matches,
                        search_term=search_pattern,
                        replace_term=replacement_pattern,
                        context=job_description or f"Refactor: rename '{search_pattern}' to '{replacement_pattern}'",
                        provider_id=provider_id,
                        model_id=model_id,
                        llm_call_func=llm_call if _LLM_CALL_AVAILABLE else None,
                        vision_context=vision_context,  # v2.4: Pass vision context
                    )
                    if refactor_plan:
                        classification_markdown = format_human_readable(refactor_plan)
                        classification_json = format_machine_readable(refactor_plan)
                        confirmation_message = format_confirmation_message(refactor_plan)
                except Exception as classify_err:
                    logger.error("[multi_file_detection] v2.1 LLM classification also failed: %s", classify_err)
        
        file_preview = classification_markdown if classification_markdown else result.get_summary_report()
        
        if rename_plan:
            target_files = list(set(m.file_path for m in rename_plan.safe_to_rename))
        elif refactor_plan:
            target_files = list(set(m.file_path for m in refactor_plan.get_change_matches()))
        else:
            target_files = [fm.path for fm in result.files]
        
        return MultiFileOperation(
            is_multi_file=True,
            operation_type=operation_type,
            search_pattern=search_pattern,
            replacement_pattern=replacement_pattern,
            target_files=target_files,
            total_files=result.total_files,
            total_occurrences=result.total_occurrences,
            file_filter=file_filter,
            file_preview=file_preview,
            discovery_truncated=result.truncated,
            discovery_duration_ms=result.duration_ms,
            roots_searched=result.roots_searched,
            requires_confirmation=(operation_type == "refactor"),
            confirmed=False,
            error_message=None,
            refactor_plan=refactor_plan,
            classification_markdown=classification_markdown,
            classification_json=classification_json,
            confirmation_message=confirmation_message,
        )
        
    except Exception as e:
        logger.error("[multi_file_detection] v2.1 Multi-file operation failed: %s", e)
        return MultiFileOperation(
            is_multi_file=True,
            operation_type=operation_type,
            search_pattern=search_pattern,
            replacement_pattern=replacement_pattern,
            requires_confirmation=(operation_type == "refactor"),
            error_message=f"Discovery failed: {str(e)}",
        )
