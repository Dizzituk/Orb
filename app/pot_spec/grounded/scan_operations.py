# FILE: app/pot_spec/grounded/scan_operations.py
"""
Scan Operations Security and Parameter Extraction (v1.19, v1.21)

CRITICAL SECURITY: This module contains the scan security gates that prevent
scanning the host PC filesystem. Scans are ALWAYS constrained to sandbox
workspace directories only.

Version Notes:
-------------
v1.19 (2026-01): Added scan parameter extraction
v1.21 (2026-01): CRITICAL SECURITY FIX - host PC protection
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# SCAN SAFETY CONSTANTS (v1.21 - CRITICAL SECURITY)
# =============================================================================

# Default exclusions for filesystem scans
DEFAULT_SCAN_EXCLUSIONS = [
    ".git", "node_modules", "__pycache__", ".venv", "venv", 
    "dist", "build", ".next", ".nuxt", "target",
    ".idea", ".vscode", ".vs", "*.pyc", "*.pyo",
    ".mypy_cache", ".pytest_cache", ".tox", "egg-info",
]

# SAFE_DEFAULT_SCAN_ROOTS: The ONLY directories that SCAN_ONLY jobs can target.
# "Entire D drive" MUST be interpreted as "entire allowed workspace", NOT literal D:\
SAFE_DEFAULT_SCAN_ROOTS = ["D:\\Orb", "D:\\orb-desktop"]

# FORBIDDEN_SCAN_ROOTS: Patterns that MUST NEVER be used as scan roots.
FORBIDDEN_SCAN_ROOTS = [
    "C:\\",
    "D:\\",
    "E:\\",
    "F:\\",
]


# =============================================================================
# SECURITY VALIDATION FUNCTIONS
# =============================================================================

def _is_path_within_allowed_roots(path: str, allowed_roots: List[str] = None) -> bool:
    """
    v1.21 SECURITY GATE: Check if a path is within allowed scan roots.
    
    Args:
        path: The path to validate
        allowed_roots: List of allowed root paths (defaults to SAFE_DEFAULT_SCAN_ROOTS)
    
    Returns:
        True if path is within an allowed root, False otherwise
    """
    if allowed_roots is None:
        allowed_roots = SAFE_DEFAULT_SCAN_ROOTS
    
    path_normalized = path.replace('/', '\\').rstrip('\\').lower()
    
    # Check if it's a bare drive letter (FORBIDDEN)
    if len(path_normalized) <= 3 and path_normalized.endswith(':'):
        logger.warning("[scan_operations] v1.21 SECURITY: Rejected bare drive root: %s", path)
        return False
    if len(path_normalized) <= 3 and path_normalized.endswith(':\\'):
        logger.warning("[scan_operations] v1.21 SECURITY: Rejected bare drive root: %s", path)
        return False
    
    for allowed_root in allowed_roots:
        allowed_normalized = allowed_root.replace('/', '\\').rstrip('\\').lower()
        
        if path_normalized == allowed_normalized:
            return True
        
        if path_normalized.startswith(allowed_normalized + '\\'):
            return True
    
    logger.warning(
        "[scan_operations] v1.21 SECURITY: Path '%s' is NOT within allowed roots %s",
        path, allowed_roots
    )
    return False


def validate_scan_roots(scan_roots: List[str]) -> Tuple[List[str], List[str]]:
    """
    v1.21 SECURITY GATE: Validate and filter scan roots to only allowed paths.
    
    This is a HARD security gate that ensures scan operations can ONLY target
    the allowed workspace directories (D:\\Orb, D:\\orb-desktop), never the
    host filesystem root.
    
    Args:
        scan_roots: List of proposed scan root paths
    
    Returns:
        Tuple of (valid_roots, rejected_roots)
    """
    valid_roots = []
    rejected_roots = []
    
    for root in scan_roots:
        if _is_path_within_allowed_roots(root):
            normalized = root.replace('/', '\\').rstrip('\\')
            if normalized not in valid_roots:
                valid_roots.append(normalized)
        else:
            rejected_roots.append(root)
    
    if rejected_roots:
        logger.warning(
            "[scan_operations] v1.21 SECURITY GATE: Rejected %d scan root(s): %s",
            len(rejected_roots), rejected_roots
        )
    
    if valid_roots:
        logger.info(
            "[scan_operations] v1.21 SECURITY GATE: Accepted %d scan root(s): %s",
            len(valid_roots), valid_roots
        )
    else:
        logger.warning(
            "[scan_operations] v1.21 SECURITY GATE: No valid roots, falling back to SAFE_DEFAULT_SCAN_ROOTS: %s",
            SAFE_DEFAULT_SCAN_ROOTS
        )
        valid_roots = SAFE_DEFAULT_SCAN_ROOTS.copy()
    
    return valid_roots, rejected_roots


# =============================================================================
# SCAN PARAMETER EXTRACTION
# =============================================================================

def extract_scan_params(
    combined_text: str,
    intent: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    v1.19: Extract scan parameters from user request text.
    v1.21: SECURITY HARDENED - all paths validated through security gate.
    
    Args:
        combined_text: Full text combining user input and Weaver output
        intent: Parsed intent dict
    
    Returns:
        Dict with scan parameters, or None if extraction fails
    """
    if not combined_text:
        logger.warning("[scan_operations] v1.20 extract_scan_params: combined_text is EMPTY")
        return None
    
    logger.info(
        "[scan_operations] v1.20 extract_scan_params CALLED: text_len=%d, preview='%s'",
        len(combined_text), combined_text[:300].replace('\n', ' ')
    )
    
    text = combined_text
    text_lower = combined_text.lower()
    
    scan_roots: List[str] = []
    scan_terms: List[str] = []
    scan_targets: List[str] = []
    scan_case_mode: str = "case_insensitive"
    scan_exclusions: List[str] = DEFAULT_SCAN_EXCLUSIONS.copy()
    
    # =========================================================================
    # 1. SCAN ROOTS EXTRACTION (v1.21 - SECURITY HARDENED)
    # =========================================================================
    
    candidate_paths: List[str] = []
    user_wants_full_workspace = False
    
    # Detect "entire drive" / "whole drive" requests
    full_workspace_patterns = [
        r'entire\s+([a-z])\s*drive',
        r'whole\s+([a-z])\s*drive',
        r'\b([a-z])\s+drive\b',
        r'scan\s+([a-z]):\s*(?![\\\w])',
        r'\bon\s+([a-z]):',
        r'\bfiles/folders\s+on\s+([a-z]):',
        r'\bfolders\s+on\s+([a-z]):',
        r'\bfiles\s+on\s+([a-z]):',
    ]
    
    for pattern in full_workspace_patterns:
        if re.search(pattern, text_lower):
            user_wants_full_workspace = True
            logger.info(
                "[scan_operations] v1.21 SECURITY: 'Full drive' request detected. "
                "Interpreting as 'entire allowed workspace', NOT bare drive root."
            )
            break
    
    # Extract explicit paths from text
    path_patterns = [
        r'([A-Za-z]:\\[\w\-\\]+)',
        r'([A-Za-z]:/[\w\-/]+)',
    ]
    
    for pattern in path_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            normalized = match.replace('/', '\\').rstrip('\\')
            if normalized not in candidate_paths:
                candidate_paths.append(normalized)
                logger.info("[scan_operations] v1.21 Extracted candidate path: %s", normalized)
    
    # SECURITY GATE - Validate all candidate paths
    if candidate_paths:
        valid_roots, rejected_roots = validate_scan_roots(candidate_paths)
        scan_roots = valid_roots
        
        if rejected_roots:
            logger.warning(
                "[scan_operations] v1.21 SECURITY: Rejected paths outside allowed workspace: %s",
                rejected_roots
            )
    
    # Handle "full workspace" request or no valid paths
    if user_wants_full_workspace:
        scan_roots = SAFE_DEFAULT_SCAN_ROOTS.copy()
        logger.info(
            "[scan_operations] v1.21 SECURITY: 'Full workspace' scan requested. "
            "Using SAFE_DEFAULT_SCAN_ROOTS: %s",
            scan_roots
        )
    elif not scan_roots:
        scan_roots = SAFE_DEFAULT_SCAN_ROOTS.copy()
        logger.info(
            "[scan_operations] v1.21 No valid scan roots extracted, using SAFE_DEFAULT_SCAN_ROOTS: %s",
            scan_roots
        )
    
    # FINAL SECURITY CHECK - Ensure NO bare drive letters slipped through
    final_roots = []
    for root in scan_roots:
        normalized = root.replace('/', '\\').rstrip('\\')
        if len(normalized) <= 3:
            logger.error(
                "[scan_operations] v1.21 SECURITY VIOLATION: Bare drive letter '%s' rejected in final check!",
                root
            )
            continue
        final_roots.append(normalized)
    
    if not final_roots:
        final_roots = SAFE_DEFAULT_SCAN_ROOTS.copy()
        logger.warning(
            "[scan_operations] v1.21 SECURITY: Final check removed all roots, using safe defaults: %s",
            final_roots
        )
    
    scan_roots = final_roots
    
    # =========================================================================
    # 2. SCAN TERMS EXTRACTION
    # =========================================================================
    
    ignored_slash_groups = {
        "file/folder", "folder/file", "files/folders", "folders/files",
        "and/or", "yes/no", "true/false", "on/off",
        "input/output", "read/write", "start/stop",
    }
    
    # Slash-separated variants
    slash_pattern = r'([\w]+(?:/[\w]+)+)'
    slash_matches = re.findall(slash_pattern, text)
    for match in slash_matches:
        if match.lower() in ignored_slash_groups:
            continue
        if '\\' in match or len(match.split('/')) > 5:
            continue
        
        terms = match.split('/')
        lower_terms = [t.lower() for t in terms]
        unique_lower = set(lower_terms)
        
        is_case_variant_group = len(unique_lower) == 1 and len(terms) > 1
        
        if is_case_variant_group:
            for term in terms:
                term = term.strip()
                if term and term not in scan_terms and len(term) >= 2:
                    scan_terms.append(term)
    
    # Quoted strings
    all_quoted_matches = re.findall(r'"([\w\-\.]+)"', text)
    for match in all_quoted_matches:
        term = match.strip()
        if term and term not in scan_terms and len(term) >= 2:
            scan_terms.append(term)
    
    single_quoted_matches = re.findall(r"'([\w\-\.]+)'", text)
    for match in single_quoted_matches:
        term = match.strip()
        if term and term not in scan_terms and len(term) >= 2:
            scan_terms.append(term)
    
    # Descriptive patterns
    term_stopwords = {
        "the", "a", "an", "all", "any", "every", "each", "some", "this", "that",
        "in", "on", "at", "to", "of", "for", "from", "with", "by", "into",
        "files", "file", "folders", "folder", "directories", "directory",
        "names", "name", "references", "reference", "mentions", "mention",
        "paths", "path", "contents", "content", "code", "structure",
        "drive", "entire", "whole", "project", "codebase", "scan",
        "find", "search", "report", "show", "list", "full",
        "look", "inside", "why", "explain", "there",
        "and", "or", "but", "also",
    }
    
    descriptive_patterns = [
        r'references?\s+to\s+([\w\-\.]+)',
        r'references?\s+of\s+([\w\-\.]+)',
        r'mentions?\s+of\s+([\w\-\.]+)',
        r'occurrences?\s+of\s+([\w\-\.]+)',
        r'instances?\s+of\s+([\w\-\.]+)',
        r'containing\s+([\w\-\.]+)',
        r'(?:named|called)\s+["\']?([\w\-\.]+)["\']?',
        r'\bfor\s+([\w\-\.]+)\s+in\s+file',
        r'search(?:ing)?\s+for\s+([\w\-\.]+)',
        r'find(?:ing)?\s+([\w\-\.]+)\s+(?:in|on|anywhere)',
        r'looking\s+for\s+([\w\-\.]+)',
    ]
    
    for pattern in descriptive_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            term = match.strip()
            if term and term not in term_stopwords and term not in [t.lower() for t in scan_terms] and len(term) >= 2:
                original_match = re.search(re.escape(term), text, re.IGNORECASE)
                if original_match:
                    term = original_match.group(0)
                if term not in scan_terms:
                    scan_terms.append(term)
    
    # =========================================================================
    # 3. SCAN TARGETS EXTRACTION
    # =========================================================================
    
    names_keywords = [
        "file name", "filename", "folder name", "directory name",
        "named", "called", "names", "in file/folder structure",
        "file/folder", "in file and folder",
    ]
    
    contents_keywords = [
        "inside code", "in code", "code reference", "within files",
        "file content", "contents", "inside files", "in files",
        "source code", "import", "references in code",
        "within any code",
    ]
    
    has_names = any(kw in text_lower for kw in names_keywords)
    has_contents = any(kw in text_lower for kw in contents_keywords)
    
    if has_names:
        scan_targets.append("names")
    if has_contents:
        scan_targets.append("contents")
    
    if not scan_targets or "all references" in text_lower or "references" in text_lower:
        if "names" not in scan_targets:
            scan_targets.append("names")
        if "contents" not in scan_targets:
            scan_targets.append("contents")
    
    # =========================================================================
    # 4. CASE MODE EXTRACTION
    # =========================================================================
    
    if "case insensitive" in text_lower or "case-insensitive" in text_lower:
        scan_case_mode = "case_insensitive"
    elif "case sensitive" in text_lower or "case-sensitive" in text_lower:
        scan_case_mode = "case_sensitive"
    elif len(scan_terms) > 1:
        lower_terms = set(t.lower() for t in scan_terms)
        if len(lower_terms) < len(scan_terms):
            scan_case_mode = "case_sensitive"
    
    # =========================================================================
    # 5. BUILD RESULT
    # =========================================================================
    
    result = {
        "scan_roots": scan_roots,
        "scan_terms": scan_terms,
        "scan_targets": scan_targets,
        "scan_case_mode": scan_case_mode,
        "scan_exclusions": scan_exclusions,
    }
    
    logger.info(
        "[scan_operations] v1.20 extract_scan_params RESULT: roots=%s, terms=%s, targets=%s, case=%s",
        scan_roots, scan_terms, scan_targets, scan_case_mode
    )
    
    return result
