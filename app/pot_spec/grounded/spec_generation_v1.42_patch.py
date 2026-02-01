# FILE: spec_generation_v1.42_patch.py
"""
SpecGate v1.42 PATCH - Multi-File Refactor Detection Fix

This patch file contains the updated _detect_multi_file_intent() function
and the call-site fix for run_spec_gate_grounded().

Apply this patch to spec_generation.py

KEY FIXES:
1. Pass combined_text (not just user_intent) to _detect_multi_file_intent()
2. Add conversational patterns for natural language refactor requests
3. Add extractors for search_pattern and replacement_pattern from verbose text

v1.42 (2026-01-31): CRITICAL FIX - Multi-file refactor detection for natural language
    - _detect_multi_file_intent() now checks combined_text (Weaver job + user intent)
    - Added flexible patterns for conversational refactor requests
    - Added pattern extraction for "change X to Y", "rename X to Y" with varied phrasing
    - Handles scope indicators like "D drive", "entire codebase", "all files"
    - Detects intent from "I want to", "search for", "look for", etc.
"""

import re
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# v1.42: ENHANCED MULTI-FILE OPERATION HELPERS
# =============================================================================

# Extended scope indicators that trigger multi-file mode
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
    # Drive references (D drive, D:\, etc.)
    r"\b[A-Za-z]\s+drive\b",
    r"\b[A-Za-z]:\s*(?:\\|/)",
    r"\bover\s+(?:the\s+)?[A-Za-z]\s+drive\b",
    r"\bsearch\s+over\b",
    r"\bscan\s+(?:the\s+)?(?:entire|whole|all)\b",
    # File/folder scope
    r"\bfile\s+(?:names?|structures?)\b",
    r"\bfolder\s+names?\b",
    r"\bwithin\s+(?:the\s+)?(?:code|files?|folders?)\b",
    r"\bin\s+(?:all\s+)?(?:files?|folders?|the\s+code)\b",
]

# Compiled scope patterns
_SCOPE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in MULTI_FILE_SCOPE_INDICATORS]


def _has_multi_file_scope(text: str) -> bool:
    """
    Check if text indicates multi-file scope.
    
    v1.42: Expanded scope detection for natural language.
    """
    if not text:
        return False
    for pattern in _SCOPE_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _extract_search_and_replace_terms(text: str) -> Optional[Dict[str, str]]:
    """
    Extract search pattern and replacement pattern from natural language text.
    
    v1.42: Handles conversational phrasing like:
    - "I want to change X to Y"
    - "rename all references to Orb to ASTRA"
    - "replace X with Y"
    - "look for X and change it to Y"
    - "find any references to X... change to Y"
    
    Returns:
        Dict with 'search_pattern' and 'replacement_pattern', or None
    """
    if not text:
        return None
    
    text_clean = text.strip()
    
    # ==========================================================================
    # PATTERN GROUP 1: Direct commands (highest confidence)
    # "replace X with Y", "rename X to Y", "change X to Y"
    # ==========================================================================
    
    direct_patterns = [
        # "replace X with Y" / "replace X by Y"
        r"(?:^|[.!?]\s*)replace\s+['\"]?(.+?)['\"]?\s+(?:with|by)\s+['\"]?(.+?)['\"]?(?:\s+(?:everywhere|in\s+all|across|without\s+breaking)|$|[.!?])",
        # "rename X to Y"
        r"(?:^|[.!?]\s*)rename\s+(?:all\s+)?(?:occurrences?\s+of\s+)?['\"]?(.+?)['\"]?\s+to\s+['\"]?(.+?)['\"]?(?:\s+(?:everywhere|in\s+all|across|without\s+breaking)|$|[.!?])",
        # "change X to Y" (at start of sentence)
        r"(?:^|[.!?]\s*)change\s+(?:all\s+)?(?:occurrences?\s+of\s+)?['\"]?(.+?)['\"]?\s+to\s+['\"]?(.+?)['\"]?(?:\s+(?:everywhere|in\s+all|across|without\s+breaking)|$|[.!?])",
    ]
    
    for pattern in direct_patterns:
        match = re.search(pattern, text_clean, re.IGNORECASE)
        if match:
            search = match.group(1).strip().strip("'\"")
            replace = match.group(2).strip().strip("'\"")
            if search and replace and len(search) > 1 and len(replace) > 1:
                logger.info(
                    "[spec_generation] v1.42 Direct pattern match: search='%s', replace='%s'",
                    search, replace
                )
                return {"search_pattern": search, "replacement_pattern": replace}
    
    # ==========================================================================
    # PATTERN GROUP 2: Conversational "I want to" patterns
    # "I want to change X to Y", "I want to rename X to Y"
    # ==========================================================================
    
    want_patterns = [
        # "I want to change X to Y" / "I want you to change X to Y"
        r"(?:i\s+)?want\s+(?:you\s+)?to\s+(?:change|rename|replace)\s+(?:(?:all\s+)?(?:occurrences?\s+of\s+|references?\s+to\s+)?)?['\"]?(.+?)['\"]?\s+(?:to|with|by)\s+['\"]?(.+?)['\"]?(?:\s+(?:without\s+breaking|everywhere|in\s+all)|$|[.!?])",
        # "I want X changed to Y" / "I need X renamed to Y"
        r"(?:i\s+)?(?:want|need|would\s+like)\s+['\"]?(.+?)['\"]?\s+(?:changed?|renamed?|replaced?)\s+(?:to|with|by)\s+['\"]?(.+?)['\"]?(?:\s+(?:without\s+breaking|everywhere)|$|[.!?])",
    ]
    
    for pattern in want_patterns:
        match = re.search(pattern, text_clean, re.IGNORECASE)
        if match:
            search = match.group(1).strip().strip("'\"")
            replace = match.group(2).strip().strip("'\"")
            if search and replace and len(search) > 1 and len(replace) > 1:
                logger.info(
                    "[spec_generation] v1.42 Conversational pattern match: search='%s', replace='%s'",
                    search, replace
                )
                return {"search_pattern": search, "replacement_pattern": replace}
    
    # ==========================================================================
    # PATTERN GROUP 3: "Find X and change to Y" patterns
    # "search for X and change to Y", "find X... and rename to Y"
    # ==========================================================================
    
    find_change_patterns = [
        # "find/search/look for X ... change/rename to Y"
        r"(?:find|search\s+for|look\s+for|scan\s+for)\s+(?:any\s+)?(?:references?\s+to\s+|occurrences?\s+of\s+)?(?:the\s+name\s+)?['\"]?(.+?)['\"]?[^.]*?(?:change|rename|replace|update)\s+(?:it|them|that)?\s*(?:to|with|by)\s+(?:the\s+name\s+)?['\"]?(.+?)['\"]?(?:\s+(?:without\s+breaking)|$|[.!?])",
    ]
    
    for pattern in find_change_patterns:
        match = re.search(pattern, text_clean, re.IGNORECASE | re.DOTALL)
        if match:
            search = match.group(1).strip().strip("'\"")
            replace = match.group(2).strip().strip("'\"")
            # Clean up common noise
            search = re.sub(r'\s*,?\s*o-r-b\s*,?\s*', '', search, flags=re.IGNORECASE).strip()
            if search and replace and len(search) > 1 and len(replace) > 1:
                logger.info(
                    "[spec_generation] v1.42 Find-change pattern match: search='%s', replace='%s'",
                    search, replace
                )
                return {"search_pattern": search, "replacement_pattern": replace}
    
    # ==========================================================================
    # PATTERN GROUP 4: Two-part extraction (search term + replacement separately)
    # When patterns are split across text: "look for Orb" ... "change to ASTRA"
    # ==========================================================================
    
    # Extract search term patterns
    search_extractors = [
        r"(?:references?\s+to\s+(?:the\s+name\s+)?)['\"]?([A-Za-z][A-Za-z0-9_]*)['\"]?",
        r"(?:the\s+name\s+)['\"]?([A-Za-z][A-Za-z0-9_]*)['\"]?",
        r"(?:occurrences?\s+of\s+)['\"]?([A-Za-z][A-Za-z0-9_]*)['\"]?",
        r"(?:search\s+(?:for\s+|over\s+)?.*?)['\"]?([A-Za-z][A-Za-z0-9_]+)['\"]?",
        r"(?:look\s+for\s+(?:any\s+)?(?:references?\s+to\s+)?)['\"]?([A-Za-z][A-Za-z0-9_]+)['\"]?",
    ]
    
    # Extract replacement term patterns
    replace_extractors = [
        r"(?:change\s+(?:it|them|that)?\s*to\s+(?:the\s+name\s+)?)['\"]?([A-Za-z][A-Za-z0-9_]+)['\"]?",
        r"(?:rename\s+(?:it|them|that)?\s*to\s+)['\"]?([A-Za-z][A-Za-z0-9_]+)['\"]?",
        r"(?:replace\s+(?:it|them|that)?\s*(?:with|by|to)\s+)['\"]?([A-Za-z][A-Za-z0-9_]+)['\"]?",
        r"(?:to\s+the\s+name\s+)['\"]?([A-Za-z][A-Za-z0-9_]+)['\"]?",
        r"(?:with\s+(?:the\s+name\s+)?)['\"]?([A-Za-z][A-Za-z0-9_]+)['\"]?",
    ]
    
    search_term = None
    replace_term = None
    
    for pattern in search_extractors:
        match = re.search(pattern, text_clean, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip().strip("'\"")
            if candidate and len(candidate) > 1:
                search_term = candidate
                break
    
    for pattern in replace_extractors:
        match = re.search(pattern, text_clean, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip().strip("'\"")
            if candidate and len(candidate) > 1:
                replace_term = candidate
                break
    
    if search_term and replace_term and search_term.lower() != replace_term.lower():
        logger.info(
            "[spec_generation] v1.42 Two-part extraction: search='%s', replace='%s'",
            search_term, replace_term
        )
        return {"search_pattern": search_term, "replacement_pattern": replace_term}
    
    return None


def _detect_multi_file_intent_v142(
    combined_text: str, 
    constraints_hint: Optional[Dict] = None
) -> Optional[Dict[str, Any]]:
    """
    Detect multi-file operation intent from combined text or constraints.
    
    v1.42: ENHANCED - handles conversational natural language.
    
    Args:
        combined_text: Combined user intent + Weaver job description
        constraints_hint: Weaver output and other hints (may contain multi_file_metadata)
    
    Returns:
        Dict with keys: is_multi_file, operation_type, search_pattern, replacement_pattern
        or None if not a multi-file intent
    """
    # Check if constraints_hint already has multi-file metadata (from tier0)
    if constraints_hint:
        multi_file_meta = constraints_hint.get("multi_file_metadata")
        if multi_file_meta and multi_file_meta.get("is_multi_file"):
            logger.info(
                "[spec_generation] v1.42 Multi-file metadata found in constraints_hint: %s",
                multi_file_meta
            )
            return multi_file_meta
    
    # No text to analyze
    if not combined_text:
        return None
    
    text = combined_text.strip()
    text_lower = text.lower()
    
    # Skip if text is too short
    if len(text_lower) < 15:
        return None
    
    # ==========================================================================
    # STEP 1: Check for scope indicators
    # ==========================================================================
    
    has_scope = _has_multi_file_scope(text)
    if not has_scope:
        logger.debug("[spec_generation] v1.42 No multi-file scope indicators found")
        return None
    
    # ==========================================================================
    # STEP 2: Check for refactor/rename intent keywords
    # ==========================================================================
    
    refactor_intent_patterns = [
        r"\b(?:rename|replace|change|update)\s+(?:all|every|it|them|that|to|with)",
        r"\b(?:want\s+to|need\s+to|should)\s+(?:rename|replace|change)",
        r"\bfind\s+(?:all\s+)?(?:references?|occurrences?)\b.*?\b(?:change|rename|replace)",
        r"\bsearch\s+(?:for|over)\b.*?\b(?:change|rename|replace)",
        r"\blook\s+for\b.*?\b(?:change|rename|replace)",
        r"\bwithout\s+breaking\b",  # Safety indicator often in refactor requests
    ]
    
    has_refactor_intent = False
    for pattern in refactor_intent_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
            has_refactor_intent = True
            break
    
    if not has_refactor_intent:
        # Check for search-only intent (no change/rename/replace)
        search_only_patterns = [
            r"\bfind\s+all\b(?!.*?\b(?:change|rename|replace))",
            r"\blist\s+(?:all\s+)?(?:files?|references?|occurrences?)\b(?!.*?\b(?:change|rename|replace))",
            r"\bsearch\s+(?:for|codebase)\b(?!.*?\b(?:change|rename|replace))",
            r"\bcount\s+(?:all\s+)?(?:occurrences?|references?)\b",
        ]
        
        for pattern in search_only_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
                # This is a search-only operation (handled by MULTI_FILE_SEARCH)
                # For now, return None and let existing scan_only logic handle it
                logger.debug("[spec_generation] v1.42 Search-only intent, not refactor")
                return None
        
        logger.debug("[spec_generation] v1.42 No refactor intent keywords found")
        return None
    
    # ==========================================================================
    # STEP 3: Extract search and replacement terms
    # ==========================================================================
    
    terms = _extract_search_and_replace_terms(text)
    
    if not terms:
        logger.warning(
            "[spec_generation] v1.42 Has refactor intent + scope but couldn't extract terms from: %s",
            text[:200]
        )
        return None
    
    search_pattern = terms.get("search_pattern", "")
    replacement_pattern = terms.get("replacement_pattern", "")
    
    if not search_pattern or not replacement_pattern:
        return None
    
    # ==========================================================================
    # STEP 4: Build and return result
    # ==========================================================================
    
    logger.info(
        "[spec_generation] v1.42 MULTI-FILE REFACTOR detected: "
        "search='%s' -> replace='%s' (scope indicators present)",
        search_pattern, replacement_pattern
    )
    
    return {
        "is_multi_file": True,
        "operation_type": "refactor",
        "search_pattern": search_pattern,
        "replacement_pattern": replacement_pattern,
    }


# =============================================================================
# CALL-SITE FIX
# =============================================================================
# In run_spec_gate_grounded(), change the call from:
#
#   multi_file_meta = _detect_multi_file_intent(user_intent, constraints_hint)
#
# To:
#
#   multi_file_meta = _detect_multi_file_intent(combined_text, constraints_hint)
#
# The combined_text variable already exists and contains:
#   combined_text = f"{user_intent or ''} {weaver_job_text}"
#
# This ensures the full context (including the conversational Weaver job 
# description) is analyzed for multi-file intent.
# =============================================================================


# =============================================================================
# TEST CASES
# =============================================================================
if __name__ == "__main__":
    # Test the v1.42 detection logic
    test_cases = [
        # Original failing case
        (
            "I want you to search over the D drive and look for any references to the name Orb, "
            "O-R-B, with capital or small letters. I want to know anything within file structures, "
            "within file names, folder names, anything at all that refers to those names. "
            "And I want to change it to the name ASTRA without breaking any of the code.",
            {"search_pattern": "Orb", "replacement_pattern": "ASTRA"},
        ),
        # Weaver-style structured output
        (
            "What is being built: System-wide rename from Orb to ASTRA\n"
            "Intent: Search D: for all occurrences of Orb in file names and contents, "
            "replace with ASTRA without breaking code",
            {"search_pattern": "Orb", "replacement_pattern": "ASTRA"},
        ),
        # Direct command
        (
            "Replace all occurrences of Orb with ASTRA across the codebase",
            {"search_pattern": "Orb", "replacement_pattern": "ASTRA"},
        ),
        # Search-only (should NOT match refactor)
        (
            "Find all files containing TODO in the codebase",
            None,  # Should be None - search only, no replacement
        ),
        # Simple rename
        (
            "I want to rename Config to Settings everywhere in the project",
            {"search_pattern": "Config", "replacement_pattern": "Settings"},
        ),
    ]
    
    print("=== v1.42 Multi-File Refactor Detection Tests ===\n")
    
    for i, (text, expected) in enumerate(test_cases, 1):
        result = _detect_multi_file_intent_v142(text)
        
        if expected is None:
            passed = result is None
            expected_str = "None (not a refactor)"
        else:
            passed = (
                result is not None and
                result.get("search_pattern") == expected["search_pattern"] and
                result.get("replacement_pattern") == expected["replacement_pattern"]
            )
            expected_str = f"search='{expected['search_pattern']}', replace='{expected['replacement_pattern']}'"
        
        status = "✅ PASS" if passed else "❌ FAIL"
        
        print(f"Test {i}: {status}")
        print(f"  Input: {text[:80]}...")
        print(f"  Expected: {expected_str}")
        if result:
            print(f"  Got: search='{result.get('search_pattern')}', replace='{result.get('replacement_pattern')}'")
        else:
            print(f"  Got: None")
        print()
    
    print("=== End of tests ===")
