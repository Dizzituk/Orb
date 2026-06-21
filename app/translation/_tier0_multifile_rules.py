# FILE: app/translation/_tier0_multifile_rules.py
# Purpose: Tier 0 multi-file search/refactor + refactor-codebase trigger classifiers.
# Called-by: app.translation._tier0_filesystem (shim)
# Depends-on: app.translation.schemas, app.translation.tier0_rules
# Last-renovated: 2026-06-21
"""
Tier 0 multi-file + refactor-codebase classifiers.

Split out of _tier0_filesystem.py (BATCH 4) verbatim. Detects multi-file
search/refactor operations and the self-refactor trigger.
"""
from __future__ import annotations
import re
from .schemas import CanonicalIntent
from .tier0_rules import Tier0RuleResult


MULTI_FILE_SEARCH_PATTERNS = [
    # "find all X" / "find all X in the codebase"
    (r"^find\s+all\s+(.+?)(?:\s+(?:in\s+)?(?:the\s+)?(?:codebase|repo|project|files?))?$", "find_all"),
    # "list files containing X" / "list all files with X"
    (r"^list\s+(?:all\s+)?files?\s+(?:containing|with|that\s+(?:have|contain))\s+(.+)$", "list_containing"),
    # "search for X in all files" / "search codebase for X"
    (r"^search\s+(?:for\s+)?(.+?)\s+(?:in\s+)?(?:all|every)\s+(?:files?|the\s+codebase)$", "search_all"),
    (r"^search\s+(?:the\s+)?codebase\s+for\s+(.+)$", "search_codebase"),
    # "show all X in the codebase" / "show me all X"
    (r"^show\s+(?:me\s+)?all\s+(.+?)\s+(?:in\s+)?(?:the\s+)?(?:codebase|repo|project)$", "show_all"),
    # "where is X used" / "find usages of X"
    (r"^(?:where\s+is|find\s+(?:all\s+)?usages?\s+of)\s+(.+?)(?:\s+(?:in\s+)?(?:the\s+)?codebase)?$", "find_usages"),
    # "count occurrences of X"
    (r"^count\s+(?:all\s+)?(?:occurrences?|instances?)\s+of\s+(.+)$", "count_occurrences"),
]


MULTI_FILE_REFACTOR_PATTERNS = [
    # "replace X with Y everywhere" / "replace X with Y in all files"
    (r"^replace\s+(.+?)\s+with\s+(.+?)(?:\s+(?:everywhere|(?:in\s+)?(?:all|every)\s+(?:files?|the\s+codebase)))?$", "replace"),
    # "change all X to Y" / "change X to Y everywhere"
    (r"^change\s+(?:all\s+)?(.+?)\s+to\s+(.+?)(?:\s+(?:everywhere|in\s+all\s+files?))?$", "change"),
    # "rename X to Y everywhere" / "rename X to Y across the codebase"
    (r"^rename\s+(.+?)\s+to\s+(.+?)(?:\s+(?:everywhere|(?:in\s+)?all\s+files?|across\s+the\s+codebase))?$", "rename"),
    # "remove all X from codebase" / "delete all X"
    (r"^(?:remove|delete)\s+(?:all\s+)?(.+?)(?:\s+from\s+(?:the\s+)?(?:codebase|all\s+files?))?$", "remove"),
    # "update all X to Y"
    (r"^update\s+(?:all\s+)?(.+?)\s+to\s+(.+?)(?:\s+(?:everywhere|in\s+all\s+files?))?$", "update"),
]


MULTI_FILE_SCOPE_KEYWORDS = [
    "all", "every", "everywhere", "codebase", "repo", "project",
    "all files", "every file", "across", "throughout",
]


def check_multi_file_trigger(text: str) -> Tier0RuleResult:
    """
    Detect multi-file operation patterns (v5.10 - Level 3).
    
    Returns:
        Tier0RuleResult with:
        - MULTI_FILE_SEARCH for read-only operations (find all X)
        - MULTI_FILE_REFACTOR for write operations (replace X with Y)
        - matched=False if no match
    
    Metadata stored in rule_name field includes:
        - operation_type: "search" or "refactor"
        - search_pattern: extracted pattern to find
        - replacement_pattern: extracted replacement (refactor only)
        - requires_confirmation: True for refactor operations
    
    Examples:
        "find all TODO comments" → MULTI_FILE_SEARCH
        "replace Orb with Astra everywhere" → MULTI_FILE_REFACTOR
        "remove all DEBUG = True from codebase" → MULTI_FILE_REFACTOR
    """
    text_lower = text.strip().lower()
    text_clean = text.strip()
    
    # Skip if text is too short or doesn't contain scope keywords for ambiguous patterns
    if len(text_lower) < 10:
        return Tier0RuleResult(matched=False)
    
    # ==========================================================================
    # Check REFACTOR patterns first (more specific, write operations)
    # ==========================================================================
    
    for pattern, pattern_type in MULTI_FILE_REFACTOR_PATTERNS:
        match = re.match(pattern, text_lower, re.IGNORECASE)
        if match:
            groups = match.groups()
            search_pattern = groups[0].strip() if len(groups) > 0 else ""
            replacement_pattern = groups[1].strip() if len(groups) > 1 else ""
            
            # For "remove" type, replacement is empty
            if pattern_type == "remove":
                replacement_pattern = ""
            
            # Require scope keyword for ambiguous patterns (single-word patterns)
            # e.g., "replace foo with bar" alone is ambiguous - needs "everywhere"
            has_scope = any(kw in text_lower for kw in MULTI_FILE_SCOPE_KEYWORDS)
            
            # More lenient for explicit multi-file verbs
            if not has_scope and pattern_type not in ["remove"]:
                # Check if the pattern explicitly mentions scope
                if not re.search(r'(?:everywhere|all\s+files?|codebase|across)', text_lower):
                    continue  # Skip this pattern, not clearly multi-file
            
            print(f"[MULTI_FILE] Detected REFACTOR ({pattern_type}): search='{search_pattern}', replace='{replacement_pattern}'")
            
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.MULTI_FILE_REFACTOR,
                confidence=0.98,
                rule_name=f"multi_file_refactor_{pattern_type}",
                reason=f"Multi-file refactor: {pattern_type} '{search_pattern}' → '{replacement_pattern}'",
            )
    
    # ==========================================================================
    # Check SEARCH patterns (read-only operations)
    # ==========================================================================
    
    for pattern, pattern_type in MULTI_FILE_SEARCH_PATTERNS:
        match = re.match(pattern, text_lower, re.IGNORECASE)
        if match:
            groups = match.groups()
            search_pattern = groups[0].strip() if len(groups) > 0 else ""
            
            # Require "all" or scope keyword to distinguish from single-file queries
            # "find TODO" → ambiguous (could be single file)
            # "find all TODO" → clearly multi-file
            has_scope = any(kw in text_lower for kw in MULTI_FILE_SCOPE_KEYWORDS)
            
            if not has_scope:
                # Skip patterns that don't clearly indicate multi-file scope
                continue
            
            print(f"[MULTI_FILE] Detected SEARCH ({pattern_type}): pattern='{search_pattern}'")
            
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.MULTI_FILE_SEARCH,
                confidence=0.98,
                rule_name=f"multi_file_search_{pattern_type}",
                reason=f"Multi-file search: {pattern_type} for '{search_pattern}'",
            )
    
    return Tier0RuleResult(matched=False)


_REFACTOR_PATTERNS = [
    r"^(?:[Aa]stra[,:]?\s+)?[Rr]efactor\s+(?:yourself|codebase)$",
    r"^[Ss]tart\s+(?:the\s+)?refactor(?:ing)?$",
    r"^\*refactor$",
    r"^[Rr]efactor\s+(?:the\s+)?codebase$",
]


_REFACTOR_EXACT_PHRASES = {
    "astra, refactor yourself",
    "refactor yourself",
    "refactor codebase",
    "start refactor",
    "*refactor",
    "astra, command: refactor codebase",
}


def check_refactor_codebase_trigger(text: str) -> Tier0RuleResult:
    """
    Check if the user is requesting codebase refactoring.
    
    Matches:
    - "Astra, refactor yourself"
    - "refactor codebase" 
    - "start refactor"
    - "*refactor"
    """
    text_lower = text.strip().lower()
    
    # Exact phrase match
    if text_lower in _REFACTOR_EXACT_PHRASES:
        return Tier0RuleResult(
            matched=True,
            intent=CanonicalIntent.REFACTOR_CODEBASE,
            confidence=1.0,
            rule_name="refactor_codebase_exact",
            reason=f"Exact match: '{text_lower}'",
        )
    
    # Regex pattern match
    for pattern in _REFACTOR_PATTERNS:
        if re.match(pattern, text.strip()):
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.REFACTOR_CODEBASE,
                confidence=1.0,
                rule_name="refactor_codebase_pattern",
                reason=f"Pattern match: '{pattern}'",
            )
    
    return Tier0RuleResult(matched=False)
