# FILE: app/translation/_tier0_filesystem.py
"""
Tier 0 filesystem, multi-file, refactor, and chat pattern rules.

Extracted from tier0_rules.py to reduce file size.
Contains: check_multi_file_trigger, check_refactor_codebase_trigger,
check_filesystem_query_trigger, is_user_chat_pattern, and supporting constants.
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

# Refactor patterns (write): "replace X with Y everywhere"
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

# Keywords that indicate multi-file scope (must be present for ambiguous patterns)
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


# =============================================================================
# REFACTOR CODEBASE (v1.9 - Self-refactoring loop)
# =============================================================================

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


# =============================================================================
# FILESYSTEM QUERY HANDLER (v5.8 - ORB PREFIX SUPPORT)
# =============================================================================

# Known user folder keywords (for queries that don't have explicit paths)
_KNOWN_FOLDER_KEYWORDS = {
    "desktop", "onedrive", "documents", "downloads", 
    "pictures", "videos", "music", "appdata",
}

# Allowed scan roots - reject queries outside these
_ALLOWED_FS_ROOTS = [
    r"D:\\",
    r"C:\\Users\\dizzi",
]


def _has_windows_path(text: str) -> bool:
    """Check if text contains a Windows path like C:\\ or D:\\."""
    return bool(re.search(r'[A-Za-z]:[/\\]', text))


def _has_known_folder_keyword(text: str) -> bool:
    """Check if text contains a known user folder keyword."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in _KNOWN_FOLDER_KEYWORDS)


def _is_within_allowed_roots(text: str) -> bool:
    """
    Check if query references a path within allowed roots.
    
    Allowed: D:\\ and C:\\Users\\dizzi
    Reject: C:\\Windows, C:\\Program Files, etc.
    """
    text_lower = text.lower()
    
    # If contains C:\ but NOT C:\Users\dizzi, reject
    if re.search(r'c:[/\\]', text_lower):
        if not re.search(r'c:[/\\]users[/\\]dizzi', text_lower):
            return False
    
    # D:\ is always allowed
    # Known folder keywords (Desktop, OneDrive) are under C:\Users\dizzi
    return True


def check_filesystem_query_trigger(text: str) -> Tier0RuleResult:
    """
    Special handler for filesystem listing/search queries (v5.8 - ORB PREFIX SUPPORT).
    
    v5.8 (2026-01): Added "Orb, command:" prefix support alongside "Astra, command:"
    v5.7 (2026-01): Fixed corrupted regex, removed duplicated code blocks
    
    v5.6 (2026-01): Fixed to handle text with command prefix already stripped
      - Also matches raw verbs without "command:" prefix
      - Handles case where translator.py strips prefix before calling
    
    v5.0: Added explicit command mode support:
    - "Astra, command: list <path>" or "Orb, command: list <path>"
    - "Astra, command: read <path>" or "Orb, command: read <path>"
    - "Astra, command: head <path> <n>"
    - "Astra, command: lines <path> <start> <end>"
    - "Astra, command: find <term> [under <path>]"
    
    Natural language queries:
    - "List everything on C:\\Users\\dizzi\\Desktop"
    - "What's in C:\\Users\\dizzi\\OneDrive"
    - "Find folder named Jobs under C:\\Users\\dizzi"
    - "Show me line 45-65 of stream_router.py"
    - "First 10 lines of main.py"
    
    Requirements:
    - Must have Windows path OR known folder keyword
    - Must be within allowed roots
    """
    text_stripped = text.strip()
    
    # FS command verbs - used in multiple checks below
    fs_command_verbs = [
        r'^list\s+',           # list <path>
        r'^read\s+',           # read <path> or read "<path>"
        r'^head\s+',           # head <path> [n]
        r'^lines\s+',          # lines <path> <start> <end>
        r'^find\s+',           # find <term> [under <path>]
        # Stage 1 write commands (v5.5)
        r'^append\s+',         # append <path> "content" or append <path> + fenced block
        r'^overwrite\s+',      # overwrite <path> "content" or overwrite <path> + fenced block
        r'^delete_area\s+',    # delete_area <path> (uses ASTRA_BLOCK markers)
        r'^delete_lines\s+',   # delete_lines <path> <start> <end>
    ]
    
    # ==========================================================================
    # v5.6: DIRECT COMMAND VERB DETECTION (highest priority)
    # When text is already stripped of "Astra, command:" prefix, it starts
    # directly with the verb like "append path content" or "read path"
    # ==========================================================================
    text_lower = text_stripped.lower()
    for pattern in fs_command_verbs:
        if re.match(pattern, text_lower):
            verb = text_lower.split()[0]
            print(f"[FILESYSTEM_QUERY] Direct verb detected: {text_stripped[:80]}")
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.FILESYSTEM_QUERY,
                confidence=1.0,
                rule_name="filesystem_direct_verb",
                reason=f"Filesystem command (direct verb): {verb}",
            )
    
    # ==========================================================================
    # v5.8: EXPLICIT COMMAND MODE DETECTION (with prefix) - SUPPORTS ORB + ASTRA
    # Patterns: "Astra, command: <cmd> <args>" or "Orb, command: <cmd> <args>"
    # ==========================================================================
    
    # Check for command mode prefix (v5.8: now accepts orb|astra)
    command_match = re.match(
        r'^(?:(?:astra|orb),?\s*)?command:?\s*(.+)$',
        text_stripped, re.IGNORECASE
    )
    
    if command_match:
        cmd_text = command_match.group(1).strip()
        cmd_lower = cmd_text.lower()
        
        # Check for FS commands
        for pattern in fs_command_verbs:
            if re.match(pattern, cmd_lower):
                print(f"[FILESYSTEM_QUERY] Command mode detected: {cmd_text[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=1.0,
                    rule_name="filesystem_command_mode",
                    reason=f"Filesystem command: {cmd_text.split()[0]}",
                )
    
    # ==========================================================================
    # v5.0: LINE RANGE QUERIES (natural language)
    # "Show me line 45-65 of...", "What's on lines 10-20..."
    # ==========================================================================
    
    line_range_patterns = [
        r'lines?\s+\d+\s*[-\u2013\u2014to]+\s*\d+\s+(?:of|in|from)\s+',
        r'(?:show|what\'?s)\s+(?:me\s+)?(?:on\s+)?lines?\s+\d+',
        r'first\s+\d+\s+lines?\s+(?:of|in|from)\s+',
        r'head\s+\d+\s+(?:of|in|from)\s+',
    ]
    
    text_lower_raw = text_stripped.lower()
    for pattern in line_range_patterns:
        if re.search(pattern, text_lower_raw):
            # Must have a path reference
            if _has_windows_path(text_stripped) or _has_known_folder_keyword(text_lower_raw):
                print(f"[FILESYSTEM_QUERY] Line range query: {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=1.0,
                    rule_name="filesystem_line_range",
                    reason="Filesystem line range query",
                )
    
    # Strip optional "After scan sandbox, " prefix
    text_for_match = re.sub(
        r'^[Aa]fter\s+scan\s+sandbox,?\s*',
        '', text_stripped
    ).strip()
    
    text_lower = text_for_match.lower()
    
    # ==========================================================================
    # TIGHT PATTERNS - each requires Windows path context
    # ==========================================================================
    
    # Pattern 1: "List everything/all/contents/files/folders on/in/at <path>"
    # Requires Windows path
    list_with_path_patterns = [
        r"^list\s+(?:everything|all|contents?|files?(?:\s+and\s+folders?)?|folders?(?:\s+and\s+files?)?|top[- ]?level\s+folders?)\s+(?:on|in|at|under|inside)\s+[A-Za-z]:[/\\]",
    ]
    
    for pattern in list_with_path_patterns:
        if re.match(pattern, text_lower):
            if _is_within_allowed_roots(text_lower):
                print(f"[FILESYSTEM_QUERY] Detected list with path: {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=1.0,
                    rule_name="filesystem_list_with_path",
                    reason=f"Filesystem list query with Windows path",
                )
    
    # Pattern 2: "What's in / What is in <path>"
    # Requires Windows path
    whats_in_patterns = [
        r"^what(?:'s|\s+is)\s+(?:in|on|at|inside)\s+[A-Za-z]:[/\\]",
    ]
    
    for pattern in whats_in_patterns:
        if re.match(pattern, text_lower):
            if _is_within_allowed_roots(text_lower):
                print(f"[FILESYSTEM_QUERY] Detected what's in: {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=1.0,
                    rule_name="filesystem_whats_in",
                    reason=f"Filesystem 'what's in' query with Windows path",
                )
    
    # Pattern 3: "Show me everything/contents in <path>"
    # Requires Windows path
    show_patterns = [
        r"^show\s+(?:me\s+)?(?:everything|all|contents?|files?|folders?)\s+(?:in|on|at|under|inside)\s+[A-Za-z]:[/\\]",
    ]
    
    for pattern in show_patterns:
        if re.match(pattern, text_lower):
            if _is_within_allowed_roots(text_lower):
                print(f"[FILESYSTEM_QUERY] Detected show: {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=1.0,
                    rule_name="filesystem_show",
                    reason=f"Filesystem show query with Windows path",
                )
    
    # Pattern 4: "Contents of <path>"
    # Requires Windows path
    contents_of_patterns = [
        r"^contents?\s+of\s+[A-Za-z]:[/\\]",
    ]
    
    for pattern in contents_of_patterns:
        if re.match(pattern, text_lower):
            if _is_within_allowed_roots(text_lower):
                print(f"[FILESYSTEM_QUERY] Detected contents of: {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=1.0,
                    rule_name="filesystem_contents_of",
                    reason=f"Filesystem 'contents of' query with Windows path",
                )
    
    # Pattern 5: "Find folder/file/directory named X under/in <path>"
    # Requires Windows path
    find_named_with_path_patterns = [
        r"^find\s+(?:folder|file|directory)\s+(?:named?\s+)?[\w\s]+\s+(?:under|in|on|inside)\s+[A-Za-z]:[/\\]",
    ]
    
    for pattern in find_named_with_path_patterns:
        if re.match(pattern, text_lower):
            if _is_within_allowed_roots(text_lower):
                print(f"[FILESYSTEM_QUERY] Detected find named with path: {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=1.0,
                    rule_name="filesystem_find_named_path",
                    reason=f"Filesystem find query with Windows path",
                )
    
    # Pattern 6: "Find X under/in <known folder>"
    # Requires known folder keyword (Desktop, OneDrive, etc.)
    find_under_folder_patterns = [
        r"^find\s+[\w\s]+\s+(?:under|in|inside|on)\s+(?:my\s+)?(?:desktop|onedrive|documents|downloads)",
    ]
    
    for pattern in find_under_folder_patterns:
        if re.match(pattern, text_lower):
            print(f"[FILESYSTEM_QUERY] Detected find under known folder: {text_stripped[:80]}")
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.FILESYSTEM_QUERY,
                confidence=0.98,
                rule_name="filesystem_find_under_folder",
                reason=f"Filesystem find query with known folder keyword",
            )
    
    # Pattern 7: "Find files with X in the name" 
    # ONLY if also has Windows path OR known folder keyword
    find_files_with_patterns = [
        r"^find\s+files?\s+(?:with|named|containing)\s+.+\s+(?:in\s+(?:the\s+)?name|in\s+their\s+name)",
    ]
    
    for pattern in find_files_with_patterns:
        if re.match(pattern, text_lower):
            # Must have path OR folder keyword to avoid false positives
            if _has_windows_path(text_lower) or _has_known_folder_keyword(text_lower):
                if _is_within_allowed_roots(text_lower):
                    print(f"[FILESYSTEM_QUERY] Detected find files with pattern: {text_stripped[:80]}")
                    return Tier0RuleResult(
                        matched=True,
                        intent=CanonicalIntent.FILESYSTEM_QUERY,
                        confidence=0.95,
                        rule_name="filesystem_find_files_pattern",
                        reason=f"Filesystem find files query",
                    )
    
    # Pattern 8: Generic "find files with X" - requires Windows path explicitly in query
    generic_find_with_path = [
        r"^find\s+(?:all\s+)?files?\s+(?:with|containing|named)\s+.+\s+(?:under|in|on|inside)\s+[A-Za-z]:[/\\]",
    ]
    
    for pattern in generic_find_with_path:
        if re.match(pattern, text_lower):
            if _is_within_allowed_roots(text_lower):
                print(f"[FILESYSTEM_QUERY] Detected generic find with path: {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=1.0,
                    rule_name="filesystem_generic_find_path",
                    reason=f"Filesystem find query with explicit path",
                )
    
    # Pattern 9: List with known folder keyword (no explicit path)
    # "List everything in my Desktop", "List folders in OneDrive"
    list_with_folder_keyword = [
        r"^list\s+(?:everything|all|contents?|files?(?:\s+and\s+folders?)?|folders?(?:\s+and\s+files?)?|top[- ]?level\s+folders?)\s+(?:on|in|at|under|inside)\s+(?:my\s+)?(?:desktop|onedrive|documents|downloads)",
    ]
    
    for pattern in list_with_folder_keyword:
        if re.match(pattern, text_lower):
            print(f"[FILESYSTEM_QUERY] Detected list with folder keyword: {text_stripped[:80]}")
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.FILESYSTEM_QUERY,
                confidence=0.98,
                rule_name="filesystem_list_folder_keyword",
                reason=f"Filesystem list query with known folder keyword",
            )
    
    # Pattern 10: "What's in my Desktop/OneDrive" (no explicit path)
    whats_in_folder_keyword = [
        r"^what(?:'s|\s+is)\s+(?:in|on)\s+(?:my\s+)?(?:desktop|onedrive|documents|downloads)",
    ]
    
    for pattern in whats_in_folder_keyword:
        if re.match(pattern, text_lower):
            print(f"[FILESYSTEM_QUERY] Detected what's in folder keyword: {text_stripped[:80]}")
            return Tier0RuleResult(
                matched=True,
                intent=CanonicalIntent.FILESYSTEM_QUERY,
                confidence=0.98,
                rule_name="filesystem_whats_in_folder",
                reason=f"Filesystem 'what's in' query with known folder keyword",
            )
    
    # ==========================================================================
    # v1.6: READ FILE PATTERNS (from zobie_tools.py _parse_filesystem_query)
    # Routes file read queries to FILESYSTEM_QUERY for DB-first content lookup
    # Handles apostrophe variants: straight ' and curly '
    # ==========================================================================
    
    # Normalize curly apostrophes to straight for consistent matching
    text_normalized = text_lower.replace("'", "'").replace("'", "'")
    
    # Pattern 11: "what's written in <path>" / "whats written in <path>" / "whats inside <path>"
    # Requires Windows path
    whats_written_patterns = [
        r"^what'?s\s+(?:written|inside)\s+(?:in\s+)?[A-Za-z]:[/\\]",
    ]
    
    for pattern in whats_written_patterns:
        if re.match(pattern, text_normalized):
            if _is_within_allowed_roots(text_lower):
                print(f"[FILESYSTEM_QUERY] Detected READ (what's written): {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=1.0,
                    rule_name="filesystem_read_whats_written",
                    reason="Filesystem READ query: 'what's written in'",
                )
    
    # Pattern 12: "read <path>" / "read file <path>" / "read the file <path>"
    # Requires Windows path
    read_file_patterns = [
        r"^read\s+(?:the\s+)?(?:file\s+)?[A-Za-z]:[/\\]",
    ]
    
    for pattern in read_file_patterns:
        if re.match(pattern, text_normalized):
            if _is_within_allowed_roots(text_lower):
                print(f"[FILESYSTEM_QUERY] Detected READ (read file): {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=1.0,
                    rule_name="filesystem_read_file",
                    reason="Filesystem READ query: 'read file'",
                )
    
    # Pattern 13: "show contents of <path>" / "show the contents of <path>"
    # Requires Windows path
    show_contents_of_patterns = [
        r"^show\s+(?:the\s+)?contents?\s+of\s+[A-Za-z]:[/\\]",
    ]
    
    for pattern in show_contents_of_patterns:
        if re.match(pattern, text_normalized):
            if _is_within_allowed_roots(text_lower):
                print(f"[FILESYSTEM_QUERY] Detected READ (show contents of): {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=1.0,
                    rule_name="filesystem_read_show_contents",
                    reason="Filesystem READ query: 'show contents of'",
                )
    
    # Pattern 14: "view/display/cat/open <path>" (Unix-style + generic)
    # Requires Windows path
    view_display_patterns = [
        r"^(?:view|display|cat|output|print)\s+(?:the\s+)?(?:file\s+)?[A-Za-z]:[/\\]",
    ]
    
    for pattern in view_display_patterns:
        if re.match(pattern, text_normalized):
            if _is_within_allowed_roots(text_lower):
                print(f"[FILESYSTEM_QUERY] Detected READ (view/display/cat): {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=1.0,
                    rule_name="filesystem_read_view_display",
                    reason="Filesystem READ query: 'view/display/cat'",
                )
    
    # Pattern 15: "open <path>" - only if path looks like a file (has extension)
    # Requires Windows path with file extension
    open_file_patterns = [
        r"^open\s+(?:the\s+)?(?:file\s+)?[A-Za-z]:[/\\].+\.\w+",
    ]
    
    for pattern in open_file_patterns:
        if re.match(pattern, text_normalized):
            if _is_within_allowed_roots(text_lower):
                print(f"[FILESYSTEM_QUERY] Detected READ (open file): {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=0.98,
                    rule_name="filesystem_read_open",
                    reason="Filesystem READ query: 'open file'",
                )
    
    # Pattern 16: "what does <path> say/contain" 
    # Requires Windows path
    what_does_say_patterns = [
        r"^what\s+does\s+[A-Za-z]:[/\\].+\s+(?:say|contain)",
    ]
    
    for pattern in what_does_say_patterns:
        if re.match(pattern, text_normalized):
            if _is_within_allowed_roots(text_lower):
                print(f"[FILESYSTEM_QUERY] Detected READ (what does say/contain): {text_stripped[:80]}")
                return Tier0RuleResult(
                    matched=True,
                    intent=CanonicalIntent.FILESYSTEM_QUERY,
                    confidence=0.98,
                    rule_name="filesystem_read_what_does",
                    reason="Filesystem READ query: 'what does X say/contain'",
                )
    
    return Tier0RuleResult(matched=False)


# =============================================================================
# USER CHAT PATTERN LIBRARY
# =============================================================================
# Common user chat patterns that should be learned over time.
# These are seed patterns that can be augmented by the phrase cache.

USER_CHAT_PATTERNS = [
    # Exploratory questions about the system
    r"tell me (?:about|more about)",
    r"what (?:is|are) your",
    r"describe your",
    r"explain (?:your|the|how)",
    r"show me (?:your|the|how)",
    r"how does (?:your|the)",
    r"what does (?:your|the|this|that|it)",
    
    # Casual conversation
    r"^(?:hi|hello|hey|yo|sup)",
    r"^(?:thanks|thank you|cheers)",
    r"^(?:ok|okay|sure|got it|understood)",
    r"^(?:hmm|huh|interesting|cool|nice)",
    
    # Questions about capabilities
    r"can you (?:tell|show|explain)",
    r"do you (?:have|know|understand)",
    r"are you (?:able|capable)",
]

# Compiled for efficiency
_COMPILED_USER_CHAT = [re.compile(p, re.IGNORECASE) for p in USER_CHAT_PATTERNS]


def is_user_chat_pattern(text: str) -> bool:
    """
    Check if text matches known user chat patterns.
    Used for additional chat short-circuiting.
    """
    for pattern in _COMPILED_USER_CHAT:
        if pattern.search(text):
            return True
    return False


# Legacy alias for backwards compatibility
TAZISH_CHAT_PATTERNS = USER_CHAT_PATTERNS
_COMPILED_TAZISH_CHAT = _COMPILED_USER_CHAT
is_tazish_chat = is_user_chat_pattern
