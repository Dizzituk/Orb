# FILE: app/pot_spec/grounded/sandbox_discovery.py
r"""
Sandbox Discovery and Output Mode Detection (v1.42)

Handles sandbox file discovery, output mode detection, and replacement text extraction.

Version Notes:
-------------
v1.42 (2026-01-30): CRITICAL FIX - CREATE vs READ target distinction
    - Added extract_create_targets() to identify files that should be CREATED, not searched for
    - Expanded SEPARATE_REPLY_FILE patterns in detect_output_mode() for natural language:
      - "create a new file on desktop and call it reply"
      - "in the reply file i want you to write"
      - "then create", "and create a", "make a file called"
    - Added regex patterns for complex file creation phrases
    - Fixes bug where "reply" file was being searched for instead of created
v1.39 (2026-01-30): CRITICAL FIX - Added quoted pattern without colon requirement
    - Weaver produces: "test2" file on D drive (quoted, but NO colon before "drive")
    - Previous v1.36 pattern required colon: "test2" on D: drive
    - Added pattern: [quotes]FILENAME[quotes] file on X drive (no colon)
    - Now extracts both "test" file on Desktop AND "test2" file on D drive
    - v1.38 unquoted patterns remain disabled (caused infinite hang)
v1.38 (2026-01-29): CRITICAL FIX - Added unquoted Weaver patterns for actual output format
    - Weaver actually produces: "the test file on Desktop" (unquoted, with "the" and "file")
    - Weaver actually produces: "and test2 on D: drive" (unquoted, with "and", no "file")
    - Added patterns: "the FILENAME file on LOCATION" for first target
    - Added patterns: "and FILENAME on LOCATION" for subsequent targets
    - Fixed group extraction logic to correctly handle new (filename, location) patterns
    - Now extracts BOTH quoted and unquoted multi-target requests from Weaver
v1.37 (2026-01-29): CRITICAL FIX - Added optional "file" word to v1.36 patterns
    - Weaver actually produces: "test" file on Desktop (not "test" on Desktop)
    - Weaver actually produces: "test2" file on D: drive (not "test2" on D: drive)
    - Fixed both reverse-order patterns to include optional (?:file\s+)?
    - Now correctly matches actual Weaver output format
v1.36 (2026-01-29): CRITICAL FIX - Added reverse-order Weaver patterns
    - Added pattern for 'test' on Desktop (actual Weaver format)
    - Added pattern for 'test2' on D: drive (actual Weaver format)
    - Tested and verified: extracts 2 targets, no false positives
    - Handles both orders: 'file' on location AND location 'file'
v1.35 (2026-01-29): CRITICAL FIX - Removed overly broad unquoted Weaver patterns
    - Removed unquoted drive pattern that was matching "D: file" → "file"
    - Removed unquoted desktop pattern that could match "Desktop and" → "and"
    - Now ONLY uses quoted Weaver patterns: "D: 'test2' file" and "Desktop 'test' file"
    - Prevents spurious extraction of words like "send", "micro_file_task", "none", "file"
    - Extraction should now be precise: only actual quoted filenames in Weaver output
v1.34 (2026-01-29): CRITICAL FIX - Added Weaver-output format patterns for multi-target extraction
    - Added patterns for Weaver format: "Desktop 'test' file" and "D: 'test2' file"
    - Supports both quoted ("D: 'test2' file") and unquoted ("D: test2 file") Weaver patterns
    - Fixed group structure handling: Weaver patterns have (location, filename) vs. standard (filename, location)
    - Handles "Find the Desktop 'test' file and the D: drive 'test2' file" from Weaver
    - Critical for multi-target requests to work after passing through Weaver layer
v1.33 (2026-01-29): CRITICAL BUG FIX - Context-aware multi-target extraction
    - Fixed extract_file_targets() matching common English words ("both", "their", "full", "platform")
    - Added context-aware patterns: only extract when preceded by action verbs ("read", "open")
    - Added quoted string support: "test" on D drive
    - Added file indicator patterns: "file called test"
    - Expanded FILENAME_STOPWORDS with commonly mistaken words
    - Added common_words safety check for additional protection
    - Now properly extracts ONLY actual file references, not random English words
v1.32 (2026-01-29): Multi-target file extraction for Level 2.5
    - Added extract_file_targets() for "read test on desktop and test2 on D drive"
    - Handles multiple anchors: desktop, documents, D:, C:, E:, etc.
    - Returns List[FileTarget] with individual anchors per file
v1.31 (2026-01-28): Unquoted text extraction for OVERWRITE_FULL
    - Added fallback patterns for unquoted replacement text
    - Handles: "overwrite with the message X", "replace with X", etc.
    - Quoted strings still take priority (fallback only)
v1.30 (2026-01-27): DEBUG prints for detect_output_mode troubleshooting
    - Added print statements to trace function entry and pattern matching
v1.29 (2026-01-27): Expanded REWRITE_IN_PLACE pattern matching
    - Added plural forms: "fill missing answers", "fill blank answers", "fill empty answers"
    - Added "fill missing" (without "the")
    - Added "leave those", "leave existing" (from user's "so you can leave those")
    - Added "just answer where", "answer where the answer is meant"
    - Added "format exactly the same", "format of the file exactly"
    - Added "25 questions", "25-question", "answer the 25"
v1.28 (2026-01-27): Fix OVERWRITE_FULL detection for "overwrite it" patterns
    - Added patterns: "overwrite it", "overwrite it with", "read and overwrite", etc.
v1.27 (2026-01-27): Better REWRITE_IN_PLACE detection for multi-question tasks
    - Added patterns: "answer the questions", "keep the format", "spot the trick"
    - User prompt "answer where the answer is meant to be placed" now triggers REWRITE
v1.24 (2026-01): OVERWRITE_FULL output mode
v1.16-v1.17 (2026-01): Output mode safety fixes, REWRITE_IN_PLACE
v1.13 (2026-01): Output mode detection
v1.10 (2026-01): Greenfield build fix - platform vs file context
v1.3 (2026-01): Initial sandbox discovery
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple
from enum import Enum

from .domain_detection import DOMAIN_KEYWORDS

logger = logging.getLogger(__name__)

# =============================================================================
# v1.35 BUILD VERIFICATION - Proves correct code is running
# v1.35: CRITICAL FIX - Removed overly broad unquoted Weaver patterns
# =============================================================================
SANDBOX_DISCOVERY_BUILD_ID = "2026-01-30-v1.43-weaver-output-format"
print(f"[SANDBOX_DISCOVERY_LOADED] BUILD_ID={SANDBOX_DISCOVERY_BUILD_ID}")
logger.info(f"[sandbox_discovery] Module loaded: BUILD_ID={SANDBOX_DISCOVERY_BUILD_ID}")


# =============================================================================
# OUTPUT MODE ENUM
# =============================================================================

class OutputMode(str, Enum):
    """Output mode for MICRO_FILE_TASK jobs."""
    APPEND_IN_PLACE = "append_in_place"
    REWRITE_IN_PLACE = "rewrite_in_place"
    SEPARATE_REPLY_FILE = "separate_reply_file"
    CHAT_ONLY = "chat_only"
    OVERWRITE_FULL = "overwrite_full"


# =============================================================================
# SANDBOX HINT EXTRACTION
# =============================================================================

# Comprehensive stopword list - these must NEVER be captured as subfolder names
SUBFOLDER_STOPWORDS = {
    "on", "in", "at", "to", "of", "for", "from", "with", "by",
    "the", "a", "an", "my", "your", "this", "that", "it",
    "folder", "file", "files", "directory", "dir",
    "ok", "okay", "yes", "no", "please", "thanks",
}

# v1.32: File name stopwords - should never be extracted as file names
FILENAME_STOPWORDS = {
    # Prepositions and articles
    "on", "in", "at", "to", "of", "for", "from", "with", "by",
    "the", "a", "an", "my", "your", "this", "that", "it",
    # Conjunctions
    "and", "or", "but", "so", "then", "also",
    # Action verbs (not filenames!)
    "read", "write", "open", "find", "get", "show", "display",
    "create", "delete", "remove", "move", "copy", "save", "load",
    # v1.41: Action verbs from Weaver outputs that caused false positives
    "concatenate", "combine", "merge", "aggregate", "collect",
    "output", "print", "return", "send", "insert", "append",
    # Generic words
    "file", "files", "folder", "folders", "directory", "drive",
    "desktop", "documents", "downloads",
    # v1.33: Common English words that got mistakenly extracted
    "both", "their", "full", "platform", "contents", "content",
    "all", "each", "every", "some", "any", "one", "into",
    "target", "named", "called", "labeled",
}


def extract_sandbox_hints(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract anchor and subfolder from Weaver/user text.
    
    v1.10: CRITICAL FIX - Distinguish between platform vs file context.
    v1.15: CRITICAL FIX - sandbox_file domain override.
    
    Returns:
        (anchor, subfolder) or (None, None) if no sandbox hints found
        
    Examples:
        "Desktop folder called test" → ("desktop", "test")
        "file on the desktop" → ("desktop", None)
        "Target platform: Desktop" → (None, None)  # v1.10: Platform choice
    """
    if not text:
        return None, None
    
    text_lower = text.lower()
    
    # v1.15: Check for sandbox_file domain keywords first
    sandbox_file_keywords = DOMAIN_KEYWORDS.get("sandbox_file", [])
    sandbox_keywords_found = [kw for kw in sandbox_file_keywords if kw in text_lower]
    has_sandbox_file_signals = len(sandbox_keywords_found) > 0
    
    if has_sandbox_file_signals:
        logger.info(
            "[sandbox_discovery] v1.15 extract_sandbox_hints: sandbox_file keywords detected: %s",
            sandbox_keywords_found[:5]
        )
    
    # v1.10: Platform context patterns (NOT file location)
    platform_context_patterns = [
        r"target\s+platform[:\s]+desktop",
        r"platform[:\s]+desktop",
        r"for\s+desktop",
        r"on\s+desktop[\s,]",
        r"desktop\s+app",
        r"desktop\s+game",
        r"desktop\s+version",
        r"desktop\s+platform",
        r"build.*for.*desktop",
        r"create.*for.*desktop",
        r"make.*for.*desktop",
    ]
    
    is_platform_context = any(re.search(p, text_lower) for p in platform_context_patterns)
    
    # v1.10: File context patterns
    file_context_patterns = [
        r"desktop\s+folder",
        r"folder\s+(?:on|in)\s+(?:the\s+)?desktop",
        r"file\s+(?:on|in)\s+(?:the\s+)?desktop",
        r"(?:read|find|open)\s+.*desktop",
        r"sandbox\s+desktop",
        r"in\s+the\s+desktop",
        r"from\s+(?:the\s+)?desktop",
        r"on\s+(?:the\s+)?desktop[,.]?\s+(?:there\s+)?(?:is|are)\s+(?:a\s+)?(?:folder|file)",
        r"(?:there\s+)?(?:is|are)\s+(?:a\s+)?(?:folder|file)\s+(?:on|in)\s+(?:the\s+)?desktop",
        r"desktop.*folder\s+(?:called|named)",
    ]
    
    is_file_context = any(re.search(p, text_lower) for p in file_context_patterns)
    
    # v1.15: Override for sandbox_file signals
    if has_sandbox_file_signals and not is_file_context:
        basic_file_indicators = [
            "folder", "file", "read", ".txt", ".md", ".py",
            "answer the question", "answer question", "missing question",
            "find the", "open the", "called", "named",
        ]
        has_basic_file_indicator = any(ind in text_lower for ind in basic_file_indicators)
        
        has_desktop_mention = "desktop" in text_lower
        has_documents_mention = "documents" in text_lower or "document" in text_lower
        
        if has_basic_file_indicator and (has_desktop_mention or has_documents_mention):
            is_file_context = True
            logger.info(
                "[sandbox_discovery] v1.15 OVERRIDE: sandbox_file keywords + basic indicators "
                "-> forcing is_file_context=True"
            )
    
    # v1.10: Skip if platform context but no file context
    if is_platform_context and not is_file_context:
        if not has_sandbox_file_signals:
            logger.info(
                "[sandbox_discovery] v1.10 extract_sandbox_hints: 'desktop' is PLATFORM context, not file location"
            )
            return None, None
        else:
            logger.warning(
                "[sandbox_discovery] v1.15 FORCING anchor extraction: sandbox_file keywords detected "
                "but is_file_context=False and is_platform_context=True"
            )
    
    # Detect anchor
    anchor = None
    should_extract_anchor = is_file_context or has_sandbox_file_signals or not is_platform_context
    
    if should_extract_anchor:
        # Try strict file_context_patterns first
        if any(re.search(p, text_lower) for p in file_context_patterns):
            if "desktop" in text_lower:
                anchor = "desktop"
            elif "documents" in text_lower or "document" in text_lower:
                anchor = "documents"
        
        # Fallback for prompts like "On the desktop, there is a folder called Test"
        if not anchor:
            file_operation_keywords = [
                "folder", "file", "read", "find", "open", "document",
                "called", "named", "test", ".txt", ".md", ".py",
                "answer", "question",
            ]
            has_file_operation = any(kw in text_lower for kw in file_operation_keywords)
            
            if has_file_operation or has_sandbox_file_signals:
                if "desktop" in text_lower:
                    anchor = "desktop"
                    logger.info("[sandbox_discovery] v1.15 FALLBACK anchor extraction: 'desktop'")
                elif "documents" in text_lower or "document" in text_lower:
                    anchor = "documents"
                    logger.info("[sandbox_discovery] v1.15 FALLBACK anchor extraction: 'documents'")
    
    if not anchor:
        if has_sandbox_file_signals:
            logger.warning(
                "[sandbox_discovery] v1.15 FAILED: sandbox_file keywords detected but no anchor found"
            )
        return None, None
    
    # Strip meta-instructions before extracting subfolder
    meta_patterns = [
        r'reply\s+(?:with\s+)?ok\b[^.]*',
        r'say\s+(?:with\s+)?ok\b[^.]*',
        r'just\s+say\s+ok\b[^.]*',
        r'when\s+you\s+understand[^.]*',
    ]
    cleaned_text = text_lower
    for meta_pat in meta_patterns:
        cleaned_text = re.sub(meta_pat, '', cleaned_text)
    
    # Extract subfolder
    patterns = [
        r'folder\s+(?:on|in)\s+(?:the\s+)?(?:desktop|documents)\s+(?:called|named)\s+["\']?(\w+)["\']?',
        r'(?:desktop|documents)\s+folder\s+(?:called|named)\s+["\']?(\w+)["\']?',
        r'folder\s+(?:called|named)\s+["\']?(\w+)["\']?',
        r'in\s+(?:my\s+|the\s+|a\s+)?([\w]+)\s+folder',
        r'([\w]+)\s+folder\s+(?:on|in)\s+(?:the\s+)?(?:desktop|documents)',
        r'my\s+([\w]+)\s+folder',
        r'(?:called|named)\s+["\']?(\w+)["\']?',
        r'desktop[/\\\\]+(\w+)',
        r'documents[/\\\\]+(\w+)',
    ]
    
    subfolder = None
    for pattern in patterns:
        match = re.search(pattern, cleaned_text)
        if match:
            candidate = match.group(1)
            if candidate not in SUBFOLDER_STOPWORDS and len(candidate) > 1:
                subfolder = candidate
                logger.info(
                    "[sandbox_discovery] v1.15 Extracted subfolder '%s' using pattern: %s",
                    subfolder, pattern
                )
                break
    
    # Fallback subfolder extraction
    if not subfolder:
        folder_mention = re.search(r'([\w]+)\s+folder', cleaned_text)
        if folder_mention:
            candidate = folder_mention.group(1)
            if candidate not in SUBFOLDER_STOPWORDS and len(candidate) > 1:
                subfolder = candidate
    
    # Final validation
    if subfolder is not None:
        subfolder = subfolder.strip()
        if not subfolder or subfolder in SUBFOLDER_STOPWORDS:
            logger.warning("[sandbox_discovery] v1.15 Discarding invalid subfolder: '%s'", subfolder)
            subfolder = None
    
    logger.info(
        "[sandbox_discovery] v1.15 extract_sandbox_hints result: anchor='%s', subfolder='%s'",
        anchor, subfolder
    )
    
    return anchor, subfolder


# =============================================================================
# MULTI-TARGET FILE EXTRACTION (v1.32 - Level 2.5)
# =============================================================================

def extract_file_targets(text: str) -> List[dict]:
    """
    v1.33: Extract multiple file targets with individual anchors.
    
    v1.33 CRITICAL FIX: Now context-aware - only extracts actual file references,
    not random English words like "both", "their", "full", "platform".
    
    Parses patterns like:
        "read test on desktop and test2 on D drive"
        "open file1.txt from documents and file2.txt from E:"
        "get test on the desktop and test2 on the D drive"
    
    Returns:
        List of dicts with keys: name, anchor, subfolder, explicit_path
        
    Examples:
        "read test on desktop and test2 on D drive" ->
        [
            {"name": "test", "anchor": "desktop", "subfolder": None, "explicit_path": None},
            {"name": "test2", "anchor": "D:", "subfolder": None, "explicit_path": None},
        ]
        
        "read D:\\test.txt and C:\\Users\\file.txt" ->
        [
            {"name": "test.txt", "anchor": None, "subfolder": None, "explicit_path": "D:\\test.txt"},
            {"name": "file.txt", "anchor": None, "subfolder": None, "explicit_path": "C:\\Users\\file.txt"},
        ]
    """
    if not text:
        return []
    
    text_lower = text.lower()
    targets = []
    
    logger.info("[sandbox_discovery] v1.32 extract_file_targets: parsing '%s'", text[:100])
    
    # =========================================================================
    # STEP 1: Extract explicit absolute paths first (highest priority)
    # Patterns: D:\test.txt, C:\Users\file.txt, etc.
    # =========================================================================
    
    explicit_path_pattern = r'([A-Za-z]:[\\\/][\w\\\/\.\-]+)'
    explicit_matches = re.findall(explicit_path_pattern, text)
    
    for path in explicit_matches:
        # Normalize path separators
        normalized_path = path.replace('/', '\\')
        filename = normalized_path.split('\\')[-1]
        
        targets.append({
            "name": filename,
            "anchor": None,
            "subfolder": None,
            "explicit_path": normalized_path,
        })
        logger.info(
            "[sandbox_discovery] v1.32 Found explicit path: %s -> name='%s'",
            normalized_path, filename
        )
    
    # If we found explicit paths, we might still have other targets
    # Remove explicit paths from text for further parsing
    remaining_text = text
    for path in explicit_matches:
        remaining_text = remaining_text.replace(path, ' ')
    remaining_text_lower = remaining_text.lower()
    
    # =========================================================================
    # STEP 2: Extract drive-letter references (e.g., "test2 on D drive")
    # v1.33: Added context-aware extraction - only match in file reference contexts
    # v1.34: Added Weaver-output patterns ("Find the Desktop 'test' file")
    # v1.35: CRITICAL FIX - Made unquoted patterns more restrictive to avoid spurious matches
    # v1.36: CRITICAL FIX - Added reverse-order pattern ('test2' on D: drive)
    # =========================================================================
    
    # Pattern: "filename on/in/from X drive" or "filename on/in/from X:"
    # v1.33: Look for file action verbs or quoted strings before the filename
    # v1.34: Added Weaver patterns ("D: drive 'test2' file", "D: 'test2' file")
    # v1.35: CRITICAL - Unquoted patterns REQUIRE quotes around filename now
    # v1.36: CRITICAL - Added reverse order: 'filename' on X: drive
    drive_patterns = [
        # v1.38 TEMPORARILY DISABLED - testing for hang
        # (r'the\s+(\w+(?:\.\w+)?)\s+file\s+(?:on|in|from)\s+(?:the\s+)?([A-Za-z]):\s*(?:drive)?', 'weaver_unquoted_before_drive'),
        
        # v1.38 TEMPORARILY DISABLED - testing for hang  
        # (r'and\s+(\w+(?:\.\w+)?)\s+(?:on|in|from)\s+(?:the\s+)?([A-Za-z]):\s*(?:drive)?', 'weaver_unquoted_and_context'),
        
        # v1.39 NEW: Quoted pattern WITHOUT colon requirement - "test2" file on D drive
        (r'["\']+(\w+(?:\.\w+)?)["\']\s+file\s+(?:on|in|from)\s+(?:the\s+)?([A-Za-z])\s+drive', 'weaver_quoted_no_colon'),
        
        # v1.36 NEW: Reverse order - 'test2' on D: drive / 'test2' on D drive
        # This is the ACTUAL Weaver format!
        (r"['\"](\w+(?:\.\w+)?)['\"]" + r"\s+(?:file\s+)?(?:on|in|from)\s+(?:the\s+)?([A-Za-z]):\s*(?:drive)?", 'weaver_quoted_before_drive_v2'),
        
        # v1.34: Weaver patterns WITH QUOTES (safe, high confidence)
        # "Find the D: drive 'test2' file" / "Locate D: 'filename' file"
        (r'(?:the\s+)?([A-Za-z]):\s+(?:drive\s+)?[\'"](\w+\.?\w*)[\'"](\s+file)?', 'weaver_drive_quoted'),
        
        # v1.35 REMOVED: Unquoted pattern was too broad and matched "D: file" → "file"
        # Do NOT extract unquoted filenames from drive references
        
        # Context-aware patterns (v1.33): With action verbs
        # "read test2 on D drive" / "open file on E drive"
        (r'(?:read|open|get|show|display|find|locate)\s+(?:the\s+)?(?:file\s+)?([\w\.]+)\s+(?:on|in|from)\s+(?:the\s+)?([A-Za-z])[\s:]*drive', 'action_verb'),
        
        # Quoted strings (v1.33): "test2" on D drive
        (r'["\']([\w\.]+)["\']\s+(?:on|in|from)\s+(?:the\s+)?([A-Za-z])[\s:]*drive', 'quoted'),
        
        # File indicator patterns (v1.33): "file called test2 on D drive"
        (r'(?:file|document)\s+(?:called|named)\s+([\w\.]+)\s+(?:on|in|from)\s+(?:the\s+)?([A-Za-z])[\s:]*drive', 'file_named'),
        
        # Drive colon variant: "test2 on D:" / "test2 from D:"
        (r'(?:read|open|get)\s+(?:the\s+)?([\w\.]+)\s+(?:on|in|from)\s+(?:the\s+)?([A-Za-z]):', 'drive_colon'),
    ]
    
    for pattern, pattern_type in drive_patterns:
        matches = re.finditer(pattern, remaining_text_lower)
        for match in matches:
            # v1.36: New reverse-order patterns have (filename, drive_letter)
            if pattern_type in ('weaver_quoted_before_drive_v2', 'weaver_quoted_before_location_v2'):
                # Reverse order: 'test2' on D: drive
                filename = match.group(1)
                drive_letter = match.group(2).upper()
            elif pattern_type == 'weaver_drive_quoted':
                # Original Weaver: D: drive 'test2' file (drive, filename)
                drive_letter = match.group(1).upper()
                filename = match.group(2)
            else:
                # Standard: (filename, drive_letter)
                filename = match.group(1)
                drive_letter = match.group(2).upper()
            
            # v1.33: STRICT validation
            if filename.lower() in FILENAME_STOPWORDS:
                logger.info(
                    "[sandbox_discovery] v1.33 REJECTED (stopword): '%s' (pattern: %s)",
                    filename, pattern_type
                )
                continue
            if len(filename) < 2:
                continue
            # v1.33: Reject if looks like a common English word (additional safety)
            common_words = {"both", "their", "full", "platform", "all", "some", "any", "every", "each"}
            if filename.lower() in common_words:
                logger.info(
                    "[sandbox_discovery] v1.33 REJECTED (common word): '%s'",
                    filename
                )
                continue
            
            # Check if this target is already in the list
            already_exists = any(
                t.get("name", "").lower() == filename.lower() and 
                t.get("anchor") == f"{drive_letter}:"
                for t in targets
            )
            if already_exists:
                continue
            
            targets.append({
                "name": filename,
                "anchor": f"{drive_letter}:",
                "subfolder": None,
                "explicit_path": None,
            })
            logger.info(
                "[sandbox_discovery] v1.33 Found drive target: name='%s', anchor='%s:', pattern='%s'",
                filename, drive_letter, pattern_type
            )
    
    # =========================================================================
    # STEP 3: Extract desktop/documents references
    # v1.33: Added context-aware extraction
    # v1.34: Added Weaver-output patterns ("Desktop 'test' file")
    # v1.35: CRITICAL FIX - Removed unquoted pattern to avoid spurious matches
    # =========================================================================
    
    # Pattern: "filename on/in/from desktop/documents"
    # v1.33: Context-aware - look for action verbs or file indicators
    # v1.34: Added Weaver patterns ("the Desktop 'test' file", "Desktop 'test' file")
    # v1.35: CRITICAL - Only use QUOTED patterns to avoid false positives
    location_patterns = [
        # v1.38 TEMPORARILY DISABLED - testing for hang
        # (r'the\s+(\w+(?:\.\w+)?)\s+file\s+(?:on|in|from)\s+(?:the\s+)?(Desktop|Documents)', 'weaver_unquoted_before_location'),
        
        # v1.38 TEMPORARILY DISABLED - testing for hang
        # (r'and\s+(\w+(?:\.\w+)?)\s+(?:on|in|from)\s+(?:the\s+)?(Desktop|Documents)', 'weaver_unquoted_and_context_location'),
        
        # v1.36 NEW: Reverse order - 'test' on Desktop (ACTUAL Weaver format!)
        (r"['\"](\w+(?:\.\w+)?)['\"]" + r"\s+(?:file\s+)?(?:on|in|from)\s+(?:the\s+)?(Desktop|Documents)", 'weaver_quoted_before_location_v2'),
        # v1.34: Weaver patterns WITH QUOTES (safe, high confidence)
        # "Find the Desktop 'test' file" / "Locate Documents 'filename' file"
        (r'(?:the\s+)?(desktop|documents)\s+[\'"](\w+\.?\w*)[\'"](\s+file)?', 'weaver_location_quoted'),
        
        # v1.35 REMOVED: Unquoted pattern was too broad
        # Do NOT use unquoted desktop/documents patterns
        
        # Context-aware: "read test on desktop" / "open file from documents"
        (r'(?:read|open|get|show|display|find|locate)\s+(?:the\s+)?(?:file\s+)?([\w\.]+)\s+(?:on|in|from)\s+(?:the\s+|my\s+)?(desktop|documents)', 'action_verb'),
        
        # Quoted: "test" on desktop
        (r'["\']([\w\.]+)["\']\s+(?:on|in|from)\s+(?:the\s+|my\s+)?(desktop|documents)', 'quoted'),
        
        # File indicator: "file called test on desktop"
        (r'(?:file|document)\s+(?:called|named)\s+([\w\.]+)\s+(?:on|in|from)\s+(?:the\s+|my\s+)?(desktop|documents)', 'file_named'),
    ]
    
    for pattern, pattern_type in location_patterns:
        matches = re.finditer(pattern, remaining_text_lower)
        for match in matches:
            # v1.38: Unquoted weaver patterns have (filename, location) order
            # Only the OLD weaver patterns have (location, filename) order
            if pattern_type in ('weaver_location_quoted', 'weaver_quoted_before_location_v2'):
                # OLD Weaver patterns: (location, filename) or (location first)
                if pattern_type == 'weaver_location_quoted':
                    location = match.group(1).lower()
                    filename = match.group(2)
                else:
                    # weaver_quoted_before_location_v2: (filename, location)
                    filename = match.group(1)
                    location = match.group(2).lower()
            else:
                # Standard AND new v1.38 unquoted patterns: (filename, location)
                filename = match.group(1)
                location = match.group(2).lower()
            
            # v1.33: STRICT validation
            if filename.lower() in FILENAME_STOPWORDS:
                logger.info(
                    "[sandbox_discovery] v1.33 REJECTED (stopword): '%s' (pattern: %s)",
                    filename, pattern_type
                )
                continue
            if len(filename) < 2:
                continue
            # v1.33: Additional safety check
            common_words = {"both", "their", "full", "platform", "all", "some", "any", "contents"}
            if filename.lower() in common_words:
                logger.info(
                    "[sandbox_discovery] v1.33 REJECTED (common word): '%s'",
                    filename
                )
                continue
            
            # Check if this target is already in the list
            already_exists = any(
                t.get("name", "").lower() == filename.lower() and 
                t.get("anchor") == location
                for t in targets
            )
            if already_exists:
                continue
            
            targets.append({
                "name": filename,
                "anchor": location,
                "subfolder": None,
                "explicit_path": None,
            })
            logger.info(
                "[sandbox_discovery] v1.33 Found location target: name='%s', anchor='%s', pattern='%s'",
                filename, location, pattern_type
            )
    
    # =========================================================================
    # STEP 4: Extract file-with-subfolder patterns
    # =========================================================================
    
    # Pattern: "file in Test folder on desktop"
    subfolder_patterns = [
        r'(\w+(?:\.\w+)?)\s+(?:in|from)\s+(?:the\s+)?(\w+)\s+folder\s+(?:on|in)\s+(?:the\s+)?(desktop|documents)',
        r'(?:file\s+)?(?:called|named)\s+(\w+(?:\.\w+)?)\s+(?:in|from)\s+(?:the\s+)?(\w+)\s+folder',
    ]
    
    for pattern in subfolder_patterns:
        matches = re.finditer(pattern, remaining_text_lower)
        for match in matches:
            groups = match.groups()
            if len(groups) >= 2:
                filename = groups[0]
                subfolder = groups[1]
                location = groups[2] if len(groups) > 2 else "desktop"  # Default to desktop
                
                # Validate
                if filename.lower() in FILENAME_STOPWORDS:
                    continue
                if subfolder.lower() in SUBFOLDER_STOPWORDS:
                    continue
                if len(filename) < 2:
                    continue
                
                # Check if this target is already in the list
                already_exists = any(
                    t.get("name", "").lower() == filename.lower() and 
                    t.get("anchor") == location and
                    t.get("subfolder", "").lower() == subfolder.lower()
                    for t in targets
                )
                if already_exists:
                    continue
                
                targets.append({
                    "name": filename,
                    "anchor": location,
                    "subfolder": subfolder,
                    "explicit_path": None,
                })
                logger.info(
                    "[sandbox_discovery] v1.32 Found subfolder target: name='%s', anchor='%s', subfolder='%s'",
                    filename, location, subfolder
                )
    
    # =========================================================================
    # STEP 5: Fallback - extract "file called X" without explicit location
    # These will use the default anchor from extract_sandbox_hints
    # =========================================================================
    
    if not targets:
        # Try to extract file names without explicit locations
        fallback_patterns = [
            r'(?:read|open|get|show|display)\s+(?:the\s+)?(?:file\s+)?(?:called\s+|named\s+)?(\w+(?:\.\w+)?)',
            r'(?:file|document)\s+(?:called|named)\s+(\w+(?:\.\w+)?)',
        ]
        
        default_anchor, default_subfolder = extract_sandbox_hints(text)
        
        for pattern in fallback_patterns:
            matches = re.finditer(pattern, remaining_text_lower)
            for match in matches:
                filename = match.group(1)
                
                # Validate
                if filename.lower() in FILENAME_STOPWORDS:
                    continue
                if len(filename) < 2:
                    continue
                
                # Check if this target is already in the list
                already_exists = any(
                    t.get("name", "").lower() == filename.lower()
                    for t in targets
                )
                if already_exists:
                    continue
                
                targets.append({
                    "name": filename,
                    "anchor": default_anchor,
                    "subfolder": default_subfolder,
                    "explicit_path": None,
                })
                logger.info(
                    "[sandbox_discovery] v1.32 Found fallback target: name='%s', anchor='%s', subfolder='%s'",
                    filename, default_anchor, default_subfolder
                )
    
    logger.info(
        "[sandbox_discovery] v1.32 extract_file_targets: found %d targets: %s",
        len(targets), [t.get("name") for t in targets]
    )
    
    return targets


def is_multi_target_request(text: str) -> bool:
    """
    v1.32: Check if text contains a multi-target file read request.
    
    Returns True if the text mentions multiple files with different locations.
    """
    if not text:
        return False
    
    targets = extract_file_targets(text)
    
    # Multi-target if we have 2+ targets
    if len(targets) >= 2:
        return True
    
    # Also check for explicit "and" patterns suggesting multiple files
    text_lower = text.lower()
    multi_indicators = [
        r'\band\b.*\b(?:on|in|from)\b.*\b(?:drive|desktop|documents)',
        r'(?:both|all)\s+(?:the\s+)?files?',
        r'files?\s+(?:on|in|from)\s+.+\s+and\s+',
    ]
    
    for pattern in multi_indicators:
        if re.search(pattern, text_lower):
            return True
    
    return False


# =============================================================================
# OUTPUT MODE DETECTION (v1.13, v1.16-v1.17, v1.24)
# =============================================================================

def detect_output_mode(text: str) -> OutputMode:
    """
    v1.13: Detect the intended output mode for MICRO_FILE_TASK jobs.
    v1.16: SAFETY OVERRIDE - "do not change" signals ALWAYS force CHAT_ONLY.
    v1.24: OVERWRITE_FULL - "overwrite the file", "just leave X" for destructive replacement.
    
    Priority order:
    1. CHAT_ONLY (safety override)
    2. OVERWRITE_FULL (destructive write)
    3. REWRITE_IN_PLACE (multi-question insert)
    4. SEPARATE_REPLY_FILE
    5. APPEND_IN_PLACE
    6. Default: CHAT_ONLY
    """
    # v1.30 DEBUG: Print to verify function is being called and see the input
    print(f"[DEBUG detect_output_mode] v1.30 ENTRY text_len={len(text) if text else 0}")
    if text:
        print(f"[DEBUG detect_output_mode] text_preview: {text[:200]}...")
    
    if not text:
        print("[DEBUG detect_output_mode] -> CHAT_ONLY (empty text)")
        return OutputMode.CHAT_ONLY
    
    text_lower = text.lower()
    
    # ==========================================================================
    # v1.16 SAFETY OVERRIDE: "Do not change" signals MUST force CHAT_ONLY
    # ==========================================================================
    
    simple_chat_only_phrases = [
        "do not change",
        "don't change",
        "do not modify",
        "don't modify",
        "do not alter",
        "don't alter",
        "do not touch",
        "don't touch",
        "leave it alone",
        "leave the file",
        "leave unchanged",
        "no file modif",
        "no file change",
        "without modifying",
        "without changing",
        "chat only",
        "in chat only",
        "answer in chat",
        "reply in chat",
    ]
    
    for phrase in simple_chat_only_phrases:
        if phrase in text_lower:
            logger.info(
                "[sandbox_discovery] v1.16 detect_output_mode: CHAT_ONLY (SAFETY phrase: '%s')",
                phrase
            )
            return OutputMode.CHAT_ONLY
    
    # Additional CHAT_ONLY patterns
    chat_only_patterns = [
        "just answer here",
        "answer here in chat",
        "reply here in chat",
        "don't write to the file",
        "do not write",
        "no file output",
        "no file changes",
    ]
    
    if any(pattern in text_lower for pattern in chat_only_patterns):
        logger.info("[sandbox_discovery] v1.13 detect_output_mode: CHAT_ONLY (explicit trigger)")
        return OutputMode.CHAT_ONLY
    
    # ==========================================================================
    # v1.24: OVERWRITE_FULL triggers
    # ==========================================================================
    
    overwrite_patterns = [
        "overwrite the file",
        "overwrite file",
        "overwrite it with",
        "overwrite it",
        "read and overwrite",
        "read it then overwrite",
        "then overwrite",
        "replace the file",
        "replace file",
        "replace contents",
        "replace the contents",
        "replace everything",
        "delete everything",
        "remove everything",
        "clear the file",
        "clear file",
        "wipe the file",
        "wipe file",
        "just leave",
        "only leave",
        "leave only",
        "should only contain",
        "contain only",
        "only contain",
        "should contain only",
        "replace with",
        "set to",
        "set the file to",
        "set file to",
        "file should say",
        "should say",
        "make it say",
        "have it say",
    ]
    
    if any(pattern in text_lower for pattern in overwrite_patterns):
        logger.info("[sandbox_discovery] v1.24 detect_output_mode: OVERWRITE_FULL (destructive trigger)")
        return OutputMode.OVERWRITE_FULL
    
    # ==========================================================================
    # REWRITE_IN_PLACE triggers (v1.17)
    # ==========================================================================
    
    rewrite_patterns = [
        "answer every question",
        "answer each question",
        "answer all questions",
        "answer all the questions",
        "answer the questions",
        "i want you to answer the questions",
        "want you to answer the questions",
        "answer where the answer",
        "answer where it says answer",
        "keep the format exactly",
        "keep the format of the file",
        "keep the file format",
        "preserve the format",
        "maintain the format",
        "same format",
        "has 25 questions",
        "has questions in it",
        "questions in it",
        "some of the questions have already been answered",
        "already been answered",
        "spot the trick",
        "trick question",
        "under each question",
        "beneath each question",
        "below each question",
        "directly under each question",
        "put answer under each",
        "write answer under each",
        "insert answer under each",
        "multi-question",
        "multiple questions",
        "fill in the missing",
        "fill the missing",
        "fill missing",
        "fill in the answer",
        "fill in answers",
        "fill missing answer",
        "fill missing answers",
        "fill blank answer",
        "fill blank answers",
        "fill blank",
        "fill empty answer",
        "fill empty answers",
        "fill empty",
        "populate the answer",
        "populate answers",
        "complete the answer",
        "complete answers",
        "under answer:",
        "under the answer",
        "into the file under",
        "preserve everything else",
        "leave those",
        "leave existing",
        "just answer where",
        "answer where the answer is meant",
        "format exactly the same",
        "format of the file exactly",
        "25 questions",
        "25-question",
        "answer the 25",
    ]
    
    if any(pattern in text_lower for pattern in rewrite_patterns):
        print(f"[DEBUG detect_output_mode] -> REWRITE_IN_PLACE (matched pattern)")
        logger.info("[sandbox_discovery] v1.27 detect_output_mode: REWRITE_IN_PLACE (multi-question trigger)")
        return OutputMode.REWRITE_IN_PLACE
    
    # ==========================================================================
    # SEPARATE_REPLY_FILE triggers (v1.43: EXPANDED for Weaver output format)
    # ==========================================================================
    
    separate_file_patterns = [
        # Original patterns
        "save to reply.txt",
        "save as reply.txt",
        "write to reply.txt",
        "create reply.txt",
        "write to a new file",
        "write to new file",
        "create a new file",
        "create a reply file",
        "separate file",
        "save as a new file",
        "save to a new file",
        "put reply in reply.txt",
        "output to reply.txt",
        # v1.42 NEW: Expanded patterns for natural language file creation
        "create a new file on",       # "create a new file on desktop"
        "create a file on",           # "create a file on desktop"
        "create a file called",       # "create a file called reply"
        "create file called",         # "create file called reply"
        "make a new file",            # "make a new file"
        "make a file called",         # "make a file called X"
        "make a file on",             # "make a file on desktop"
        "make file called",           # "make file called X"
        "in the reply file",          # "in the reply file write"
        "in a file called",           # "in a file called reply"
        "then create",                # "read X then create Y"
        "and create a",               # "read X and create a new file"
        "write a reply to",           # signals output expectation
        "and call it",                # "create file and call it reply"
        # v1.43 NEW: Weaver output format patterns
        "create desktop/",            # Weaver: "create Desktop/reply.txt"
        "create/overwrite",           # Weaver: "Create/overwrite Desktop/reply.txt"
        "/reply.txt",                 # Weaver: "Desktop/reply.txt"
        "/reply",                     # Weaver: "Desktop/reply"
        "synthesize",                 # Weaver: "Synthesize a coherent reply"
        "synthesized reply",          # Weaver: "write the synthesized reply"
        "composed reply",             # Weaver: "write a composed reply"
        "reply synthesis",            # Weaver: "reply synthesis (micro file task)"
        "write reply",                # Weaver: "write reply to Desktop"
        "with a composed reply",
        "with synthesized",
    ]
    
    if any(pattern in text_lower for pattern in separate_file_patterns):
        print(f"[DEBUG detect_output_mode] -> SEPARATE_REPLY_FILE (simple pattern match)")
        logger.info("[sandbox_discovery] v1.42 detect_output_mode: SEPARATE_REPLY_FILE (explicit trigger)")
        return OutputMode.SEPARATE_REPLY_FILE
    
    # v1.42 NEW: Regex patterns for complex file creation phrases
    create_file_regex_patterns = [
        # "create a [new] file [on location] [and] call it X"
        r"create\s+(?:a\s+)?(?:new\s+)?file\s+(?:on\s+\w+\s+)?(?:and\s+)?call\s+it\s+\w+",
        # "create a [new] file called/named X"
        r"create\s+(?:a\s+)?(?:new\s+)?file\s+(?:called|named)\s+\w+",
        # "in the X file i want [you] to write"
        r"in\s+the\s+\w+\s+file\s+(?:i\s+)?want\s+(?:you\s+)?to\s+write",
        # "make a file [called] X [on location]"
        r"make\s+(?:a\s+)?(?:new\s+)?file\s+(?:called\s+)?\w+\s+(?:on|in)\s+",
    ]
    
    for pattern in create_file_regex_patterns:
        if re.search(pattern, text_lower):
            print(f"[DEBUG detect_output_mode] -> SEPARATE_REPLY_FILE (regex match: {pattern[:50]})")
            logger.info("[sandbox_discovery] v1.42 detect_output_mode: SEPARATE_REPLY_FILE (regex pattern)")
            return OutputMode.SEPARATE_REPLY_FILE
    
    # ==========================================================================
    # APPEND_IN_PLACE triggers
    # ==========================================================================
    
    append_patterns = [
        "write under",
        "write below",
        "write beneath",
        "write it under",
        "write the reply under",
        "write reply under",
        "put under",
        "put below",
        "put beneath",
        "put reply under",
        "put the reply under",
        "append",
        "add below",
        "add under",
        "add beneath",
        "underneath the question",
        "beneath the question",
        "under the question",
        "below the question",
        "write it in the file",
        "write in the file",
        "write reply in the file",
        "write the reply in",
        "put it in the file",
        "add it to the file",
        "add to the file",
        "modify the file",
        "update the file",
        "edit the file",
        "in-place",
        "in place",
        "write the answer",
        "insert the answer",
        "write answers",
        "insert answers",
    ]
    
    if any(pattern in text_lower for pattern in append_patterns):
        logger.info("[sandbox_discovery] v1.13 detect_output_mode: APPEND_IN_PLACE (write/append trigger)")
        return OutputMode.APPEND_IN_PLACE
    
    # Default: CHAT_ONLY (safest)
    print(f"[DEBUG detect_output_mode] -> CHAT_ONLY (default - no patterns matched)")
    logger.info("[sandbox_discovery] v1.13 detect_output_mode: CHAT_ONLY (default)")
    return OutputMode.CHAT_ONLY


# =============================================================================
# REPLACEMENT TEXT EXTRACTION (v1.31)
# =============================================================================

def extract_replacement_text(text: str) -> Optional[str]:
    """
    v1.31: Extract the replacement text for OVERWRITE_FULL operations.
    
    Detection patterns (in priority order):
    1. Quoted strings: 'text', "text" (HIGHEST PRIORITY)
    2. Contextual quoted patterns: "just leave 'X'", etc.
    3. v1.31 NEW: Unquoted semantic patterns (FALLBACK):
       - "overwrite with the message X"
       - "replace everything with X"
       - "set the file to X"
       - "make it say X"
    4. Last quoted string fallback (if overwrite context)
    """
    if not text:
        return None
    
    text_lower = text.lower()
    
    # =========================================================================
    # STEP 1: Try quoted strings first (HIGHEST PRIORITY)
    # =========================================================================
    
    # Double-quoted strings
    double_quoted = re.findall(r'"([^"]+)"', text)
    # Single-quoted strings  
    single_quoted = re.findall(r"'([^']+)'", text)
    
    all_quoted = double_quoted + single_quoted
    
    # Filter out non-replacement quoted strings (filenames, paths)
    replacement_candidates = []
    for quoted in all_quoted:
        quoted_lower = quoted.lower()
        # Skip if looks like a filename (has extension)
        if '.' in quoted and len(quoted.split('.')[-1]) <= 4:
            continue
        # Skip if looks like a path
        if '/' in quoted or '\\' in quoted:
            continue
        # Skip very short strings
        if len(quoted) < 2:
            continue
        replacement_candidates.append(quoted)
    
    # =========================================================================
    # STEP 2: Try contextual quoted patterns
    # =========================================================================
    
    contextual_patterns = [
        r"just\s+leave\s+(?:a\s+)?(?:message\s+)?(?:for\s+\w+\s+)?['\"]([^'\"]+)['\"]?",
        r"just\s+leave\s+['\"]([^'\"]+)['\"]?",
        r"only\s+leave\s+['\"]([^'\"]+)['\"]?",
        r"leave\s+only\s+['\"]([^'\"]+)['\"]?",
        r"replace\s+(?:it\s+)?(?:the\s+file\s+)?with\s+['\"]([^'\"]+)['\"]?",
        r"replace\s+contents?\s+with\s+['\"]([^'\"]+)['\"]?",
        r"(?:should\s+)?(?:only\s+)?contain\s+['\"]([^'\"]+)['\"]?",
        r"contain\s+only\s+['\"]([^'\"]+)['\"]?",
        r"(?:file\s+)?should\s+(?:only\s+)?say\s+['\"]([^'\"]+)['\"]?",
        r"make\s+it\s+say\s+['\"]([^'\"]+)['\"]?",
        r"have\s+it\s+say\s+['\"]([^'\"]+)['\"]?",
        r"set\s+(?:the\s+file\s+)?(?:contents?\s+)?to\s+['\"]([^'\"]+)['\"]?",
    ]
    
    for pattern in contextual_patterns:
        match = re.search(pattern, text_lower)
        if match:
            extracted = match.group(1)
            if extracted and len(extracted) >= 1:
                # Preserve original case by finding in original text
                case_match = re.search(re.escape(extracted), text, re.IGNORECASE)
                if case_match:
                    extracted = case_match.group(0)
                logger.info(
                    "[sandbox_discovery] v1.31 extract_replacement_text: Found via quoted pattern: '%s'",
                    extracted[:50]
                )
                return extracted.strip()
    
    # =========================================================================
    # STEP 3: Fallback quoted string in overwrite context
    # =========================================================================
    
    if replacement_candidates:
        overwrite_indicators = [
            "overwrite", "replace", "just leave", "only leave", "leave only",
            "contain only", "should only", "should say", "make it say",
            "clear", "wipe", "delete everything", "remove everything",
        ]
        
        has_overwrite_context = any(ind in text_lower for ind in overwrite_indicators)
        
        if has_overwrite_context:
            replacement = replacement_candidates[-1]
            logger.info(
                "[sandbox_discovery] v1.31 extract_replacement_text: Using last quoted string: '%s'",
                replacement[:50] if replacement else "(empty)"
            )
            return replacement.strip()
    
    # =========================================================================
    # STEP 4 (v1.31 NEW): Try UNQUOTED semantic patterns (FALLBACK)
    # Only runs if no quoted strings were found
    # =========================================================================
    
    logger.info(
        "[sandbox_discovery] v1.31 extract_replacement_text: No quoted text found, trying unquoted patterns"
    )
    
    # Unquoted extraction patterns - capture text after semantic markers
    # These patterns capture content until end of sentence (period) or end of string
    unquoted_patterns = [
        # "overwrite with the message X" / "overwrite with the text X" / "overwrite it with X"
        (r'overwrite\s+(?:it\s+)?with\s+(?:the\s+)?(?:message|text)[,:]?\s+(.+?)(?:\.|$)', 'overwrite_with_message'),
        
        # "overwrite with X" (simpler, lower priority)
        (r'overwrite\s+(?:it\s+)?with\s+(.+?)(?:\.|$)', 'overwrite_with'),
        
        # "replace everything with X" / "replace it with X" / "replace with X"
        (r'replace\s+(?:everything\s+|it\s+)?with\s+(.+?)(?:\.|$)', 'replace_with'),
        
        # "set the file to X" / "set it to X" / "set contents to X"
        (r'set\s+(?:the\s+)?(?:file\s+|it\s+)?(?:contents?\s+)?to\s+(.+?)(?:\.|$)', 'set_to'),
        
        # "make it say X" / "have it say X"
        (r'(?:make|have)\s+it\s+say\s+(.+?)(?:\.|$)', 'make_say'),
        
        # "should contain X" / "should only contain X"
        (r'should\s+(?:only\s+)?contain\s+(.+?)(?:\.|$)', 'should_contain'),
        
        # "file should say X"
        (r'file\s+should\s+say\s+(.+?)(?:\.|$)', 'file_should_say'),
        
        # "just leave X" / "only leave X" (unquoted version)
        (r'(?:just|only)\s+leave\s+(.+?)(?:\.|$)', 'just_leave'),
    ]
    
    for pattern, pattern_name in unquoted_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            
            # Clean up the extracted text
            # Remove trailing punctuation that might have been captured
            extracted = extracted.rstrip('.,;:!?')
            
            # Skip if too short or empty
            if not extracted or len(extracted) < 2:
                continue
            
            # Skip if it looks like we captured a clause instead of content
            # (e.g., "and then do something else")
            skip_prefixes = ['and ', 'but ', 'then ', 'so ', 'or ']
            if any(extracted.lower().startswith(prefix) for prefix in skip_prefixes):
                continue
            
            # Preserve original case by finding in original text
            case_match = re.search(re.escape(extracted), text, re.IGNORECASE)
            if case_match:
                extracted = case_match.group(0)
            
            logger.info(
                "[sandbox_discovery] v1.31 extract_replacement_text: Found via UNQUOTED pattern '%s': '%s'",
                pattern_name, extracted[:50]
            )
            return extracted.strip()
    
    logger.warning(
        "[sandbox_discovery] v1.31 extract_replacement_text: Could not extract replacement text"
    )
    return None


# =============================================================================
# SYSTEM-WIDE SCAN SUPPORT (v1.40 - Level 2.5+)
# =============================================================================

# All roots to scan for system-wide searches
SYSTEM_SCAN_ROOTS = [
    # User Desktop locations
    "C:\\Users\\dizzi\\OneDrive\\Desktop",
    "C:\\Users\\dizzi\\Desktop",
    # User Documents
    "C:\\Users\\dizzi\\OneDrive\\Documents",
    "C:\\Users\\dizzi\\Documents",
    # User Downloads
    "C:\\Users\\dizzi\\Downloads",
    # Common project directories (v1.41)
    "D:\\Orb",
    "D:\\orb-desktop",
    # Drive roots (shallow scan)
    "D:\\",
    "C:\\",
    "E:\\",
]

# Patterns that indicate system-wide scan intent
SYSTEM_SCAN_INDICATORS = [
    # "on my system" variants
    r"on\s+(?:my|the|this)\s+system",
    r"in\s+(?:my|the|this)\s+system",
    r"across\s+(?:my|the)\s+(?:system|drives?|computer)",
    r"throughout\s+(?:my|the)\s+(?:system|drives?|computer)",
    # "anywhere" / "everywhere" variants
    r"(?:find|search|look)\s+(?:for\s+)?(?:them\s+)?(?:anywhere|everywhere)",
    r"wherever\s+they\s+(?:are|might\s+be)",
    # "search for" without specific location
    r"(?:search|scan|look)\s+(?:for|through)\s+(?:files?\s+)?(?:called|named)",
    # "find all files called X"
    r"find\s+(?:all\s+)?(?:the\s+)?files?\s+(?:called|named)",
    # "all files called X on my system/computer"
    r"all\s+(?:the\s+)?files?\s+(?:called|named)\s+[\w,\s]+\s+(?:on|in|across)",
]


def is_system_wide_scan_request(text: str) -> bool:
    """
    v1.40: Detect if user wants a system-wide file scan.
    
    Triggers on patterns like:
    - "Find all files called test1, test2, test3, and test4 on my system"
    - "Search for test files across my drives"
    - "Look for test.txt anywhere"
    
    Returns True if system-wide scan is requested.
    """
    if not text:
        return False
    
    text_lower = text.lower()
    
    # Check for system-wide indicators
    for pattern in SYSTEM_SCAN_INDICATORS:
        if re.search(pattern, text_lower):
            logger.info(
                "[sandbox_discovery] v1.40 is_system_wide_scan_request: DETECTED via pattern '%s'",
                pattern
            )
            return True
    
    # Also check: has file names but NO specific anchor
    # e.g., "find test1, test2, test3" without "on desktop" or "on D drive"
    has_file_names = bool(re.search(r'(?:test|file)\d+', text_lower))
    has_find_verb = any(v in text_lower for v in ['find', 'search', 'locate', 'look for', 'get'])
    has_specific_anchor = any(a in text_lower for a in ['desktop', 'documents', 'downloads', ' d:', ' c:', ' e:'])
    
    if has_file_names and has_find_verb and not has_specific_anchor:
        # Check if "system" or "drives" or similar is mentioned
        if any(s in text_lower for s in ['system', 'drive', 'computer', 'anywhere', 'everywhere']):
            logger.info(
                "[sandbox_discovery] v1.40 is_system_wide_scan_request: DETECTED via file+verb+system heuristic"
            )
            return True
    
    return False


def extract_scan_file_names(text: str) -> List[str]:
    """
    v1.40: Extract file names for system-wide scan.
    
    Extracts names from patterns like:
    - "test1, test2, test3, and test4"
    - "files called test1 test2 test3"
    - "test.txt and config.json"
    
    Returns list of file names (without paths/anchors).
    """
    if not text:
        return []
    
    text_lower = text.lower()
    file_names = []
    
    logger.info("[sandbox_discovery] v1.40 extract_scan_file_names: parsing '%s'", text[:100])
    
    # Pattern 1: "files called/named X, Y, Z, and W"
    # v1.41 FIX: Stop at "and [verb]" to avoid capturing action phrases
    # e.g., "test1, test2, test3, test4 and concatenate" -> stops before "and concatenate"
    called_pattern = r'(?:files?\s+)?(?:called|named)\s+([\w\.,\s]+?)(?:\s+and\s+(?:concatenate|combine|merge|output|print|return|then)|\s+(?:on|in|across|throughout)|$)'
    match = re.search(called_pattern, text_lower)
    if match:
        names_str = match.group(1)
        # Parse comma/and separated list
        # v1.41: Only replace "and" when followed by filename-like word (alphanumeric)
        names_str = re.sub(r'\s+and\s+(?=[a-z0-9])', ', ', names_str)
        names = [n.strip() for n in names_str.split(',') if n.strip()]
        for name in names:
            # v1.41: Skip if contains spaces (phrase, not filename)
            if ' ' in name:
                logger.info("[sandbox_discovery] v1.41 REJECTED (contains space): '%s'", name)
                continue
            if name and name not in FILENAME_STOPWORDS and len(name) >= 2:
                if name not in file_names:
                    file_names.append(name)
                    logger.info("[sandbox_discovery] v1.41 Found scan target: '%s'", name)
    
    # Pattern 2: Explicit "testN" pattern
    test_pattern = r'\b(test\d+)\b'
    test_matches = re.findall(test_pattern, text_lower)
    for name in test_matches:
        if name not in file_names:
            file_names.append(name)
            logger.info("[sandbox_discovery] v1.40 Found test pattern target: '%s'", name)
    
    # Pattern 3: Quoted file names
    quoted_pattern = r'["\']([\w\.]+)["\']'
    quoted_matches = re.findall(quoted_pattern, text)
    for name in quoted_matches:
        name_lower = name.lower()
        if name_lower not in FILENAME_STOPWORDS and len(name) >= 2:
            if name_lower not in [f.lower() for f in file_names]:
                file_names.append(name)
                logger.info("[sandbox_discovery] v1.40 Found quoted target: '%s'", name)
    
    # Pattern 4: "file1.txt, file2.txt" with extensions
    ext_pattern = r'\b([\w]+\.(?:txt|md|py|json|yaml|yml|js|ts))\b'
    ext_matches = re.findall(ext_pattern, text_lower)
    for name in ext_matches:
        if name not in file_names:
            file_names.append(name)
            logger.info("[sandbox_discovery] v1.40 Found extension target: '%s'", name)
    
    logger.info(
        "[sandbox_discovery] v1.40 extract_scan_file_names: found %d targets: %s",
        len(file_names), file_names
    )
    
    return file_names


def extract_create_targets(text: str) -> List[dict]:
    """
    v1.43: Extract file targets that should be CREATED (not searched for).
    
    CRITICAL: These targets should NOT be passed to file scanning/searching.
    They represent output files that the user wants created.
    
    Patterns detected:
    - User format: "create a new file called X"
    - User format: "create X on desktop"
    - User format: "make a file named X"
    - Weaver format: "create Desktop/reply.txt"
    - Weaver format: "Create/overwrite Desktop/X"
    
    Returns:
        List of dicts with keys: name, anchor, is_create=True
    """
    if not text:
        return []
    
    text_lower = text.lower()
    create_targets = []
    
    logger.info("[sandbox_discovery] v1.43 extract_create_targets: parsing '%s'", text[:100])
    
    # =========================================================================
    # v1.43 NEW: Weaver output format patterns (highest priority)
    # =========================================================================
    
    # Pattern W1: "create Desktop/X" or "Create/overwrite Desktop/X"
    weaver_pattern1 = r"(?:create/?overwrite|create)\s+(?:the\s+)?(?:desktop|documents)[/\\](\w+(?:\.\w+)?)"
    match = re.search(weaver_pattern1, text_lower)
    if match:
        filename = match.group(1)
        # Extract anchor from the pattern
        anchor_match = re.search(r"(desktop|documents)", text_lower)
        anchor = anchor_match.group(1) if anchor_match else "desktop"
        if filename and filename not in FILENAME_STOPWORDS:
            create_targets.append({
                "name": filename,
                "anchor": anchor,
                "is_create": True,
            })
            logger.info("[sandbox_discovery] v1.43 CREATE target (Weaver path format): '%s' on '%s'", filename, anchor)
    
    # Pattern W2: "Desktop/X.txt" or "Documents/X" standalone (with reply/output context)
    weaver_pattern2 = r"(?:desktop|documents)[/\\](\w+(?:\.\w+)?)"
    if "reply" in text_lower or "output" in text_lower or "create" in text_lower or "write" in text_lower:
        matches = re.findall(weaver_pattern2, text_lower)
        for filename in matches:
            if filename and filename not in FILENAME_STOPWORDS and len(filename) >= 2:
                # Don't add if already exists
                if not any(t["name"].lower() == filename.lower() for t in create_targets):
                    anchor_match = re.search(r"(desktop|documents)[/\\]" + re.escape(filename), text_lower)
                    if anchor_match:
                        anchor = "desktop" if "desktop" in anchor_match.group(0) else "documents"
                    else:
                        anchor = "desktop"
                    create_targets.append({
                        "name": filename,
                        "anchor": anchor,
                        "is_create": True,
                    })
                    logger.info("[sandbox_discovery] v1.43 CREATE target (Weaver standalone): '%s' on '%s'", filename, anchor)
    
    # =========================================================================
    # Original user-format patterns
    # =========================================================================
    
    # Pattern 1: "create a [new] file [on location] [and] call it X"
    pattern1 = r"create\s+(?:a\s+)?(?:new\s+)?file\s+(?:on\s+(\w+)\s+)?(?:and\s+)?call\s+it\s+(\w+)"
    match = re.search(pattern1, text_lower)
    if match:
        anchor = match.group(1)  # May be None
        filename = match.group(2)
        if filename and filename not in FILENAME_STOPWORDS:
            if not any(t["name"] == filename for t in create_targets):
                create_targets.append({
                    "name": filename,
                    "anchor": anchor,
                    "is_create": True,
                })
                logger.info("[sandbox_discovery] v1.42 CREATE target (pattern1): '%s' on '%s'", filename, anchor)
    
    # Pattern 2: "create a file called/named X"
    pattern2 = r"create\s+(?:a\s+)?(?:new\s+)?file\s+(?:called|named)\s+(\w+)"
    match = re.search(pattern2, text_lower)
    if match:
        filename = match.group(1)
        if filename and filename not in FILENAME_STOPWORDS:
            if not any(t["name"] == filename for t in create_targets):
                create_targets.append({
                    "name": filename,
                    "anchor": None,
                    "is_create": True,
                })
                logger.info("[sandbox_discovery] v1.42 CREATE target (pattern2): '%s'", filename)
    
    # Pattern 3: "in the X file i want [you] to write" (preceded by create context)
    if "create" in text_lower or "new file" in text_lower or "make a file" in text_lower:
        pattern3 = r"in\s+the\s+(\w+)\s+file\s+(?:i\s+)?want\s+(?:you\s+)?to\s+write"
        match = re.search(pattern3, text_lower)
        if match:
            filename = match.group(1)
            if filename and filename not in FILENAME_STOPWORDS:
                if not any(t["name"] == filename for t in create_targets):
                    create_targets.append({
                        "name": filename,
                        "anchor": None,
                        "is_create": True,
                    })
                    logger.info("[sandbox_discovery] v1.42 CREATE target (pattern3 - in-file): '%s'", filename)
    
    # Pattern 4: "make a [new] file [called/named] X [on location]"
    pattern4 = r"make\s+(?:a\s+)?(?:new\s+)?file\s+(?:(?:called|named)\s+)?(\w+)(?:\s+on\s+(\w+))?"
    match = re.search(pattern4, text_lower)
    if match:
        filename = match.group(1)
        anchor = match.group(2)  # May be None
        if filename and filename not in FILENAME_STOPWORDS and len(filename) >= 2:
            if not any(t["name"] == filename for t in create_targets):
                create_targets.append({
                    "name": filename,
                    "anchor": anchor,
                    "is_create": True,
                })
                logger.info("[sandbox_discovery] v1.42 CREATE target (pattern4 - make): '%s' on '%s'", filename, anchor)
    
    # Pattern 5: "then create [a] [new] [file] [called] X" / "and create X"
    pattern5 = r"(?:then|and)\s+create\s+(?:a\s+)?(?:new\s+)?(?:file\s+)?(?:called\s+)?(\w+)"
    match = re.search(pattern5, text_lower)
    if match:
        filename = match.group(1)
        if filename and filename not in FILENAME_STOPWORDS and len(filename) >= 2:
            # Extra check: "create a" should not extract "a"
            if filename != "a" and filename != "new" and filename != "file":
                if not any(t["name"] == filename for t in create_targets):
                    create_targets.append({
                        "name": filename,
                        "anchor": None,
                        "is_create": True,
                    })
                    logger.info("[sandbox_discovery] v1.42 CREATE target (pattern5 - then/and): '%s'", filename)
    
    # Pattern 6: Extract anchor from "on desktop" / "on D drive" if we have targets without anchors
    if create_targets:
        # Check for anchor mentions
        anchor_match = re.search(r"on\s+(?:the\s+)?(desktop|documents)", text_lower)
        if anchor_match:
            detected_anchor = anchor_match.group(1)
            for target in create_targets:
                if target.get("anchor") is None:
                    target["anchor"] = detected_anchor
                    logger.info(
                        "[sandbox_discovery] v1.42 Updated CREATE target '%s' with anchor '%s'",
                        target["name"], detected_anchor
                    )
    
    logger.info(
        "[sandbox_discovery] v1.43 extract_create_targets: found %d CREATE targets: %s",
        len(create_targets), [t["name"] for t in create_targets]
    )
    
    return create_targets


def get_system_scan_targets(text: str) -> List[dict]:
    """
    v1.40: Get file targets for system-wide scan.
    
    Returns targets marked for system-wide scan (anchor=None, scan_all_roots=True).
    """
    if not text:
        return []
    
    if not is_system_wide_scan_request(text):
        return []
    
    file_names = extract_scan_file_names(text)
    
    if not file_names:
        logger.warning("[sandbox_discovery] v1.40 System scan requested but no file names found")
        return []
    
    targets = []
    for name in file_names:
        targets.append({
            "name": name,
            "anchor": None,  # No specific anchor
            "subfolder": None,
            "explicit_path": None,
            "scan_all_roots": True,  # v1.40: Flag for system-wide scan
        })
    
    logger.info(
        "[sandbox_discovery] v1.40 get_system_scan_targets: %d targets for system-wide scan",
        len(targets)
    )
    
    return targets



