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
from app.pot_spec.grounded._sandbox_discovery_utils import OutputMode, SANDBOX_DISCOVERY_BUILD_ID, SYSTEM_SCAN_ROOTS, extract_create_targets, extract_replacement_text, is_multi_target_request
from app.pot_spec.grounded._sandbox_discovery_utils import SYSTEM_SCAN_INDICATORS, detect_output_mode, get_system_scan_targets
from app.pot_spec.grounded._sandbox_discovery_utils import extract_scan_file_names, is_system_wide_scan_request
from app.pot_spec.grounded._sandbox_discovery_utils import FILENAME_STOPWORDS, extract_file_targets

logger = logging.getLogger(__name__)

# =============================================================================
# v1.35 BUILD VERIFICATION - Proves correct code is running
# v1.35: CRITICAL FIX - Removed overly broad unquoted Weaver patterns
# =============================================================================
print(f"[SANDBOX_DISCOVERY_LOADED] BUILD_ID={SANDBOX_DISCOVERY_BUILD_ID}")
logger.info(f"[sandbox_discovery] Module loaded: BUILD_ID={SANDBOX_DISCOVERY_BUILD_ID}")


# =============================================================================
# OUTPUT MODE ENUM
# =============================================================================


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


# =============================================================================
# OUTPUT MODE DETECTION (v1.13, v1.16-v1.17, v1.24)
# =============================================================================


# =============================================================================
# REPLACEMENT TEXT EXTRACTION (v1.31)
# =============================================================================


# =============================================================================
# SYSTEM-WIDE SCAN SUPPORT (v1.40 - Level 2.5+)
# =============================================================================

# All roots to scan for system-wide searches

# Patterns that indicate system-wide scan intent



