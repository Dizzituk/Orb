from __future__ import annotations
import logging
import re
from enum import Enum
from typing import List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


SANDBOX_DISCOVERY_BUILD_ID = "2026-01-30-v1.43-weaver-output-format"

class OutputMode(str, Enum):
    """Output mode for MICRO_FILE_TASK jobs."""
    APPEND_IN_PLACE = "append_in_place"
    REWRITE_IN_PLACE = "rewrite_in_place"
    SEPARATE_REPLY_FILE = "separate_reply_file"
    CHAT_ONLY = "chat_only"
    OVERWRITE_FULL = "overwrite_full"

def is_multi_target_request(text: str) -> bool:
    """
    v1.32: Check if text contains a multi-target file read request.
    
    Returns True if the text mentions multiple files with different locations.
    """
    from .sandbox_discovery import extract_file_targets
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
    from .sandbox_discovery import FILENAME_STOPWORDS
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
