import logging
import re
from typing import List
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


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
