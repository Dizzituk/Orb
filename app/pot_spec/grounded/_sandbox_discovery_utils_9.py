from __future__ import annotations
import logging
import re
from app.pot_spec.grounded._sandbox_discovery_utils_8 import OutputMode
from typing import List
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


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

def get_system_scan_targets(text: str) -> List[dict]:
    """
    v1.40: Get file targets for system-wide scan.
    
    Returns targets marked for system-wide scan (anchor=None, scan_all_roots=True).
    """
    from .sandbox_discovery import extract_scan_file_names, is_system_wide_scan_request
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
