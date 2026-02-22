from __future__ import annotations
import logging
import re
from app.pot_spec.grounded._sandbox_discovery_utils_9 import SYSTEM_SCAN_INDICATORS
from typing import List
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


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
