from __future__ import annotations
import base64
import logging
import re
from app.overwatcher.implementer import logger
from typing import List, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


IMPLEMENTER_BUILD_ID = "2026-02-14-v1.15-temp-path-fix-dynamic-parent-dir"

INLINE_BASE64_CHAR_LIMIT = 24000  # Safe threshold with overhead for PS wrapper

def _escape_powershell_string(s: str) -> str:
    """Escape a string for use in PowerShell double-quoted strings.
    
    NOTE: For complex content with embedded quotes/newlines, use
    _build_powershell_write_command_base64() instead - it's more reliable.
    """
    return s.replace('`', '``').replace('"', '`"').replace('$', '`$')

def _encode_for_powershell_base64(content: str) -> str:
    """
    Encode content as Base64 for safe PowerShell transmission.
    
    v1.8: This is more robust than escaping for complex content with
    embedded quotes, newlines, and special characters.
    """
    encoded = base64.b64encode(content.encode('utf-8')).decode('ascii')
    return encoded

def _build_powershell_write_command_base64(path: str, content: str) -> str:
    """
    Build a PowerShell command that writes content using Base64 encoding.
    
    v1.8: Uses Base64 to safely transmit complex content with quotes/newlines.
    This avoids all escaping issues that occur with embedded quotes like
    "works on my machine".
    """
    encoded = _encode_for_powershell_base64(content)
    # Decode Base64 in PowerShell, then write to file
    return (
        f'[System.Text.Encoding]::UTF8.GetString('
        f'[System.Convert]::FromBase64String("{encoded}"))'
        f' | Set-Content -Path "{path}" -NoNewline -Encoding UTF8'
    )

def _generate_sandbox_path_candidates(path: str) -> List[str]:
    """Generate candidate paths for sandbox resolution."""
    candidates = [path]
    
    # Match: C:\Users\<username>\OneDrive\Desktop\<rest>
    onedrive_match = re.match(
        r'^([A-Za-z]):\\Users\\([^\\]+)\\OneDrive\\Desktop\\(.*)$',
        path,
        re.IGNORECASE
    )
    if onedrive_match:
        drive = onedrive_match.group(1)
        username = onedrive_match.group(2)
        rest = onedrive_match.group(3)
        
        non_onedrive = f"{drive}:\\Users\\{username}\\Desktop\\{rest}"
        if non_onedrive not in candidates:
            candidates.append(non_onedrive)
        
        wdag_onedrive = f"{drive}:\\Users\\WDAGUtilityAccount\\OneDrive\\Desktop\\{rest}"
        if wdag_onedrive not in candidates:
            candidates.append(wdag_onedrive)
        
        wdag = f"{drive}:\\Users\\WDAGUtilityAccount\\Desktop\\{rest}"
        if wdag not in candidates:
            candidates.append(wdag)
        
        return candidates
    
    # Match: C:\Users\<username>\Desktop\<rest> (non-OneDrive)
    desktop_match = re.match(
        r'^([A-Za-z]):\\Users\\([^\\]+)\\Desktop\\(.*)$',
        path,
        re.IGNORECASE
    )
    if desktop_match:
        drive = desktop_match.group(1)
        username = desktop_match.group(2)
        rest = desktop_match.group(3)
        
        wdag = f"{drive}:\\Users\\WDAGUtilityAccount\\Desktop\\{rest}"
        if wdag not in candidates:
            candidates.append(wdag)
        
        return candidates
    
    return candidates

def _find_question_block_starts(text: str) -> List[Tuple[int, int, str]]:
    """
    Find all question block start positions in text.
    
    Returns list of (line_number, char_position, question_identifier) tuples.
    
    Question detection patterns:
    - Pattern 1: Lines starting with "Question N:" (case-insensitive)
    - Pattern 2: Lines ending with "?" that start with "N." or "N)" 
    - Pattern 3 (v1.7): Standalone numbered lines like "1)" or "2."
    """
    blocks: List[Tuple[int, int, str]] = []
    lines = text.split('\n')
    char_pos = 0
    
    for line_num, line in enumerate(lines):
        line_stripped = line.strip()
        
        # Pattern 1: "Question N:" (case-insensitive)
        question_match = re.match(r'^question\s*(\d+)\s*[:\.]', line_stripped, re.IGNORECASE)
        if question_match:
            q_num = question_match.group(1)
            blocks.append((line_num, char_pos, q_num))
            char_pos += len(line) + 1
            continue
        
        # Pattern 2: Numbered line ending with "?" (e.g., "1. How?" or "2) Why?")
        numbered_question_match = re.match(r'^(\d+)[.\)]\s*', line_stripped)
        if numbered_question_match and line_stripped.rstrip().endswith('?'):
            q_num = numbered_question_match.group(1)
            blocks.append((line_num, char_pos, q_num))
            char_pos += len(line) + 1
            continue
        
        # Pattern 3 (v1.7): Standalone numbered line (e.g., "1)" or "2.")
        standalone_match = re.match(r'^(\d+)[.\)]\s*$', line_stripped)
        if standalone_match:
            q_num = standalone_match.group(1)
            blocks.append((line_num, char_pos, q_num))
            logger.debug("[implementer] v1.7 Pattern 3 matched: line %d = %r -> Q%s", line_num, line_stripped, q_num)
        
        char_pos += len(line) + 1
    
    return blocks

def _block_has_answer(block_text: str) -> bool:
    """Check if a question block already has a non-empty answer.
    
    v1.9: Detect both "Answer" and "Answer:" patterns (with or without colon).
    """
    lines = block_text.split('\n')
    
    for i, line in enumerate(lines):
        line_stripped = line.strip().lower()
        
        # Match "answer" or "answer:" (with or without colon)
        if line_stripped == 'answer' or line_stripped.startswith('answer:'):
            # Check content AFTER this line
            remaining_lines = lines[i + 1:]
            remaining_text = '\n'.join(remaining_lines).strip()
            
            if remaining_text:
                # There's content after the Answer line
                return True
    
    return False
