# FILE: app/llm/local_tools/zobie/fs_command_parser.py
# Purpose: Command parsing for filesystem queries and writes.
# Called-by: app.llm.local_tools.zobie._fs_command_parser_utils_2, app.llm.local_tools.zobie.streams._fs_query_utils_3, app.llm.local_tools.zobie.streams.fs_query
# Depends-on: app.llm.local_tools.zobie._fs_command_parser_utils_2, app.llm.local_tools.zobie.config, app.llm.local_tools.zobie.fs_path_utils
# Last-renovated: 2026-06-11
"""Command parsing for filesystem queries and writes.

This module handles parsing of both:
- Explicit command mode: "Astra, command: read C:\\path"
- Natural language: "What's in my desktop?"

v5.7 (2026-01): Fixed multi-line quoted content parsing for overwrite/append
  - Now properly handles quoted strings that span multiple lines
  - Added _extract_quoted_arguments() tokenizer that respects quotes across newlines
  - Better error messages showing what was parsed
  - Fenced block support unchanged (already worked)
v5.6 (2026-01): Fixed find command parsing - quoted value is search_term, not path
  - find "ME" -> search_term="ME", path=None
  - find "ME" in "C:\\path" -> search_term="ME", path="C:\\path"
v5.5 (2026-01): Added Stage 1 write commands (append, overwrite, delete_area)
v5.2 (2026-01): Extracted from fs_query.py for modularity
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .config import KNOWN_FOLDER_PATHS
from .fs_path_utils import normalize_path
from app.llm.local_tools.zobie._fs_command_parser_utils_2 import _extract_fenced_content, _extract_path_from_text, _extract_quoted_arguments, _parse_find_command, parse_filesystem_query, parse_natural_language


def parse_command_mode(message: str) -> Optional[Dict]:
    """
    Parse explicit command mode messages.
    
    Formats supported:
    - Astra, command: <cmd> <args>
    - command: <cmd> <args>
    - <cmd> <args>  (if starts with known command)
    
    Read Commands:
    - list <path>
    - find <term> [under <path>]
    - find "<term>" [in "<path>"]
    - read <path>
    - read "<path with spaces>"
    - head <path> [n]
    - lines <path> <start> <end>
    
    Write Commands (Stage 1):
    - append <path> "content"
    - append <path>
      ```
      multiline content
      ```
    - overwrite <path> "content"
    - overwrite <path>
      ```
      multiline content
      ```
    - delete_area <path>  (uses default markers)
    - delete_lines <path> <start> <end>
    
    Returns dict with parsed command or None if not a command.
    """
    text = message.strip()
    
    # Strip "Astra, command:" or "command:" prefix
    prefix_match = re.match(
        r'^(?:astra,?\s*)?command:?\s*',
        text, re.IGNORECASE
    )
    if prefix_match:
        text = text[prefix_match.end():].strip()
    
    if not text:
        return None
    
    # Normalize curly quotes to straight quotes for easier parsing
    text_normalized = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
    
    # Also strip carriage returns (but NOT newlines - we need those for multi-line content!)
    text_normalized = text_normalized.replace('\r', '')
    
    # Check for write commands FIRST (they have special content parsing)
    write_result = _parse_write_command(text_normalized)
    if write_result:
        return write_result
    
    # Check for find command with quoted search term BEFORE general quoted matching
    find_result = _parse_find_command(text_normalized)
    if find_result:
        return find_result
    
    # Initialize variables for read commands
    cmd = None
    path = ''
    extra = ''
    
    # Try to match: <command> "<quoted_path>" [additional_args]
    # Pattern matches: cmd "path" optional_extra
    quoted_match = re.match(
        r'^(\w+)\s+["\']([^"\']+)["\']\s*(.*)$',
        text_normalized
    )
    
    if quoted_match:
        cmd = quoted_match.group(1).lower()
        path = quoted_match.group(2)
        extra = quoted_match.group(3).strip()
    else:
        # No quotes - split on whitespace but be smart about Windows paths
        parts = text_normalized.split()
        if not parts:
            return None
        cmd = parts[0].lower()
        
        # For commands that take a path, try to reconstruct it
        if cmd in ('list', 'read', 'head', 'lines', 'find'):
            # Find where the path likely ends
            # A Windows path starts with X:\ and continues until we hit
            # something that looks like a number (for head/lines) or keyword
            path_parts = []
            extra_parts = []
            in_path = False
            path_done = False
            
            for i, part in enumerate(parts[1:], 1):
                if not in_path:
                    # Check if this looks like start of Windows path
                    if re.match(r'^[A-Za-z]:[/\\]', part):
                        in_path = True
                        path_parts.append(part)
                    elif part.lower() in KNOWN_FOLDER_PATHS:
                        # Known folder keyword
                        path_parts.append(KNOWN_FOLDER_PATHS[part.lower()])
                        path_done = True
                    else:
                        # For 'find', first arg might be search term
                        if cmd == 'find' and not path_parts:
                            extra_parts.append(part)
                        else:
                            extra_parts.append(part)
                elif in_path and not path_done:
                    # Check if this part looks like end of path
                    # Numbers for head/lines, or "under/in" for find
                    if part.isdigit():
                        path_done = True
                        extra_parts.append(part)
                    elif part.lower() in ('under', 'in', 'inside', 'on'):
                        path_done = True
                        extra_parts.append(part)
                    else:
                        path_parts.append(part)
                else:
                    extra_parts.append(part)
            
            path = ' '.join(path_parts) if path_parts else ''
            extra = ' '.join(extra_parts)
        else:
            path = ''
            extra = ' '.join(parts[1:])
    
    # Normalize the path (uses the robust normalize_path from fs_path_utils)
    path = normalize_path(path, debug=True) if path else ''
    
    result = {
        "command": cmd,
        "path": path,
        "extra": extra,
        "query_type": None,
        "start_line": None,
        "end_line": None,
        "head_lines": None,
        "search_term": None,
        "content": None,  # For write commands
    }
    
    # Parse command-specific arguments
    if cmd == "list":
        result["query_type"] = "list"
        
    elif cmd == "read":
        result["query_type"] = "read"
        
    elif cmd == "head":
        result["query_type"] = "head"
        # Parse optional line count
        extra_parts = extra.split()
        if extra_parts and extra_parts[0].isdigit():
            result["head_lines"] = int(extra_parts[0])
        else:
            result["head_lines"] = 20  # Default
            
    elif cmd == "lines":
        result["query_type"] = "lines"
        # Parse start and end line numbers
        extra_parts = extra.split()
        if len(extra_parts) >= 2:
            try:
                result["start_line"] = int(extra_parts[0])
                result["end_line"] = int(extra_parts[1])
            except ValueError:
                pass
                
    elif cmd == "find":
        result["query_type"] = "find"
        # Parse: find <term> [under <path>]
        extra_lower = extra.lower()
        under_match = re.search(r'\s+(?:under|in|inside|on)\s+', extra_lower)
        if under_match:
            result["search_term"] = extra[:under_match.start()].strip()
            path_part = extra[under_match.end():].strip()
            if path_part:
                result["path"] = normalize_path(path_part, debug=True)
        else:
            result["search_term"] = extra.strip() if extra else None
    else:
        # Unknown command
        return None
    
    return result


def _parse_write_command(text: str) -> Optional[Dict]:
    """
    Parse write commands: append, overwrite, delete_area, delete_lines.
    
    Supported formats:
    1. Inline quoted (single-line): append <path> "content here"
    2. Inline quoted (multi-line): append "<path>" "multi
       line content here"
    3. Fenced block:
       append <path>
       ```
       content here
       ```
    4. Delete area (markers): delete_area <path>
    5. Delete lines: delete_lines <path> <start> <end>
    
    Returns parsed dict or None if not a write command.
    
    v5.7: Fixed multi-line quoted content by using proper tokenizer.
    Previously split on newlines first which broke multi-line quoted strings.
    """
    # Get first line to determine command type
    first_newline = text.find('\n')
    first_line = text[:first_newline] if first_newline != -1 else text
    first_line = first_line.strip()
    first_line_lower = first_line.lower()
    
    # Check for write commands
    write_commands = ['append', 'overwrite', 'delete_area', 'delete_lines']
    
    cmd = None
    for wc in write_commands:
        if first_line_lower.startswith(wc + ' ') or first_line_lower == wc:
            cmd = wc
            break
    
    if not cmd:
        return None
    
    result = {
        "command": cmd,
        "path": "",
        "extra": "",
        "query_type": cmd,  # append, overwrite, delete_area, delete_lines
        "start_line": None,
        "end_line": None,
        "head_lines": None,
        "search_term": None,
        "content": None,
    }
    
    # Get everything after the command name on the first line
    rest_of_first_line = first_line[len(cmd):].strip()
    
    # ==========================================================================
    # DELETE_AREA: delete_area <path>
    # ==========================================================================
    if cmd == "delete_area":
        # Just needs path (on first line only)
        path = _extract_path_from_text(rest_of_first_line)
        if path:
            result["path"] = normalize_path(path, debug=True)
        return result
    
    # ==========================================================================
    # DELETE_LINES: delete_lines <path> <start> <end>
    # ==========================================================================
    if cmd == "delete_lines":
        # Parse: delete_lines <path> <start> <end>
        # or: delete_lines "<path with spaces>" <start> <end>
        
        # Try quoted path first
        quoted_match = re.match(r'^["\']([^"\']+)["\']\s+(\d+)\s+(\d+)', rest_of_first_line)
        if quoted_match:
            result["path"] = normalize_path(quoted_match.group(1), debug=True)
            result["start_line"] = int(quoted_match.group(2))
            result["end_line"] = int(quoted_match.group(3))
            return result
        
        # Try unquoted path (numbers at end)
        parts = rest_of_first_line.split()
        if len(parts) >= 3:
            try:
                end_line = int(parts[-1])
                start_line = int(parts[-2])
                path = ' '.join(parts[:-2])
                result["path"] = normalize_path(path, debug=True)
                result["start_line"] = start_line
                result["end_line"] = end_line
                return result
            except ValueError:
                pass
        
        return result
    
    # ==========================================================================
    # APPEND / OVERWRITE: Need path + content
    # ==========================================================================
    
    # Split full text into lines for fenced block detection
    lines = text.split('\n')
    
    # Check for fenced block content FIRST (```...```)
    # Fenced blocks start on line 2 or later
    fenced_content = _extract_fenced_content(lines[1:]) if len(lines) > 1 else None
    
    if fenced_content is not None:
        # Path is on first line after command
        path = _extract_path_from_text(rest_of_first_line)
        if path:
            result["path"] = normalize_path(path, debug=True)
        result["content"] = fenced_content
        return result
    
    # ==========================================================================
    # MULTI-LINE QUOTED CONTENT HANDLING (v5.7 fix)
    # ==========================================================================
    # For append/overwrite with quoted content, we need the FULL text
    # (not just first line) to handle multi-line quoted strings.
    
    # Get everything after the command name in the FULL text
    # Find where the command starts and skip past it
    cmd_match = re.match(r'^' + re.escape(cmd) + r'\s*', text, re.IGNORECASE)
    if cmd_match:
        full_args = text[cmd_match.end():]
    else:
        full_args = text[len(cmd):].strip()
    
    # Use tokenizer to extract arguments (respects quotes across newlines)
    args = _extract_quoted_arguments(full_args, debug=True)
    
    if len(args) >= 2:
        # First arg is path, second arg is content
        result["path"] = normalize_path(args[0], debug=True)
        result["content"] = args[1]
        return result
    elif len(args) == 1:
        # Only one argument - is it path or content?
        arg = args[0]
        # If it looks like a path (has drive letter or is clearly a file path)
        if re.match(r'^[A-Za-z]:[/\\]', arg) or '\\' in arg or '/' in arg:
            result["path"] = normalize_path(arg, debug=True)
            # Content is None - caller will see "no content" error
        else:
            # Doesn't look like a path - could be path without drive letter
            # or could be content without path (which is an error anyway)
            result["path"] = normalize_path(arg, debug=True)
        return result
    
    # No arguments found
    return result
