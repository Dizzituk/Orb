# FILE: app/llm/local_tools/zobie/fs_command_parser.py
"""Command parsing for filesystem queries.

This module handles parsing of both:
- Explicit command mode: "Astra, command: read C:\\path"
- Natural language: "What's in my desktop?"

v5.2 (2026-01): Extracted from fs_query.py for modularity
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional

from .config import KNOWN_FOLDER_PATHS
from .fs_path_utils import normalize_path


def parse_command_mode(message: str) -> Optional[Dict]:
    """
    Parse explicit command mode messages.
    
    Formats supported:
    - Astra, command: <cmd> <args>
    - command: <cmd> <args>
    - <cmd> <args>  (if starts with known command)
    
    Commands:
    - list <path>
    - find <n> [under <path>]
    - read <path>
    - read "<path with spaces>"
    - head <path> [n]
    - lines <path> <start> <end>
    
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
    
    # Also strip hidden characters that can sneak in
    text_normalized = text_normalized.replace('\r', '').replace('\n', '').replace('\t', '')
    
    # Initialize variables
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


def parse_natural_language(message: str) -> Dict:
    """
    Parse natural language filesystem queries.
    
    Examples:
    - "What's in my desktop?"
    - "Show me line 45-65 of stream_router.py"
    - "First 10 lines of D:\\Orb\\main.py"
    - "Where does router.py live?"
    - "Find config.py"
    
    Returns dict with parsed info.
    """
    text = message.strip()
    text_lower = text.lower()
    
    # Strip common prefixes
    text_clean = re.sub(
        r'^(?:astra,?\s*)?(?:can you\s+)?(?:please\s+)?(?:show me\s+)?(?:tell me\s+)?',
        '', text, flags=re.IGNORECASE
    ).strip()
    
    result = {
        "query_type": None,
        "path": None,
        "search_term": None,
        "start_line": None,
        "end_line": None,
        "head_lines": None,
    }
    
    # Extract Windows path if present
    path_match = re.search(r'([A-Za-z]:[/\\][^\s"\'<>|?*\n]*)', text)
    if path_match:
        result["path"] = normalize_path(path_match.group(1), debug=True)
    else:
        # Check for known folder keywords
        for folder, path in KNOWN_FOLDER_PATHS.items():
            if folder in text_lower:
                result["path"] = path
                break
    
    # Detect line range queries: "line 45-65", "lines 45 to 65", "lines 45-65"
    line_range_match = re.search(
        r'lines?\s+(\d+)\s*[-–—to]+\s*(\d+)',
        text_lower
    )
    if line_range_match:
        result["query_type"] = "lines"
        result["start_line"] = int(line_range_match.group(1))
        result["end_line"] = int(line_range_match.group(2))
        return result
    
    # Detect head queries: "first N lines", "top N lines", "head N"
    head_match = re.search(
        r'(?:first|top|head)\s+(\d+)\s+lines?',
        text_lower
    )
    if head_match:
        result["query_type"] = "head"
        result["head_lines"] = int(head_match.group(1))
        return result
    
    # Detect "what's in line X" - single line
    single_line_match = re.search(
        r"(?:what'?s\s+(?:in|on|at)\s+)?line\s+(\d+)",
        text_lower
    )
    if single_line_match and not line_range_match:
        result["query_type"] = "lines"
        line_num = int(single_line_match.group(1))
        result["start_line"] = line_num
        result["end_line"] = line_num
        return result
    
    # Detect "where does X live" / "where is X"
    where_match = re.search(
        r'where\s+(?:does|is)\s+([^\s?]+)',
        text_lower
    )
    if where_match:
        result["query_type"] = "find"
        result["search_term"] = where_match.group(1)
        return result
    
    # Detect read queries
    read_patterns = [
        r"what'?s\s+(?:written|inside)\s+(?:in\s+)?",
        r"read\s+(?:the\s+)?(?:file\s+)?",
        r"show\s+(?:me\s+)?(?:the\s+)?contents?\s+of\s+",
        r"(?:display|view|print|output|cat)\s+(?:the\s+)?(?:file\s+)?",
        r"what\s+does\s+.+\s+(?:say|contain)",
        r"open\s+(?:the\s+)?(?:file\s+)?",
    ]
    for pattern in read_patterns:
        if re.search(pattern, text_lower):
            if result["path"]:
                # Check if path looks like a file
                basename = os.path.basename(result["path"])
                if basename and '.' in basename:
                    result["query_type"] = "read"
                    return result
    
    # Detect list queries
    list_patterns = [
        r"^list\s+",
        r"^show\s+(?:me\s+)?(?:everything|all|contents?|files?|folders?)\s+(?:in|on|at|under)",
        r"what'?s\s+(?:in|on)\s+",
        r"contents?\s+of\s+",
    ]
    for pattern in list_patterns:
        if re.search(pattern, text_lower):
            result["query_type"] = "list"
            return result
    
    # Detect find queries
    find_patterns = [
        r"^find\s+",
        r"search\s+for\s+",
        r"locate\s+",
    ]
    for pattern in find_patterns:
        if re.search(pattern, text_lower):
            result["query_type"] = "find"
            # Extract search term
            find_match = re.search(
                r'(?:find|search\s+for|locate)\s+(?:file\s+|folder\s+)?([^\s]+)',
                text_lower
            )
            if find_match:
                result["search_term"] = find_match.group(1)
            return result
    
    # Default: if we have a path that looks like a file, treat as read
    if result["path"]:
        basename = os.path.basename(result["path"])
        if basename and '.' in basename:
            result["query_type"] = "read"
        else:
            result["query_type"] = "list"
    
    return result


def parse_filesystem_query(message: str) -> Dict:
    """
    Unified parser that handles both command mode and natural language.
    
    Returns dict with:
    - query_type: "list", "find", "read", "head", or "lines"
    - path: Target path (normalized)
    - search_term: For find queries
    - start_line, end_line: For lines queries
    - head_lines: For head queries
    - source: "command" or "natural"
    """
    # Try command mode first
    cmd_result = parse_command_mode(message)
    if cmd_result and cmd_result.get("query_type"):
        cmd_result["source"] = "command"
        return cmd_result
    
    # Fall back to natural language parsing
    nl_result = parse_natural_language(message)
    nl_result["source"] = "natural"
    return nl_result
