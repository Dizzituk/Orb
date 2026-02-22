import os
import re
import sys
from .config import KNOWN_FOLDER_PATHS
from .fs_path_utils import normalize_path
from typing import Dict, List, Optional


def _parse_find_command(text: str) -> Optional[Dict]:
    """
    Parse find command with special handling for quoted search terms.
    
    Supported formats:
    - find "search term"
    - find "search term" in "C:\\path"
    - find "search term" under "C:\\path"
    - find "search term" inside "C:\\path"
    - find "search term" on "C:\\path"
    
    For find, the first quoted value is ALWAYS the search term, NOT a path.
    Optional second quoted value (after in/under/inside/on) is the path.
    
    Returns parsed dict or None if not a find command.
    """
    text_lower = text.lower()
    
    # Check if this starts with "find"
    if not text_lower.startswith('find ') and text_lower != 'find':
        return None
    
    # Remove "find " prefix
    rest = text[5:].strip() if len(text) > 5 else ""
    
    if not rest:
        return None
    
    result = {
        "command": "find",
        "path": None,
        "extra": "",
        "query_type": "find",
        "start_line": None,
        "end_line": None,
        "head_lines": None,
        "search_term": None,
        "content": None,
    }
    
    # Check for: find "term" [in/under/inside/on "path"]
    # Pattern: "term" [keyword "path"]
    
    # Match first quoted value (search term)
    first_quote_match = re.match(r'^["\']([^"\']+)["\'](.*)$', rest)
    
    if first_quote_match:
        search_term = first_quote_match.group(1)
        remainder = first_quote_match.group(2).strip()
        
        result["search_term"] = search_term
        
        # Check for optional path: in/under/inside/on "path" or in/under/inside/on path
        if remainder:
            path_keyword_match = re.match(
                r'^(?:in|under|inside|on)\s+(.+)$',
                remainder,
                re.IGNORECASE
            )
            if path_keyword_match:
                path_part = path_keyword_match.group(1).strip()
                # Remove quotes if present
                if len(path_part) >= 2:
                    if (path_part.startswith('"') and path_part.endswith('"')) or \
                       (path_part.startswith("'") and path_part.endswith("'")):
                        path_part = path_part[1:-1]
                if path_part:
                    result["path"] = normalize_path(path_part, debug=True)
        
        return result
    
    # No quotes - check if it's a simple unquoted search term
    # Only handle this if there's no Windows path pattern (let the main parser handle paths)
    if not re.match(r'^[A-Za-z]:[/\\]', rest):
        # Check for: find term [in/under/inside/on path]
        keyword_match = re.search(r'\s+(?:in|under|inside|on)\s+', rest, re.IGNORECASE)
        if keyword_match:
            search_term = rest[:keyword_match.start()].strip()
            path_part = rest[keyword_match.end():].strip()
            result["search_term"] = search_term
            if path_part:
                # Remove quotes if present
                if len(path_part) >= 2:
                    if (path_part.startswith('"') and path_part.endswith('"')) or \
                       (path_part.startswith("'") and path_part.endswith("'")):
                        path_part = path_part[1:-1]
                result["path"] = normalize_path(path_part, debug=True)
            return result
        else:
            # Simple: find term (no path)
            result["search_term"] = rest.strip()
            return result
    
    # Has a Windows path pattern - let the main parser handle it
    return None

def _extract_quoted_arguments(text: str, debug: bool = False) -> List[str]:
    """
    Extract quoted arguments from text, handling multi-line strings.
    
    Tokenizes the text respecting quoted strings that may span multiple lines.
    Returns list of extracted quoted/unquoted arguments.
    
    Examples:
        '"path" "content"' -> ['path', 'content']
        '"path" "multi\\nline"' -> ['path', 'multi\\nline']
        'unquoted_path "content"' -> ['unquoted_path', 'content']
        '"D:\\path\\file.py" "# comment\\ndef foo():\\n    pass"' 
            -> ['D:\\path\\file.py', '# comment\\ndef foo():\\n    pass']
    
    v5.7: New function to properly handle multi-line quoted content.
    """
    args = []
    pos = 0
    text_len = len(text)
    
    if debug:
        print(f"[TOKENIZER] input_len={text_len} preview={repr(text[:80])}{'...' if len(text) > 80 else ''}", file=sys.stderr)
    
    while pos < text_len:
        # Skip whitespace (spaces, tabs, newlines between arguments)
        while pos < text_len and text[pos] in ' \t\n':
            pos += 1
        
        if pos >= text_len:
            break
        
        char = text[pos]
        
        # Quoted argument - handles multi-line content within quotes
        if char in '"\'':
            quote_char = char
            pos += 1  # Skip opening quote
            arg_start = pos
            
            # Find closing quote (can span multiple lines!)
            # We do NOT stop at newlines - the quote must be closed
            while pos < text_len:
                if text[pos] == quote_char:
                    # Found closing quote (not escaped)
                    # Check if previous char was backslash (escape)
                    # But be careful about \\\" (escaped backslash + quote)
                    num_backslashes = 0
                    check_pos = pos - 1
                    while check_pos >= arg_start and text[check_pos] == '\\':
                        num_backslashes += 1
                        check_pos -= 1
                    
                    # If odd number of backslashes, the quote is escaped
                    if num_backslashes % 2 == 0:
                        # Even backslashes (or zero) = real closing quote
                        arg_content = text[arg_start:pos]
                        args.append(arg_content)
                        pos += 1  # Skip closing quote
                        if debug:
                            print(f"[TOKENIZER] found quoted arg: {repr(arg_content[:50])}{'...' if len(arg_content) > 50 else ''}", file=sys.stderr)
                        break
                    else:
                        # Escaped quote - continue
                        pos += 1
                else:
                    pos += 1
            else:
                # No closing quote found - take rest as partial arg
                arg_content = text[arg_start:]
                args.append(arg_content)
                if debug:
                    print(f"[TOKENIZER] unclosed quote, took rest: {repr(arg_content[:50])}...", file=sys.stderr)
        else:
            # Unquoted argument - read until whitespace or quote
            arg_start = pos
            while pos < text_len and text[pos] not in ' \t\n"\'':
                pos += 1
            if pos > arg_start:
                arg_content = text[arg_start:pos]
                args.append(arg_content)
                if debug:
                    print(f"[TOKENIZER] found unquoted arg: {repr(arg_content)}", file=sys.stderr)
    
    if debug:
        print(f"[TOKENIZER] result: {len(args)} args", file=sys.stderr)
    
    return args

def _extract_path_from_text(text: str) -> str:
    """Extract a path from text, handling quotes."""
    text = text.strip()
    
    # Remove surrounding quotes
    if len(text) >= 2:
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1]
    
    return text

def _extract_fenced_content(lines: list) -> Optional[str]:
    """
    Extract content from a fenced code block.
    
    Looks for:
    ```
    content here
    ```
    
    Returns the content between fences, or None if no valid fence found.
    """
    if not lines:
        return None
    
    # Find opening fence
    start_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('```'):
            start_idx = i
            break
    
    if start_idx is None:
        return None
    
    # Find closing fence
    end_idx = None
    for i in range(start_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped == '```' or stripped.startswith('```'):
            end_idx = i
            break
    
    if end_idx is None:
        return None
    
    # Extract content between fences
    content_lines = lines[start_idx + 1:end_idx]
    return '\n'.join(content_lines)

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
        "content": None,
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
    - query_type: "list", "find", "read", "head", "lines", 
                  "append", "overwrite", "delete_area", "delete_lines"
    - path: Target path (normalized)
    - search_term: For find queries
    - start_line, end_line: For lines/delete_lines queries
    - head_lines: For head queries
    - content: For append/overwrite queries
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
