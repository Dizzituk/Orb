# FILE: app/llm/local_tools/zobie/fs_path_utils.py
"""Filesystem path utilities for the filesystem query system.

This module handles:
- Path normalization (quotes, hidden chars, slashes)
- Path safety validation (allowlist/blocklist)

v5.3 (2026-01): Added print() debugging for OneDrive path diagnosis
v5.2 (2026-01): Fixed path normalization for hidden characters
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Tuple

from .config import (
    FILESYSTEM_QUERY_ALLOWED_ROOTS,
    FILESYSTEM_QUERY_BLOCKED_PATHS,
)

logger = logging.getLogger(__name__)


def normalize_path(path: str, debug: bool = False) -> str:
    """
    Normalize a Windows path robustly.
    
    Handles:
    - Hidden characters (\\r, \\n, \\t)
    - Curly/smart quotes -> straight quotes
    - Quote stripping (multi-pass)
    - Forward/backward slash normalization
    - Trailing backslash removal
    - os.path.normpath() for proper Windows path handling
    
    Args:
        path: Raw path string from user input
        debug: If True, print repr() of raw and normalized paths
    
    Returns:
        Normalized path string
    """
    if not path:
        return path
    
    raw_path = path  # Keep original for debug
    
    # Step 1: Strip hidden control characters FIRST
    # These can sneak in from command parsing
    path = path.replace("\r", "")
    path = path.replace("\n", "")
    path = path.replace("\t", "")
    
    # Step 2: Normalize smart/curly quotes to straight quotes
    # Left/right double quotes
    path = path.replace(""", '"')
    path = path.replace(""", '"')
    # Left/right single quotes
    path = path.replace("'", "'")
    path = path.replace("'", "'")
    # Other unicode quote variants
    path = path.replace("‹", "'")
    path = path.replace("›", "'")
    path = path.replace("«", '"')
    path = path.replace("»", '"')
    
    # Step 3: Strip whitespace
    path = path.strip()
    
    # Step 4: Multi-pass quote stripping
    # Handle quoted paths: "C:\path" or 'C:\path' or nested quotes
    quote_chars = ['"', "'"]
    for _ in range(3):  # Multiple passes for nested quotes
        for q in quote_chars:
            # Strip matching quotes from both ends
            if len(path) > 1 and path.startswith(q) and path.endswith(q):
                path = path[1:-1]
            # Also strip individual leading/trailing quotes
            path = path.strip(q)
        path = path.strip()
    
    # Step 5: Normalize slashes (forward -> backward for Windows)
    path = path.replace('/', '\\')
    
    # Step 6: Use os.path.normpath for proper Windows path handling
    # This handles:
    # - Redundant slashes: C:\\\\foo\\\\bar -> C:\foo\bar  
    # - Parent refs: C:\foo\bar\..\baz -> C:\foo\baz
    # - Current dir: C:\foo\.\bar -> C:\foo\bar
    path = os.path.normpath(path)
    
    # Step 7: Remove trailing backslash (unless it's a root like D:\)
    if len(path) > 3 and path.endswith('\\'):
        path = path.rstrip('\\')
    
    # Debug output - use print() for immediate visibility in console
    if debug:
        print(f"[PATH_NORM] raw={repr(raw_path)} -> norm={repr(path)}", file=sys.stderr)
    
    return path


def is_path_allowed(path: str) -> Tuple[bool, str]:
    """
    Check if a path is within allowed roots and not in blocked list.
    
    The allowlist/blocklist is defined in config.py and controls
    which paths can be accessed via filesystem queries.
    
    Args:
        path: Normalized path to check
    
    Returns:
        (allowed: bool, reason: str)
    """
    path_lower = path.lower().replace('/', '\\')
    
    # Check blocked paths first (these take precedence)
    for blocked in FILESYSTEM_QUERY_BLOCKED_PATHS:
        blocked_lower = blocked.lower().replace('/', '\\')
        if blocked_lower in path_lower:
            return False, f"Path contains blocked segment: {blocked}"
    
    # Check if under allowed roots
    for allowed_root in FILESYSTEM_QUERY_ALLOWED_ROOTS:
        allowed_lower = allowed_root.lower().replace('/', '\\')
        if path_lower.startswith(allowed_lower):
            return True, "Path is within allowed roots"
    
    return False, "Path is not within allowed roots"


def get_file_extension(path: str) -> str:
    """Get lowercase file extension from path."""
    return os.path.splitext(path)[1].lower()


def get_basename(path: str) -> str:
    """Get basename from path."""
    return os.path.basename(path)


def looks_like_file(path: str) -> bool:
    """
    Check if path looks like a file (has extension) vs directory.
    
    Used for heuristics when we can't check the filesystem directly.
    """
    basename = get_basename(path)
    if not basename:
        return False
    # Has extension OR is a known extensionless file type
    return '.' in basename or basename.lower() in {
        'dockerfile', 'makefile', 'readme', 'license', 'gemfile',
        'rakefile', 'procfile', 'vagrantfile', 'jenkinsfile',
    }


def get_language_from_extension(path: str) -> str:
    """
    Detect language/syntax highlighting from file extension.
    
    Returns language string for code fence, or empty string if unknown.
    """
    ext = get_file_extension(path)
    lang_map = {
        '.py': 'python', '.pyw': 'python', '.pyi': 'python',
        '.js': 'javascript', '.mjs': 'javascript', '.cjs': 'javascript',
        '.jsx': 'jsx',
        '.ts': 'typescript', '.mts': 'typescript', '.cts': 'typescript',
        '.tsx': 'tsx',
        '.json': 'json', '.jsonc': 'json',
        '.yaml': 'yaml', '.yml': 'yaml',
        '.toml': 'toml',
        '.md': 'markdown', '.markdown': 'markdown',
        '.html': 'html', '.htm': 'html',
        '.css': 'css', '.scss': 'scss', '.sass': 'sass', '.less': 'less',
        '.sql': 'sql',
        '.sh': 'bash', '.bash': 'bash', '.zsh': 'zsh',
        '.ps1': 'powershell', '.psm1': 'powershell',
        '.bat': 'batch', '.cmd': 'batch',
        '.xml': 'xml',
        '.ini': 'ini', '.cfg': 'ini', '.conf': 'ini',
        '.txt': '',
        '.rst': 'rst',
        '.rs': 'rust',
        '.go': 'go',
        '.java': 'java',
        '.kt': 'kotlin', '.kts': 'kotlin',
        '.c': 'c', '.h': 'c',
        '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp', '.hpp': 'cpp',
        '.cs': 'csharp',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.r': 'r',
        '.lua': 'lua',
        '.pl': 'perl', '.pm': 'perl',
        '.ex': 'elixir', '.exs': 'elixir',
        '.erl': 'erlang',
        '.clj': 'clojure', '.cljs': 'clojure', '.cljc': 'clojure',
        '.hs': 'haskell',
        '.scala': 'scala',
        '.dart': 'dart',
        '.vue': 'vue',
        '.svelte': 'svelte',
    }
    return lang_map.get(ext, '')
