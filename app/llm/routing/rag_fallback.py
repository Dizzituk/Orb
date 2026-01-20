# FILE: app/llm/routing/rag_fallback.py
"""
RAG fallback detection for architecture/codebase queries.

v1.0 (2026-01-20): Extracted from stream_router.py (v4.12 RAG fallback).

This module provides:
- High-precision patterns for detecting architecture questions
- `is_architecture_query()` - Check if message should trigger RAG fallback

Used when translation layer fails or returns None to catch plain-English
codebase questions that would otherwise fall through to chat mode.
"""

from __future__ import annotations

import re
from typing import List

# =============================================================================
# ARCHITECTURE QUERY PATTERNS (v4.12)
# =============================================================================

# High-precision patterns for architecture/codebase questions
# These catch queries when translation layer fails/returns None
_ARCHITECTURE_QUERY_PATTERNS: List[re.Pattern] = [
    # "Where is X" patterns (broad - catches most arch questions)
    re.compile(
        r"^[Ww]here\s+is\s+(?:the\s+)?(?:main\s+)?(?:\w+\s+){0,6}"
        r"(?:entrypoint|entry\s*point|router|stream|handler|function|class|module|file|config|constant|routing|implementation|trigger|pipeline|gate)[s]?"
    ),
    
    # "Show me where X" patterns
    re.compile(
        r"^[Ss]how\s+(?:me\s+)?where\s+.+"
        r"(?:is\s+)?(?:implemented|defined|located|triggered|called|used|loaded|handled|routed|processed)"
    ),
    
    # "Find where X" patterns
    re.compile(
        r"^[Ff]ind\s+(?:where|the\s+file\s+where)\s+.+"
        r"(?:is\s+)?(?:implemented|defined|located|triggered|called|used|loaded|handled|routed|processed|routes)"
    ),
    
    # "Find the file where X" patterns
    re.compile(
        r"^[Ff]ind\s+(?:the\s+)?(?:file|module|class|function)\s+(?:where|that)\s+.+"
    ),
    
    # "Find/List/Show call sites/callers of X" patterns
    re.compile(
        r"^(?:[Ff]ind|[Ll]ist|[Ss]how)\s+(?:the\s+)?(?:call\s*sites?|callers?)\s+(?:of|for)\s+.+"
    ),
    
    # "Who calls X" patterns
    re.compile(r"^[Ww]ho\s+calls\s+.+"),
    
    # Codebase-specific questions with known ASTRA terms
    re.compile(
        r"^(?:[Ww]here|[Hh]ow|[Ww]hat)\s+.+"
        r"(?:[Ss]pec\s*[Gg]ate|[Oo]verwatcher|[Ww]eaver|[Cc]ritical\s*[Pp]ipeline|"
        r"[Ss]tream\s*[Rr]outer|[Tt]ranslation\s*[Ll]ayer|[Rr][Aa][Gg]|[Ee]mbedding)"
        r"\s*.+[?.!]?$"
    ),
]

# Command prefix pattern - never route explicit commands to RAG fallback
_COMMAND_PREFIX_PATTERN = re.compile(r"^[Aa]stra[,:]?\s*command:\s*", re.IGNORECASE)


def is_architecture_query(message: str) -> bool:
    """
    Check if a message is a plain-English architecture/codebase question.
    
    Used as a fallback when translation layer fails or returns None.
    High-precision patterns only - no broad catch-alls.
    
    Args:
        message: User message to check
    
    Returns:
        True if message matches an architecture query pattern
        False for explicit command prefixes (e.g. "Astra, command: ...")
    """
    text = message.strip()
    
    # Never trigger for explicit command prefix
    if _COMMAND_PREFIX_PATTERN.match(text):
        return False
    
    # Check against high-precision patterns
    for pattern in _ARCHITECTURE_QUERY_PATTERNS:
        if pattern.match(text):
            return True
    
    return False


def get_pattern_count() -> int:
    """Get the number of architecture query patterns (for testing)."""
    return len(_ARCHITECTURE_QUERY_PATTERNS)


__all__ = [
    "is_architecture_query",
    "get_pattern_count",
]
