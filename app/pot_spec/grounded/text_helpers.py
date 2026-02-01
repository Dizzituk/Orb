# FILE: app/pot_spec/grounded/text_helpers.py
"""
Text Extraction Helpers for SpecGate

This module provides utility functions for extracting meaningful information
from text content, including file paths and keywords.

Responsibilities:
- Extract file/directory paths from text using pattern matching
- Extract meaningful keywords from text with stopword filtering
- Support path patterns for common project structures (app/, src/, tests/)

Used by:
- grounding_engine.py for path verification
- Other modules needing text analysis

Version: v2.0 (2026-02-01) - Extracted from spec_generation.py
"""

from __future__ import annotations

import re
from typing import List

__all__ = [
    "_extract_paths_from_text",
    "_extract_keywords",
]


def _extract_paths_from_text(text: str) -> List[str]:
    """
    Extract file/directory paths from text.
    
    Supports:
    - Backtick-quoted paths with common extensions
    - Single/double quoted paths with common extensions
    - Bare paths starting with app/, src/, tests/
    
    Args:
        text: Text to extract paths from
        
    Returns:
        List of extracted path strings
    """
    if not text:
        return []
    
    patterns = [
        r'`([^`]+\.(?:py|ts|tsx|js|jsx|json|md|yaml|yml))`',  # backtick paths
        r'[\'"]([^\'"]+\.(?:py|ts|tsx|js|jsx|json|md|yaml|yml))[\'"]',  # quoted paths
        r'(?:^|\s)(app/[^\s]+)',  # app/ paths
        r'(?:^|\s)(src/[^\s]+)',  # src/ paths
        r'(?:^|\s)(tests/[^\s]+)',  # tests/ paths
    ]
    
    paths = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        paths.extend(matches)
    
    return paths


def _extract_keywords(text: str) -> List[str]:
    """
    Extract meaningful keywords from text.
    
    Filters out common stopwords and short words to extract
    meaningful technical/domain keywords.
    
    Args:
        text: Text to extract keywords from
        
    Returns:
        List of unique keywords in order of first occurrence
    """
    if not text:
        return []
    
    # Remove common words
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'this', 'that', 'these', 'those', 'it', 'its', 'i', 'you', 'we',
        'they', 'he', 'she', 'what', 'which', 'who', 'whom', 'when', 'where',
        'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())
    
    # Filter and score by length (longer = more meaningful)
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    
    # Dedupe while preserving order
    seen = set()
    result = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            result.append(kw)
    
    return result
