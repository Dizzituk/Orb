# FILE: app/llm/_weaver_substantive_filter.py
# Purpose: v4.3: Substantive assistant content detector for Weaver.
# Called-by: app.llm._weaver_stream_prepare
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
v4.3: Substantive assistant content detector for Weaver.

The Weaver historically only kept assistant messages matching vision
patterns (screenshot analysis). This missed rich technical responses
from video+tool pipelines (e.g. Gemini extracting requirements from a
video, reading codebase files, producing component specs). Those
responses contain the *actual* extracted requirements the Weaver needs.

This module provides _is_substantive_assistant_content() which broadens
the filter to include any assistant message with meaningful technical
content — code analysis, component specs, CSS patterns, API routes, etc.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Minimum length — short replies ("Sure!", "Got it.") are not substantive
_SUBSTANTIVE_MIN_LENGTH = 400

_SUBSTANTIVE_PATTERNS = [
    # Code / file references
    r"```",                          # code fences
    r"\.tsx\b",                      # TypeScript/React files
    r"\.ts\b",
    r"\.py\b",
    r"\.css\b",
    r"\bimport\s+[\{a-zA-Z]",       # JS/TS/Python imports
    r"\bfrom\s+['\"]",              # JS/TS imports
    r"\bclass\s+\w+",              # class definitions
    r"\binterface\s+\w+",         # TS interfaces
    r"\bdef\s+\w+\(",             # Python functions
    # Component / architecture specs
    r"\bcomponent\b",
    r"\brouter\b",
    r"\bendpoint\b",
    r"\bschema\b",
    r"\bmodel[s]?\b.*\b(?:creat|defin|implement)",
    r"\bphase\s+\d",               # "Phase 1:", "Phase 2:"
    # CSS / styling
    r"var\(--",                     # CSS custom properties
    r"border-radius",
    r"background:\s*",
    r"rgba\(",
    r"grid-template",
    r"display:\s*(?:flex|grid)",
    # API / backend
    r"\b(?:GET|POST|PUT|DELETE|PATCH)\s+/",  # REST endpoints
    r"\bSQLAlchemy\b",
    r"\bPydantic\b",
    r"\bFastAPI\b",
    # Structural markers (specs, analysis docs)
    r"##\s+\w",                     # markdown H2+ headers
    r"\*\*.*?:\*\*",               # bold labels like **Title:**
    r"\d+\.\s+[A-Z]\w+",          # numbered lists starting with caps
]


def _is_substantive_assistant_content(content: str) -> bool:
    """
    Detect if an assistant message contains substantive technical content
    that the Weaver should include when building a job description.

    v4.3: Broadens the old vision-only filter to catch rich responses from
    video+tool pipelines (Gemini codebase analysis, extracted specs, etc.).

    Requires 3+ pattern matches to avoid false positives on casual messages
    that happen to mention one keyword.

    Returns True if the message contains extracted requirements, code
    analysis, component specs, CSS patterns, or other technical content.
    """
    if not content or len(content) < _SUBSTANTIVE_MIN_LENGTH:
        return False

    # Check for substantive technical patterns — require 3+ distinct matches
    hit_count = 0
    for pattern in _SUBSTANTIVE_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            hit_count += 1
            if hit_count >= 3:
                logger.info(
                    "[WEAVER] v4.3 Substantive assistant content detected "
                    "(%d pattern hits, %d chars)",
                    hit_count, len(content),
                )
                return True

    return False