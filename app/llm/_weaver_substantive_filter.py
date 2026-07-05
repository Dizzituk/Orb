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

# Minimum length — short replies ("Sure!", "Got it.") are not substantive.
# 2026-07-04 (live6, Taz): lowered from 400 — Astra's conversational replies
# ('Yeah man, it's a self-contained retro Tetris game with... controls: A/D
# to slide, features...') carry load-bearing spec content the Weaver MUST
# read, but they are prose, not code, so the old code-heavy filter dropped
# them. A long assistant reply is substantive by default now.
_SUBSTANTIVE_MIN_LENGTH = 250
_SUBSTANTIVE_LONG_LENGTH = 700  # any reply this long is substantive regardless of pattern hits

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
    # 2026-07-04 (live6): conversational build/spec vocabulary — Astra
    # describing what it built or should build is load-bearing even in prose.
    r"\b(?:app|application|game|feature|button|control|screen|window|menu)s?\b",
    r"\b(?:build|built|create[ds]?|implement|standalone|self-contained)\b",
    r"\b(?:arrow|keyboard|key|mouse|click|drag)\b",
    r"\b(?:colou?r|theme|retro|layout|design|style)s?\b",
    r"\b(?:file|folder|path|index\.html|\.exe|\.py|\.js)\b",
    r"[-*•]\s+\w",            # bullet lists
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

    # A long reply is substantive by default (Astra rarely rambles 700+ chars
    # of pure pleasantry — and if it does, the Weaver ignores the fluff).
    if len(content) >= _SUBSTANTIVE_LONG_LENGTH:
        logger.info(
            "[WEAVER] live6 Substantive assistant content (long reply, %d chars)",
            len(content),
        )
        return True

    # Shorter replies need 2+ distinct pattern hits (was 3 + code-only).
    hit_count = 0
    for pattern in _SUBSTANTIVE_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            hit_count += 1
            if hit_count >= 2:
                logger.info(
                    "[WEAVER] live6 Substantive assistant content (%d pattern hits, %d chars)",
                    hit_count, len(content),
                )
                return True

    return False