# FILE: app/debug/executors/filesystem_guards.py
"""
Filesystem guards: pre-write and post-write validation for ASTRA's tool loop.

These guards prevent a class of self-inflicted corruption that has happened
in practice: ASTRA reads a file with line-number prefixes (e.g. "  12: foo"),
then writes that decorated output back as content, producing files that
look like  "  1:   1: from __future__..."  and refuse to parse.

Three independent guards:
  - strip_line_number_prefixes(content):  decorates -> raw, idempotent.
  - looks_like_line_numbered_content(c):  detects corruption before write.
  - check_syntax(path, content):          AST/JSON parse based on extension.

All functions are pure (no I/O, no globals). They live in their own module
so filesystem.py stays under the 30KB hard cap.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Pattern that matches a line-number prefix at the start of a line.
# Tolerates 1-6 leading spaces, 1-6 digits, then ":" + at least one space.
# Examples that match:
#   "  1: foo"
#   "   42: bar"
#   "1234: baz"
# Examples that DO NOT match (legitimate code):
#   "foo: bar"           - no leading digits
#   "    if x == 1:"     - colon is mid-line, not after digits
#   "1.5: comment"       - non-digit between digits and colon
_LINE_NUMBER_PREFIX = re.compile(r"^[ \t]{0,6}\d{1,6}:[ \t]+")

# Detection threshold: if N of the first M non-blank lines match the prefix,
# we treat the content as line-numbered. Conservative to avoid false positives
# on legitimate code that happens to start with "12: " (rare but possible in
# YAML config or markdown).
_DETECTION_SAMPLE = 8       # Inspect this many non-blank lines.
_DETECTION_THRESHOLD = 5    # This many must match to count as "decorated".


# =============================================================================
# Line-number prefix handling
# =============================================================================

def strip_line_number_prefixes(content: str) -> Tuple[str, bool]:
    """Remove leading "N: " line-number prefixes from each line of `content`.

    Returns (cleaned_content, was_stripped).

    Idempotent: calling on already-clean content returns it unchanged with
    was_stripped=False. Safe to call on any text input.

    Strategy: only strips the prefix when the content as a whole looks
    line-numbered (more than _DETECTION_THRESHOLD matches in the first
    _DETECTION_SAMPLE non-blank lines). Per-line stripping after that point
    is unconditional so trailing decorated lines aren't left behind.
    """
    if not content:
        return content, False

    if not looks_like_line_numbered_content(content):
        return content, False

    # Loop in case the content was read-and-rewritten more than once,
    # producing nested prefixes like "  1:   1: foo". Capped at 4
    # passes to avoid any pathological case.
    was_stripped = False
    for _ in range(4):
        if not looks_like_line_numbered_content(content):
            break
        lines_out = []
        for line in content.splitlines(keepends=False):
            stripped = _LINE_NUMBER_PREFIX.sub("", line, count=1)
            lines_out.append(stripped)
        cleaned = "\n".join(lines_out)
        if content.endswith("\n") and not cleaned.endswith("\n"):
            cleaned += "\n"
        content = cleaned
        was_stripped = True

    return content, was_stripped


def looks_like_line_numbered_content(content: str) -> bool:
    """Heuristic: does `content` look like the output of a line-numbered reader?

    Used as a pre-write guard. Returns True when enough of the early lines
    match the "  N: " prefix pattern to make accidental write-back likely.

    Conservative by design — false positives would reject legitimate writes.
    """
    if not content:
        return False

    matches = 0
    inspected = 0
    for line in content.splitlines():
        if not line.strip():
            continue  # Skip blank lines; they carry no prefix.
        inspected += 1
        if _LINE_NUMBER_PREFIX.match(line):
            matches += 1
        if inspected >= _DETECTION_SAMPLE:
            break

    return matches >= _DETECTION_THRESHOLD


# =============================================================================
# Syntax check (post-write validation)
# =============================================================================

# Map of file extensions to a cheap parser. Returning None means "OK",
# returning a string means "broken, here's why".
def _check_python(content: str) -> Optional[str]:
    try:
        ast.parse(content)
        return None
    except SyntaxError as e:
        # Format consistent with Python's own error display.
        return f"SyntaxError: {e.msg} (line {e.lineno}, col {e.offset})"
    except ValueError as e:
        # Raised for e.g. null bytes in source.
        return f"Invalid Python source: {e}"


def _check_json(content: str) -> Optional[str]:
    if not content.strip():
        return None  # Empty file is allowed at filesystem level.
    try:
        json.loads(content)
        return None
    except json.JSONDecodeError as e:
        return f"JSONDecodeError: {e.msg} (line {e.lineno}, col {e.colno})"


_SYNTAX_CHECKERS = {
    ".py":   _check_python,
    ".json": _check_json,
}


def check_syntax(path: str, content: str) -> Optional[str]:
    """Run a cheap syntax check for known file types.

    Returns None if the content is syntactically valid (or the extension
    isn't covered). Returns an error message string if parsing fails.

    Only languages with fast in-process parsers are covered (.py, .json).
    TypeScript / JSX / YAML would need external tools and aren't worth
    the latency for every write.
    """
    if not content:
        return None  # Empty file: nothing to parse.

    ext = os.path.splitext(path)[1].lower()
    checker = _SYNTAX_CHECKERS.get(ext)
    if not checker:
        return None  # Unsupported extension; treat as valid.

    return checker(content)


# =============================================================================
# Diagnostic helpers (used by error messages)
# =============================================================================

def describe_line_numbered_corruption(content: str, sample_lines: int = 3) -> str:
    """Build a short diagnostic showing what the corruption looks like.

    Used in error messages so ASTRA can see exactly what was rejected.
    """
    preview_lines = []
    for line in content.splitlines()[:sample_lines]:
        if len(line) > 80:
            line = line[:77] + "..."
        preview_lines.append(f"    {line!r}")
    return "\n".join(preview_lines) if preview_lines else "(empty)"
