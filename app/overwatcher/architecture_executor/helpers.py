"""
Helper utilities for architecture executor.

Provides low-level text processing functions used across the executor package.
These functions are pure utilities with no external dependencies (stdlib only).

v1.2 (2026-02-19): All-prose rejection in _sanitise_python_content.
  When the entire LLM output is markdown/prose with zero Python, the
  sanitiser now returns empty string + ALL_PROSE warning instead of
  passing the content through to waste 3 syntax-guard strikes.
  Fixes job sg-41756d01 where the Implementer wrote architecture
  instructions into _evidence.py 3 times.
v1.1 (2026-02-19): Added _sanitise_python_content and _check_python_syntax
  for deterministic pre-write validation of LLM-generated Python files.
  Fixes job sg-8d29f79f where the Implementer wrote markdown commentary
  directly into a .py file, causing a SyntaxError at boot.
"""

import ast
import logging
import re
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "_extract_llm_content",
    "_strip_markdown_fences",
    "_sanitise_python_content",
    "_check_python_syntax",
]


def _extract_llm_content(llm_result: Any) -> str:
    """
    Extract string content from an LLM result object.

    Handles both string responses and structured objects with a 'content' attribute.

    Args:
        llm_result: The LLM response object (str or object with .content)

    Returns:
        The extracted string content

    Raises:
        ValueError: If content cannot be extracted from the result
    """
    if isinstance(llm_result, str):
        return llm_result
    if hasattr(llm_result, "content"):
        content = llm_result.content
        if isinstance(content, str):
            return content
    raise ValueError(f"Cannot extract content from LLM result of type {type(llm_result)}")


def _strip_markdown_fences(content: str) -> str:
    """
    Remove markdown code fences from content if present.

    Strips leading ```[language] and trailing ``` markers, preserving the code inside.
    If no fences are found, returns the content unchanged.

    Args:
        content: The content string, possibly wrapped in markdown fences

    Returns:
        The content with fences removed (if present)
    """
    lines = content.strip().split("\n")
    if not lines:
        return content

    # Check for opening fence
    if lines[0].startswith("```"):
        lines = lines[1:]

    # Check for closing fence
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines)


# =============================================================================
# v1.1: DETERMINISTIC PYTHON VALIDATION
# =============================================================================

# Markdown patterns that should NEVER appear as top-level Python code.
# These indicate the LLM dumped architecture notes into the .py file.
_MARKDOWN_HEADING = re.compile(r'^\s*#{1,6}\s+\w')
_MARKDOWN_BOLD = re.compile(r'^\s*\*\*[A-Z]')
_MARKDOWN_FENCE = re.compile(r'^\s*```')
_MARKDOWN_TABLE = re.compile(r'^\s*\|\s*\w.*\|')
_MARKDOWN_HRULE = re.compile(r'^\s*-{3,}\s*$')
_MARKDOWN_QUOTE = re.compile(r'^\s*>{1,2}\s+')

_MARKDOWN_LINE_PATTERNS = [
    _MARKDOWN_HEADING,
    _MARKDOWN_BOLD,
    _MARKDOWN_FENCE,
    _MARKDOWN_TABLE,
    _MARKDOWN_HRULE,
    _MARKDOWN_QUOTE,
]

# Valid Python opening patterns — once we see one of these, the preamble is over.
_PY_COMMENT = re.compile(r'^\s*#')
_PY_DOCSTRING_DQ = re.compile(r'^\s*"""')
_PY_DOCSTRING_SQ = re.compile(r"^\s*'''")
_PY_FROM_IMPORT = re.compile(r'^\s*from\s+')
_PY_IMPORT = re.compile(r'^\s*import\s+')
_PY_DEF = re.compile(r'^\s*def\s+')
_PY_ASYNC_DEF = re.compile(r'^\s*async\s+def\s+')
_PY_CLASS = re.compile(r'^\s*class\s+')
_PY_DECORATOR = re.compile(r'^\s*@')
_PY_CONSTANT = re.compile(r'^\s*[A-Z_][A-Z_0-9]*\s*=')
_PY_DUNDER = re.compile(r'^\s*__')
_PY_BLANK = re.compile(r'^\s*$')

_PYTHON_START_PATTERNS = [
    _PY_COMMENT,
    _PY_DOCSTRING_DQ,
    _PY_DOCSTRING_SQ,
    _PY_FROM_IMPORT,
    _PY_IMPORT,
    _PY_DEF,
    _PY_ASYNC_DEF,
    _PY_CLASS,
    _PY_DECORATOR,
    _PY_CONSTANT,
    _PY_DUNDER,
    _PY_BLANK,
]


def _sanitise_python_content(content: str, file_path: str = "") -> Tuple[str, list]:
    """Strip non-Python preamble from LLM-generated content.

    The Implementer LLM sometimes prepends markdown architecture notes,
    binding contract text, or other commentary before the actual Python code.
    This function detects and removes that preamble.

    Args:
        content: Raw LLM output (after markdown fence stripping).
        file_path: For logging context.

    Returns:
        (cleaned_content, warnings) where warnings is a list of strings
        describing what was stripped.
    """
    if not content or not file_path.endswith('.py'):
        return content, []

    lines = content.split('\n')
    warnings = []
    preamble_end = 0
    found_python_start = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Blank lines before code starts are fine
        if not stripped:
            continue

        # Check if this looks like valid Python
        is_python = any(p.match(line) for p in _PYTHON_START_PATTERNS)
        if is_python:
            found_python_start = True
            preamble_end = i
            break

        # Check if this looks like markdown
        is_markdown = any(p.match(line) for p in _MARKDOWN_LINE_PATTERNS)
        if is_markdown:
            continue  # Keep scanning — maybe Python starts later

        # Unrecognised line before any Python — could be markdown text
        # without obvious markers (e.g. "Now generate the file:")
        # Keep scanning until we find Python
        continue

    if not found_python_start:
        # v1.2: Detect if content is ALL prose/markdown.
        # Count how many lines look like markdown vs unknown.
        non_blank_lines = [l for l in lines if l.strip()]
        markdown_count = sum(
            1 for l in non_blank_lines
            if any(p.match(l) for p in _MARKDOWN_LINE_PATTERNS)
        )
        # Heuristic: if >30% of non-blank lines are markdown, or if we
        # see common prose indicators, treat this as all-prose output.
        _content_lower = content.lower()
        _prose_indicators = [
            "the following", "copy implementation", "exports:",
            "do not rename", "must define", "this file",
            "source files are being", "## ", "**",
        ]
        _has_prose_markers = any(ind in _content_lower for ind in _prose_indicators)
        _high_markdown_ratio = (
            len(non_blank_lines) > 0
            and (markdown_count / len(non_blank_lines)) > 0.3
        )
        if _has_prose_markers or _high_markdown_ratio or len(non_blank_lines) > 3:
            # This is 100% prose — reject it entirely
            warning = (
                f"v1.2 ALL_PROSE_REJECTED: {file_path} — content is 100% "
                f"markdown/prose with no Python code detected "
                f"({len(non_blank_lines)} lines, {markdown_count} markdown). "
                f"The Implementer wrote architecture instructions instead of "
                f"Python source code. First line: {non_blank_lines[0][:100] if non_blank_lines else '(empty)'}..."
            )
            logger.error("[helpers] %s", warning)
            return "", [warning]
        # Truly empty or trivial content — return unchanged for syntax check
        return content, []

    if preamble_end > 0:
        stripped_lines = lines[:preamble_end]
        stripped_text = '\n'.join(stripped_lines).strip()
        if stripped_text:  # Don't warn about blank-only preamble
            warnings.append(
                f"v1.1 SANITISE: Stripped {preamble_end} line(s) of non-Python "
                f"preamble from {file_path}: {stripped_text[:150]}..."
            )
            logger.warning(
                "[helpers] v1.1 Stripped %d lines of markdown preamble from %s: %s",
                preamble_end, file_path, stripped_text[:100],
            )
        content = '\n'.join(lines[preamble_end:])

    return content, warnings


def _check_python_syntax(content: str, file_path: str = "") -> Optional[str]:
    """Run ast.parse() on Python content to catch syntax errors.

    This is a zero-cost deterministic check that catches the exact class of
    bug that killed job sg-8d29f79f: the Implementer wrote markdown commentary
    (``**IMPORTANT**: ...``) directly into a .py file, causing a SyntaxError
    that wasn't caught until the boot test.

    Args:
        content: Python source code to validate.
        file_path: For error context.

    Returns:
        None if syntax is valid, or an error string if parsing fails.
    """
    if not content or not file_path.endswith('.py'):
        return None

    try:
        ast.parse(content, filename=file_path)
        return None
    except SyntaxError as e:
        error_msg = (
            f"SyntaxError in {file_path} at line {e.lineno}: {e.msg}"
        )
        # Include the offending line for strike error context
        if e.text:
            error_msg += f"\n  Offending line: {e.text.strip()[:200]}"
        # Check if it looks like markdown leaked through
        if e.text and any(p.match(e.text) for p in _MARKDOWN_LINE_PATTERNS):
            error_msg += (
                "\n  CAUSE: This looks like markdown/architecture commentary "
                "that was written into the Python file instead of code. "
                "Output ONLY valid Python — no markdown headings, bold text, "
                "or table syntax."
            )
        logger.error("[helpers] v1.1 %s", error_msg)
        return error_msg
