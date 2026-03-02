# FILE: app/orchestrator/syntax_feedback.py
"""
Syntax Feedback Formatter — formats validation errors for the implementer.

Takes ValidationBatchResult from syntax_validator and produces targeted,
actionable feedback that can be injected into the implementer retry prompt.
No LLM involvement — pure deterministic error formatting.

v1.0 (2026-03-01): Initial implementation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def format_syntax_errors_for_retry(
    errors_by_file: Dict[str, List[Any]],
) -> str:
    """Format syntax errors as a concise feedback block for the implementer.

    The output is injected into the implementer's retry prompt so it
    sees the exact errors and line numbers to fix.

    Args:
        errors_by_file: {file_path: [SyntaxError_]} from validation.

    Returns:
        Formatted feedback string ready for prompt injection.
    """
    if not errors_by_file:
        return ""

    total_errors = sum(len(errs) for errs in errors_by_file.values())
    parts = [
        "## ⚠️ SYNTAX ERRORS (deterministic — must fix before proceeding)",
        "",
        f"The syntax validator found **{total_errors} error(s)** in "
        f"**{len(errors_by_file)} file(s)**. These are compiler/parser errors, "
        f"not opinions — they must be fixed exactly.",
        "",
    ]

    for file_path, errors in errors_by_file.items():
        parts.append(f"### `{file_path}` ({len(errors)} error(s))")
        parts.append("")

        for err in errors[:10]:  # Cap at 10 per file
            loc = f"line {err.line}" if err.line else ""
            code = f"[{err.code}]" if err.code else ""
            parts.append(f"- {code} {loc}: {err.message}")

        if len(errors) > 10:
            parts.append(f"- ... and {len(errors) - 10} more")
        parts.append("")

    parts.extend([
        "**Instructions:**",
        "1. Fix ALL errors listed above in the affected files",
        "2. Do NOT modify files that passed validation",
        "3. Pay attention to line numbers — the error is at that exact location",
        "4. Common fixes: remove duplicate imports, fix missing semicolons, "
        "close unclosed braces, correct type mismatches",
        "",
    ])

    return "\n".join(parts)


def should_retry_without_overwatcher(
    errors_by_file: Dict[str, List[Any]],
) -> bool:
    """Determine if errors are simple enough to retry without Overwatcher.

    Simple errors (syntax, duplicate imports, brace mismatch) can be
    retried with just the error feedback — no Overwatcher diagnosis needed.
    Complex errors (logic, type mismatches across files) need Overwatcher.

    Args:
        errors_by_file: Errors from validation.

    Returns:
        True if a direct retry (skip Overwatcher) is appropriate.
    """
    if not errors_by_file:
        return False

    # Simple error codes that don't need Overwatcher diagnosis
    _SIMPLE_CODES = {
        "TS2300",   # Duplicate identifier
        "TS6133",   # Declared but never used
        "TS1005",   # Expected ';'
        "TS1003",   # Expected identifier
        "TS1128",   # Declaration or statement expected
        "TS1161",   # Unterminated string literal
        "CSS_BRACE",  # Brace mismatch
        "PY_SYNTAX",  # Python syntax error
    }

    all_errors = [e for errs in errors_by_file.values() for e in errs]
    simple_count = sum(1 for e in all_errors if getattr(e, "code", "") in _SIMPLE_CODES)

    # If >80% of errors are simple, skip Overwatcher
    if len(all_errors) > 0 and simple_count / len(all_errors) >= 0.8:
        return True

    # If total errors are very few (1-3), always try direct retry
    if len(all_errors) <= 3:
        return True

    return False
