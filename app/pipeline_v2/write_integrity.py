# FILE: app/pipeline_v2/write_integrity.py
"""
Write Integrity Checker — validates file content survives the write path.

Catches a class of bugs where content is syntactically correct in the LLM output
but gets corrupted during transport (shell escaping, encoding, string interpolation).

Root causes this catches:
  1. Template literal corruption (backticks stripped by PowerShell)
  2. String interpolation ($variable consumed by shell layer)
  3. Encoding mismatches (UTF-8 vs UTF-16 vs Latin-1)
  4. Truncated writes (connection timeout, partial response)
  5. Path separator mangling (backslash eaten as escape)

Usage:
    from app.pipeline_v2.write_integrity import verify_write, check_content_integrity

    # After writing a file:
    ok, issues = await verify_write(path, original_content)

    # Before writing (pre-flight check for known dangerous patterns):
    warnings = check_content_integrity(content, file_extension)
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Known corruption patterns ──────────────────────────────────

# Characters that are commonly eaten by shell layers
SHELL_DANGER_CHARS = {
    "`": "backtick (PowerShell escape char)",
    "$": "dollar sign (variable interpolation)",
    "%": "percent (cmd.exe variable)",
}

# Patterns that indicate template literal corruption in JS/TS files
JS_CORRUPTION_PATTERNS = [
    # Template literal backticks stripped — leaves bare ${...}
    (r'(?<![`\'"])(\$\{[^}]+\})(?![`\'"])', "bare template expression outside quotes — backticks likely stripped"),
    # String concatenation where template literal was expected
    (r'return\s+\$\{', "return followed by bare ${} — missing backticks"),
    # className with bare interpolation
    (r'className=\{[^}]*\$\{', "className with bare ${} — template literal corruption"),
]

# Patterns that indicate encoding issues
ENCODING_PATTERNS = [
    (r'\x00', "null bytes detected — likely UTF-16 encoding issue"),
    (r'\\x[0-9a-fA-F]{2}', "hex escape sequences — possible encoding mismatch"),
]

# Patterns for truncation detection
TRUNCATION_PATTERNS = {
    ".tsx": ["export default", "export function", "export {"],
    ".ts": ["export default", "export function", "export {", "export const"],
    ".py": ["if __name__", "def ", "class "],
    ".css": ["}"],
    ".json": ["}"],
}


# ─── Pre-write content check ───────────────────────────────────

def check_content_integrity(content: str, file_extension: str) -> list[dict]:
    """
    Pre-flight check on content before writing.
    Detects patterns that are likely to be corrupted during transport.

    Returns a list of warning dicts:
        [{"line": int, "pattern": str, "message": str, "severity": "warn"|"error"}]
    """
    warnings = []

    if not content or not content.strip():
        warnings.append({
            "line": 0,
            "pattern": "empty_content",
            "message": "Content is empty or whitespace-only",
            "severity": "error",
        })
        return warnings

    lines = content.split("\n")

    # Check for JS/TS specific corruption patterns
    if file_extension in (".js", ".jsx", ".ts", ".tsx"):
        for i, line in enumerate(lines, 1):
            for pattern, message in JS_CORRUPTION_PATTERNS:
                if re.search(pattern, line):
                    warnings.append({
                        "line": i,
                        "pattern": "template_corruption",
                        "message": message,
                        "severity": "error",
                        "content": line.strip()[:100],
                    })

    # Check for encoding issues (all file types)
    for i, line in enumerate(lines, 1):
        for pattern, message in ENCODING_PATTERNS:
            if re.search(pattern, line):
                warnings.append({
                    "line": i,
                    "pattern": "encoding_issue",
                    "message": message,
                    "severity": "error",
                    "content": line.strip()[:100],
                })

    # Check for truncation
    expected_endings = TRUNCATION_PATTERNS.get(file_extension, [])
    if expected_endings and len(lines) > 5:
        last_meaningful = ""
        for line in reversed(lines):
            stripped = line.strip()
            if stripped:
                last_meaningful = stripped
                break

        # If the file doesn't end with any expected pattern, it might be truncated
        has_valid_ending = any(
            last_meaningful.startswith(e) or last_meaningful.endswith(e)
            for e in expected_endings
        )
        if not has_valid_ending and file_extension in (".tsx", ".ts", ".jsx", ".js"):
            # More specific: check for unclosed braces
            open_braces = content.count("{")
            close_braces = content.count("}")
            if open_braces > close_braces:
                warnings.append({
                    "line": len(lines),
                    "pattern": "truncation",
                    "message": "File may be truncated: " + str(open_braces) + " open braces vs " + str(close_braces) + " close braces",
                    "severity": "error",
                })

    # Check bracket/paren balance (all code files)
    if file_extension in (".js", ".jsx", ".ts", ".tsx", ".py", ".json"):
        pairs = [("{", "}"), ("(", ")"), ("[", "]")]
        for opener, closer in pairs:
            o_count = content.count(opener)
            c_count = content.count(closer)
            if o_count != c_count:
                diff = abs(o_count - c_count)
                which = "unclosed" if o_count > c_count else "extra closing"
                warnings.append({
                    "line": 0,
                    "pattern": "bracket_mismatch",
                    "message": str(diff) + " " + which + " '" + (opener if o_count > c_count else closer) + "' characters",
                    "severity": "warn" if diff <= 1 else "error",
                })

    return warnings


# ─── Post-write verification ───────────────────────────────────

async def verify_write(path: str, expected_content: str) -> tuple[bool, list[dict]]:
    """
    After writing a file to the sandbox, read it back and verify integrity.

    Compares:
      1. Exact byte-level match (ideal)
      2. If not exact, checks for known corruption patterns
      3. Reports line-level differences

    Returns:
      (ok: bool, issues: list[dict])
    """
    from app.pipeline_v2.sandbox_tools import read_file

    actual = await read_file(path)
    if actual is None:
        return False, [{
            "type": "read_failure",
            "message": "Could not read back file after write: " + path,
            "severity": "error",
        }]

    issues = []

    # Normalise line endings for comparison
    expected_norm = expected_content.replace("\r\n", "\n")
    actual_norm = actual.replace("\r\n", "\n")

    if expected_norm == actual_norm:
        return True, []

    # Content differs — find out how
    expected_lines = expected_norm.split("\n")
    actual_lines = actual_norm.split("\n")

    # Length check
    if len(expected_lines) != len(actual_lines):
        issues.append({
            "type": "line_count_mismatch",
            "message": "Expected " + str(len(expected_lines)) + " lines, got " + str(len(actual_lines)),
            "severity": "error",
        })

    # Line-by-line diff (first 10 differences)
    diff_count = 0
    max_diffs = 10
    for i, (exp, act) in enumerate(zip(expected_lines, actual_lines)):
        if exp != act:
            diff_count += 1
            if diff_count <= max_diffs:
                issues.append({
                    "type": "line_diff",
                    "line": i + 1,
                    "expected": exp[:120],
                    "actual": act[:120],
                    "severity": "error",
                })

    if diff_count > max_diffs:
        issues.append({
            "type": "too_many_diffs",
            "message": str(diff_count) + " total line differences (showing first " + str(max_diffs) + ")",
            "severity": "error",
        })

    # Check for specific corruption signatures
    if any("`" in exp and "`" not in act for exp, act in zip(expected_lines, actual_lines)):
        issues.append({
            "type": "backtick_stripped",
            "message": "Backticks present in source but missing in written file — shell escaping likely corrupted the content",
            "severity": "error",
            "fix_hint": "Use raw byte writes or base64 encoding instead of shell pipe",
        })

    if any("$" in exp and "$" not in act for exp, act in zip(expected_lines, actual_lines)):
        issues.append({
            "type": "dollar_interpolated",
            "message": "Dollar signs present in source but missing in written file — variable interpolation likely consumed them",
            "severity": "error",
            "fix_hint": "Escape dollar signs or use a write method that doesn't interpret content",
        })

    return False, issues


# ─── Integration helper ─────────────────────────────────────────

async def safe_write_file(path: str, content: str, file_extension: str = "") -> tuple[bool, list[dict]]:
    """
    Write a file with full integrity checking:
      1. Pre-flight content check
      2. Write to sandbox
      3. Read-back verification

    Returns (ok, issues). If ok is False, the file should NOT be trusted.
    """
    from app.pipeline_v2.sandbox_tools import write_file

    # Detect extension if not provided
    if not file_extension and "." in path:
        file_extension = "." + path.rsplit(".", 1)[-1]

    # Pre-flight check
    pre_issues = check_content_integrity(content, file_extension)
    errors = [w for w in pre_issues if w["severity"] == "error"]

    if errors:
        logger.warning(
            "[write_integrity] Pre-flight check found %d errors for %s",
            len(errors), path
        )
        # Don't block the write — the content might still be valid
        # (e.g. intentional bare ${} in a string). But log it.

    # Write
    ok = await write_file(path, content)
    if not ok:
        return False, [{"type": "write_failed", "message": "Sandbox write returned failure", "severity": "error"}]

    # Verify
    verify_ok, verify_issues = await verify_write(path, content)

    all_issues = pre_issues + verify_issues
    return verify_ok, all_issues
