# FILE: app/orchestrator/phase_checkout_frontend.py
"""
Frontend Syntax Validation for Phase Checkout.

v1.0 (2026-03-01): Lightweight TypeScript/TSX/JSX syntax check.
    Detects SSE contamination, markdown prose, and non-code content in
    frontend files produced by the pipeline. Does NOT require node_modules
    or tsc — uses the same pattern-matching approach as the Python sanitiser.
v1.1 (2026-03-01): Added [LLM_FILL] scaffold marker detection.
    Files containing unfilled scaffold placeholders are rejected as
    incomplete implementations.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# File extensions to validate
_FRONTEND_EXTENSIONS = {'.ts', '.tsx', '.jsx', '.js'}

# Valid TypeScript/JSX opening line patterns.
# A file that doesn't start with one of these (after blank lines)
# is very likely contaminated.
_VALID_TS_START = re.compile(
    r'^\s*('
    r'import\s|export\s|const\s|let\s|var\s|'
    r'class\s|function\s|async\s+function\s|'
    r'interface\s|type\s+\w|enum\s|'
    r'//|/\*|'
    r'@|'
    r'"use |\'use '
    r')'
)

# Known SSE/contamination markers
_CONTAMINATION_PATTERNS = [
    re.compile(r'[\u2699\U0001f4cb\u2705\U0001f9e9\U0001f9e0\U0001f3f7\U0001f3d7\U0001f4c1\U0001f4da\U0001f527]'),
    re.compile(r'^\s*#{1,6}\s+\w'),          # Markdown headings
    re.compile(r'^\s*\*\*[A-Z]'),             # Markdown bold
    re.compile(r'^\s*\|.*\|.*\|'),            # Markdown tables
    re.compile(r'^\s*-{3,}\s*$'),             # Horizontal rules
]

# v1.1: Unfilled scaffold marker pattern.
# Scaffold templates use [LLM_FILL: ...] as placeholders for the implementer.
# Any file still containing these after implementation is incomplete.
_SCAFFOLD_MARKER = re.compile(r'\[LLM_FILL[:\s]')


def check_frontend_syntax(
    state: Any,
    sandbox_base: str,
    emit: Optional[Any] = None,
) -> Dict[str, Any]:
    """Validate frontend files are actual code, not SSE/markdown garbage.

    Reads each frontend file produced by the pipeline and checks:
    1. The file starts with a valid TypeScript/JSX pattern (after blanks)
    2. The file doesn't contain known SSE contamination markers
    3. The file isn't predominantly markdown prose
    4. v1.1: The file doesn't contain unfilled [LLM_FILL] scaffold markers

    Args:
        state: JobState with segment output files.
        sandbox_base: Path to sandbox root (e.g. D:\\Orb).
        emit: Optional SSE callback for progress messages.

    Returns:
        Dict with 'status' ('pass'/'fail'), 'files_checked', 'failures'.
    """
    _emit = emit or (lambda msg: None)

    # Collect frontend output files from all segments
    frontend_files: List[str] = []
    for seg_id, seg_state in state.segments.items():
        for f in (seg_state.output_files or []):
            ext = os.path.splitext(f)[1].lower()
            if ext in _FRONTEND_EXTENSIONS:
                frontend_files.append(f)

    if not frontend_files:
        return {"status": "pass", "files_checked": 0, "failures": []}

    failures: List[Dict[str, str]] = []

    for rel_path in frontend_files:
        result = _validate_single_file(rel_path, sandbox_base)
        if result:
            failures.append(result)
            _emit(f"    FAIL {rel_path}: {result['reason']}")
        else:
            _emit(f"    OK {rel_path}")

    status = "fail" if failures else "pass"
    logger.info(
        "[frontend_check] v1.1 Checked %d frontend files: %d passed, %d failed",
        len(frontend_files),
        len(frontend_files) - len(failures),
        len(failures),
    )

    return {
        "status": status,
        "files_checked": len(frontend_files),
        "failures": failures,
    }


def _validate_single_file(
    rel_path: str,
    sandbox_base: str,
) -> Optional[Dict[str, str]]:
    """Validate a single frontend file.

    Returns None if valid, or a dict with 'file' and 'reason' if invalid.
    """
    content = _read_frontend_file(rel_path, sandbox_base)
    if content is None:
        logger.debug(
            "[frontend_check] Cannot read %s in sandbox (may be path issue)",
            rel_path,
        )
        return None

    if not content.strip():
        return {"file": rel_path, "reason": "File is empty"}

    lines = content.split('\n')
    non_blank = [l for l in lines if l.strip()]

    if not non_blank:
        return {"file": rel_path, "reason": "File contains only blank lines"}

    # v1.1: Check for unfilled scaffold markers FIRST — this is the most
    # definitive signal that implementation is incomplete.
    scaffold_hits = []
    for i, line in enumerate(lines):
        if _SCAFFOLD_MARKER.search(line):
            scaffold_hits.append((i + 1, line.strip()[:80]))
    if scaffold_hits:
        first_hit = scaffold_hits[0]
        return {
            "file": rel_path,
            "reason": (
                f"Unfilled scaffold marker detected ({len(scaffold_hits)} "
                f"occurrence(s)). First at line {first_hit[0]}: "
                f"{first_hit[1]}"
            ),
        }

    # Check 1: Does the file start with valid code?
    first_code_line = non_blank[0].strip()
    has_valid_start = _VALID_TS_START.match(first_code_line) is not None

    # Check 2: Does the file contain SSE/contamination markers?
    contamination_count = 0
    for line in non_blank[:20]:  # Check first 20 non-blank lines
        for pattern in _CONTAMINATION_PATTERNS:
            if pattern.search(line):
                contamination_count += 1
                break

    # Check 3: Is the file predominantly markdown?
    markdown_lines = 0
    for line in non_blank:
        stripped = line.strip()
        if (stripped.startswith('#') and not stripped.startswith('#!')
                and not stripped.startswith('//')):
            markdown_lines += 1
        elif stripped.startswith('**') or stripped.startswith('|'):
            markdown_lines += 1
        elif stripped.startswith('- ') or stripped.startswith('* '):
            markdown_lines += 1

    markdown_ratio = markdown_lines / len(non_blank) if non_blank else 0

    # Decision logic
    if contamination_count >= 3:
        return {
            "file": rel_path,
            "reason": (
                f"SSE contamination detected ({contamination_count} "
                f"marker lines in first 20 lines)"
            ),
        }

    if not has_valid_start and markdown_ratio > 0.3:
        return {
            "file": rel_path,
            "reason": (
                f"File appears to be markdown prose, not TypeScript "
                f"({markdown_ratio:.0%} markdown lines, "
                f"first line: {first_code_line[:80]})"
            ),
        }

    if not has_valid_start and contamination_count > 0:
        return {
            "file": rel_path,
            "reason": (
                f"File does not start with valid TypeScript and contains "
                f"contamination markers (first line: {first_code_line[:80]})"
            ),
        }

    return None


def _read_frontend_file(rel_path: str, sandbox_base: str) -> Optional[str]:
    """Read a frontend file from the sandbox, trying multiple path resolutions.

    v1.1: Checks orb-desktop paths for frontend files, including bare src/
    prefixes that resolve to D:\\orb-desktop.
    """
    norm_path = rel_path.replace('/', os.sep).replace('\\', os.sep)

    candidates = [
        os.path.join(sandbox_base, norm_path),
    ]

    # If the path starts with orb-desktop or src/components, try D:\orb-desktop
    if 'orb-desktop' in rel_path or rel_path.startswith('src/') or rel_path.startswith('src\\'):
        stripped = rel_path
        for prefix in ['orb-desktop/', 'orb-desktop\\', 'D:\\orb-desktop\\', 'D:/orb-desktop/']:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
                break
        candidates.append(os.path.join('D:\\orb-desktop', stripped.replace('/', os.sep)))
        # Also try inside sandbox under D:\Orb (phantom path)
        candidates.append(os.path.join(sandbox_base, stripped.replace('/', os.sep)))

    # Try sandbox filesystem first
    try:
        from app.sandbox_fs import sandbox_read_text
        for candidate in candidates:
            try:
                content = sandbox_read_text(candidate)
                if content is not None:
                    return content
            except Exception:
                continue
    except ImportError:
        pass

    # Fallback: try host filesystem
    for candidate in candidates:
        if os.path.isfile(candidate):
            try:
                with open(candidate, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read()
            except Exception:
                continue

    return None