# FILE: app/orchestrator/scaffold/validator.py
"""
Scaffold Validator.

Post-generation validation that the scaffold is syntactically valid
and structurally sound before handing to the Implementer.

Checks:
1. Python scaffolds pass ast.parse()
2. All LLM_FILL markers have matching FILL_END markers
3. Locked regions don't overlap with fill regions
4. Fill IDs are unique within a file
5. File size within limits

v1.0 (2026-03-01): Batch 2 — deterministic validation.
"""
from __future__ import annotations

import ast
import logging
import re
from typing import List, Optional

from .models import (
    FileLanguage,
    ScaffoldFile,
    ScaffoldResult,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)

logger = logging.getLogger(__name__)

# Size limits
MAX_SCAFFOLD_KB = 30
MAX_SCAFFOLD_LINES = 800


def validate_scaffold_result(result: ScaffoldResult) -> ValidationResult:
    """Validate all scaffolded files in a result.

    Returns a ValidationResult aggregating all issues found.
    """
    all_issues: List[ValidationIssue] = []

    for sf in result.files:
        issues = validate_single_file(sf)
        all_issues.extend(issues)

    status = "pass"
    if any(i.severity == ValidationSeverity.ERROR for i in all_issues):
        status = "fail"
    elif any(i.severity == ValidationSeverity.WARNING for i in all_issues):
        status = "warning"

    vr = ValidationResult(
        status=status,
        issues=all_issues,
    )

    logger.info(
        "[scaffold_validator] Validated %d files: %s (%d issues)",
        len(result.files), status, len(all_issues),
    )
    return vr


def validate_single_file(sf: ScaffoldFile) -> List[ValidationIssue]:
    """Validate a single scaffold file. Returns list of issues."""
    issues: List[ValidationIssue] = []

    # Check 1: Syntax validation
    if sf.language == FileLanguage.PYTHON:
        syntax_issue = _check_python_syntax(sf)
        if syntax_issue:
            issues.append(syntax_issue)

    # Check 2: Fill marker integrity
    fill_issues = _check_fill_markers(sf)
    issues.extend(fill_issues)

    # Check 3: Locked region sanity
    lock_issues = _check_locked_regions(sf)
    issues.extend(lock_issues)

    # Check 4: Size limits
    size_issue = _check_size_limits(sf)
    if size_issue:
        issues.append(size_issue)

    # Check 5: No accidental SSE contamination
    sse_issue = _check_sse_contamination(sf)
    if sse_issue:
        issues.append(sse_issue)

    return issues


# =============================================================================
# INDIVIDUAL CHECKS
# =============================================================================

def _check_python_syntax(sf: ScaffoldFile) -> Optional[ValidationIssue]:
    """Check Python scaffold passes ast.parse()."""
    # Replace fill markers with pass for syntax check
    code = sf.content
    code = re.sub(r'# LLM_FILL:.*', '# fill_placeholder', code)
    code = re.sub(r'# FILL_END:.*', '', code)

    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return ValidationIssue(
            file_path=sf.file_path,
            severity=ValidationSeverity.ERROR,
            message=f"Python syntax error at line {e.lineno}: {e.msg}",
            line_number=e.lineno or 0,
        )


def _check_fill_markers(sf: ScaffoldFile) -> List[ValidationIssue]:
    """Validate fill marker integrity."""
    issues: List[ValidationIssue] = []
    lines = sf.content.split('\n')

    # Find all FILL/FILL_END markers in the actual content
    fill_opens = {}  # {fill_id: line_number}
    fill_closes = {}

    for i, line in enumerate(lines, 1):
        m_open = re.search(r'LLM_FILL:\s*(FILL_\d+)', line)
        if m_open:
            fid = m_open.group(1)
            if fid in fill_opens:
                issues.append(ValidationIssue(
                    file_path=sf.file_path,
                    severity=ValidationSeverity.ERROR,
                    message=f"Duplicate fill marker {fid} at line {i} (first at {fill_opens[fid]})",
                    line_number=i,
                ))
            fill_opens[fid] = i

        m_close = re.search(r'FILL_END:\s*(FILL_\d+)', line)
        if m_close:
            fid = m_close.group(1)
            fill_closes[fid] = i

    # Check all opens have closes
    for fid, line_no in fill_opens.items():
        if fid not in fill_closes:
            issues.append(ValidationIssue(
                file_path=sf.file_path,
                severity=ValidationSeverity.ERROR,
                message=f"Fill marker {fid} at line {line_no} has no FILL_END",
                line_number=line_no,
            ))

    # Check manifest matches content
    manifest_ids = {f.id for f in sf.manifest.fills}
    content_ids = set(fill_opens.keys())

    for mid in manifest_ids - content_ids:
        issues.append(ValidationIssue(
            file_path=sf.file_path,
            severity=ValidationSeverity.WARNING,
            message=f"Fill {mid} in manifest but not in scaffold content",
        ))

    for cid in content_ids - manifest_ids:
        issues.append(ValidationIssue(
            file_path=sf.file_path,
            severity=ValidationSeverity.WARNING,
            message=f"Fill {cid} in scaffold content but not in manifest",
        ))

    return issues


def _check_locked_regions(sf: ScaffoldFile) -> List[ValidationIssue]:
    """Check locked regions don't overlap with fills."""
    issues: List[ValidationIssue] = []

    for lock in sf.manifest.locked_regions:
        for fill in sf.manifest.fills:
            if (lock.line_start <= fill.line_end
                    and lock.line_end >= fill.line_start):
                issues.append(ValidationIssue(
                    file_path=sf.file_path,
                    severity=ValidationSeverity.ERROR,
                    message=(
                        f"Locked region {lock.region_type} (lines {lock.line_start}-"
                        f"{lock.line_end}) overlaps with fill {fill.id} "
                        f"(lines {fill.line_start}-{fill.line_end})"
                    ),
                    line_number=lock.line_start,
                ))

    return issues


def _check_size_limits(sf: ScaffoldFile) -> Optional[ValidationIssue]:
    """Check scaffold file size."""
    kb = len(sf.content.encode('utf-8')) / 1024
    if kb > MAX_SCAFFOLD_KB:
        return ValidationIssue(
            file_path=sf.file_path,
            severity=ValidationSeverity.WARNING,
            message=f"Scaffold is {kb:.1f}KB (limit {MAX_SCAFFOLD_KB}KB)",
        )

    if sf.line_count > MAX_SCAFFOLD_LINES:
        return ValidationIssue(
            file_path=sf.file_path,
            severity=ValidationSeverity.WARNING,
            message=f"Scaffold has {sf.line_count} lines (limit {MAX_SCAFFOLD_LINES})",
        )

    return None


def _check_sse_contamination(sf: ScaffoldFile) -> Optional[ValidationIssue]:
    """Check for SSE emoji contamination in scaffold content."""
    sse_markers = ['\u2699', '\U0001f4cb', '\u2705', '\U0001f9e9', '\U0001f9e0']
    for marker in sse_markers:
        if marker in sf.content:
            return ValidationIssue(
                file_path=sf.file_path,
                severity=ValidationSeverity.ERROR,
                message="SSE contamination detected in scaffold content",
            )
    return None
