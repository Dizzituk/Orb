# FILE: app/llm/local_tools/zobie/streams/_codebase_report_structural.py
"""
Structural analysis integration for codebase reports.

Runs the smart health scanner on qualifying Python files and collects
genuine structural concerns (monolithic functions, god classes,
extractable blocks, multi-responsibility files).

Replaces the old raw size/line flagging with actionable intelligence.

BUILD_ID: 2026-02-23-v1.0-structural-report-integration
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Minimum line count to bother running structural analysis.
# Files below this are fine regardless of shape.
MIN_LINES_FOR_STRUCTURAL_SCAN = 200


@dataclass
class StructuralFinding:
    """A single structural concern found by the smart scanner."""
    file_path: str
    category: str       # monolithic_function, god_class, extractable_block, multi_responsibility
    severity: str       # info, warning, error
    description: str
    suggestion: str
    symbol_name: str = ""
    line_number: int = 0


@dataclass
class StructuralReport:
    """Aggregated structural analysis results."""
    findings: List[StructuralFinding] = field(default_factory=list)
    files_scanned: int = 0
    files_skipped_data_only: int = 0
    files_skipped_orchestrator: int = 0
    files_skipped_small: int = 0
    scan_errors: List[str] = field(default_factory=list)

    @property
    def has_concerns(self) -> bool:
        return len(self.findings) > 0

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "warning")

    @property
    def info_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "info")


def run_structural_scan(
    file_entries: list,
    project_roots: Optional[List[str]] = None,
) -> StructuralReport:
    """
    Run the smart codebase scanner on qualifying Python files.

    Only scans files that are:
    - Python (.py)
    - Above MIN_LINES_FOR_STRUCTURAL_SCAN
    - Not already known to be tiny

    Returns a StructuralReport with genuine concerns only.
    """
    from app.orchestrator.codebase_scanner import scan_file
    from app.orchestrator.codebase_scanner_models import HealthCategory

    # Categories we care about for the report (structural concerns only)
    STRUCTURAL_CATEGORIES = {
        HealthCategory.MONOLITHIC_FUNCTION,
        HealthCategory.GOD_CLASS,
        HealthCategory.EXTRACTABLE_BLOCK,
        HealthCategory.MULTI_RESPONSIBILITY,
    }

    report = StructuralReport()

    for entry in file_entries:
        rel_path = entry.relative_path
        abs_path = entry.path

        # Only Python files
        if not rel_path.endswith(".py"):
            continue

        # Only files above the line threshold
        if entry.line_count is not None and entry.line_count < MIN_LINES_FOR_STRUCTURAL_SCAN:
            report.files_skipped_small += 1
            continue

        # Read source
        try:
            source = Path(abs_path).read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            report.scan_errors.append(f"Could not read {rel_path}: {e}")
            continue

        # Run the scanner
        try:
            result = scan_file(rel_path, source)
        except Exception as e:
            report.scan_errors.append(f"Scan failed for {rel_path}: {e}")
            continue

        report.files_scanned += 1

        # Collect only structural findings
        for issue in result.health_issues:
            if issue.category not in STRUCTURAL_CATEGORIES:
                continue

            report.findings.append(StructuralFinding(
                file_path=issue.file_path,
                category=issue.category.value,
                severity=issue.severity.value,
                description=issue.description,
                suggestion=issue.suggestion,
                symbol_name=issue.symbol_name,
                line_number=issue.line_number,
            ))

    # Sort: warnings first, then by file path
    report.findings.sort(key=lambda f: (0 if f.severity == "warning" else 1, f.file_path))

    logger.info(
        "[structural_scan] Scanned %d files, found %d concerns (%d warnings, %d info)",
        report.files_scanned, len(report.findings),
        report.warning_count, report.info_count,
    )
    return report
