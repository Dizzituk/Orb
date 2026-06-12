# FILE: app/pipeline_v2/spec_review/models.py
# Purpose: Data models for the always-on spec reviewer.
# Called-by: app.pipeline_v2.spec_review, app.pipeline_v2.spec_review.parser, app.pipeline_v2.spec_review.reviewer
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Data models for the always-on spec reviewer.

The reviewer consumes Weaver intent + SpecGate spec + the builder's file
output + build log, and produces a structured ReviewReport where each
entry is a specific, citable finding with severity and category.

v1.0 (2026-04-18): Initial implementation for Stage 2 verifier work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Severity(str, Enum):
    """How serious a finding is."""
    CRITICAL = "critical"   # Spec requirement definitely not met; blocks release
    MAJOR = "major"         # Likely bug / significant spec gap
    MINOR = "minor"         # Polish item, edge case, low-risk gap
    INFO = "info"           # Observation worth logging, not a defect


class Category(str, Enum):
    """What kind of issue the finding represents.

    Keep this list small and load-bearing. If nothing fits, use OTHER.
    """
    UNWIRED = "unwired"                     # Observer/listener/VM never connected
    BLOCKING_IO = "blocking_io"             # I/O without Dispatchers.IO / async
    MISSING_CONNECTION = "missing_connection"  # A calls B but B isn't wired
    API_MISMATCH = "api_mismatch"           # Method signature / API-level mismatch
    SPEC_GAP = "spec_gap"                   # Spec requirement not implemented
    CONTRACT_BREAK = "contract_break"       # Cross-file interface broken
    RESOURCE_LEAK = "resource_leak"         # Unclosed resource, leaked scope
    OTHER = "other"


class Verdict(str, Enum):
    """Overall outcome of the review."""
    PASS = "pass"                                   # No findings
    PASS_WITH_WARNINGS = "pass_with_warnings"       # Only minor / info findings
    SPEC_GAPS_FOUND = "spec_gaps_found"             # Major findings present
    CRITICAL_ISSUES_FOUND = "critical_issues_found"  # One or more criticals


@dataclass
class Finding:
    """A single issue the reviewer identified.

    Every finding MUST cite a file or a specific code reference so the
    reader can act on it. The reviewer is instructed to refuse to emit
    a finding it cannot ground in real code.
    """
    severity: Severity
    category: Category
    title: str                          # Short headline (< 100 chars)
    description: str                    # Plain-language explanation
    file: Optional[str] = None          # Path relative to project root
    line: Optional[int] = None          # Line number (1-indexed)
    evidence: str = ""                  # Quoted code or specific reasoning
    spec_reference: str = ""            # Which requirement from the spec
    fix_hint: str = ""                  # Suggested surgical fix (optional)

    def one_line(self) -> str:
        where = ""
        if self.file:
            where = f" [{self.file}"
            if self.line:
                where += f":{self.line}"
            where += "]"
        return f"{self.severity.value.upper()} {self.category.value}: {self.title}{where}"


@dataclass
class ReviewReport:
    """Complete reviewer output.

    ``verdict`` is the aggregate signal; ``findings`` is the actionable list;
    the counters + cost fields let the pipeline surface per-run stats.
    """
    verdict: Verdict = Verdict.PASS
    summary: str = ""                           # One-paragraph overview
    findings: List[Finding] = field(default_factory=list)
    requirements_covered: List[str] = field(default_factory=list)
    requirements_unmet: List[str] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    duration_seconds: float = 0.0
    estimated_cost_usd: float = 0.0
    model_used: str = ""
    raw_response: str = ""                      # Kept for debugging

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)

    @property
    def major_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.MAJOR)

    @property
    def minor_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.MINOR)

    @property
    def info_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.INFO)

    @property
    def passed(self) -> bool:
        """True if the reviewer didn't find anything serious."""
        return self.verdict in (Verdict.PASS, Verdict.PASS_WITH_WARNINGS)

    def summary_line(self) -> str:
        return (
            f"{self.verdict.value}: "
            f"{self.critical_count} critical, "
            f"{self.major_count} major, "
            f"{self.minor_count} minor, "
            f"{self.info_count} info"
        )

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "summary": self.summary,
            "findings": [
                {
                    "severity": f.severity.value,
                    "category": f.category.value,
                    "title": f.title,
                    "file": f.file,
                    "line": f.line,
                    "description": f.description,
                    "evidence": f.evidence,
                    "spec_reference": f.spec_reference,
                    "fix_hint": f.fix_hint,
                }
                for f in self.findings
            ],
            "requirements_covered": self.requirements_covered,
            "requirements_unmet": self.requirements_unmet,
            "counts": {
                "critical": self.critical_count,
                "major": self.major_count,
                "minor": self.minor_count,
                "info": self.info_count,
            },
            "duration_seconds": self.duration_seconds,
            "estimated_cost_usd": self.estimated_cost_usd,
            "model_used": self.model_used,
            "tokens": {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
            },
        }
