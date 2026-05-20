# FILE: app/pipeline_v2/spec_review/__init__.py
"""
Always-on spec reviewer package.

Public API:
    run_spec_review(spec, build_result, profile, ...)  -> ReviewReport

    ReviewReport, Finding, Severity, Category, Verdict
        -> dataclasses and enums for consuming the review output.

v1.0 (2026-04-18): Initial implementation.
"""
from app.pipeline_v2.spec_review.models import (
    Category,
    Finding,
    ReviewReport,
    Severity,
    Verdict,
)
from app.pipeline_v2.spec_review.reviewer import run_spec_review

__all__ = [
    "run_spec_review",
    "ReviewReport",
    "Finding",
    "Severity",
    "Category",
    "Verdict",
]
