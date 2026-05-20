# FILE: app/pipeline_v2/spec_review/parser.py
"""
Parses Opus's JSON response into a ReviewReport.

The reviewer is prompted to return strict JSON with no preamble. In
practice frontier models sometimes wrap the payload in markdown fences
or emit a stray explanation line before the JSON. The parser handles
both gracefully and, on total parse failure, degrades to a single
info-severity finding describing the parse failure rather than crashing
the pipeline.

v1.0 (2026-04-18): Initial implementation.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.pipeline_v2.spec_review.models import (
    Category,
    Finding,
    ReviewReport,
    Severity,
    Verdict,
)

logger = logging.getLogger(__name__)


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_review_response(raw: str) -> ReviewReport:
    """Parse Opus's response string into a ReviewReport.

    Never raises. On unparseable input, returns a ReviewReport with one
    info-severity finding describing what went wrong, so the pipeline
    can continue and the user sees the raw output for debugging.
    """
    raw = (raw or "").strip()
    if not raw:
        return _parse_failure_report(raw, "Reviewer returned an empty response.")

    payload = _extract_json_object(raw)
    if payload is None:
        return _parse_failure_report(raw, "No JSON object found in reviewer response.")

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        return _parse_failure_report(
            raw,
            f"Reviewer response was not valid JSON: {exc}",
        )

    if not isinstance(data, dict):
        return _parse_failure_report(
            raw,
            f"Reviewer returned {type(data).__name__}, expected an object.",
        )

    report = ReviewReport(raw_response=raw)
    report.summary = str(data.get("summary", "") or "")
    report.verdict = _parse_verdict(data.get("verdict"))
    report.requirements_covered = _string_list(data.get("requirements_covered"))
    report.requirements_unmet = _string_list(data.get("requirements_unmet"))
    report.findings = _parse_findings(data.get("findings"))

    # Reconcile verdict with findings in case the model contradicted itself.
    report.verdict = _reconcile_verdict(report)

    return report


# ═══════════════════════════════════════════════════════════════════
# JSON extraction
# ═══════════════════════════════════════════════════════════════════

def _extract_json_object(raw: str) -> Optional[str]:
    """Pull the JSON object out of the response.

    Tries (in order):
      1. Parse the full string as-is (fastest path).
      2. Strip ```json ... ``` fences.
      3. Fall back to the widest {...} match in the string.
    """
    stripped = raw.strip()

    # Path 1: raw is already the JSON object.
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    # Path 2: fenced code block.
    fence_match = _FENCE_RE.search(raw)
    if fence_match:
        return fence_match.group(1).strip()

    # Path 3: greedy brace match anywhere in the string.
    obj_match = _OBJECT_RE.search(raw)
    if obj_match:
        return obj_match.group(0).strip()

    return None


# ═══════════════════════════════════════════════════════════════════
# Field parsers
# ═══════════════════════════════════════════════════════════════════

def _parse_verdict(raw: Any) -> Verdict:
    if isinstance(raw, str):
        try:
            return Verdict(raw.strip().lower())
        except ValueError:
            pass
    return Verdict.PASS


def _parse_severity(raw: Any) -> Optional[Severity]:
    if isinstance(raw, str):
        try:
            return Severity(raw.strip().lower())
        except ValueError:
            return None
    return None


def _parse_category(raw: Any) -> Optional[Category]:
    if isinstance(raw, str):
        try:
            return Category(raw.strip().lower())
        except ValueError:
            return None
    return None


def _string_list(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if item is not None]


def _parse_findings(raw: Any) -> List[Finding]:
    if not isinstance(raw, list):
        return []

    findings: List[Finding] = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        severity = _parse_severity(item.get("severity"))
        category = _parse_category(item.get("category"))
        if severity is None or category is None:
            # Skip malformed findings rather than crash the whole report.
            logger.debug(
                "[spec_review] Skipping finding with invalid severity=%r "
                "or category=%r",
                item.get("severity"),
                item.get("category"),
            )
            continue

        line = item.get("line")
        try:
            line_int = int(line) if line is not None else None
        except (TypeError, ValueError):
            line_int = None

        findings.append(
            Finding(
                severity=severity,
                category=category,
                title=str(item.get("title", "") or "")[:200],
                description=str(item.get("description", "") or ""),
                file=_optional_str(item.get("file")),
                line=line_int,
                evidence=str(item.get("evidence", "") or ""),
                spec_reference=str(item.get("spec_reference", "") or ""),
                fix_hint=str(item.get("fix_hint", "") or ""),
            )
        )

    return findings


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


# ═══════════════════════════════════════════════════════════════════
# Failure fallback
# ═══════════════════════════════════════════════════════════════════

def _parse_failure_report(raw: str, reason: str) -> ReviewReport:
    """Return a report that records the parse failure without crashing."""
    logger.warning("[spec_review] %s", reason)
    return ReviewReport(
        verdict=Verdict.PASS_WITH_WARNINGS,
        summary=(
            "Spec reviewer ran but its response could not be parsed as the "
            "expected JSON structure. The raw response is preserved on the "
            "report for debugging. Treat this as 'review inconclusive' "
            "rather than a pass."
        ),
        findings=[
            Finding(
                severity=Severity.INFO,
                category=Category.OTHER,
                title="Reviewer response not parseable",
                description=reason,
                evidence=(raw[:400] + "\u2026") if len(raw) > 400 else raw,
            )
        ],
        raw_response=raw,
    )


# ═══════════════════════════════════════════════════════════════════
# Verdict reconciliation
# ═══════════════════════════════════════════════════════════════════

def _reconcile_verdict(report: ReviewReport) -> Verdict:
    """Ensure the verdict matches the findings severity distribution.

    Frontier models occasionally say 'pass' while listing a critical
    finding, or 'critical_issues_found' with an empty findings list.
    The findings are ground truth; the verdict is derived.
    """
    if report.critical_count > 0:
        return Verdict.CRITICAL_ISSUES_FOUND
    if report.major_count > 0:
        return Verdict.SPEC_GAPS_FOUND
    if report.minor_count > 0 or report.info_count > 0:
        return Verdict.PASS_WITH_WARNINGS
    return Verdict.PASS
