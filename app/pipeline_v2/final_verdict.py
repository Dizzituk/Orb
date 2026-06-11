# FILE: app/pipeline_v2/final_verdict.py
"""
JOB 3 (2026-06-10) - One merged verdict instead of three contradicting reports.

After a build, three verification layers each produce their own report:
  - BVL (tier1 sanity / tier2 behavioural / tier3 adversarial)
  - the contract (multi-target) verifier
  - the always-on spec review
They have repeatedly disagreed (BVL blocked, review screaming criticals,
contract verifier PASS) and the human had to do forensics. This module folds
them into ONE verdict with the conflict named, plus per-criterion rows where
the spec carries structured acceptance criteria.

Statuses: VERIFIED / FAILED / UNTESTED / CONFLICTING-EVIDENCE.
Per-criterion runtime testing lands with the agentic verifier (Job 11); until
then criterion rows are derived from the spec review's unmet-requirements
list and default to UNTESTED rather than guessing.

Single responsibility: verdict assembly + rendering. No LLM calls.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

VERIFIED = "VERIFIED"
FAILED = "FAILED"
UNTESTED = "UNTESTED"
CONFLICTING = "CONFLICTING-EVIDENCE"


@dataclass
class SourceVerdict:
    source: str          # "bvl" | "contract" | "spec_review"
    status: str          # "pass" | "fail" | "warn" | "skipped"
    detail: str = ""


@dataclass
class CriterionVerdict:
    criterion: str
    status: str          # VERIFIED / FAILED / UNTESTED
    evidence: str = ""


@dataclass
class FinalVerdict:
    overall: str
    conflict_note: str = ""
    sources: List[SourceVerdict] = field(default_factory=list)
    criteria: List[CriterionVerdict] = field(default_factory=list)

    def render_lines(self) -> List[str]:
        lines: List[str] = []
        lines.append("")
        lines.append("=" * 60)
        lines.append("FINAL VERDICT (merged: BVL + contract + spec review)")
        lines.append("=" * 60)
        for s in self.sources:
            mark = {"pass": "PASS", "fail": "FAIL", "warn": "WARN", "skipped": "SKIP"}.get(s.status, "?")
            lines.append(f"   [{mark}] {s.source}: {s.detail[:160]}" if s.detail else f"   [{mark}] {s.source}")
        if self.criteria:
            lines.append("   -- Acceptance criteria --")
            for c in self.criteria:
                lines.append(f"   [{c.status}] {c.criterion[:140]}" + (f" - {c.evidence[:100]}" if c.evidence else ""))
        if self.conflict_note:
            lines.append(f"   CONFLICT: {self.conflict_note}")
        lines.append(f"   OVERALL: {self.overall}")
        lines.append("=" * 60)
        return lines


def build_final_verdict(
    pipeline_result: Any,
    contract_report: Optional[Any] = None,
    spec: Optional[Dict[str, Any]] = None,
) -> FinalVerdict:
    """Assemble the merged verdict. Never raises."""
    sources: List[SourceVerdict] = []

    # -- BVL ----------------------------------------------------------
    bvl = getattr(pipeline_result, "bvl_report", None)
    if bvl is None:
        sources.append(SourceVerdict("bvl", "skipped", "BVL not run for this target"))
        bvl_status = "skipped"
    else:
        try:
            if getattr(bvl, "all_signed_off", False):
                bvl_status = "pass"
                detail = "all tiers signed off"
            else:
                bvl_status = "fail"
                blocked = getattr(bvl, "blocked_components", []) or []
                detail = (blocked[0].block_reason[:200] if blocked else "verification incomplete")
        except Exception as exc:
            bvl_status, detail = "skipped", f"unreadable BVL report: {exc}"
        sources.append(SourceVerdict("bvl", bvl_status, detail))

    # -- Contract / multi-target verifier ------------------------------
    if contract_report is None:
        sources.append(SourceVerdict("contract", "skipped", "contract verifier not run"))
        contract_status = "skipped"
    else:
        try:
            contract_status = "pass" if contract_report.is_passing() else "fail"
            detail = contract_report.summary()
        except Exception as exc:
            contract_status, detail = "skipped", f"unreadable contract report: {exc}"
        sources.append(SourceVerdict("contract", contract_status, detail))

    # -- Spec review ----------------------------------------------------
    review = getattr(pipeline_result, "spec_review_report", None)
    unmet: List[str] = []
    if review is None:
        sources.append(SourceVerdict("spec_review", "skipped", "spec review not run"))
        review_status = "skipped"
    else:
        try:
            findings = getattr(review, "findings", []) or []
            crit = sum(1 for f in findings if getattr(getattr(f, "severity", None), "value", "") == "critical")
            major = sum(1 for f in findings if getattr(getattr(f, "severity", None), "value", "") == "major")
            unmet = list(getattr(review, "requirements_unmet", []) or [])
            if crit > 0:
                review_status = "fail"
            elif major > 0 or unmet:
                review_status = "warn"
            else:
                review_status = "pass"
            detail = getattr(review, "summary", "") or f"{crit} critical, {major} major"
        except Exception as exc:
            review_status, detail = "skipped", f"unreadable review report: {exc}"
        sources.append(SourceVerdict("spec_review", review_status, detail))

    # -- Per-criterion rows (structured criteria only) -------------------
    criteria_rows: List[CriterionVerdict] = []
    for crit_text in _extract_criteria(spec):
        status = UNTESTED
        evidence = ""
        for um in unmet:
            if um and (um.lower() in crit_text.lower() or crit_text.lower() in um.lower()):
                status = FAILED
                evidence = "spec review: unmet"
                break
        criteria_rows.append(CriterionVerdict(crit_text, status, evidence))

    # -- Overall + conflict detection -----------------------------------
    statuses = {s.source: s.status for s in sources}
    active = {k: v for k, v in statuses.items() if v != "skipped"}
    conflict_note = ""
    if "pass" in active.values() and "fail" in active.values():
        passers = [k for k, v in active.items() if v == "pass"]
        failers = [k for k, v in active.items() if v == "fail"]
        conflict_note = (
            f"{' & '.join(passers)} report PASS while {' & '.join(failers)} report FAIL - "
            "inspect both before trusting either"
        )
        overall = CONFLICTING
    elif "fail" in active.values():
        overall = FAILED
    elif not active:
        overall = UNTESTED
    elif all(v in ("pass", "warn") for v in active.values()):
        overall = VERIFIED
        if any(v == "warn" for v in active.values()):
            conflict_note = "passed with spec-review warnings - read the findings before sign-off"
    else:
        overall = UNTESTED

    return FinalVerdict(
        overall=overall,
        conflict_note=conflict_note,
        sources=sources,
        criteria=criteria_rows,
    )


def _extract_criteria(spec: Optional[Dict[str, Any]]) -> List[str]:
    """Pull structured acceptance criteria out of the spec if present."""
    if not isinstance(spec, dict):
        return []
    for key in ("acceptance_criteria", "acceptance", "criteria"):
        val = spec.get(key)
        if isinstance(val, list) and val:
            return [str(v)[:300] for v in val if v][:25]
    # Segment-level requirements as a fallback
    rows: List[str] = []
    for seg in spec.get("segments", []) or []:
        if isinstance(seg, dict):
            for req in seg.get("requirements", []) or []:
                if req:
                    rows.append(str(req)[:300])
    return rows[:25]


def push_verdict_narrative(verdict: FinalVerdict) -> None:
    """Record the verdict on the current build project (best-effort)."""
    try:
        from app.pipeline_v2.orchestrator import _push_narrative
        _push_narrative(
            stage="final_checkout",
            title=f"Final Verdict: {verdict.overall}",
            output_summary=verdict.conflict_note or verdict.overall,
            sections=[
                {
                    "heading": "Sources",
                    "body": "\n".join(f"- {s.source}: {s.status} {s.detail[:160]}" for s in verdict.sources),
                },
                {
                    "heading": "Acceptance criteria",
                    "body": "\n".join(f"- [{c.status}] {c.criterion}" for c in verdict.criteria) or "(no structured criteria in spec)",
                },
            ],
        )
    except Exception as exc:
        logger.debug("[final_verdict] narrative push failed: %s", exc)
