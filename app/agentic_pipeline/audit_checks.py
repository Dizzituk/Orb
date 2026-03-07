# FILE: app/agentic_pipeline/audit_checks.py
"""
Phase 1: Deterministic Checker Audit.

Analyses historical job data to classify each check by false-positive rate.
Runs the check_runner against completed jobs' arch docs and compares
against actual outcomes (did the job succeed despite the check firing?).

This is a research/diagnostic tool, not a production component.

Usage:
    from app.agentic_pipeline.audit_checks import run_audit
    results = run_audit()  # Analyses last N completed jobs
    print(results.to_report())

v1.0 (2026-03-05): Initial implementation.
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

JOBS_ROOT = os.path.join("D:\\Orb", "jobs", "jobs")


@dataclass
class CheckFiring:
    """Record of a single check firing in a historical job."""
    job_id: str
    segment_id: str
    check_id: str
    severity: str  # from cohesion: "warning" or "blocking"
    category: str
    description: str
    file_path: str = ""
    was_genuine: Optional[bool] = None  # True = real issue, False = false positive
    reason: str = ""


@dataclass
class CheckClassification:
    """Aggregated classification for one check type."""
    check_id: str
    total_firings: int = 0
    genuine_count: int = 0
    false_positive_count: int = 0
    undetermined_count: int = 0
    recommended_confidence: str = ""  # HARD_BLOCK, WARNING, or INFO
    notes: str = ""

    @property
    def false_positive_rate(self) -> float:
        determined = self.genuine_count + self.false_positive_count
        if determined == 0:
            return 0.0
        return self.false_positive_count / determined

    @property
    def is_safe_for_hard_block(self) -> bool:
        """Zero false positives required for HARD_BLOCK."""
        return self.false_positive_count == 0 and self.genuine_count > 0


@dataclass
class AuditResult:
    """Complete audit results across all analysed jobs."""
    jobs_analysed: int = 0
    total_firings: int = 0
    classifications: Dict[str, CheckClassification] = field(default_factory=dict)
    cohesion_analysis: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_report(self) -> str:
        lines = [
            "# Deterministic Checker Audit Report",
            f"Jobs analysed: {self.jobs_analysed}",
            f"Total check firings: {self.total_firings}",
            "",
            "## Check Classifications",
            "",
            "| Check ID | Firings | Genuine | FP | FP Rate | Recommended |",
            "|----------|---------|---------|-----|---------|-------------|",
        ]
        for cid, c in sorted(self.classifications.items()):
            fp_pct = f"{c.false_positive_rate:.0%}" if (c.genuine_count + c.false_positive_count) > 0 else "N/A"
            lines.append(
                f"| {cid} | {c.total_firings} | {c.genuine_count} | "
                f"{c.false_positive_count} | {fp_pct} | {c.recommended_confidence} |"
            )

        if self.cohesion_analysis:
            lines.extend(["", "## Cohesion Check Analysis"])
            for category, data in sorted(self.cohesion_analysis.items()):
                lines.append(f"### {category}")
                lines.append(f"  Total: {data.get('total', 0)}")
                lines.append(f"  Genuine: {data.get('genuine', 0)}")
                lines.append(f"  False positive: {data.get('false_positive', 0)}")
                if data.get("examples"):
                    lines.append("  Examples:")
                    for ex in data["examples"][:3]:
                        lines.append(f"    - {ex}")
                lines.append("")

        if self.errors:
            lines.extend(["", "## Errors"])
            for e in self.errors:
                lines.append(f"  - {e}")

        return "\n".join(lines)


def run_audit(max_jobs: int = 5) -> AuditResult:
    """Run the full audit across recent completed jobs.

    Analyses:
    1. Cohesion check results (from cohesion_check.json)
    2. Arch doc extractability (run check_runner against real arch docs)
    3. Cross-reference with job success/failure status
    """
    result = AuditResult()

    # Find jobs with arch docs and results
    jobs = _find_auditable_jobs(max_jobs)
    result.jobs_analysed = len(jobs)

    if not jobs:
        result.errors.append("No auditable jobs found")
        return result

    for job_id, job_info in jobs.items():
        try:
            _audit_single_job(job_id, job_info, result)
        except Exception as e:
            result.errors.append(f"Job {job_id}: {e}")
            logger.warning("[audit] Failed to audit %s: %s", job_id, e)

    # Classify checks based on aggregated data
    _classify_checks(result)

    return result


def _find_auditable_jobs(max_jobs: int) -> Dict[str, Dict]:
    """Find jobs with arch docs, skeleton contracts, and cohesion results."""
    jobs = {}

    if not os.path.isdir(JOBS_ROOT):
        return jobs

    for name in sorted(os.listdir(JOBS_ROOT), reverse=True):
        if "__" in name or not name.startswith("sg-"):
            continue

        job_dir = os.path.join(JOBS_ROOT, name)
        seg_dir = os.path.join(job_dir, "segments")

        if not os.path.isdir(seg_dir):
            continue

        manifest_path = os.path.join(seg_dir, "manifest.json")
        skeleton_path = os.path.join(seg_dir, "skeleton_contract.json")
        cohesion_path = os.path.join(seg_dir, "cohesion_check.json")

        if not os.path.isfile(manifest_path):
            continue

        # Count arch docs
        arch_count = 0
        arch_docs = {}
        for seg_name in os.listdir(seg_dir):
            seg_path = os.path.join(seg_dir, seg_name)
            if os.path.isdir(seg_path):
                arch_file = os.path.join(seg_path, "arch", "arch_v1.md")
                if os.path.isfile(arch_file):
                    arch_count += 1
                    arch_docs[seg_name] = arch_file

        if arch_count == 0:
            continue

        # Load job state for outcome
        state_path = os.path.join(job_dir, "state.json")
        overall_status = "unknown"
        if os.path.isfile(state_path):
            try:
                with open(state_path, "r") as f:
                    state = json.load(f)
                overall_status = state.get("overall_status", "unknown")
            except Exception:
                pass

        jobs[name] = {
            "job_dir": job_dir,
            "seg_dir": seg_dir,
            "manifest_path": manifest_path,
            "skeleton_path": skeleton_path,
            "cohesion_path": cohesion_path,
            "arch_docs": arch_docs,
            "arch_count": arch_count,
            "overall_status": overall_status,
        }

        if len(jobs) >= max_jobs:
            break

    return jobs


def _audit_single_job(job_id: str, job_info: Dict, result: AuditResult) -> None:
    """Audit a single job's check results."""
    logger.info("[audit] Auditing job %s (%d arch docs)", job_id, job_info["arch_count"])

    # 1. Analyse cohesion check results
    cohesion_path = job_info["cohesion_path"]
    if os.path.isfile(cohesion_path):
        _audit_cohesion(job_id, cohesion_path, job_info, result)

    # 2. Run check_runner against arch docs (simulated)
    _audit_arch_docs(job_id, job_info, result)


def _audit_cohesion(
    job_id: str, cohesion_path: str, job_info: Dict, result: AuditResult,
) -> None:
    """Analyse cohesion check results for false positives."""
    try:
        with open(cohesion_path, "r") as f:
            cohesion = json.load(f)
    except Exception as e:
        result.errors.append(f"Cohesion load failed for {job_id}: {e}")
        return

    issues = cohesion.get("issues", [])
    job_succeeded = job_info["overall_status"] in ("complete", "partial")

    for issue in issues:
        if isinstance(issue, str):
            # PowerShell serialised — parse manually
            issue = _parse_ps_object(issue)

        issue_id = issue.get("issue_id", "unknown")
        severity = issue.get("severity", "warning")
        category = issue.get("category", "unknown")
        description = issue.get("description", "")
        expected = issue.get("expected", "")
        actual = issue.get("actual", "")

        # Determine if this was a genuine issue or false positive
        is_fp = _classify_cohesion_issue(category, expected, actual, description)

        firing = CheckFiring(
            job_id=job_id, segment_id=issue.get("source_segment", ""),
            check_id=f"COHESION_{category.upper()}",
            severity=severity, category=category,
            description=description[:200],
            file_path=issue.get("file_path", ""),
            was_genuine=not is_fp,
            reason="type prefix mismatch" if is_fp else "genuine",
        )
        result.total_firings += 1

        # Aggregate by category
        if category not in result.cohesion_analysis:
            result.cohesion_analysis[category] = {
                "total": 0, "genuine": 0, "false_positive": 0, "examples": [],
            }
        cat_data = result.cohesion_analysis[category]
        cat_data["total"] += 1
        if is_fp:
            cat_data["false_positive"] += 1
        else:
            cat_data["genuine"] += 1
        if len(cat_data["examples"]) < 3:
            cat_data["examples"].append(f"[{job_id}] {description[:100]}")

        # Aggregate into classifications
        cid = firing.check_id
        if cid not in result.classifications:
            result.classifications[cid] = CheckClassification(check_id=cid)
        cls = result.classifications[cid]
        cls.total_firings += 1
        if is_fp:
            cls.false_positive_count += 1
        else:
            cls.genuine_count += 1


def _classify_cohesion_issue(
    category: str, expected: str, actual: str, description: str,
) -> bool:
    """Determine if a cohesion issue is a false positive.

    Returns True if the issue is a false positive.
    """
    # Type prefix mismatch: "type DebugJobSummary" vs "DebugJobSummary"
    # The cohesion checker compares with TS type keyword prefixes
    if category == "interface_mismatch" and expected and actual:
        bare_expected = expected.replace("type ", "").replace("interface ", "").strip()
        # Check if the bare name IS in the actual exports
        actual_names = [a.strip() for a in actual.split(",")]
        if bare_expected in actual_names:
            return True  # False positive — the symbol exists, just with a type prefix

    # Missing export for files that are CREATE targets
    if category == "missing_export":
        # These fire because the file doesn't exist yet during cohesion check
        # The architecture will create it — this is expected
        if "should export" in description and "consumed by" in description:
            # Check if the description mentions a file that's in the segment's scope
            # For now, classify all missing_export as false positives when the
            # job eventually succeeded
            return False  # Conservative — leave as genuine until proven otherwise

    return False


def _audit_arch_docs(job_id: str, job_info: Dict, result: AuditResult) -> None:
    """Run check_runner against historical arch docs to test extraction."""
    try:
        from app.agentic_pipeline.checks.check_runner import run_deterministic_checks
        from app.agentic_pipeline.preflight_evidence import gather_preflight_evidence
    except ImportError:
        result.errors.append("Could not import check_runner or preflight_evidence")
        return

    # Load manifest for file scopes
    try:
        with open(job_info["manifest_path"], "r") as f:
            manifest = json.load(f)
    except Exception:
        return

    # Build segment file scopes
    segment_file_scopes = {}
    for seg in manifest.get("segments", []):
        sid = seg.get("segment_id", "")
        segment_file_scopes[sid] = seg.get("file_scope", [])

    # Gather preflight (will hit sandbox — may fail if sandbox is down)
    all_files = list(set(f for files in segment_file_scopes.values() for f in files))
    preflight = gather_preflight_evidence(all_files)

    # Read arch docs and run checks
    arch_docs = {}
    for seg_name, arch_path in job_info["arch_docs"].items():
        try:
            with open(arch_path, "r", encoding="utf-8") as f:
                arch_docs[seg_name] = f.read()
        except Exception:
            continue

    if not arch_docs:
        return

    report = run_deterministic_checks(
        arch_docs=arch_docs,
        preflight=preflight,
        segment_file_scopes=segment_file_scopes,
    )

    # Record findings
    for check_result in report.results:
        cid = check_result.check_id
        if cid not in result.classifications:
            result.classifications[cid] = CheckClassification(check_id=cid)
        cls = result.classifications[cid]
        cls.total_firings += 1
        # For arch doc checks, we can't easily determine genuine vs FP
        # without manual review — mark as undetermined
        cls.undetermined_count += 1
        result.total_firings += 1


def _classify_checks(result: AuditResult) -> None:
    """Assign recommended confidence levels based on aggregated data."""
    for cid, cls in result.classifications.items():
        if cls.is_safe_for_hard_block:
            cls.recommended_confidence = "HARD_BLOCK"
            cls.notes = "Zero false positives across all jobs"
        elif cls.false_positive_rate > 0.3:
            cls.recommended_confidence = "INFO"
            cls.notes = f"High FP rate ({cls.false_positive_rate:.0%}) — demote to informational"
        elif cls.false_positive_rate > 0:
            cls.recommended_confidence = "WARNING"
            cls.notes = f"Some FPs ({cls.false_positive_rate:.0%}) — cannot be HARD_BLOCK"
        elif cls.undetermined_count > 0 and cls.genuine_count == 0:
            cls.recommended_confidence = "WARNING"
            cls.notes = "No confirmed genuine firings — conservative WARNING"
        else:
            cls.recommended_confidence = "WARNING"
            cls.notes = "Insufficient data for classification"


def _parse_ps_object(s: str) -> Dict[str, str]:
    """Parse PowerShell-serialised object string back to dict."""
    result = {}
    # Format: @{key1=value1; key2=value2; ...}
    s = s.strip()
    if s.startswith("@{") and s.endswith("}"):
        s = s[2:-1]
    for pair in s.split("; "):
        if "=" in pair:
            key, _, val = pair.partition("=")
            result[key.strip()] = val.strip()
    return result


def save_audit_report(result: AuditResult, output_dir: str = JOBS_ROOT) -> str:
    """Save the audit report to disk."""
    report = result.to_report()
    out_path = os.path.join(output_dir, "checker_audit_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("[audit] Report saved to %s", out_path)
    return out_path
