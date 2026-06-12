# FILE: app/llm/pipeline/critique_parts/deterministic_verdict.py
# Purpose: Deterministic Verdict Engine — Orchestrates all deterministic checks.
# Called-by: app.llm.pipeline.critique
# Depends-on: app.llm.pipeline.critique_parts.coverage_checks, app.llm.pipeline.critique_parts.inventory_checks, app.llm.pipeline.critique_parts.ordering_checks, app.llm.pipeline.critique_parts.prohibition_checks (+3 more)
# Last-renovated: 2026-06-11
"""
Deterministic Verdict Engine — Orchestrates all deterministic checks.

Single entry point for critique.py to call. Runs all seven deterministic
checks and produces a unified verdict with structured pass/fail results.

The verdict determines whether the LLM critique should fire:
- If deterministic checks find blocking issues: FAIL, skip LLM
- If all pass AND needle ≤ 6: PASS, skip LLM
- If all pass AND needle ≥ 7: PASS, optionally run LLM safety net

v1.0 (2026-02-27): Initial implementation — Stage 1 of deterministic
verification migration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.llm.pipeline.critique_schemas import CritiqueIssue, CritiqueResult

# Import each check module directly (no facades)
from app.llm.pipeline.critique_parts.inventory_checks import (
    check_file_inventory_compliance,
    check_create_modify_classification,
    check_import_contracts,
)
from app.llm.pipeline.critique_parts.coverage_checks import (
    check_acceptance_criteria_coverage,
)
from app.llm.pipeline.critique_parts.signature_checks import (
    check_function_signatures,
)
from app.llm.pipeline.critique_parts.ordering_checks import (
    check_dependency_order,
)
from app.llm.pipeline.critique_parts.prohibition_checks import (
    check_prohibited_definitions,
)
from app.llm.pipeline.critique_parts.size_checks import (
    check_size_estimation,
)

logger = logging.getLogger(__name__)

DETERMINISTIC_VERDICT_BUILD_ID = "2026-02-27-v1.0-unified-verdict"


# =========================================================================
# RESULT TYPE
# =========================================================================

@dataclass
class DeterministicVerdict:
    """Result of all deterministic checks."""
    passed: bool
    blocking_count: int = 0
    warning_count: int = 0
    issues: List[Dict[str, Any]] = field(default_factory=list)
    checks_run: int = 0
    skipped_checks: List[str] = field(default_factory=list)

    def to_critique_result(self, model_label: str = "deterministic_v3.0") -> CritiqueResult:
        """
        Convert to a CritiqueResult that downstream stages expect.

        This ensures the skip-LLM path produces a valid response object.
        """
        blocking = []
        non_blocking = []

        for issue in self.issues:
            ci = CritiqueIssue(
                id=issue.get("rule_id", "DET-UNKNOWN"),
                spec_ref=issue.get("spec_ref"),
                arch_ref=issue.get("arch_ref"),
                category=issue.get("rule_id", "deterministic").split("-")[1].lower()
                         if "-" in issue.get("rule_id", "") else "deterministic",
                severity=issue.get("severity", "warning"),
                description=issue.get("description", ""),
                fix_suggestion=issue.get("suggested_fix", ""),
            )
            if issue.get("severity") == "blocking":
                blocking.append(ci)
            else:
                non_blocking.append(ci)

        summary = (
            f"Deterministic verdict: {self.checks_run} checks run, "
            f"{self.blocking_count} blocking, {self.warning_count} warnings"
        )
        if self.passed:
            summary = f"Architecture PASSED deterministic verification ({self.checks_run} checks, {self.warning_count} warnings)"

        return CritiqueResult(
            summary=summary,
            critique_model=model_label,
            critique_failed=False,
            critique_mode="deterministic",
            blocking_issues=blocking,
            non_blocking_issues=non_blocking,
        )


# =========================================================================
# MAIN ENTRY POINT
# =========================================================================

def run_deterministic_verdict(
    *,
    arch_content: str,
    segment_id: Optional[str] = None,
    spec_json: Optional[str] = None,
    spec_markdown: Optional[str] = None,
    segment_spec: Optional[Dict[str, Any]] = None,
    skeleton_contract: Optional[Dict[str, Any]] = None,
    skeleton_file_scope: Optional[List[str]] = None,
    enrichment_data: Optional[Dict[str, Any]] = None,
    manifest_dict: Optional[Dict[str, Any]] = None,
    needle_estimate: Optional[int] = None,
) -> DeterministicVerdict:
    """
    Run all seven deterministic checks and produce a unified verdict.

    This is the single entry point for critique.py. It runs every
    applicable check and collects results.

    Args:
        arch_content: Architecture markdown document
        segment_id: This segment's ID (None for non-segmented jobs)
        spec_json: JSON string of the spec
        spec_markdown: Spec markdown
        segment_spec: Segment spec dict (has file_scope, acceptance_criteria)
        skeleton_contract: Full skeleton_contract.json dict
        skeleton_file_scope: File scope list from skeleton for this segment
        enrichment_data: Enrichment data keyed by segment_id
        manifest_dict: Full manifest dict
        needle_estimate: Complexity estimate for this segment

    Returns:
        DeterministicVerdict with all check results
    """
    all_issues: List[Dict[str, Any]] = []
    checks_run = 0
    skipped: List[str] = []

    # ---------------------------------------------------------------
    # Check 1: File inventory compliance
    # ---------------------------------------------------------------
    try:
        inv_issues = check_file_inventory_compliance(
            arch_content=arch_content,
            spec_json=spec_json,
            segment_spec=segment_spec,
            skeleton_file_scope=skeleton_file_scope,
        )
        all_issues.extend(inv_issues)
        checks_run += 1
    except Exception as e:
        logger.warning("[det_verdict] Check 1 (inventory) failed: %s", e)
        skipped.append(f"inventory: {e}")

    # ---------------------------------------------------------------
    # Check 8: CREATE vs MODIFY classification accuracy (v1.1)
    # ---------------------------------------------------------------
    try:
        classify_issues = check_create_modify_classification(
            arch_content=arch_content,
            segment_spec=segment_spec,
            skeleton_file_scope=skeleton_file_scope,
        )
        all_issues.extend(classify_issues)
        checks_run += 1
    except Exception as e:
        logger.warning("[det_verdict] Check 8 (classification) failed: %s", e)
        skipped.append(f"classification: {e}")

    # ---------------------------------------------------------------
    # Check 2: Acceptance criteria coverage
    # ---------------------------------------------------------------
    try:
        cov_issues = check_acceptance_criteria_coverage(
            arch_content=arch_content,
            spec_json=spec_json,
            spec_markdown=spec_markdown,
            segment_spec=segment_spec,
        )
        all_issues.extend(cov_issues)
        checks_run += 1
    except Exception as e:
        logger.warning("[det_verdict] Check 2 (coverage) failed: %s", e)
        skipped.append(f"coverage: {e}")

    # ---------------------------------------------------------------
    # Check 3: Import contract validation
    # ---------------------------------------------------------------
    if segment_id:
        try:
            imp_issues = check_import_contracts(
                arch_content=arch_content,
                segment_id=segment_id,
                skeleton_contract=skeleton_contract,
                enrichment_data=enrichment_data,
                manifest_dict=manifest_dict,
            )
            all_issues.extend(imp_issues)
            checks_run += 1
        except Exception as e:
            logger.warning("[det_verdict] Check 3 (imports) failed: %s", e)
            skipped.append(f"imports: {e}")
    else:
        skipped.append("imports: no segment_id")

    # ---------------------------------------------------------------
    # Check 4: Function signature matching
    # ---------------------------------------------------------------
    if segment_id:
        try:
            sig_issues = check_function_signatures(
                arch_content=arch_content,
                segment_id=segment_id,
                skeleton_contract=skeleton_contract,
                enrichment_data=enrichment_data,
            )
            all_issues.extend(sig_issues)
            checks_run += 1
        except Exception as e:
            logger.warning("[det_verdict] Check 4 (signatures) failed: %s", e)
            skipped.append(f"signatures: {e}")
    else:
        skipped.append("signatures: no segment_id")

    # ---------------------------------------------------------------
    # Check 5: Dependency order verification
    # ---------------------------------------------------------------
    if segment_id and manifest_dict:
        try:
            ord_issues = check_dependency_order(
                arch_content=arch_content,
                segment_id=segment_id,
                manifest_dict=manifest_dict,
            )
            all_issues.extend(ord_issues)
            checks_run += 1
        except Exception as e:
            logger.warning("[det_verdict] Check 5 (ordering) failed: %s", e)
            skipped.append(f"ordering: {e}")
    else:
        skipped.append("ordering: no segment_id or manifest")

    # ---------------------------------------------------------------
    # Check 6: Prohibited pattern detection
    # ---------------------------------------------------------------
    if segment_id:
        try:
            proh_issues = check_prohibited_definitions(
                arch_content=arch_content,
                segment_id=segment_id,
                skeleton_contract=skeleton_contract,
                enrichment_data=enrichment_data,
                manifest_dict=manifest_dict,
            )
            all_issues.extend(proh_issues)
            checks_run += 1
        except Exception as e:
            logger.warning("[det_verdict] Check 6 (prohibition) failed: %s", e)
            skipped.append(f"prohibition: {e}")
    else:
        skipped.append("prohibition: no segment_id")

    # ---------------------------------------------------------------
    # Check 7: Size estimation
    # ---------------------------------------------------------------
    try:
        size_issues = check_size_estimation(arch_content=arch_content)
        all_issues.extend(size_issues)
        checks_run += 1
    except Exception as e:
        logger.warning("[det_verdict] Check 7 (size) failed: %s", e)
        skipped.append(f"size: {e}")

    # ---------------------------------------------------------------
    # Aggregate
    # ---------------------------------------------------------------
    blocking_count = sum(1 for i in all_issues if i.get("severity") == "blocking")
    warning_count = sum(1 for i in all_issues if i.get("severity") != "blocking")
    passed = blocking_count == 0

    verdict = DeterministicVerdict(
        passed=passed,
        blocking_count=blocking_count,
        warning_count=warning_count,
        issues=all_issues,
        checks_run=checks_run,
        skipped_checks=skipped,
    )

    logger.info(
        "[det_verdict] %s: %d checks, %d blocking, %d warnings, %d skipped. needle=%s",
        "PASS" if passed else "FAIL",
        checks_run, blocking_count, warning_count, len(skipped),
        needle_estimate,
    )
    print(
        f"[DEBUG] [det_verdict] {'PASS' if passed else 'FAIL'}: "
        f"{checks_run} checks, {blocking_count} blocking, "
        f"{warning_count} warnings, {len(skipped)} skipped. "
        f"needle={needle_estimate}"
    )

    return verdict


def should_skip_llm_critique(
    verdict: DeterministicVerdict,
    needle_estimate: Optional[int] = None,
) -> bool:
    """
    Determine whether to skip the LLM critique based on deterministic
    verdict and complexity.

    Rules:
    - If deterministic checks found blocking issues: doesn't matter,
      we return early with the blocking result (handled by caller)
    - If all pass AND needle ≤ 6: skip LLM
    - If all pass AND needle ≥ 7: don't skip (run LLM safety net)
    - If needle unknown: skip LLM (assume simple)
    """
    if not verdict.passed:
        return True  # Blocking issues found — skip LLM, return failures

    if needle_estimate is None:
        return True  # No needle info — skip LLM, trust deterministic

    if needle_estimate >= 7:
        logger.info(
            "[det_verdict] High-complexity segment (needle=%d) — LLM safety net enabled",
            needle_estimate,
        )
        return False  # High complexity — run LLM safety net

    return True  # Low/medium complexity — skip LLM


__all__ = [
    "DeterministicVerdict",
    "run_deterministic_verdict",
    "should_skip_llm_critique",
    "DETERMINISTIC_VERDICT_BUILD_ID",
]
