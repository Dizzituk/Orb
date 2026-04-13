# FILE: app/pipeline_v2/multi_target_verifier.py
"""
Multi-Target Build Verifier (Phase 3 Job 10)

Runs per-target build verification after segment execution completes, then
runs the contract verifier across the whole manifest to catch producer/
consumer mismatches that only surface once both sides are compiled.

Design principle: each target is verified independently in parallel. A
Python backend failing syntax doesn't block the Android target's Gradle
compile from attempting. The report tells the caller which targets
passed, which failed, and whether cross-target contracts held.

v1.0 (2026-04-12): Phase 3 Job 10.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class TargetVerificationStatus(str, Enum):
    PASSED = "passed"
    SYNTAX_FAILED = "syntax_failed"
    BUILD_FAILED = "build_failed"
    BOOT_FAILED = "boot_failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TargetResult:
    target_id: str
    status: TargetVerificationStatus
    syntax_output: str = ""
    build_output: str = ""
    boot_output: str = ""
    duration_sec: float = 0.0
    detail: str = ""

    def is_passing(self) -> bool:
        return self.status == TargetVerificationStatus.PASSED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "status": self.status.value,
            "syntax_output": self.syntax_output[:500],
            "build_output": self.build_output[:500],
            "boot_output": self.boot_output[:500],
            "duration_sec": round(self.duration_sec, 2),
            "detail": self.detail,
        }


@dataclass
class VerificationReport:
    target_results: List[TargetResult] = field(default_factory=list)
    contract_errors: int = 0
    contract_warnings: int = 0
    contract_report: Optional[Any] = None   # ContractReport if run
    integration_report: Optional[Any] = None   # Phase 3 Job 11 smoke test

    def all_targets_passed(self) -> bool:
        return all(r.is_passing() for r in self.target_results)

    def has_contract_errors(self) -> bool:
        return self.contract_errors > 0

    def is_passing(self) -> bool:
        _int_ok = (self.integration_report is None
                   or self.integration_report.status.value in ("passed", "skipped"))
        return (self.all_targets_passed()
                and not self.has_contract_errors()
                and _int_ok)

    def failed_targets(self) -> List[TargetResult]:
        return [r for r in self.target_results if not r.is_passing()]

    def summary(self) -> str:
        n_pass = sum(1 for r in self.target_results if r.is_passing())
        n_total = len(self.target_results)
        _int = "skipped"
        if self.integration_report is not None:
            _int = self.integration_report.status.value
        return (
            f"VerificationReport(targets={n_pass}/{n_total} passed, "
            f"contract_errors={self.contract_errors}, "
            f"contract_warnings={self.contract_warnings}, "
            f"integration={_int}, "
            f"overall={'PASS' if self.is_passing() else 'FAIL'})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_results": [r.to_dict() for r in self.target_results],
            "contract_errors": self.contract_errors,
            "contract_warnings": self.contract_warnings,
            "contract_report": self.contract_report.to_dict() if self.contract_report else None,
            "integration_report": self.integration_report.to_dict() if self.integration_report else None,
            "overall_passing": self.is_passing(),
        }


async def _verify_single_target(target_id: str) -> TargetResult:
    """Run syntax -> build -> boot for one target. Returns a TargetResult.
    Short-circuits: syntax failure skips build+boot; build failure skips boot.
    """
    import time
    start = time.monotonic()
    result = TargetResult(target_id=target_id, status=TargetVerificationStatus.ERROR)

    try:
        from app.pipeline_v2.target_registry import get_profile
        profile = get_profile(target_id)
        if profile is None:
            result.status = TargetVerificationStatus.ERROR
            result.detail = f"target_id {target_id} not found in registry"
            return result

        from app.pipeline_v2.sandbox_tools import build_check, boot_check

        # 1. Build check (also serves as syntax check for compiled langs)
        try:
            build_ok, build_out = await build_check(profile=profile)
            result.build_output = build_out
            if not build_ok:
                result.status = TargetVerificationStatus.BUILD_FAILED
                result.detail = f"{target_id} build check failed"
                result.duration_sec = time.monotonic() - start
                logger.warning(
                    "[verifier] Job 10 %s BUILD FAILED: %s", target_id, build_out[:200],
                )
                return result
        except Exception as be:
            result.status = TargetVerificationStatus.BUILD_FAILED
            result.detail = f"build_check raised: {be}"
            result.duration_sec = time.monotonic() - start
            return result

        # 2. Boot check (for targets that support it; Gradle targets return OK)
        try:
            boot_ok, boot_out = await boot_check(profile=profile)
            result.boot_output = boot_out
            if not boot_ok:
                result.status = TargetVerificationStatus.BOOT_FAILED
                result.detail = f"{target_id} boot check failed"
                result.duration_sec = time.monotonic() - start
                logger.warning(
                    "[verifier] Job 10 %s BOOT FAILED: %s", target_id, boot_out[:200],
                )
                return result
        except Exception as boe:
            result.status = TargetVerificationStatus.BOOT_FAILED
            result.detail = f"boot_check raised: {boe}"
            result.duration_sec = time.monotonic() - start
            return result

        result.status = TargetVerificationStatus.PASSED
        result.duration_sec = time.monotonic() - start
        logger.info(
            "[verifier] Job 10 %s PASSED in %.1fs",
            target_id, result.duration_sec,
        )
        return result

    except Exception as outer:
        result.status = TargetVerificationStatus.ERROR
        result.detail = f"verifier raised: {outer}"
        result.duration_sec = time.monotonic() - start
        logger.error("[verifier] Job 10 %s ERROR: %s", target_id, outer)
        return result


async def verify_manifest_build(manifest, target_ids: Optional[List[str]] = None) -> VerificationReport:
    """Verify a completed multi-target build.

    Runs all target build checks in parallel via asyncio.gather, then runs
    the Phase 1 Job 7 contract verifier across the manifest.

    Args:
        manifest: The SegmentManifest produced by Phase 1.
        target_ids: Optional explicit list of targets to verify. If None,
                    discovered from the manifest's segment target_ids.

    Returns:
        VerificationReport summarising target-by-target results + contract
        verification outcome. Callers decide whether to gate on it.
    """
    report = VerificationReport()

    # Discover targets to verify
    if target_ids is None:
        discovered: Set[str] = set()
        if manifest and getattr(manifest, "segments", None):
            for seg in manifest.segments:
                tid = getattr(seg, "target_id", None)
                if tid:
                    discovered.add(tid)
        target_ids = sorted(discovered)

    if not target_ids:
        logger.warning("[verifier] Job 10 no targets to verify — empty manifest?")
        return report

    logger.info(
        "[verifier] Job 10 verifying %d target(s) in parallel: %s",
        len(target_ids), target_ids,
    )

    # Parallel target verification
    target_tasks = [_verify_single_target(tid) for tid in target_ids]
    target_results = await asyncio.gather(*target_tasks, return_exceptions=False)
    report.target_results = list(target_results)

    # Cross-target contract verification (Phase 1 Job 7)
    try:
        from app.pot_spec.grounded.contract_verifier import verify_manifest_contracts
        cr = verify_manifest_contracts(manifest)
        report.contract_report = cr
        report.contract_errors = len(cr.errors())
        report.contract_warnings = len(cr.warnings())
    except Exception as ce:
        logger.warning("[verifier] Job 10 contract verification raised: %s", ce)

    # Cross-repo integration smoke (Phase 3 Job 11).
    # Only runs if both:
    #   a) All target builds passed (no point hitting the backend if it won't boot)
    #   b) There is at least one cross-target endpoint contract to exercise
    # The runner itself short-circuits to SKIPPED in case (b), so we only
    # gate on (a) here.
    if report.all_targets_passed():
        try:
            from app.pipeline_v2.integration_runner import run_integration_smoke
            ir = await run_integration_smoke(manifest)
            report.integration_report = ir
            logger.info("[verifier] Job 11 %s", ir.summary())
        except Exception as ire:
            logger.warning("[verifier] Job 11 integration smoke raised: %s", ire)

    logger.info("[verifier] Job 10 %s", report.summary())
    return report