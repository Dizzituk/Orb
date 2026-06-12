# FILE: app/pipeline_v2/bvl/bvl_orchestrator.py
# Purpose: BVL Orchestrator — coordinates the three-tier verification cascade.
# Called-by: app.pipeline_v2.orchestrator
# Depends-on: app.pipeline_v2.build_targets, app.pipeline_v2.bvl.bvl_models, app.pipeline_v2.bvl.emulator_bridge, app.pipeline_v2.bvl.preflight (+6 more)
# Last-renovated: 2026-06-11
"""
BVL Orchestrator — coordinates the three-tier verification cascade.

This is the single entry point the pipeline calls. It:
  1. Runs Tier 1 (Sanity Gate)
  2. On T1 pass → generates test flows → runs Tier 2 (Behavioral Gate)
  3. On T2 pass → runs Tier 3 (Adversarial Gate)
  4. On any failure → triggers the retry loop
  5. Produces a BVLReport for the debrief system

The orchestrator is NOT smart. It is reliable. It routes data between
tiers, manages the retry budget, and assembles the final report.

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-BVL-001.
v1.1 (2026-04-18): Fix double-append bug — retry results now REPLACE failed
                   results for the same tier instead of stacking alongside
                   them. The previous behaviour caused the final
                   all(tr.passed ...) sign-off check to see the old failed
                   result and mark the component as not-signed-off even when
                   retries succeeded. Helper _replace_tier_result keeps only
                   the latest result per tier.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from app.pipeline_v2.bvl.bvl_models import (
    BVLReport,
    ComponentResult,
    TierName,
    TierResult,
    TierStatus,
)
from app.pipeline_v2.bvl.retry_loop import (
    MAX_RETRIES_PER_TIER,
    build_component_result,
    run_retry_loop,
)

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile
    from app.pipeline_v2.models import ScaffoldResult

logger = logging.getLogger(__name__)


async def run_bvl(
    job_id: str,
    spec: Dict[str, Any],
    manifest: Dict[str, Any],
    scaffold: "ScaffoldResult",
    profile: "BuildTargetProfile",
    job_dir: str,
    existing_messages: Optional[List[Dict]] = None,
    emit: Optional[Callable[[str], None]] = None,
) -> BVLReport:
    """Run the complete BVL cascade.

    This is the main entry point called by the pipeline orchestrator.
    Replaces the current verification step for Android/emulator builds.

    Args:
        job_id: Pipeline job identifier.
        spec: Verified spec from SpecGate.
        manifest: Segment manifest.
        scaffold: Scaffold result from the Scaffold Engine.
        profile: Build target profile.
        job_dir: Job directory for artifacts.
        existing_messages: Builder conversation history.
        emit: Progress callback.

    Returns:
        BVLReport with full verification results.
    """
    emit = emit or (lambda msg: None)
    t_start = time.time()
    messages = existing_messages or []
    total_llm_calls = 0
    total_retries = 0
    tier_results: List[TierResult] = []

    emit(f"\n{'='*60}")
    emit("🔬 BEHAVIORAL VERIFICATION LAYER (BVL)")
    emit(f"{'='*60}")


    # v2.3: Set the active profile on the emulator bridge so ADB commands
    # route to the correct shell (host for Android, sandbox for ASTRA).
    from app.pipeline_v2.bvl.emulator_bridge import set_active_profile
    set_active_profile(profile)
    emit(f"   Target: {profile.project_name} ({profile.language})")

    # ── PRE-FLIGHT (Job 6, 2026-06-10): get the app into a testable state ──
    emit(f"\n{'─'*40}")
    emit("🛫 PRE-FLIGHT: environment readiness")
    emit(f"{'─'*40}")
    try:
        from app.pipeline_v2.bvl.preflight import run_preflight
        _pf = await run_preflight(profile, emit)
        for _name, _ok, _msg in _pf.steps:
            emit(f"   {'✅' if _ok else '⚠️'} {_name}" + (f": {_msg}" if _msg else ""))
        if _pf.ready:
            emit("   🛫 Pre-flight: app is testable (logged in)")
        else:
            emit(f"   ⚠️ Pre-flight incomplete: {_pf.reason} — continuing; failures will be classified")
    except Exception as _pf_exc:
        logger.warning("[bvl] preflight crashed (non-fatal): %s", _pf_exc)
        emit(f"   ⚠️ Pre-flight crashed (non-fatal): {_pf_exc}")

    # ── TIER 1: Sanity Gate ──
    emit(f"\n{'─'*40}")
    emit("🔍 TIER 1: SANITY GATE")
    emit(f"{'─'*40}")

    from app.pipeline_v2.bvl.tier1_sanity import run_tier1

    spec_text = _spec_to_text(spec)
    t1_result = await run_tier1(
        profile=profile,
        spec_text=spec_text,
        emit=emit,
    )
    tier_results.append(t1_result)

    if not t1_result.passed:
        # Retry loop for T1
        async def rerun_t1():
            return await run_tier1(profile=profile, spec_text=spec_text, emit=emit)

        t1_result, llm_calls, messages = await run_retry_loop(
            component_path=profile.project_root,
            failed_tier=t1_result,
            tier_runner=rerun_t1,
            spec=spec,
            profile=profile,
            manifest=manifest,
            scaffold=scaffold,
            job_dir=job_dir,
            existing_messages=messages,
            emit=emit,
        )
        total_llm_calls += llm_calls
        total_retries += min(MAX_RETRIES_PER_TIER, 3)
        _replace_tier_result(tier_results, t1_result)

        if not t1_result.passed:
            # BLOCKED at T1 — cannot proceed
            return _build_report(
                job_id, profile, tier_results,
                total_retries, total_llm_calls, t_start,
                blocked=True,
                block_reason=t1_result.error_context,
                emit=emit,
            )

    # ── TIER 2: Behavioral Gate ──
    emit(f"\n{'─'*40}")
    emit("🧪 TIER 2: BEHAVIORAL GATE")
    emit(f"{'─'*40}")

    from app.pipeline_v2.bvl.tier2_test_generator import generate_test_flows
    from app.pipeline_v2.bvl.tier2_behavioral import run_tier2

    flows = await generate_test_flows(spec=spec, profile=profile, emit=emit)
    total_llm_calls += 1  # Test generation call

    t2_result = await run_tier2(flows=flows, profile=profile, emit=emit)
    tier_results.append(t2_result)

    if not t2_result.passed and t2_result.status != TierStatus.SKIPPED:
        # Retry loop for T2 — must re-run T1 then T2
        async def rerun_t1_t2():
            t1 = await run_tier1(profile=profile, spec_text=spec_text, emit=emit)
            if not t1.passed:
                return t1
            return await run_tier2(flows=flows, profile=profile, emit=emit)

        t2_result, llm_calls, messages = await run_retry_loop(
            component_path=profile.project_root,
            failed_tier=t2_result,
            tier_runner=rerun_t1_t2,
            spec=spec,
            profile=profile,
            manifest=manifest,
            scaffold=scaffold,
            job_dir=job_dir,
            existing_messages=messages,
            emit=emit,
        )
        total_llm_calls += llm_calls
        total_retries += min(MAX_RETRIES_PER_TIER, 3)
        _replace_tier_result(tier_results, t2_result)

        if not t2_result.passed:
            return _build_report(
                job_id, profile, tier_results,
                total_retries, total_llm_calls, t_start,
                blocked=True,
                block_reason=t2_result.error_context,
                emit=emit,
            )

    # ── TIER 3: Adversarial Gate ──
    emit(f"\n{'─'*40}")
    emit("💥 TIER 3: ADVERSARIAL GATE")
    emit(f"{'─'*40}")

    from app.pipeline_v2.bvl.tier3_adversarial import run_tier3

    t3_result = await run_tier3(profile=profile, emit=emit)
    tier_results.append(t3_result)

    if not t3_result.passed:
        # Retry loop for T3 — must re-run full cascade
        async def rerun_full_cascade():
            t1 = await run_tier1(profile=profile, spec_text=spec_text, emit=emit)
            if not t1.passed:
                return t1
            t2 = await run_tier2(flows=flows, profile=profile, emit=emit)
            if not t2.passed:
                return t2
            return await run_tier3(profile=profile, emit=emit)

        t3_result, llm_calls, messages = await run_retry_loop(
            component_path=profile.project_root,
            failed_tier=t3_result,
            tier_runner=rerun_full_cascade,
            spec=spec,
            profile=profile,
            manifest=manifest,
            scaffold=scaffold,
            job_dir=job_dir,
            existing_messages=messages,
            emit=emit,
        )
        total_llm_calls += llm_calls
        total_retries += min(MAX_RETRIES_PER_TIER, 3)
        _replace_tier_result(tier_results, t3_result)

    # ── All tiers complete ──
    all_passed = all(
        tr.passed for tr in tier_results
        if tr.status != TierStatus.SKIPPED
    )

    return _build_report(
        job_id, profile, tier_results,
        total_retries, total_llm_calls, t_start,
        blocked=not all_passed and total_retries >= MAX_RETRIES_PER_TIER,
        block_reason=tier_results[-1].error_context if not all_passed else "",
        emit=emit,
    )


# ═══════════════════════════════════════════════════════════════════
# Report builder
# ═══════════════════════════════════════════════════════════════════

def _build_report(
    job_id: str,
    profile: "BuildTargetProfile",
    tier_results: List[TierResult],
    total_retries: int,
    total_llm_calls: int,
    t_start: float,
    blocked: bool = False,
    block_reason: str = "",
    emit: Optional[Callable[[str], None]] = None,
) -> BVLReport:
    """Assemble the final BVLReport."""
    emit = emit or (lambda msg: None)
    duration = time.time() - t_start

    component = build_component_result(
        component_path=profile.project_root,
        tier_results=tier_results,
        retries=total_retries,
    )
    if blocked:
        component.blocked = True
        component.block_reason = block_reason
    elif all(tr.passed for tr in tier_results if tr.status != TierStatus.SKIPPED):
        component.signed_off = True

    # Estimate cost: ~$0.01 per verification call, ~$0.03 per retry
    estimated_cost = (total_llm_calls * 0.01) + (total_retries * 0.03)

    report = BVLReport(
        job_id=job_id,
        component_results=[component],
        total_duration_seconds=duration,
        total_retries=total_retries,
        total_llm_calls=total_llm_calls,
        estimated_cost_usd=estimated_cost,
    )

    # Emit summary
    emit(f"\n{'='*60}")
    if report.all_signed_off:
        emit("✅ BVL: ALL TIERS PASSED — BUILD SIGNED OFF")
    else:
        emit("❌ BVL: VERIFICATION INCOMPLETE")
    emit(report.summary())
    emit(f"{'='*60}")

    return report


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _spec_to_text(spec: Dict[str, Any]) -> str:
    """Convert spec dict to text for visual comparison."""
    import json
    if isinstance(spec, dict):
        return json.dumps(spec, indent=2)
    return str(spec)


def _replace_tier_result(
    tier_results: List[TierResult],
    new_result: TierResult,
) -> None:
    """Replace the existing result for this tier with the new one.

    If no prior result exists for this tier, append. Otherwise, overwrite
    the existing entry. This ensures the final sign-off check
    (``all(tr.passed ...)``) sees only the latest state per tier, so a
    failed-then-retried-successfully tier is treated as passed, not failed.

    v1.1 (2026-04-18): Introduced to fix the "always fails" bug where
    retry results were being appended alongside the original failed result,
    poisoning the aggregate sign-off check.
    """
    for i, tr in enumerate(tier_results):
        if tr.tier == new_result.tier:
            tier_results[i] = new_result
            return
    tier_results.append(new_result)
