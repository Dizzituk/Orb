# FILE: app/pipeline_v2/bvl/retry_loop.py
# Purpose: BVL Retry Loop — surgical fix attempts when a tier fails.
# Called-by: app.pipeline_v2.bvl.bvl_orchestrator
# Depends-on: app.pipeline_v2.agentic_builder, app.pipeline_v2.build_targets, app.pipeline_v2.bvl.bvl_models, app.pipeline_v2.bvl.cross_model_diagnostic (+3 more)
# Last-renovated: 2026-06-11
"""
BVL Retry Loop — surgical fix attempts when a tier fails.

When any tier fails, this module assembles a minimal context package
(~1,200–3,800 tokens) and sends it to the Agentic Builder for a
targeted fix. Max 3 retries per component per tier.

The retry context is cumulative: each attempt includes all previous
error signals so the builder can see what's already been tried.

Token cost per retry: ~$0.01–$0.03 with GPT-5.4.

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-BVL-001.
v1.1 (2026-04-06): Cross-model diagnostic — Opus diagnoses failures before GPT-5.4 fixes.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from app.pipeline_v2.bvl.bvl_models import (
    ComponentResult,
    RetryContext,
    TierName,
    TierResult,
    TierStatus,
)

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile
    from app.pipeline_v2.models import BuildResult, ScaffoldResult

logger = logging.getLogger(__name__)

MAX_RETRIES_PER_TIER = 3


async def run_retry_loop(
    component_path: str,
    failed_tier: TierResult,
    tier_runner: Callable,
    spec: Dict[str, Any],
    profile: "BuildTargetProfile",
    manifest: Dict[str, Any],
    scaffold: "ScaffoldResult",
    job_dir: str,
    existing_messages: Optional[List[Dict]] = None,
    emit: Optional[Callable[[str], None]] = None,
) -> tuple:
    """Run the retry loop for a failed component.

    Attempts up to MAX_RETRIES_PER_TIER fixes. Each retry:
      1. Assembles surgical context (failing file + error + spec slice)
      2. Sends to Agentic Builder for fix
      3. Re-runs the tier that failed
      4. If still failing, accumulates error context for next attempt

    Args:
        component_path: The file/component that failed.
        failed_tier: The TierResult that triggered the retry.
        tier_runner: Async function to re-run the failed tier.
        spec: Full spec dict.
        profile: Build target profile.
        manifest: Build manifest.
        scaffold: Scaffold result.
        job_dir: Job directory.
        existing_messages: Builder conversation history for continuation.
        emit: Progress callback.

    Returns:
        (final_tier_result, total_llm_calls, updated_messages)
    """
    emit = emit or (lambda msg: None)
    total_llm_calls = 0
    previous_errors = []
    messages = existing_messages or []

    spec_slice = _extract_spec_slice(spec, component_path)

    for attempt in range(1, MAX_RETRIES_PER_TIER + 1):
        # JOB 7 (2026-06-10): environment-vs-code gate. Environmental failures
        # (no emulator, login screen, backend unreachable) never reach the
        # builder — they get ONE preflight remediation attempt, then block
        # with the reason, consuming ZERO builder fix cycles.
        try:
            from app.pipeline_v2.bvl.failure_classifier import ENVIRONMENT, classify_failure
            _cls = await classify_failure(failed_tier, profile)
        except Exception as _cls_err:
            logger.warning("[retry_loop] failure classification crashed: %s", _cls_err)
            _cls = None

        if _cls is not None and _cls.kind == ENVIRONMENT:
            emit(f"   🌍 ENVIRONMENT failure (not code): {_cls.reason}")
            if _cls.remediation:
                emit(f"      Remediation: {_cls.remediation}")
            if attempt == 1:
                emit("   🔧 Attempting automatic remediation (preflight)...")
                try:
                    from app.pipeline_v2.bvl.preflight import run_preflight
                    await run_preflight(profile, emit)
                except Exception as _pf_err:
                    logger.warning("[retry_loop] preflight remediation crashed: %s", _pf_err)
                emit(f"   🔍 Re-running {failed_tier.tier.value} after remediation...")
                new_result = await tier_runner()
                if new_result.passed:
                    emit(f"   ✅ {failed_tier.tier.value} PASSED after environment remediation")
                    return new_result, total_llm_calls, messages
                previous_errors.append(new_result.error_context[:500])
                failed_tier = new_result
                continue
            failed_tier.error_context = (
                f"ENVIRONMENT: {_cls.reason} | {_cls.remediation} | "
                + (failed_tier.error_context or "")
            )[:2000]
            emit("   🚫 Blocked by ENVIRONMENT, not code — no builder fix attempts spent")
            return failed_tier, total_llm_calls, messages

        emit(f"   🔄 Retry {attempt}/{MAX_RETRIES_PER_TIER} for {component_path}")

        # Build retry context
        ctx = RetryContext(
            component_path=component_path,
            tier=failed_tier.tier,
            attempt=attempt,
            error_signal=failed_tier.error_context[:2000],
            spec_slice=spec_slice,
            previous_errors=previous_errors.copy(),
            logcat_excerpt=failed_tier.logcat_excerpt[:1000],
        )

        emit(f"   📦 Retry context: ~{ctx.estimated_tokens} tokens")

        # v1.1: Cross-model diagnosis before builder fix
        # Send the failure to Opus for diagnosis, then give that
        # diagnosis to GPT-5.4 for the actual fix. Two perspectives
        # on every failure instead of the builder debugging itself.
        fix_prompt = ctx.to_builder_prompt()

        try:
            from app.pipeline_v2.bvl.cross_model_diagnostic import (
                diagnose_failure,
                read_source_file,
            )
            source_code = await read_source_file(component_path, profile)
            diagnosis = await diagnose_failure(
                component_path=component_path,
                error_signal=failed_tier.error_context[:2000],
                logcat_excerpt=failed_tier.logcat_excerpt[:1000],
                spec_slice=spec_slice,
                source_code=source_code,
                previous_errors=previous_errors,
                profile=profile,
                emit=emit,
                full_spec=spec,
                manifest=manifest,
            )
            if diagnosis:
                fix_prompt = diagnosis
                total_llm_calls += 1
            else:
                emit("   ⚠️ Cross-model diagnosis unavailable, using raw error context")
        except Exception as _diag_err:
            logger.warning("[retry_loop] Cross-model diagnosis failed: %s", _diag_err)
            emit("   ⚠️ Cross-model diagnosis failed, using raw error context")

        from app.pipeline_v2.agentic_builder import run_agentic_builder

        fix_result = await run_agentic_builder(
            spec=spec,
            manifest=manifest,
            scaffold=scaffold,
            job_dir=job_dir,
            handover_context=fix_prompt,
            on_progress=emit,
            existing_messages=messages,
            profile=profile,
        )
        total_llm_calls += fix_result.total_llm_calls
        messages = fix_result.messages_history

        if fix_result.all_files_written:
            emit(f"   🔧 Builder fixed {len(fix_result.all_files_written)} file(s)")
        else:
            emit("   ⚠️ Builder made no changes")

        # Re-run the failed tier
        emit(f"   🔍 Re-running {failed_tier.tier.value}...")
        new_result = await tier_runner()

        if new_result.passed:
            emit(f"   ✅ {failed_tier.tier.value} PASSED on retry {attempt}")
            return new_result, total_llm_calls, messages

        # Accumulate error for next attempt
        previous_errors.append(new_result.error_context[:500])
        failed_tier = new_result
        emit(f"   ❌ Still failing after retry {attempt}")

    # Exhausted retries
    emit(f"   🚫 {component_path} BLOCKED after {MAX_RETRIES_PER_TIER} retries")
    return failed_tier, total_llm_calls, messages


def build_component_result(
    component_path: str,
    tier_results: List[TierResult],
    retries: int = 0,
) -> ComponentResult:
    """Build a ComponentResult from accumulated tier results.

    A component is signed off if all tiers passed. It's blocked
    if any tier failed after exhausting retries.
    """
    result = ComponentResult(
        component_path=component_path,
        tier_results=tier_results,
        retry_attempts=retries,
    )

    # Check if all tiers passed
    tier_passed = {tr.tier: tr.passed for tr in tier_results}
    all_passed = (
        tier_passed.get(TierName.SANITY, False)
        and tier_passed.get(TierName.BEHAVIORAL, False)
        and tier_passed.get(TierName.ADVERSARIAL, False)
    )

    if all_passed:
        result.signed_off = True
    elif retries >= MAX_RETRIES_PER_TIER:
        result.blocked = True
        # Find the failing tier
        for tr in tier_results:
            if not tr.passed:
                result.block_reason = (
                    f"Failed {tr.tier.value} after {retries} retries: "
                    f"{tr.error_context[:200]}"
                )
                break

    # Total duration
    result.total_duration_ms = sum(tr.duration_ms for tr in tier_results)

    return result


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _extract_spec_slice(
    spec: Dict[str, Any],
    component_path: str,
) -> str:
    """Extract the relevant portion of the spec for a component.

    Looks for segments that reference the component's file path
    and returns their requirements and acceptance criteria.
    """
    if not isinstance(spec, dict):
        return str(spec)[:2000]

    # Normalise the component path for matching
    norm_path = component_path.replace("\\", "/").lower()
    file_name = norm_path.rsplit("/", 1)[-1] if "/" in norm_path else norm_path

    relevant_parts = []

    # Search segments for file references
    segments = spec.get("segments", [])
    if not segments:
        # Flat spec — return summary
        return json.dumps(spec, indent=2)[:2000]

    for seg in segments:
        files = seg.get("file_scope", [])
        file_names = [f.replace("\\", "/").lower() for f in files]

        if any(file_name in f or norm_path in f for f in file_names):
            relevant_parts.append(
                f"Segment: {seg.get('segment_id', '')}\n"
                f"Requirements: {json.dumps(seg.get('requirements', []))}\n"
                f"Files: {json.dumps(files)}"
            )

    if relevant_parts:
        return "\n\n".join(relevant_parts)[:2000]

    # Fallback: return spec summary
    summary = spec.get("summary", spec.get("title", ""))
    return summary[:2000] if summary else json.dumps(spec, indent=2)[:2000]
