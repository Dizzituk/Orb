# FILE: app/pipeline_v2/orchestrator.py
"""
ASTRA v2.1 Orchestrator.

Simplified from v2.0. Four stages:
  Weaver → SpecGate → Scaffold Engine → Agentic Builder (with verify loop)

The orchestrator is not smart. It is reliable. It routes data between
stages, tracks progress, and manages the verify loop.

v2.1 (2026-03-07): Simplified from v2.0 (removed Architect, tiered Builder).
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, Optional

from app.pipeline_v2.config import MAX_VERIFY_LOOPS, MAX_FALLBACK_ATTEMPTS
from app.pipeline_v2.models import PipelineResult

logger = logging.getLogger(__name__)


async def run_v2_pipeline(
    job_id: str,
    manifest: Dict[str, Any],
    spec: Dict[str, Any],
    intent_text: str,
    job_dir: str,
    on_progress: Optional[Callable[[str], None]] = None,
) -> PipelineResult:
    """Run the complete ASTRA v2.1 pipeline.

    Stages 1-2 (Weaver + SpecGate) have already run.
    This function runs stages 3-4: Scaffold Engine → Agentic Builder.

    Args:
        job_id: The job identifier.
        manifest: Segment manifest from SpecGate.
        spec: The verified spec content.
        intent_text: The Weaver's intent document (for verification).
        job_dir: Job directory for saving artifacts.
        on_progress: Progress callback for UI updates.
    """
    result = PipelineResult(job_id=job_id)
    t_start = time.time()
    emit = on_progress or (lambda msg: None)

    emit(f"\n{'='*60}")
    emit(f"🚀 ASTRA v2.1 Pipeline — Job {job_id}")
    emit(f"{'='*60}")

    # ------------------------------------------------------------------
    # STAGE 3: Scaffold Engine (deterministic, no LLM)
    # ------------------------------------------------------------------
    emit(f"\n{'='*60}")
    emit("🏗️ STAGE 3: SCAFFOLD ENGINE")
    emit(f"{'='*60}")

    from app.pipeline_v2.scaffold_engine import run_scaffold_engine

    scaffold = await run_scaffold_engine(
        manifest=manifest,
        spec=spec,
        job_dir=job_dir,
        on_progress=emit,
    )
    result.scaffold_result = scaffold

    if not scaffold.files:
        result.errors.append("Scaffold Engine produced no files")
        result.total_duration_seconds = time.time() - t_start
        emit("❌ Scaffold failed — no files produced")
        return result

    # ------------------------------------------------------------------
    # STAGE 4: Agentic Builder (one model, one loop)
    # ------------------------------------------------------------------
    emit(f"\n{'='*60}")
    emit("🤖 STAGE 4: AGENTIC BUILDER")
    emit(f"{'='*60}")

    from app.pipeline_v2.agentic_builder import run_agentic_builder

    build_result = await run_agentic_builder(
        spec=spec,
        manifest=manifest,
        scaffold=scaffold,
        job_dir=job_dir,
        on_progress=emit,
    )
    result.build_result = build_result
    result.total_llm_calls += build_result.total_llm_calls

    if not build_result.all_files_written:
        result.errors.append("Builder wrote no files")
        result.total_duration_seconds = time.time() - t_start
        emit("❌ Builder failed — no files written")
        return result

    # ------------------------------------------------------------------
    # VERIFICATION LOOP: screenshot → verify → fix → repeat
    # ------------------------------------------------------------------
    emit(f"\n{'='*60}")
    emit("📸 VERIFICATION LOOP")
    emit(f"{'='*60}")

    from app.pipeline_v2.verification import verify_visually

    spec_text = json.dumps(spec, indent=2) if isinstance(spec, dict) else str(spec)

    for attempt in range(1, MAX_VERIFY_LOOPS + 1):
        emit(f"\n--- Verification attempt {attempt}/{MAX_VERIFY_LOOPS} ---")

        # Boot the app first
        emit("   🔄 Booting application...")
        from app.pipeline_v2.sandbox_tools import boot_check
        boot_ok, boot_output = await boot_check()

        if not boot_ok:
            emit(f"   ❌ Boot failed: {boot_output[:200]}")
            # TODO: Feed boot errors back to Builder for fix
            # For now, continue to verification which will catch the blank screen
        else:
            emit("   ✅ Boot OK")

        # Visual verification
        verify_result = await verify_visually(
            spec_text=spec_text,
            attempt=attempt,
            emit=emit,
        )
        result.verify_results.append(verify_result)
        result.total_llm_calls += 1

        if verify_result.passed:
            emit("   ✅ VERIFICATION PASSED")
            # Report Verifier stage as passed to the build tracker
            try:
                from app.builds.pipeline_bridge import notify_stage_passed
                _build_project_id = os.environ.get("ASTRA_CURRENT_BUILD_PROJECT_ID", "")
                if _build_project_id:
                    notify_stage_passed(_build_project_id, "final_checkout")
                    emit("   ✅ Verifier stage reported as PASSED")
            except Exception as _e:
                logger.debug("[orchestrator] Could not report verifier stage: %s", _e)
            break

        if attempt >= MAX_VERIFY_LOOPS:
            emit(f"   ⚠️ Verification failed after {MAX_VERIFY_LOOPS} attempts")
            break

        # Feed feedback back to Builder
        emit(f"   🔄 Sending feedback to Builder...")
        # TODO: Re-run agentic builder with verification feedback
        # This requires keeping the builder's context or starting fresh
        # with a handover that includes the feedback.
        emit(f"   Feedback: {verify_result.feedback[:200]}")

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------
    final_verify = result.verify_results[-1] if result.verify_results else None
    result.success = build_result.success and (
        final_verify.passed if final_verify else False
    )
    result.total_duration_seconds = time.time() - t_start

    # Estimate cost
    result.estimated_cost_usd = _estimate_cost(result)

    status = "COMPLETE ✅" if result.success else "FINISHED WITH ISSUES ⚠️"
    emit(f"\n{'='*60}")
    emit(f"🚀 ASTRA v2.1 Pipeline {status}")
    emit(f"   Files: {len(build_result.all_files_written)}")
    emit(f"   Tool calls: {build_result.total_tool_calls}")
    emit(f"   LLM calls: {result.total_llm_calls}")
    emit(f"   Duration: {result.total_duration_seconds:.1f}s")
    emit(f"   Est. cost: ${result.estimated_cost_usd:.2f}")
    if result.errors:
        emit(f"   Errors: {result.errors}")
    emit(f"{'='*60}")

    return result


def _estimate_cost(result: PipelineResult) -> float:
    """Rough cost estimate based on token usage."""
    if not result.build_result:
        return 0.0

    # GPT-5.4 rates: $2.50/MTok input, $15.00/MTok output
    input_cost = (result.build_result.total_input_tokens / 1_000_000) * 2.50
    output_cost = (result.build_result.total_output_tokens / 1_000_000) * 15.00

    # Verification: ~$0.01 per call
    verify_cost = len(result.verify_results) * 0.01

    return input_cost + output_cost + verify_cost
