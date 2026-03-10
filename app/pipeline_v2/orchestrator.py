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


def _push_narrative(
    stage: str,
    title: str,
    input_summary: str = "",
    output_summary: str = "",
    sections: list = None,
    files_touched: list = None,
    warnings: list = None,
    model_used: str = "",
    duration_ms: int = 0,
    tokens_used: int = 0,
) -> None:
    """Push a narrative entry to the build project's stage log.

    Uses the current build project from the most recent DB session.
    Fails silently if no build project context is available.
    """
    try:
        from app.db import SessionLocal
        from app.builds.pipeline_bridge import notify_narrative
        from app.builds.models import BuildProject
        db = SessionLocal()
        # Find the most recently updated project with a running stage
        project = db.query(BuildProject).order_by(BuildProject.updated_at.desc()).first()
        if project:
            notify_narrative(
                db, project.id, stage,
                title=title,
                input_summary=input_summary,
                output_summary=output_summary,
                sections=sections or [],
                files_touched=files_touched or [],
                warnings=warnings or [],
                model_used=model_used,
                duration_ms=duration_ms,
                tokens_used=tokens_used,
            )
        db.close()
    except Exception as e:
        logger.debug("[orchestrator] Narrative push failed: %s", e)


def _update_stage_status(stage: str, status: str, detail: str = "") -> None:
    """Update a pipeline stage status on the current build project.

    Creates its own DB session (same pattern as _push_narrative).
    """
    try:
        from app.db import SessionLocal
        from app.builds.pipeline_bridge import (
            notify_stage_start, notify_stage_passed, notify_stage_failed,
        )
        from app.builds.models import BuildProject
        db = SessionLocal()
        project = db.query(BuildProject).order_by(BuildProject.updated_at.desc()).first()
        if project:
            pid = str(project.id)
            if status == "running":
                notify_stage_start(db, pid, stage, detail=detail)
            elif status == "passed":
                notify_stage_passed(db, pid, stage, detail=detail)
            elif status == "failed":
                notify_stage_failed(db, pid, stage, detail=detail)
        db.close()
    except Exception as e:
        logger.debug("[orchestrator] Stage status update failed: %s", e)


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

    # v2.1.1: Early abort if manifest has zero file scope
    _total_manifest_files = sum(
        len(seg.get("file_scope", []))
        for seg in manifest.get("segments", [])
    )
    if _total_manifest_files == 0:
        result.errors.append(
            "Manifest contains no files — SpecGate likely needs feature "
            "clarification (check for [HUMAN_REQUIRED] in the spec)"
        )
        result.total_duration_seconds = time.time() - t_start
        emit("❌ Manifest has 0 files across all segments — nothing to scaffold")
        emit("   This usually means SpecGate couldn't extract concrete requirements.")
        emit("   Provide a text description of the feature and re-run the pipeline.")
        return result

    from app.pipeline_v2.scaffold_engine import run_scaffold_engine

    scaffold = await run_scaffold_engine(
        manifest=manifest,
        spec=spec,
        job_dir=job_dir,
        on_progress=emit,
    )
    result.scaffold_result = scaffold

    # Push scaffold narrative
    create_files = [f.path for f in scaffold.files if f.is_new]
    modify_files = [f.path for f in scaffold.files if not f.is_new]
    _push_narrative(
        stage="critical_pipeline",
        title="Scaffold Engine Complete",
        output_summary=f"{len(create_files)} skeletons written, {len(modify_files)} MODIFY files queued",
        files_touched=create_files[:10],
        duration_ms=int(scaffold.duration_seconds * 1000),
        sections=[
            {"heading": "CREATE files", "body": "\n".join(f"- {f}" for f in create_files)},
            {"heading": "MODIFY files", "body": "\n".join(f"- {f}" for f in modify_files)},
        ] if create_files or modify_files else [],
    )

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

    # Push builder narrative
    _push_narrative(
        stage="critical_pipeline",
        title=f"Agentic Builder {'Complete' if build_result.success else 'Finished with Issues'}",
        model_used=f"{os.getenv('ASTRA_V2_BUILDER_PROVIDER', 'openai')}/{os.getenv('ASTRA_V2_BUILDER_MODEL', 'gpt-5.4')}",
        output_summary=f"{len(build_result.all_files_written)} files written, {build_result.total_tool_calls} tool calls",
        files_touched=build_result.all_files_written[:15],
        duration_ms=int(build_result.total_duration_seconds * 1000),
        tokens_used=build_result.total_input_tokens + build_result.total_output_tokens,
        warnings=build_result.errors[:5] if build_result.errors else [],
    )

    # v2.1.2: Keep builder conversation history for verify→fix continuation
    _builder_messages = build_result.messages_history

    if not build_result.all_files_written:
        result.errors.append("Builder wrote no files")
        result.total_duration_seconds = time.time() - t_start
        emit("❌ Builder failed — no files written")
        return result

    # ------------------------------------------------------------------
    # VERIFICATION: enhanced checkout or legacy screenshot loop
    # ------------------------------------------------------------------
    from app.pipeline_v2.config import ENHANCED_CHECKOUT

    # v2.1.3: Set verifier stage to running when verification loop starts
    _update_stage_status("final_checkout", "running", "Verification loop started")

    # v2.2: Enhanced multi-step checkout
    if ENHANCED_CHECKOUT:
        from app.pipeline_v2.checkout import run_enhanced_checkout
        checkout_report = await run_enhanced_checkout(spec=spec, emit=emit)
        result.total_llm_calls += 1  # visual verify call

        if checkout_report.overall_passed:
            _update_stage_status("final_checkout", "passed", "Enhanced checkout passed")
            emit("   ✅ Verifier stage → PASSED")
        else:
            # Feed failures back to builder for one fix attempt
            feedback = checkout_report.to_builder_feedback()
            if feedback:
                emit("   🔧 Sending checkout feedback to Builder for fix...")
                fix_result = await run_agentic_builder(
                    spec=spec,
                    manifest=manifest,
                    scaffold=scaffold,
                    job_dir=job_dir,
                    handover_context=feedback,
                    on_progress=emit,
                    existing_messages=_builder_messages,
                )
                result.total_llm_calls += fix_result.total_llm_calls
                _builder_messages = fix_result.messages_history

                # Re-run checkout after fix
                emit("   🔄 Re-running checkout after fix...")
                checkout_report_2 = await run_enhanced_checkout(spec=spec, emit=emit)
                result.total_llm_calls += 1

                if checkout_report_2.overall_passed:
                    _update_stage_status("final_checkout", "passed", "Enhanced checkout passed after fix")
                    emit("   ✅ Verifier stage → PASSED (after fix)")
                else:
                    _update_stage_status("final_checkout", "failed", "Enhanced checkout failed after fix attempt")
                    emit("   ❌ Verifier stage → FAILED (after fix attempt)")
            else:
                _update_stage_status("final_checkout", "failed", "Enhanced checkout failed")

        # Skip legacy loop
        result.total_duration_seconds = time.time() - t_start
        result.estimated_cost_usd = _estimate_cost(result)

        final_verify_passed = checkout_report.overall_passed if 'checkout_report_2' not in dir() else checkout_report_2.overall_passed
        result.success = final_verify_passed

        emit(f"\n{'='*60}")
        emit(f"{'✅ BUILD COMPLETE' if result.success else '❌ BUILD FINISHED WITH ISSUES'}")
        cost_str = f"${result.estimated_cost_usd:.2f}"
        emit(f"Duration: {result.total_duration_seconds:.1f}s | LLM calls: {result.total_llm_calls} | Est. cost: {cost_str}")
        emit(f"{'='*60}")
        return result

    # ------------------------------------------------------------------
    # LEGACY VERIFICATION LOOP: screenshot → verify → fix → repeat
    # ------------------------------------------------------------------
    emit(f"\n{'='*60}")
    emit("📸 VERIFICATION LOOP (legacy)")
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

            # v2.1.2: Feed boot errors back to Builder for surgical fix
            if attempt < MAX_VERIFY_LOOPS:
                emit("   🔧 Sending boot errors to Builder for fix...")
                _fix_prompt = (
                    f"The application failed to boot. Here are the errors:\n\n"
                    f"```\n{boot_output[:4000]}\n```\n\n"
                    f"Read the relevant files, diagnose the issue, and fix it. "
                    f"Focus on import errors, missing modules, syntax errors, "
                    f"and database model registration issues."
                )
                fix_result = await run_agentic_builder(
                    spec=spec,
                    manifest=manifest,
                    scaffold=scaffold,
                    job_dir=job_dir,
                    handover_context=_fix_prompt,
                    on_progress=emit,
                    existing_messages=_builder_messages,
                )
                result.total_llm_calls += fix_result.total_llm_calls
                _builder_messages = fix_result.messages_history  # Update for next iteration
                if fix_result.all_files_written:
                    emit(f"   🔧 Builder fixed {len(fix_result.all_files_written)} file(s)")
                continue  # Re-attempt boot + verify
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
            _update_stage_status("final_checkout", "passed", "Visual verification passed")
            emit("   ✅ Verifier stage → PASSED")
            break

        if attempt >= MAX_VERIFY_LOOPS:
            emit(f"   ⚠️ Verification failed after {MAX_VERIFY_LOOPS} attempts")
            _update_stage_status("final_checkout", "failed", f"Failed after {MAX_VERIFY_LOOPS} attempts")
            break

        # v2.1.2: Feed verification feedback back to Builder for surgical fix
        emit(f"   🔧 Sending verification feedback to Builder for fix...")
        _fix_prompt = (
            f"Visual verification FAILED on attempt {attempt}. "
            f"The verifier reported these issues:\n\n"
            f"{verify_result.feedback[:4000]}\n\n"
            f"Read the relevant files, diagnose what's wrong, and fix it. "
            f"Focus on the specific UI issues reported above. "
            f"Do NOT rewrite files from scratch — make surgical, targeted changes."
        )
        fix_result = await run_agentic_builder(
            spec=spec,
            manifest=manifest,
            scaffold=scaffold,
            job_dir=job_dir,
            handover_context=_fix_prompt,
            on_progress=emit,
            existing_messages=_builder_messages,
        )
        result.total_llm_calls += fix_result.total_llm_calls
        _builder_messages = fix_result.messages_history  # Update for next iteration
        if fix_result.all_files_written:
            emit(f"   🔧 Builder fixed {len(fix_result.all_files_written)} file(s)")
        emit(f"   Feedback: {verify_result.feedback[:200]}")

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------
    final_verify = result.verify_results[-1] if result.verify_results else None
    # v2.1.3: The verifier is the final authority. If it passed, the job
    # succeeded — even if the initial build needed fix iterations.
    # The old logic (build_result.success AND verify.passed) would report
    # failure when the builder needed fixes but the verifier confirmed the
    # final output was correct.
    result.success = final_verify.passed if final_verify else False
    result.total_duration_seconds = time.time() - t_start

    # Estimate cost
    result.estimated_cost_usd = _estimate_cost(result)

    # ------------------------------------------------------------------
    # COMPILE REPORT & SEND TO DEBUG
    # ------------------------------------------------------------------
    try:
        _compile_and_send_to_debug(result, job_id, spec, emit)
    except Exception as _report_err:
        logger.warning("[orchestrator] Report compilation failed: %s", _report_err)

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


def _compile_and_send_to_debug(
    result: PipelineResult,
    job_id: str,
    spec: Dict,
    emit: Any,
) -> None:
    """Compile the build report and create a debug project with it.

    This runs after the pipeline finishes (pass or fail) and creates
    a debug project pre-loaded with the full build context so the user
    can immediately reference it from the Debug tab.
    """
    emit("\n📝 Compiling build report...")

    # 1. Compile the build report via the existing service
    try:
        from app.db import SessionLocal
        from app.builds.service import compile_build_report
        from app.builds.models import BuildProject
        db = SessionLocal()
        project = db.query(BuildProject).order_by(BuildProject.updated_at.desc()).first()
        if project:
            report_json = compile_build_report(db, project.id)
            emit(f"   📝 Report compiled: {len(report_json or '')} chars")
        else:
            report_json = None
            emit("   ⚠️ No build project found for report")
        db.close()
    except Exception as e:
        emit(f"   ⚠️ Report compilation failed: {e}")
        report_json = None

    # 2. Create a debug project with the build context
    try:
        from app.debug.project_service import create_project as create_debug_project

        # Build a summary title
        spec_summary = ""
        if isinstance(spec, dict):
            spec_summary = spec.get("summary", spec.get("title", ""))[:100]
        title = f"Build: {spec_summary or job_id}"

        # Build description with key stats
        build = result.build_result
        lines = [
            f"Job ID: {job_id}",
            f"Status: {'PASSED' if result.success else 'ISSUES'}",
            f"Files: {len(build.all_files_written) if build else 0}",
            f"Tool calls: {build.total_tool_calls if build else 0}",
            f"Duration: {result.total_duration_seconds:.0f}s",
            f"Est. cost: ${result.estimated_cost_usd:.2f}",
        ]
        if result.errors:
            lines.append(f"Errors: {'; '.join(result.errors[:3])}")

        description = "\n".join(lines)
        if report_json:
            description += f"\n\n--- BUILD REPORT ---\n{report_json[:5000]}"

        debug_project = create_debug_project(
            title=title,
            description=description,
            error_summary="; ".join(result.errors[:3]) if result.errors else "",
        )
        emit(f"   🐛 Debug project created: {debug_project.get('id', '?')}")
    except Exception as e:
        emit(f"   ⚠️ Debug project creation failed: {e}")
        logger.warning("[orchestrator] Debug project creation failed: %s", e)


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
