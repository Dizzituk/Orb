# FILE: app/pipeline_v2/orchestrator.py
# Purpose: ASTRA v2.2 Orchestrator.
# Called-by: app.orchestrator.segment_loop, app.pipeline_v2.agentic_builder, app.pipeline_v2.final_verdict
# Depends-on: app.builds.models, app.builds.pipeline_bridge, app.builds.service, app.db (+16 more)
# Last-renovated: 2026-06-11
"""
ASTRA v2.2 Orchestrator.

Simplified from v2.0. Four stages:
  Weaver → SpecGate → Scaffold Engine → Agentic Builder (with verify loop)

The orchestrator is not smart. It is reliable. It routes data between
stages, tracks progress, and manages the verify loop.

v2.1 (2026-03-07): Simplified from v2.0 (removed Architect, tiered Builder).
v2.2 (2026-03-10): Multi-project targeting. Accepts BuildTargetProfile and
    passes it through to every stage. Profile determines language, paths,
    build commands, and verification strategy.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from app.pipeline_v2.config import MAX_VERIFY_LOOPS, MAX_FALLBACK_ATTEMPTS
from app.pipeline_v2.models import PipelineResult

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile

logger = logging.getLogger(__name__)


# Re-export the helpers moved out 2026-06-21 (BATCH 6) so importers resolve unchanged.
# (segment_loop <- run_v2_pipeline [below]; final_verdict + agentic_builder <- _push_narrative)
from app.pipeline_v2.orchestrator_notify import _push_narrative, _update_stage_status
from app.pipeline_v2.orchestrator_report import _compile_and_send_to_debug, _estimate_cost, _emit_ledger_summary
from app.pipeline_v2.orchestrator_spec_review import _run_spec_review_stage, _extract_build_output_from_bvl


async def run_v2_pipeline(
    job_id: str,
    manifest: Dict[str, Any],
    spec: Dict[str, Any],
    intent_text: str,
    job_dir: str,
    on_progress: Optional[Callable[[str], None]] = None,
    profile: Optional["BuildTargetProfile"] = None,
) -> PipelineResult:
    """Run the complete ASTRA v2.2 pipeline.

    Stages 1-2 (Weaver + SpecGate) have already run.
    This function runs stages 3-4: Scaffold Engine → Agentic Builder.

    Args:
        job_id: The job identifier.
        manifest: Segment manifest from SpecGate.
        spec: The verified spec content.
        intent_text: The Weaver's intent document (for verification).
        job_dir: Job directory for saving artifacts.
        on_progress: Progress callback for UI updates.
        profile: Build target profile. If None, uses default from config.
    """
    # Resolve profile — check ProjectSession first, then env default
    if profile is None:
        try:
            from app.shared_context.project_session import _sessions as _proj_sessions
            from app.pipeline_v2.target_registry import get_profile, get_default_profile
            _session_profile = None
            for _session in _proj_sessions.values():
                if _session.is_set and _session.project_id:
                    _session_profile = get_profile(_session.project_id)
                    if _session_profile:
                        break
            profile = _session_profile or get_default_profile()
        except Exception:
            from app.pipeline_v2.target_registry import get_default_profile
            profile = get_default_profile()

    result = PipelineResult(job_id=job_id)
    t_start = time.time()
    emit = on_progress or (lambda msg: None)

    emit(f"\n{'='*60}")
    emit(f"🚀 ASTRA v2.2 Pipeline — Job {job_id}")
    emit(f"   Target: {profile.project_name} ({profile.language}/{profile.framework})")
    emit(f"   Root: {profile.project_root}")
    emit(f"{'='*60}")

    # ------------------------------------------------------------------
    # DECISION LEDGER — Piece 1 of v2.2 stability upgrade
    # Seed with decisions already implied by the finalised spec + profile.
    # Stages read from it via load_ledger(job_dir); conflicts on write
    # surface drift rather than hide it.
    # ------------------------------------------------------------------
    try:
        from app.pipeline_v2.config import LEDGER_ENABLED
        if LEDGER_ENABLED:
            from app.pipeline_v2.ledger import load_or_create_ledger
            from app.pipeline_v2.ledger_seed import seed_ledger_from_spec
            _ledger = load_or_create_ledger(job_id, job_dir)
            if _ledger.entry_count == 0:
                _seeded = seed_ledger_from_spec(_ledger, spec, job_dir, profile=profile)
                emit(f"   📒 Decision ledger seeded with {_seeded} entries")
            else:
                emit(f"   📒 Decision ledger loaded ({_ledger.entry_count} entries)")
    except Exception as _lerr:
        logger.warning("[orchestrator] Ledger seed failed: %s", _lerr)

    # ------------------------------------------------------------------
    # STAGE 3: Scaffold Engine (deterministic, no LLM)
    # ------------------------------------------------------------------
    emit(f"\n{'='*60}")
    emit("🏗️ STAGE 3: SCAFFOLD ENGINE")
    emit(f"{'='*60}")

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
        return result

    from app.pipeline_v2.scaffold_engine import run_scaffold_engine

    scaffold = await run_scaffold_engine(
        manifest=manifest,
        spec=spec,
        job_dir=job_dir,
        on_progress=emit,
        profile=profile,
    )
    result.scaffold_result = scaffold

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

    # v2.2 Piece 5: stage summary for context threading
    try:
        from app.pipeline_v2.config import LEDGER_ENABLED
        if LEDGER_ENABLED:
            from app.pipeline_v2.stage_summaries import write_summary
            _body = (
                f"Produced {len(scaffold.files)} files "
                f"({len(create_files)} new, {len(modify_files)} modify). "
                f"Target: {profile.language}/{profile.framework}."
            )
            _handover_bits = []
            if create_files:
                _handover_bits.append(
                    f"New files needing implementation: {', '.join(create_files[:10])}"
                    + (" …" if len(create_files) > 10 else "")
                )
            if modify_files:
                _handover_bits.append(
                    f"Existing files to modify: {', '.join(modify_files[:10])}"
                    + (" …" if len(modify_files) > 10 else "")
                )
            write_summary(
                job_dir=job_dir,
                stage="scaffold_engine",
                body=_body,
                handover_notes=" ".join(_handover_bits),
                duration_s=scaffold.duration_seconds,
                status="passed" if scaffold.files else "failed",
            )
    except Exception as _serr:
        logger.debug("[orchestrator] Scaffold summary skipped: %s", _serr)

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
        profile=profile,
    )
    result.build_result = build_result
    result.total_llm_calls += build_result.total_llm_calls

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

    # v2.2 Piece 5: stage summary for context threading
    try:
        from app.pipeline_v2.config import LEDGER_ENABLED
        if LEDGER_ENABLED:
            from app.pipeline_v2.stage_summaries import write_summary
            _bwritten = build_result.all_files_written
            _bbody = (
                f"Builder wrote {len(_bwritten)} files in "
                f"{build_result.total_tool_calls} tool calls. "
                f"Input tokens: {build_result.total_input_tokens:,}, "
                f"output tokens: {build_result.total_output_tokens:,}."
            )
            _bhandover = ""
            if build_result.errors:
                _bhandover = f"Errors encountered: {'; '.join(build_result.errors[:3])}"
            elif _bwritten:
                _bhandover = (
                    f"Files written: {', '.join(_bwritten[:10])}"
                    + (" …" if len(_bwritten) > 10 else "")
                )
            _bstatus = "passed" if build_result.success else "failed"
            write_summary(
                job_dir=job_dir,
                stage="agentic_builder",
                body=_bbody,
                handover_notes=_bhandover,
                duration_s=build_result.total_duration_seconds,
                status=_bstatus,
            )
    except Exception as _berr:
        logger.debug("[orchestrator] Builder summary skipped: %s", _berr)

    _builder_messages = build_result.messages_history

    if not build_result.all_files_written:
        result.errors.append("Builder wrote no files")
        result.total_duration_seconds = time.time() - t_start
        emit("❌ Builder failed — no files written")
        return result

    # ------------------------------------------------------------------
    # VERIFICATION: profile-aware checkout
    # ------------------------------------------------------------------
    from app.pipeline_v2.config import ENHANCED_CHECKOUT, BVL_ENABLED

    _update_stage_status("final_checkout", "running", "Verification loop started")

    # For Android/Kotlin: BVL three-tier verification cascade
    if BVL_ENABLED and (profile.verification_mode == "emulator" or profile.language == "kotlin"):
        from app.pipeline_v2.bvl.bvl_orchestrator import run_bvl

        _update_stage_status("final_checkout", "running", "BVL verification started")

        bvl_report = await run_bvl(
            job_id=job_id,
            spec=spec,
            manifest=manifest,
            scaffold=scaffold,
            profile=profile,
            job_dir=job_dir,
            existing_messages=_builder_messages,
            emit=emit,
        )

        result.total_llm_calls += bvl_report.total_llm_calls
        result.bvl_report = bvl_report

        if bvl_report.all_signed_off:
            _update_stage_status("final_checkout", "passed", "BVL: all tiers passed")
            result.success = True
        else:
            blocked = bvl_report.blocked_components
            detail = blocked[0].block_reason[:200] if blocked else "BVL verification incomplete"
            _update_stage_status("final_checkout", "failed", detail)
            result.success = False

        # v1.3 (2026-04-18): Always-on spec reviewer.
        # Runs AFTER BVL regardless of outcome. The reviewer reads the spec
        # + every file the builder wrote + the BVL results and emits a
        # structured report of any unmet requirements, unwired hooks, or
        # real-bug patterns. Advisory only — does not fail the build.
        try:
            await _run_spec_review_stage(
                result=result,
                spec=spec,
                intent_text=intent_text,
                profile=profile,
                emit=emit,
                bvl_report=bvl_report,
                job_id=job_id,
                manifest=manifest,
            )
        except Exception as _sr_exc:
            logger.warning("[orchestrator] Spec review failed non-fatally: %s", _sr_exc)
            emit(f"   ⚠️ Spec review skipped: {_sr_exc}")

        result.total_duration_seconds = time.time() - t_start
        result.estimated_cost_usd = _estimate_cost(result) + bvl_report.estimated_cost_usd

        emit(f"\n{'='*60}")
        emit(f"{'✅ BUILD COMPLETE' if result.success else '❌ BUILD FINISHED WITH ISSUES'}")
        cost_str = f"${result.estimated_cost_usd:.2f}"
        emit(f"Duration: {result.total_duration_seconds:.1f}s | LLM calls: {result.total_llm_calls} | Est. cost: {cost_str}")
        emit(f"{'='*60}")

        try:
            _compile_and_send_to_debug(result, job_id, spec, emit, profile)
        except Exception as _e:
            logger.warning("[orchestrator] Report compilation failed: %s", _e)

        return result

    # ------------------------------------------------------------------
    # ENHANCED CHECKOUT (browser-based — ASTRA backend/frontend)
    # ------------------------------------------------------------------
    if ENHANCED_CHECKOUT:
        from app.pipeline_v2.checkout import run_enhanced_checkout
        checkout_report = await run_enhanced_checkout(spec=spec, emit=emit, profile=profile)
        result.total_llm_calls += 1

        if checkout_report.overall_passed:
            _update_stage_status("final_checkout", "passed", "Enhanced checkout passed")
            emit("   ✅ Verifier stage → PASSED")
        else:
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
                    profile=profile,
                )
                result.total_llm_calls += fix_result.total_llm_calls
                _builder_messages = fix_result.messages_history

                emit("   🔄 Re-running checkout after fix...")
                checkout_report_2 = await run_enhanced_checkout(spec=spec, emit=emit, profile=profile)
                result.total_llm_calls += 1

                if checkout_report_2.overall_passed:
                    _update_stage_status("final_checkout", "passed", "Enhanced checkout passed after fix")
                    emit("   ✅ Verifier stage → PASSED (after fix)")
                else:
                    _update_stage_status("final_checkout", "failed", "Enhanced checkout failed after fix attempt")
                    emit("   ❌ Verifier stage → FAILED (after fix attempt)")
            else:
                _update_stage_status("final_checkout", "failed", "Enhanced checkout failed")

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

        emit("   🔄 Booting application...")
        from app.pipeline_v2.sandbox_tools import boot_check
        boot_ok, boot_output = await boot_check(profile=profile)

        if not boot_ok:
            emit(f"   ❌ Boot failed: {boot_output[:200]}")

            if attempt < MAX_VERIFY_LOOPS:
                emit("   🔧 Sending boot errors to Builder for fix...")
                _fix_prompt = (
                    f"The application failed to boot. Here are the errors:\n\n"
                    f"```\n{boot_output[:4000]}\n```\n\n"
                    f"Read the relevant files, diagnose the issue, and fix it."
                )
                fix_result = await run_agentic_builder(
                    spec=spec,
                    manifest=manifest,
                    scaffold=scaffold,
                    job_dir=job_dir,
                    handover_context=_fix_prompt,
                    on_progress=emit,
                    existing_messages=_builder_messages,
                    profile=profile,
                )
                result.total_llm_calls += fix_result.total_llm_calls
                _builder_messages = fix_result.messages_history
                if fix_result.all_files_written:
                    emit(f"   🔧 Builder fixed {len(fix_result.all_files_written)} file(s)")
                continue
        else:
            emit("   ✅ Boot OK")

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
            break

        if attempt >= MAX_VERIFY_LOOPS:
            emit(f"   ⚠️ Verification failed after {MAX_VERIFY_LOOPS} attempts")
            _update_stage_status("final_checkout", "failed", f"Failed after {MAX_VERIFY_LOOPS} attempts")
            break

        emit(f"   🔧 Sending verification feedback to Builder for fix...")
        _fix_prompt = (
            f"Visual verification FAILED on attempt {attempt}. "
            f"The verifier reported these issues:\n\n"
            f"{verify_result.feedback[:4000]}\n\n"
            f"Read the relevant files, diagnose what's wrong, and fix it. "
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
            profile=profile,
        )
        result.total_llm_calls += fix_result.total_llm_calls
        _builder_messages = fix_result.messages_history
        if fix_result.all_files_written:
            emit(f"   🔧 Builder fixed {len(fix_result.all_files_written)} file(s)")
        emit(f"   Feedback: {verify_result.feedback[:200]}")

    # Final result
    final_verify = result.verify_results[-1] if result.verify_results else None
    result.success = final_verify.passed if final_verify else False
    result.total_duration_seconds = time.time() - t_start
    result.estimated_cost_usd = _estimate_cost(result)

    try:
        _compile_and_send_to_debug(result, job_id, spec, emit, profile)
    except Exception as _report_err:
        logger.warning("[orchestrator] Report compilation failed: %s", _report_err)

    status = "COMPLETE ✅" if result.success else "FINISHED WITH ISSUES ⚠️"
    emit(f"\n{'='*60}")
    emit(f"🚀 ASTRA v2.2 Pipeline {status}")
    emit(f"   Files: {len(build_result.all_files_written)}")
    emit(f"   Tool calls: {build_result.total_tool_calls}")
    emit(f"   LLM calls: {result.total_llm_calls}")
    emit(f"   Duration: {result.total_duration_seconds:.1f}s")
    emit(f"   Est. cost: ${result.estimated_cost_usd:.2f}")
    if result.errors:
        emit(f"   Errors: {result.errors}")
    _emit_ledger_summary(job_dir, emit)
    emit(f"{'='*60}")

    return result
