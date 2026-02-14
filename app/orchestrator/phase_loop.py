# FILE: app/orchestrator/phase_loop.py
"""
Phase Loop — Multi-Phase Execution Orchestrator.

Iterates through phases in a ConstructionPlan, running the segment loop
for each phase and gating with Phase Checkout before advancing.

For single-phase plans, this is a thin pass-through to run_segmented_job.
For multi-phase plans, it:
1. Runs each phase's segments via segment_loop
2. Runs Phase Checkout after each phase
3. Only advances if checkout passes
4. Logs routing info if checkout fails (auto-retry not yet implemented)

v1.0 (2026-02-14): Initial implementation.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Optional

from .construction_planner_models import ConstructionPlan, PhaseDefinition
from .segment_state import JobState

logger = logging.getLogger(__name__)

PHASE_LOOP_BUILD_ID = "2026-02-14-v1.0-initial"
print(f"[PHASE_LOOP_LOADED] BUILD_ID={PHASE_LOOP_BUILD_ID}")

ProgressCallback = Optional[Callable[[str], None]]


async def run_construction_plan(
    plan: ConstructionPlan,
    parent_spec: dict,
    db: Any = None,
    project_id: int = 0,
    on_progress: ProgressCallback = None,
) -> ConstructionPlan:
    """
    Execute a construction plan phase by phase.

    For single-phase: delegates directly to run_segmented_job.
    For multi-phase: iterates phases with checkout gating.

    Returns the updated ConstructionPlan with execution state.
    """
    _emit = on_progress or (lambda msg: None)

    if not plan.is_multi_phase:
        # Single-phase: run segment loop directly (existing flow)
        return await _run_single_phase(plan, parent_spec, db, project_id, _emit)

    # Multi-phase execution
    from .construction_skeleton import generate_phase_interface, format_phase_contract_markdown

    _emit(f"\n{'='*60}")
    _emit(f"🏗️ MULTI-PHASE BUILD: {plan.total_phases} phases, {plan.total_files} files")
    _emit(f"   {plan.reasoning}")

    plan.status = "running"

    for phase in plan.phases:
        # Check dependencies
        if not _dependencies_met(phase, plan):
            _emit(f"\n⏸️ Phase {phase.phase_number} '{phase.title}' blocked — "
                  f"dependency not complete")
            phase.status = "blocked"
            continue

        _emit(f"\n{'='*50}")
        _emit(f"📦 Phase {phase.phase_number}/{plan.total_phases}: {phase.title}")
        _emit(f"   {len(phase.file_scope)} files, ~{phase.estimated_segments} segments")

        # Generate phase interface contract (upstream deliverables)
        if phase.phase_number > 1:
            phase_contract = generate_phase_interface(plan, phase)
            if phase_contract.upstream_deliverables:
                contract_md = format_phase_contract_markdown(phase_contract)
                _emit(f"   📋 Phase interface: {len(phase_contract.upstream_deliverables)} "
                      f"upstream files available")
                if phase_contract.missing_deliverables:
                    _emit(f"   ⚠️ {len(phase_contract.missing_deliverables)} missing!")
                # Store contract markdown in parent_spec for injection into prompts
                parent_spec["_phase_contract_markdown"] = contract_md

        plan.current_phase = phase.phase_number
        phase.status = "running"

        # Save progress
        _save_plan_state(plan)

        try:
            job_state = await _run_phase(
                phase, plan, parent_spec, db, project_id, _emit,
            )

            if job_state and job_state.overall_status == "complete":
                # Check Phase Checkout result
                checkout_status = _get_checkout_status(job_state)
                phase.checkout_status = checkout_status

                if checkout_status == "pass":
                    phase.status = "complete"
                    _emit(f"✅ Phase {phase.phase_number} complete + checkout passed")
                else:
                    phase.status = "failed"
                    _emit(f"❌ Phase {phase.phase_number} checkout {checkout_status}")
                    # Log routing but don't auto-retry yet
                    _log_phase_failure(phase, job_state, _emit)
                    break  # Stop phase progression on failure
            else:
                status = job_state.overall_status if job_state else "error"
                phase.status = "failed"
                _emit(f"❌ Phase {phase.phase_number} failed: {status}")
                break

        except Exception as exc:
            logger.exception("[phase_loop] Phase %d error: %s", phase.phase_number, exc)
            phase.status = "failed"
            _emit(f"❌ Phase {phase.phase_number} error: {exc}")
            break

    # Aggregate plan status
    if plan.all_complete:
        plan.status = "complete"
        _emit(f"\n🎉 ALL {plan.total_phases} PHASES COMPLETE")

        # Run Final Project Checkout (Stage 10)
        try:
            from .final_checkout import run_final_checkout
            from .segment_loop import get_job_dir as _get_plan_job_dir
            all_files = []
            for p in plan.phases:
                all_files.extend(p.file_scope)
            _final = run_final_checkout(
                job_id=plan.job_id,
                plan=plan,
                original_file_scope=all_files,
                job_dir=_get_plan_job_dir(plan.job_id),
                emit=_emit,
            )
            if not _final.status == "pass":
                plan.status = "failed"
                logger.warning("[phase_loop] Final checkout FAILED for %s", plan.job_id)
        except Exception as _fc_err:
            logger.warning("[phase_loop] Final checkout error: %s", _fc_err)
            _emit(f"⚠️ Final checkout could not run: {_fc_err}")

    elif any(p.status == "complete" for p in plan.phases):
        plan.status = "partial"
    elif any(p.status == "failed" for p in plan.phases):
        plan.status = "failed"

    _save_plan_state(plan)
    return plan


async def _run_single_phase(
    plan: ConstructionPlan,
    parent_spec: dict,
    db: Any,
    project_id: int,
    _emit: Callable,
) -> ConstructionPlan:
    """Run a single-phase plan via existing segment loop."""
    from .segment_loop import run_segmented_job

    phase = plan.phases[0]
    plan.status = "running"
    plan.current_phase = 1
    phase.status = "running"

    if not phase.manifest_path:
        # Single-phase plans use the job's existing manifest
        from .segment_loop import get_job_dir
        job_dir = get_job_dir(plan.job_id)
        manifest_path = os.path.join(job_dir, "segments", "manifest.json")
        phase.manifest_path = manifest_path

    job_state = await run_segmented_job(
        job_id=plan.job_id,
        manifest_path=phase.manifest_path,
        parent_spec=parent_spec,
        db=db,
        project_id=project_id,
        on_progress=_emit,
    )

    phase.status = job_state.overall_status if job_state else "failed"
    phase.checkout_status = _get_checkout_status(job_state) if job_state else None
    plan.status = phase.status

    return plan


async def _run_phase(
    phase: PhaseDefinition,
    plan: ConstructionPlan,
    parent_spec: dict,
    db: Any,
    project_id: int,
    _emit: Callable,
) -> Optional[JobState]:
    """Run a single phase through the segment loop."""
    from .segment_loop import run_segmented_job, get_job_dir

    if not phase.manifest_path:
        # Multi-phase: each phase needs its own manifest
        # For now, this requires spec_runner to have generated it
        job_dir = get_job_dir(plan.job_id)
        manifest_path = os.path.join(
            job_dir, "segments", f"phase_{phase.phase_number}_manifest.json"
        )
        if not os.path.isfile(manifest_path):
            # Fall back to main manifest (single manifest for now)
            manifest_path = os.path.join(job_dir, "segments", "manifest.json")
        phase.manifest_path = manifest_path

    return await run_segmented_job(
        job_id=plan.job_id,
        manifest_path=phase.manifest_path,
        parent_spec=parent_spec,
        db=db,
        project_id=project_id,
        on_progress=_emit,
    )


def _dependencies_met(phase: PhaseDefinition, plan: ConstructionPlan) -> bool:
    """Check if all phase dependencies are complete."""
    for dep_id in phase.depends_on:
        dep = next((p for p in plan.phases if p.phase_id == dep_id), None)
        if not dep or dep.status != "complete":
            return False
    return True


def _get_checkout_status(job_state: JobState) -> str:
    """Extract Phase Checkout status from JobState."""
    if not job_state.integration_check:
        return job_state.phase_checkout_boot or "unknown"
    checkout = job_state.integration_check.get("phase_checkout", {})
    return checkout.get("status", job_state.phase_checkout_boot or "unknown")


def _log_phase_failure(
    phase: PhaseDefinition, job_state: JobState, _emit: Callable,
) -> None:
    """Log routing info from a failed Phase Checkout."""
    if not job_state.integration_check:
        return
    checkout = job_state.integration_check.get("phase_checkout", {})
    routing = checkout.get("routing")
    if routing:
        _emit(f"  Route to: {routing.get('target_stage', '?')}")
        if routing.get("target_segment"):
            _emit(f"  Segment: {routing['target_segment']}")
        _emit(f"  Reason: {routing.get('reason', '?')}")


def _save_plan_state(plan: ConstructionPlan) -> None:
    """Save updated plan state to disk."""
    from .segment_loop import get_job_dir
    try:
        job_dir = get_job_dir(plan.job_id)
        path = os.path.join(job_dir, "construction_plan.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(plan.to_dict(), f, indent=2, ensure_ascii=False)
    except Exception as exc:
        logger.warning("[phase_loop] Failed to save plan state: %s", exc)
