# FILE: app/orchestrator/segment_loop.py
"""
Core orchestrator segment loop.

Reads a segment manifest, processes segments in dependency order through
the existing pipeline (Critical Pipeline → Critique → Overwatcher →
Implementer), threads evidence forward between segments, and tracks
state for crash recovery.

Phase 2 of Pipeline Segmentation.

v1.0 (2026-02-08): Initial implementation
v7.0 (2026-02-23): Decomposed into sub-modules for maintainability.
    - seg_pipeline_step0..step3: Per-segment pipeline stages
    - seg_job_init: Job initialisation stages
    - seg_job_post: Post-execution stages
    - seg_job_cohesion: Cohesion check loop
    - seg_job_loop: Main segment processing loop
    - segment_pipeline_ctx: Shared context dataclasses
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from app.orchestrator._segment_loop_utils_6 import (
    SEGMENT_LOOP_BUILD_ID,
    _find_latest_arch,
    _load_source_file_evidence,
    _now_iso,
    collect_segment_outputs,
    is_segment_blocked,
)
from app.orchestrator._segment_loop_utils_7 import (
    _is_facade_segment,
    _save_execution_trace,
    build_evidence_bundle,
    build_segment_context,
    can_execute_segment,
    mark_dependents_blocked,
    unblock_recovered_segments,
    verify_contracts_fulfilled,
)
from app.orchestrator._segment_loop_utils_8 import update_segment_status

logger = logging.getLogger(__name__)
print(f"[SEGMENT_LOOP_LOADED] BUILD_ID={SEGMENT_LOOP_BUILD_ID}")


# --- Internal imports ---
from app.pot_spec.grounded.segment_schemas import (
    SegmentManifest,
    SegmentSpec,
    SegmentStatus,
    InterfaceContract,
)
from app.orchestrator.segment_state import (
    JobState,
    SegmentState,
    load_or_init_state,
    save_state,
    get_job_dir,
)

# --- Pipeline stage imports (optional — graceful degradation) ---
try:
    from app.llm.critical_pipeline_stream import generate_critical_pipeline_stream
    _CRITICAL_PIPELINE_AVAILABLE = True
except ImportError:
    _CRITICAL_PIPELINE_AVAILABLE = False

try:
    from app.overwatcher.overwatcher import run_overwatcher, run_pot_spec_execution
    _OVERWATCHER_AVAILABLE = True
except ImportError:
    _OVERWATCHER_AVAILABLE = False

try:
    from app.overwatcher.implementer import run_implementer
    _IMPLEMENTER_AVAILABLE = True
except ImportError:
    _IMPLEMENTER_AVAILABLE = False

try:
    from app.overwatcher.architecture_executor import run_architecture_execution
    from app.overwatcher.spec_resolution import resolve_latest_spec, ResolvedSpec
    from app.llm.overwatcher_stream import create_overwatcher_llm_fn
    _ARCH_EXECUTOR_AVAILABLE = True
except ImportError as _ae:
    _ARCH_EXECUTOR_AVAILABLE = False
    logger.warning("[SEGMENT_LOOP] Architecture executor not available: %s", _ae)
    print(f"[SEGMENT_LOOP] [WARNING] Architecture executor import failed: {_ae}")

try:
    from app.orchestrator.interface_reconciliation import (
        read_dependency_interfaces_from_sandbox,
        inject_reconciliation_into_architecture,
    )
    _RECONCILIATION_AVAILABLE = True
except ImportError:
    _RECONCILIATION_AVAILABLE = False
    logger.debug("[SEGMENT_LOOP] Interface reconciliation not available")


# Type alias for progress callback
ProgressCallback = Optional[Callable[[str], None]]


# =============================================================================
# PER-SEGMENT PIPELINE (decomposed into step modules)
# =============================================================================


async def run_segment_through_pipeline(
    segment: SegmentSpec,
    segment_context: Dict[str, Any],
    job_id: str,
    db: Any,
    project_id: int,
    on_progress: ProgressCallback = None,
    contract_set: Any = None,
    job_dir_path: str = "",
    manifest: Any = None,
    parent_spec: Any = None,
    quarantine_result: Any = None,
) -> Dict[str, Any]:
    """
    Run a single segment through: Critical Pipeline → Critique → Overwatcher → Implementer.

    v7.0: Decomposed — each step is now a separate function in seg_pipeline_step*.py.
    """
    from app.orchestrator.seg_pipeline_step0 import load_failure_feedback
    from app.orchestrator.seg_pipeline_step1 import (
        generate_architecture, sanitise_architecture,
        save_architecture, show_file_inventory,
    )
    from app.orchestrator.seg_pipeline_step1b import validate_imports_and_regen
    from app.orchestrator.seg_pipeline_step2 import check_approval_gate
    from app.orchestrator.seg_pipeline_step3 import (
        run_preflight, execute_architecture,
    )

    result = {
        "success": False,
        "output_files": [],
        "error": None,
        "critique_warnings": [],
    }

    seg_id = segment.segment_id
    emit = on_progress or (lambda msg: None)
    seg_job_id = f"{job_id}__{seg_id}"
    is_deterministic = segment_context.get("segment_spec", {}).get(
        "deterministic_refactor", False
    )

    # --- Step 0.5: Load previous failure feedback ---
    load_failure_feedback(seg_id, job_id, segment_context, emit)

    # --- Step 1: Generate architecture ---
    arch_result = await generate_architecture(
        seg_id, seg_job_id, segment_context, project_id, db,
        is_deterministic, job_id, emit,
    )

    # Deterministic fallback: if pre-generated arch missing, retry as LLM
    if arch_result is None and is_deterministic:
        is_deterministic = False
        arch_result = await generate_architecture(
            seg_id, seg_job_id, segment_context, project_id, db,
            False, job_id, emit,
        )

    if arch_result is None or "error" in arch_result:
        result["error"] = (arch_result or {}).get("error", "Architecture generation failed")
        return result

    arch_text = arch_result["arch_text"]

    # --- Step 1: Sanitise ---
    file_scope = segment_context.get("file_scope", segment.file_scope)
    arch_text = sanitise_architecture(arch_text, seg_id, file_scope, emit)

    # --- Step 1: Save + show inventory ---
    seg_arch_path = save_architecture(arch_text, seg_id, job_id, emit)
    show_file_inventory(arch_text, emit)

    # --- Step 1a: Record architectural decisions to ledger ---
    _seg_ledger = segment_context.get("_ledger")
    if _seg_ledger:
        try:
            from app.orchestrator.ledger_arch_extractor import extract_and_record_decisions
            extract_and_record_decisions(
                arch_text=arch_text, seg_id=seg_id,
                ledger=_seg_ledger, job_dir=job_dir_path, emit=emit,
            )
        except Exception as _led_err:
            logger.debug("[SEGMENT_LOOP] Ledger decision extraction failed (non-fatal): %s", _led_err)

    # --- Step 1b: Import validation ---
    arch_text = await validate_imports_and_regen(
        arch_text, seg_id, seg_job_id, job_id, segment_context,
        project_id, db, is_deterministic, seg_arch_path, emit,
    )

    # --- Step 2: Human approval gate ---
    gate_result = check_approval_gate(seg_id, job_id, segment_context, seg_arch_path, emit)
    if gate_result is not None:
        return gate_result

    # --- Step 3a: Overwatcher pre-flight ---
    emit(f"  🔧 Running Overwatcher for {seg_id}...")

    if not _ARCH_EXECUTOR_AVAILABLE:
        emit(f"  ⚠️ Architecture executor not available — architecture generated only")
        result["success"] = True
        return result

    preflight_result = await run_preflight(
        seg_id, arch_text, segment_context, contract_set, manifest,
        parent_spec, job_id, job_dir_path, seg_arch_path, emit,
    )
    if preflight_result is not None:
        return preflight_result

    # --- Step 3b: Architecture execution ---
    return await execute_architecture(
        seg_id, seg_job_id, arch_text, seg_arch_path, segment_context,
        segment, manifest, job_id, job_dir_path, project_id, db,
        contract_set, quarantine_result, emit,
    )


# =============================================================================
# MAIN JOB ORCHESTRATOR (decomposed into init/loop/post modules)
# =============================================================================


async def run_segmented_job(
    job_id: str,
    manifest_path: str,
    parent_spec: dict,
    db: Any = None,
    project_id: int = 0,
    on_progress: ProgressCallback = None,
    implement_only: bool = False,
) -> JobState:
    """
    Main entry point for segmented execution.

    v7.0: Decomposed — stages are now in seg_job_init.py, seg_job_loop.py,
    seg_job_cohesion.py, and seg_job_post.py.
    """
    from app.orchestrator.segment_pipeline_ctx import JobCtx
    from app.orchestrator.seg_job_init import (
        load_manifest,
        try_single_segment_fast_path,
        init_skeleton_contracts,
        init_source_evidence,
        init_enrichment,
        augment_skeleton,
        init_evidence_ledger,
        init_quarantine_and_state,
    )
    from app.orchestrator.seg_job_loop import (
        compute_execution_order,
        process_segments,
    )
    from app.orchestrator.seg_job_cohesion import run_cohesion_loop
    from app.orchestrator.seg_job_post import (
        run_post_execution_reconciliation,
        run_deferred_consumer_reconciliation,
        run_integration_check,
        run_deferred_quarantine,
        run_phase_checkout,
        run_final_checkout,
        distill_journal,
        emit_quarantine_status,
        emit_final_summary,
        compact_evidence_ledger,
    )

    emit = on_progress or (lambda msg: None)

    # --- Build shared context ---
    ctx = JobCtx(
        job_id=job_id,
        manifest_path=manifest_path,
        parent_spec=parent_spec,
        db=db,
        project_id=project_id,
        on_progress=on_progress,
        implement_only=implement_only,
    )

    # v5.16: Set journal context
    try:
        from app.experience.context import set_job_context
        set_job_context(job_id=job_id, job_dir=ctx.job_dir_path, job_type="segmented")
    except Exception:
        pass

    # --- Phase 1: Load manifest ---
    if not load_manifest(ctx):
        state = JobState(job_id=job_id, overall_status="failed")
        return state

    # --- Phase 1C: Single-segment fast path ---
    fast_result = await try_single_segment_fast_path(ctx)
    if fast_result is not None:
        return fast_result

    # --- Phase 2: Multi-segment initialisation ---
    init_skeleton_contracts(ctx)
    init_source_evidence(ctx)
    await init_enrichment(ctx)
    augment_skeleton(ctx)
    init_evidence_ledger(ctx)
    init_quarantine_and_state(ctx)

    # --- Phase 3: Process segments ---
    compute_execution_order(ctx)
    await process_segments(ctx)

    # --- Phase 4: Post-execution reconciliation ---
    run_post_execution_reconciliation(ctx)
    run_deferred_consumer_reconciliation(ctx)

    # --- Phase 5: Cohesion check loop ---
    await run_cohesion_loop(ctx)

    # --- Phase 6: Integration + quarantine + checkout ---
    run_integration_check(ctx)
    run_deferred_quarantine(ctx)
    await run_phase_checkout(ctx)
    await run_final_checkout(ctx)

    # --- Phase 7: Finalise ---
    distill_journal(ctx)
    emit_quarantine_status(ctx)
    compact_evidence_ledger(ctx)
    emit_final_summary(ctx)

    return ctx.state
