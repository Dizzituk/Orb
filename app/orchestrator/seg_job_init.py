# FILE: app/orchestrator/seg_job_init.py
"""
Initialisation stages for run_segmented_job:
- Load manifest
- Single-segment fast path
- Skeleton contracts
- Enrichment
- Evidence ledger
- Quarantine
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from app.orchestrator.segment_pipeline_ctx import JobCtx

logger = logging.getLogger(__name__)


def load_manifest(ctx: JobCtx) -> bool:
    """Load manifest from disk. Returns True on success, False on failure."""
    logger.info("[SEGMENT_LOOP] Starting segmented execution for job %s", ctx.job_id)
    ctx.emit(f"📋 Loading manifest from {ctx.manifest_path}...")

    try:
        from app.pot_spec.grounded.segment_schemas import SegmentManifest
        with open(ctx.manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        ctx.manifest = SegmentManifest.from_dict(manifest_data)
        ctx.emit(f"📋 Manifest loaded: {ctx.manifest.total_segments} segment(s)")
        return True
    except Exception as e:
        logger.error("[SEGMENT_LOOP] Failed to load manifest: %s", e)
        ctx.emit(f"❌ Failed to load manifest: {e}")
        return False


async def try_single_segment_fast_path(ctx: JobCtx) -> Optional[Any]:
    """
    v5.4: When manifest has exactly 1 segment, skip all multi-segment ceremony.
    Returns JobState if fast path was taken, None otherwise.
    """
    if ctx.manifest.total_segments != 1:
        return None

    from app.orchestrator.segment_state import JobState, SegmentState
    from app.pot_spec.grounded.segment_schemas import SegmentStatus

    seg_spec = ctx.manifest.segments[0]
    seg_id = seg_spec.segment_id
    ctx.emit(f"⚡ Single-segment fast path: {seg_id}")
    ctx.emit(
        f"  Files: {', '.join(seg_spec.file_scope[:5])}"
        f"{'...' if len(seg_spec.file_scope) > 5 else ''}"
    )

    segment_context = {
        "segment_id": seg_id,
        "segment_spec": seg_spec.to_dict(),
        "parent_spec": ctx.parent_spec,
        "file_scope": seg_spec.file_scope,
        "evidence": [],
        "exposes": None,
        "consumes": None,
        "requirements": seg_spec.requirements,
        "acceptance_criteria": seg_spec.acceptance_criteria,
        "dependencies": [],
    }

    from app.orchestrator.segment_loop import run_segment_through_pipeline
    try:
        pipeline_result = await run_segment_through_pipeline(
            segment=seg_spec,
            segment_context=segment_context,
            job_id=ctx.job_id,
            db=ctx.db,
            project_id=ctx.project_id,
            on_progress=ctx.on_progress,
        )
    except Exception as e:
        pipeline_result = {
            "success": False, "output_files": [], "error": str(e),
            "critique_warnings": [],
        }
        logger.exception("[SEGMENT_LOOP] Single-segment error: %s", e)

    state = JobState(job_id=ctx.job_id)
    state.segments[seg_id] = SegmentState(
        segment_id=seg_id,
        status=(
            SegmentStatus.COMPLETE.value if pipeline_result["success"]
            else SegmentStatus.FAILED.value
        ),
        output_files=pipeline_result.get("output_files", []),
        error=pipeline_result.get("error"),
    )
    state.overall_status = "complete" if pipeline_result["success"] else "failed"

    output_count = len(pipeline_result.get("output_files", []))
    if pipeline_result["success"]:
        ctx.emit(f"\n✅ Pipeline complete ({output_count} file(s) written)")
    else:
        ctx.emit(f"\n❌ Pipeline failed: {pipeline_result.get('error', 'Unknown')}")

    logger.info(
        "[SEGMENT_LOOP] v5.4 Single-segment fast path %s: %s",
        state.overall_status, ctx.job_id,
    )
    return state


def init_skeleton_contracts(ctx: JobCtx) -> None:
    """v5.6: Generate or load skeleton contracts (deterministic, zero LLM)."""
    try:
        from app.orchestrator.skeleton_contracts import (
            generate_skeleton_contract, save_skeleton_contract,
            load_skeleton_contract,
        )
    except ImportError:
        logger.debug("[SEGMENT_LOOP] Skeleton contracts not available")
        return

    ctx.contract_set = load_skeleton_contract(ctx.job_dir_path)
    if ctx.contract_set and ctx.contract_set.skeletons:
        ctx.emit(
            f"🦴 Loaded existing skeleton contract: "
            f"{ctx.contract_set.total_segments} segment(s), "
            f"{len(ctx.contract_set.cross_segment_bindings)} binding(s)"
        )
        return

    ctx.emit("🦴 Generating skeleton contracts (deterministic)...")
    try:
        ctx.contract_set = generate_skeleton_contract(
            manifest_dict=ctx.manifest.to_dict(),
            job_id=ctx.job_id,
        )
        if ctx.contract_set.skeletons:
            save_skeleton_contract(ctx.contract_set, ctx.job_dir_path)
            total_exports = sum(len(s.exports) for s in ctx.contract_set.skeletons)
            ctx.emit(
                f"🦴 Skeleton: {ctx.contract_set.total_segments} segments, "
                f"{total_exports} exports, "
                f"{len(ctx.contract_set.cross_segment_bindings)} cross-segment bindings"
            )
            for binding in ctx.contract_set.cross_segment_bindings:
                ctx.emit(
                    f"  🔗 {binding['from_segment']} → {binding['to_segment']}: "
                    f"`{binding['file_path']}` ({binding['binding_type']})"
                )
        else:
            ctx.emit("ℹ️ No cross-segment bindings detected (segments may be independent)")
    except Exception as skel_err:
        logger.warning(
            "[SEGMENT_LOOP] Skeleton generation failed (non-fatal): %s", skel_err
        )
        ctx.emit(f"⚠️ Skeleton generation failed (non-fatal): {skel_err}")
        ctx.contract_set = None


def init_source_evidence(ctx: JobCtx) -> None:
    """v2.2: Pre-load source file evidence for refactor jobs."""
    from app.orchestrator._segment_loop_utils_6 import _load_source_file_evidence
    ctx.source_evidence = _load_source_file_evidence(ctx.manifest) or {}


async def init_enrichment(ctx: JobCtx) -> None:
    """v5.17 Stage 4B: Segment enrichment."""
    if not ctx.source_evidence or ctx.manifest.total_segments <= 1:
        return

    try:
        from app.orchestrator.segment_enrichment import enrich_segments
    except ImportError:
        return

    ctx.emit("🔬 Running segment enrichment (Stage 4B)...")
    try:
        ctx.enrichment_data = await enrich_segments(
            manifest=ctx.manifest,
            source_evidence=ctx.source_evidence,
            job_dir_path=ctx.job_dir_path,
            db=ctx.db,
            project_id=ctx.project_id,
        )
        if ctx.enrichment_data:
            n_enriched = len(ctx.enrichment_data)
            total_symbols = sum(
                e.get("extraction_stats", {}).get("constants", 0)
                + e.get("extraction_stats", {}).get("functions", 0)
                + e.get("extraction_stats", {}).get("classes", 0)
                for e in ctx.enrichment_data.values()
            )
            n_unresolved = sum(
                len(e.get("unresolved", []))
                for e in ctx.enrichment_data.values()
            )
            ctx.emit(
                f"🔬 Segment enrichment complete: {n_enriched} segment(s), "
                f"{total_symbols} symbol(s) extracted"
            )
            if n_unresolved:
                ctx.emit(f"  ⚠️ {n_unresolved} unresolved symbol(s) detected")
            for seg_id, seg_enrich in ctx.enrichment_data.items():
                stats = seg_enrich.get("extraction_stats", {})
                risk = seg_enrich.get("risk_level", "low")
                order = seg_enrich.get("implementation_order", 0)
                risk_icon = "🔴" if risk == "high" else "🟡" if risk == "medium" else "🟢"
                ctx.emit(
                    f"  {risk_icon} {seg_id}: "
                    f"{stats.get('constants', 0)}C/{stats.get('functions', 0)}F/"
                    f"{stats.get('classes', 0)}Cl "
                    f"risk={risk} order={order}"
                )
        else:
            ctx.emit("🔬 Segment enrichment: no data produced (pipeline continues as before)")
    except Exception as enrich_err:
        logger.warning(
            "[SEGMENT_LOOP] Segment enrichment failed (non-fatal): %s", enrich_err
        )
        ctx.emit(f"⚠️ Segment enrichment failed (non-fatal): {enrich_err}")
        ctx.enrichment_data = {}


def augment_skeleton(ctx: JobCtx) -> None:
    """v5.21: Post-enrichment skeleton augmentation."""
    if not ctx.enrichment_data or not ctx.contract_set:
        return
    try:
        from app.orchestrator.skeleton_contracts import augment_skeleton_with_enrichment
        augmented = augment_skeleton_with_enrichment(
            contract_set=ctx.contract_set,
            enrichment_data=ctx.enrichment_data,
            job_dir=ctx.job_dir_path,
        )
        if augmented:
            ctx.emit(
                f"🦴 Skeleton augmented: {augmented} export binding(s) now have named symbols"
            )
    except Exception as aug_err:
        logger.warning(
            "[SEGMENT_LOOP] v5.21 Skeleton augmentation failed (non-fatal): %s", aug_err
        )
        ctx.emit(f"⚠️ Skeleton augmentation failed (non-fatal): {aug_err}")


def init_evidence_ledger(ctx: JobCtx) -> None:
    """v2.2: Evidence Ledger — create/load and seed with source files."""
    try:
        from app.orchestrator.evidence_ledger import (
            create_ledger, load_ledger, seed_ledger_with_source_files,
        )
        ctx.ledger = load_ledger(ctx.job_dir_path)
        if ctx.ledger is None:
            ctx.ledger = create_ledger(ctx.job_id, ctx.job_dir_path)
            if ctx.source_evidence:
                seed_ledger_with_source_files(
                    ctx.ledger, ctx.job_dir_path, ctx.source_evidence,
                )
        else:
            ctx.emit(f"📚 Evidence ledger loaded: {ctx.ledger.entry_count} entries")
    except Exception as ledger_err:
        logger.warning(
            "[SEGMENT_LOOP] Evidence ledger init failed (non-fatal): %s", ledger_err
        )
        ctx.ledger = None


def init_quarantine_and_state(ctx: JobCtx) -> None:
    """
    v5.7/v5.15/v6.1: Pre-execution quarantine + state init + auto-recovery.
    """
    from app.orchestrator.segment_state import (
        load_or_init_state, save_state,
    )
    from app.pot_spec.grounded.segment_schemas import SegmentStatus

    # Init state BEFORE quarantine (v5.19)
    ctx.state = load_or_init_state(ctx.job_id, ctx.manifest)
    ctx.emit(f"📊 State: {ctx.state.summary()}")

    # v6.1 FIX 9 + FIX 13: Deterministic refactor → quarantine immediately
    if ctx.manifest.deterministic_sources:
        logger.info(
            "[SEGMENT_LOOP] v6.1 Deterministic job — quarantine before execution: %s",
            ctx.manifest.deterministic_sources,
        )
        _run_quarantine(ctx)
    elif not ctx.implement_only:
        logger.debug(
            "[SEGMENT_LOOP] v5.15 Skipping quarantine (run segments mode)"
        )
    else:
        # v5.22: Auto-recover FAILED/BLOCKED segments on retry
        _auto_recover_segments(ctx)


def _run_quarantine(ctx: JobCtx) -> None:
    """Execute package quarantine."""
    try:
        from app.orchestrator.package_quarantine import run_quarantine
        from app.overwatcher.sandbox_client import get_sandbox_client

        client = get_sandbox_client()
        sandbox_base = os.getenv("ORB_SANDBOX_BASE", "D:\\Orb")

        ctx.quarantine_result = run_quarantine(
            manifest_dict=ctx.manifest.to_dict(),
            sandbox_base=sandbox_base,
            client=client,
            on_progress=ctx.emit,
        )
        if ctx.quarantine_result.has_quarantined:
            logger.info(
                "[SEGMENT_LOOP] v6.1 Quarantine complete: %d file(s), %d dir(s)",
                len([e for e in ctx.quarantine_result.entries if e.status == 'quarantined']),
                len(ctx.quarantine_result.directories_created),
            )
            ctx.emit("📦 v6.1 Quarantine: monolith moved before execution")
        else:
            logger.info("[SEGMENT_LOOP] v6.1 Quarantine ran but nothing to move")
    except ImportError:
        logger.debug("[SEGMENT_LOOP] Package quarantine not available")
    except Exception as q_err:
        logger.warning("[SEGMENT_LOOP] v6.1 Quarantine failed (non-fatal): %s", q_err)
        ctx.emit(f"⚠️ Quarantine failed (non-fatal): {q_err}")


def _auto_recover_segments(ctx: JobCtx) -> None:
    """v5.22: Auto-recover FAILED/BLOCKED segments on retry."""
    from app.orchestrator.segment_state import save_state
    from app.pot_spec.grounded.segment_schemas import SegmentStatus

    failed_or_blocked = [
        (sid, s) for sid, s in ctx.state.segments.items()
        if s.status in (SegmentStatus.FAILED.value, SegmentStatus.BLOCKED.value)
    ]
    if not failed_or_blocked:
        return

    recovered = []
    for fb_sid, fb_state in failed_or_blocked:
        fb_arch_dir = os.path.join(ctx.job_dir_path, "segments", fb_sid, "arch")
        has_arch = (
            os.path.isdir(fb_arch_dir)
            and any(f.endswith(".md") for f in os.listdir(fb_arch_dir))
        )
        if has_arch:
            fb_state.status = SegmentStatus.APPROVED.value
            fb_state.error = None
            fb_state.started_at = None
            fb_state.completed_at = None
            recovered.append(fb_sid)
            logger.info(
                "[SEGMENT_LOOP] v5.22 Auto-recovered %s: FAILED/BLOCKED -> APPROVED (retry)",
                fb_sid,
            )

    if recovered:
        save_state(ctx.state, ctx.job_dir_path)
        ctx.emit(
            f"🔄 Auto-recovered {len(recovered)} segment(s) for retry: "
            f"{', '.join(recovered[:5])}{'...' if len(recovered) > 5 else ''}"
        )

    logger.info("[SEGMENT_LOOP] v5.31 Quarantine deferred to Phase Checkout")
