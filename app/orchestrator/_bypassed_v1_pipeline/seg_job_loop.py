# FILE: app/orchestrator/seg_job_loop.py
"""
Main segment processing loop for run_segmented_job.
Processes segments in dependency order through multi-pass execution.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from app.orchestrator.segment_pipeline_ctx import JobCtx
from app.orchestrator._segment_loop_utils_6 import (
    _find_latest_arch, collect_segment_outputs, is_segment_blocked,
)
from app.orchestrator._segment_loop_utils_7 import (
    _is_facade_segment, build_segment_context, can_execute_segment,
    mark_dependents_blocked, unblock_recovered_segments,
    verify_contracts_fulfilled,
)
from app.orchestrator._segment_loop_utils_8 import update_segment_status
from app.orchestrator.segment_state import save_state

logger = logging.getLogger(__name__)


# =========================================================================
# EXECUTION ORDER
# =========================================================================


def compute_execution_order(ctx: JobCtx) -> None:
    """Compute complexity-sorted execution order from manifest."""
    raw_order = ctx.manifest.get_execution_order()

    def _complexity_sort(order: List[str]) -> List[str]:
        completed: set = set()
        tiers: List[List[str]] = []
        remaining = list(order)
        while remaining:
            ready = [
                sid for sid in remaining
                if all(
                    d in completed
                    for d in (ctx.manifest.get_segment(sid).dependencies or [])
                )
            ]
            if not ready:
                tiers.append(remaining)
                break
            ready.sort(
                key=lambda sid: len(
                    ctx.manifest.get_segment(sid).dependencies or []
                )
            )
            tiers.append(ready)
            completed.update(ready)
            remaining = [sid for sid in remaining if sid not in completed]
        return [sid for tier in tiers for sid in tier]

    ctx.execution_order = _complexity_sort(raw_order)

    if ctx.execution_order != raw_order:
        ctx.emit(
            f"⚙️ v5.40 Complexity-sorted order: "
            f"{' → '.join(s.split('-', 2)[-1][:20] for s in ctx.execution_order)}"
        )

    ctx.emit(
        f"🔄 Processing {len(ctx.execution_order)} segment(s) "
        f"in dependency order...\n"
    )


# =========================================================================
# MULTI-PASS LOOP
# =========================================================================


async def process_segments(ctx: JobCtx) -> None:
    """Multi-pass segment processing loop."""
    from app.pot_spec.grounded.segment_schemas import SegmentStatus

    total = len(ctx.execution_order)
    pass_number = 0
    MAX_PASSES = 5

    while pass_number < MAX_PASSES:
        pass_number += 1
        progress_this_pass = 0

        if pass_number > 1:
            unblocked = unblock_recovered_segments(
                ctx.state, ctx.manifest, ctx.job_dir_path,
            )
            if unblocked:
                ctx.emit(
                    f"\n🔓 Unblocked {len(unblocked)} segment(s) "
                    f"(blocker recovered): {unblocked}"
                )
                progress_this_pass += len(unblocked)

        for idx, seg_id in enumerate(ctx.execution_order, 1):
            result = await _process_single_segment(ctx, seg_id, idx, total)
            if result == "progress":
                progress_this_pass += 1

        if progress_this_pass == 0:
            logger.info(
                "[SEGMENT_LOOP] v5.11 Pass %d: no progress — stopping",
                pass_number,
            )
            break

        remaining = sum(
            1 for ss in ctx.state.segments.values()
            if ss.status not in (
                SegmentStatus.COMPLETE.value,
                SegmentStatus.FAILED.value,
                SegmentStatus.BLOCKED.value,
            )
        )
        logger.info(
            "[SEGMENT_LOOP] v5.11 Pass %d: %d progressed, %d remaining",
            pass_number, progress_this_pass, remaining,
        )
        if remaining == 0:
            break
        ctx.emit(
            f"\n🔄 Pass {pass_number} complete "
            f"({progress_this_pass} progressed, {remaining} remaining) "
            f"— continuing...\n"
        )


# =========================================================================
# SINGLE SEGMENT DISPATCH
# =========================================================================


async def _process_single_segment(
    ctx: JobCtx, seg_id: str, idx: int, total: int,
) -> str:
    """
    Process a single segment. Returns "progress" or "skip".
    """
    from app.pot_spec.grounded.segment_schemas import SegmentStatus

    seg_state = ctx.state.segments.get(seg_id)
    seg_spec = ctx.manifest.get_segment(seg_id)

    if seg_state is None or seg_spec is None:
        logger.error(
            "[SEGMENT_LOOP] Missing state/spec for segment %s", seg_id
        )
        return "skip"

    if seg_state.status == SegmentStatus.COMPLETE.value:
        ctx.emit(f"⏭️ [{idx}/{total}] {seg_id}: already COMPLETE (skipping)")
        return "skip"

    if seg_state.status == SegmentStatus.BLOCKED.value:
        return _handle_blocked(ctx, seg_id, seg_state, seg_spec, idx, total)

    if seg_state.status == SegmentStatus.APPROVED.value:
        return await _handle_approved(ctx, seg_id, seg_spec, idx, total)

    if ctx.implement_only and seg_state.status == SegmentStatus.PENDING.value:
        return await _handle_pending_implement_only(
            ctx, seg_id, seg_spec, idx, total,
        )

    if is_segment_blocked(seg_spec, ctx.state):
        update_segment_status(
            ctx.state, seg_id, SegmentStatus.BLOCKED, ctx.job_dir_path,
            error="Dependency failed or blocked",
        )
        ctx.emit(f"🚫 [{idx}/{total}] {seg_id}: BLOCKED by failed dependency")
        return "skip"

    is_facade = _is_facade_segment(seg_spec, ctx.manifest)
    if is_facade and not ctx.implement_only:
        ctx.emit(
            f"⏭️ [{idx}/{total}] {seg_id}: FACADE — deferred to implementation phase"
        )
        return "skip"

    if not can_execute_segment(
        seg_spec, ctx.state, require_complete=is_facade,
    ):
        label = "FACADE — waiting for all dependencies" if is_facade else "waiting on dependencies"
        ctx.emit(f"⏳ [{idx}/{total}] {seg_id}: {label} (skipping)")
        return "skip"

    return await _execute_segment(ctx, seg_id, seg_spec, idx, total)


# =========================================================================
# HANDLERS
# =========================================================================


def _handle_blocked(ctx, seg_id, seg_state, seg_spec, idx, total):
    """Handle BLOCKED segment — check if blocker recovered."""
    from app.pot_spec.grounded.segment_schemas import SegmentStatus

    if not is_segment_blocked(seg_spec, ctx.state):
        seg_arch_dir = os.path.join(
            ctx.job_dir_path, "segments", seg_id, "arch",
        )
        has_arch = (
            os.path.isdir(seg_arch_dir)
            and any(f.endswith(".md") for f in os.listdir(seg_arch_dir))
        )
        restore = SegmentStatus.APPROVED if has_arch else SegmentStatus.PENDING
        update_segment_status(
            ctx.state, seg_id, restore, ctx.job_dir_path, error=None,
        )
        ctx.emit(
            f"🔓 [{idx}/{total}] {seg_id}: UNBLOCKED → {restore.value}"
        )
        return "progress"

    ctx.emit(
        f"🚫 [{idx}/{total}] {seg_id}: BLOCKED — "
        f"{seg_state.error or 'dependency failed'}"
    )
    return "skip"


async def _handle_approved(ctx, seg_id, seg_spec, idx, total):
    """Handle APPROVED segment — execute if conditions met."""
    from app.pot_spec.grounded.segment_schemas import SegmentStatus

    is_det = bool(ctx.manifest.deterministic_sources)

    if not ctx.implement_only and not is_det:
        ctx.emit(
            f"⏸️ [{idx}/{total}] {seg_id}: APPROVED — "
            f"awaiting 'implement segments' command"
        )
        return "skip"

    if is_det and not ctx.implement_only:
        ctx.emit(
            f"⚡ [{idx}/{total}] {seg_id}: Deterministic — auto-executing"
        )

    if is_segment_blocked(seg_spec, ctx.state):
        update_segment_status(
            ctx.state, seg_id, SegmentStatus.BLOCKED, ctx.job_dir_path,
            error="Dependency failed or blocked",
        )
        ctx.emit(
            f"🚫 [{idx}/{total}] {seg_id}: BLOCKED by failed dependency"
        )
        return "skip"

    # Deps must be COMPLETE for execution (not just APPROVED)
    for dep_id in (seg_spec.dependencies or []):
        dep_st = ctx.state.segments.get(dep_id)
        if dep_st and dep_st.status != SegmentStatus.COMPLETE.value:
            ctx.emit(
                f"⏳ [{idx}/{total}] {seg_id}: APPROVED but "
                f"dependencies not yet COMPLETE (skipping)"
            )
            return "skip"

    ctx.emit(f"\n✅ [{idx}/{total}] {seg_id}: APPROVED — executing...")
    ctx.emit(
        f"  Files: {', '.join(seg_spec.file_scope[:5])}"
        f"{'...' if len(seg_spec.file_scope) > 5 else ''}"
    )
    update_segment_status(
        ctx.state, seg_id, SegmentStatus.IN_PROGRESS, ctx.job_dir_path,
    )

    # Load saved architecture
    seg_dir = os.path.join(ctx.job_dir_path, "segments", seg_id)
    arch_path = _find_latest_arch(seg_dir)

    if arch_path is None or not os.path.isfile(arch_path):
        update_segment_status(
            ctx.state, seg_id, SegmentStatus.FAILED, ctx.job_dir_path,
            error=f"Architecture file not found: {arch_path}",
        )
        ctx.emit(f"  ❌ Architecture file missing: {arch_path}")
        blocked = mark_dependents_blocked(
            ctx.state, seg_id, ctx.manifest, ctx.job_dir_path,
        )
        if blocked:
            ctx.emit(f"  🚫 Blocked {len(blocked)} dependent segment(s)")
        return "skip"

    with open(arch_path, 'r', encoding='utf-8') as f:
        arch_text = f.read()
    ctx.emit(f"  📄 Loaded architecture: {arch_path} ({len(arch_text)} chars)")

    # Sanitise
    try:
        from app.orchestrator.architecture_sanitiser import sanitise_architecture
        arch_text, san_result = sanitise_architecture(
            arch_text=arch_text,
            file_scope=seg_spec.file_scope,
            segment_id=seg_id,
        )
        if san_result.had_fixes:
            ctx.emit(
                f"  🧹 Sanitiser: {san_result.fix_count} fix(es) applied"
            )
            try:
                with open(arch_path, "w", encoding="utf-8") as sf:
                    sf.write(arch_text)
            except Exception:
                pass
    except (ImportError, Exception):
        pass

    # Build context + execute
    segment_context = build_segment_context(
        seg_spec, ctx.state, ctx.parent_spec, ctx.job_dir_path,
        contract_set=ctx.contract_set,
        source_file_evidence=ctx.source_evidence,
        enrichment=ctx.enrichment_data.get(seg_spec.segment_id),
        ledger=ctx.ledger,
    )

    from app.orchestrator.seg_pipeline_step3 import execute_architecture
    pipeline_result = await execute_architecture(
        seg_id=seg_id,
        seg_job_id=f"{ctx.job_id}__{seg_id}",
        arch_text=arch_text,
        seg_arch_path=arch_path,
        segment_context=segment_context,
        segment=seg_spec,
        manifest=ctx.manifest,
        job_id=ctx.job_id,
        job_dir_path=ctx.job_dir_path,
        project_id=ctx.project_id,
        db=ctx.db,
        contract_set=ctx.contract_set,
        quarantine_result=ctx.quarantine_result,
        emit=ctx.emit,
    )

    return _apply_segment_result(
        ctx, seg_id, seg_spec, pipeline_result, idx, total,
    )


async def _handle_pending_implement_only(ctx, seg_id, seg_spec, idx, total):
    """Handle PENDING segments in implement_only mode (facade auto-gen)."""
    from app.pot_spec.grounded.segment_schemas import SegmentStatus
    from app.orchestrator.segment_loop import run_segment_through_pipeline

    if not _is_facade_segment(seg_spec, ctx.manifest):
        ctx.emit(
            f"⏭️ [{idx}/{total}] {seg_id}: PENDING — "
            f"needs architecture first (run 'run segments')"
        )
        return "skip"

    if not can_execute_segment(seg_spec, ctx.state, require_complete=True):
        ctx.emit(
            f"⏳ [{idx}/{total}] {seg_id}: FACADE — "
            f"waiting for all dependencies to be COMPLETE"
        )
        return "skip"

    ctx.emit(
        f"\n🏗️ [{idx}/{total}] {seg_id}: FACADE — "
        f"all deps COMPLETE, auto-generating + implementing"
    )
    ctx.emit(f"  Files: {', '.join(seg_spec.file_scope[:5])}")
    update_segment_status(
        ctx.state, seg_id, SegmentStatus.IN_PROGRESS, ctx.job_dir_path,
    )

    segment_context = build_segment_context(
        seg_spec, ctx.state, ctx.parent_spec, ctx.job_dir_path,
        contract_set=ctx.contract_set,
        source_file_evidence=ctx.source_evidence,
        enrichment=ctx.enrichment_data.get(seg_spec.segment_id),
        ledger=ctx.ledger,
    )
    segment_context["_facade_auto_execute"] = True

    # v5.26: Inject dependency file contents
    _inject_dep_file_evidence(ctx, seg_spec, segment_context)

    try:
        pipeline_result = await run_segment_through_pipeline(
            segment=seg_spec,
            segment_context=segment_context,
            job_id=ctx.job_id,
            db=ctx.db,
            project_id=ctx.project_id,
            on_progress=ctx.on_progress,
            contract_set=ctx.contract_set,
            job_dir_path=ctx.job_dir_path,
            manifest=ctx.manifest,
            parent_spec=ctx.parent_spec,
            quarantine_result=ctx.quarantine_result,
        )
    except Exception as e:
        pipeline_result = {
            "success": False, "error": str(e), "output_files": [],
        }
        logger.exception(
            "[SEGMENT_LOOP] v5.26 Facade pipeline error for %s", seg_id,
        )

    return _apply_segment_result(
        ctx, seg_id, seg_spec, pipeline_result, idx, total,
    )


async def _execute_segment(ctx, seg_id, seg_spec, idx, total):
    """Execute a PENDING segment through the full pipeline."""
    from app.pot_spec.grounded.segment_schemas import SegmentStatus
    from app.orchestrator.segment_loop import run_segment_through_pipeline

    ctx.emit(f"\n⚙️ [{idx}/{total}] {seg_id}: {seg_spec.title}")
    ctx.emit(
        f"  Files: {', '.join(seg_spec.file_scope[:5])}"
        f"{'...' if len(seg_spec.file_scope) > 5 else ''}"
    )
    ctx.emit(f"  Dependencies: {seg_spec.dependencies or 'none'}")

    update_segment_status(
        ctx.state, seg_id, SegmentStatus.IN_PROGRESS, ctx.job_dir_path,
    )

    segment_context = build_segment_context(
        seg_spec, ctx.state, ctx.parent_spec, ctx.job_dir_path,
        contract_set=ctx.contract_set,
        source_file_evidence=ctx.source_evidence,
        enrichment=ctx.enrichment_data.get(seg_spec.segment_id),
        ledger=ctx.ledger,
    )

    # Inject cohesion feedback for regen
    seg_state = ctx.state.segments.get(seg_id)
    if (
        seg_state
        and seg_state.error
        and seg_state.error.startswith("Cohesion regen:")
    ):
        segment_context["cohesion_feedback"] = seg_state.error
        ctx.emit(
            f"  🔄 Re-generating with cohesion feedback: "
            f"{seg_state.error[:120]}"
        )

    try:
        pipeline_result = await run_segment_through_pipeline(
            segment=seg_spec,
            segment_context=segment_context,
            job_id=ctx.job_id,
            db=ctx.db,
            project_id=ctx.project_id,
            on_progress=ctx.on_progress,
            contract_set=ctx.contract_set,
            job_dir_path=ctx.job_dir_path,
            manifest=ctx.manifest,
            parent_spec=ctx.parent_spec,
            quarantine_result=ctx.quarantine_result,
        )
    except Exception as e:
        pipeline_result = {
            "success": False, "output_files": [], "error": str(e),
            "critique_warnings": [],
        }
        logger.exception(
            "[SEGMENT_LOOP] Unexpected error processing %s", seg_id,
        )

    return _apply_segment_result(
        ctx, seg_id, seg_spec, pipeline_result, idx, total,
    )


# =========================================================================
# HELPERS
# =========================================================================


def _apply_segment_result(ctx, seg_id, seg_spec, pipeline_result, idx, total):
    """Apply pipeline result — mark COMPLETE/APPROVED/FAILED."""
    from app.pot_spec.grounded.segment_schemas import SegmentStatus

    if pipeline_result.get("success"):
        if pipeline_result.get("awaiting_approval"):
            update_segment_status(
                ctx.state, seg_id, SegmentStatus.APPROVED, ctx.job_dir_path,
            )
            ctx.emit(f"  ✅ {seg_id}: APPROVED — architecture ready for review")

            # v1.0 Fix 3: Back-propagate exports to skeleton contracts
            try:
                from app.orchestrator.cross_segment_interfaces import (
                    backpropagate_exports_to_skeleton,
                )
                _skel_path = os.path.join(
                    ctx.job_dir_path, "segments", "skeleton_contract.json",
                )
                _arch_path = os.path.join(
                    ctx.job_dir_path, "segments", seg_id, "arch", "arch_v1.md",
                )
                if os.path.isfile(_arch_path):
                    with open(_arch_path, "r", encoding="utf-8") as _af:
                        _arch_text = _af.read()
                    _bp_count = backpropagate_exports_to_skeleton(
                        seg_id, _arch_text, _skel_path,
                    )
                    if _bp_count:
                        ctx.emit(
                            f"  🔗 Back-propagated {_bp_count} export(s) "
                            f"to skeleton contracts"
                        )
            except Exception as _bp_err:
                logger.debug(
                    "[SEGMENT_LOOP] Export backprop failed (non-fatal): %s",
                    _bp_err,
                )

            return "progress"

        output_files = pipeline_result.get("output_files", [])
        if not output_files:
            output_files = collect_segment_outputs(seg_id, ctx.job_dir_path)

        update_segment_status(
            ctx.state, seg_id, SegmentStatus.COMPLETE, ctx.job_dir_path,
            output_files=output_files,
            impl_model=pipeline_result.get("impl_model"),
            impl_provider=pipeline_result.get("impl_provider"),
        )

        contract_warnings = verify_contracts_fulfilled(
            seg_id, ctx.state, ctx.manifest,
        )
        if contract_warnings:
            ctx.emit(f"  ⚠️ Contract warnings: {len(contract_warnings)}")

        _impl_model = pipeline_result.get("impl_model", "")
        _impl_tag = f" via {_impl_model}" if _impl_model else ""
        ctx.emit(
            f"  ✅ {seg_id}: COMPLETE ({len(output_files)} file(s){_impl_tag})"
        )
        return "progress"

    # FAILED
    error_msg = pipeline_result.get("error", "Unknown error")
    update_segment_status(
        ctx.state, seg_id, SegmentStatus.FAILED, ctx.job_dir_path,
        error=error_msg,
    )
    ctx.emit(f"  ❌ {seg_id}: FAILED — {error_msg}")
    print(
        f"[SEGMENT_LOOP] v3.1 ❌ SEGMENT FAILED: {seg_id} — {error_msg}"
    )

    blocked = mark_dependents_blocked(
        ctx.state, seg_id, ctx.manifest, ctx.job_dir_path,
    )
    if blocked:
        ctx.emit(f"  🚫 Blocked {len(blocked)} dependent segment(s): {blocked}")
        print(f"[SEGMENT_LOOP] v3.1 🚫 BLOCKED dependents: {blocked}")

    return "skip"


def _inject_dep_file_evidence(ctx, seg_spec, segment_context):
    """v5.26: Read dependency output files and inject as evidence.

    v3.4-fix: Reads from SANDBOX via sandbox_read_text, not host open().
    Dependency files are created by earlier segments inside the sandbox.
    They do not exist on the host filesystem.
    """
    from app.pot_spec.grounded.segment_schemas import SegmentStatus
    try:
        from app.sandbox_fs import sandbox_read_text as _sbx_read
        _sbx_ok = True
    except ImportError:
        _sbx_ok = False

    dep_file_contents: Dict[str, str] = {}
    for dep_id in seg_spec.dependencies:
        dep_state = ctx.state.segments.get(dep_id)
        if (
            dep_state
            and dep_state.status == SegmentStatus.COMPLETE.value
        ):
            for dep_file in (dep_state.output_files or []):
                try:
                    # v3.4-fix: Always read from sandbox — these files
                    # were written by the implementer inside the sandbox
                    # and do NOT exist on the host.
                    dep_content = None
                    if _sbx_ok:
                        dep_content = _sbx_read(dep_file)
                        if dep_content:
                            dep_content = dep_content[:60_000]

                    if not dep_content:
                        logger.warning(
                            "[SEGMENT_LOOP] v3.4 Dep file not readable via sandbox: %s",
                            dep_file,
                        )
                        continue

                    rel_path = dep_file
                    for root in [
                        "D:\\Orb\\", "D:\\orb-desktop\\",
                        "D:/Orb/", "D:/orb-desktop/",
                    ]:
                        if dep_file.startswith(root):
                            rel_path = dep_file[len(root):]
                            break
                    dep_file_contents[rel_path] = dep_content
                except Exception as read_err:
                    logger.warning(
                        "[SEGMENT_LOOP] v5.26 Failed to read dep file %s: %s",
                        dep_file, read_err,
                    )

    if dep_file_contents:
        existing = segment_context.get("source_file_evidence", {})
        existing.update(dep_file_contents)
        segment_context["source_file_evidence"] = existing
        ctx.emit(
            f"  📚 Injected {len(dep_file_contents)} dependency file(s) as evidence"
        )
        for dfp in sorted(dep_file_contents.keys()):
            ctx.emit(f"    → {dfp} ({len(dep_file_contents[dfp]):,} chars)")
