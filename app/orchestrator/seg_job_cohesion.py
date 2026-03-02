# FILE: app/orchestrator/seg_job_cohesion.py
"""
Cohesion check loop for run_segmented_job.
v5.16 PHASE 2C: Cohesion Check + Automated Regen Loop.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from app.orchestrator.segment_pipeline_ctx import JobCtx
from app.orchestrator._segment_loop_utils_7 import (
    build_segment_context, can_execute_segment,
)
from app.orchestrator._segment_loop_utils_8 import update_segment_status
from app.orchestrator.segment_state import save_state, get_job_dir

logger = logging.getLogger(__name__)

# v3.2-fix: Sandbox-aware filesystem checks for codebase paths.
try:
    from app.sandbox_fs import (
        sandbox_isfile as _sbx_isfile,
        sandbox_isdir as _sbx_isdir,
        sandbox_exists as _sbx_exists,
        sandbox_read_text as _sbx_read_text,
    )
    _SBX_FS_OK = True
except ImportError:
    _SBX_FS_OK = False

MAX_COHESION_RETRIES = 3


async def run_cohesion_loop(ctx: JobCtx) -> None:
    """
    After architecture generation, run cohesion check. If blocking issues
    remain after auto-fix, auto-regenerate flagged segments through
    Critical Pipeline with cohesion feedback, then re-check.

    v3.4-fix: Skip entirely during implementation runs (implement_only=True).
    Cohesion validates architecture-level interface contracts. During
    implementation, the arch_exec writes files in dependency order — seg-04
    reads what seg-01/02/03 actually created. Cohesion at this stage would
    check against files that may not exist yet, generating false positives.
    """
    from app.pot_spec.grounded.segment_schemas import SegmentStatus

    # v3.4-fix: Cohesion is an architecture-phase check only
    if getattr(ctx, "implement_only", False):
        logger.info("[SEGMENT_LOOP] v3.4 Skipping cohesion check (implement_only=True)")
        ctx.cohesion_passed = True
        ctx.cohesion_retry_count = 0
        return

    ctx.cohesion_retry_count = 0
    ctx.cohesion_passed = False

    while ctx.cohesion_retry_count < MAX_COHESION_RETRIES and not ctx.cohesion_passed:
        approved_ids = [
            sid for sid, ss in ctx.state.segments.items()
            if ss.status in (
                SegmentStatus.APPROVED.value,
                SegmentStatus.COMPLETE.value,
            )
        ]
        if len(approved_ids) < 2:
            break

        ctx.cohesion_retry_count += 1
        ctx.emit(f"\n{'='*50}")
        if ctx.cohesion_retry_count == 1:
            ctx.emit("🔍 Running cross-segment cohesion check...")
        else:
            ctx.emit(
                f"🔍 Cohesion re-check "
                f"(attempt {ctx.cohesion_retry_count}/{MAX_COHESION_RETRIES})..."
            )

        try:
            result = await _run_single_cohesion_check(ctx, approved_ids)
            if result == "passed":
                ctx.cohesion_passed = True
            elif result == "break":
                break
            # else "continue" → loop continues for retry
        except ImportError:
            logger.debug("[SEGMENT_LOOP] Cohesion check module not available")
            break
        except Exception as coh_err:
            logger.warning(
                "[SEGMENT_LOOP] Cohesion check failed (non-fatal): %s", coh_err,
            )
            ctx.emit(f"⚠️ Cohesion check error (non-fatal): {coh_err}")
            break

    # Log final status
    if ctx.cohesion_passed:
        logger.info(
            "[SEGMENT_LOOP] v5.16 Cohesion passed after %d attempt(s)",
            ctx.cohesion_retry_count,
        )
    elif ctx.cohesion_retry_count > 0:
        logger.warning(
            "[SEGMENT_LOOP] v5.16 Cohesion not resolved after %d attempt(s)",
            ctx.cohesion_retry_count,
        )

    # v5.34 Cohesion halt gate
    if ctx.cohesion_retry_count > 0 and not ctx.cohesion_passed:
        ctx.cohesion_halted = True
        ctx.emit(f"\n{'='*50}")
        ctx.emit("🛑 PIPELINE HALTED: Cohesion check has unresolved blocking issues.")
        ctx.emit("   Phase Checkout, boot test, and Final Checkout are SKIPPED.")
        ctx.emit("   Resolve cohesion issues first, then re-run.")
        ctx.emit(f"{'='*50}")
        logger.warning(
            "[SEGMENT_LOOP] v5.34 COHESION HALT GATE — skipping downstream stages",
        )
        ctx.state.overall_status = "cohesion_failed"
        ctx.state.phase_checkout_boot = "skipped"
        save_state(ctx.state, ctx.job_dir_path)


async def _run_single_cohesion_check(
    ctx: JobCtx, approved_ids: List[str],
) -> str:
    """
    Run one cohesion check pass.
    Returns: "passed", "break" (no more retries / error), or "continue" (retry).
    """
    from app.pot_spec.grounded.segment_schemas import SegmentStatus
    from app.orchestrator.cohesion_check import (
        run_cohesion_check, save_cohesion_result,
    )

    contract_json = ctx.contract_set.to_json() if ctx.contract_set else None

    # Detect deterministic job
    is_det_job = bool(ctx.manifest.deterministic_sources)

    cohesion_result = await run_cohesion_check(
        job_id=ctx.job_id,
        job_dir=ctx.job_dir_path,
        segment_ids=approved_ids,
        contract_json=contract_json,
        source_file_evidence=ctx.source_evidence,
        skip_llm_layer=is_det_job,
    )

    # v1.0 Fix 2: Deterministic cross-segment interface validation
    try:
        from app.orchestrator.cross_segment_interfaces import (
            validate_cross_segment_interfaces,
        )
        import json as _json
        _manifest_path = os.path.join(
            ctx.job_dir_path, "segments", "manifest.json",
        )
        if os.path.isfile(_manifest_path):
            with open(_manifest_path, "r", encoding="utf-8") as _mf:
                _manifest_data = _json.load(_mf)
            _iface_issues = validate_cross_segment_interfaces(
                ctx.job_dir_path, _manifest_data.get("segments", []),
            )
            if _iface_issues:
                logger.info(
                    "[cohesion] Fix 2: Found %d interface issue(s)",
                    len(_iface_issues),
                )
                # Merge into cohesion result
                from app.orchestrator._cohesion_check_utils_10 import CohesionIssue
                for _ii in _iface_issues:
                    _ci = CohesionIssue(
                        issue_id=_ii["issue_id"],
                        severity=_ii["severity"],
                        category=_ii["category"],
                        description=_ii["description"],
                        source_segment=_ii.get("source_segment", ""),
                        related_segment=_ii.get("related_segment", ""),
                        file_path=_ii.get("file_path", ""),
                        expected=_ii.get("expected", ""),
                        actual=_ii.get("actual", ""),
                        suggested_fix=_ii.get("suggested_fix", ""),
                        auto_fix_tier=_ii.get("auto_fix_tier", 0),
                    )
                    cohesion_result.issues.append(_ci)
                # Recompute status if we added blocking issues
                _new_blocking = [
                    i for i in cohesion_result.issues
                    if i.severity == "blocking" and not i.auto_fixed
                ]
                if _new_blocking and cohesion_result.status == "pass":
                    cohesion_result.status = "fail"
                    ctx.emit(
                        f"⚠️ Interface validation found "
                        f"{len(_new_blocking)} blocking mismatch(es)"
                    )
    except Exception as _iv_err:
        logger.debug("[cohesion] Interface validation error: %s", _iv_err)

    save_cohesion_result(cohesion_result, ctx.job_dir_path)

    # Journal emission
    _emit_cohesion_journal(ctx, cohesion_result)

    # Show auto-fixed issues
    auto_fixed = [
        ci for ci in cohesion_result.issues
        if ci.auto_fixed or ci.severity == "resolved"
    ]
    if auto_fixed:
        ctx.emit(f"🔧 Auto-fixed {len(auto_fixed)} issue(s):")
        for ci in auto_fixed:
            tier_label = f"T{ci.auto_fix_tier}" if ci.auto_fix_tier else "?"
            ctx.emit(
                f"  ✅ {ci.issue_id} [{tier_label}] "
                f"{ci.auto_fix_note or ci.description[:100]}"
            )

    if cohesion_result.status == "pass":
        if auto_fixed:
            ctx.emit("✅ Cohesion check PASSED — all issues resolved by auto-fix!")
        else:
            ctx.emit("✅ Cohesion check PASSED — all segments are compatible")
        return "passed"

    if cohesion_result.status != "fail":
        ctx.emit(f"⚠️ Cohesion check error: {cohesion_result.notes or 'unknown'}")
        return "break"

    n_blocking = len(cohesion_result.blocking_issues)
    n_warning = len(cohesion_result.warning_issues)

    if ctx.cohesion_retry_count >= MAX_COHESION_RETRIES:
        _handle_exhausted_retries(ctx, cohesion_result, n_blocking, n_warning)
        return "break"

    # Auto-regen failing segments
    regen_segs = cohesion_result.segments_needing_regen
    if not regen_segs:
        ctx.emit(
            f"❌ Cohesion FAILED but no segments flagged for regen — cannot auto-fix"
        )
        return "break"

    ctx.emit(
        f"🔄 Cohesion found {n_blocking} blocking issue(s) — "
        f"auto-regenerating {len(regen_segs)} segment(s)..."
    )

    await _regen_flagged_segments(ctx, cohesion_result, regen_segs)
    ctx.emit(f"  🔄 Re-generation complete — re-running cohesion check...")
    return "continue"


def _handle_exhausted_retries(ctx, cohesion_result, n_blocking, n_warning):
    """Report failures when cohesion retries exhausted."""
    from app.pot_spec.grounded.segment_schemas import SegmentStatus

    ctx.emit(
        f"❌ Cohesion check FAILED after {MAX_COHESION_RETRIES} attempts — "
        f"{n_blocking} blocking, {n_warning} warning(s)"
    )
    for ci in cohesion_result.blocking_issues:
        tier_label = f"T{ci.auto_fix_tier}" if ci.auto_fix_tier else "?"
        ctx.emit(
            f"  🚫 {ci.issue_id} [{ci.category}/{tier_label}] "
            f"{ci.source_segment} ↔ {ci.related_segment}"
        )
        ctx.emit(f"     {ci.description}")
        if ci.suggested_fix:
            ctx.emit(f"     Fix: {ci.suggested_fix}")
    for ci in cohesion_result.warning_issues:
        ctx.emit(f"  ⚠️ {ci.issue_id} [{ci.category}] {ci.description}")

    regen_segs = cohesion_result.segments_needing_regen
    if regen_segs:
        for regen_seg_id in regen_segs:
            if regen_seg_id in ctx.state.segments:
                feedback = _build_cohesion_feedback(
                    cohesion_result, regen_seg_id,
                )
                ctx.state.segments[regen_seg_id].status = SegmentStatus.PENDING.value
                ctx.state.segments[regen_seg_id].error = feedback
        ctx.emit(
            f"  🔄 Marked {len(regen_segs)} segment(s) for manual re-generation"
        )
        ctx.emit(
            f"  💡 Say 'Astra, command: run segments' to retry architecture generation"
        )
    save_state(ctx.state, get_job_dir(ctx.job_id))


async def _regen_flagged_segments(ctx, cohesion_result, regen_segs):
    """Re-generate architecture for flagged segments."""
    from app.pot_spec.grounded.segment_schemas import SegmentStatus
    from app.orchestrator.segment_loop import run_segment_through_pipeline

    # v5.35: Protect files from completed segments not being regenerated
    protected_files = _collect_protected_files(ctx, regen_segs)
    if protected_files:
        ctx.emit(
            f"  🛡️ {len(protected_files)} files from completed segments "
            f"are protected during regen"
        )

    # Mark flagged segments PENDING with cohesion feedback
    for regen_seg_id in regen_segs:
        if regen_seg_id in ctx.state.segments:
            feedback = _build_cohesion_feedback(cohesion_result, regen_seg_id)
            ctx.state.segments[regen_seg_id].status = SegmentStatus.PENDING.value
            ctx.state.segments[regen_seg_id].error = feedback
    save_state(ctx.state, get_job_dir(ctx.job_id))

    # Re-run flagged segments through Critical Pipeline
    for regen_seg_id in regen_segs:
        seg_spec = ctx.manifest.get_segment(regen_seg_id)
        if seg_spec is None:
            continue
        if not can_execute_segment(seg_spec, ctx.state):
            ctx.emit(
                f"  ⏳ {regen_seg_id}: waiting on dependencies (skipping regen)"
            )
            continue

        ctx.emit(f"  🔄 Re-generating architecture for {regen_seg_id}...")
        update_segment_status(
            ctx.state, regen_seg_id, SegmentStatus.IN_PROGRESS, ctx.job_dir_path,
        )

        segment_context = build_segment_context(
            seg_spec, ctx.state, ctx.parent_spec, ctx.job_dir_path,
            contract_set=ctx.contract_set,
            source_file_evidence=ctx.source_evidence,
            enrichment=ctx.enrichment_data.get(seg_spec.segment_id),
            ledger=ctx.ledger,
        )

        seg_state = ctx.state.segments.get(regen_seg_id)
        if (
            seg_state
            and seg_state.error
            and seg_state.error.startswith("Cohesion regen:")
        ):
            segment_context["cohesion_issues"] = seg_state.error
            ctx.emit(
                f"  🧩 Injected cohesion issues for {regen_seg_id} "
                f"(arch-only, no approval bypass)"
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
            )
            if pipeline_result.get("success"):
                if pipeline_result.get("awaiting_approval"):
                    update_segment_status(
                        ctx.state, regen_seg_id,
                        SegmentStatus.APPROVED, ctx.job_dir_path,
                    )
                ctx.emit(f"  ✅ {regen_seg_id}: architecture re-generated")
            else:
                ctx.emit(
                    f"  ❌ {regen_seg_id}: regen failed — "
                    f"{pipeline_result.get('error', 'unknown')}"
                )
        except Exception as regen_err:
            logger.exception(
                "[SEGMENT_LOOP] v5.16 Regen failed for %s: %s",
                regen_seg_id, regen_err,
            )
            ctx.emit(f"  ❌ {regen_seg_id}: regen error — {regen_err}")

    save_state(ctx.state, get_job_dir(ctx.job_id))

    # v5.35: Post-regen file protection check
    if protected_files:
        _check_protected_files(ctx, protected_files)


def _build_cohesion_feedback(cohesion_result, seg_id: str) -> str:
    """Build structured cohesion feedback for a segment."""
    fb_parts = []
    for ci in cohesion_result.blocking_issues:
        if ci.source_segment != seg_id and ci.related_segment != seg_id:
            continue
        part = f"[{ci.issue_id}] {ci.category}: {ci.description}"
        if ci.expected:
            part += f" | Expected: {ci.expected[:200]}"
        if ci.actual:
            part += f" | Actual: {ci.actual[:200]}"
        if ci.suggested_fix:
            part += f" | Fix: {ci.suggested_fix[:200]}"
        if ci.auto_fix_note and "FAILED" in ci.auto_fix_note:
            part += f" | Autofix FAILED: {ci.auto_fix_note}"
        fb_parts.append(part)
    return (
        "Cohesion regen:\n" + "\n".join(fb_parts)
        if fb_parts
        else f"Cohesion regen: blocking issues for {seg_id}"
    )


def _collect_protected_files(ctx, regen_segs) -> set:
    """Collect output files from completed segments NOT being regenerated."""
    from app.pot_spec.grounded.segment_schemas import SegmentStatus
    protected = set()
    for ps_id, ps_state in ctx.state.segments.items():
        if (
            ps_id not in regen_segs
            and ps_state.status == SegmentStatus.COMPLETE.value
        ):
            for pf in (ps_state.output_files or []):
                protected.add(pf.replace("\\", "/"))
    return protected


# v3.4-fix: Frontend path prefixes that resolve to D:\orb-desktop
_COH_FE_ROOT = r"D:\orb-desktop"
_COH_FE_BARE = ("src/", "src\\", "public/", "public\\")


def _check_protected_files(ctx, protected_files):
    """v5.35: Verify protected files survived regen.

    v3.4-fix: Frontend path resolution — bare src/ or public/ paths
    resolve to D:\\orb-desktop, matching the write path used by arch_exec.
    """
    missing = []
    for pf in protected_files:
        normalized_fwd = pf.replace("\\", "/")
        candidates = [
            pf,
            os.path.join("D:/Orb", pf),
            pf.replace("/", os.sep),
        ]
        # v3.4-fix: Add frontend path candidate
        if any(normalized_fwd.startswith(bp.replace("\\", "/")) for bp in _COH_FE_BARE):
            candidates.append(os.path.join(_COH_FE_ROOT, pf.replace("/", os.sep)))
        elif normalized_fwd.startswith("orb-desktop/"):
            fe_rel = normalized_fwd[len("orb-desktop/"):]
            candidates.append(os.path.join(_COH_FE_ROOT, fe_rel.replace("/", os.sep)))
        if not any((_sbx_isfile(c) if _SBX_FS_OK else os.path.isfile(c)) for c in candidates):
            missing.append(pf)

    if missing:
        logger.error(
            "[SEGMENT_LOOP] v5.35 PROTECTION VIOLATION: %d files missing: %s",
            len(missing), missing[:5],
        )
        ctx.emit(
            f"  ⚠️ PROTECTION VIOLATION: {len(missing)} "
            f"completed segment files missing after regen!"
        )
        ctx.emit(
            f"     Missing: "
            f"{', '.join(os.path.basename(f) for f in missing[:5])}"
        )
    else:
        logger.info(
            "[SEGMENT_LOOP] v5.35 All %d protected files intact",
            len(protected_files),
        )


def _emit_cohesion_journal(ctx, cohesion_result):
    """v5.29: Emit cohesion issues to journal."""
    try:
        from app.experience.journal_writer import emit_journal_entry
        from app.experience.schemas import JournalEventType

        evt_map = {
            "import_mismatch": JournalEventType.COHESION_MISMATCH,
            "missing_export": JournalEventType.COHESION_MISMATCH,
            "naming_mismatch": JournalEventType.COHESION_NAMING_DRIFT,
            "shape_mismatch": JournalEventType.COHESION_INTERFACE_BREAK,
            "contract_violation": JournalEventType.COHESION_INTERFACE_BREAK,
            "scope_violation": JournalEventType.COHESION_MISMATCH,
            "phantom_segment": JournalEventType.COHESION_MISMATCH,
            "endpoint_mismatch": JournalEventType.COHESION_INTERFACE_BREAK,
        }

        for ci in cohesion_result.issues:
            evt = evt_map.get(ci.category, JournalEventType.COHESION_MISMATCH)
            emit_journal_entry(
                ctx.job_id,
                ctx.job_dir_path,
                stage="cohesion_check",
                event_type=evt.value,
                severity="blocking" if ci.severity == "blocking" else "warning",
                description=ci.description[:300],
                root_cause=ci.category,
                resolution=ci.auto_fix_note if ci.auto_fixed else ci.suggested_fix,
                file_scope=ci.file_path,
                segment_id=ci.source_segment,
                details={
                    "issue_id": ci.issue_id,
                    "expected": ci.expected[:200] if ci.expected else "",
                    "actual": ci.actual[:200] if ci.actual else "",
                    "related_segment": ci.related_segment,
                    "auto_fixed": ci.auto_fixed,
                    "auto_fix_tier": ci.auto_fix_tier,
                },
            )
    except Exception as jrn_err:
        logger.debug(
            "[SEGMENT_LOOP] v5.29 cohesion journal emit failed: %s", jrn_err,
        )
