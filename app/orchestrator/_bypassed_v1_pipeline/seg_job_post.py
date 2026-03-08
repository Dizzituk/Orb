# FILE: app/orchestrator/seg_job_post.py
"""
Post-execution stages for run_segmented_job:
- Post-execution reconciliation
- Deferred consumer reconciliation
- Cross-segment integration check
- Deferred quarantine
- Phase Checkout
- Final Checkout
- Journal distillation
- Final summary
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from app.orchestrator.segment_pipeline_ctx import JobCtx
from app.orchestrator.segment_state import save_state

logger = logging.getLogger(__name__)


def run_post_execution_reconciliation(ctx: JobCtx) -> None:
    """v5.12: Post-execution reconciliation (Option B fallback)."""
    from app.pot_spec.grounded.segment_schemas import SegmentStatus

    any_complete = any(
        ss.status == SegmentStatus.COMPLETE.value
        for ss in ctx.state.segments.values()
    )
    if not any_complete or not ctx.implement_only:
        return

    try:
        from app.orchestrator.post_execution_reconciliation import (
            run_post_execution_reconciliation as _run_recon,
        )
        ctx.emit(f"\n{'='*50}")
        recon_result = _run_recon(
            manifest=ctx.manifest,
            state=ctx.state,
            on_progress=ctx.emit,
        )
        if recon_result.fixes_applied:
            logger.info(
                "[SEGMENT_LOOP] v5.12 Post-execution reconciliation: %d fix(es) in %d file(s)",
                len(recon_result.fixes_applied), recon_result.files_fixed,
            )
            any_failed = any(
                ss.status == SegmentStatus.FAILED.value
                for ss in ctx.state.segments.values()
            )
            if any_failed:
                ctx.emit(
                    "  💡 Fixes applied to files from failed segment(s) — "
                    "these may resolve the failure on retry"
                )
    except ImportError:
        logger.debug("[SEGMENT_LOOP] Post-execution reconciliation not available")
    except Exception as recon_err:
        logger.warning(
            "[SEGMENT_LOOP] v5.12 Post-execution reconciliation error (non-fatal): %s",
            recon_err,
        )
        ctx.emit(f"⚠️ Post-execution reconciliation error (non-fatal): {recon_err}")


def run_deferred_consumer_reconciliation(ctx: JobCtx) -> None:
    """v5.18: Deferred consumer reconciliation."""
    from app.pot_spec.grounded.segment_schemas import SegmentStatus

    deferred = getattr(ctx.manifest, 'deferred_consumer_files', []) or []
    any_complete = any(
        ss.status == SegmentStatus.COMPLETE.value
        for ss in ctx.state.segments.values()
    )
    if not deferred or not any_complete or not ctx.implement_only:
        return

    try:
        from app.orchestrator.post_execution_reconciliation import (
            reconcile_deferred_consumers,
        )
        consumer_result = reconcile_deferred_consumers(
            manifest=ctx.manifest,
            on_progress=ctx.emit,
        )
        if consumer_result.errors:
            logger.warning(
                "[SEGMENT_LOOP] v5.18 Deferred consumer issues: %s",
                consumer_result.errors,
            )
    except ImportError:
        logger.debug("[SEGMENT_LOOP] Deferred consumer recon not available")
    except Exception as dc_err:
        logger.warning(
            "[SEGMENT_LOOP] v5.18 Deferred consumer recon error (non-fatal): %s",
            dc_err,
        )


def run_integration_check(ctx: JobCtx) -> None:
    """Cross-segment integration check (Phase 3)."""
    from app.pot_spec.grounded.segment_schemas import SegmentStatus

    any_complete = any(
        s.status == SegmentStatus.COMPLETE.value
        for s in ctx.state.segments.values()
    )
    if not any_complete or ctx.cohesion_halted:
        return

    ctx.emit(f"\n{'='*50}")
    ctx.emit("🔗 Running cross-segment integration check...")

    try:
        from app.orchestrator.integration_check import (
            run_integration_check as _run_ic,
        )
        result = _run_ic(
            manifest=ctx.manifest,
            state=ctx.state,
            job_dir=ctx.job_dir_path,
            on_progress=ctx.on_progress,
        )
        ctx.state.integration_check = result.to_dict()
        save_state(ctx.state, ctx.job_dir_path)

        status_msg = {
            "fail": f"[SEGMENT_LOOP] Integration check FAILED -- {result.error_count} error(s), {result.warning_count} warning(s)",
            "warn": f"[SEGMENT_LOOP] Integration check passed with {result.warning_count} warning(s)",
            "error": f"[SEGMENT_LOOP] Integration check encountered an error: {result.error_message}",
            "skipped": "[SEGMENT_LOOP] Integration check skipped (no complete segments)",
        }
        ctx.emit(status_msg.get(result.status, "[SEGMENT_LOOP] Integration check PASSED"))

    except Exception as e:
        logger.exception("[SEGMENT_LOOP] Integration check failed to run: %s", e)
        ctx.emit(f"[SEGMENT_LOOP] Integration check error: {e}")


def run_deferred_quarantine(ctx: JobCtx) -> None:
    """v5.35: Deferred quarantine — just before Phase Checkout."""
    from app.pot_spec.grounded.segment_schemas import SegmentStatus

    if not ctx.implement_only or ctx.quarantine_result is not None or ctx.cohesion_halted:
        return

    all_complete = all(
        s.status == SegmentStatus.COMPLETE.value
        for s in ctx.state.segments.values()
    )
    any_failed = any(
        s.status in (SegmentStatus.FAILED.value, SegmentStatus.BLOCKED.value)
        for s in ctx.state.segments.values()
    )

    if any_failed:
        logger.info(
            "[SEGMENT_LOOP] v5.35 Quarantine SKIPPED — incomplete job"
        )
        ctx.emit("📦 Quarantine: SKIPPED — not all segments complete, monolith preserved")
        return

    if not all_complete:
        return

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
            ctx.emit(f"📦 Quarantine: monolith moved aside for boot test")
        if not ctx.quarantine_result.all_ok:
            for q_err in ctx.quarantine_result.errors:
                ctx.emit(f"  ⚠️ Quarantine warning: {q_err}")
    except ImportError:
        logger.debug("[SEGMENT_LOOP] Package quarantine not available")
    except Exception as q_err:
        logger.warning(
            "[SEGMENT_LOOP] v5.31 Deferred quarantine failed (non-fatal): %s", q_err
        )
        ctx.emit(f"⚠️ Quarantine check failed (non-fatal): {q_err}")


async def run_phase_checkout(ctx: JobCtx) -> None:
    """v5.0 Phase Checkout — Stage 9 Full Verification."""
    from app.pot_spec.grounded.segment_schemas import SegmentStatus

    total = len(ctx.execution_order)
    any_complete = any(
        s.status == SegmentStatus.COMPLETE.value
        for s in ctx.state.segments.values()
    )
    no_in_progress = not any(
        s.status == SegmentStatus.IN_PROGRESS.value
        for s in ctx.state.segments.values()
    )
    impl_pass_done = any_complete and no_in_progress and total > 0

    if not impl_pass_done or ctx.cohesion_halted:
        return

    incomplete = [
        sid for sid, s in ctx.state.segments.items()
        if s.status != SegmentStatus.COMPLETE.value
    ]
    all_complete = len(incomplete) == 0

    if incomplete and not all_complete:
        ctx.emit(
            f"\n⚠️ {len(incomplete)} segment(s) incomplete "
            f"({', '.join(incomplete[:3])}{'...' if len(incomplete) > 3 else ''}) "
            f"— running Phase Checkout on completed segments"
        )

    try:
        from app.orchestrator.phase_checkout import run_phase_checkout as _run_pc
        from app.orchestrator.skeleton_contracts import load_skeleton_contract

        skeleton = load_skeleton_contract(ctx.job_dir_path)
        checkout_result = await _run_pc(
            job_id=ctx.job_id,
            job_dir=ctx.job_dir_path,
            state=ctx.state,
            manifest=ctx.manifest,
            skeleton=skeleton,
            attempt=1,
            emit=ctx.emit,
        )

        if checkout_result.boot_test:
            ctx.state.phase_checkout_boot = checkout_result.boot_test.status
            if checkout_result.boot_test.error_summary:
                ctx.state.phase_checkout_error = checkout_result.boot_test.error_summary[:500]

        ctx.state.integration_check = ctx.state.integration_check or {}
        ctx.state.integration_check["phase_checkout"] = checkout_result.to_dict()

        if checkout_result.passed:
            logger.info("[SEGMENT_LOOP] v5.0 Phase Checkout PASSED")
        elif checkout_result.routing:
            _sev = getattr(checkout_result.routing, 'severity', 'unknown')
            _stage = checkout_result.routing.target_stage
            _scoped = getattr(checkout_result.routing, 'scoped_files', None)

            if _sev == "major":
                # v1.3: Major failures fail cleanly — no LLM heroics.
                logger.warning(
                    "[SEGMENT_LOOP] v5.0 Phase Checkout FAILED (MAJOR) — "
                    "not auto-fixable. Reason: %s",
                    checkout_result.routing.reason,
                )
                ctx.emit(
                    f"❌ Phase Checkout FAILED (major severity) — "
                    f"not sending to auto-fix. Reason: {checkout_result.routing.reason}"
                )
            else:
                logger.warning(
                    "[SEGMENT_LOOP] v5.0 Phase Checkout FAILED → route to %s "
                    "(seg=%s, severity=%s, scoped=%s)",
                    _stage,
                    checkout_result.routing.target_segment or "all",
                    _sev,
                    [os.path.basename(f) for f in _scoped] if _scoped else "all",
                )
                if _scoped:
                    ctx.emit(
                        f"🔧 Phase Checkout: surgical fix needed in "
                        f"{', '.join(os.path.basename(f) for f in _scoped)}"
                    )

                    # v1.4: Deterministic surgical fix for minor failures.
                    # Preamble contamination (content before first import) is
                    # fixable without LLM — strip everything before the first
                    # import line. This was the gap that let job sg-bc6118fe
                    # pass Phase Checkout with a known-broken file.
                    _reason = getattr(checkout_result.routing, 'reason', '')
                    _is_preamble = (
                        'garbage' in _reason.lower()
                        or 'preamble' in _reason.lower()
                        or 'contamina' in _reason.lower()
                        or 'non-code' in _reason.lower()
                    )
                    if _is_preamble and _scoped:
                        _fixed_count = _deterministic_preamble_fix(
                            _scoped, ctx.emit,
                        )
                        if _fixed_count:
                            ctx.emit(
                                f"  ✅ Deterministic preamble fix applied to "
                                f"{_fixed_count} file(s)"
                            )

    except (ImportError, Exception) as pc_err:
        logger.warning("[SEGMENT_LOOP] v5.0 Phase Checkout error: %s", pc_err)
        ctx.emit(f"⚠️ Phase Checkout could not run: {pc_err}")
        ctx.state.phase_checkout_boot = "error"

    save_state(ctx.state, ctx.job_dir_path)


def _deterministic_preamble_fix(
    scoped_files: list,
    emit: Any = None,
) -> int:
    """Deterministic preamble fix — strip content before first import.

    v1.4 (2026-03-04): Fixes preamble contamination where scaffold stubs
    or arch-doc snippets are prepended before the import block. This is
    a pure deterministic fix — no LLM needed.

    Operates in the sandbox via sandbox_client for frontend files,
    or directly on host for backend files.

    Returns number of files fixed.
    """
    _emit = emit or (lambda msg: None)
    fixed = 0

    for filepath in scoped_files:
        try:
            norm = filepath.replace("\\", "/")
            is_frontend = (
                "orb-desktop" in norm
                or norm.endswith(".tsx")
                or norm.endswith(".ts")
                or norm.endswith(".jsx")
            )

            if is_frontend:
                from app.overwatcher.sandbox_client import get_sandbox_client
                client = get_sandbox_client()
                result = client.shell_run(
                    f'Get-Content "{filepath}" -Raw -Encoding UTF8',
                    cwd_target="REPO",
                )
                file_content = result.stdout or ""
            else:
                from app.sandbox_fs import sandbox_read_text
                file_content = sandbox_read_text(filepath) or ""

            if not file_content.strip():
                continue

            lines = file_content.split("\n")

            # Find first import line
            first_import = -1
            for i, line in enumerate(lines):
                stripped = line.strip()
                if (
                    stripped.startswith("import ")
                    or stripped.startswith("from ")
                    or stripped.startswith("import{")
                ):
                    first_import = i
                    break

            if first_import <= 0:
                # No preamble (import is first line) or no imports at all
                continue

            # Check if there's actual code content before the import
            pre_import = "\n".join(lines[:first_import]).strip()
            if not pre_import or pre_import.startswith("//") or pre_import.startswith("/*"):
                # Only comments before import — that's fine
                comment_only = all(
                    l.strip().startswith("//")
                    or l.strip().startswith("/*")
                    or l.strip().startswith("*")
                    or l.strip() == ""
                    for l in lines[:first_import]
                )
                if comment_only:
                    continue

            # Strip everything before the first import
            cleaned = "\n".join(lines[first_import:])
            _emit(
                f"    [PREAMBLE FIX] {os.path.basename(filepath)}: "
                f"stripped {first_import} line(s) before first import"
            )
            logger.info(
                "[PREAMBLE_FIX] %s: stripped %d lines of preamble",
                filepath, first_import,
            )

            if is_frontend:
                import base64
                encoded = base64.b64encode(
                    cleaned.encode("utf-8")
                ).decode("ascii")
                client.shell_run(
                    f'[System.IO.File]::WriteAllBytes("{filepath}", '
                    f'[Convert]::FromBase64String("{encoded}"))',
                    cwd_target="REPO",
                )
            else:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(cleaned)

            fixed += 1

        except Exception as exc:
            logger.warning(
                "[PREAMBLE_FIX] Failed to fix %s: %s", filepath, exc,
            )
            _emit(f"    ⚠️ Preamble fix failed for {os.path.basename(filepath)}: {exc}")

    return fixed


async def run_final_checkout(ctx: JobCtx) -> None:
    """v5.14 Final Checkout — Stage 10 (Autonomous Closer + Learning Report)."""
    from app.pot_spec.grounded.segment_schemas import SegmentStatus

    total = len(ctx.execution_order)
    all_complete = all(
        s.status == SegmentStatus.COMPLETE.value
        for s in ctx.state.segments.values()
    )
    if not all_complete or total == 0 or ctx.state.phase_checkout_boot != "pass" or ctx.cohesion_halted:
        return

    ctx.emit(f"\n{'='*50}")
    ctx.emit("🏁 Running Final Checkout (Stage 10)...")

    try:
        from app.orchestrator.final_checkout import run_final_checkout as _run_fc

        original_spec = None
        if isinstance(ctx.parent_spec, dict):
            original_spec = ctx.parent_spec.get("spec_markdown") or ctx.parent_spec.get("content", "")
            if not original_spec:
                try:
                    original_spec = json.dumps(ctx.parent_spec)[:8000]
                except Exception:
                    pass
        elif isinstance(ctx.parent_spec, str):
            original_spec = ctx.parent_spec

        final_result = await _run_fc(
            job_id=ctx.job_id,
            job_dir=ctx.job_dir_path,
            sandbox_base=os.getenv("ORB_SANDBOX_BASE", r"D:\Orb"),
            original_spec=original_spec,
            state=ctx.state,
            manifest=ctx.manifest,
            emit=ctx.emit,
        )

        ctx.state.integration_check = ctx.state.integration_check or {}
        ctx.state.integration_check["final_checkout"] = final_result.to_dict()
        save_state(ctx.state, ctx.job_dir_path)

        if final_result.status == "pass":
            ctx.emit("🏁 Final Checkout PASSED")
        else:
            ctx.emit(f"🏁 Final Checkout FAILED — see final_checkout_result.json")

    except ImportError:
        logger.debug("[SEGMENT_LOOP] Final Checkout module not available")
    except Exception as fc_err:
        logger.warning("[SEGMENT_LOOP] v5.14 Final Checkout error: %s", fc_err)
        ctx.emit(f"⚠️ Final Checkout could not run: {fc_err}")


def distill_journal(ctx: JobCtx) -> None:
    """v5.20: ALWAYS distill journal — no matter how the job ends."""
    total = len(ctx.execution_order)
    if total == 0:
        return

    try:
        from app.experience.distillation import distill_job
        from app.db import get_db_session
        distill_db = get_db_session()
        patterns = distill_job(distill_db, ctx.job_id, ctx.job_dir_path)
        if patterns:
            ctx.emit(f"🧠 Distilled {len(patterns)} experience pattern(s) from journal")
            logger.info(
                "[SEGMENT_LOOP] Distilled %d patterns for job %s",
                len(patterns), ctx.job_id,
            )
        distill_db.close()
    except Exception as distill_err:
        logger.debug("[SEGMENT_LOOP] Distillation skipped: %s", distill_err)


def emit_quarantine_status(ctx: JobCtx) -> None:
    """v5.7/v5.26: Quarantine status report (no auto-delete)."""
    if not ctx.quarantine_result or not ctx.quarantine_result.has_quarantined:
        return

    final_status = ctx.state.compute_overall_status()
    if final_status == "complete":
        ctx.emit("\n📦 Quarantine: All segments COMPLETE.")
        ctx.emit("  Original files preserved in .quarantined/ folders.")
        ctx.emit("  To clean up: manually delete .quarantined/ dirs when satisfied.")
        ctx.emit("  To rollback: 'Astra, command: rollback quarantine'")
    elif final_status == "failed":
        ctx.emit(
            "\n📦 Quarantine: Job FAILED — original files safe in .quarantined/ folders."
        )
        ctx.emit("  To rollback: 'Astra, command: rollback quarantine'")


def compact_evidence_ledger(ctx: JobCtx) -> None:
    """v4.3: Compact ledger after all segments complete."""
    if not getattr(ctx, "ledger", None):
        return
    try:
        from app.orchestrator.ledger_compactor import compact_ledger
        compact_ledger(
            ledger=ctx.ledger,
            job_dir=ctx.job_dir_path,
            pass_number=1,
            emit=ctx.emit,
        )
    except Exception as exc:
        logger.debug("[SEGMENT_LOOP] Ledger compaction failed (non-fatal): %s", exc)


def emit_final_summary(ctx: JobCtx) -> None:
    """Final summary output."""
    from app.pot_spec.grounded.segment_schemas import SegmentStatus

    total = len(ctx.execution_order)
    ctx.state.overall_status = ctx.state.compute_overall_status()
    save_state(ctx.state, ctx.job_dir_path)

    counts = ctx.state.count_by_status()
    approved_count = sum(
        1 for seg in ctx.state.segments.values()
        if seg.status == SegmentStatus.APPROVED.value
    )

    ctx.emit(f"\n{'='*50}")
    ctx.emit(f"📊 SEGMENTED EXECUTION COMPLETE")
    ctx.emit(f"   Status: {ctx.state.overall_status.upper()}")
    ctx.emit(f"   Complete: {counts.get('complete', 0)}/{total}")
    if approved_count:
        ctx.emit(f"   ⏸️ Approved (awaiting execution): {approved_count} segment(s)")
        ctx.emit(
            f"   Say 'Astra, command: implement segments' to execute approved segments"
        )
    if counts.get("failed", 0):
        ctx.emit(f"   Failed: {counts.get('failed', 0)}")
    if counts.get("blocked", 0):
        ctx.emit(f"   Blocked: {counts.get('blocked', 0)}")

    boot = ctx.state.phase_checkout_boot
    boot_labels = {
        "pass": "🏁 Boot check: PASSED",
        "fail": "🏁 Boot check: FAILED",
        "skipped": "🏁 Boot check: SKIPPED (cohesion unresolved)",
        "error": "🏁 Boot check: ERROR (could not run)",
    }
    if boot in boot_labels:
        ctx.emit(f"   {boot_labels[boot]}")
    ctx.emit(f"{'='*50}")

    logger.info(
        "[SEGMENT_LOOP] Job %s finished: %s", ctx.job_id, ctx.state.summary()
    )
    print(f"[SEGMENT_LOOP] DONE: {ctx.state.summary()}")

    try:
        from app.experience.context import clear_job_context
        clear_job_context()
    except Exception:
        pass
