# FILE: app/pipeline_v2/segmented/segmented_builder.py
# Purpose: Derek phase 5 — segmented builder entry: schedule -> scoped workers -> integrator -> BuildResult.
# Called-by: app.pipeline_v2.orchestrator (ASTRA_BUILDER_MODE=segmented)
# Depends-on: app.pipeline_v2.segmented.scheduler, app.pipeline_v2.segmented.worker, app.pipeline_v2.segmented.integrator
# Last-renovated: 2026-07-04
"""
The furnace, split: segments become execution units.

Drop-in replacement for run_agentic_builder's stage slot — same BuildResult
contract, so everything downstream (BVL, spec review, verdict) is untouched.
Sequential execution (waves flattened); parallelism arrives with Derek.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from app.pipeline_v2.models import BuildResult, BuildSession, ToolCall

logger = logging.getLogger(__name__)


def _segments_from_manifest(manifest: Any) -> List[Any]:
    """Accept a SegmentManifest object or a plain manifest dict."""
    segs = getattr(manifest, "segments", None)
    if segs is None and isinstance(manifest, dict):
        segs = manifest.get("segments") or []
    segs = list(segs or [])
    if segs and isinstance(segs[0], dict):
        try:
            from app.pot_spec.grounded.segment_schemas import SegmentSpec
            segs = [SegmentSpec.from_dict(s) if isinstance(s, dict) else s for s in segs]
        except Exception as exc:
            logger.warning("[segmented_builder] SegmentSpec coercion failed: %s", exc)
    return segs


async def run_segmented_builder(
    spec: Dict[str, Any],
    manifest: Any,
    scaffold: Any,
    job_dir: str,
    job_id: str = "",
    handover_context: Optional[str] = None,
    on_progress: Optional[Callable[[str], None]] = None,
    profile: Any = None,
) -> BuildResult:
    """Schedule segments topologically, run one scoped HANDS worker per
    segment, then the HEAVY integrator (contracts + syntax + targeted
    re-dispatch + APEX escalation). Returns a classic BuildResult."""
    from app.pipeline_v2.segmented.scheduler import (
        SchedulingError, parallel_requested, schedule_order,
    )
    from app.pipeline_v2.segmented.worker import run_segment_worker
    from app.pipeline_v2.segmented.integrator import run_integration

    t_start = time.time()
    emit = on_progress or (lambda m: None)
    result = BuildResult()
    session = BuildSession(session_number=1)
    result.sessions.append(session)

    if not job_id and job_dir:
        import os as _os
        job_id = _os.path.basename(_os.path.normpath(job_dir))

    segments = _segments_from_manifest(manifest)
    if not segments:
        result.errors.append("segmented builder: manifest has no segments")
        emit("❌ Segmented builder: no segments in manifest")
        result.total_duration_seconds = time.time() - t_start
        return result

    parallel_requested()  # logs + ignores ASTRA_SEGMENT_PARALLEL=true
    try:
        order = schedule_order(segments)
    except SchedulingError as exc:
        result.errors.append(f"segment scheduling failed: {exc}")
        emit(f"❌ Segment scheduling failed: {exc}")
        result.total_duration_seconds = time.time() - t_start
        return result

    emit(f"🧩 SEGMENTED BUILD: {len(order)} segment(s) in dependency order:")
    for s in order:
        deps = getattr(s, "dependencies", None) or []
        emit(f"   {getattr(s, 'segment_id', '?')}"
             + (f"  (after {', '.join(deps)})" if deps else ""))

    ledger = None
    try:
        from app.pipeline_v2.config import LEDGER_ENABLED
        if LEDGER_ENABLED:
            from app.pipeline_v2.ledger import load_or_create_ledger
            ledger = load_or_create_ledger(job_id, job_dir)
    except Exception as exc:
        logger.warning("[segmented_builder] ledger unavailable: %s", exc)

    reports: Dict[str, Any] = {}
    done: List[Any] = []
    for seg in order:
        sid = getattr(seg, "segment_id", "?")
        emit(f"\n   🔨 Segment {sid} — worker starting (fresh context)")
        rep = await run_segment_worker(
            seg,
            upstream=[s for s in done],
            job_dir=job_dir, job_id=job_id, profile=profile, ledger=ledger,
            handover_context=handover_context if not done else None,
            on_progress=on_progress,
        )
        reports[sid] = rep
        done.append(seg)
        session.tool_calls.extend(
            ToolCall(tool="segment_worker", args=sid) for _ in range(max(rep.tool_calls, 1))
        )
        session.total_input_tokens += rep.input_tokens
        session.total_output_tokens += rep.output_tokens
        for f in rep.files_written:
            if f not in session.files_created:
                session.files_created.append(f)
        if rep.error:
            result.errors.append(f"[{sid}] {rep.error}")
        if rep.scope_violations:
            result.errors.append(
                f"[{sid}] {len(rep.scope_violations)} write(s) outside file_scope rejected: "
                + ", ".join(rep.scope_violations[:3])
            )

    emit(f"\n   🔗 INTEGRATOR — verifying cross-segment wiring + syntax")
    integration = await run_integration(
        order, reports, job_dir=job_dir, job_id=job_id,
        profile=profile, ledger=ledger, on_progress=on_progress,
    )

    # Fold re-dispatch token/file counts back in
    session.total_input_tokens = sum(r.input_tokens for r in reports.values())
    session.total_output_tokens = sum(r.output_tokens for r in reports.values())
    all_files: List[str] = []
    for r in reports.values():
        for f in r.files_written:
            if f not in all_files:
                all_files.append(f)
    session.files_created = all_files

    if not integration.passed:
        for f in integration.failures[:8]:
            result.errors.append(f"[integration:{f.segment_id or 'cross'}] {f.detail}")

    session.completed = integration.passed and all(
        r.completed or r.files_written for r in reports.values()
    )
    result.all_files_written = all_files
    result.total_tool_calls = sum(r.tool_calls for r in reports.values())
    result.total_llm_calls = result.total_tool_calls  # per-call parity with classic builder
    result.total_input_tokens = session.total_input_tokens
    result.total_output_tokens = session.total_output_tokens
    result.total_duration_seconds = time.time() - t_start
    result.success = session.completed and bool(all_files)

    emit(
        f"\n   {'✅' if result.success else '❌'} SEGMENTED BUILD "
        f"{'complete' if result.success else 'finished with issues'}: "
        f"{len(all_files)} file(s), {result.total_tool_calls} tool calls, "
        f"{result.total_input_tokens + result.total_output_tokens:,} tokens, "
        f"{integration.redispatches} re-dispatch(es), {integration.escalations} escalation(s)"
    )
    return result


__all__ = ["run_segmented_builder"]
