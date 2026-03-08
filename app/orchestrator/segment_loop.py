# FILE: app/orchestrator/segment_loop.py
"""
Core orchestrator — routes to ASTRA v2.1 Pipeline.

v9.0 (2026-03-07): V2.1 is the ONLY pipeline. The old per-segment
architecture→critique→overwatcher→implementer loop and the v8.0
agentic pipeline have been removed.

Flow: segment_loop_stream.py → run_segmented_job() → run_v2_pipeline()
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Optional

from app.orchestrator.segment_state import (
    JobState,
    get_job_dir,
)

logger = logging.getLogger(__name__)

SEGMENT_LOOP_BUILD_ID = "2026-03-08-v9.1-v2-only"
print(f"[SEGMENT_LOOP_LOADED] BUILD_ID={SEGMENT_LOOP_BUILD_ID}")

# Type alias for progress callback
ProgressCallback = Optional[Callable[[str], None]]


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
    Main entry point for pipeline execution.

    v9.1: Routes exclusively to the ASTRA v2.1 Pipeline
    (Scaffold Engine → Agentic Builder → Visual Verifier).
    """
    emit = on_progress or (lambda msg: None)

    logger.info("[SEGMENT_LOOP] v9.1 ASTRA V2.1 Pipeline — loading job %s", job_id)

    try:
        from app.pipeline_v2.orchestrator import run_v2_pipeline
    except ImportError as _imp_err:
        logger.error("[SEGMENT_LOOP] v9.1 pipeline_v2 import failed: %s", _imp_err)
        emit(f"❌ Pipeline v2.1 module not available: {_imp_err}")
        return JobState(job_id=job_id, overall_status="failed", total_segments=0)

    _v2_job_dir = os.path.join("D:\\Orb", "jobs", "jobs", job_id)

    # Load manifest
    _v2_manifest = {}
    _v2_spec = {}
    _v2_intent = ""
    try:
        with open(manifest_path, "r", encoding="utf-8") as _mf:
            _v2_manifest = json.load(_mf)

        # Try loading spec from segments dir
        _spec_path = os.path.join(os.path.dirname(manifest_path), "..", "spec.json")
        if os.path.isfile(_spec_path):
            with open(_spec_path, "r", encoding="utf-8") as _sf:
                _v2_spec = json.load(_sf)

        # Load intent from weaver if available
        _intent_path = os.path.join(_v2_job_dir, "intent.txt")
        if os.path.isfile(_intent_path):
            with open(_intent_path, "r", encoding="utf-8") as _if:
                _v2_intent = _if.read()
        elif parent_spec:
            _v2_intent = parent_spec.get("summary", str(parent_spec)[:2000])
    except Exception as _load_err:
        logger.error("[SEGMENT_LOOP] v9.1 Failed to load v2 inputs: %s", _load_err)

    if not _v2_manifest:
        emit("❌ Could not load manifest — cannot run pipeline")
        return JobState(job_id=job_id, overall_status="failed", total_segments=0)

    try:
        v2_result = await run_v2_pipeline(
            job_id=job_id,
            manifest=_v2_manifest,
            spec=_v2_spec or parent_spec,
            intent_text=_v2_intent or str(parent_spec)[:2000],
            job_dir=_v2_job_dir,
            on_progress=on_progress,
        )
        return JobState(
            job_id=job_id,
            overall_status="complete" if v2_result.success else "failed",
            total_segments=len(_v2_manifest.get("segments", [])),
        )
    except Exception as _v2_err:
        logger.error(
            "[SEGMENT_LOOP] v9.1 V2 pipeline CRASHED: %s", _v2_err, exc_info=True,
        )
        emit(f"❌ Pipeline crashed: {_v2_err}")
        return JobState(job_id=job_id, overall_status="failed", total_segments=0)