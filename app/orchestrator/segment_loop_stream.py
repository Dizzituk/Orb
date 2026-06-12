# FILE: app/orchestrator/segment_loop_stream.py
# Purpose: SSE streaming handler for segmented job execution.
# Called-by: app.llm.routing.handler_registry
# Depends-on: app.llm.spec_flow_state, app.llm.stage_models, app.orchestrator.segment_loop, app.orchestrator.segment_state (+2 more)
# Last-renovated: 2026-06-11
"""
SSE streaming handler for segmented job execution.

Thin wrapper around segment_loop.run_segmented_job() that yields SSE
events for frontend consumption. Follows the same pattern as
generate_critical_pipeline_stream() in stream_handler.py.

This handler is invoked when the user says "run segments" after SpecGate
has produced a segment manifest. It provides real-time progress for each
segment and supports crash recovery (re-running resumes from state.json).

Phase 2 of Pipeline Segmentation.

v1.0 (2026-02-08): Initial implementation
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncGenerator, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

SEGMENT_LOOP_STREAM_BUILD_ID = "2026-02-10-v1.1-unified-pipeline"
print(f"[SEGMENT_LOOP_STREAM_LOADED] BUILD_ID={SEGMENT_LOOP_STREAM_BUILD_ID}")

# --- Internal imports ---
from app.orchestrator.segment_loop import run_segmented_job
from app.orchestrator.segment_state import get_job_dir, load_state

# --- Path construction ---
try:
    from app.pot_spec.spec_gate_persistence import artifact_root as _artifact_root
except ImportError:
    def _artifact_root():
        return os.path.abspath(os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"))

# --- Spec service (to load the parent spec from DB) ---
try:
    from app.specs.service import get_latest_validated_spec, get_spec
    _SPEC_SERVICE_AVAILABLE = True
except ImportError:
    _SPEC_SERVICE_AVAILABLE = False
    get_latest_validated_spec = None
    get_spec = None

# --- Flow state management ---
try:
    from app.llm.spec_flow_state import (
        get_active_flow,
        SpecFlowStage,
    )
    _FLOW_STATE_AVAILABLE = True
except ImportError:
    _FLOW_STATE_AVAILABLE = False
    get_active_flow = None

# --- Stage model config ---
try:
    from app.llm.stage_models import get_pipeline_model_config
except ImportError:
    def get_pipeline_model_config():
        return {"provider": "anthropic", "model": "claude-sonnet-4-20250514"}


# =============================================================================
# SSE helpers (same pattern as critical_pipeline/stream_handler.py)
# =============================================================================

def _sse(event_type: str, content: str = "", **extra) -> str:
    payload = {"type": event_type}
    if content:
        payload["content"] = content
    payload.update(extra)
    return "data: " + json.dumps(payload) + "\n\n"


def _token(text: str) -> str:
    return _sse("token", text)


def _done(**fields) -> str:
    return _sse("done", **fields)


# =============================================================================
# MANIFEST DISCOVERY
# =============================================================================

def _find_manifest_path(job_id: str) -> Optional[str]:
    """
    Find the manifest.json for a job.

    Checks the canonical location: <job_dir>/segments/manifest.json
    """
    job_dir_path = get_job_dir(job_id)
    manifest_path = os.path.join(job_dir_path, "segments", "manifest.json")
    if os.path.isfile(manifest_path):
        return manifest_path
    return None


def _find_latest_job_with_manifest(project_id: int, db: Session) -> Optional[str]:
    """
    v5.4: Find the most recent job ID that has a segment manifest.

    Since Phase 1A (always-manifest), every validated spec produces a manifest,
    whether single or multi-segment. Scans the jobs directory for manifests,
    returns the most recently modified.
    """
    jobs_root = os.path.join(_artifact_root(), "jobs")
    if not os.path.isdir(jobs_root):
        return None

    # Scan for job directories with manifests, sorted by mtime (newest first)
    candidates = []
    try:
        for entry in os.scandir(jobs_root):
            if entry.is_dir():
                manifest = os.path.join(entry.path, "segments", "manifest.json")
                if os.path.isfile(manifest):
                    candidates.append((entry.name, os.path.getmtime(manifest)))
    except OSError:
        return None

    if not candidates:
        return None

    # Return most recently modified
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def _load_parent_spec(project_id: int, db: Session) -> Optional[dict]:
    """
    Load the parent SPoT spec from the database.

    The parent spec is the full spec that was validated by SpecGate
    before segmentation. It's used as reference context for segments.
    """
    if not _SPEC_SERVICE_AVAILABLE or not get_latest_validated_spec:
        return None

    try:
        db_spec = get_latest_validated_spec(db, project_id)
        if db_spec:
            # Extract spec data from the DB model
            spec_data = {}
            if hasattr(db_spec, "spec_json") and db_spec.spec_json:
                spec_data = (
                    json.loads(db_spec.spec_json)
                    if isinstance(db_spec.spec_json, str)
                    else db_spec.spec_json
                )
            if hasattr(db_spec, "content_markdown") and db_spec.content_markdown:
                spec_data["spot_markdown"] = db_spec.content_markdown
            return spec_data
    except Exception as e:
        logger.warning("[SEGMENT_LOOP] Failed to load parent spec: %s", e)

    return None


# =============================================================================
# MAIN SSE STREAM HANDLER
# =============================================================================

async def generate_segment_loop_stream(
    project_id: int,
    db: Session,
    job_id: Optional[str] = None,
    trace: Optional[Any] = None,
    conversation_id: Optional[str] = None,
    implement_only: bool = False,
) -> AsyncGenerator[str, None]:
    """
    SSE stream handler for segmented job execution.

    Loads the segment manifest, executes each segment through the pipeline
    in dependency order, and yields real-time progress events.

    If no job_id is provided, discovers the most recent segmented job
    for this project.

    Supports crash recovery: if a previous run was interrupted, resumes
    from the last saved state.
    """
    response_parts = []
    model_cfg = get_pipeline_model_config()
    pipeline_provider = model_cfg.get("provider", "anthropic")
    pipeline_model = model_cfg.get("model", "unknown")

    def _emit(text):
        response_parts.append(text)
        return _token(text + "\n")

    try:
        yield _emit("🔀 **Pipeline Executor**\n")

        # =================================================================
        # Step 1: Find the segmented job
        # =================================================================

        if not job_id:
            job_id = _find_latest_job_with_manifest(project_id, db)

        if not job_id:
            yield _emit("❌ No job with manifest found. Run SpecGate first to validate a spec.")
            yield _done(
                provider=pipeline_provider, model=pipeline_model,
                total_length=sum(len(p) for p in response_parts),
            )
            return

        yield _emit(f"📋 Job ID: `{job_id}`")

        # =================================================================
        # Step 2: Find manifest
        # =================================================================

        manifest_path = _find_manifest_path(job_id)
        if not manifest_path:
            yield _emit(f"❌ No manifest found for job `{job_id}`. SpecGate may not have segmented this job.")
            yield _done(
                provider=pipeline_provider, model=pipeline_model,
                total_length=sum(len(p) for p in response_parts),
            )
            return

        yield _emit(f"📋 Manifest: `{manifest_path}`\n")

        # =================================================================
        # Step 3: Check for existing state (crash recovery)
        # =================================================================

        job_dir_path = get_job_dir(job_id)
        existing_state = load_state(job_dir_path)

        if existing_state:
            counts = existing_state.count_by_status()
            completed = counts.get("complete", 0)
            total = existing_state.total_segments

            if existing_state.overall_status == "complete":
                yield _emit(f"✅ Job already COMPLETE ({completed}/{total} segments).")
                yield _emit("All segments have been executed. No work to do.")
                yield _done(
                    provider=pipeline_provider, model=pipeline_model,
                    total_length=sum(len(p) for p in response_parts),
                    job_status="complete",
                )
                return

            if completed > 0:
                yield _emit(
                    f"🔄 **Resuming from crash recovery** — "
                    f"{completed}/{total} segments already complete"
                )
            else:
                yield _emit(f"🔄 Resuming execution — status: {existing_state.overall_status}")

        # =================================================================
        # Step 4: Load parent spec
        # =================================================================

        parent_spec = _load_parent_spec(project_id, db) or {}
        if parent_spec:
            yield _emit("📄 Parent spec loaded from database")
        else:
            yield _emit("⚠️ Parent spec not found in DB — segments will execute with local context only")

        yield _emit("")  # blank line before segment execution

        # =================================================================
        # Step 5: Execute the segment loop
        # =================================================================

        # JOB 4 (2026-06-10): live progress + watchdog. Progress messages are
        # pushed onto an asyncio queue by the pipeline's on_progress callback
        # and yielded AS THEY HAPPEN, instead of being buffered until the
        # whole pipeline returns (which made long builds look like silence
        # and hid wedged calls). A heartbeat event fires every 15s of quiet
        # so the frontend can distinguish "working" from "dead".
        progress_queue: asyncio.Queue = asyncio.Queue()

        def on_progress(msg: str):
            try:
                progress_queue.put_nowait(msg)
            except Exception:
                pass

        # Emit a "pipeline_started" event for the frontend
        yield _sse(
            "segment_loop_started",
            job_id=job_id,
            manifest_path=manifest_path,
        )

        # Run the orchestrator loop as a task so we can stream while it runs
        run_task = asyncio.create_task(run_segmented_job(
            job_id=job_id,
            manifest_path=manifest_path,
            parent_spec=parent_spec,
            db=db,
            project_id=project_id,
            on_progress=on_progress,
            implement_only=implement_only,
        ))

        _HEARTBEAT_SEC = 15.0
        _last_beat = time.monotonic()
        while True:
            if run_task.done() and progress_queue.empty():
                break
            try:
                msg = await asyncio.wait_for(progress_queue.get(), timeout=1.0)
                yield _emit(msg)
            except asyncio.TimeoutError:
                pass
            _now = time.monotonic()
            if _now - _last_beat >= _HEARTBEAT_SEC:
                _last_beat = _now
                yield _sse(
                    "heartbeat",
                    job_id=job_id,
                    elapsed_quiet_s=int(_HEARTBEAT_SEC),
                )

        # Re-raise any crash from the pipeline task (handled by outer except)
        final_state = run_task.result()

        # =================================================================
        # Step 5b: Emit integration check events (Phase 3)
        # =================================================================

        integration_data = final_state.integration_check
        if integration_data:
            yield _sse(
                "integration_check_start",
                job_id=job_id,
            )

            ic_status = integration_data.get("status", "unknown")
            ic_tier1 = integration_data.get("tier1_issues", [])
            ic_tier2 = integration_data.get("tier2_issues", [])
            ic_checked = integration_data.get("segments_checked", [])
            ic_skipped = integration_data.get("segments_skipped", [])

            total_issues = len(ic_tier1) + len(ic_tier2)
            error_count = sum(1 for i in ic_tier1 + ic_tier2 if i.get("severity") == "error")
            warning_count = sum(1 for i in ic_tier1 + ic_tier2 if i.get("severity") == "warning")

            # Emit per-issue details for failures/warnings
            for issue in ic_tier1 + ic_tier2:
                if issue.get("severity") in ("error", "warning"):
                    yield _sse(
                        "integration_check_issue",
                        severity=issue.get("severity", "error"),
                        check_type=issue.get("check_type", "unknown"),
                        message=issue.get("message", ""),
                        segment_a=issue.get("segment_a", ""),
                        segment_b=issue.get("segment_b", ""),
                    )

            yield _sse(
                "integration_check_complete",
                job_id=job_id,
                status=ic_status,
                total_issues=total_issues,
                error_count=error_count,
                warning_count=warning_count,
                segments_checked=len(ic_checked),
                segments_skipped=len(ic_skipped),
            )

        # =================================================================
        # Step 6: Emit final summary
        # =================================================================

        yield _sse(
            "segment_loop_complete",
            job_id=job_id,
            overall_status=final_state.overall_status,
            **final_state.count_by_status(),
        )

        # Final status message
        if final_state.overall_status == "complete":
            yield _emit("\n🎉 **All segments executed successfully!**")
        elif final_state.overall_status == "partial":
            counts = final_state.count_by_status()
            yield _emit(
                f"\n⚠️ **Partial completion:** "
                f"{counts.get('complete', 0)} complete, "
                f"{counts.get('failed', 0)} failed, "
                f"{counts.get('blocked', 0)} blocked"
            )
        else:
            yield _emit(f"\n❌ **Segmented execution {final_state.overall_status}**")

        # JOB 1 (2026-06-10): explicit machine-readable failure signal.
        # stage_hooks only flips a stage to FAILED on an SSE event of type
        # "error" — plain ❌ tokens were recording crashed builds as passed.
        if final_state.overall_status != "complete":
            yield _sse(
                "error",
                content=f"Pipeline finished with status: {final_state.overall_status}",
                job_id=job_id,
                overall_status=final_state.overall_status,
            )

        yield _done(
            provider=pipeline_provider,
            model=pipeline_model,
            total_length=sum(len(p) for p in response_parts),
            job_id=job_id,
            job_status=final_state.overall_status,
        )

    except Exception as e:
        logger.exception("[SEGMENT_LOOP] Stream failed: %s", e)
        yield _emit(f"\n❌ **Segment loop error:** {e}")
        # JOB 1 (2026-06-10): machine-readable crash signal for stage_hooks.
        yield _sse("error", content=f"Segment loop crashed: {e}")
        yield _done(
            provider=pipeline_provider,
            model=pipeline_model,
            total_length=sum(len(p) for p in response_parts),
            error=str(e),
        )


__all__ = ["generate_segment_loop_stream"]
