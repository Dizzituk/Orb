from __future__ import annotations
import json
import logging
from app.llm.critical_pipeline._stream_handler_utils import _handle_micro, _handle_scan
from app.llm.critical_pipeline.config import PIPELINE_AVAILABLE, SCHEMAS_AVAILABLE, SPECS_SERVICE_AVAILABLE, get_latest_validated_spec, get_pipeline_model_config, get_spec, memory_schemas, memory_service
from app.llm.critical_pipeline.job_classification import JobKind, classify_job_kind
from sqlalchemy.orm import Session
from typing import Any, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


def _token(text: str) -> str:
    return _sse("token", text)

def _done(**fields) -> str:
    return _sse("done", **fields)

def _save_to_memory(
    db: Session,
    project_id: int,
    content: str,
    provider: str,
    model: str,
) -> None:
    if memory_service and memory_schemas:
        try:
            memory_service.create_message(
                db,
                memory_schemas.MessageCreate(
                    project_id=project_id,
                    role="assistant",
                    content=content,
                    provider=provider,
                    model=model,
                ),
            )
        except Exception as e:
            logger.warning("[critical_pipeline] Failed to save to memory: %s", e)

async def generate_critical_pipeline_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[Any] = None,
    conversation_id: Optional[str] = None,
    spec_id: Optional[str] = None,
    spec_hash: Optional[str] = None,
    job_id: Optional[str] = None,
    segment_context: Optional[dict] = None,
):
    """Generate SSE stream for Critical Pipeline execution."""
    response_parts = []
    model_cfg = get_pipeline_model_config()
    pipeline_provider = model_cfg["provider"]
    pipeline_model = model_cfg["model"]

    # v5.5 PHASE 3C: Needle-based model selection — override for segments
    _needle_model_override = None
    if segment_context:
        try:
            from app.llm.critical_pipeline.needle_model_selector import select_model_for_segment
            _needle_model_override = select_model_for_segment(
                segment_context=segment_context,
                grounding_data=segment_context.get("_grounding_data"),
                default_config={"provider": pipeline_provider, "model": pipeline_model},
            )
            pipeline_provider = _needle_model_override["provider"]
            pipeline_model = _needle_model_override["model"]
        except (ImportError, Exception) as _nm_err:
            logger.debug("[critical_pipeline] Needle model selector unavailable: %s", _nm_err)

    def _emit(text):
        response_parts.append(text)
        return _token(text)

    try:
        yield _emit("\u2699\ufe0f **Critical Pipeline**\n\n")

        # =================================================================
        # Validation
        # =================================================================
        if not PIPELINE_AVAILABLE:
            msg = (
                "\u274c **Pipeline modules not available.**\n\n"
                "The high-stakes pipeline modules (app.llm.pipeline.*) failed to import.\n"
            )
            yield _emit(msg)
            if trace:
                trace.finalize(success=False, error_message="Pipeline modules not available")
            yield _done(
                provider=pipeline_provider, model=pipeline_model,
                total_length=sum(len(p) for p in response_parts),
            )
            return

        if not SCHEMAS_AVAILABLE:
            yield _emit("\u274c **Schema imports failed.** Check backend logs.\n")
            yield _done(
                provider=pipeline_provider, model=pipeline_model,
                total_length=sum(len(p) for p in response_parts),
            )
            return

        # =================================================================
        # Step 1: Load validated spec
        # =================================================================
        yield _emit("\ud83d\udccb **Loading validated spec...**\n")

        db_spec = None
        if spec_id and SPECS_SERVICE_AVAILABLE and get_spec:
            try:
                db_spec = get_spec(db, spec_id)
            except Exception as e:
                logger.warning("[critical_pipeline] Failed to get spec by ID: %s", e)

        if not db_spec and SPECS_SERVICE_AVAILABLE and get_latest_validated_spec:
            try:
                db_spec = get_latest_validated_spec(db, project_id)
            except Exception as e:
                logger.warning("[critical_pipeline] Failed to get latest validated spec: %s", e)

        if not db_spec:
            yield _emit(
                "\u274c **No validated spec found.**\n\n"
                "Please complete Spec Gate validation first:\n"
                "1. Describe what you want to build\n"
                "2. Say `Astra, command: how does that look all together`\n"
                "3. Say `Astra, command: critical architecture` to validate\n"
                "4. Once validated, retry `run critical pipeline`\n"
            )
            yield _done(
                provider=pipeline_provider, model=pipeline_model,
                total_length=sum(len(p) for p in response_parts),
            )
            return

        spec_id = db_spec.spec_id
        spec_hash = db_spec.spec_hash
        spec_json = db_spec.content_json
        spec_markdown = db_spec.content_markdown

        try:
            spec_data = json.loads(spec_json) if isinstance(spec_json, str) else (spec_json or {})
        except Exception:
            spec_data = {}

        yield _emit(f"\u2705 Spec loaded: `{spec_id[:16]}...`\n")

        # =================================================================
        # v5.1: Segment context info (UI visibility)
        # =================================================================
        if segment_context:
            _seg_id = segment_context.get("segment_id", "unknown")
            _seg_deps = segment_context.get("dependencies", [])
            _seg_files = segment_context.get("file_scope", [])
            _seg_reqs = segment_context.get("requirements", [])
            _seg_exposes = segment_context.get("exposes")
            _seg_consumes = segment_context.get("consumes")

            seg_info = f"\ud83e\udde9 **Segment:** `{_seg_id}`\n"
            if _seg_deps:
                seg_info += f"   \u2514\u2500 Dependencies: {', '.join(f'`{d}`' for d in _seg_deps)}\n"
            if _seg_files:
                seg_info += f"   \u2514\u2500 Files in scope: {len(_seg_files)}\n"
            if _seg_reqs:
                seg_info += f"   \u2514\u2500 Requirements: {len(_seg_reqs)}\n"
            if _seg_exposes:
                seg_info += f"   \u2514\u2500 Exposes: interface contracts for downstream\n"
            if _seg_consumes:
                seg_info += f"   \u2514\u2500 Consumes: interface contracts from upstream\n"
            yield _emit(seg_info)
        # v5.5 PHASE 3C: Show model selection reason
        if _needle_model_override:
            _nm_tier = _needle_model_override.get("tier", "?")
            _nm_reason = _needle_model_override.get("reason", "")
            yield _emit(f"🧠 **Model tier:** `{_nm_tier}` → `{pipeline_model}`\n")
            if _nm_reason:
                yield _emit(f"   └─ {_nm_reason}\n")

        # =================================================================
        # Mechanical guard: pending_evidence / blocked / error
        # =================================================================
        validation_status = spec_data.get("validation_status", "validated")

        if validation_status == "pending_evidence":
            # v5.0: Softened from hard block to warning. SpecGate v4.0+ fulfils
            # its own ERs, so pending_evidence should no longer occur. If a legacy
            # spec arrives with this status, warn but proceed — the Critical
            # Pipeline's own evidence_loop in high_stakes.py can attempt to fulfil
            # remaining ERs during architecture generation. The old hard block
            # caused a deadlock where SpecGate and Critical Pipeline each told
            # the user to go to the other.
            warn_msg = (
                "\n\u26a0\ufe0f **Warning: Spec has unfulfilled evidence requirements**\n\n"
                "SpecGate marked this spec as `pending_evidence`. This usually means "
                "evidence fulfilment was partially unsuccessful. Proceeding anyway \u2014 "
                "the architecture stage will attempt to gather remaining evidence.\n\n"
            )
            logger.warning(
                "[critical_pipeline] v5.0 SOFT GUARD: pending_evidence (proceeding), spec_id=%s",
                spec_id,
            )
            yield _emit(warn_msg)

        if validation_status in ("blocked", "error", "needs_clarification"):
            yield _emit(
                f"\n\ud83d\udeab **BLOCKED: Spec status is `{validation_status}`**\n\n"
                "Please resolve the spec issues and re-validate before retrying.\n"
            )
            yield _done(
                provider=pipeline_provider, model=pipeline_model,
                total_length=sum(len(p) for p in response_parts),
                blocked=True, blocked_reason=validation_status, spec_id=spec_id,
            )
            if trace:
                trace.finalize(success=False, error_message=f"Spec status is {validation_status}")
            return

        logger.info("[critical_pipeline] validation_status=%s \u2014 proceeding", validation_status)

        # =================================================================
        # Step 1a: Check for segmented spec (v5.1)
        # =================================================================
        # If SpecGate decomposed this job into segments, the critical pipeline
        # should NOT process the parent spec as a blob. Redirect to segment loop.
        _spec_context = spec_data.get("context", {})
        # v5.4: Check for MULTI-segment specs only. Single-segment manifests
        # (Phase 1 always-manifest) should flow through normally.
        _total_segs = (
            spec_data.get("total_segments", 0)
            or _spec_context.get("total_segments", 0)
        )
        _is_multi_segmented = (
            _spec_context.get("segmented", False)
            or _total_segs > 1
        )

        if _is_multi_segmented and not segment_context:
            # This spec has segments but was called directly (not via segment loop).
            # The segment loop passes segment_context when calling per-segment.
            seg_msg = (
                f"\n\u26a0\ufe0f **This spec has been segmented into {_total_segs} segments.**\n\n"
                f"The critical pipeline should process each segment individually, "
                f"not the parent spec as a single blob.\n\n"
                f"Please use: **'Astra, command: run segments'** to execute "
                f"all segments through the pipeline in dependency order.\n"
            )
            logger.warning(
                "[critical_pipeline] v5.1 SEGMENT GUARD: spec %s has %d segments "
                "but was called directly — redirecting user to segment loop",
                spec_id, _total_segs,
            )
            print(f"[DEBUG] [critical_pipeline] v5.1 SEGMENT GUARD: {_total_segs} segments detected, blocking direct execution")
            yield _emit(seg_msg)
            yield _done(
                provider=pipeline_provider, model=pipeline_model,
                total_length=sum(len(p) for p in response_parts),
            )
            if trace:
                trace.finalize(success=False, error_message=f"Segmented spec ({_total_segs} segments) — use 'run segments' command")
            return

        # =================================================================
        # Step 1b: Classify job type
        # =================================================================
        job_kind = classify_job_kind(spec_data, message)
        yield _emit(f"\ud83c\udff7\ufe0f **Job Type:** `{job_kind}`\n")

        # =================================================================
        # MICRO_EXECUTION PATH
        # =================================================================
        if job_kind == JobKind.MICRO_EXECUTION:
            async for chunk in _handle_micro(
                spec_data, message, spec_id, job_id, project_id, db,
                trace, pipeline_provider, pipeline_model, response_parts,
            ):
                yield chunk
            return

        # =================================================================
        # SCAN_ONLY PATH
        # =================================================================
        if job_kind == JobKind.SCAN_ONLY:
            async for chunk in _handle_scan(
                spec_data, message, spec_id, job_id, project_id, db,
                trace, pipeline_provider, pipeline_model, response_parts,
            ):
                yield chunk
            return

        # =================================================================
        # ARCHITECTURE PATH
        # =================================================================
        async for chunk in _handle_architecture(
            spec_data, message, spec_id, spec_hash, spec_json, spec_markdown,
            job_id, job_kind, project_id, db, trace, conversation_id,
            pipeline_provider, pipeline_model, response_parts,
            segment_context=segment_context,
        ):
            yield chunk

    except Exception as e:
        logger.exception("[critical_pipeline] Stream failed: %s", e)
        if trace:
            trace.finalize(success=False, error_message=str(e))
        yield _sse("error", error=str(e))
