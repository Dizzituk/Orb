# FILE: app/llm/spec_gate_stream.py
"""
Spec Gate streaming handler for ASTRA command flow.

v2.5 (2026-02-24): Refactored — persistence extracted to _sg_stream_persist.py
v2.4 (2026-02-03): POT SPEC PERSISTENCE FIX
v2.3 (2026-02-01): VISION CONTEXT FLOW FIX
v2.2 (2026-01-20): Caller-side persistence (v1.5 SpecGate support)
v2.1 (2026-01-04): Blocking Validation Support
v2.0: Original implementation with Weaver spec validation.
"""

from __future__ import annotations

import json
import logging
import asyncio
from typing import Optional, Any, AsyncGenerator, Dict

from sqlalchemy.orm import Session
from app.llm._spec_gate_stream_utils_2 import (
    _USE_GROUNDED_SPEC_GATE,
    _get_weaver_vision_context_from_flow,
    _load_latest_weaver_spec_json,
    _resolve_spec_gate_model,
    _safe_json_event,
)
from app.llm._sg_stream_persist import persist_validated_spec

logger = logging.getLogger(__name__)

# Import Spec Gate v2
try:
    from app.pot_spec.spec_gate_v2 import run_spec_gate_v2, SpecGateResult
    _SPEC_GATE_V2_AVAILABLE = True
except Exception as e:
    _SPEC_GATE_V2_AVAILABLE = False
    run_spec_gate_v2 = None
    SpecGateResult = None
    logger.warning("[spec_gate_stream] spec_gate_v2 not available: %s", e)

# Import Spec Gate Grounded (Contract v1)
import os

try:
    from app.pot_spec.spec_gate_grounded import run_spec_gate_grounded
    _SPEC_GATE_GROUNDED_AVAILABLE = True
except Exception as e:
    _SPEC_GATE_GROUNDED_AVAILABLE = False
    run_spec_gate_grounded = None
    logger.warning("[spec_gate_stream] spec_gate_grounded not available: %s", e)

# Flow state management (optional)
try:
    from app.llm.spec_flow_state import (
        get_active_flow,
        advance_to_spec_gate_questions,
        advance_to_spec_validated,
        advance_to_spec_segmented,
        cancel_flow,
    )
    _FLOW_STATE_AVAILABLE = True
except Exception:
    _FLOW_STATE_AVAILABLE = False
    get_active_flow = None
    advance_to_spec_gate_questions = None
    advance_to_spec_validated = None
    advance_to_spec_segmented = None
    cancel_flow = None

# Job service (optional)
try:
    from app.jobs.service import get_active_job_for_project
except Exception:
    get_active_job_for_project = None

# Memory service (optional)
try:
    from app.memory import service as memory_service
    from app.memory import schemas as memory_schemas
except Exception:
    memory_service = None
    memory_schemas = None

# Audit logger (optional)
try:
    from app.llm.audit_logger import RoutingTrace
except Exception:
    RoutingTrace = None


async def generate_spec_gate_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[Any] = None,
    conversation_id: Optional[str] = None,
    is_clarification_response: bool = False,
) -> AsyncGenerator[str, None]:
    """Generate SSE stream for Spec Gate validation (v2.1+)."""
    response_parts: list[str] = []

    spec_gate_provider, spec_gate_model = _resolve_spec_gate_model()
    logger.info(
        "[spec_gate_stream] provider=%s, model=%s, project=%s",
        spec_gate_provider, spec_gate_model, project_id,
    )

    try:
        if not _SPEC_GATE_V2_AVAILABLE or not run_spec_gate_v2:
            msg = "⚠️ Spec Gate v2 is not available. Please ensure app.pot_spec.spec_gate_v2 imports cleanly.\n"
            yield _safe_json_event({"type": "token", "content": msg})
            response_parts.append(msg)
            yield _safe_json_event({"type": "done", "provider": spec_gate_provider, "model": spec_gate_model, "total_length": sum(len(p) for p in response_parts)})
            return

        if not spec_gate_provider or not spec_gate_model:
            msg = "❌ Spec Gate model config missing. Check ENV/stage_models.\n"
            yield _safe_json_event({"type": "token", "content": msg})
            response_parts.append(msg)
            yield _safe_json_event({"type": "done", "provider": spec_gate_provider, "model": spec_gate_model, "total_length": sum(len(p) for p in response_parts)})
            return

        # Flow state
        existing_flow = None
        if _FLOW_STATE_AVAILABLE and get_active_flow:
            try:
                existing_flow = get_active_flow(project_id)
            except Exception as e:
                logger.debug("[spec_gate_stream] get_active_flow failed: %s", e)

        # Job context
        job_id = None
        clarification_round = 1

        if existing_flow:
            job_id = getattr(existing_flow, "job_id", None)
            clarification_round = (getattr(existing_flow, "clarification_round", 0) or 0) + 1
        else:
            if get_active_job_for_project:
                try:
                    active_job = get_active_job_for_project(db, project_id)
                    if active_job:
                        job_id = active_job.id
                except Exception:
                    pass

        if not job_id:
            import uuid
            job_id = f"sg-{uuid.uuid4().hex[:8]}"

        # Load Weaver spec
        weaver_spec_json, weaver_prov = _load_latest_weaver_spec_json(db, project_id)
        constraints_hint: dict = {"project_id": project_id}

        # v2.3: Vision context
        vision_context = _get_weaver_vision_context_from_flow(project_id)
        if vision_context:
            constraints_hint["vision_context"] = vision_context

        if weaver_spec_json:
            constraints_hint["weaver_spec_json"] = weaver_spec_json
            constraints_hint.update({k: v for k, v in weaver_prov.items() if v})

            if weaver_spec_json.get("source") == "weaver_simple":
                job_desc = weaver_spec_json.get("job_description", "")
                constraints_hint["weaver_job_description_text"] = job_desc
                constraints_hint["weaver_source"] = "weaver_simple"
                found_msg = f"✓ Found Weaver job description ({len(job_desc)} chars)\n  Source: Simple Weaver (v3.0)\n\n"
            else:
                content_verbatim = weaver_spec_json.get("content_verbatim") or weaver_spec_json.get("metadata", {}).get("content_verbatim")
                location = weaver_spec_json.get("location") or weaver_spec_json.get("metadata", {}).get("location")
                found_msg = "✓ Found Weaver spec to validate.\n"
                if content_verbatim:
                    found_msg += f"  Content: \"{str(content_verbatim)[:50]}{'...' if len(str(content_verbatim)) > 50 else ''}\"\n"
                if location:
                    found_msg += f"  Location: {location}\n"
                found_msg += "\n"

            yield _safe_json_event({"type": "token", "content": found_msg})
            response_parts.append(found_msg)
        else:
            warn = "⚠️ No Weaver output found.\nSpec Gate may drift if it must infer the job from raw chat.\nConsider running Weaver first: `how does that look all together`\n\n"
            yield _safe_json_event({"type": "token", "content": warn})
            response_parts.append(warn)

        validating_msg = f"Validating spec (Round {clarification_round}/3)...\n\n"
        yield _safe_json_event({"type": "token", "content": validating_msg})
        response_parts.append(validating_msg)
        await asyncio.sleep(0.05)

        user_intent = (message or "").strip()

        # v3.0: Choose grounded or v2
        use_grounded = _USE_GROUNDED_SPEC_GATE and _SPEC_GATE_GROUNDED_AVAILABLE and run_spec_gate_grounded

        if use_grounded:
            version_msg = "🔬 Using **SpecGate v3.0** (grounded, evidence-based, deterministic)\n\n"
            yield _safe_json_event({"type": "token", "content": version_msg})
            response_parts.append(version_msg)

        # Execute SpecGate
        try:
            if use_grounded:
                result = await run_spec_gate_grounded(
                    db=db, job_id=job_id, user_intent=user_intent,
                    provider_id=spec_gate_provider, model_id=spec_gate_model,
                    project_id=project_id, constraints_hint=constraints_hint,
                    spec_version=clarification_round,
                )
            else:
                result = await run_spec_gate_v2(
                    db=db, job_id=job_id, user_intent=user_intent,
                    provider_id=spec_gate_provider, model_id=spec_gate_model,
                    project_id=project_id, constraints_hint=constraints_hint,
                    spec_version=clarification_round,
                )
        except Exception as e:
            error_msg = f"❌ Spec Gate failed: {e}\n"
            yield _safe_json_event({"type": "token", "content": error_msg})
            response_parts.append(error_msg)
            if trace:
                try: trace.finalize(success=False, error_message=str(e))
                except Exception: pass
            yield _safe_json_event({"type": "done", "provider": spec_gate_provider, "model": spec_gate_model, "total_length": sum(len(p) for p in response_parts)})
            return

        # --- Handle result branches ---
        # Hard stop
        if getattr(result, "hard_stopped", False):
            reason = getattr(result, "hard_stop_reason", None) or "Spec Gate has blocked this request."
            yield _safe_json_event({"type": "token", "content": f"🛑 **HARD STOP**\n\n{reason}\n"})
            if _FLOW_STATE_AVAILABLE and cancel_flow:
                try: cancel_flow(project_id)
                except Exception: pass
            yield _safe_json_event({"type": "done", "provider": spec_gate_provider, "model": spec_gate_model, "total_length": sum(len(p) for p in response_parts), "hard_stopped": True})
            return

        open_q = getattr(result, "open_questions", None) or []
        blocking_issues = getattr(result, "blocking_issues", None) or []
        validation_status = getattr(result, "validation_status", "pending")
        ready = bool(getattr(result, "ready_for_pipeline", False))

        # Branch: blocking issues
        if blocking_issues and not ready:
            async for chunk in _stream_blocking(result, open_q, clarification_round, project_id, job_id, spec_gate_provider, spec_gate_model, response_parts):
                yield chunk

        # Branch: non-blocking questions
        elif open_q and not ready:
            async for chunk in _stream_questions(result, open_q, clarification_round, project_id, job_id, response_parts):
                yield chunk

        # Branch: segmented
        elif validation_status == "segmented":
            async for chunk in _stream_segmented(result, db, project_id, job_id, clarification_round, spec_gate_provider, spec_gate_model, response_parts):
                yield chunk

        # Branch: ready (validated)
        else:
            async for chunk in _stream_ready(result, db, project_id, job_id, clarification_round, spec_gate_provider, spec_gate_model, validation_status, blocking_issues, ready, response_parts):
                yield chunk

        # Persist assistant message
        if memory_service and memory_schemas:
            try:
                memory_service.create_message(
                    db,
                    memory_schemas.MessageCreate(
                        project_id=project_id, role="assistant",
                        content="".join(response_parts),
                        provider=spec_gate_provider, model=spec_gate_model,
                    ),
                )
            except Exception as e:
                logger.warning("[spec_gate_stream] Failed to save message: %s", e)

        yield _safe_json_event({
            "type": "done", "provider": spec_gate_provider, "model": spec_gate_model,
            "total_length": sum(len(p) for p in response_parts),
            "spec_id": getattr(result, "spec_id", None) if "result" in dir() else None,
            "db_persisted": False,
            "validation_status": validation_status if "validation_status" in dir() else "unknown",
        })

    except Exception as e:
        logger.exception("[spec_gate_stream] Stream failed: %s", e)
        if trace:
            try: trace.finalize(success=False, error_message=str(e))
            except Exception: pass
        yield _safe_json_event({"type": "error", "error": str(e)})


# =========================================================================
# Stream branch helpers
# =========================================================================

async def _stream_blocking(result, open_q, clarification_round, project_id, job_id, provider, model, parts):
    """Yield SSE events for blocking validation failure."""
    blocking_issues = getattr(result, "blocking_issues", None) or []
    header = f"⛔ **BLOCKING VALIDATION FAILED** (Round {clarification_round}/3)\n\n"
    yield _safe_json_event({"type": "token", "content": header})
    parts.append(header)

    explain = "The spec cannot proceed to the pipeline because critical information is missing.\nPlease provide the following:\n\n"
    yield _safe_json_event({"type": "token", "content": explain})
    parts.append(explain)

    for i, q in enumerate(open_q, 1):
        line = f"**{i}.** {q}\n\n"
        yield _safe_json_event({"type": "token", "content": line})
        parts.append(line)
        await asyncio.sleep(0.01)

    guidance = (
        "---\n**How to resolve:**\n1. Answer the questions above in your next message\n"
        f"2. Then say **'Astra, command: critical architecture'** to retry validation\n\n"
        f"Round {clarification_round}/3 - Round 3 will finalize even with gaps (not recommended).\n"
    )
    yield _safe_json_event({"type": "token", "content": guidance})
    parts.append(guidance)

    if _FLOW_STATE_AVAILABLE and advance_to_spec_gate_questions:
        try:
            advance_to_spec_gate_questions(
                project_id=project_id, job_id=job_id,
                spec_id=getattr(result, "spec_id", "") or "",
                spec_hash=getattr(result, "spec_hash", "") or "",
                questions=open_q, clarification_round=clarification_round,
            )
        except Exception:
            pass

    yield _safe_json_event({
        "type": "spec_blocked", "spec_id": getattr(result, "spec_id", None),
        "blocking_issues": blocking_issues, "questions": open_q, "round": clarification_round,
    })


async def _stream_questions(result, open_q, clarification_round, project_id, job_id, parts):
    """Yield SSE events for non-blocking clarification questions."""
    header = f"❓ **Clarification Needed** (Round {clarification_round}/3)\n\n"
    yield _safe_json_event({"type": "token", "content": header})
    parts.append(header)

    for q in open_q:
        line = f"• {q}\n\n"
        yield _safe_json_event({"type": "token", "content": line})
        parts.append(line)
        await asyncio.sleep(0.01)

    followup = "Please answer these questions, then say **'Astra, command: critical architecture'** again.\n"
    yield _safe_json_event({"type": "token", "content": followup})
    parts.append(followup)

    if _FLOW_STATE_AVAILABLE and advance_to_spec_gate_questions:
        try:
            advance_to_spec_gate_questions(
                project_id=project_id, job_id=job_id,
                spec_id=getattr(result, "spec_id", "") or "",
                spec_hash=getattr(result, "spec_hash", "") or "",
                questions=open_q, clarification_round=clarification_round,
            )
        except Exception:
            pass


async def _stream_segmented(result, db, project_id, job_id, clarification_round, provider, model, parts):
    """Yield SSE events for segmented job result."""
    seg_data = (getattr(result, 'grounding_data', None) or {}).get('segmentation', {})
    seg_count = seg_data.get('total_segments', 0)
    seg_ids = seg_data.get('segment_ids', [])

    seg_header = f"🔀 **Job Segmented** — {seg_count} segments ready for execution\n\n"
    yield _safe_json_event({"type": "token", "content": seg_header})
    parts.append(seg_header)

    seg_detail = f"SpecGate has decomposed this job into **{seg_count} segments** that will be executed in dependency order:\n\n"
    for i, sid in enumerate(seg_ids, 1):
        seg_detail += f"  {i}. `{sid}`\n"
    seg_detail += "\n"
    yield _safe_json_event({"type": "token", "content": seg_detail})
    parts.append(seg_detail)

    # Persist
    db_persisted = getattr(result, "db_persisted", False)
    if not db_persisted:
        db_persisted, _ = persist_validated_spec(
            db=db, project_id=project_id, result=result,
            clarification_round=clarification_round,
            spec_gate_provider=provider, spec_gate_model=model,
            job_id=job_id, title_suffix=" — Segmented",
        )

    db_msg = "💾 Parent spec persisted to database.\n\n" if db_persisted else "⚠️ Parent spec NOT persisted to database.\n\n"
    yield _safe_json_event({"type": "token", "content": db_msg})
    parts.append(db_msg)

    spot_md = getattr(result, "spot_markdown", None)
    if spot_md:
        yield _safe_json_event({"type": "token", "content": spot_md})
        parts.append(spot_md)

    if _FLOW_STATE_AVAILABLE and advance_to_spec_segmented:
        try:
            advance_to_spec_segmented(
                project_id=project_id,
                spec_id=getattr(result, "spec_id", "") or "",
                spec_hash=getattr(result, "spec_hash", "") or "",
                job_id=job_id, total_segments=seg_count,
                spec_version=clarification_round,
            )
        except Exception:
            pass

    next_step = f"\n🚀 **Segments Ready.**\nSay **'Astra, command: run segments'** to execute all {seg_count} segments through the pipeline in dependency order.\n"
    yield _safe_json_event({"type": "token", "content": next_step})
    parts.append(next_step)

    yield _safe_json_event({
        "type": "spec_segmented", "spec_id": getattr(result, "spec_id", None),
        "spec_hash": getattr(result, "spec_hash", None), "job_id": job_id,
        "total_segments": seg_count, "segment_ids": seg_ids,
        "db_persisted": db_persisted, "validation_status": "segmented",
    })


async def _stream_ready(result, db, project_id, job_id, clarification_round, provider, model, validation_status, blocking_issues, ready, parts):
    """Yield SSE events for validated/ready spec."""
    is_round3_with_issues = ready and validation_status == "validated_with_issues" and blocking_issues

    if is_round3_with_issues:
        header = "⚠️ **Spec Finalized (Round 3) - With Unresolved Items**\n\n"
        warn = "**Note:** This spec was finalized on Round 3 but contains unresolved questions.\nSpecGate did NOT fill in assumptions - gaps are explicitly marked.\n\n"
        yield _safe_json_event({"type": "token", "content": header})
        parts.append(header)
        yield _safe_json_event({"type": "token", "content": warn})
        parts.append(warn)
    elif validation_status == "pending_evidence":
        header = "🔬 **Spec Ready — Evidence Required Before Architecture**\n\n"
        yield _safe_json_event({"type": "token", "content": header})
        parts.append(header)
        note = "**Note:** This spec contains CRITICAL `EVIDENCE_REQUEST`(s) that must be fulfilled before the Critical Pipeline can produce a grounded architecture.\n\n"
        yield _safe_json_event({"type": "token", "content": note})
        parts.append(note)
    else:
        header = "✅ **Spec Validated - SPoT Generated**\n\n"
        yield _safe_json_event({"type": "token", "content": header})
        parts.append(header)

    # Persist
    db_persisted = getattr(result, "db_persisted", False)
    persist_error = None
    if not db_persisted:
        db_persisted, persist_error = persist_validated_spec(
            db=db, project_id=project_id, result=result,
            clarification_round=clarification_round,
            spec_gate_provider=provider, spec_gate_model=model,
            job_id=job_id,
        )

    if db_persisted:
        db_msg = "💾 Spec persisted to database.\n\n"
    else:
        db_msg = "⚠️ Spec NOT persisted to database - may not survive restart.\n"
        if persist_error:
            db_msg += f"   Reason: {persist_error}\n"
        db_msg += "\n"
    yield _safe_json_event({"type": "token", "content": db_msg})
    parts.append(db_msg)

    spot_md = getattr(result, "spot_markdown", None)
    if spot_md:
        yield _safe_json_event({"type": "token", "content": spot_md})
        parts.append(spot_md)

    meta = (
        f"\n\n---\n**Spec ID:** `{getattr(result,'spec_id',None)}`\n"
        f"**Spec Hash:** `{str(getattr(result,'spec_hash','') or '')[:16]}...`\n"
        f"**Round:** {clarification_round}\n**Status:** {validation_status}\n"
    )
    yield _safe_json_event({"type": "token", "content": meta})
    parts.append(meta)

    if validation_status == "pending_evidence":
        next_step = "\n🔬 **Evidence Required Before Architecture.**\nSay **'Astra, command: run critical pipeline'** to proceed.\n"
    else:
        next_step = "\n🚀 **Spec Complete.**\nSay **'Astra, command: run critical pipeline'** to proceed to architecture generation.\n"
    yield _safe_json_event({"type": "token", "content": next_step})
    parts.append(next_step)

    if _FLOW_STATE_AVAILABLE and advance_to_spec_validated:
        try:
            advance_to_spec_validated(
                project_id=project_id,
                spec_id=getattr(result, "spec_id", "") or "",
                spec_hash=getattr(result, "spec_hash", "") or "",
                spec_version=clarification_round,
            )
        except Exception:
            pass

    yield _safe_json_event({
        "type": "spec_ready", "spec_id": getattr(result, "spec_id", None),
        "spec_hash": getattr(result, "spec_hash", None), "job_id": job_id,
        "db_persisted": db_persisted, "validation_status": validation_status,
    })


__all__ = ["generate_spec_gate_stream"]
