# FILE: app/llm/spec_gate_stream.py
"""
Spec Gate streaming handler for ASTRA command flow.

v2.2 (2026-01-20): Caller-side persistence (v1.5 SpecGate support)
- Persist spec to DB after validation (caller responsibility, not SpecGate)
- SpecGate remains read-only in runtime
- Actual persistence status reflected in events

v2.1 (2026-01-04): Blocking Validation Support
- Shows blocking issues prominently when validation fails
- Clear distinction between blocking vs informational questions
- Better user guidance on how to resolve blocking issues

v2.0: Original implementation with Weaver spec validation.
"""

from __future__ import annotations

import json
import logging
import asyncio
from typing import Optional, Any, AsyncGenerator, Dict

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Centralized stage model selection (ENV-driven)
try:
    from app.llm.stage_models import get_spec_gate_config
except ImportError:
    get_spec_gate_config = None

# Import Spec Gate v2
try:
    from app.pot_spec.spec_gate_v2 import run_spec_gate_v2, SpecGateResult
    _SPEC_GATE_V2_AVAILABLE = True
except Exception as e:
    _SPEC_GATE_V2_AVAILABLE = False
    run_spec_gate_v2 = None
    SpecGateResult = None
    logger.warning("[spec_gate_stream] spec_gate_v2 module not available: %s", e)

# Import Spec Gate Grounded (Contract v1)
import os
_USE_GROUNDED_SPEC_GATE = os.getenv("USE_GROUNDED_SPEC_GATE", "1") == "1"

try:
    from app.pot_spec.spec_gate_grounded import run_spec_gate_grounded
    _SPEC_GATE_GROUNDED_AVAILABLE = True
except Exception as e:
    _SPEC_GATE_GROUNDED_AVAILABLE = False
    run_spec_gate_grounded = None
    logger.warning("[spec_gate_stream] spec_gate_grounded module not available: %s", e)

# Import Spec Gate Persistence (v1.5 - persist after validation)
try:
    from app.pot_spec.spec_gate_persistence import (
        persist_spec,
        build_spec_schema,
        safe_summary_from_objective,
        write_spec_artifacts,
        compute_spec_hash,
    )
    _SPEC_PERSISTENCE_AVAILABLE = True
except Exception as e:
    _SPEC_PERSISTENCE_AVAILABLE = False
    persist_spec = None
    build_spec_schema = None
    safe_summary_from_objective = None
    write_spec_artifacts = None
    compute_spec_hash = None
    logger.warning("[spec_gate_stream] spec_gate_persistence module not available: %s", e)

# Flow state management (optional)
try:
    from app.llm.spec_flow_state import (
        get_active_flow,
        advance_to_spec_gate_questions,
        advance_to_spec_validated,
        cancel_flow,
    )
    _FLOW_STATE_AVAILABLE = True
except Exception:
    _FLOW_STATE_AVAILABLE = False
    get_active_flow = None
    advance_to_spec_gate_questions = None
    advance_to_spec_validated = None
    cancel_flow = None

# Job service (optional)
try:
    from app.jobs.service import get_active_job_for_project
except Exception:
    get_active_job_for_project = None

# Spec service for Weaver draft
try:
    from app.specs import service as specs_service
except Exception:
    specs_service = None

# Memory service (optional)
try:
    from app.memory import service as memory_service
except Exception:
    memory_service = None

# Audit logger (optional)
try:
    from app.llm.audit_logger import RoutingTrace
except Exception:
    RoutingTrace = None


def _safe_json_event(payload: Dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _resolve_spec_gate_model() -> tuple[str, str]:
    if not get_spec_gate_config:
        return "", ""
    cfg = get_spec_gate_config()
    return (cfg.provider or "", cfg.model or "")


def _get_weaver_job_description_from_flow(project_id: int) -> Optional[str]:
    """Get simple Weaver job description from flow state (v3.0)."""
    if not _FLOW_STATE_AVAILABLE or not get_active_flow:
        return None
    try:
        flow = get_active_flow(project_id)
        if flow:
            return getattr(flow, 'weaver_job_description', None)
    except Exception as e:
        logger.debug("[spec_gate_stream] get_active_flow failed: %s", e)
    return None


def _load_latest_weaver_spec_json(db: Session, project_id: int) -> tuple[Optional[dict], dict]:
    """Load Weaver output - checks flow state first (v3.0), then DB."""
    
    # v3.0: First check flow state for simple Weaver job description
    job_description = _get_weaver_job_description_from_flow(project_id)
    if job_description:
        logger.info("[spec_gate_stream] Found Weaver job description in flow state (%d chars)", len(job_description))
        # Return job description wrapped in a format Spec Gate can use
        return {
            "job_description": job_description,
            "source": "weaver_simple",
            "title": "Job Description from Weaver",
        }, {"weaver_source": "flow_state"}
    
    # Fallback: Load from DB (v2.x behaviour)
    if not specs_service:
        return None, {}

    try:
        spec_rec = specs_service.get_latest_draft_spec(db, project_id)
        if not spec_rec:
            return None, {}

        content_json = getattr(spec_rec, "content_json", None)
        if not content_json:
            return None, {}

        try:
            weaver_spec = json.loads(content_json) if isinstance(content_json, str) else content_json
        except Exception as e:
            logger.warning("[spec_gate_stream] Failed to parse Weaver content_json: %s", e)
            return None, {}

        provenance = {
            "weaver_spec_id": getattr(spec_rec, "spec_id", None),
            "weaver_spec_hash": getattr(spec_rec, "spec_hash", None),
            "weaver_spec_version": getattr(spec_rec, "spec_version", None),
        }
        return (weaver_spec if isinstance(weaver_spec, dict) else None), provenance
    except Exception as e:
        logger.warning("[spec_gate_stream] Could not load latest Weaver draft spec: %s", e)
        return None, {}


async def generate_spec_gate_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[Any] = None,
    conversation_id: Optional[str] = None,
    is_clarification_response: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Generate SSE stream for Spec Gate validation with blocking validation support (v2.1).

    Flow:
    1) Resolve provider/model from stage_models (ENV-driven)
    2) Determine job_id + clarification_round
    3) Load latest Weaver draft spec JSON from DB
    4) Call run_spec_gate_v2() with constraints_hint["weaver_spec_json"]
    5) Stream either:
       - BLOCKING issues (if validation failed) with clear resolution guidance
       - clarification questions (non-blocking)
       - SPoT markdown (if ready)
       - hard stop (if blocked)
    """
    response_parts: list[str] = []

    spec_gate_provider, spec_gate_model = _resolve_spec_gate_model()
    logger.info(
        "[spec_gate_stream] Using provider=%s, model=%s, project_id=%s",
        spec_gate_provider, spec_gate_model, project_id
    )

    try:
        if not _SPEC_GATE_V2_AVAILABLE or not run_spec_gate_v2:
            msg = "⚠️ Spec Gate v2 is not available. Please ensure app.pot_spec.spec_gate_v2 imports cleanly.\n"
            yield _safe_json_event({"type": "token", "content": msg})
            response_parts.append(msg)
            yield _safe_json_event({
                "type": "done",
                "provider": spec_gate_provider,
                "model": spec_gate_model,
                "total_length": sum(len(p) for p in response_parts),
            })
            return

        if not spec_gate_provider or not spec_gate_model:
            msg = "❌ Spec Gate model config missing. Check ENV/stage_models for Spec Gate provider/model.\n"
            yield _safe_json_event({"type": "token", "content": msg})
            response_parts.append(msg)
            yield _safe_json_event({
                "type": "done",
                "provider": spec_gate_provider,
                "model": spec_gate_model,
                "total_length": sum(len(p) for p in response_parts),
            })
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

        if weaver_spec_json:
            constraints_hint["weaver_spec_json"] = weaver_spec_json
            constraints_hint.update({k: v for k, v in weaver_prov.items() if v})
            
            # v3.0: Check if this is from simple Weaver (job_description) or v2.x (full spec)
            if weaver_spec_json.get("source") == "weaver_simple":
                # Simple Weaver output - job description text
                # Add dedicated key so Spec Gate knows this is raw text, not JSON spec
                job_desc = weaver_spec_json.get("job_description", "")
                constraints_hint["weaver_job_description_text"] = job_desc  # Dedicated key for v3.0
                constraints_hint["weaver_source"] = "weaver_simple"
                
                found_msg = f"✓ Found Weaver job description ({len(job_desc)} chars)\n"
                found_msg += "  Source: Simple Weaver (v3.0)\n"
                # Show preview
                preview = job_desc[:150].replace('\n', ' ')[:100]
                if preview:
                    found_msg += f"  Preview: \"{preview}...\"\n"
                found_msg += "\n"
            else:
                # v2.1: Show content preservation info if present (v2.x spec)
                content_verbatim = weaver_spec_json.get("content_verbatim") or weaver_spec_json.get("metadata", {}).get("content_verbatim")
                location = weaver_spec_json.get("location") or weaver_spec_json.get("metadata", {}).get("location")
                
                found_msg = "✓ Found Weaver spec to validate.\n"
                if content_verbatim:
                    found_msg += f"  Content: \"{content_verbatim[:50]}{'...' if len(str(content_verbatim)) > 50 else ''}\"\n"
                if location:
                    found_msg += f"  Location: {location}\n"
                found_msg += "\n"
            
            yield _safe_json_event({"type": "token", "content": found_msg})
            response_parts.append(found_msg)
        else:
            warn = (
                "⚠️ No Weaver output found.\n"
                "Spec Gate may drift if it must infer the job from raw chat.\n"
                "Consider running Weaver first: `how does that look all together`\n\n"
            )
            yield _safe_json_event({"type": "token", "content": warn})
            response_parts.append(warn)

        validating_msg = f"Validating spec (Round {clarification_round}/3)...\n\n"
        yield _safe_json_event({"type": "token", "content": validating_msg})
        response_parts.append(validating_msg)
        await asyncio.sleep(0.05)

        user_intent = (message or "").strip()

        # v3.0: Choose grounded (Contract v1) or v2 implementation
        use_grounded = _USE_GROUNDED_SPEC_GATE and _SPEC_GATE_GROUNDED_AVAILABLE and run_spec_gate_grounded
        
        if use_grounded:
            version_msg = "🔬 Using **SpecGate Contract v1** (grounded, evidence-based, read-only)\n\n"
            yield _safe_json_event({"type": "token", "content": version_msg})
            response_parts.append(version_msg)
        
        try:
            if use_grounded:
                # Contract v1: Grounded, evidence-based, read-only
                result: SpecGateResult = await run_spec_gate_grounded(
                    db=db,
                    job_id=job_id,
                    user_intent=user_intent,
                    provider_id=spec_gate_provider,
                    model_id=spec_gate_model,
                    project_id=project_id,
                    constraints_hint=constraints_hint,
                    spec_version=clarification_round,
                )
            else:
                # Fallback: v2 implementation (DB writes enabled)
                result: SpecGateResult = await run_spec_gate_v2(
                    db=db,
                    job_id=job_id,
                    user_intent=user_intent,
                    provider_id=spec_gate_provider,
                    model_id=spec_gate_model,
                    project_id=project_id,
                    constraints_hint=constraints_hint,
                    spec_version=clarification_round,
                )
        except Exception as e:
            error_msg = f"❌ Spec Gate failed: {e}\n"
            yield _safe_json_event({"type": "token", "content": error_msg})
            response_parts.append(error_msg)
            if trace:
                try:
                    trace.finalize(success=False, error_message=str(e))
                except Exception:
                    pass
            yield _safe_json_event({
                "type": "done",
                "provider": spec_gate_provider,
                "model": spec_gate_model,
                "total_length": sum(len(p) for p in response_parts),
            })
            return

        # Hard stop
        if getattr(result, "hard_stopped", False):
            reason = getattr(result, "hard_stop_reason", None) or "Spec Gate has blocked this request."
            hard_stop_msg = f"🛑 **HARD STOP**\n\n{reason}\n"
            yield _safe_json_event({"type": "token", "content": hard_stop_msg})
            response_parts.append(hard_stop_msg)

            if _FLOW_STATE_AVAILABLE and cancel_flow:
                try:
                    cancel_flow(project_id)
                except Exception:
                    pass

            yield _safe_json_event({
                "type": "done",
                "provider": spec_gate_provider,
                "model": spec_gate_model,
                "total_length": sum(len(p) for p in response_parts),
                "hard_stopped": True,
            })
            return

        open_q = getattr(result, "open_questions", None) or []
        blocking_issues = getattr(result, "blocking_issues", None) or []
        validation_status = getattr(result, "validation_status", "pending")
        ready = bool(getattr(result, "ready_for_pipeline", False))

        # v2.1: Handle blocking issues with clear guidance
        if blocking_issues and not ready:
            header = f"⛔ **BLOCKING VALIDATION FAILED** (Round {clarification_round}/3)\n\n"
            yield _safe_json_event({"type": "token", "content": header})
            response_parts.append(header)
            
            explain = (
                "The spec cannot proceed to the pipeline because critical information is missing.\n"
                "Please provide the following:\n\n"
            )
            yield _safe_json_event({"type": "token", "content": explain})
            response_parts.append(explain)
            
            for i, q in enumerate(open_q, 1):
                line = f"**{i}.** {q}\n\n"
                yield _safe_json_event({"type": "token", "content": line})
                response_parts.append(line)
                await asyncio.sleep(0.01)
            
            guidance = (
                "---\n"
                "**How to resolve:**\n"
                "1. Answer the questions above in your next message\n"
                "2. Then say **'Astra, command: critical architecture'** to retry validation\n\n"
                f"Round {clarification_round}/3 - Round 3 will finalize even with gaps (not recommended).\n"
            )
            yield _safe_json_event({"type": "token", "content": guidance})
            response_parts.append(guidance)

            if _FLOW_STATE_AVAILABLE and advance_to_spec_gate_questions:
                try:
                    advance_to_spec_gate_questions(
                        project_id=project_id,
                        job_id=job_id,
                        spec_id=getattr(result, "spec_id", "") or "",
                        spec_hash=getattr(result, "spec_hash", "") or "",
                        questions=open_q,
                        clarification_round=clarification_round,
                    )
                except Exception as e:
                    logger.debug("[spec_gate_stream] advance_to_spec_gate_questions failed: %s", e)

            # Emit blocking event for frontend
            yield _safe_json_event({
                "type": "spec_blocked",
                "spec_id": getattr(result, "spec_id", None),
                "blocking_issues": blocking_issues,
                "questions": open_q,
                "round": clarification_round,
            })

        # Non-blocking questions (rare case - validation passed but questions remain)
        elif open_q and not ready:
            header = f"❓ **Clarification Needed** (Round {clarification_round}/3)\n\n"
            yield _safe_json_event({"type": "token", "content": header})
            response_parts.append(header)

            for q in open_q:
                line = f"• {q}\n\n"
                yield _safe_json_event({"type": "token", "content": line})
                response_parts.append(line)
                await asyncio.sleep(0.01)

            followup = (
                "Please answer these questions, then say **'Astra, command: critical architecture'** again.\n"
            )
            yield _safe_json_event({"type": "token", "content": followup})
            response_parts.append(followup)

            if _FLOW_STATE_AVAILABLE and advance_to_spec_gate_questions:
                try:
                    advance_to_spec_gate_questions(
                        project_id=project_id,
                        job_id=job_id,
                        spec_id=getattr(result, "spec_id", "") or "",
                        spec_hash=getattr(result, "spec_hash", "") or "",
                        questions=open_q,
                        clarification_round=clarification_round,
                    )
                except Exception as e:
                    logger.debug("[spec_gate_stream] advance_to_spec_gate_questions failed: %s", e)

        # Ready: stream SPoT markdown
        else:
            # Check if this is Round 3 with unresolved questions
            is_round3_with_issues = (
                ready and 
                validation_status == "validated_with_issues" and
                blocking_issues
            )
            
            if is_round3_with_issues:
                success_header = "⚠️ **Spec Finalized (Round 3) - With Unresolved Items**\n\n"
                yield _safe_json_event({"type": "token", "content": success_header})
                response_parts.append(success_header)
                
                warn_msg = (
                    "**Note:** This spec was finalized on Round 3 but contains unresolved questions.\n"
                    "SpecGate did NOT fill in assumptions - gaps are explicitly marked.\n\n"
                )
                yield _safe_json_event({"type": "token", "content": warn_msg})
                response_parts.append(warn_msg)
            else:
                success_header = "✅ **Spec Validated - SPoT Generated**\n\n"
                yield _safe_json_event({"type": "token", "content": success_header})
                response_parts.append(success_header)

            # v1.5: Persist spec after validation (caller responsibility)
            db_persisted = getattr(result, "db_persisted", False)
            persist_error = None
            
            if not db_persisted and _SPEC_PERSISTENCE_AVAILABLE and persist_spec and build_spec_schema:
                try:
                    # Build spec schema for persistence
                    spot_md = getattr(result, "spot_markdown", "") or ""
                    spec_id = getattr(result, "spec_id", "") or ""
                    spec_hash = getattr(result, "spec_hash", "") or ""
                    
                    # Extract info from spot_markdown for schema
                    goal = ""
                    if "## Goal" in spot_md:
                        goal_start = spot_md.index("## Goal") + len("## Goal")
                        goal_end = spot_md.find("##", goal_start)
                        if goal_end == -1:
                            goal_end = len(spot_md)
                        goal = spot_md[goal_start:goal_end].strip()
                    
                    summary = safe_summary_from_objective(goal) if safe_summary_from_objective else goal[:180]
                    
                    spec_schema = build_spec_schema(
                        spec_id=spec_id,
                        title="SPoT Spec (SpecGate Contract v1)",
                        summary=summary,
                        objective=goal,
                        outputs=[],  # Extracted from spec if needed
                        steps=[],    # Extracted from spec if needed
                        acceptance=[],  # Extracted from spec if needed
                        context={"source": "spec_gate_grounded", "round": clarification_round},
                        job_id=job_id,
                        provider_id=spec_gate_provider,
                        model_id=spec_gate_model,
                    )
                    
                    if spec_schema:
                        success, db_spec_id, db_spec_hash, error = persist_spec(
                            db=db,
                            project_id=project_id,
                            spec_schema=spec_schema,
                            provider_id=spec_gate_provider,
                            model_id=spec_gate_model,
                        )
                        if success:
                            db_persisted = True
                            logger.info("[spec_gate_stream] v1.5: Spec persisted to DB: %s", db_spec_id)
                        else:
                            persist_error = error
                            logger.warning("[spec_gate_stream] v1.5: Spec persistence failed: %s", error)
                except Exception as e:
                    persist_error = str(e)
                    logger.exception("[spec_gate_stream] v1.5: Spec persistence exception: %s", e)
            
            if db_persisted:
                db_msg = "💾 Spec persisted to database.\n\n"
            else:
                db_msg = "⚠️ Spec NOT persisted to database - may not survive restart.\n"
                if persist_error:
                    db_msg += f"   Reason: {persist_error}\n"
                db_msg += "\n"
            yield _safe_json_event({"type": "token", "content": db_msg})
            response_parts.append(db_msg)

            spot_md = getattr(result, "spot_markdown", None)
            if spot_md:
                yield _safe_json_event({"type": "token", "content": spot_md})
                response_parts.append(spot_md)

            meta = (
                f"\n\n---\n"
                f"**Spec ID:** `{getattr(result,'spec_id',None)}`\n"
                f"**Spec Hash:** `{str(getattr(result,'spec_hash', '') or '')[:16]}...`\n"
                f"**Round:** {clarification_round}\n"
                f"**Status:** {validation_status}\n"
            )
            yield _safe_json_event({"type": "token", "content": meta})
            response_parts.append(meta)

            next_step = (
                "\n🚀 **Spec Complete.**\n"
                "Say **'Astra, command: run critical pipeline'** to proceed to architecture generation.\n"
            )
            yield _safe_json_event({"type": "token", "content": next_step})
            response_parts.append(next_step)

            if _FLOW_STATE_AVAILABLE and advance_to_spec_validated:
                try:
                    advance_to_spec_validated(
                        project_id=project_id,
                        spec_id=getattr(result, "spec_id", "") or "",
                        spec_hash=getattr(result, "spec_hash", "") or "",
                        spec_version=clarification_round,
                    )
                except Exception as e:
                    logger.debug("[spec_gate_stream] advance_to_spec_validated failed: %s", e)

            yield _safe_json_event({
                "type": "spec_ready",
                "spec_id": getattr(result, "spec_id", None),
                "spec_hash": getattr(result, "spec_hash", None),
                "job_id": job_id,
                "db_persisted": db_persisted,  # v1.5: Use actual persistence result
                "validation_status": validation_status,
            })

        # Persist assistant message
        if memory_service:
            try:
                full_response = "".join(response_parts)
                memory_service.create_message(
                    db=db,
                    project_id=project_id,
                    role="assistant",
                    content=full_response,
                )
            except Exception:
                pass

        yield _safe_json_event({
            "type": "done",
            "provider": spec_gate_provider,
            "model": spec_gate_model,
            "total_length": sum(len(p) for p in response_parts),
            "spec_id": getattr(result, "spec_id", None) if "result" in locals() else None,
            "db_persisted": db_persisted if "db_persisted" in locals() else False,  # v1.5: Use actual persistence result
            "validation_status": validation_status if "validation_status" in locals() else "unknown",
        })

    except Exception as e:
        logger.exception("[spec_gate_stream] Stream failed: %s", e)
        if trace:
            try:
                trace.finalize(success=False, error_message=str(e))
            except Exception:
                pass
        yield _safe_json_event({"type": "error", "error": str(e)})


__all__ = ["generate_spec_gate_stream"]