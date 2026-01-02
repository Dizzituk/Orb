# FILE: app/llm/spec_gate_stream.py
"""
Spec Gate streaming handler for ASTRA command flow.

Receives refined spec from Weaver and validates through Spec Gate.
Returns questions or SPoT (Singular Point of Truth) markdown.

v1.4 (2026-01): CRITICAL FIX - Pass project_id to run_spec_gate_v2 for DB persistence
v1.3 (2026-01): Fixed to use runtime env lookup for model selection (no hardcoded models)
v1.2 (2026-01): Added 3-round cap enforcement - auto-validates after round 3
v1.1 (2026-01): Added flow state management for clarification routing
v1.0 (2026-01): Initial implementation
"""

import json
import logging
import asyncio
from typing import Optional, Any

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Import spec_gate_v2
try:
    from app.pot_spec.spec_gate_v2 import run_spec_gate_v2, SpecGateResult
    _SPEC_GATE_V2_AVAILABLE = True
except ImportError:
    _SPEC_GATE_V2_AVAILABLE = False
    logger.warning("[spec_gate_stream] spec_gate_v2 module not available")

# Import memory service for persistence
try:
    from app.memory import service as memory_service, schemas as memory_schemas
except ImportError:
    memory_service = None
    memory_schemas = None

# Import audit logger
try:
    from app.llm.audit_logger import RoutingTrace
except ImportError:
    RoutingTrace = None

# Import job service for spec state
try:
    from app.jobs.service import get_active_job_for_project, get_latest_draft_spec
except ImportError:
    get_active_job_for_project = None
    get_latest_draft_spec = None

# Import flow state management
try:
    from app.llm.spec_flow_state import (
        get_active_flow,
        advance_to_spec_gate_questions,
        advance_to_spec_validated,
        SpecFlowStage,
    )
    _FLOW_STATE_AVAILABLE = True
except ImportError:
    _FLOW_STATE_AVAILABLE = False
    get_active_flow = None

# Import centralized runtime model lookup (v1.4: uses stage_models)
from app.llm.stage_models import get_spec_gate_config


async def generate_spec_gate_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[Any] = None,
    conversation_id: Optional[str] = None,
    is_clarification_response: bool = False,
):
    """
    Generate SSE stream for Spec Gate validation.
    
    Flow:
    1. Check if this is a clarification response (mid-flow)
    2. Get latest draft spec from Weaver (from DB or job artifacts)
    3. Call run_spec_gate_v2() with the spec
    4. If result.open_questions: stream questions to user, update flow state
    5. If result.ready_for_pipeline: stream SPoT markdown + prompt, update flow state
    6. If result.hard_stopped: stream hard stop reason
    
    v1.4: Now passes project_id to run_spec_gate_v2 for DB persistence (restart survival).
    """
    response_parts = []
    
    # Get provider/model at runtime from centralized stage_models
    spec_gate_cfg = get_spec_gate_config()
    spec_gate_provider = spec_gate_cfg.provider
    spec_gate_model = spec_gate_cfg.model
    
    # AUDIT: Log the resolved model
    logger.info(f"[spec_gate_stream] AUDIT: Using provider={spec_gate_provider}, model={spec_gate_model}")
    print(f"[SPEC_GATE_STREAM] project_id={project_id}, provider={spec_gate_provider}, model={spec_gate_model}")
    
    try:
        yield "data: " + json.dumps({"type": "token", "content": "🔍 **Spec Gate Validation**\n\n"}) + "\n\n"
        response_parts.append("🔍 **Spec Gate Validation**\n\n")
        
        if not _SPEC_GATE_V2_AVAILABLE:
            error_msg = "⚠️ Spec Gate module not available. Please ensure app.pot_spec.spec_gate_v2 is installed.\n"
            yield "data: " + json.dumps({"type": "token", "content": error_msg}) + "\n\n"
            response_parts.append(error_msg)
            if trace:
                trace.finalize(success=False, error_message="spec_gate_v2 not available")
            yield "data: " + json.dumps({
                "type": "done", "provider": spec_gate_provider, "model": spec_gate_model,
                "total_length": sum(len(p) for p in response_parts)
            }) + "\n\n"
            return
        
        # Check for existing flow state
        existing_flow = None
        if _FLOW_STATE_AVAILABLE and get_active_flow:
            existing_flow = get_active_flow(project_id)
        
        # Get job context
        job_id = None
        user_intent = message
        clarification_round = 1
        
        if existing_flow and existing_flow.job_id:
            job_id = existing_flow.job_id
            clarification_round = existing_flow.clarification_round + 1
            
            # Enforce 3-round maximum
            if clarification_round > 3:
                cap_msg = "⚠️ **Maximum clarification rounds (3) reached.** Forcing spec validation...\n\n"
                yield "data: " + json.dumps({"type": "token", "content": cap_msg}) + "\n\n"
                response_parts.append(cap_msg)
                clarification_round = 3  # Cap for downstream processing
            else:
                yield "data: " + json.dumps({"type": "token", "content": f"Continuing job: `{job_id}` (Round {clarification_round}/3)\n"}) + "\n\n"
                response_parts.append(f"Continuing job: `{job_id}` (Round {clarification_round}/3)\n")
        elif get_active_job_for_project:
            try:
                active_job = get_active_job_for_project(db, project_id)
                if active_job:
                    job_id = active_job.id
                    yield "data: " + json.dumps({"type": "token", "content": f"Using job: `{job_id}`\n"}) + "\n\n"
                    response_parts.append(f"Using job: `{job_id}`\n")
            except Exception as e:
                logger.warning(f"[spec_gate_stream] Could not get active job: {e}")
        
        if not job_id:
            # Create a new job ID if none exists
            from uuid import uuid4
            job_id = f"sg-{uuid4().hex[:8]}"
            yield "data: " + json.dumps({"type": "token", "content": f"Creating new job: `{job_id}`\n"}) + "\n\n"
            response_parts.append(f"Creating new job: `{job_id}`\n")
        
        # Get draft spec from Weaver if available
        if get_latest_draft_spec:
            try:
                draft_spec = get_latest_draft_spec(db, project_id)
                if draft_spec:
                    user_intent = draft_spec.get("user_intent", message)
                    yield "data: " + json.dumps({"type": "token", "content": "Found draft spec from Weaver.\n\n"}) + "\n\n"
                    response_parts.append("Found draft spec from Weaver.\n\n")
            except Exception as e:
                logger.debug(f"[spec_gate_stream] No draft spec: {e}")
        
        # Display the actual model being used (dynamic, not hardcoded)
        validating_msg = f"Validating spec with {spec_gate_model}...\n\n"
        yield "data: " + json.dumps({"type": "token", "content": validating_msg}) + "\n\n"
        response_parts.append(validating_msg)
        await asyncio.sleep(0.1)  # Small delay for UX
        
        # Run Spec Gate v2
        # v1.4 CRITICAL FIX: Pass project_id for DB persistence
        try:
            print(f"[SPEC_GATE_STREAM] Calling run_spec_gate_v2 with project_id={project_id}, job_id={job_id}")
            result: SpecGateResult = await run_spec_gate_v2(
                db=db,
                job_id=job_id,
                user_intent=user_intent,
                provider_id=spec_gate_provider,
                model_id=spec_gate_model,
                project_id=project_id,  # v1.4: CRITICAL - enables DB persistence for restart survival
                spec_version=clarification_round,
            )
            print(f"[SPEC_GATE_STREAM] Result: ready_for_pipeline={result.ready_for_pipeline}, db_persisted={result.db_persisted}")
        except Exception as e:
            error_msg = f"❌ Spec Gate failed: {e}\n"
            yield "data: " + json.dumps({"type": "token", "content": error_msg}) + "\n\n"
            response_parts.append(error_msg)
            if trace:
                trace.finalize(success=False, error_message=str(e))
            yield "data: " + json.dumps({
                "type": "done", "provider": spec_gate_provider, "model": spec_gate_model,
                "total_length": sum(len(p) for p in response_parts)
            }) + "\n\n"
            return
        
        # Handle result based on state
        if result.hard_stopped:
            hard_stop_msg = f"🛑 **HARD STOP**\n\n{result.hard_stop_reason}\n\nSpec Gate has blocked this request. Please revise and try again.\n"
            yield "data: " + json.dumps({"type": "token", "content": hard_stop_msg}) + "\n\n"
            response_parts.append(hard_stop_msg)
            
            # Clear flow state on hard stop
            if _FLOW_STATE_AVAILABLE:
                from app.llm.spec_flow_state import cancel_flow
                cancel_flow(project_id)
        
        elif result.open_questions:
            # If we're at round 3 and still have questions, force validation instead
            if clarification_round >= 3:
                force_msg = "⚠️ **Round 3 complete - auto-validating spec despite open questions.**\n\n"
                yield "data: " + json.dumps({"type": "token", "content": force_msg}) + "\n\n"
                response_parts.append(force_msg)
                
                # List remaining questions as notes
                notes_header = "📋 **Remaining questions (for reference):**\n"
                yield "data: " + json.dumps({"type": "token", "content": notes_header}) + "\n\n"
                response_parts.append(notes_header)
                
                for i, q in enumerate(result.open_questions, 1):
                    q_line = f"- {q}\n"
                    yield "data: " + json.dumps({"type": "token", "content": q_line}) + "\n\n"
                    response_parts.append(q_line)
                
                # Proceed as if ready for pipeline
                next_step = "\n🚀 **Spec Complete.**\nIf you're happy with this spec, say **'Astra, command: run critical pipeline'** to proceed.\n"
                yield "data: " + json.dumps({"type": "token", "content": next_step}) + "\n\n"
                response_parts.append(next_step)
                
                # Update flow state - force to validated
                if _FLOW_STATE_AVAILABLE:
                    advance_to_spec_validated(
                        project_id=project_id,
                        spec_id=result.spec_id,
                        spec_hash=result.spec_hash,
                        spec_version=result.spec_version,
                    )
                
                # Emit spec ready event
                yield "data: " + json.dumps({
                    "type": "spec_ready",
                    "spec_id": result.spec_id,
                    "spec_hash": result.spec_hash,
                    "spec_version": result.spec_version,
                    "job_id": job_id,
                    "db_persisted": result.db_persisted,  # v1.4: Include DB status
                    "forced_validation": True,
                }) + "\n\n"
            else:
                # Normal case: show questions and request answers
                questions_header = f"❓ **Clarification Needed** (Round {result.clarification_round}/3)\n\n"
                yield "data: " + json.dumps({"type": "token", "content": questions_header}) + "\n\n"
                response_parts.append(questions_header)
                
                for i, q in enumerate(result.open_questions, 1):
                    q_line = f"{i}. {q}\n"
                    yield "data: " + json.dumps({"type": "token", "content": q_line}) + "\n\n"
                    response_parts.append(q_line)
                    await asyncio.sleep(0.02)
                
                followup = "\nPlease answer these questions, then say **'Astra, command: critical architecture'** again to continue.\n"
                yield "data: " + json.dumps({"type": "token", "content": followup}) + "\n\n"
                response_parts.append(followup)
                
                # Update flow state to track we're awaiting clarification
                if _FLOW_STATE_AVAILABLE:
                    advance_to_spec_gate_questions(
                        project_id=project_id,
                        job_id=job_id,
                        spec_id=result.spec_id,
                        spec_hash=result.spec_hash,
                        questions=result.open_questions,
                        clarification_round=result.clarification_round,
                    )
        
        elif result.ready_for_pipeline:
            success_header = "✅ **Spec Validated - SPoT Generated**\n\n"
            yield "data: " + json.dumps({"type": "token", "content": success_header}) + "\n\n"
            response_parts.append(success_header)
            
            # v1.4: Show DB persistence status
            if result.db_persisted:
                db_msg = "💾 Spec persisted to database (restart-safe).\n\n"
            else:
                db_msg = "⚠️ Spec NOT persisted to database - may not survive restart.\n\n"
            yield "data: " + json.dumps({"type": "token", "content": db_msg}) + "\n\n"
            response_parts.append(db_msg)
            
            # Stream the SPoT markdown
            if result.spec_summary_markdown:
                # Stream in chunks for better UX
                chunk_size = 100
                md = result.spec_summary_markdown
                for i in range(0, len(md), chunk_size):
                    chunk = md[i:i + chunk_size]
                    yield "data: " + json.dumps({"type": "token", "content": chunk}) + "\n\n"
                    response_parts.append(chunk)
                    await asyncio.sleep(0.01)
            
            # Add spec metadata
            meta = f"\n\n---\n**Spec ID:** `{result.spec_id}`\n**Hash:** `{result.spec_hash[:16]}...`\n**Version:** {result.spec_version}\n"
            yield "data: " + json.dumps({"type": "token", "content": meta}) + "\n\n"
            response_parts.append(meta)
            
            # Prompt for next step
            next_step = "\n🚀 **Spec Complete.**\nIf you're happy with this spec, say **'Astra, command: run critical pipeline'** to proceed.\n"
            yield "data: " + json.dumps({"type": "token", "content": next_step}) + "\n\n"
            response_parts.append(next_step)
            
            # Update flow state - spec is now validated (SPoT ready)
            if _FLOW_STATE_AVAILABLE:
                advance_to_spec_validated(
                    project_id=project_id,
                    spec_id=result.spec_id,
                    spec_hash=result.spec_hash,
                    spec_version=result.spec_version,
                )
            
            # Emit spec ready event for frontend
            yield "data: " + json.dumps({
                "type": "spec_ready",
                "spec_id": result.spec_id,
                "spec_hash": result.spec_hash,
                "spec_version": result.spec_version,
                "job_id": job_id,
                "db_persisted": result.db_persisted,  # v1.4: Include DB status
            }) + "\n\n"
        
        # Save to memory
        full_response = "".join(response_parts)
        if memory_service and memory_schemas:
            memory_service.create_message(db, memory_schemas.MessageCreate(
                project_id=project_id, role="user", content=message, provider="local"
            ))
            memory_service.create_message(db, memory_schemas.MessageCreate(
                project_id=project_id, role="assistant", content=full_response,
                provider=spec_gate_provider, model=spec_gate_model
            ))
        
        if trace:
            trace.finalize(success=True)
        
        yield "data: " + json.dumps({
            "type": "done", 
            "provider": spec_gate_provider, 
            "model": spec_gate_model,
            "total_length": len(full_response),
            "spec_id": result.spec_id if result else None,
            "db_persisted": result.db_persisted if result else False,  # v1.4
        }) + "\n\n"
        
    except Exception as e:
        logger.exception("[spec_gate_stream] Stream failed: %s", e)
        if trace:
            trace.finalize(success=False, error_message=str(e))
        yield "data: " + json.dumps({"type": "error", "error": str(e)}) + "\n\n"


__all__ = ["generate_spec_gate_stream"]