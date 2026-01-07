# FILE: app/llm/high_stakes_stream.py
"""
High-stakes critique stream generator.

v3.5 (2026-01): Integrated stage tracing for full command visibility
v3.4 (2026-01): Fixed spec gate model env var precedence (OPENAI_SPEC_GATE_MODEL takes priority)
v3.3 (2026-01): Uses run_spec_gate_v2 with project_id for DB persistence (restart survival)
v3.2 (2025-12): Added comprehensive debug logging for streaming diagnosis
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Optional, List

from sqlalchemy.orm import Session

from app.memory import service as memory_service, schemas as memory_schemas
from app.llm.schemas import LLMTask
from app.llm.audit_logger import RoutingTrace

from app.llm.router import (
    run_high_stakes_with_critique,
    synthesize_envelope_from_task,
)

# v3.3: Use spec_gate_v2 for DB persistence
from app.pot_spec.spec_gate_v2 import run_spec_gate_v2
from app.pot_spec.spec_gate import detect_user_questions

from app.jobs.stage3_locks import (
    build_spec_echo_instruction,
    verify_and_store_stage3,
    append_stage3_ledger_event,
)

from .stream_utils import (
    parse_reasoning_tags,
    extract_usage_tokens,
)

# v3.5: Stage tracing
try:
    from .stage_trace import StageTrace, log_model_resolution
    _STAGE_TRACE_AVAILABLE = True
except ImportError:
    _STAGE_TRACE_AVAILABLE = False
    StageTrace = None

# v3.6: Centralized stage models
try:
    from .stage_models import get_spec_gate_config, get_critique_config, get_revision_config
    _STAGE_MODELS_AVAILABLE = True
except ImportError:
    _STAGE_MODELS_AVAILABLE = False


def _get_spec_gate_config() -> tuple[str, str]:
    """
    Get Spec Gate provider and model from env vars AT RUNTIME.
    
    v3.4: Ensures env vars are respected without hardcoded overrides.
    
    Precedence for model:
    1. OPENAI_SPEC_GATE_MODEL (explicit spec gate override)
    2. OPENAI_DEFAULT_MODEL (general default)
    3. "gpt-4.1-mini" (hardcoded fallback only if both env vars unset)
    
    Returns: (provider, model)
    """
    provider = os.getenv("SPEC_GATE_PROVIDER", "openai")
    
    # v3.4: Explicit precedence - OPENAI_SPEC_GATE_MODEL wins over everything
    model = os.getenv("OPENAI_SPEC_GATE_MODEL")
    if not model:
        model = os.getenv("OPENAI_DEFAULT_MODEL")
    if not model:
        model = "gpt-4.1-mini"  # Last resort hardcoded default
    
    return provider, model

logger = logging.getLogger(__name__)


async def generate_high_stakes_critique_stream(
    project_id: int,
    message: str,
    provider: str,
    model: str,
    system_prompt: str,
    messages: List[dict],
    full_context: str,
    job_type_str: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
    enable_reasoning: bool = False,
    continue_job_id: Optional[str] = None,
):
    """Generate SSE stream for high-stakes critique pipeline.
    
    This function:
    1. Runs Spec Gate for architecture_design jobs
    2. Injects SPEC_ID/SPEC_HASH header instruction into system_prompt
    3. Rebuilds task and envelope with updated system_prompt
    4. Runs high-stakes critique pipeline
    5. Verifies spec hash in response (Stage 3)
    6. Fails fast on mismatch
    
    v3.3: Now persists validated specs to DB via run_spec_gate_v2 for restart survival.
    v3.5: Stage tracing for full command visibility.
    """
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    trace_finished = False
    
    # v3.5: Create stage trace
    stage_trace = None
    if _STAGE_TRACE_AVAILABLE and StageTrace:
        stage_trace = StageTrace.start(
            command_type=f"high_stakes_{job_type_str}",
            project_id=project_id,
            job_id=continue_job_id,
        )
    
    print(f"[DEBUG] ========== HIGH STAKES STREAM START ==========")
    print(f"[DEBUG] job_type={job_type_str}, provider={provider}, model={model}")
    print(f"[DEBUG] project_id={project_id}, message length: {len(message)}, messages count: {len(messages)}")

    spec_id: Optional[str] = None
    spec_hash: Optional[str] = None
    needs_verification = False

    try:
        print(f"[DEBUG] Building initial LLMTask...")
        task = LLMTask(
            project_id=project_id,
            user_message=message,
            system_prompt=system_prompt,
            messages=messages,
            full_context=full_context,
            job_type=job_type_str,
            enable_reasoning=enable_reasoning,
        )
        print(f"[DEBUG] LLMTask built successfully")

        envelope = synthesize_envelope_from_task(task)
        print(f"[DEBUG] Envelope synthesized: job_id={envelope.job_id}")
        
        if continue_job_id:
            envelope.job_id = continue_job_id
            print(f"[DEBUG] Continuing existing job: {continue_job_id}")

        # =====================================================================
        # Spec Gate (only stage allowed to ask user questions)
        # =====================================================================
        if (job_type_str or "").strip().lower() == "architecture_design":
            print(f"[DEBUG] Architecture job detected, running Spec Gate v2...")
            
            # v3.4: Use helper function for proper env var precedence
            spec_gate_provider, spec_gate_model = _get_spec_gate_config()
            
            # v3.4: Explicit audit logging of what we're about to use
            print(f"[SPEC_GATE_AUDIT] Provider: {spec_gate_provider}")
            print(f"[SPEC_GATE_AUDIT] Model: {spec_gate_model}")
            print(f"[SPEC_GATE_AUDIT] OPENAI_SPEC_GATE_MODEL env: {os.getenv('OPENAI_SPEC_GATE_MODEL', '<not set>')}")
            print(f"[SPEC_GATE_AUDIT] SPEC_GATE_PROVIDER env: {os.getenv('SPEC_GATE_PROVIDER', '<not set>')}")
            print(f"[DEBUG] Spec Gate: provider={spec_gate_provider}, model={spec_gate_model}")
            
            # Determine spec_version based on continuation
            spec_version = 1
            if continue_job_id and messages:
                # Count previous spec gate rounds from messages
                spec_gate_rounds = sum(1 for msg in messages 
                                       if msg.get("role") == "assistant" 
                                       and "Spec Gate" in msg.get("content", ""))
                spec_version = spec_gate_rounds + 1
                print(f"[DEBUG] Continuation detected, spec_version={spec_version}")
            
            if continue_job_id and messages:
                conversation_parts = []
                for msg in messages:
                    role = msg.get("role", "user").upper()
                    content = msg.get("content", "")
                    if role == "ASSISTANT" and "Spec Gate" in content:
                        conversation_parts.append(f"SPEC_GATE: {content}")
                    elif role == "USER":
                        conversation_parts.append(f"USER: {content}")
                
                conversation_parts.append(f"USER: {message}")
                
                spec_gate_user_intent = (
                    "Previous conversation:\n"
                    + "\n\n".join(conversation_parts)
                    + "\n\n---\nBased on the above conversation, update the spec with the user's answers. "
                    "If all critical questions have been answered, set open_questions to []. "
                    "Only ask NEW questions if the answers revealed new ambiguities."
                )
                print(f"[DEBUG] Spec Gate continuation with {len(messages)} messages of context")
            else:
                spec_gate_user_intent = message

            append_stage3_ledger_event(
                envelope.job_id,
                {
                    "event": "STAGE_STARTED",
                    "job_id": envelope.job_id,
                    "stage": "spec_gate_streaming",
                    "ts_utc": datetime.utcnow().isoformat() + "Z",
                },
            )

            # v3.5: Enter spec gate stage
            if stage_trace:
                stage_trace.set_job_id(envelope.job_id)
                stage_trace.enter_stage("spec_gate", provider=spec_gate_provider, model=spec_gate_model,
                                       spec_version=spec_version)

            # v3.3: Use run_spec_gate_v2 with project_id for DB persistence
            print(f"[DEBUG] Calling run_spec_gate_v2 with project_id={project_id}...")
            spec_gate_result = await run_spec_gate_v2(
                db=db,
                job_id=envelope.job_id,
                user_intent=spec_gate_user_intent,
                provider_id=spec_gate_provider,
                model_id=spec_gate_model,
                project_id=project_id,  # CRITICAL: enables DB persistence
                spec_version=spec_version,
                constraints_hint={"stability_accuracy": "high", "allowed_tools": "free_only"},
            )
            
            # Unpack result
            spec_id = spec_gate_result.spec_id
            spec_hash = spec_gate_result.spec_hash
            open_questions = spec_gate_result.open_questions
            
            print(f"[DEBUG] Spec gate completed: spec_id={spec_id}, hash={spec_hash[:16] if spec_hash else None}...")
            print(f"[DEBUG] questions={len(open_questions) if open_questions else 0}, ready_for_pipeline={spec_gate_result.ready_for_pipeline}")
            print(f"[DEBUG] db_persisted={spec_gate_result.db_persisted}")

            if open_questions:
                print(f"[DEBUG] Open questions found, returning early")
                
                # v3.5: Exit spec gate stage (paused for questions)
                if stage_trace:
                    stage_trace.exit_stage("spec_gate", success=True, 
                                          questions_count=len(open_questions),
                                          outcome="needs_clarification")
                    stage_trace.finish(success=True, outcome="paused_for_spec_clarification")
                
                questions_text = "\n\n".join([f"**Q{i+1}:** {q}" for i, q in enumerate(open_questions)])
                intro = "🔍 **Spec Gate - Clarification Needed**\n\nBefore I can design the architecture, I need to clarify a few things:\n\n"
                full_response = intro + questions_text + "\n\n---\n*Please answer these questions so I can create a complete specification.*"
                
                chunk_size = 50
                for i in range(0, len(full_response), chunk_size):
                    chunk = full_response[i : i + chunk_size]
                    yield "data: " + json.dumps({'type': 'token', 'content': chunk}) + "\n\n"
                    await asyncio.sleep(0.01)
                
                memory_service.create_message(db, memory_schemas.MessageCreate(
                    project_id=project_id, role="user", content=message, provider="local"
                ))
                memory_service.create_message(db, memory_schemas.MessageCreate(
                    project_id=project_id, role="assistant", content=full_response, 
                    provider="spec_gate", model=spec_gate_model
                ))
                
                yield "data: " + json.dumps({
                    "type": "pause",
                    "pause_state": "needs_spec_clarification",
                    "job_id": envelope.job_id,
                    "spec_id": spec_id,
                    "spec_hash": spec_hash,
                    "open_questions": open_questions,
                    "spec_version": spec_version,
                    "artifacts_written": f"jobs/{envelope.job_id}/spec/ + jobs/{envelope.job_id}/ledger/",
                }) + "\n\n"
                
                if trace and not trace_finished:
                    trace.finalize(success=True)
                    trace_finished = True
                return

            # Spec is complete (no questions) - proceed to architecture generation
            print(f"[DEBUG] Spec complete, injecting SPEC_ID/SPEC_HASH header instruction")
            
            # v3.5: Exit spec gate stage (complete)
            if stage_trace:
                stage_trace.exit_stage("spec_gate", success=True, 
                                      spec_id=spec_id, spec_hash=spec_hash[:16] if spec_hash else None,
                                      db_persisted=spec_gate_result.db_persisted)
            
            # Emit ready for pipeline (spec_gate_v2 already emits this, but log here too)
            if spec_gate_result.ready_for_pipeline:
                print(f"[DEBUG] Spec ready for pipeline, db_persisted={spec_gate_result.db_persisted}")
            
            needs_verification = True
            spec_header_instruction = build_spec_echo_instruction(spec_id, spec_hash)
            
            updated_system_prompt = (
                spec_header_instruction
                + "\n\n---\n\n"
                + system_prompt
            )
            
            task = LLMTask(
                project_id=project_id,
                user_message=message,
                system_prompt=updated_system_prompt,
                messages=messages,
                full_context=full_context,
                job_type=job_type_str,
                enable_reasoning=enable_reasoning,
            )
            envelope = synthesize_envelope_from_task(task)
            envelope.job_id = continue_job_id or envelope.job_id
            print(f"[DEBUG] Task rebuilt with spec header, job_id={envelope.job_id}")

        # =====================================================================
        # Run high-stakes critique pipeline
        # =====================================================================
        print(f"[DEBUG] ========== CALLING run_high_stakes_with_critique ==========")
        print(f"[DEBUG] provider={provider}, model={model}")
        print(f"[DEBUG] system_prompt length: {len(task.system_prompt)}")
        
        # v3.5: Enter critique pipeline stage
        if stage_trace:
            stage_trace.enter_stage("critique_pipeline", provider=provider, model=model)
        
        result = await run_high_stakes_with_critique(
            db=db,
            envelope=envelope,
            provider_id=provider,
            model_id=model,
            trace=trace,
            use_json_critique=True,
        )
        print(f"[DEBUG] ========== run_high_stakes_with_critique RETURNED ==========")
        print(f"[DEBUG] Result type: {type(result)}")

        usage_obj = getattr(result, "usage", None)
        if usage_obj is None and isinstance(result, dict):
            usage_obj = result.get("usage")
        hs_prompt_tokens, hs_completion_tokens = extract_usage_tokens(usage_obj)
        duration_ms = max(0, int(loop.time() * 1000) - started_ms)
        print(f"[DEBUG] Usage: prompt={hs_prompt_tokens}, completion={hs_completion_tokens}, duration={duration_ms}ms")

        error_message = getattr(result, "error_message", None)
        content = getattr(result, "content", None)
        if isinstance(result, dict):
            error_message = error_message or result.get("error_message") or result.get("error")
            content = content if content is not None else result.get("content")
        
        print(f"[DEBUG] error_message={error_message}, content_len={len(content) if content else 0}")

        if error_message:
            print(f"[DEBUG] ERROR: {error_message}")
            if trace and not trace_finished:
                trace.log_model_call("primary", provider, model, "primary", hs_prompt_tokens, hs_completion_tokens, duration_ms, success=False, error=str(error_message))
                trace.finalize(success=False, error_message=str(error_message))
                trace_finished = True
            yield "data: " + json.dumps({'type': 'error', 'error': str(error_message)}) + "\n\n"
            return

        content = content or ""
        if content:
            print(f"[DEBUG] Content preview (first 300): {content[:300]}")
        
        # =====================================================================
        # Stage 3 verification
        # =====================================================================
        if needs_verification and spec_id and spec_hash:
            print(f"[DEBUG] ========== STAGE 3 VERIFICATION ==========")
            verified, error_msg = verify_and_store_stage3(
                job_id=envelope.job_id,
                stage_name="high_stakes_streaming",
                spec_id=spec_id,
                expected_spec_hash=spec_hash,
                raw_output=content,
                provider=provider,
                model=model,
            )
            print(f"[DEBUG] verified={verified}, error_msg={error_msg}")
            
            if not verified:
                print(f"[DEBUG] VERIFICATION FAILED")
                if trace and not trace_finished:
                    trace.log_model_call("primary", provider, model, "primary", hs_prompt_tokens, hs_completion_tokens, duration_ms, success=False, error=error_msg)
                    trace.finalize(success=False, error_message=error_msg)
                    trace_finished = True
                yield "data: " + json.dumps({
                    'type': 'error',
                    'error': error_msg,
                    'error_type': 'STAGE_SPEC_HASH_MISMATCH',
                    'job_id': envelope.job_id,
                }) + "\n\n"
                return

        print(f"[DEBUG] ========== PARSING REASONING TAGS ==========")
        print(f"[DEBUG] Content length: {len(content)}")
        print(f"[DEBUG] Has THINKING: {'<THINKING>' in content.upper()}")
        print(f"[DEBUG] Has ANSWER: {'<ANSWER>' in content.upper()}")
        
        final_answer, reasoning = parse_reasoning_tags(content)
        
        print(f"[DEBUG] final_answer={len(final_answer)} chars, reasoning={len(reasoning)} chars")
        if not final_answer:
            print(f"[DEBUG] WARNING: final_answer is EMPTY!")

        # =====================================================================
        # Check for user questions (policy violation)
        # =====================================================================
        if (job_type_str or "").strip().lower() == "architecture_design":
            print(f"[DEBUG] ========== CHECKING FOR USER QUESTIONS ==========")
            has_questions = detect_user_questions(final_answer or "")
            print(f"[DEBUG] has_questions={has_questions}")
            
            if has_questions:
                print(f"[DEBUG] POLICY VIOLATION - rerouting to Spec Gate")
                from app.pot_spec.ledger import append_event
                from app.pot_spec.service import get_job_artifact_root

                job_root = get_job_artifact_root()
                append_event(
                    job_artifact_root=job_root,
                    job_id=envelope.job_id,
                    event={
                        "event": "POLICY_VIOLATION_STAGE_ASKED_QUESTIONS",
                        "job_id": envelope.job_id,
                        "stage": "HIGH_STAKES_PRIMARY",
                        "status": "rejected",
                    },
                )

                # v3.4: Use helper function for proper env var precedence
                spec_gate_provider, spec_gate_model = _get_spec_gate_config()
                print(f"[SPEC_GATE_AUDIT] Reroute - Provider: {spec_gate_provider}, Model: {spec_gate_model}")

                # v3.3: Use run_spec_gate_v2 with project_id for reroute case too
                reroute_result = await run_spec_gate_v2(
                    db=db,
                    job_id=envelope.job_id,
                    user_intent=message,
                    provider_id=spec_gate_provider,
                    model_id=spec_gate_model,
                    project_id=project_id,  # CRITICAL: enables DB persistence
                    reroute_reason="Downstream stage asked the user questions. Only Spec Gate may ask questions.",
                    downstream_output_excerpt=(final_answer or "")[:2000],
                )
                
                spec_id = reroute_result.spec_id
                spec_hash = reroute_result.spec_hash
                open_questions = reroute_result.open_questions

                yield "data: " + json.dumps({
                    "type": "pause",
                    "pause_state": "needs_spec_clarification",
                    "job_id": envelope.job_id,
                    "spec_id": spec_id,
                    "spec_hash": spec_hash,
                    "open_questions": open_questions,
                    "artifacts_written": f"jobs/{envelope.job_id}/spec/ + jobs/{envelope.job_id}/ledger/",
                }) + "\n\n"
                return

        # =====================================================================
        # Stream response to UI
        # =====================================================================
        print(f"[DEBUG] ========== STREAMING TO UI ==========")
        print(f"[DEBUG] Streaming {len(final_answer)} chars")
        
        if trace and not trace_finished:
            trace.log_model_call("primary", provider, model, "primary", hs_prompt_tokens, hs_completion_tokens, duration_ms, success=True)

        chunk_size = 50
        chunks_sent = 0
        for i in range(0, len(final_answer), chunk_size):
            chunk = final_answer[i : i + chunk_size]
            yield "data: " + json.dumps({'type': 'token', 'content': chunk}) + "\n\n"
            chunks_sent += 1
            await asyncio.sleep(0.01)
        
        print(f"[DEBUG] Streamed {chunks_sent} chunks")

        if enable_reasoning and reasoning:
            yield "data: " + json.dumps({'type': 'reasoning', 'content': reasoning}) + "\n\n"

        # =====================================================================
        # Save to memory
        # =====================================================================
        print(f"[DEBUG] ========== SAVING TO MEMORY ==========")
        memory_service.create_message(db, memory_schemas.MessageCreate(project_id=project_id, role="user", content=message, provider="local"))
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id,
                role="assistant",
                content=final_answer,
                provider=provider,
                model=model,
                reasoning=reasoning if reasoning else None,
            ),
        )

        if trace and not trace_finished:
            trace.finalize(success=True)
            trace_finished = True

        # v3.5: Exit critique pipeline and finish trace
        if stage_trace:
            stage_trace.log_model_call(provider, model, "critique_pipeline", 
                                      tokens_used=hs_prompt_tokens + hs_completion_tokens,
                                      success=True)
            stage_trace.exit_stage("critique_pipeline", success=True,
                                  tokens_used=hs_prompt_tokens + hs_completion_tokens)
            stage_trace.finish(success=True, outcome="completed")

        print(f"[DEBUG] ========== STREAMING COMPLETE ==========")
        yield "data: " + json.dumps({'type': 'done', 'provider': provider, 'model': model, 'total_length': len(final_answer)}) + "\n\n"

    except asyncio.CancelledError:
        print(f"[DEBUG] ========== CANCELLED ==========")
        if trace and not trace_finished:
            trace.log_warning("STREAM", "client_disconnect")
            trace.finalize(success=False, error_message="client_disconnect")
            trace_finished = True
        if stage_trace:
            stage_trace.finish(success=False, error="client_disconnect")
        raise
    except Exception as e:
        logger.exception("[high_stakes] Stream failed: %s", e)
        print(f"[DEBUG] ========== EXCEPTION ==========")
        print(f"[DEBUG] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        if trace and not trace_finished:
            trace.log_model_call("primary", provider, model, "primary", 0, 0, 0, success=False, error=str(e))
            trace.finalize(success=False, error_message=str(e))
            trace_finished = True
        if stage_trace:
            stage_trace.finish(success=False, error=str(e))
        yield "data: " + json.dumps({'type': 'error', 'error': str(e)}) + "\n\n"
        return