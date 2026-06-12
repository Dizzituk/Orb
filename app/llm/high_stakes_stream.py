# FILE: app/llm/high_stakes_stream.py
# Purpose: High-stakes critique stream generator.
# Called-by: app.llm.routing.handler_registry
# Depends-on: app.llm.audit_logger, app.llm.router, app.llm.schemas, app.llm.stage_trace (+4 more)
# Last-renovated: 2026-06-11
"""
High-stakes critique stream generator.

v3.7 (2026-04): Spec Gate auto-trigger removed. Spec Gate is now invoked manually
                from within the pipeline (critical_pipeline / agentic_builder /
                overwatcher), never from chat-mode classification. Dead code and
                unused imports stripped. Stage 3 spec-hash verification removed
                (was only reachable via the Spec Gate path).
v3.5 (2026-01): Integrated stage tracing for full command visibility
v3.4 (2026-01): Fixed spec gate model env var precedence (OPENAI_SPEC_GATE_MODEL takes priority)
v3.3 (2026-01): Uses run_spec_gate_v2 with project_id for DB persistence (restart survival)
v3.2 (2025-12): Added comprehensive debug logging for streaming diagnosis
"""

import json
import asyncio
import logging
from typing import Optional, List

from sqlalchemy.orm import Session

from app.memory import service as memory_service, schemas as memory_schemas
from app.llm.schemas import LLMTask
from app.llm.audit_logger import RoutingTrace

from app.llm.router import (
    run_high_stakes_with_critique,
    synthesize_envelope_from_task,
)

from .stream_utils import (
    parse_reasoning_tags,
    extract_usage_tokens,
)

# v3.5: Stage tracing
try:
    from .stage_trace import StageTrace
    _STAGE_TRACE_AVAILABLE = True
except ImportError:
    _STAGE_TRACE_AVAILABLE = False
    StageTrace = None

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
    """Generate SSE stream for the high-stakes critique pipeline.

    This function:
    1. Builds the LLMTask + envelope from caller-supplied context
    2. Runs the high-stakes critique pipeline (primary + critique + optional revision)
    3. Parses reasoning tags from the response
    4. Streams the final answer to the UI
    5. Persists user + assistant messages to memory

    Spec Gate is NOT invoked from here (v3.7). The pipeline invokes it explicitly
    when it needs a structured spec; chat-mode routing never touches it.
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
        # Run high-stakes critique pipeline
        # =====================================================================
        print(f"[DEBUG] ========== CALLING run_high_stakes_with_critique ==========")
        print(f"[DEBUG] provider={provider}, model={model}")
        print(f"[DEBUG] system_prompt length: {len(task.system_prompt)}")

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
                trace.log_model_call(
                    "primary", provider, model, "primary",
                    hs_prompt_tokens, hs_completion_tokens, duration_ms,
                    success=False, error=str(error_message),
                )
                trace.finalize(success=False, error_message=str(error_message))
                trace_finished = True
            yield "data: " + json.dumps({'type': 'error', 'error': str(error_message)}) + "\n\n"
            return

        content = content or ""
        if content:
            print(f"[DEBUG] Content preview (first 300): {content[:300]}")

        # =====================================================================
        # Parse reasoning tags
        # =====================================================================
        print(f"[DEBUG] ========== PARSING REASONING TAGS ==========")
        print(f"[DEBUG] Content length: {len(content)}")
        print(f"[DEBUG] Has THINKING: {'<THINKING>' in content.upper()}")
        print(f"[DEBUG] Has ANSWER: {'<ANSWER>' in content.upper()}")

        final_answer, reasoning = parse_reasoning_tags(content)

        print(f"[DEBUG] final_answer={len(final_answer)} chars, reasoning={len(reasoning)} chars")
        if not final_answer:
            print(f"[DEBUG] WARNING: final_answer is EMPTY!")

        # =====================================================================
        # Stream response to UI
        # =====================================================================
        print(f"[DEBUG] ========== STREAMING TO UI ==========")
        print(f"[DEBUG] Streaming {len(final_answer)} chars")

        if trace and not trace_finished:
            trace.log_model_call(
                "primary", provider, model, "primary",
                hs_prompt_tokens, hs_completion_tokens, duration_ms, success=True,
            )

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
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id, role="user", content=message, provider="local",
            ),
        )
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

        if stage_trace:
            stage_trace.log_model_call(
                provider, model, "critique_pipeline",
                tokens_used=hs_prompt_tokens + hs_completion_tokens, success=True,
            )
            stage_trace.exit_stage(
                "critique_pipeline", success=True,
                tokens_used=hs_prompt_tokens + hs_completion_tokens,
            )
            stage_trace.finish(success=True, outcome="completed")

        print(f"[DEBUG] ========== STREAMING COMPLETE ==========")
        yield "data: " + json.dumps({
            'type': 'done', 'provider': provider, 'model': model,
            'total_length': len(final_answer),
        }) + "\n\n"

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
            trace.log_model_call(
                "primary", provider, model, "primary",
                0, 0, 0, success=False, error=str(e),
            )
            trace.finalize(success=False, error_message=str(e))
            trace_finished = True
        if stage_trace:
            stage_trace.finish(success=False, error=str(e))
        yield "data: " + json.dumps({'type': 'error', 'error': str(e)}) + "\n\n"
        return
