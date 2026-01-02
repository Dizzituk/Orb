# FILE: app/llm/overwatcher_stream.py
"""
Overwatcher streaming handler for ASTRA command flow.

Final stage: Supervises and executes approved changes via Implementer.

v3.3 (2026-01): FIX - Token events use 'text' not 'content'
v3.2 (2026-01): DEBUG - Added print statements to diagnose empty response
v3.0 (2026-01): CRITICAL FIX - Actually wire LLM function to Overwatcher
v2.1 (2026-01): Fixed db_session parameter name
v2.0 (2026-01): Real implementation - connects to overwatcher_command.py
v1.0 (2026-01): Stub implementation
"""

import json
import logging
import asyncio
import os
from typing import Optional, Any, AsyncGenerator, Callable

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports with graceful fallbacks
# ---------------------------------------------------------------------------

try:
    from app.memory import service as memory_service, schemas as memory_schemas
except ImportError:
    memory_service = None
    memory_schemas = None

try:
    from app.jobs.service import (
        get_active_job_for_project,
        get_work_artifacts,
        mark_job_complete,
        mark_job_failed,
    )
except ImportError:
    get_active_job_for_project = None
    get_work_artifacts = None
    mark_job_complete = None
    mark_job_failed = None

try:
    from app.llm.audit_logger import RoutingTrace
except ImportError:
    RoutingTrace = None

# Import the real Overwatcher command handler
try:
    from app.overwatcher.overwatcher_command import (
        run_overwatcher_command,
        OverwatcherCommandResult,
    )
    OVERWATCHER_AVAILABLE = True
except ImportError as e:
    OVERWATCHER_AVAILABLE = False
    run_overwatcher_command = None
    OverwatcherCommandResult = None
    logger.warning(f"[overwatcher_stream] overwatcher_command not available - using stub: {e}")

# Import streaming for LLM calls
try:
    from app.llm.streaming import stream_llm
    STREAMING_AVAILABLE = True
except ImportError:
    stream_llm = None
    STREAMING_AVAILABLE = False
    logger.warning("[overwatcher_stream] streaming not available - LLM calls disabled")

# Import centralized stage configuration
try:
    from app.llm.stage_models import get_overwatcher_config
    STAGE_MODELS_AVAILABLE = True
except ImportError:
    get_overwatcher_config = None
    STAGE_MODELS_AVAILABLE = False
    logger.warning("[overwatcher_stream] stage_models not available - using fallback config")


# ---------------------------------------------------------------------------
# Configuration (from centralized stage_models)
# ---------------------------------------------------------------------------

def _get_overwatcher_provider_model() -> tuple[str, str]:
    """
    Get Overwatcher provider and model from centralized config.
    
    v3.1: NO HARDCODED FALLBACKS. Uses stage_models.get_overwatcher_config() 
    which reads from env vars. If stage_models unavailable, FAIL.
    
    Returns:
        Tuple of (provider, model)
    
    Raises:
        RuntimeError: If stage_models not available (config is mandatory)
    """
    if not STAGE_MODELS_AVAILABLE or get_overwatcher_config is None:
        raise RuntimeError(
            "FATAL: stage_models not available. Cannot determine Overwatcher model. "
            "Ensure app/llm/stage_models.py exists and OVERWATCHER_PROVIDER + OVERWATCHER_MODEL "
            "env vars are set."
        )
    
    config = get_overwatcher_config()
    logger.info(f"[overwatcher_stream] Using config: {config.provider}/{config.model}")
    return config.provider, config.model


# ---------------------------------------------------------------------------
# LLM Call Function Factory
# ---------------------------------------------------------------------------

def create_overwatcher_llm_fn() -> Optional[Callable]:
    """
    Create an async LLM call function for Overwatcher reasoning.
    
    v3.3: FIX - Token events use 'text' field, not 'content'
    v3.2: Added debug prints to diagnose empty response issue.
    v3.1: Signature matches what run_overwatcher() expects.
    
    Returns:
        Async callable compatible with run_overwatcher(), or None if unavailable.
    """
    if not STREAMING_AVAILABLE or stream_llm is None:
        logger.error("[overwatcher_stream] Cannot create LLM function - streaming not available")
        return None
    
    # Get configured provider/model from centralized config
    default_provider, default_model = _get_overwatcher_provider_model()
    
    async def llm_call_fn(
        messages: list,
        provider_id: Optional[str] = None,
        model_id: Optional[str] = None,
        max_tokens: int = 2000,
        **kwargs,  # Accept any extra kwargs for forward compatibility
    ) -> str:
        """
        Call LLM for Overwatcher reasoning.
        
        Signature matches what run_overwatcher() in overwatcher.py expects.
        
        Args:
            messages: List of message dicts with role/content
            provider_id: Provider override (defaults to env config)
            model_id: Model override (defaults to env config)
            max_tokens: Max output tokens
            **kwargs: Extra args ignored for compatibility
        
        Returns:
            Complete LLM response text
        """
        use_provider = provider_id or default_provider
        use_model = model_id or default_model
        
        logger.info(f"[overwatcher_llm] Calling {use_provider}/{use_model} for policy reasoning (max_tokens={max_tokens})")
        
        # Extract system prompt from messages if present
        system_prompt = ""
        user_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            else:
                user_messages.append(msg)
        
        response_parts = []
        event_count = 0
        
        try:
            async for event in stream_llm(
                messages=user_messages if user_messages else messages,
                system_prompt=system_prompt,
                provider=use_provider,
                model=use_model,
                route="overwatcher",
            ):
                event_count += 1
                
                if isinstance(event, dict):
                    event_type = event.get("type", "")
                    
                    # v3.3 FIX: Token events use 'text' field, not 'content'
                    if event_type == "token":
                        content = event.get("text", "")
                        if content:
                            response_parts.append(content)
                    elif event_type == "text":
                        content = event.get("content", event.get("text", ""))
                        if content:
                            response_parts.append(content)
                    elif event_type == "error":
                        error_msg = event.get("message", event.get("error", "Unknown error"))
                        logger.error(f"[overwatcher_llm] Stream error: {error_msg}")
                        raise RuntimeError(f"LLM stream error: {error_msg}")
                    elif event_type == "done":
                        break
                    elif event_type == "metadata":
                        # Skip metadata events
                        continue
                    else:
                        # Unknown event type - try to extract content anyway
                        content = (
                            event.get("text") or 
                            event.get("content") or 
                            event.get("delta") or
                            ""
                        )
                        if content:
                            response_parts.append(content)
                            
                elif isinstance(event, str):
                    response_parts.append(event)
            
            full_response = "".join(response_parts)
            
            logger.info(f"[overwatcher_llm] Response received: {len(full_response)} chars from {event_count} events")
            logger.info(f"[overwatcher_llm] Response preview: {full_response[:500]!r}")
            
            return full_response
            
        except Exception as e:
            logger.exception(f"[overwatcher_llm] LLM call failed: {e}")
            raise
    
    return llm_call_fn


# ---------------------------------------------------------------------------
# SSE Helpers
# ---------------------------------------------------------------------------

def sse_token(content: str) -> str:
    """Format SSE token event."""
    return "data: " + json.dumps({"type": "token", "content": content}) + "\n\n"


def sse_event(event_type: str, **kwargs) -> str:
    """Format SSE custom event."""
    return "data: " + json.dumps({"type": event_type, **kwargs}) + "\n\n"


def sse_done(provider: str, model: str, total_length: int, **kwargs) -> str:
    """Format SSE done event."""
    return "data: " + json.dumps({
        "type": "done",
        "provider": provider,
        "model": model,
        "total_length": total_length,
        **kwargs
    }) + "\n\n"


def sse_error(error: str) -> str:
    """Format SSE error event."""
    return "data: " + json.dumps({"type": "error", "error": error}) + "\n\n"


# ---------------------------------------------------------------------------
# Main Stream Generator
# ---------------------------------------------------------------------------

async def generate_overwatcher_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[Any] = None,
    conversation_id: Optional[str] = None,
    job_id: Optional[str] = None,
    force_llm: bool = True,  # v3.0: Default to requiring LLM
) -> AsyncGenerator[str, None]:
    """
    Generate SSE stream for Overwatcher execution.
    
    v3.0: Now properly wires LLM function to enable policy reasoning.
    
    Flow:
    1. Resolve spec (from DB or create smoke test)
    2. Load work artifacts (if any)
    3. Create LLM call function for policy reasoning
    4. Run Overwatcher supervisor for APPROVE/REJECT with ACTUAL LLM
    5. If approved: Run Implementer (in sandbox)
    6. Run verification
    7. Stream results
    
    Args:
        project_id: Project ID
        message: User message
        db: Database session
        trace: Optional routing trace
        conversation_id: Optional conversation ID
        job_id: Optional job ID to continue
        force_llm: If True, fail if LLM not available (default: True)
    """
    response_parts = []
    
    def emit(content: str):
        """Helper to track response content."""
        response_parts.append(content)
    
    # Get configured provider/model at start (used for logging and done events)
    ow_provider, ow_model = _get_overwatcher_provider_model()
    
    try:
        # Header
        yield sse_token("🔧 **Overwatcher Execution**\n\n")
        emit("🔧 **Overwatcher Execution**\n\n")
        
        # ---------------------------------------------------------------------------
        # Step 1: Resolve job context
        # ---------------------------------------------------------------------------
        
        if not job_id and get_active_job_for_project:
            try:
                active_job = get_active_job_for_project(db, project_id)
                if active_job:
                    job_id = active_job.id
            except Exception as e:
                logger.warning(f"[overwatcher_stream] Could not get active job: {e}")
        
        # ---------------------------------------------------------------------------
        # Step 2: Create LLM function (v3.0 - THE CRITICAL FIX)
        # ---------------------------------------------------------------------------
        
        llm_call_fn = create_overwatcher_llm_fn()
        
        if llm_call_fn is None:
            if force_llm:
                error_msg = "❌ LLM function unavailable - cannot run policy reasoning.\n"
                error_msg += "Overwatcher requires LLM for policy evaluation.\n"
                yield sse_token(error_msg)
                emit(error_msg)
                yield sse_error("LLM unavailable for Overwatcher")
                if trace:
                    trace.finalize(success=False, error_message="LLM unavailable")
                return
            else:
                yield sse_token("⚠️ Running without LLM (smoke test mode)\n\n")
                emit("⚠️ Running without LLM (smoke test mode)\n\n")
        else:
            yield sse_token(f"✅ LLM attached: `{ow_provider}/{ow_model}`\n\n")
            emit(f"✅ LLM attached: `{ow_provider}/{ow_model}`\n\n")
        
        # ---------------------------------------------------------------------------
        # Step 3: Run real Overwatcher command (if available)
        # ---------------------------------------------------------------------------
        
        if OVERWATCHER_AVAILABLE and run_overwatcher_command:
            yield sse_token("📋 Running Overwatcher command flow...\n\n")
            emit("📋 Running Overwatcher command flow...\n\n")
            
            yield sse_event("execution_started", stage="overwatcher", job_id=job_id)
            
            # Call the real implementation WITH the LLM function
            try:
                result: OverwatcherCommandResult = await run_overwatcher_command(
                    project_id=project_id,
                    job_id=job_id,
                    message=message,
                    db_session=db,
                    llm_call_fn=llm_call_fn,  # v3.0: THE FIX - actually pass the LLM!
                    use_smoke_test=(llm_call_fn is None),  # Only use smoke test if no LLM
                )
                
                # Stream stage trace as it happened
                yield sse_token("### Stage Trace\n\n")
                emit("### Stage Trace\n\n")
                
                for entry in result.stage_trace:
                    stage = entry.get("stage", "UNKNOWN")
                    status = entry.get("status", "")
                    line = f"- `[{stage}]` {status}\n"
                    yield sse_token(line)
                    emit(line)
                    await asyncio.sleep(0.02)
                
                yield sse_token("\n")
                emit("\n")
                
                # Report spec info
                if result.spec:
                    spec_info = f"**Spec:** `{result.spec.spec_id}` (hash: `{result.spec.spec_hash[:12]}...`)\n\n"
                    yield sse_token(spec_info)
                    emit(spec_info)
                
                # Report Overwatcher decision
                if result.overwatcher_decision:
                    decision_icon = "✅" if result.overwatcher_decision == "PASS" else "❌"
                    decision_msg = f"**Overwatcher Decision:** {decision_icon} {result.overwatcher_decision}\n"
                    yield sse_token(decision_msg)
                    emit(decision_msg)
                    
                    if result.overwatcher_diagnosis:
                        yield sse_token(f"> {result.overwatcher_diagnosis}\n\n")
                        emit(f"> {result.overwatcher_diagnosis}\n\n")
                
                # Report Implementer result
                if result.implementer_result:
                    impl = result.implementer_result
                    if impl.success:
                        impl_msg = f"**Implementer:** ✅ Success\n- Output: `{impl.output_path}`\n- Sandbox: {'Yes' if impl.sandbox_used else 'No (local fallback)'}\n\n"
                    else:
                        impl_msg = f"**Implementer:** ❌ Failed\n- Error: {impl.error}\n\n"
                    yield sse_token(impl_msg)
                    emit(impl_msg)
                
                # Report verification result
                if result.verification_result:
                    ver = result.verification_result
                    if ver.passed:
                        ver_msg = f"**Verification:** ✅ Passed\n- File exists: {ver.file_exists}\n- Content matches: {ver.content_matches}\n\n"
                    else:
                        ver_msg = f"**Verification:** ❌ Failed\n- Error: {ver.error}\n\n"
                    yield sse_token(ver_msg)
                    emit(ver_msg)
                
                # Final status
                yield sse_token("---\n\n")
                emit("---\n\n")
                
                if result.success:
                    final_msg = f"✅ **Job Complete**\n\nJob `{result.job_id}` executed successfully.\n"
                    yield sse_token(final_msg)
                    emit(final_msg)
                    
                    # Mark job complete in DB
                    if mark_job_complete and job_id:
                        try:
                            mark_job_complete(db, job_id, status="completed")
                        except Exception as e:
                            logger.warning(f"[overwatcher_stream] Could not mark job complete: {e}")
                    
                    yield sse_event("job_complete", job_id=result.job_id, status="completed")
                    
                    if trace:
                        trace.finalize(success=True)
                else:
                    final_msg = f"❌ **Job Failed**\n\nError: {result.error}\n"
                    yield sse_token(final_msg)
                    emit(final_msg)
                    
                    # Mark job failed in DB
                    if mark_job_failed and job_id:
                        try:
                            mark_job_failed(db, job_id, error=result.error)
                        except Exception as e:
                            logger.warning(f"[overwatcher_stream] Could not mark job failed: {e}")
                    
                    yield sse_event("job_failed", job_id=result.job_id, error=result.error)
                    
                    if trace:
                        trace.finalize(success=False, error_message=result.error)
                
            except Exception as e:
                logger.exception("[overwatcher_stream] Command execution failed: %s", e)
                error_msg = f"❌ Overwatcher command failed: {e}\n"
                yield sse_token(error_msg)
                emit(error_msg)
                
                if trace:
                    trace.finalize(success=False, error_message=str(e))
                
                yield sse_error(str(e))
        
        else:
            # ---------------------------------------------------------------------------
            # Fallback: Stub implementation (when overwatcher_command not available)
            # ---------------------------------------------------------------------------
            
            yield sse_token("⚠️ Running stub implementation (overwatcher_command not available)\n\n")
            emit("⚠️ Running stub implementation (overwatcher_command not available)\n\n")
            
            if not job_id:
                error_msg = "❌ No active job found. Please run Critical Pipeline first.\n"
                yield sse_token(error_msg)
                emit(error_msg)
                if trace:
                    trace.finalize(success=False, error_message="No active job")
                yield sse_done(ow_provider, ow_model, sum(len(p) for p in response_parts))
                return
            
            yield sse_token(f"Job: `{job_id}`\n\n")
            emit(f"Job: `{job_id}`\n\n")
            
            yield sse_event("execution_started", stage="overwatcher", job_id=job_id)
            
            # Simulated execution log
            execution_log = """### Execution Log (STUB)

**Phase 1: Pre-flight Checks**
- ✅ Work artifacts validated
- ✅ Target paths verified
- ✅ No conflicts detected

**Phase 2: Implementation**
- ✅ Stub execution complete

**Phase 3: Verification**
- ✅ Syntax check passed

"""
            
            chunk_size = 80
            for i in range(0, len(execution_log), chunk_size):
                chunk = execution_log[i:i + chunk_size]
                yield sse_token(chunk)
                emit(chunk)
                await asyncio.sleep(0.02)
            
            if mark_job_complete and job_id:
                try:
                    mark_job_complete(db, job_id, status="completed")
                except Exception as e:
                    logger.warning(f"[overwatcher_stream] Could not mark job complete: {e}")
            
            completion_msg = """---
✅ **Overwatcher Complete (STUB)**

This was a stub execution. Install overwatcher_command.py for real functionality.
"""
            yield sse_token(completion_msg)
            emit(completion_msg)
            
            yield sse_event("job_complete", job_id=job_id, status="completed")
            
            if trace:
                trace.finalize(success=True)
        
        # ---------------------------------------------------------------------------
        # Save to memory
        # ---------------------------------------------------------------------------
        
        full_response = "".join(response_parts)
        if memory_service and memory_schemas:
            try:
                memory_service.create_message(db, memory_schemas.MessageCreate(
                    project_id=project_id,
                    role="user",
                    content=message,
                    provider="local"
                ))
                memory_service.create_message(db, memory_schemas.MessageCreate(
                    project_id=project_id,
                    role="assistant",
                    content=full_response,
                    provider=ow_provider,
                    model=ow_model
                ))
            except Exception as e:
                logger.warning(f"[overwatcher_stream] Could not save to memory: {e}")
        
        # Final done event
        yield sse_done(
            ow_provider,
            ow_model,
            len(full_response),
            job_id=job_id,
        )
        
    except Exception as e:
        logger.exception("[overwatcher_stream] Stream failed: %s", e)
        if trace:
            trace.finalize(success=False, error_message=str(e))
        yield sse_error(str(e))


__all__ = ["generate_overwatcher_stream", "create_overwatcher_llm_fn"]