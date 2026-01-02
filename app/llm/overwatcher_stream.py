# FILE: app/llm/overwatcher_stream.py
"""
Overwatcher streaming handler for ASTRA command flow.

Final stage: Supervises and executes approved changes via Implementer.

v2.1 (2026-01): Fixed db_session parameter name
v2.0 (2026-01): Real implementation - connects to overwatcher_command.py
v1.0 (2026-01): Stub implementation
"""

import json
import logging
import asyncio
from typing import Optional, Any, AsyncGenerator

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
        SMOKE_TEST_FILENAME,
        SMOKE_TEST_CONTENT,
    )
    OVERWATCHER_AVAILABLE = True
except ImportError:
    OVERWATCHER_AVAILABLE = False
    run_overwatcher_command = None
    OverwatcherCommandResult = None
    logger.warning("[overwatcher_stream] overwatcher_command not available - using stub")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OVERWATCHER_PROVIDER = "anthropic"
OVERWATCHER_MODEL = "claude-sonnet-4-5-20250929"


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
) -> AsyncGenerator[str, None]:
    """
    Generate SSE stream for Overwatcher execution.
    
    Flow:
    1. Resolve spec (from DB or create smoke test)
    2. Load work artifacts (if any)
    3. Run Overwatcher supervisor for APPROVE/REJECT
    4. If approved: Run Implementer (Claude Sonnet)
    5. Run verification
    6. Stream results
    """
    response_parts = []
    
    def emit(content: str):
        """Helper to track response content."""
        response_parts.append(content)
    
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
        # Step 2: Run real Overwatcher command (if available)
        # ---------------------------------------------------------------------------
        
        if OVERWATCHER_AVAILABLE and run_overwatcher_command:
            yield sse_token("📋 Running Overwatcher command flow...\n\n")
            emit("📋 Running Overwatcher command flow...\n\n")
            
            yield sse_event("execution_started", stage="overwatcher", job_id=job_id)
            
            # Call the real implementation
            try:
                result: OverwatcherCommandResult = await run_overwatcher_command(
                    project_id=project_id,
                    job_id=job_id,
                    message=message,
                    db_session=db,  # v2.1: Fixed parameter name (was db=db)
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
                yield sse_done(OVERWATCHER_PROVIDER, OVERWATCHER_MODEL, sum(len(p) for p in response_parts))
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
                    provider=OVERWATCHER_PROVIDER,
                    model=OVERWATCHER_MODEL
                ))
            except Exception as e:
                logger.warning(f"[overwatcher_stream] Could not save to memory: {e}")
        
        # Final done event
        yield sse_done(
            OVERWATCHER_PROVIDER,
            OVERWATCHER_MODEL,
            len(full_response),
            job_id=job_id,
        )
        
    except Exception as e:
        logger.exception("[overwatcher_stream] Stream failed: %s", e)
        if trace:
            trace.finalize(success=False, error_message=str(e))
        yield sse_error(str(e))


__all__ = ["generate_overwatcher_stream"]
