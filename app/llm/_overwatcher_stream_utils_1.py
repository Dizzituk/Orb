from __future__ import annotations
import asyncio
import json
import logging
import os
from app.llm._overwatcher_stream_utils import _build_evidence_bundle, _get_overwatcher_provider_model, _load_artifact_bindings, _resolve_job_id, _validate_artifact_bindings, sse_error, sse_event, sse_token
from sqlalchemy.orm import Session
from typing import Any, AsyncGenerator, Callable, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
memory_service = None
memory_schemas = None
specs_service = None
get_work_artifacts = None
mark_job_complete = None
mark_job_failed = None
OVERWATCHER_AVAILABLE = True
run_overwatcher_command = None
OverwatcherCommandResult = None


ARTIFACT_ROOT = os.getenv("ORB_JOB_ARTIFACT_ROOT", r"D:\Orb\jobs")

def create_overwatcher_llm_fn() -> Optional[Callable]:
    """Create LLM call function for Overwatcher reasoning.
    
    v3.7: Uses call_llm_text from streaming.py which handles:
    - All provider routing (openai/anthropic/gemini)
    - Retry logic for transient failures  
    - Non-streaming fallback for OpenAI
    
    Returns:
        Async callable matching Overwatcher contract, or None if unavailable
    """
    # Import the unified LLM call function
    try:
        from app.llm.streaming import call_llm_text, get_available_streaming_provider
    except ImportError as e:
        logger.warning(f"[overwatcher_stream] Cannot import streaming module: {e}")
        return None
    
    # Check if any provider is available
    if not get_available_streaming_provider():
        logger.warning("[overwatcher_stream] No LLM providers available (missing API keys)")
        return None
    
    # Get default provider/model from stage config
    try:
        default_provider, default_model = _get_overwatcher_provider_model()
    except Exception as e:
        logger.warning(f"[overwatcher_stream] Could not get Overwatcher config: {e}")
        default_provider = "openai"
        default_model = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4.1-mini")
    
    async def llm_call_fn(
        messages: list,
        provider_id: Optional[str] = None,
        model_id: Optional[str] = None,
        max_tokens: int = 2000,
        **kwargs,
    ) -> str:
        """
        LLM call function matching Overwatcher contract.
        
        Args:
            messages: OpenAI-style message list [{"role": ..., "content": ...}]
            provider_id: Provider to use (defaults to Overwatcher config)
            model_id: Model to use (defaults to Overwatcher config)
            max_tokens: Max output tokens (passed through to call_llm_text)
        
        Returns:
            Response text as string
        """
        use_provider = provider_id or default_provider
        use_model = model_id or default_model
        
        logger.info(f"[overwatcher_llm] Calling {use_provider}/{use_model}")
        
        # Extract system prompt from messages (if present)
        system_prompt = ""
        user_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            else:
                user_messages.append(msg)
        
        # Get user prompt (last user message)
        user_prompt = ""
        if user_messages:
            user_prompt = user_messages[-1].get("content", "")
        
        try:
            result = await call_llm_text(
                provider=use_provider,
                model=use_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                messages=user_messages if len(user_messages) > 1 else None,
                max_tokens=max_tokens,
            )
            logger.info(f"[overwatcher_llm] Response length: {len(result)}")
            return result
        except Exception as e:
            logger.exception(f"[overwatcher_llm] LLM call failed: {e}")
            raise
    
    return llm_call_fn

async def generate_overwatcher_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[Any] = None,
    conversation_id: Optional[str] = None,
    job_id: Optional[str] = None,
    force_llm: bool = True,
) -> AsyncGenerator[str, None]:
    """
    Generate SSE stream for Overwatcher execution (v3.5).
    
    Enhanced flow:
    1. Resolve job context (multiple fallbacks)
    2. Load spec data for acceptance criteria
    3. Load and validate artifact bindings
    4. Build evidence bundle with actual file content
    5. Run Overwatcher with evidence
    6. Stream results
    """
    response_parts = []
    
    def emit(content: str):
        response_parts.append(content)
    
    ow_provider, ow_model = _get_overwatcher_provider_model()
    
    try:
        yield sse_token("🔧 **Overwatcher Execution**\n\n")
        emit("🔧 **Overwatcher Execution**\n\n")
        
        # =====================================================================
        # Step 1: Load spec data (needed for job resolution and evidence)
        # =====================================================================
        
        spec_data = None
        spec_id = None
        
        try:
            if specs_service:
                # Try validated spec first, then draft
                latest_spec = specs_service.get_latest_validated_spec(db, project_id)
                if not latest_spec:
                    latest_spec = specs_service.get_latest_draft_spec(db, project_id)
                
                if latest_spec:
                    spec_id = latest_spec.spec_id
                    if isinstance(latest_spec.content_json, str):
                        spec_data = json.loads(latest_spec.content_json)
                    else:
                        spec_data = latest_spec.content_json
                    logger.info(f"[overwatcher_stream] Loaded spec: {spec_id}")
        except Exception as e:
            logger.warning(f"[overwatcher_stream] Could not load spec: {e}")
        
        # =====================================================================
        # Step 2: Resolve job context (v3.5 enhanced)
        # =====================================================================
        
        resolved_job_id, resolution_method = _resolve_job_id(
            db, project_id, spec_id=spec_id, provided_job_id=job_id
        )
        
        if resolved_job_id:
            job_msg = f"📁 **Job ID:** `{resolved_job_id}` (resolved via {resolution_method})\n\n"
        else:
            job_msg = "⚠️ **Warning:** Could not resolve job_id - artifact loading may fail\n\n"
        
        yield sse_token(job_msg)
        emit(job_msg)
        
        # =====================================================================
        # Step 3: Load work artifacts
        # =====================================================================
        
        work_artifacts = None
        if resolved_job_id and get_work_artifacts:
            try:
                work_artifacts = get_work_artifacts(db, resolved_job_id)
            except Exception as e:
                logger.warning(f"[overwatcher_stream] Could not get work artifacts: {e}")
        
        # =====================================================================
        # Step 4: Load and validate artifact bindings
        # =====================================================================
        
        artifact_bindings = _load_artifact_bindings(resolved_job_id, work_artifacts, spec_data=spec_data)
        bindings_valid, binding_issues = _validate_artifact_bindings(artifact_bindings)
        
        if artifact_bindings:
            binding_msg = f"📦 **Artifact Bindings:** {len(artifact_bindings)} loaded\n\n"
            for b in artifact_bindings:
                path = b.get("path", "unknown")
                content = b.get("content_verbatim", "")
                binding_msg += f"  - `{path}`"
                if content:
                    preview = content[:40].replace('\n', ' ')
                    binding_msg += f"\n    Content: \"{preview}{'...' if len(content) > 40 else ''}\""
                binding_msg += "\n"
            yield sse_token(binding_msg + "\n")
            emit(binding_msg + "\n")
        else:
            warn_msg = "⚠️ **No artifact bindings found.**\n"
            warn_msg += "Overwatcher needs artifact bindings to verify implementation.\n"
            if binding_issues:
                warn_msg += f"Issues: {', '.join(binding_issues)}\n"
            warn_msg += "\n"
            yield sse_token(warn_msg)
            emit(warn_msg)
        
        # =====================================================================
        # Step 5: Build evidence bundle (v3.5 NEW)
        # =====================================================================
        
        evidence = _build_evidence_bundle(artifact_bindings, spec_data, resolved_job_id)
        
        yield sse_token("📋 **Evidence Bundle Built**\n")
        emit("📋 **Evidence Bundle Built**\n")
        
        if evidence["artifacts"]:
            for art in evidence["artifacts"]:
                status_icon = {
                    "MATCH": "✅",
                    "EXISTS": "✅",
                    "MISMATCH": "⚠️",
                    "MISSING": "❌",
                    "READ_ERROR": "⚠️",
                }.get(art.get("verification"), "❓")
                
                evidence_line = f"  {status_icon} `{art.get('artifact_id')}`: {art.get('verification')}"
                if art.get("actual_content"):
                    content_preview = art["actual_content"][:30].replace('\n', ' ')
                    evidence_line += f" (content: \"{content_preview}...\")"
                evidence_line += "\n"
                yield sse_token(evidence_line)
                emit(evidence_line)
        
        overall_msg = f"\n**Overall Verification:** {evidence.get('overall_result', 'UNKNOWN')}\n\n"
        yield sse_token(overall_msg)
        emit(overall_msg)
        
        # =====================================================================
        # Step 6: Create LLM function
        # =====================================================================
        
        llm_call_fn = create_overwatcher_llm_fn()
        
        if llm_call_fn is None:
            if force_llm:
                error_msg = "❌ LLM function unavailable.\n"
                yield sse_token(error_msg)
                emit(error_msg)
                yield sse_error("LLM unavailable")
                yield sse_event("done", error="LLM unavailable")
                return
            else:
                yield sse_token("⚠️ Running without LLM (smoke test mode)\n\n")
                emit("⚠️ Running without LLM (smoke test mode)\n\n")
        else:
            yield sse_token(f"✅ LLM attached: `{ow_provider}/{ow_model}`\n\n")
            emit(f"✅ LLM attached: `{ow_provider}/{ow_model}`\n\n")
        
        # =====================================================================
        # Step 7: Run Overwatcher with evidence
        # =====================================================================
        
        overwatcher_error = None  # Track errors for done event
        
        if OVERWATCHER_AVAILABLE and run_overwatcher_command:
            yield sse_token("📋 Running Overwatcher command flow...\n\n")
            emit("📋 Running Overwatcher command flow...\n\n")
            
            yield sse_event("execution_started", stage="overwatcher", job_id=resolved_job_id)
            
            try:
                result: OverwatcherCommandResult = await run_overwatcher_command(
                    project_id=project_id,
                    job_id=resolved_job_id,
                    message=message,
                    db_session=db,
                    llm_call_fn=llm_call_fn,
                    use_smoke_test=(llm_call_fn is None),
                    artifact_bindings=artifact_bindings,  # Pass bindings
                    evidence_bundle=evidence,  # Pass evidence
                )
                
                # Stream stage trace
                yield sse_token("### Stage Trace\n\n")
                emit("### Stage Trace\n\n")
                
                for entry in result.stage_trace:
                    stage = entry.get("stage", "UNKNOWN")
                    status = entry.get("status", "")
                    line = f"- `[{stage}]` {status}\n"
                    yield sse_token(line)
                    emit(line)
                    await asyncio.sleep(0.02)
                
                # Show spec info if available
                if result.spec:
                    spec_info = f"\n**Spec:** `{result.spec.spec_id}` (hash: `{result.spec.spec_hash[:12]}...`)\n\n"
                    yield sse_token(spec_info)
                    emit(spec_info)
                
                # Decision and artifacts
                decision_icon = {"APPROVED": "✅", "REJECTED": "❌", "NEEDS_INFO": "❓"}.get(
                    result.decision, "❓"
                )
                decision_msg = f"**Overwatcher Decision:** {decision_icon} {result.decision}\n\n"
                yield sse_token(decision_msg)
                emit(decision_msg)
                
                if result.reason:
                    yield sse_token(f"> {result.reason}\n\n")
                    emit(f"> {result.reason}\n\n")
                
                if result.artifacts_written:
                    yield sse_token(f"**Artifacts Written:** {len(result.artifacts_written)}\n\n")
                    emit(f"**Artifacts Written:** {len(result.artifacts_written)}\n\n")
                    for art in result.artifacts_written:
                        yield sse_token(f"  - `{art}`\n")
                        emit(f"  - `{art}`\n")
                
                # Final status
                if result.success:
                    final_msg = f"\n✅ **Job Complete**\n\nJob `{result.job_id}` executed successfully.\n"
                    if mark_job_complete and resolved_job_id:
                        try:
                            mark_job_complete(db, resolved_job_id, status="completed")
                        except Exception as e:
                            logger.warning(f"Failed to mark job complete: {e}")
                    yield sse_event("job_complete", job_id=result.job_id, status="completed")
                else:
                    final_msg = f"\n❌ **Job Failed**\n\n"
                    if result.error:
                        final_msg += f"Error: {result.error}\n"
                    if mark_job_failed and resolved_job_id:
                        try:
                            mark_job_failed(db, resolved_job_id, error=result.error)
                        except Exception as e:
                            logger.warning(f"Failed to mark job failed: {e}")
                    yield sse_event("job_failed", job_id=result.job_id, error=result.error)
                
                yield sse_token(final_msg)
                emit(final_msg)
                
            except Exception as e:
                logger.exception(f"[overwatcher_stream] Overwatcher command failed: {e}")
                error_msg = f"❌ **Overwatcher Error:** {e}\n"
                yield sse_token(error_msg)
                emit(error_msg)
                overwatcher_error = str(e)
        
        else:
            # Fallback when Overwatcher module not available
            yield sse_token("⚠️ Overwatcher module not available. Running verification only...\n\n")
            emit("⚠️ Overwatcher module not available. Running verification only...\n\n")
            
            # Report evidence results as final output
            if evidence.get("overall_result") == "PASS":
                yield sse_token("✅ **Verification PASSED** - All artifacts exist with expected content.\n")
                emit("✅ **Verification PASSED** - All artifacts exist with expected content.\n")
            else:
                yield sse_token(f"❌ **Verification FAILED** - {evidence.get('overall_result')}\n")
                emit(f"❌ **Verification FAILED** - {evidence.get('overall_result')}\n")
        
        # Persist to message history for cross-model context continuity
        if memory_service and memory_schemas:
            try:
                full_response = "".join(response_parts)
                memory_service.create_message(
                    db,
                    memory_schemas.MessageCreate(
                        project_id=project_id,
                        role="assistant",
                        content=full_response,
                        provider=ow_provider,
                        model=ow_model,
                    ),
                )
            except Exception as e:
                logger.warning("[overwatcher_stream] Failed to save message: %s", e)

        # Done event - ALWAYS yield this
        yield sse_event(
            "done",
            provider=ow_provider,
            model=ow_model,
            job_id=resolved_job_id,
            evidence_result=evidence.get("overall_result"),
            error=overwatcher_error,
        )
        
    except Exception as e:
        logger.exception(f"[overwatcher_stream] Fatal error: {e}")
        yield sse_error(str(e))
        yield sse_event("done", error=str(e))
