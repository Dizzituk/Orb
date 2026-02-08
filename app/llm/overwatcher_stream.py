# FILE: app/llm/overwatcher_stream.py
"""
Overwatcher streaming handler for ASTRA command flow.

v3.7 (2026-01-04): LLM Wiring Fix - Use call_llm_text
- FIXED: Use call_llm_text from streaming.py instead of non-existent provider functions
- Removed broken imports: stream_openai_response, stream_anthropic_response, stream_gemini_response
- Simplified availability checking - uses get_available_streaming_provider() at runtime
- call_llm_text provides: retry logic, non-streaming fallback, unified provider routing

v3.6 (2026-01-04): LLM Wiring Fix (BROKEN - referenced non-existent functions)
v3.5 (2026-01-04): Job ID Resolution + Evidence Building Fixes
v3.4: Artifact Binding Support
v3.3: Token event 'text' field fix
v3.0: LLM function wiring
"""

import json
import logging
import asyncio
import os
import glob
from datetime import datetime
from typing import Optional, Any, AsyncGenerator, Callable, List, Dict

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
    from app.specs import service as specs_service
except ImportError:
    specs_service = None

try:
    from app.jobs.service import (
        get_active_job_for_project,
        get_work_artifacts,
        mark_job_complete,
        mark_job_failed,
        get_job_for_spec,  # NEW: Get job by spec_id
    )
except ImportError:
    get_active_job_for_project = None
    get_work_artifacts = None
    mark_job_complete = None
    mark_job_failed = None
    get_job_for_spec = None

try:
    from app.llm.audit_logger import RoutingTrace
except ImportError:
    RoutingTrace = None

try:
    from app.overwatcher.overwatcher_command import (
        run_overwatcher_command,
        OverwatcherCommandResult,
    )
    OVERWATCHER_AVAILABLE = True
except ImportError:
    run_overwatcher_command = None
    OverwatcherCommandResult = None
    OVERWATCHER_AVAILABLE = False

# v3.7: LLM availability is checked at runtime in create_overwatcher_llm_fn()
# No module-level streaming imports needed - call_llm_text handles all provider routing

try:
    from app.llm.stage_models import get_overwatcher_config
    STAGE_MODELS_AVAILABLE = True
except ImportError:
    get_overwatcher_config = None
    STAGE_MODELS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default artifact root
ARTIFACT_ROOT = os.getenv("ORB_JOB_ARTIFACT_ROOT", r"D:\Orb\jobs")


def _get_overwatcher_provider_model() -> tuple[str, str]:
    """Get Overwatcher model configuration."""
    if not STAGE_MODELS_AVAILABLE or get_overwatcher_config is None:
        raise RuntimeError("stage_models not available for Overwatcher config")
    
    config = get_overwatcher_config()
    return config.provider, config.model


# ---------------------------------------------------------------------------
# Job ID Resolution (v3.5 NEW)
# ---------------------------------------------------------------------------

def _resolve_job_id(
    db: Session,
    project_id: int,
    spec_id: Optional[str] = None,
    provided_job_id: Optional[str] = None,
) -> tuple[Optional[str], str]:
    """
    Resolve job_id using multiple fallback strategies.
    
    Returns (job_id, resolution_method)
    
    Strategies (in order):
    1. Use provided job_id if given
    2. Look up job by spec_id in database
    3. Get active job for project from database
    4. Find most recent cp-* folder in filesystem
    """
    # Strategy 1: Provided job_id
    if provided_job_id:
        logger.info(f"[job_resolve] Using provided job_id: {provided_job_id}")
        return provided_job_id, "provided"
    
    # Strategy 2: Look up by spec_id
    if spec_id and get_job_for_spec:
        try:
            job = get_job_for_spec(db, spec_id)
            if job:
                logger.info(f"[job_resolve] Found job {job.id} for spec {spec_id}")
                return job.id, "spec_lookup"
        except Exception as e:
            logger.warning(f"[job_resolve] Spec lookup failed: {e}")
    
    # Strategy 3: Get active job for project
    if get_active_job_for_project:
        try:
            active_job = get_active_job_for_project(db, project_id)
            if active_job:
                logger.info(f"[job_resolve] Found active job: {active_job.id}")
                return active_job.id, "active_project"
        except Exception as e:
            logger.warning(f"[job_resolve] Active job lookup failed: {e}")
    
    # Strategy 4: Filesystem fallback - find most recent cp-* folder
    jobs_dir = os.path.join(ARTIFACT_ROOT, "jobs")
    if os.path.isdir(jobs_dir):
        try:
            cp_folders = glob.glob(os.path.join(jobs_dir, "cp-*"))
            if cp_folders:
                # Sort by modification time, newest first
                cp_folders.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                newest = os.path.basename(cp_folders[0])
                logger.info(f"[job_resolve] Found recent job folder: {newest}")
                return newest, "filesystem"
        except Exception as e:
            logger.warning(f"[job_resolve] Filesystem lookup failed: {e}")
    
    logger.warning("[job_resolve] Could not resolve job_id via any method")
    return None, "none"


# ---------------------------------------------------------------------------
# Artifact Binding Loading (v3.5 Enhanced)
# ---------------------------------------------------------------------------

def _load_artifact_bindings(
    job_id: Optional[str],
    work_artifacts: Optional[Any] = None,
    spec_data: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Load artifact bindings from multiple sources.
    
    Priority:
    1. work_artifacts.artifact_bindings (from job system)
    2. work_artifacts.metadata.outputs (fallback)
    3. spec_data outputs + content_verbatim + location (fallback)
    4. Load from job directory filesystem (NEW)
    """
    bindings: List[Dict[str, Any]] = []
    
    # Try work_artifacts first
    if work_artifacts:
        if hasattr(work_artifacts, 'artifact_bindings') and work_artifacts.artifact_bindings:
            bindings = list(work_artifacts.artifact_bindings)
            logger.info("[artifact_load] Loaded %d bindings from work_artifacts", len(bindings))
            return bindings
        
        if hasattr(work_artifacts, 'metadata'):
            metadata = work_artifacts.metadata or {}
            outputs = metadata.get("outputs") or []
            for i, out in enumerate(outputs):
                bindings.append({
                    "artifact_id": f"output_{i+1}",
                    "action": out.get("action", "create"),
                    "path": out.get("path", ""),
                    "content_type": "text",
                    "content_verbatim": out.get("content", ""),
                })
            if bindings:
                logger.info("[artifact_load] Loaded %d bindings from metadata.outputs", len(bindings))
                return bindings
    
    # Try spec_data fallback
    if spec_data:
        outputs = spec_data.get("outputs", [])
        content_verbatim = spec_data.get("content_verbatim") or ""
        location = spec_data.get("location") or ""
        
        for i, out in enumerate(outputs):
            if isinstance(out, dict):
                name = out.get("name", "")
                path = out.get("path", "") or location
                content = out.get("content", "") or content_verbatim
                action = out.get("action", "create")
            else:
                name = str(out)
                path = location
                content = content_verbatim
                action = "create"
            
            if not name:
                continue
            
            full_path = os.path.join(path, name) if path and not path.endswith(name) else (path or name)
            
            bindings.append({
                "artifact_id": f"output_{i+1}",
                "action": action,
                "path": full_path,
                "content_type": "text",
                "content_verbatim": content,
            })
        
        if bindings:
            logger.info("[artifact_load] Constructed %d bindings from spec_data", len(bindings))
            return bindings
    
    # Try loading from job directory (NEW v3.5)
    if job_id:
        job_outputs_dir = os.path.join(ARTIFACT_ROOT, "jobs", job_id, "outputs")
        if os.path.isdir(job_outputs_dir):
            try:
                for filename in os.listdir(job_outputs_dir):
                    filepath = os.path.join(job_outputs_dir, filename)
                    if os.path.isfile(filepath):
                        # Read content for evidence
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content = f.read()
                        except Exception:
                            content = ""
                        
                        bindings.append({
                            "artifact_id": filename,
                            "action": "created",  # Already created by critical pipeline
                            "path": filepath,
                            "content_type": "text",
                            "content_verbatim": content,
                            "actual_file": True,  # Mark as actual file
                        })
                
                if bindings:
                    logger.info("[artifact_load] Loaded %d bindings from job directory", len(bindings))
                    return bindings
            except Exception as e:
                logger.warning(f"[artifact_load] Failed to read job outputs: {e}")
    
    logger.warning("[artifact_load] No artifact bindings found from any source")
    return bindings


def _validate_artifact_bindings(bindings: List[Dict[str, Any]]) -> tuple[bool, List[str]]:
    """Validate artifact bindings have required fields."""
    issues: List[str] = []
    
    if not bindings:
        issues.append("No artifact bindings found")
        return False, issues
    
    for i, binding in enumerate(bindings):
        if not binding.get("path"):
            issues.append(f"Binding {i+1}: missing 'path'")
        if not binding.get("action"):
            binding["action"] = "create"
    
    is_valid = len(issues) == 0
    return is_valid, issues


# ---------------------------------------------------------------------------
# Evidence Building (v3.5 NEW)
# ---------------------------------------------------------------------------

def _build_evidence_bundle(
    artifact_bindings: List[Dict[str, Any]],
    spec_data: Optional[Dict[str, Any]] = None,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build evidence bundle for Overwatcher verification.
    
    Evidence bundle contains:
    - expected: What the spec says should exist
    - actual: What actually exists (file content, hash)
    - result: Verification status (match/mismatch/missing)
    """
    evidence = {
        "job_id": job_id,
        "timestamp": datetime.utcnow().isoformat(),
        "artifacts": [],
        "acceptance_criteria": [],
        "verification_results": [],
    }
    
    # Add acceptance criteria from spec
    if spec_data:
        evidence["acceptance_criteria"] = spec_data.get("acceptance_criteria", [])
        evidence["objective"] = spec_data.get("objective", "")
        evidence["content_verbatim"] = spec_data.get("content_verbatim", "")
        evidence["location"] = spec_data.get("location", "")
    
    # Process each artifact binding
    for binding in artifact_bindings:
        artifact_evidence = {
            "artifact_id": binding.get("artifact_id", "unknown"),
            "expected_path": binding.get("path", ""),
            "expected_content": binding.get("content_verbatim", ""),
            "action": binding.get("action", "create"),
        }
        
        path = binding.get("path", "")
        
        # Check if file actually exists and read content
        if path and os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    actual_content = f.read()
                
                artifact_evidence["actual_content"] = actual_content
                artifact_evidence["actual_exists"] = True
                artifact_evidence["file_size_bytes"] = os.path.getsize(path)
                
                # Compare expected vs actual
                expected = binding.get("content_verbatim", "").strip()
                actual = actual_content.strip()
                
                if expected and actual == expected:
                    artifact_evidence["verification"] = "MATCH"
                elif expected and actual != expected:
                    artifact_evidence["verification"] = "MISMATCH"
                    artifact_evidence["diff"] = {
                        "expected": expected,
                        "actual": actual,
                    }
                else:
                    artifact_evidence["verification"] = "EXISTS"
                    
            except Exception as e:
                artifact_evidence["actual_exists"] = True
                artifact_evidence["read_error"] = str(e)
                artifact_evidence["verification"] = "READ_ERROR"
        else:
            artifact_evidence["actual_exists"] = False
            artifact_evidence["verification"] = "MISSING"
        
        evidence["artifacts"].append(artifact_evidence)
        evidence["verification_results"].append({
            "artifact_id": artifact_evidence["artifact_id"],
            "result": artifact_evidence["verification"],
        })
    
    # Overall verification
    results = [a.get("verification") for a in evidence["artifacts"]]
    if all(r in ("MATCH", "EXISTS") for r in results):
        evidence["overall_result"] = "PASS"
    elif any(r == "MISSING" for r in results):
        evidence["overall_result"] = "FAIL_MISSING"
    elif any(r == "MISMATCH" for r in results):
        evidence["overall_result"] = "FAIL_MISMATCH"
    else:
        evidence["overall_result"] = "UNKNOWN"
    
    return evidence


# ---------------------------------------------------------------------------
# LLM Call Function Factory (v3.7: Fixed - uses call_llm_text)
# ---------------------------------------------------------------------------

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
            max_tokens: Max output tokens (advisory - call_llm_text uses env config)
        
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
            )
            logger.info(f"[overwatcher_llm] Response length: {len(result)}")
            return result
        except Exception as e:
            logger.exception(f"[overwatcher_llm] LLM call failed: {e}")
            raise
    
    return llm_call_fn


# ---------------------------------------------------------------------------
# SSE Helpers
# ---------------------------------------------------------------------------

def sse_token(content: str) -> str:
    return "data: " + json.dumps({"type": "token", "content": content}) + "\n\n"


def sse_event(event_type: str, **kwargs) -> str:
    return "data: " + json.dumps({"type": event_type, **kwargs}) + "\n\n"


def sse_error(error: str) -> str:
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


__all__ = ["generate_overwatcher_stream", "create_overwatcher_llm_fn"]