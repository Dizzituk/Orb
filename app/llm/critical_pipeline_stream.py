# FILE: app/llm/critical_pipeline_stream.py
"""
Critical Pipeline streaming handler for ASTRA command flow.

Wires the streaming handler to the existing Block 4-6 pipeline in app/llm/pipeline/:
- high_stakes.py: Main orchestrator (run_high_stakes_with_critique)
- critique.py: JSON critique with blocking/non-blocking issues
- revision.py: Spec-anchored revision loop
- critique_schemas.py: CritiqueResult/CritiqueIssue schemas

v2.0 (2026-01): COMPLETE REWRITE - Now calls real pipeline orchestration
v1.1 (2026-01): Fixed spec_id lookup (was stub returning fake output)
v1.0 (2026-01): Initial stub implementation
"""

import json
import logging
import asyncio
import os
from typing import Optional, Any
from uuid import uuid4

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# =============================================================================
# Pipeline Imports (Block 4-6)
# =============================================================================

try:
    from app.llm.pipeline.high_stakes import (
        run_high_stakes_with_critique,
        store_architecture_artifact,
        get_environment_context,
        HIGH_STAKES_JOB_TYPES,
    )
    _PIPELINE_AVAILABLE = True
except ImportError as e:
    _PIPELINE_AVAILABLE = False
    logger.warning(f"[critical_pipeline] Pipeline modules not available: {e}")

try:
    from app.llm.pipeline.critique_schemas import CritiqueResult
except ImportError:
    CritiqueResult = None

# =============================================================================
# Schema Imports
# =============================================================================

try:
    from app.llm.schemas import LLMTask, JobType
    from app.jobs.schemas import (
        JobEnvelope,
        JobType as Phase4JobType,
        Importance,
        DataSensitivity,
        Modality,
        JobBudget,
        OutputContract,
    )
    _SCHEMAS_AVAILABLE = True
except ImportError as e:
    _SCHEMAS_AVAILABLE = False
    logger.warning(f"[critical_pipeline] Schema imports failed: {e}")

# =============================================================================
# Spec Service Imports
# =============================================================================

try:
    from app.specs.service import get_spec, get_latest_validated_spec, get_spec_schema
    _SPECS_SERVICE_AVAILABLE = True
except ImportError:
    _SPECS_SERVICE_AVAILABLE = False
    get_spec = None
    get_latest_validated_spec = None
    get_spec_schema = None

# =============================================================================
# Memory Service Imports
# =============================================================================

try:
    from app.memory import service as memory_service, schemas as memory_schemas
except ImportError:
    memory_service = None
    memory_schemas = None

# =============================================================================
# Audit Logger Imports
# =============================================================================

try:
    from app.llm.audit_logger import RoutingTrace
except ImportError:
    RoutingTrace = None

# =============================================================================
# Stage Models (env-driven model resolution)
# =============================================================================

try:
    from app.llm.stage_models import get_critical_pipeline_config
    _STAGE_MODELS_AVAILABLE = True
except ImportError:
    _STAGE_MODELS_AVAILABLE = False

# =============================================================================
# Configuration
# =============================================================================

def _get_pipeline_model_config() -> dict:
    """Get Critical Pipeline model configuration from env vars AT RUNTIME."""
    if _STAGE_MODELS_AVAILABLE:
        try:
            cfg = get_critical_pipeline_config()
            return {
                "provider": cfg.provider,
                "model": cfg.model,
            }
        except Exception:
            pass
    
    # Fallback to env vars
    return {
        "provider": os.getenv("CRITICAL_PIPELINE_PROVIDER", "anthropic"),
        "model": os.getenv("ANTHROPIC_OPUS_MODEL", "claude-opus-4-5-20251101"),
    }


# =============================================================================
# Main Stream Handler
# =============================================================================

async def generate_critical_pipeline_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[Any] = None,
    conversation_id: Optional[str] = None,
    spec_id: Optional[str] = None,
    spec_hash: Optional[str] = None,
    job_id: Optional[str] = None,
):
    """
    Generate SSE stream for Critical Pipeline execution.
    
    This handler wires to the existing Block 4-6 pipeline:
    1. Load validated spec from DB
    2. Build JobEnvelope and LLMTask
    3. Call run_high_stakes_with_critique() from high_stakes.py
    4. Stream progress and final result
    5. Real artifacts are written by the pipeline
    
    v2.0: Now calls real pipeline orchestration instead of returning fake output.
    """
    response_parts = []
    
    # Get model config from env
    model_cfg = _get_pipeline_model_config()
    pipeline_provider = model_cfg["provider"]
    pipeline_model = model_cfg["model"]
    
    try:
        yield "data: " + json.dumps({"type": "token", "content": "⚙️ **Critical Pipeline**\n\n"}) + "\n\n"
        response_parts.append("⚙️ **Critical Pipeline**\n\n")
        
        # =====================================================================
        # Validation: Check required modules are available
        # =====================================================================
        
        if not _PIPELINE_AVAILABLE:
            error_msg = (
                "❌ **Pipeline modules not available.**\n\n"
                "The high-stakes pipeline modules (app.llm.pipeline.*) failed to import.\n"
                "Check backend logs for import errors.\n"
            )
            yield "data: " + json.dumps({"type": "token", "content": error_msg}) + "\n\n"
            response_parts.append(error_msg)
            if trace:
                trace.finalize(success=False, error_message="Pipeline modules not available")
            yield "data: " + json.dumps({
                "type": "done", "provider": pipeline_provider, "model": pipeline_model,
                "total_length": sum(len(p) for p in response_parts)
            }) + "\n\n"
            return
        
        if not _SCHEMAS_AVAILABLE:
            error_msg = "❌ **Schema imports failed.** Check backend logs.\n"
            yield "data: " + json.dumps({"type": "token", "content": error_msg}) + "\n\n"
            response_parts.append(error_msg)
            if trace:
                trace.finalize(success=False, error_message="Schema imports failed")
            yield "data: " + json.dumps({
                "type": "done", "provider": pipeline_provider, "model": pipeline_model,
                "total_length": sum(len(p) for p in response_parts)
            }) + "\n\n"
            return
        
        # =====================================================================
        # Step 1: Load validated spec from DB
        # =====================================================================
        
        yield "data: " + json.dumps({"type": "token", "content": "📋 **Loading validated spec from database...**\n"}) + "\n\n"
        response_parts.append("📋 **Loading validated spec from database...**\n")
        
        # Try to get spec by ID first, then by project
        db_spec = None
        spec_json = None
        
        if spec_id and _SPECS_SERVICE_AVAILABLE and get_spec:
            try:
                db_spec = get_spec(db, spec_id)
            except Exception as e:
                logger.warning(f"[critical_pipeline] Failed to get spec by ID: {e}")
        
        if not db_spec and _SPECS_SERVICE_AVAILABLE and get_latest_validated_spec:
            try:
                db_spec = get_latest_validated_spec(db, project_id)
            except Exception as e:
                logger.warning(f"[critical_pipeline] Failed to get latest validated spec: {e}")
        
        if not db_spec:
            error_msg = (
                "❌ **No validated spec found.**\n\n"
                "Please complete Spec Gate validation first:\n"
                "1. Describe what you want to build\n"
                "2. Say `Astra, command: how does that look all together`\n"
                "3. Say `Astra, command: critical architecture` to validate\n"
                "4. Once validated, retry `run critical pipeline`\n"
            )
            yield "data: " + json.dumps({"type": "token", "content": error_msg}) + "\n\n"
            response_parts.append(error_msg)
            if trace:
                trace.finalize(success=False, error_message="No validated spec")
            yield "data: " + json.dumps({
                "type": "done", "provider": pipeline_provider, "model": pipeline_model,
                "total_length": sum(len(p) for p in response_parts)
            }) + "\n\n"
            return
        
        # Extract spec details
        spec_id = db_spec.spec_id
        spec_hash = db_spec.spec_hash
        spec_json = db_spec.content_json  # This is the canonical JSON
        
        yield "data: " + json.dumps({"type": "token", "content": f"✅ Spec loaded: `{spec_id[:16]}...`\n"}) + "\n\n"
        response_parts.append(f"✅ Spec loaded: `{spec_id[:16]}...`\n")
        yield "data: " + json.dumps({"type": "token", "content": f"   Hash: `{spec_hash[:16]}...`\n"}) + "\n\n"
        response_parts.append(f"   Hash: `{spec_hash[:16]}...`\n")
        yield "data: " + json.dumps({"type": "token", "content": f"   Status: `{db_spec.status}`\n\n"}) + "\n\n"
        response_parts.append(f"   Status: `{db_spec.status}`\n\n")
        
        # =====================================================================
        # Step 2: Create job ID if not provided
        # =====================================================================
        
        if not job_id:
            job_id = f"cp-{uuid4().hex[:8]}"
        
        yield "data: " + json.dumps({"type": "token", "content": f"📁 **Job ID:** `{job_id}`\n\n"}) + "\n\n"
        response_parts.append(f"📁 **Job ID:** `{job_id}`\n\n")
        
        # =====================================================================
        # Step 3: Build JobEnvelope and LLMTask
        # =====================================================================
        
        yield "data: " + json.dumps({"type": "token", "content": "🔧 **Initializing pipeline...**\n\n"}) + "\n\n"
        response_parts.append("🔧 **Initializing pipeline...**\n\n")
        
        # Extract original request from spec or message
        original_request = message
        if spec_json:
            try:
                spec_data = json.loads(spec_json) if isinstance(spec_json, str) else spec_json
                original_request = spec_data.get("goal", "") or spec_data.get("objective", "") or message
            except:
                pass
        
        # Build messages for the LLM
        system_prompt = f"""You are Claude Opus, generating a detailed architecture document.

SPEC_ID: {spec_id}
SPEC_HASH: {spec_hash}

You are working from a validated PoT Spec. Your architecture MUST:
1. Address all MUST requirements from the spec
2. Consider all constraints
3. Be buildable by a solo developer on Windows 11
4. Include the SPEC_ID and SPEC_HASH header at the top of your output

Generate a complete, detailed architecture document."""

        task_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate architecture for:\n\n{original_request}\n\nSpec:\n{spec_json}"},
        ]
        
        # Build LLMTask
        task = LLMTask(
            messages=task_messages,
            job_type=JobType.ARCHITECTURE_DESIGN if hasattr(JobType, 'ARCHITECTURE_DESIGN') else list(JobType)[0],
            attachments=[],
        )
        
        # Build JobEnvelope
        envelope = JobEnvelope(
            job_id=job_id,
            session_id=conversation_id or f"session-{uuid4().hex[:8]}",
            project_id=project_id,
            job_type=getattr(Phase4JobType, "APP_ARCHITECTURE", list(Phase4JobType)[0]),
            importance=Importance.CRITICAL,
            data_sensitivity=DataSensitivity.INTERNAL,
            modalities_in=[Modality.TEXT],
            budget=JobBudget(
                max_tokens=16384,
                max_cost_estimate=1.00,  # Allow higher cost for Opus
                max_wall_time_seconds=600,  # 10 minutes for full pipeline
            ),
            output_contract=OutputContract.TEXT_RESPONSE,
            messages=task_messages,
            metadata={
                "spec_id": spec_id,
                "spec_hash": spec_hash,
                "pipeline": "critical",
            },
            allow_multi_model_review=True,
            needs_tools=[],
        )
        
        # =====================================================================
        # Step 4: Run the actual pipeline
        # =====================================================================
        
        yield "data: " + json.dumps({"type": "token", "content": f"🏗️ **Starting Block 4-6 Pipeline with {pipeline_model}...**\n\n"}) + "\n\n"
        response_parts.append(f"🏗️ **Starting Block 4-6 Pipeline with {pipeline_model}...**\n\n")
        
        yield "data: " + json.dumps({"type": "token", "content": "This may take 2-5 minutes. Stages:\n"}) + "\n\n"
        response_parts.append("This may take 2-5 minutes. Stages:\n")
        yield "data: " + json.dumps({"type": "token", "content": "  1. 📝 Architecture generation (Opus)\n"}) + "\n\n"
        response_parts.append("  1. 📝 Architecture generation (Opus)\n")
        yield "data: " + json.dumps({"type": "token", "content": "  2. 🔍 Critique (Gemini)\n"}) + "\n\n"
        response_parts.append("  2. 🔍 Critique (Gemini)\n")
        yield "data: " + json.dumps({"type": "token", "content": "  3. ✏️ Revision loop (Opus) - up to 3 rounds\n\n"}) + "\n\n"
        response_parts.append("  3. ✏️ Revision loop (Opus) - up to 3 rounds\n\n")
        
        yield "data: " + json.dumps({
            "type": "pipeline_started",
            "stage": "critical_pipeline",
            "job_id": job_id,
            "spec_id": spec_id,
            "provider": pipeline_provider,
            "model": pipeline_model,
        }) + "\n\n"
        
        # Emit stage trace
        print(f"[STAGE_TRACE] ┌─ ENTER: architecture_generation")
        print(f"[STAGE_TRACE] │  job_id={job_id}, spec_id={spec_id}")
        print(f"[STAGE_TRACE] │  provider={pipeline_provider}, model={pipeline_model}")
        
        try:
            # Call the real pipeline
            result = await run_high_stakes_with_critique(
                task=task,
                provider_id=pipeline_provider,
                model_id=pipeline_model,
                envelope=envelope,
                job_type_str="architecture_design",
                file_map=None,
                db=db,
                spec_id=spec_id,
                spec_hash=spec_hash,
                spec_json=spec_json,
                use_json_critique=True,
            )
            
            print(f"[STAGE_TRACE] └─ EXIT: pipeline_complete")
            print(f"[STAGE_TRACE]    result_length={len(result.content) if result else 0}")
            
        except Exception as e:
            logger.exception(f"[critical_pipeline] Pipeline failed: {e}")
            error_msg = f"❌ **Pipeline failed:** {e}\n"
            yield "data: " + json.dumps({"type": "token", "content": error_msg}) + "\n\n"
            response_parts.append(error_msg)
            if trace:
                trace.finalize(success=False, error_message=str(e))
            yield "data: " + json.dumps({
                "type": "done", "provider": pipeline_provider, "model": pipeline_model,
                "total_length": sum(len(p) for p in response_parts), "error": str(e)
            }) + "\n\n"
            return
        
        # =====================================================================
        # Step 5: Stream the result
        # =====================================================================
        
        if not result or not result.content:
            error_msg = "❌ **Pipeline returned empty result.**\n"
            yield "data: " + json.dumps({"type": "token", "content": error_msg}) + "\n\n"
            response_parts.append(error_msg)
            if trace:
                trace.finalize(success=False, error_message="Empty pipeline result")
            yield "data: " + json.dumps({
                "type": "done", "provider": pipeline_provider, "model": pipeline_model,
                "total_length": sum(len(p) for p in response_parts)
            }) + "\n\n"
            return
        
        # Extract routing decision metadata
        routing_decision = getattr(result, 'routing_decision', {}) or {}
        arch_id = routing_decision.get('arch_id', 'unknown')
        final_version = routing_decision.get('final_version', 1)
        critique_passed = routing_decision.get('critique_passed', False)
        blocking_issues = routing_decision.get('blocking_issues', 0)
        
        # Stream pipeline summary
        summary_header = "✅ **Pipeline Complete**\n\n"
        yield "data: " + json.dumps({"type": "token", "content": summary_header}) + "\n\n"
        response_parts.append(summary_header)
        
        summary_details = f"""**Architecture ID:** `{arch_id}`
**Final Version:** v{final_version}
**Critique Status:** {"✅ PASSED" if critique_passed else f"⚠️ {blocking_issues} blocking issues remain"}
**Provider:** {result.provider}
**Model:** {result.model}
**Tokens:** {result.total_tokens:,}
**Cost:** ${result.cost_usd:.4f}

---

"""
        yield "data: " + json.dumps({"type": "token", "content": summary_details}) + "\n\n"
        response_parts.append(summary_details)
        
        # Stream the architecture content in chunks
        yield "data: " + json.dumps({"type": "token", "content": "### Architecture Document\n\n"}) + "\n\n"
        response_parts.append("### Architecture Document\n\n")
        
        content = result.content
        chunk_size = 200
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            yield "data: " + json.dumps({"type": "token", "content": chunk}) + "\n\n"
            response_parts.append(chunk)
            await asyncio.sleep(0.01)
        
        # =====================================================================
        # Step 6: Emit completion events
        # =====================================================================
        
        # Emit work artifacts event
        yield "data: " + json.dumps({
            "type": "work_artifacts",
            "spec_id": spec_id,
            "job_id": job_id,
            "arch_id": arch_id,
            "final_version": final_version,
            "critique_passed": critique_passed,
            "artifacts": [
                f"arch_v{final_version}.md",
                f"critique_v{final_version}.json",
                f"critique_v{final_version}.md",
            ],
        }) + "\n\n"
        
        # Only suggest Overwatcher if critique passed
        if critique_passed:
            next_step = """

---
✅ **Ready for Implementation**

All blocking issues resolved. Architecture is approved.

🔧 **Next Step:** Say **'Astra, command: send to overwatcher'** to have Overwatcher implement these changes.
"""
        else:
            next_step = f"""

---
⚠️ **Critique Not Fully Passed**

{blocking_issues} blocking issues remain after max revision iterations.
Review the architecture document above for outstanding issues.

You may:
- Re-run the pipeline with updated spec
- Manually address blocking issues
- Proceed to Overwatcher with caution
"""
        
        yield "data: " + json.dumps({"type": "token", "content": next_step}) + "\n\n"
        response_parts.append(next_step)
        
        # =====================================================================
        # Step 7: Save to memory
        # =====================================================================
        
        full_response = "".join(response_parts)
        if memory_service and memory_schemas:
            try:
                memory_service.create_message(db, memory_schemas.MessageCreate(
                    project_id=project_id, role="user", content=message, provider="local"
                ))
                memory_service.create_message(db, memory_schemas.MessageCreate(
                    project_id=project_id, role="assistant", content=full_response,
                    provider=pipeline_provider, model=pipeline_model
                ))
            except Exception as e:
                logger.warning(f"[critical_pipeline] Failed to save to memory: {e}")
        
        if trace:
            trace.finalize(success=True)
        
        yield "data: " + json.dumps({
            "type": "done",
            "provider": pipeline_provider,
            "model": pipeline_model,
            "total_length": len(full_response),
            "spec_id": spec_id,
            "job_id": job_id,
            "arch_id": arch_id,
            "final_version": final_version,
            "critique_passed": critique_passed,
            "tokens": result.total_tokens,
            "cost_usd": result.cost_usd,
        }) + "\n\n"
        
    except Exception as e:
        logger.exception("[critical_pipeline] Stream failed: %s", e)
        if trace:
            trace.finalize(success=False, error_message=str(e))
        yield "data: " + json.dumps({"type": "error", "error": str(e)}) + "\n\n"


__all__ = ["generate_critical_pipeline_stream"]