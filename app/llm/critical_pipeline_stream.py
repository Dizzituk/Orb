# FILE: app/llm/critical_pipeline_stream.py
"""
Critical Pipeline streaming handler for ASTRA command flow.

v2.1 (2026-01-04): Artifact Binding Support
- Extracts artifact bindings from spec for Overwatcher
- Includes content_verbatim, location, scope_constraints in architecture prompt
- Generates concrete file paths for implementation

v2.0: Real pipeline integration with Block 4-6.
"""

import json
import logging
import asyncio
import os
from typing import Optional, Any, List, Dict
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
    if _STAGE_MODELS_AVAILABLE:
        try:
            cfg = get_critical_pipeline_config()
            return {"provider": cfg.provider, "model": cfg.model}
        except Exception:
            pass
    return {
        "provider": os.getenv("CRITICAL_PIPELINE_PROVIDER", "anthropic"),
        "model": os.getenv("ANTHROPIC_OPUS_MODEL", "claude-opus-4-5-20251101"),
    }


# =============================================================================
# Artifact Binding (v2.1)
# =============================================================================

# Path template variables for artifact binding
PATH_VARIABLES = {
    "{JOB_ID}": lambda ctx: ctx.get("job_id", "unknown"),
    "{JOB_ROOT}": lambda ctx: os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"),
    "{SANDBOX_DESKTOP}": lambda ctx: "C:/Users/WDAGUtilityAccount/Desktop",
    "{REPO_ROOT}": lambda ctx: ctx.get("repo_root", "."),
}


def _resolve_path_template(template: str, context: Dict[str, Any]) -> str:
    """Resolve path template variables."""
    result = template
    for var, resolver in PATH_VARIABLES.items():
        if var in result:
            result = result.replace(var, str(resolver(context)))
    return result


def _extract_artifact_bindings(spec_data: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract and resolve artifact bindings from spec for Overwatcher.
    
    Returns list of bindings with resolved paths:
    [
        {
            "artifact_id": "output_1",
            "action": "create",
            "path": "/resolved/path/to/file.txt",
            "content_type": "text",
            "content_verbatim": "hello",  # if specified
            "description": "Output file"
        }
    ]
    """
    bindings: List[Dict[str, Any]] = []
    
    # Get outputs from spec
    outputs = spec_data.get("outputs", [])
    if not outputs:
        # Try metadata
        metadata = spec_data.get("metadata", {}) or {}
        outputs = metadata.get("outputs", [])
    
    # Get content preservation fields
    content_verbatim = (
        spec_data.get("content_verbatim") or
        spec_data.get("context", {}).get("content_verbatim") or
        spec_data.get("metadata", {}).get("content_verbatim")
    )
    location = (
        spec_data.get("location") or
        spec_data.get("context", {}).get("location") or
        spec_data.get("metadata", {}).get("location")
    )
    
    for i, output in enumerate(outputs):
        if isinstance(output, str):
            output = {"name": output, "path": "", "description": ""}
        
        name = output.get("name", f"output_{i+1}")
        path = output.get("path", "")
        description = output.get("description", output.get("notes", ""))
        
        # Resolve path
        if path:
            resolved_path = _resolve_path_template(path, context)
        elif location:
            # Use location from content preservation
            resolved_path = _resolve_path_template(location, context)
            if name and not resolved_path.endswith(name):
                resolved_path = os.path.join(resolved_path, name)
        else:
            # Default to job artifacts directory
            resolved_path = os.path.join(
                context.get("job_root", "jobs"),
                "jobs",
                context.get("job_id", "unknown"),
                "outputs",
                name
            )
        
        binding = {
            "artifact_id": f"output_{i+1}",
            "action": "create",
            "path": resolved_path,
            "content_type": _infer_content_type(name),
            "description": description or name,
        }
        
        # Include content_verbatim if this is the primary output
        if i == 0 and content_verbatim:
            binding["content_verbatim"] = content_verbatim
        
        bindings.append(binding)
    
    logger.info("[critical_pipeline] Extracted %d artifact bindings", len(bindings))
    return bindings


def _infer_content_type(filename: str) -> str:
    """Infer content type from filename."""
    ext = os.path.splitext(filename.lower())[1]
    type_map = {
        ".txt": "text",
        ".md": "markdown",
        ".json": "json",
        ".py": "python",
        ".js": "javascript",
        ".html": "html",
        ".css": "css",
        ".yaml": "yaml",
        ".yml": "yaml",
    }
    return type_map.get(ext, "text")


def _build_artifact_binding_prompt(bindings: List[Dict[str, Any]]) -> str:
    """Build prompt section for artifact bindings."""
    if not bindings:
        return ""
    
    lines = [
        "\n## ARTIFACT BINDINGS (for Overwatcher)",
        "",
        "The following artifacts MUST be created with these EXACT paths:",
        ""
    ]
    
    for binding in bindings:
        lines.append(f"- **{binding['artifact_id']}**: `{binding['path']}`")
        lines.append(f"  - Action: {binding['action']}")
        lines.append(f"  - Type: {binding['content_type']}")
        if binding.get("content_verbatim"):
            lines.append(f"  - Content: \"{binding['content_verbatim']}\" (EXACT)")
        lines.append("")
    
    lines.append("Overwatcher will use these bindings to write files. Do NOT invent different paths.")
    
    return "\n".join(lines)


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
    Generate SSE stream for Critical Pipeline execution with artifact binding (v2.1).
    
    Flow:
    1. Load validated spec from DB
    2. Extract artifact bindings with resolved paths
    3. Build JobEnvelope and LLMTask with binding context
    4. Call run_high_stakes_with_critique()
    5. Stream progress and final result
    """
    response_parts = []
    
    model_cfg = _get_pipeline_model_config()
    pipeline_provider = model_cfg["provider"]
    pipeline_model = model_cfg["model"]
    
    try:
        yield "data: " + json.dumps({"type": "token", "content": "⚙️ **Critical Pipeline**\n\n"}) + "\n\n"
        response_parts.append("⚙️ **Critical Pipeline**\n\n")
        
        # =====================================================================
        # Validation
        # =====================================================================
        
        if not _PIPELINE_AVAILABLE:
            error_msg = (
                "❌ **Pipeline modules not available.**\n\n"
                "The high-stakes pipeline modules (app.llm.pipeline.*) failed to import.\n"
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
            yield "data: " + json.dumps({
                "type": "done", "provider": pipeline_provider, "model": pipeline_model,
                "total_length": sum(len(p) for p in response_parts)
            }) + "\n\n"
            return
        
        # =====================================================================
        # Step 1: Load validated spec
        # =====================================================================
        
        yield "data: " + json.dumps({"type": "token", "content": "📋 **Loading validated spec...**\n"}) + "\n\n"
        response_parts.append("📋 **Loading validated spec...**\n")
        
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
            yield "data: " + json.dumps({
                "type": "done", "provider": pipeline_provider, "model": pipeline_model,
                "total_length": sum(len(p) for p in response_parts)
            }) + "\n\n"
            return
        
        spec_id = db_spec.spec_id
        spec_hash = db_spec.spec_hash
        spec_json = db_spec.content_json
        
        # Parse spec JSON
        try:
            spec_data = json.loads(spec_json) if isinstance(spec_json, str) else (spec_json or {})
        except Exception:
            spec_data = {}
        
        yield "data: " + json.dumps({"type": "token", "content": f"✅ Spec loaded: `{spec_id[:16]}...`\n"}) + "\n\n"
        response_parts.append(f"✅ Spec loaded: `{spec_id[:16]}...`\n")
        
        # =====================================================================
        # Step 2: Create job ID and extract artifact bindings (v2.1)
        # =====================================================================
        
        if not job_id:
            job_id = f"cp-{uuid4().hex[:8]}"
        
        # Build context for path resolution
        binding_context = {
            "job_id": job_id,
            "job_root": os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"),
            "repo_root": os.getenv("REPO_ROOT", "."),
        }
        
        # Extract artifact bindings
        artifact_bindings = _extract_artifact_bindings(spec_data, binding_context)
        
        yield "data: " + json.dumps({"type": "token", "content": f"📁 **Job ID:** `{job_id}`\n"}) + "\n\n"
        response_parts.append(f"📁 **Job ID:** `{job_id}`\n")
        
        if artifact_bindings:
            binding_msg = f"📦 **Artifact Bindings:** {len(artifact_bindings)} output(s)\n"
            for b in artifact_bindings[:3]:  # Show first 3
                binding_msg += f"  - `{b['path']}`\n"
            if len(artifact_bindings) > 3:
                binding_msg += f"  - ... and {len(artifact_bindings) - 3} more\n"
            yield "data: " + json.dumps({"type": "token", "content": binding_msg}) + "\n\n"
            response_parts.append(binding_msg)
        
        # =====================================================================
        # Step 3: Build prompt with content preservation and bindings
        # =====================================================================
        
        yield "data: " + json.dumps({"type": "token", "content": "🔧 **Building architecture prompt...**\n\n"}) + "\n\n"
        response_parts.append("🔧 **Building architecture prompt...**\n\n")
        
        # Extract content preservation fields
        content_verbatim = (
            spec_data.get("content_verbatim") or
            spec_data.get("context", {}).get("content_verbatim") or
            spec_data.get("metadata", {}).get("content_verbatim")
        )
        location = (
            spec_data.get("location") or
            spec_data.get("context", {}).get("location") or
            spec_data.get("metadata", {}).get("location")
        )
        scope_constraints = (
            spec_data.get("scope_constraints") or
            spec_data.get("context", {}).get("scope_constraints") or
            spec_data.get("metadata", {}).get("scope_constraints") or
            []
        )
        
        # Build artifact binding prompt section
        binding_prompt = _build_artifact_binding_prompt(artifact_bindings)
        
        # Build system prompt with all context
        original_request = message
        if spec_data:
            original_request = spec_data.get("goal", "") or spec_data.get("objective", "") or message
        
        system_prompt = f"""You are Claude Opus, generating a detailed architecture document.

SPEC_ID: {spec_id}
SPEC_HASH: {spec_hash}

You are working from a validated PoT Spec. Your architecture MUST:
1. Address all MUST requirements from the spec
2. Consider all constraints
3. Be buildable by a solo developer on Windows 11
4. Include the SPEC_ID and SPEC_HASH header at the top of your output

## CONTENT PRESERVATION (CRITICAL)
"""
        
        if content_verbatim:
            system_prompt += f"""
**EXACT FILE CONTENT REQUIRED:**
The file content MUST be EXACTLY: "{content_verbatim}"
Do NOT paraphrase, summarize, or modify this content in any way.
"""
        
        if location:
            system_prompt += f"""
**EXACT LOCATION REQUIRED:**
The output MUST be written to: {location}
Use this EXACT path - do not substitute or normalize it.
"""
        
        if scope_constraints:
            system_prompt += f"""
**SCOPE CONSTRAINTS:**
{chr(10).join(f'- {c}' for c in scope_constraints)}
The implementation MUST NOT operate outside these boundaries.
"""
        
        system_prompt += binding_prompt
        system_prompt += "\n\nGenerate a complete, detailed architecture document."
        
        task_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate architecture for:\n\n{original_request}\n\nSpec:\n{json.dumps(spec_data, indent=2)}"},
        ]
        
        # Build LLMTask
        task = LLMTask(
            messages=task_messages,
            job_type=JobType.ARCHITECTURE_DESIGN if hasattr(JobType, 'ARCHITECTURE_DESIGN') else list(JobType)[0],
            attachments=[],
        )
        
        # Build JobEnvelope with artifact bindings in metadata
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
                max_cost_estimate=1.00,
                max_wall_time_seconds=600,
            ),
            output_contract=OutputContract.TEXT_RESPONSE,
            messages=task_messages,
            metadata={
                "spec_id": spec_id,
                "spec_hash": spec_hash,
                "pipeline": "critical",
                # v2.1: Include artifact bindings for Overwatcher
                "artifact_bindings": artifact_bindings,
                "content_verbatim": content_verbatim,
                "location": location,
                "scope_constraints": scope_constraints,
            },
            allow_multi_model_review=True,
            needs_tools=[],
        )
        
        # =====================================================================
        # Step 4: Run the pipeline
        # =====================================================================
        
        yield "data: " + json.dumps({"type": "token", "content": f"🏗️ **Starting Block 4-6 Pipeline with {pipeline_model}...**\n\n"}) + "\n\n"
        response_parts.append(f"🏗️ **Starting Block 4-6 Pipeline with {pipeline_model}...**\n\n")
        
        yield "data: " + json.dumps({"type": "token", "content": "This may take 2-5 minutes. Stages:\n"}) + "\n\n"
        yield "data: " + json.dumps({"type": "token", "content": "  1. 📝 Architecture generation\n"}) + "\n\n"
        yield "data: " + json.dumps({"type": "token", "content": "  2. 🔍 Critique\n"}) + "\n\n"
        yield "data: " + json.dumps({"type": "token", "content": "  3. ✏️ Revision loop (up to 3 rounds)\n\n"}) + "\n\n"
        
        yield "data: " + json.dumps({
            "type": "pipeline_started",
            "stage": "critical_pipeline",
            "job_id": job_id,
            "spec_id": spec_id,
            "artifact_bindings": len(artifact_bindings),
        }) + "\n\n"
        
        try:
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
            
        except Exception as e:
            logger.exception(f"[critical_pipeline] Pipeline failed: {e}")
            error_msg = f"❌ **Pipeline failed:** {e}\n"
            yield "data: " + json.dumps({"type": "token", "content": error_msg}) + "\n\n"
            response_parts.append(error_msg)
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
            yield "data: " + json.dumps({
                "type": "done", "provider": pipeline_provider, "model": pipeline_model,
                "total_length": sum(len(p) for p in response_parts)
            }) + "\n\n"
            return
        
        routing_decision = getattr(result, 'routing_decision', {}) or {}
        arch_id = routing_decision.get('arch_id', 'unknown')
        final_version = routing_decision.get('final_version', 1)
        critique_passed = routing_decision.get('critique_passed', False)
        blocking_issues = routing_decision.get('blocking_issues', 0)
        
        summary_header = "✅ **Pipeline Complete**\n\n"
        yield "data: " + json.dumps({"type": "token", "content": summary_header}) + "\n\n"
        response_parts.append(summary_header)
        
        summary_details = f"""**Architecture ID:** `{arch_id}`
**Final Version:** v{final_version}
**Critique Status:** {"✅ PASSED" if critique_passed else f"⚠️ {blocking_issues} blocking issues"}
**Provider:** {result.provider}
**Model:** {result.model}
**Tokens:** {result.total_tokens:,}
**Cost:** ${result.cost_usd:.4f}
**Artifact Bindings:** {len(artifact_bindings)}

---

"""
        yield "data: " + json.dumps({"type": "token", "content": summary_details}) + "\n\n"
        response_parts.append(summary_details)
        
        # Stream architecture content
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
        # Step 6: Emit completion events with artifact bindings
        # =====================================================================
        
        yield "data: " + json.dumps({
            "type": "work_artifacts",
            "spec_id": spec_id,
            "job_id": job_id,
            "arch_id": arch_id,
            "final_version": final_version,
            "critique_passed": critique_passed,
            "artifact_bindings": artifact_bindings,  # v2.1: Include for Overwatcher
            "artifacts": [
                f"arch_v{final_version}.md",
                f"critique_v{final_version}.json",
            ],
        }) + "\n\n"
        
        if critique_passed:
            next_step = f"""

---
✅ **Ready for Implementation**

Architecture approved with {len(artifact_bindings)} artifact binding(s).

🔧 **Next Step:** Say **'Astra, command: send to overwatcher'** to implement.
"""
        else:
            next_step = f"""

---
⚠️ **Critique Not Fully Passed**

{blocking_issues} blocking issues remain. Review the architecture above.

You may:
- Re-run with updated spec
- Proceed to Overwatcher with caution
"""
        
        yield "data: " + json.dumps({"type": "token", "content": next_step}) + "\n\n"
        response_parts.append(next_step)
        
        # Save to memory
        full_response = "".join(response_parts)
        if memory_service and memory_schemas:
            try:
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
            "artifact_bindings": len(artifact_bindings),
            "tokens": result.total_tokens,
            "cost_usd": result.cost_usd,
        }) + "\n\n"
        
    except Exception as e:
        logger.exception("[critical_pipeline] Stream failed: %s", e)
        if trace:
            trace.finalize(success=False, error_message=str(e))
        yield "data: " + json.dumps({"type": "error", "error": str(e)}) + "\n\n"


__all__ = ["generate_critical_pipeline_stream"]