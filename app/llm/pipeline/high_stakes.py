# FILE: app/llm/pipeline/high_stakes.py
"""High-stakes critique pipeline - Main orchestrator.

Implements Blocks 4, 5, 6 of the PoT (Proof of Thought) system:

Block 4: Architecture generation as versioned artifact with spec traceability
Block 5: Structured JSON critique with blocking/non-blocking issues (critique.py)
Block 6: Revision loop until critique passes (revision.py)

v4.1 (2026-01):
- Uses stage_models for provider/model configuration (env-driven)
- ARCHITECTURE_PROVIDER/ARCHITECTURE_MODEL from env controls draft generation

v4.0 (2025-12):
- REFACTORED: Split into 3 files for maintainability:
  - high_stakes.py: Orchestrator, routing, architecture storage (~400 lines)
  - critique.py: Critique callers, prompts, parsing (~420 lines)
  - revision.py: Revision loop, spec-anchored prompts (~380 lines)
- Spec-anchored pipeline prevents drift from reviewer suggestions
- Debug logging throughout for visibility

SPEC ANCHORING:
The PoT Spec serves as the authoritative anchor point throughout the pipeline:
1. Spec Gate creates the spec (user intent → requirements)
2. Claude Opus generates architecture (must include SPEC_ID/SPEC_HASH header)
3. Gemini critiques architecture against spec (verify alignment)
4. Claude Opus revises based on critique, BUT verifies suggestions against spec first
   - Reject suggestions that add/contradict spec requirements
   - This prevents "spec drift" where reviewers inadvertently change scope
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from app.llm.schemas import JobType, LLMResult, LLMTask
from app.jobs.schemas import (
    JobEnvelope,
    JobType as Phase4JobType,
    Importance,
    DataSensitivity,
    Modality,
    JobBudget,
    OutputContract,
)
from app.providers.registry import llm_call as registry_llm_call
from app.llm.job_classifier import compute_modality_flags
from app.llm.gemini_vision import transcribe_video_for_context

# Import from sibling modules
from app.llm.pipeline.critique import (
    call_json_critic,
    store_critique_artifact,
    call_gemini_critic,
    build_critique_prompt,
    GEMINI_CRITIC_MODEL,
)
from app.llm.pipeline.revision import (
    call_revision,
    run_revision_loop,
    call_opus_revision,
    _map_to_phase4_job_type,
    OPUS_REVISION_MAX_TOKENS,
    MAX_REVISION_ITERATIONS,
)
from app.llm.pipeline.critique_schemas import CritiqueResult

# Audit logging (Spec §12)
try:
    from app.llm.audit_logger import (
        get_audit_logger,
        RoutingTrace,
        AuditEventType,
    )
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False

# Artefact service (Block 4)
try:
    from app.artefacts.service import ArtefactService, write_architecture_doc
    ARTEFACTS_AVAILABLE = True
except ImportError:
    ARTEFACTS_AVAILABLE = False

# Ledger events (Block 4)
try:
    from app.pot_spec.ledger import (
        emit_arch_created,
        emit_arch_mirror_written,
    )
    from app.pot_spec.service import get_job_artifact_root
    LEDGER_AVAILABLE = True
except ImportError:
    LEDGER_AVAILABLE = False

# Stage 3 spec echo (for verification)
try:
    from app.jobs.stage3_locks import build_spec_echo_instruction
    STAGE3_AVAILABLE = True
except ImportError:
    STAGE3_AVAILABLE = False

# Stage models (env-driven model resolution)
try:
    from app.llm.stage_models import get_architecture_config
    _STAGE_MODELS_AVAILABLE = True
except ImportError:
    _STAGE_MODELS_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

AUDIT_ENABLED = os.getenv("ORB_AUDIT_ENABLED", "1") == "1"

logger = logging.getLogger(__name__)

MIN_CRITIQUE_CHARS = int(os.getenv("ORB_MIN_CRITIQUE_CHARS", "1500"))


def _get_architecture_draft_config() -> tuple[str, str, int, int]:
    """Get architecture draft provider/model from stage_models or env vars AT RUNTIME.
    
    Returns: (provider, model, max_tokens, timeout)
    """
    if _STAGE_MODELS_AVAILABLE:
        try:
            cfg = get_architecture_config()
            return cfg.provider, cfg.model, cfg.max_output_tokens, cfg.timeout_seconds
        except Exception:
            pass
    
    # Fallback to legacy env vars
    provider = os.getenv("ARCHITECTURE_PROVIDER", "anthropic")
    model = os.getenv("ARCHITECTURE_MODEL") or os.getenv("ANTHROPIC_OPUS_MODEL", "claude-opus-4-5-20251101")
    max_tokens = int(os.getenv("ARCHITECTURE_MAX_OUTPUT_TOKENS") or os.getenv("OPUS_DRAFT_MAX_TOKENS", "60000"))
    timeout = int(os.getenv("ARCHITECTURE_TIMEOUT_SECONDS") or os.getenv("OPUS_TIMEOUT_SECONDS", "300"))
    return provider, model, max_tokens, timeout


# Legacy exports (for backward compatibility)
OPUS_DRAFT_MAX_TOKENS = int(os.getenv("ARCHITECTURE_MAX_OUTPUT_TOKENS") or os.getenv("OPUS_DRAFT_MAX_TOKENS", "60000"))
OPUS_TIMEOUT_SECONDS = int(os.getenv("ARCHITECTURE_TIMEOUT_SECONDS") or os.getenv("OPUS_TIMEOUT_SECONDS", "300"))

HIGH_STAKES_JOB_TYPES = {
    "architecture_design",
    "security_review",
    "high_stakes_infra",
    "big_architecture",
    "compliance_review",
    "high_stakes_legal",
    "high_stakes_medical",
    "orchestrator",
}


# =============================================================================
# Audit Integration
# =============================================================================

def _maybe_start_trace(
    task: LLMTask,
    envelope: JobEnvelope,
    *,
    job_type_str: str,
    provider_id: str,
    model_id: str,
):
    """Start an audit trace if auditing is enabled."""
    if not (AUDIT_AVAILABLE and AUDIT_ENABLED):
        return None, None

    try:
        audit_logger = get_audit_logger()
        if not audit_logger:
            return None, None
        
        # Extract IDs from envelope for trace
        job_id = str(getattr(envelope, "job_id", "unknown"))
        session_id = str(getattr(envelope, "session_id", "unknown"))
        project_id = int(getattr(envelope, "project_id", 0))
        
        # Create trace with correct dataclass parameters
        trace = RoutingTrace(
            logger=audit_logger,
            request_id=job_id,
            session_id=session_id,
            project_id=project_id,
        )
        
        # Log the routing decision with metadata
        trace.log_routing_decision(
            job_type=job_type_str,
            provider=provider_id,
            model=model_id,
            reason="high_stakes_critique pipeline",
        )
        
        return audit_logger, trace
    except Exception as exc:
        logger.warning(f"[audit] Failed to start trace: {exc}")
        return None, None


def _maybe_complete_trace(audit_logger, trace, *, success: bool = True, error_message: str = "") -> None:
    """Complete an audit trace if one exists."""
    if not trace or not audit_logger:
        return
    try:
        audit_logger.complete_trace(trace, success=success, error_message=error_message)
    except Exception:
        pass


def _trace_step(trace, step: str, **kv) -> None:
    """Log a step/warning to the trace."""
    if not trace:
        return
    try:
        # RoutingTrace doesn't have add_step, use log_warning for step tracking
        trace.log_warning(f"step:{step}", **kv)
    except Exception:
        pass


def _trace_error(trace, step: str, message: str) -> None:
    """Log an error to the trace."""
    if not trace:
        return
    try:
        trace.log_error(step, message)
    except Exception:
        pass


# =============================================================================
# Environment Context
# =============================================================================

def get_environment_context() -> Dict[str, Any]:
    """Get environment context for architecture/infrastructure prompts."""
    return {
        "deployment": {
            "type": "single_host",
            "os": "Windows 11",
            "scope": "local_only",
            "network": "LAN",
            "resources": "solo_dev_workstation",
        },
        "constraints": {
            "cloud_services": False,
            "external_hosting": False,
            "multi_user": False,
            "scale": "personal_project",
        },
        "tech_stack": {
            "backend": "Python/FastAPI",
            "frontend": "React/Electron",
            "database": "SQLite",
            "llm_providers": ["anthropic", "openai", "google"],
        },
    }


# =============================================================================
# Routing Helpers
# =============================================================================

def normalize_job_type_for_high_stakes(job_type_str: str, reason: str = "") -> str:
    """Normalize various job type strings to canonical high-stakes types."""
    jt = (job_type_str or "").strip().lower().replace(" ", "_")
    
    # Map common variants
    mappings = {
        "architecture": "architecture_design",
        "arch": "architecture_design",
        "big_arch": "big_architecture",
        "security": "security_review",
        "sec_review": "security_review",
        "infra": "high_stakes_infra",
        "infrastructure": "high_stakes_infra",
        "compliance": "compliance_review",
        "legal": "high_stakes_legal",
        "medical": "high_stakes_medical",
    }
    
    return mappings.get(jt, jt)


def is_high_stakes_job(job_type_str: str) -> bool:
    """Check if job type qualifies for high-stakes pipeline."""
    normalized = normalize_job_type_for_high_stakes(job_type_str)
    return normalized in HIGH_STAKES_JOB_TYPES


def is_opus_model(model_id: str) -> bool:
    """Check if model is an Opus-tier model."""
    return "opus" in (model_id or "").lower()


def is_long_enough_for_critique(text: str) -> bool:
    """Check if response is long enough to warrant critique."""
    return len(text or "") >= MIN_CRITIQUE_CHARS


# =============================================================================
# Block 4: Architecture Artifact Storage
# =============================================================================

def _compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content (truncated to 16 chars)."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _utc_iso() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def store_architecture_artifact(
    *,
    db,
    job_id: str,
    project_id: int,
    arch_content: str,
    spec_id: str,
    spec_hash: str,
    arch_version: int = 1,
    model: str = "",
    previous_arch_id: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Store architecture as versioned artifact with spec traceability.
    
    Creates:
    - DB record via ArtefactService (if available)
    - Filesystem mirror at jobs/{job_id}/arch/arch_v{version}.md
    
    Returns (arch_id, arch_hash, path)
    """
    arch_id = str(uuid4())
    arch_hash = _compute_content_hash(arch_content)
    path = ""
    
    # 1. Store in ArtefactService (if available) - uses static methods
    if ARTEFACTS_AVAILABLE and db:
        try:
            ArtefactService.write_artefact(
                db=db,
                project_id=project_id,
                artefact_type="architecture_doc",
                name=f"arch_{job_id}_v{arch_version}",
                content=arch_content,
                metadata={
                    "arch_id": arch_id,
                    "arch_hash": arch_hash,
                    "spec_id": spec_id,
                    "spec_hash": spec_hash,
                    "version": arch_version,
                    "model": model,
                    "previous_arch_id": previous_arch_id,
                },
                created_by_job_id=job_id,
            )
            logger.info(f"[arch] Stored in ArtefactService: {arch_id}")
        except Exception as e:
            logger.warning(f"[arch] ArtefactService storage failed: {e}")
    
    # 2. Write filesystem mirror
    if LEDGER_AVAILABLE:
        try:
            job_root = get_job_artifact_root()
            arch_dir = Path(job_root) / "jobs" / job_id / "arch"
            arch_dir.mkdir(parents=True, exist_ok=True)
            
            path = str(arch_dir / f"arch_v{arch_version}.md")
            Path(path).write_text(arch_content, encoding="utf-8")
            
            # Emit ledger events
            emit_arch_created(
                job_artifact_root=job_root,
                job_id=job_id,
                arch_id=arch_id,
                arch_version=arch_version,
                arch_hash=arch_hash,
                spec_id=spec_id,
                spec_hash=spec_hash,
                model=model,
            )
            
            emit_arch_mirror_written(
                job_artifact_root=job_root,
                job_id=job_id,
                arch_id=arch_id,
                arch_version=arch_version,
                mirror_path=path,
                checksum=arch_hash,
            )
            
            logger.info(f"[arch] Mirror written: {path}")
        except Exception as e:
            logger.warning(f"[arch] Filesystem mirror failed: {e}")
    
    return arch_id, arch_hash, path


# =============================================================================
# Main Pipeline Entry Point
# =============================================================================

async def run_high_stakes_with_critique(
    task: LLMTask,
    provider_id: str,
    model_id: str,
    envelope: JobEnvelope,
    job_type_str: str,
    file_map: Optional[str] = None,
    *,
    # Block 4-6 params (optional, passed from Spec Gate)
    db=None,
    spec_id: Optional[str] = None,
    spec_hash: Optional[str] = None,
    spec_json: Optional[str] = None,
    use_json_critique: bool = True,
) -> LLMResult:
    """Run high-stakes critique pipeline.
    
    If spec_id/spec_hash are provided (from Spec Gate), uses Block 4-6 pipeline:
    - Stores architecture as versioned artifact
    - Uses JSON critique schema
    - Runs revision loop until pass or max iterations
    
    Otherwise uses legacy prose-based critique.
    """
    logger.info(f"[critic] High-stakes pipeline: job_type={job_type_str} model={model_id}")
    
    audit_logger, trace = _maybe_start_trace(
        task, envelope, job_type_str=job_type_str, provider_id=provider_id, model_id=model_id
    )
    
    # Pre-step: Video transcription
    attachments = task.attachments or []
    flags = compute_modality_flags(attachments)
    video_attachments = flags.get("video_attachments", [])
    
    transcripts_text = ""
    if video_attachments:
        for video_att in video_attachments:
            try:
                video_path = getattr(video_att, "path", None)
                if video_path:
                    transcript = await transcribe_video_for_context(video_path)
                    transcripts_text += f"\n\n=== Video: {video_att.filename} ===\n{transcript}"
            except Exception:
                pass
    
    # Step 1: Generate draft
    draft_messages = list(envelope.messages)
    
    # Inject spec echo instruction for Stage 3 verification
    if spec_id and spec_hash and STAGE3_AVAILABLE:
        spec_echo_instruction = build_spec_echo_instruction(spec_id, spec_hash)
        draft_messages.append({"role": "system", "content": spec_echo_instruction})
    
    if transcripts_text:
        draft_messages.append({"role": "system", "content": f"Video context:\n{transcripts_text.strip()}"})
    
    if file_map:
        draft_messages.append({"role": "system", "content": f"{file_map}\n\nRefer to files using [FILE_X] identifiers."})
    
    if trace:
        _trace_step(trace, 'draft')
    
    # Get architecture config for max_tokens and timeout (use stage_models if available)
    _, _, arch_max_tokens, arch_timeout = _get_architecture_draft_config()
    print(f"[DEBUG] [high_stakes] Draft generation: provider={provider_id}, model={model_id}, max_tokens={arch_max_tokens}")
    
    try:
        draft_result = await registry_llm_call(
            provider_id=provider_id,
            model_id=model_id,
            messages=draft_messages,
            job_envelope=envelope,
            max_tokens=arch_max_tokens,
            timeout_seconds=arch_timeout,
        )
    except Exception as exc:
        err_msg = f"High-stakes draft failed: {exc}"
        if trace:
            _trace_error(trace, 'draft', err_msg)
        _maybe_complete_trace(audit_logger, trace, success=False, error_message=err_msg)
        return LLMResult(
            content=err_msg, provider=provider_id, model=model_id,
            finish_reason="error", error_message=err_msg,
            prompt_tokens=0, completion_tokens=0, total_tokens=0, cost_usd=0.0, raw_response=None,
        )
    
    draft = LLMResult(
        content=draft_result.content,
        provider=provider_id,
        model=model_id,
        finish_reason="stop",
        error_message=None,
        prompt_tokens=draft_result.usage.prompt_tokens,
        completion_tokens=draft_result.usage.completion_tokens,
        total_tokens=draft_result.usage.total_tokens,
        cost_usd=draft_result.usage.cost_estimate,
        raw_response=draft_result.raw_response,
    )
    
    if trace:
        _trace_step(trace, 'draft_done')
    
    # Check if critique needed
    if not is_long_enough_for_critique(draft.content):
        logger.warning("[critic] Draft too short for critique")
        _maybe_complete_trace(audit_logger, trace, success=True)
        draft.routing_decision = {"job_type": job_type_str, "provider": provider_id, "model": model_id, "reason": "draft too short"}
        return draft
    
    # Extract original request
    user_messages = [m for m in task.messages if m.get("role") == "user"]
    original_request = user_messages[-1].get("content", "") if user_messages else ""
    
    # =========================================================================
    # Block 4-6: Full artifact pipeline (if spec provided)
    # =========================================================================
    if spec_id and spec_hash and use_json_critique:
        logger.info("[critic] Using Block 4-6 artifact pipeline")
        
        job_id = str(envelope.job_id)
        project_id = int(getattr(envelope, "project_id", 0))
        
        # Store initial architecture (Block 4)
        arch_id, arch_hash, _ = store_architecture_artifact(
            db=db,
            job_id=job_id,
            project_id=project_id,
            arch_content=draft.content,
            spec_id=spec_id,
            spec_hash=spec_hash,
            arch_version=1,
            model=model_id,
        )
        
        if trace:
            _trace_step(trace, 'arch_stored', arch_id=arch_id)
        
        # Run revision loop (Block 5 + 6)
        env_context = get_environment_context() if job_type_str in HIGH_STAKES_JOB_TYPES else None
        
        final_content, final_version, passed, final_critique = await run_revision_loop(
            db=db,
            job_id=job_id,
            project_id=project_id,
            arch_content=draft.content,
            arch_id=arch_id,
            spec_id=spec_id,
            spec_hash=spec_hash,
            spec_json=spec_json,
            original_request=original_request,
            opus_model_id=model_id,
            envelope=envelope,
            env_context=env_context,
            store_architecture_fn=store_architecture_artifact,
        )
        
        if trace:
            _trace_step(trace, 'revision_loop_done', version=final_version, passed=passed)
        
        _maybe_complete_trace(audit_logger, trace, success=True)
        
        return LLMResult(
            content=final_content,
            provider=provider_id,
            model=model_id,
            finish_reason="stop",
            error_message=None,
            prompt_tokens=draft.prompt_tokens,
            completion_tokens=draft.completion_tokens,
            total_tokens=draft.total_tokens,
            cost_usd=draft.cost_usd,
            raw_response=None,
            routing_decision={
                "job_type": job_type_str,
                "provider": provider_id,
                "model": model_id,
                "reason": f"Block 4-6 pipeline: v{final_version}, passed={passed}",
                "arch_id": arch_id,
                "final_version": final_version,
                "critique_passed": passed,
                "blocking_issues": len(final_critique.blocking_issues),
            },
        )
    
    # =========================================================================
    # Legacy pipeline (prose critique, single revision)
    # =========================================================================
    logger.info("[critic] Using legacy prose critique pipeline")
    
    # Step 2: Critique
    env_context = get_environment_context() if job_type_str in HIGH_STAKES_JOB_TYPES else None
    critique = await call_gemini_critic(
        original_task=task,
        draft_result=draft,
        job_type_str=job_type_str,
        envelope=envelope,
        env_context=env_context,
    )
    
    if not critique:
        logger.warning("[critic] Critique failed; returning draft")
        _maybe_complete_trace(audit_logger, trace, success=True)
        draft.routing_decision = {"job_type": job_type_str, "provider": provider_id, "model": model_id, "reason": "critique failed"}
        return draft
    
    if trace:
        _trace_step(trace, 'critique_done')
    
    # Step 3: Revision
    revision = await call_opus_revision(
        original_task=task, draft_result=draft, critique_result=critique,
        opus_model_id=model_id, envelope=envelope
    )
    
    if not revision:
        logger.warning("[critic] Revision failed; returning draft")
        _maybe_complete_trace(audit_logger, trace, success=True)
        draft.routing_decision = {"job_type": job_type_str, "provider": provider_id, "model": model_id, "reason": "revision failed"}
        return draft
    
    if trace:
        _trace_step(trace, 'revision_done')
    
    _maybe_complete_trace(audit_logger, trace, success=True)
    
    revision.routing_decision = {
        "job_type": job_type_str,
        "provider": provider_id,
        "model": revision.model,
        "reason": "Legacy: Opus draft → Gemini critique → Opus revision",
        "critique_pipeline": {
            "draft_tokens": draft.total_tokens,
            "critique_tokens": critique.total_tokens,
            "revision_tokens": revision.total_tokens,
            "total_cost": draft.cost_usd + critique.cost_usd + revision.cost_usd,
        },
    }
    
    return revision


__all__ = [
    # Configuration
    "HIGH_STAKES_JOB_TYPES",
    "MIN_CRITIQUE_CHARS",
    # Routing helpers
    "normalize_job_type_for_high_stakes",
    "is_high_stakes_job",
    "is_opus_model",
    "is_long_enough_for_critique",
    "get_environment_context",
    "_map_to_phase4_job_type",
    # Block 4: Architecture storage
    "store_architecture_artifact",
    # Re-exports from critique.py
    "call_json_critic",
    "store_critique_artifact",
    "call_gemini_critic",
    "build_critique_prompt",
    # Re-exports from revision.py
    "call_revision",
    "run_revision_loop",
    "call_opus_revision",
    # Main entry
    "run_high_stakes_with_critique",
]