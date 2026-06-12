# FILE: app/llm/pipeline/_high_stakes_helpers.py
# Purpose: High-stakes pipeline helper functions.
# Called-by: app.llm.pipeline._high_stakes_pipelines, app.llm.pipeline.high_stakes
# Depends-on: app.artefacts.service, app.llm.audit_logger, app.llm.pipeline._high_stakes_utils, app.llm.schemas (+3 more)
# Last-renovated: 2026-06-11
"""
High-stakes pipeline helper functions.

Extracted from high_stakes.py: configuration, routing, audit,
environment context, and architecture artifact storage.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from app.llm.schemas import LLMResult
from app.llm.pipeline._high_stakes_utils import (
    AUDIT_ENABLED, _compute_content_hash, _utc_iso,
)

# Audit logging (Spec S12)
try:
    from app.llm.audit_logger import get_audit_logger, RoutingTrace, AuditEventType
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
    from app.pot_spec.ledger import emit_arch_created, emit_arch_mirror_written
    from app.pot_spec.service import get_job_artifact_root
    LEDGER_AVAILABLE = True
except ImportError:
    LEDGER_AVAILABLE = False

# Stage models (env-driven model resolution)
try:
    from app.llm.stage_models import get_critical_pipeline_config as get_architecture_config
    _STAGE_MODELS_AVAILABLE = True
except ImportError:
    _STAGE_MODELS_AVAILABLE = False

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
    timeout = int(os.getenv("ARCHITECTURE_TIMEOUT_SECONDS") or os.getenv("OPUS_TIMEOUT_SECONDS", "600"))
    return provider, model, max_tokens, timeout


# Legacy exports (for backward compatibility)
OPUS_DRAFT_MAX_TOKENS = int(os.getenv("ARCHITECTURE_MAX_OUTPUT_TOKENS") or os.getenv("OPUS_DRAFT_MAX_TOKENS", "60000"))
OPUS_TIMEOUT_SECONDS = int(os.getenv("ARCHITECTURE_TIMEOUT_SECONDS") or os.getenv("OPUS_TIMEOUT_SECONDS", "600"))

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


# =============================================================================
# Environment Context
# =============================================================================

def get_environment_context(spec_json: Optional[str] = None) -> Dict[str, Any]:
    """Get environment context from spec, NOT hardcoded defaults.
    
    v1.1 (2026-01-22): CRITICAL FIX - Phantom Constraint Bug
    - Tech stack constraints MUST come from the spec, not hardcoded defaults
    - If spec has implementation_stack, use that
    - If spec has no implementation_stack, DO NOT inject default tech_stack
    - This prevents critique from rejecting architectures for phantom requirements
    
    Args:
        spec_json: The SpecGate JSON spec (contains implementation_stack if user specified)
    
    Returns:
        Environment context dict with deployment info and spec-derived constraints
    """
    # Base deployment context (platform-specific, always included)
    context = {
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
    }
    
    # ==========================================================================
    # v1.1 CRITICAL FIX: Extract tech_stack FROM SPEC, not hardcoded defaults
    # ==========================================================================
    # The old code had hardcoded React/Electron/FastAPI/SQLite as defaults.
    # This caused critique to reject architectures for not meeting phantom
    # requirements that the user never specified.
    #
    # Now we ONLY include tech_stack if the spec explicitly provides it.
    
    if spec_json:
        try:
            spec_data = json.loads(spec_json) if isinstance(spec_json, str) else spec_json
            impl_stack = spec_data.get("implementation_stack")
            
            if impl_stack and isinstance(impl_stack, dict):
                # Build tech_stack from spec's implementation_stack
                stack_info = {}
                
                if impl_stack.get("language"):
                    stack_info["language"] = impl_stack["language"]
                if impl_stack.get("framework"):
                    stack_info["framework"] = impl_stack["framework"]
                if impl_stack.get("runtime"):
                    stack_info["runtime"] = impl_stack["runtime"]
                
                # Include lock status so critique knows how strict to be
                stack_info["stack_locked"] = impl_stack.get("stack_locked", False)
                stack_info["source"] = impl_stack.get("source", "spec")
                
                if stack_info.get("language") or stack_info.get("framework"):
                    context["tech_stack"] = stack_info
                    logger.info(
                        "[get_environment_context] v1.1 Using spec-defined tech_stack: %s (locked=%s)",
                        stack_info, stack_info.get("stack_locked")
                    )
                    print(f"[DEBUG] [env_context] v1.1 Spec tech_stack: {stack_info}")
                else:
                    logger.info("[get_environment_context] v1.1 Spec has implementation_stack but no language/framework - skipping tech_stack")
            else:
                logger.info("[get_environment_context] v1.1 No implementation_stack in spec - NO tech_stack constraints")
                print("[DEBUG] [env_context] v1.1 No implementation_stack in spec - critique will NOT check tech stack")
        except Exception as e:
            logger.warning("[get_environment_context] v1.1 Failed to parse spec_json: %s", e)
    else:
        logger.info("[get_environment_context] v1.1 No spec_json provided - NO tech_stack constraints")
    
    # NOTE: We do NOT add a default tech_stack!
    # If the user didn't specify a tech stack, critique should NOT enforce one.
    # This is the FIX for the phantom constraint bug.
    
    return context


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
# v3.2: Evidence Loop Adapter for Architecture Draft
# =============================================================================


# =============================================================================
# Block 4: Architecture Artifact Storage
# =============================================================================


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
