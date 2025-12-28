# FILE: app/llm/pipeline/high_stakes.py
"""High-stakes critique pipeline with architecture artifacts and revision loop.

Implements Blocks 4, 5, 6 of the PoT (Proof of Thought) system:

Block 4: Architecture generation as versioned artifact with spec traceability
Block 5: Structured JSON critique with blocking/non-blocking issues
Block 6: Revision loop until critique passes (max iterations safety)

v3 (2025-12):
- Architecture artifacts stored via ArtefactService + filesystem mirror
- JSON critique schema with machine-driven pass/fail
- Revision loop controller with ledger events
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import textwrap
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
from app.llm.job_classifier import (
    GEMINI_FRONTIER_MODEL_ID,
    ANTHROPIC_FRONTIER_MODEL_ID,
    OPENAI_FRONTIER_MODEL_ID,
    compute_modality_flags,
)
from app.llm.gemini_vision import transcribe_video_for_context

# Critique schemas (Block 5)
from app.llm.pipeline.critique_schemas import (
    CritiqueResult,
    CritiqueIssue,
    parse_critique_output,
    build_json_critique_prompt,
    build_json_revision_prompt,
)

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

# Ledger events (Blocks 4, 5, 6)
try:
    from app.pot_spec.ledger import (
        append_event,
        emit_arch_created,
        emit_arch_mirror_written,
        emit_critique_created,
        emit_critique_pass,
        emit_critique_fail,
        emit_revision_loop_started,
        emit_arch_revised,
        emit_revision_loop_terminated,
    )
    from app.pot_spec.service import get_job_artifact_root
    LEDGER_AVAILABLE = True
except ImportError:
    LEDGER_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

AUDIT_ENABLED = os.getenv("ORB_AUDIT_ENABLED", "1") == "1"

logger = logging.getLogger(__name__)

MIN_CRITIQUE_CHARS = int(os.getenv("ORB_MIN_CRITIQUE_CHARS", "1500"))
GEMINI_CRITIC_MODEL = os.getenv("GEMINI_OPUS_CRITIC_MODEL", "gemini-2.0-flash")
GEMINI_CRITIC_MAX_TOKENS = int(os.getenv("GEMINI_CRITIC_MAX_TOKENS", "2048"))
OPUS_REVISION_MAX_TOKENS = int(os.getenv("OPUS_REVISION_MAX_TOKENS", "4096"))

# Block 6: Revision loop config
MAX_REVISION_ITERATIONS = int(os.getenv("ORB_MAX_REVISION_ITERATIONS", "3"))

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
    except Exception:
        return None, None

    if not audit_logger:
        return None, None

    request_id = str(getattr(envelope, "job_id", None) or getattr(task, "request_id", None) or str(uuid4()))
    session_id = str(getattr(envelope, "session_id", None) or getattr(task, "session_id", None) or "session-unknown")
    project_id = int(getattr(envelope, "project_id", None) or getattr(task, "project_id", None) or 0)
    sandbox_mode = bool(getattr(task, "sandbox_mode", False))

    try:
        trace = audit_logger.start_trace(
            session_id=session_id,
            project_id=project_id,
            user_text=None,
            is_critical=True,
            sandbox_mode=sandbox_mode,
            request_id=request_id,
        )
        trace.log_routing_decision(
            job_type=job_type_str,
            provider=provider_id,
            model=model_id,
            reason="high_stakes_pipeline",
        )
    except Exception:
        return audit_logger, None

    return audit_logger, trace


def _maybe_complete_trace(audit_logger, trace, *, success: bool = True, error_message: str = "") -> None:
    if not audit_logger or not trace:
        return
    try:
        audit_logger.complete_trace(trace, success=bool(success), error_message=str(error_message or ""))
    except Exception:
        return


def _trace_step(trace, step: str, **kv) -> None:
    if not trace:
        return
    try:
        extra = " ".join([f"{k}={v}" for k, v in (kv or {}).items()])
        msg = f"{step}" + (f" {extra}" if extra else "")
        trace.log_warning("PIPELINE_STEP", msg)
    except Exception:
        return


def _trace_error(trace, step: str, message: str) -> None:
    if not trace:
        return
    try:
        trace.log_error("PIPELINE_ERROR", f"{step}: {message}")
    except Exception:
        return


# =============================================================================
# Environment Context
# =============================================================================

def get_environment_context() -> Dict[str, Any]:
    """Get current environment context for architecture critique."""
    return {
        "deployment_type": "single_host",
        "os": "Windows 11",
        "repos": ["D:\\Orb\\", "D:\\SandboxOrb\\"],
        "team_size": "solo_developer",
        "infrastructure": "local_only",
        "phase": "early_self_improvement_pipeline",
        "constraints": {
            "no_kubernetes": True,
            "no_docker_orchestration": True,
            "no_multi_host": True,
            "no_vlans": True,
            "no_external_ci": True,
            "no_separate_vms": True,
            "prefer_local_controls": True,
            "prefer_file_permissions": True,
            "prefer_process_isolation": True,
        },
        "acceptable_infra": [
            "Windows security features",
            "Local sandboxing",
            "File-based audit logs",
            "Local credentials manager",
            "Single machine deployment",
        ],
    }


# =============================================================================
# High-Stakes Routing Helpers
# =============================================================================

def normalize_job_type_for_high_stakes(job_type_str: str, reason: str = "") -> str:
    """Normalize job_type_str for high-stakes pipeline."""
    if not job_type_str:
        return "general_high_stakes"

    job_type_str = str(job_type_str).strip().lower()

    if job_type_str in HIGH_STAKES_JOB_TYPES and job_type_str != "orchestrator":
        return job_type_str

    if job_type_str == "orchestrator":
        reason_lower = reason.lower()

        if any(kw in reason_lower for kw in ["architecture", "system design", "architect"]):
            return "architecture_design"
        if any(kw in reason_lower for kw in ["security", "security review"]):
            return "security_review"
        if any(kw in reason_lower for kw in ["infrastructure", "infra", "deployment"]):
            return "high_stakes_infra"
        return "big_architecture"

    if "security" in job_type_str:
        return "security_review"
    if "arch" in job_type_str:
        return "architecture_design"
    if "infra" in job_type_str or "deploy" in job_type_str:
        return "high_stakes_infra"

    return "general_high_stakes"


def is_high_stakes_job(job_type_str: str) -> bool:
    """Check if job type requires high-stakes critique pipeline."""
    return (job_type_str or "").strip().lower() in HIGH_STAKES_JOB_TYPES


def is_opus_model(model_id: str) -> bool:
    """Check if model is a Claude Opus variant."""
    return bool(model_id and "opus" in model_id.lower())


def is_long_enough_for_critique(text: str) -> bool:
    """Only critique if draft is long enough to justify the extra tokens."""
    return bool(text and len(text) >= MIN_CRITIQUE_CHARS)


# =============================================================================
# Block 4: Architecture Artifact Storage
# =============================================================================

def _compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _utc_iso() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def store_architecture_artifact(
    *,
    db,  # SQLAlchemy Session
    job_id: str,
    project_id: int,
    arch_content: str,
    spec_id: str,
    spec_hash: str,
    arch_version: int = 1,
    model: str = "",
    previous_arch_id: Optional[str] = None,
) -> Tuple[str, str, str]:
    """Store architecture document as versioned artifact with filesystem mirror.
    
    Returns (arch_id, arch_hash, mirror_path)
    """
    arch_id = str(uuid4())
    arch_hash = _compute_content_hash(arch_content)
    mirror_path = ""
    artefact_id = None
    
    # 1. Store in DB via ArtefactService
    if ARTEFACTS_AVAILABLE and db is not None:
        try:
            metadata = {
                "spec_id": spec_id,
                "spec_hash": spec_hash,
                "arch_version": arch_version,
                "model": model,
                "previous_arch_id": previous_arch_id,
            }
            artefact = write_architecture_doc(
                db=db,
                project_id=project_id,
                name=f"arch_v{arch_version}",
                content=arch_content,
                job_id=job_id,
                metadata=metadata,
            )
            artefact_id = artefact.id
            logger.info(f"[arch] Stored architecture artifact: {artefact_id}")
        except Exception as e:
            logger.warning(f"[arch] Failed to store in DB: {e}")
    
    # 2. Mirror to filesystem
    if LEDGER_AVAILABLE:
        try:
            job_root = get_job_artifact_root()
            arch_dir = Path(job_root) / job_id / "arch"
            arch_dir.mkdir(parents=True, exist_ok=True)
            
            mirror_path = str(arch_dir / f"arch_v{arch_version}.md")
            Path(mirror_path).write_text(arch_content, encoding="utf-8")
            
            # Emit mirror event
            emit_arch_mirror_written(
                job_artifact_root=job_root,
                job_id=job_id,
                arch_id=arch_id,
                arch_version=arch_version,
                mirror_path=mirror_path,
                checksum=arch_hash,
            )
            
            logger.info(f"[arch] Mirrored to: {mirror_path}")
        except Exception as e:
            logger.warning(f"[arch] Failed to mirror to filesystem: {e}")
    
    # 3. Emit ARCH_CREATED event
    if LEDGER_AVAILABLE:
        try:
            job_root = get_job_artifact_root()
            emit_arch_created(
                job_artifact_root=job_root,
                job_id=job_id,
                arch_id=arch_id,
                arch_version=arch_version,
                arch_hash=arch_hash,
                spec_id=spec_id,
                spec_hash=spec_hash,
                artefact_id=artefact_id,
                mirror_path=mirror_path,
                model=model,
            )
        except Exception as e:
            logger.warning(f"[arch] Failed to emit ledger event: {e}")
    
    return arch_id, arch_hash, mirror_path


# =============================================================================
# Block 5: Structured JSON Critique
# =============================================================================

def store_critique_artifact(
    *,
    job_id: str,
    arch_id: str,
    arch_version: int,
    critique: CritiqueResult,
) -> Tuple[str, str, str]:
    """Store critique as JSON + MD artifacts.
    
    Returns (critique_id, json_path, md_path)
    """
    critique_id = str(uuid4())
    json_path = ""
    md_path = ""
    
    if LEDGER_AVAILABLE:
        try:
            job_root = get_job_artifact_root()
            critique_dir = Path(job_root) / job_id / "critique"
            critique_dir.mkdir(parents=True, exist_ok=True)
            
            # Write JSON artifact
            json_path = str(critique_dir / f"critique_v{arch_version}.json")
            Path(json_path).write_text(critique.to_json(), encoding="utf-8")
            
            # Write MD artifact (human-readable)
            md_path = str(critique_dir / f"critique_v{arch_version}.md")
            Path(md_path).write_text(critique.to_markdown(), encoding="utf-8")
            
            # Emit events
            emit_critique_created(
                job_artifact_root=job_root,
                job_id=job_id,
                critique_id=critique_id,
                arch_id=arch_id,
                arch_version=arch_version,
                blocking_count=len(critique.blocking_issues),
                non_blocking_count=len(critique.non_blocking_issues),
                overall_pass=critique.overall_pass,
                model=critique.critique_model,
                json_path=json_path,
                md_path=md_path,
            )
            
            # Emit pass/fail event
            if critique.overall_pass:
                emit_critique_pass(
                    job_artifact_root=job_root,
                    job_id=job_id,
                    critique_id=critique_id,
                    arch_id=arch_id,
                    arch_version=arch_version,
                )
            else:
                emit_critique_fail(
                    job_artifact_root=job_root,
                    job_id=job_id,
                    critique_id=critique_id,
                    arch_id=arch_id,
                    arch_version=arch_version,
                    blocking_issues=[i.id for i in critique.blocking_issues],
                )
            
            logger.info(f"[critique] Stored: {json_path}")
        except Exception as e:
            logger.warning(f"[critique] Failed to store artifacts: {e}")
    
    return critique_id, json_path, md_path


async def call_json_critic(
    *,
    arch_content: str,
    original_request: str,
    spec_json: Optional[str] = None,
    env_context: Optional[Dict[str, Any]] = None,
    envelope: JobEnvelope,
) -> CritiqueResult:
    """Call Gemini critic with JSON output schema.
    
    Returns structured CritiqueResult.
    """
    critique_prompt = build_json_critique_prompt(
        draft_text=arch_content,
        original_request=original_request,
        spec_json=spec_json,
        env_context=env_context,
    )
    
    critique_messages = [
        {"role": "system", "content": "You are a critical architecture reviewer. Output ONLY valid JSON."},
        {"role": "user", "content": critique_prompt},
    ]
    
    try:
        critic_envelope = JobEnvelope(
            job_id=str(uuid4()),
            session_id=getattr(envelope, 'session_id', 'session-unknown'),
            project_id=int(getattr(envelope, 'project_id', 0)),
            job_type=getattr(Phase4JobType, "CRITIQUE_REVIEW", list(Phase4JobType)[0]),
            importance=Importance.HIGH,
            data_sensitivity=DataSensitivity.INTERNAL,
            modalities_in=[Modality.TEXT],
            budget=JobBudget(
                max_tokens=GEMINI_CRITIC_MAX_TOKENS,
                max_cost_estimate=0.05,
                max_wall_time_seconds=90,
            ),
            output_contract=OutputContract.TEXT_RESPONSE,
            messages=critique_messages,
            metadata={"critic": "gemini_json"},
            allow_multi_model_review=False,
            needs_tools=[],
        )
        
        result = await registry_llm_call(
            provider_id="google",
            model_id=GEMINI_CRITIC_MODEL,
            messages=critique_messages,
            job_envelope=critic_envelope,
        )
        
        if not result or not result.content:
            logger.warning("[critic] Empty response from Gemini critic")
            return CritiqueResult(
                summary="Critique failed: empty response",
                critique_model=GEMINI_CRITIC_MODEL,
            )
        
        critique = parse_critique_output(result.content, model=GEMINI_CRITIC_MODEL)
        return critique
        
    except Exception as exc:
        logger.warning(f"[critic] JSON critic call failed: {exc}")
        return CritiqueResult(
            summary=f"Critique failed: {exc}",
            critique_model=GEMINI_CRITIC_MODEL,
        )


# =============================================================================
# Block 6: Revision Loop Controller
# =============================================================================

async def call_revision(
    *,
    arch_content: str,
    original_request: str,
    critique: CritiqueResult,
    spec_json: Optional[str] = None,
    opus_model_id: str,
    envelope: JobEnvelope,
) -> Optional[str]:
    """Call Opus to revise architecture based on blocking issues.
    
    Returns revised architecture content or None on failure.
    """
    revision_prompt = build_json_revision_prompt(
        draft_text=arch_content,
        original_request=original_request,
        critique=critique,
        spec_json=spec_json,
    )
    
    revision_messages = [
        {"role": "system", "content": "You are revising an architecture document. Output the complete revised document."},
        {"role": "user", "content": revision_prompt},
    ]
    
    try:
        opus_model = os.getenv("ANTHROPIC_OPUS_MODEL", opus_model_id)
        
        revision_envelope = JobEnvelope(
            job_id=str(uuid4()),
            session_id=getattr(envelope, 'session_id', 'session-unknown'),
            project_id=int(getattr(envelope, 'project_id', 0)),
            job_type=getattr(Phase4JobType, "APP_ARCHITECTURE", list(Phase4JobType)[0]),
            importance=Importance.HIGH,
            data_sensitivity=DataSensitivity.INTERNAL,
            modalities_in=[Modality.TEXT],
            budget=JobBudget(
                max_tokens=OPUS_REVISION_MAX_TOKENS,
                max_cost_estimate=0.15,
                max_wall_time_seconds=120,
            ),
            output_contract=OutputContract.TEXT_RESPONSE,
            messages=revision_messages,
            metadata={"revision": True},
            allow_multi_model_review=False,
            needs_tools=[],
        )
        
        result = await registry_llm_call(
            provider_id="anthropic",
            model_id=opus_model,
            messages=revision_messages,
            job_envelope=revision_envelope,
        )
        
        if not result or not result.content:
            return None
        
        return result.content
        
    except Exception as exc:
        logger.warning(f"[revision] Revision call failed: {exc}")
        return None


async def run_revision_loop(
    *,
    db,
    job_id: str,
    project_id: int,
    arch_content: str,
    arch_id: str,
    spec_id: str,
    spec_hash: str,
    spec_json: Optional[str],
    original_request: str,
    opus_model_id: str,
    envelope: JobEnvelope,
    env_context: Optional[Dict[str, Any]] = None,
) -> Tuple[str, int, bool, CritiqueResult]:
    """Run the revision loop until critique passes or max iterations.
    
    Returns (final_content, final_version, passed, final_critique)
    """
    current_content = arch_content
    current_version = 1
    iterations_used = 0
    
    # Emit loop start
    if LEDGER_AVAILABLE:
        try:
            job_root = get_job_artifact_root()
            emit_revision_loop_started(
                job_artifact_root=job_root,
                job_id=job_id,
                arch_id=arch_id,
                max_iterations=MAX_REVISION_ITERATIONS,
            )
        except Exception:
            pass
    
    final_critique = CritiqueResult()
    
    for iteration in range(MAX_REVISION_ITERATIONS):
        iterations_used = iteration + 1
        logger.info(f"[revision_loop] Iteration {iterations_used}/{MAX_REVISION_ITERATIONS}")
        
        # 1. Critique current architecture
        critique = await call_json_critic(
            arch_content=current_content,
            original_request=original_request,
            spec_json=spec_json,
            env_context=env_context,
            envelope=envelope,
        )
        
        # 2. Store critique artifact
        store_critique_artifact(
            job_id=job_id,
            arch_id=arch_id,
            arch_version=current_version,
            critique=critique,
        )
        
        final_critique = critique
        
        # 3. Check if passed
        if critique.overall_pass:
            logger.info(f"[revision_loop] Critique passed at iteration {iterations_used}")
            break
        
        # 4. If not last iteration, revise
        if iteration < MAX_REVISION_ITERATIONS - 1:
            logger.info(f"[revision_loop] Revising to address {len(critique.blocking_issues)} blocking issues")
            
            revised_content = await call_revision(
                arch_content=current_content,
                original_request=original_request,
                critique=critique,
                spec_json=spec_json,
                opus_model_id=opus_model_id,
                envelope=envelope,
            )
            
            if revised_content:
                old_version = current_version
                current_version += 1
                current_content = revised_content
                
                # Store revised architecture
                new_arch_id, new_hash, _ = store_architecture_artifact(
                    db=db,
                    job_id=job_id,
                    project_id=project_id,
                    arch_content=current_content,
                    spec_id=spec_id,
                    spec_hash=spec_hash,
                    arch_version=current_version,
                    model=opus_model_id,
                    previous_arch_id=arch_id,
                )
                
                # Emit revision event
                if LEDGER_AVAILABLE:
                    try:
                        job_root = get_job_artifact_root()
                        emit_arch_revised(
                            job_artifact_root=job_root,
                            job_id=job_id,
                            arch_id=arch_id,
                            old_version=old_version,
                            new_version=current_version,
                            new_hash=new_hash,
                            addressed_issues=[i.id for i in critique.blocking_issues],
                            model=opus_model_id,
                        )
                    except Exception:
                        pass
            else:
                logger.warning("[revision_loop] Revision failed, stopping loop")
                break
    
    # Emit loop termination
    if LEDGER_AVAILABLE:
        try:
            job_root = get_job_artifact_root()
            reason = "pass" if final_critique.overall_pass else "max_iterations"
            emit_revision_loop_terminated(
                job_artifact_root=job_root,
                job_id=job_id,
                arch_id=arch_id,
                final_version=current_version,
                reason=reason,
                iterations_used=iterations_used,
                final_pass=final_critique.overall_pass,
            )
        except Exception:
            pass
    
    return current_content, current_version, final_critique.overall_pass, final_critique


# =============================================================================
# Legacy Prompt Builders (kept for backward compatibility)
# =============================================================================

def build_critique_prompt_for_architecture(
    draft_text: str,
    original_request: str,
    env_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build architecture-specific critique prompt (legacy prose format)."""
    env_text = ""
    if env_context:
        env_text = f"\n\nENVIRONMENT CONTEXT:\n{env_context}\n"

    return textwrap.dedent(
        f"""
        You are a senior architecture reviewer. Critique the following draft response for:
        - Technical correctness
        - Completeness against the original request
        - Security implications (if relevant)
        - Practicality in the given environment (single-host, Windows 11, solo dev, local-only)
        - Clarity and actionability

        Provide a structured critique with:
        1) Critical issues (must-fix)
        2) Important improvements (should-fix)
        3) Nice-to-haves (could-fix)
        4) Missing assumptions / unknowns
        5) Suggested revised outline

        ORIGINAL REQUEST:
        {original_request}

        DRAFT RESPONSE:
        {draft_text}
        {env_text}
        """
    ).strip()


def build_critique_prompt_for_security(draft_text: str, original_request: str) -> str:
    """Build security-specific critique prompt."""
    return textwrap.dedent(
        f"""
        You are a senior security reviewer. Critique the following draft response for:
        - Security correctness (threats, mitigations, assumptions)
        - Missing controls or hardening steps
        - Risk prioritization
        - Practicality for a solo developer on Windows 11
        - Potential policy/safety issues

        ORIGINAL REQUEST:
        {original_request}

        DRAFT RESPONSE:
        {draft_text}
        """
    ).strip()


def build_critique_prompt_for_general(draft_text: str, original_request: str, job_type_str: str) -> str:
    """Build general critique prompt for non-architecture/security high-stakes."""
    return textwrap.dedent(
        f"""
        You are a critical reviewer. Critique the following draft response for:
        - Correctness
        - Completeness
        - Clarity
        - Logical consistency
        - Actionability

        Job type context: {job_type_str}

        ORIGINAL REQUEST:
        {original_request}

        DRAFT RESPONSE:
        {draft_text}
        """
    ).strip()


def build_critique_prompt(
    draft_text: str,
    original_request: str,
    job_type_str: str,
    env_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Dispatch critique prompt builder based on job type (legacy)."""
    jt = (job_type_str or "").strip().lower()
    if jt in ("architecture_design", "big_architecture", "high_stakes_infra", "architecture", "orchestrator"):
        return build_critique_prompt_for_architecture(draft_text, original_request, env_context=env_context)
    if jt in ("security_review", "compliance_review"):
        return build_critique_prompt_for_security(draft_text, original_request)
    return build_critique_prompt_for_general(draft_text, original_request, job_type_str=jt)


# =============================================================================
# Job Type Mapping
# =============================================================================

def _map_to_phase4_job_type(job_type: JobType) -> Phase4JobType:
    """Map router JobType to Phase 4 JobType."""
    def _p4(*names: str) -> Phase4JobType:
        for n in names:
            v = getattr(Phase4JobType, n, None)
            if v is not None:
                return v
        return list(Phase4JobType)[0]

    name = getattr(job_type, "name", "") or ""
    value = getattr(job_type, "value", "") or ""
    key = f"{name} {value} {job_type}".strip().lower()

    if "critique" in key or "review" in key:
        return _p4("CRITIQUE_REVIEW", "CHAT_RESEARCH")
    if "video" in key:
        return _p4("VIDEO_ADVANCED", "VIDEO_SIMPLE")
    if any(tok in key for tok in ("image", "vision", "ocr")):
        return _p4("VISION_COMPLEX", "VISION_SIMPLE")
    if any(tok in key for tok in ("arch", "architecture")):
        return _p4("APP_ARCHITECTURE", "ORCHESTRATION_PLAN")
    if any(tok in key for tok in ("repo", "refactor", "code")):
        return _p4("CODE_REPO", "CODE_SMALL")

    return _p4("CHAT_RESEARCH", "CHAT_SIMPLE")


# =============================================================================
# Legacy Critique Pipeline (Prose-based)
# =============================================================================

async def call_gemini_critic(
    original_task: LLMTask,
    draft_result: LLMResult,
    job_type_str: str,
    envelope: JobEnvelope,
) -> Optional[LLMResult]:
    """Call Gemini to critique the Opus draft (legacy prose format)."""
    user_messages = [m for m in original_task.messages if m.get("role") == "user"]
    original_request = user_messages[-1].get("content", "") if user_messages else ""

    env_context = None
    if job_type_str in ["architecture_design", "big_architecture", "high_stakes_infra"]:
        env_context = get_environment_context()

    critique_prompt = build_critique_prompt(
        draft_text=draft_result.content,
        original_request=original_request,
        job_type_str=job_type_str,
        env_context=env_context,
    )

    critique_messages = [
        {"role": "system", "content": "You are a critical reviewer. Provide direct critique."},
        {"role": "user", "content": critique_prompt},
    ]

    try:
        critic_envelope = JobEnvelope(
            job_id=str(uuid4()),
            session_id=getattr(envelope, 'session_id', 'session-unknown'),
            project_id=int(getattr(envelope, 'project_id', 0)),
            job_type=getattr(Phase4JobType, "CRITIQUE_REVIEW", list(Phase4JobType)[0]),
            importance=Importance.HIGH,
            data_sensitivity=DataSensitivity.INTERNAL,
            modalities_in=[Modality.TEXT],
            budget=JobBudget(
                max_tokens=GEMINI_CRITIC_MAX_TOKENS,
                max_cost_estimate=0.05,
                max_wall_time_seconds=60,
            ),
            output_contract=OutputContract.TEXT_RESPONSE,
            messages=critique_messages,
            metadata={"critic": "gemini"},
            allow_multi_model_review=False,
            needs_tools=[],
        )

        result = await registry_llm_call(
            provider_id="google",
            model_id=GEMINI_CRITIC_MODEL,
            messages=critique_messages,
            job_envelope=critic_envelope,
        )

        if not result:
            return None

        return LLMResult(
            content=result.content,
            provider="google",
            model=GEMINI_CRITIC_MODEL,
            finish_reason="stop",
            error_message=None,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
            cost_usd=result.usage.cost_estimate,
            raw_response=result.raw_response,
        )

    except Exception as exc:
        logger.warning(f"[critic] Gemini critic call failed: {exc}")
        return None


async def call_opus_revision(
    original_task: LLMTask,
    draft_result: LLMResult,
    critique_result: LLMResult,
    opus_model_id: str,
    envelope: JobEnvelope,
) -> Optional[LLMResult]:
    """Call Opus to revise its draft based on critique (legacy)."""
    user_messages = [m for m in original_task.messages if m.get("role") == "user"]
    original_request = user_messages[-1].get("content", "") if user_messages else ""

    revision_prompt = f"""You are revising your own previous answer using a critique.

ORIGINAL REQUEST:
{original_request}

YOUR DRAFT ANSWER:
{draft_result.content}

CRITIQUE:
{critique_result.content}
"""

    revision_messages = [
        {"role": "system", "content": "You are revising your own answer. Output only the improved final answer."},
        {"role": "user", "content": revision_prompt},
    ]

    try:
        opus_model = os.getenv("ANTHROPIC_OPUS_MODEL", opus_model_id)
        phase4_job_type = _map_to_phase4_job_type(original_task.job_type)

        revision_envelope = JobEnvelope(
            job_id=str(uuid4()),
            session_id=getattr(envelope, 'session_id', 'session-unknown'),
            project_id=int(getattr(envelope, 'project_id', 0)),
            job_type=phase4_job_type,
            importance=Importance.HIGH,
            data_sensitivity=DataSensitivity.INTERNAL,
            modalities_in=[Modality.TEXT],
            budget=JobBudget(
                max_tokens=OPUS_REVISION_MAX_TOKENS,
                max_cost_estimate=0.10,
                max_wall_time_seconds=60,
            ),
            output_contract=OutputContract.TEXT_RESPONSE,
            messages=revision_messages,
            metadata={"revision_of_draft": True},
            allow_multi_model_review=False,
            needs_tools=[],
        )

        result = await registry_llm_call(
            provider_id="anthropic",
            model_id=opus_model,
            messages=revision_messages,
            job_envelope=revision_envelope,
        )

        if not result:
            return None

        return LLMResult(
            content=result.content,
            provider="anthropic",
            model=opus_model,
            finish_reason="stop",
            error_message=None,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
            cost_usd=result.usage.cost_estimate,
            raw_response=result.raw_response,
        )

    except Exception as exc:
        logger.warning(f"[critic] Opus revision call failed: {exc}")
        return None


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
    
    if transcripts_text:
        draft_messages.append({"role": "system", "content": f"Video context:\n{transcripts_text.strip()}"})
    
    if file_map:
        draft_messages.append({"role": "system", "content": f"{file_map}\n\nRefer to files using [FILE_X] identifiers."})
    
    if trace:
        _trace_step(trace, 'draft')
    
    try:
        draft_result = await registry_llm_call(
            provider_id=provider_id,
            model_id=model_id,
            messages=draft_messages,
            job_envelope=envelope,
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
    critique = await call_gemini_critic(original_task=task, draft_result=draft, job_type_str=job_type_str, envelope=envelope)
    
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
    # Routing helpers
    "normalize_job_type_for_high_stakes",
    "is_high_stakes_job",
    "is_opus_model",
    "is_long_enough_for_critique",
    "get_environment_context",
    "HIGH_STAKES_JOB_TYPES",
    # Block 4: Architecture storage
    "store_architecture_artifact",
    # Block 5: JSON critique
    "call_json_critic",
    "store_critique_artifact",
    # Block 6: Revision loop
    "call_revision",
    "run_revision_loop",
    # Main entry
    "run_high_stakes_with_critique",
    # Legacy
    "call_gemini_critic",
    "call_opus_revision",
    "build_critique_prompt",
]