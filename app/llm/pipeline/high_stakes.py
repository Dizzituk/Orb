# FILE: app/llm/pipeline/high_stakes.py
"""High-stakes critique pipeline configuration and prompt builders.

Extracted from app.llm.router to keep the router thin and make sanity checks easier.
"""

from __future__ import annotations

import logging
import os
import textwrap
from typing import Any, Dict, Optional
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


# =============================================================================
# Audit integration (stable API via app.llm.audit_logger)
# =============================================================================

def _maybe_start_trace(
    task: LLMTask,
    envelope: JobEnvelope,
    *,
    job_type_str: str,
    provider_id: str,
    model_id: str,
):
    """Start an audit trace if auditing is enabled. Audit must never break routing."""
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
        # Canonical routing decision event (enum-safe)
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

AUDIT_ENABLED = os.getenv("ORB_AUDIT_ENABLED", "1") == "1"

logger = logging.getLogger(__name__)

MIN_CRITIQUE_CHARS = int(os.getenv("ORB_MIN_CRITIQUE_CHARS", "1500"))
GEMINI_CRITIC_MODEL = os.getenv("GEMINI_OPUS_CRITIC_MODEL", "gemini-3-pro")
GEMINI_CRITIC_MAX_TOKENS = int(os.getenv("GEMINI_CRITIC_MAX_TOKENS", "1024"))
OPUS_REVISION_MAX_TOKENS = int(os.getenv("OPUS_REVISION_MAX_TOKENS", "2048"))

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
# ENVIRONMENT CONTEXT
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
# HIGH-STAKES ROUTING HELPERS
# =============================================================================


def normalize_job_type_for_high_stakes(job_type_str: str, reason: str = "") -> str:
    """
    Normalize job_type_str for high-stakes pipeline.
    We keep the job classifier flexible, but map relevant categories into stable pipeline types.
    """
    if not job_type_str:
        return "general_high_stakes"

    job_type_str = str(job_type_str).strip().lower()

    if job_type_str in HIGH_STAKES_JOB_TYPES and job_type_str != "orchestrator":
        return job_type_str

    # ORCHESTRATOR can mean many things; map based on reason heuristics
    if job_type_str == "orchestrator":
        reason_lower = reason.lower()

        architecture_keywords = ["architecture", "system design", "architect", "system architecture"]
        if any(kw in reason_lower for kw in architecture_keywords):
            logger.info("[router] Normalized orchestrator → architecture_design (reason: %s)", reason[:80])
            return "architecture_design"

        security_keywords = ["security", "security review", "security audit"]
        if any(kw in reason_lower for kw in security_keywords):
            logger.info("[router] Normalized orchestrator → security_review (reason: %s)", reason[:80])
            return "security_review"

        infra_keywords = ["infrastructure", "infra", "deployment"]
        if any(kw in reason_lower for kw in infra_keywords):
            logger.info("[router] Normalized orchestrator → high_stakes_infra (reason: %s)", reason[:80])
            return "high_stakes_infra"

        logger.info("[router] Normalized orchestrator → big_architecture (default, reason: %s)", reason[:80])
        return "big_architecture"

    # Generic mapping for other job types that might still be high impact
    if "security" in job_type_str:
        return "security_review"
    if "arch" in job_type_str:
        return "architecture_design"
    if "infra" in job_type_str or "deploy" in job_type_str:
        return "high_stakes_infra"

    return "general_high_stakes"


def is_high_stakes_job(job_type_str: str) -> bool:
    """Check if job type requires high-stakes critique pipeline."""
    job_type_str = (job_type_str or "").strip().lower()
    return job_type_str in HIGH_STAKES_JOB_TYPES


def is_opus_model(model_id: str) -> bool:
    """Check if model is a Claude Opus variant."""
    if not model_id:
        return False
    m = model_id.lower()
    return "opus" in m


def is_long_enough_for_critique(text: str) -> bool:
    """Only critique if draft is long enough to justify the extra tokens."""
    return bool(text and len(text) >= MIN_CRITIQUE_CHARS)


# =============================================================================
# CRITIQUE PROMPT BUILDERS
# =============================================================================


def build_critique_prompt_for_architecture(
    draft_text: str,
    original_request: str,
    env_context: Optional[Dict[str, Any]] = None,
) -> str:
    """Build architecture-specific critique prompt."""
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

        Provide a structured critique with:
        1) Critical security gaps
        2) Risk-ranked recommendations
        3) Potential misconfigurations
        4) Attack surface considerations
        5) Suggested revised outline

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

        Provide a structured critique with:
        1) Critical issues
        2) Important improvements
        3) Missing details
        4) Suggested revised outline

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
    """Dispatch critique prompt builder based on job type."""
    jt = (job_type_str or "").strip().lower()
    if jt in ("architecture_design", "big_architecture", "high_stakes_infra", "architecture", "orchestrator"):
        return build_critique_prompt_for_architecture(draft_text, original_request, env_context=env_context)
    if jt in ("security_review", "compliance_review"):
        return build_critique_prompt_for_security(draft_text, original_request)
    return build_critique_prompt_for_general(draft_text, original_request, job_type_str=jt)


def _map_to_phase4_job_type(job_type: JobType) -> Phase4JobType:
    """Map router JobType enum (app.llm.schemas.JobType) to Phase 4 JobType enum (app.jobs.schemas.JobType).

    Router JobType and Phase-4 JobType evolve independently. This mapper must:
    - never reference missing enum members directly (would raise AttributeError),
    - return a *valid* Phase4JobType member for all inputs.
    """

    def _p4(*names: str) -> Phase4JobType:
        for n in names:
            v = getattr(Phase4JobType, n, None)
            if v is not None:
                return v
        return list(Phase4JobType)[0]

    # Build a robust key for fuzzy routing across legacy names.
    name = getattr(job_type, "name", "") or ""
    value = getattr(job_type, "value", "") or ""
    key = f"{name} {value} {job_type}".strip().lower()

    # === Explicit primary-route mappings (8-route system) ===
    if getattr(JobType, "CHAT_LIGHT", None) is not None and job_type == getattr(JobType, "CHAT_LIGHT"):
        return _p4("CHAT_SIMPLE")

    if getattr(JobType, "TEXT_HEAVY", None) is not None and job_type == getattr(JobType, "TEXT_HEAVY"):
        return _p4("CHAT_RESEARCH", "CHAT_SIMPLE")

    if getattr(JobType, "CODE_MEDIUM", None) is not None and job_type == getattr(JobType, "CODE_MEDIUM"):
        return _p4("CODE_SMALL", "CODE_REPO")

    if getattr(JobType, "ORCHESTRATOR", None) is not None and job_type == getattr(JobType, "ORCHESTRATOR"):
        # ORCHESTRATOR spans architecture + multi-file code; pick best-effort based on legacy hints.
        if "arch" in key:
            return _p4("APP_ARCHITECTURE", "ORCHESTRATION_PLAN")
        if any(tok in key for tok in ("code", "refactor", "migration", "bug", "repo")):
            return _p4("CODE_REPO", "ORCHESTRATION_PLAN")
        return _p4("ORCHESTRATION_PLAN", "CHAT_RESEARCH")

    if getattr(JobType, "IMAGE_SIMPLE", None) is not None and job_type == getattr(JobType, "IMAGE_SIMPLE"):
        return _p4("VISION_SIMPLE", "VISION_COMPLEX")

    if getattr(JobType, "IMAGE_COMPLEX", None) is not None and job_type == getattr(JobType, "IMAGE_COMPLEX"):
        return _p4("VISION_COMPLEX", "VISION_SIMPLE")

    if getattr(JobType, "DOCUMENT_PDF_TEXT", None) is not None and job_type == getattr(JobType, "DOCUMENT_PDF_TEXT"):
        return _p4("CHAT_RESEARCH", "CHAT_SIMPLE")

    if getattr(JobType, "DOCUMENT_PDF_VISION", None) is not None and job_type == getattr(JobType, "DOCUMENT_PDF_VISION"):
        return _p4("VISION_COMPLEX", "VISION_SIMPLE")

    if getattr(JobType, "VIDEO_HEAVY", None) is not None and job_type == getattr(JobType, "VIDEO_HEAVY"):
        return _p4("VIDEO_ADVANCED", "VIDEO_SIMPLE")

    if getattr(JobType, "VIDEO_CODE_DEBUG", None) is not None and job_type == getattr(JobType, "VIDEO_CODE_DEBUG"):
        return _p4("VIDEO_ADVANCED", "VIDEO_SIMPLE")

    if getattr(JobType, "OPUS_CRITIC", None) is not None and job_type == getattr(JobType, "OPUS_CRITIC"):
        return _p4("CRITIQUE_REVIEW", "CHAT_RESEARCH")

    # === Fuzzy fallback for legacy names (safe) ===
    if "critique" in key or "review" in key:
        return _p4("CRITIQUE_REVIEW", "CHAT_RESEARCH")

    if "video" in key:
        return _p4("VIDEO_ADVANCED", "VIDEO_SIMPLE")

    if any(tok in key for tok in ("image", "vision", "ocr", "screenshot", "pdf_vision")):
        return _p4("VISION_COMPLEX", "VISION_SIMPLE")

    if any(tok in key for tok in ("arch", "architecture")):
        return _p4("APP_ARCHITECTURE", "ORCHESTRATION_PLAN")

    if any(tok in key for tok in ("repo", "refactor", "migration", "code", "bugfix", "bug_analysis", "bug")):
        return _p4("CODE_REPO", "CODE_SMALL")

    # Security / privacy / infra (Phase-4 spec has no dedicated security job type)
    if any(tok in key for tok in ("security", "privacy", "infra", "high_stakes")):
        return _p4("ORCHESTRATION_PLAN", "CHAT_RESEARCH")

    return _p4("CHAT_RESEARCH", "CHAT_SIMPLE")


# =============================================================================
# CRITIQUE PIPELINE (Opus draft → Gemini critique → Opus revision)
# =============================================================================


async def call_gemini_critic(
    original_task: LLMTask,
    draft_result: LLMResult,
    job_type_str: str,
    envelope: JobEnvelope,
) -> Optional[LLMResult]:
    """Call Gemini 3 Pro to critique the Opus draft."""
    user_messages = [m for m in original_task.messages if m.get("role") == "user"]
    original_request = user_messages[-1].get("content", "") if user_messages else ""

    env_context = None
    if job_type_str in ["architecture_design", "big_architecture", "high_stakes_infra",
                         "architecture", "orchestrator"]:
        env_context = get_environment_context()
        logger.info("[critic] Passing environment context to architecture critique")

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
            session_id=getattr(envelope, 'session_id', None) or getattr(original_task, 'session_id', None) or 'session-unknown',
            project_id=int(getattr(envelope, 'project_id', None) or getattr(original_task, 'project_id', None) or 0),
            job_type=getattr(Phase4JobType, "CRITIQUE_REVIEW", getattr(Phase4JobType, "CHAT_RESEARCH", list(Phase4JobType)[0])),
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
        logger.warning("[critic] Gemini critic call failed: %s", exc, exc_info=False)
        return None


async def call_opus_revision(
    original_task: LLMTask,
    draft_result: LLMResult,
    critique_result: LLMResult,
    opus_model_id: str,
    envelope: JobEnvelope,
) -> Optional[LLMResult]:
    """Call Opus to revise its draft based on Gemini's critique."""
    user_messages = [m for m in original_task.messages if m.get("role") == "user"]
    original_request = user_messages[-1].get("content", "") if user_messages else ""

    revision_prompt = f"""You are revising your own previous answer using a critique. Keep the user's intent and constraints. Do not mention the critique explicitly. Fix errors, add missing details, and improve clarity while maintaining your technical accuracy. Produce an improved final answer.

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
            session_id=getattr(envelope, 'session_id', None) or getattr(original_task, 'session_id', None) or 'session-unknown',
            project_id=int(getattr(envelope, 'project_id', None) or getattr(original_task, 'project_id', None) or 0),
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
        logger.warning("[critic] Opus revision call failed: %s", exc, exc_info=False)
        return None


async def run_high_stakes_with_critique(
    task: LLMTask,
    provider_id: str,
    model_id: str,
    envelope: JobEnvelope,
    job_type_str: str,
    file_map: Optional[str] = None,
) -> LLMResult:
    """Run 3-step critique pipeline for high-stakes Opus work."""
    logger.info("[critic] High-stakes pipeline enabled: job_type=%s model=%s", job_type_str, model_id)    # v0.15.0: Start audit trace (audit must never break routing)
    audit_logger, trace = _maybe_start_trace(
        task,
        envelope,
        job_type_str=job_type_str,
        provider_id=provider_id,
        model_id=model_id,
    )

    # -------------------------------------------------------------------------
    # Pre-step: If video exists, transcribe for context and append to prompt.
    # -------------------------------------------------------------------------
    attachments = task.attachments or []
    flags = compute_modality_flags(attachments)
    video_attachments = flags.get("video_attachments", [])

    video_transcripts = []
    if video_attachments:
        for video_att in video_attachments:
            try:
                video_path = video_att.path if hasattr(video_att, "path") else None
                if video_path:
                    logger.info("[critic] Pre-step: Transcribing %s", getattr(video_att, "filename", "video"))
                    try:
                        transcript = await transcribe_video_for_context(video_path)
                        video_transcripts.append({
                            "filename": video_att.filename,
                            "transcript": transcript,
                        })
                    except Exception as e:
                        logger.warning("[critic] Pre-step: Transcription failed: %s", e, exc_info=False)
            except Exception:
                pass

    transcripts_text = ""
    if video_transcripts:
        for vt in video_transcripts:
            transcripts_text += f"\n\n=== Video: {vt['filename']} ===\n{vt['transcript']}"
        transcripts_text = transcripts_text.strip()

    # -------------------------------------------------------------------------
    # Step 1: Generate draft (Opus)
    # -------------------------------------------------------------------------
    draft_messages = list(envelope.messages)

    if transcripts_text:
        draft_messages = list(draft_messages) + [
            {"role": "system", "content": f"Additional context from video transcription:\n{transcripts_text}"}
        ]

    if file_map:
        draft_messages = list(draft_messages) + [
            {"role": "system", "content": f"{file_map}\n\nIMPORTANT: Refer to files using [FILE_X] identifiers."}
        ]

    if trace is not None:
        _trace_step(trace, 'draft')

    try:
        draft_result = await registry_llm_call(
            provider_id=provider_id,
            model_id=model_id,
            messages=draft_messages,
            job_envelope=envelope,
        )
    except Exception as exc:
        err_msg = f"High-stakes draft call failed: {type(exc).__name__}: {exc}"
        if trace is not None:
            _trace_error(trace, 'draft', err_msg)
        _maybe_complete_trace(audit_logger, trace, success=False, error_message=err_msg)
        return LLMResult(
            content=err_msg,
            provider=provider_id,
            model=model_id,
            finish_reason="error",
            error_message=err_msg,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            raw_response=None,
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

    if trace is not None:
        _trace_step(trace, 'draft_done')

    # Only critique if the draft is long enough.
    if not is_long_enough_for_critique(draft.content):
        logger.warning("[critic] Draft too short for critique; returning draft.")
        if trace is not None:
            _trace_step(trace, 'skip_critique_short_draft')
            _maybe_complete_trace(audit_logger, trace, success=True)

        draft.routing_decision = {
            "job_type": job_type_str,
            "provider": provider_id,
            "model": model_id,
            "reason": "High-stakes pipeline: draft returned (too short for critique)",
        }
        return draft

    # -------------------------------------------------------------------------
    # Step 2: Critique (Gemini)
    # -------------------------------------------------------------------------
    critique = await call_gemini_critic(original_task=task, draft_result=draft, job_type_str=job_type_str, envelope=envelope)

    if not critique:
        logger.warning("[critic] Critique failed; returning draft.")
        if trace is not None:
            _trace_step(trace, 'critique_failed')
            _maybe_complete_trace(audit_logger, trace, success=True)

        draft.routing_decision = {
            "job_type": job_type_str,
            "provider": provider_id,
            "model": model_id,
            "reason": "High-stakes pipeline: critique failed; returning Opus draft",
        }
        return draft

    if trace is not None:
        _trace_step(trace, 'critique_done')

    # -------------------------------------------------------------------------
    # Step 3: Revision (Opus)
    # -------------------------------------------------------------------------
    revision = await call_opus_revision(
        original_task=task,
        draft_result=draft,
        critique_result=critique,
        opus_model_id=model_id,
        envelope=envelope,
    )

    if not revision:
        logger.warning("[critic] Revision failed; returning draft.")
        if trace is not None:
            _trace_step(trace, 'revision_failed')
            _maybe_complete_trace(audit_logger, trace, success=True)

        draft.routing_decision = {
            "job_type": job_type_str,
            "provider": provider_id,
            "model": model_id,
            "reason": "High-stakes pipeline: revision failed; returning Opus draft",
        }
        return draft

    if trace is not None:
        _trace_step(trace, 'revision_done')
        _maybe_complete_trace(audit_logger, trace, success=True)

    revision.routing_decision = {
        "job_type": job_type_str,
        "provider": provider_id,
        "model": revision.model,
        "reason": "High-stakes pipeline: Opus draft → Gemini critique → Opus revision",
        "critique_pipeline": {
            "draft_tokens": draft.total_tokens,
            "critique_tokens": critique.total_tokens,
            "revision_tokens": revision.total_tokens,
            "total_cost": draft.cost_usd + critique.cost_usd + revision.cost_usd,
        }
    }

    _maybe_complete_trace(audit_logger, trace, success=True)

    return revision