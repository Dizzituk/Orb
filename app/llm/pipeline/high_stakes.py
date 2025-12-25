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
            print(f"[router] Normalized orchestrator → architecture_design (reason: {reason[:80]})")
            return "architecture_design"

        security_keywords = ["security", "security review", "security audit"]
        if any(kw in reason_lower for kw in security_keywords):
            print(f"[router] Normalized orchestrator → security_review (reason: {reason[:80]})")
            return "security_review"

        infra_keywords = ["infrastructure", "infra", "deployment"]
        if any(kw in reason_lower for kw in infra_keywords):
            print(f"[router] Normalized orchestrator → high_stakes_infra (reason: {reason[:80]})")
            return "high_stakes_infra"

        print(f"[router] Normalized orchestrator → big_architecture (default, reason: {reason[:80]})")
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
    """Map router JobType enum to Phase 4 JobType enum.

    NOTE:
    The router's JobType enum has changed over time (e.g. SECURITY vs SECURITY_REVIEW).
    This mapper must be resilient to missing enum members to avoid AttributeError during
    envelope synthesis.
    """
    # Prefer the enum value string when available; fall back to str(job_type)
    try:
        key = (job_type.value or "").strip().lower()  # type: ignore[attr-defined]
    except Exception:
        key = (str(job_type) or "").strip().lower()

    # Guarded direct comparisons (safe even if enum members are missing)
    if getattr(JobType, "CHAT_LIGHT", None) is not None and job_type == getattr(JobType, "CHAT_LIGHT"):
        return Phase4JobType.QUICK_CHAT
    if getattr(JobType, "TEXT_HEAVY", None) is not None and job_type == getattr(JobType, "TEXT_HEAVY"):
        return Phase4JobType.TEXT_HEAVY
    if getattr(JobType, "CODE_MEDIUM", None) is not None and job_type == getattr(JobType, "CODE_MEDIUM"):
        return Phase4JobType.CODE_MEDIUM
    if getattr(JobType, "ORCHESTRATOR", None) is not None and job_type == getattr(JobType, "ORCHESTRATOR"):
        return Phase4JobType.ORCHESTRATOR
    if getattr(JobType, "ARCHITECTURE", None) is not None and job_type == getattr(JobType, "ARCHITECTURE"):
        return Phase4JobType.ARCHITECTURE
    # SECURITY is historically either JobType.SECURITY or JobType.SECURITY_REVIEW (or similar).
    if getattr(JobType, "SECURITY", None) is not None and job_type == getattr(JobType, "SECURITY"):
        return Phase4JobType.SECURITY
    if getattr(JobType, "SECURITY_REVIEW", None) is not None and job_type == getattr(JobType, "SECURITY_REVIEW"):
        return Phase4JobType.SECURITY
    if getattr(JobType, "INFRASTRUCTURE", None) is not None and job_type == getattr(JobType, "INFRASTRUCTURE"):
        return Phase4JobType.INFRASTRUCTURE
    if getattr(JobType, "MIXED_FILE", None) is not None and job_type == getattr(JobType, "MIXED_FILE"):
        return Phase4JobType.MIXED_FILE
    if getattr(JobType, "VIDEO_CODE_DEBUG", None) is not None and job_type == getattr(JobType, "VIDEO_CODE_DEBUG"):
        return Phase4JobType.VIDEO_CODE_DEBUG

    # String-based fallbacks for forward compatibility
    if "security" in key:
        return Phase4JobType.SECURITY
    if "infra" in key or "deploy" in key:
        return Phase4JobType.INFRASTRUCTURE
    if "arch" in key:
        return Phase4JobType.ARCHITECTURE
    if "mixed" in key or "file" in key:
        return Phase4JobType.MIXED_FILE
    if "video" in key:
        return Phase4JobType.VIDEO_CODE_DEBUG
    if "code" in key:
        return Phase4JobType.CODE_MEDIUM
    if "chat" in key:
        return Phase4JobType.QUICK_CHAT

    return Phase4JobType.TEXT_HEAVY
# =============================================================================
# CRITIQUE PIPELINE (Opus draft → Gemini critique → Opus revision)
# =============================================================================


async def call_gemini_critic(
    original_task: LLMTask,
    draft_result: LLMResult,
    job_type_str: str,
) -> Optional[LLMResult]:
    """Call Gemini 3 Pro to critique the Opus draft."""
    user_messages = [m for m in original_task.messages if m.get("role") == "user"]
    original_request = user_messages[-1].get("content", "") if user_messages else ""

    env_context = None
    if job_type_str in ["architecture_design", "big_architecture", "high_stakes_infra",
                         "architecture", "orchestrator"]:
        env_context = get_environment_context()
        print(f"[critic] Passing environment context to architecture critique")

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
            envelope_id=str(uuid4()),
            job_type=Phase4JobType.TEXT_HEAVY,
            importance=Importance.HIGH,
            data_sensitivity=DataSensitivity.INTERNAL,
            modalities=[Modality.TEXT],
            budget=JobBudget(
                max_tokens=GEMINI_CRITIC_MAX_TOKENS,
                max_cost_estimate=0.05,
                max_wall_time_seconds=60,
            ),
            output_contract=OutputContract.TEXT_RESPONSE,
            messages=critique_messages,
            metadata={"critique_for_job_type": job_type_str},
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
        logger.exception("[critic] Gemini critic call failed: %s", exc)
        return None


async def call_opus_revision(
    original_task: LLMTask,
    draft_result: LLMResult,
    critique_result: LLMResult,
    opus_model_id: str,
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

        revision_envelope = JobEnvelope(
            envelope_id=str(uuid4()),
            job_type=Phase4JobType.TEXT_HEAVY,
            importance=Importance.HIGH,
            data_sensitivity=DataSensitivity.INTERNAL,
            modalities=[Modality.TEXT],
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
        logger.exception("[critic] Opus revision call failed: %s", exc)
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
    print(f"[critic] High-stakes pipeline enabled: job_type={job_type_str} model={model_id}")

    # v0.15.0: Start audit trace if available
    trace = None
    if AUDIT_AVAILABLE and AUDIT_ENABLED:
        audit_logger = get_audit_logger()
        if audit_logger:
            trace = RoutingTrace(
                session_id=getattr(task, "session_id", None),
                envelope_id=envelope.envelope_id,
                job_type=job_type_str,
                provider=provider_id,
                model=model_id,
            )
            trace.add_event(AuditEventType.ROUTER_DECISION, {
                "pipeline": "high_stakes_critique",
                "job_type": job_type_str,
                "provider": provider_id,
                "model": model_id,
            })

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
                    print(f"[critic] Pre-step: Transcribing {video_att.filename}")
                    try:
                        transcript = await transcribe_video_for_context(video_path)
                        video_transcripts.append({
                            "filename": video_att.filename,
                            "transcript": transcript,
                        })
                    except Exception as e:
                        print(f"[critic] Pre-step: Transcription failed: {e}")
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
        trace.add_event(AuditEventType.PIPELINE_STEP, {
            "step": "draft",
            "provider": provider_id,
            "model": model_id,
            "messages_count": len(draft_messages),
        })

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
            trace.add_event(AuditEventType.ERROR, {"step": "draft", "error": err_msg})
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
        trace.add_event(AuditEventType.PIPELINE_STEP, {
            "step": "draft_done",
            "tokens": draft.total_tokens,
            "cost": draft.cost_usd,
        })

    # Only critique if the draft is long enough.
    if not is_long_enough_for_critique(draft.content):
        print("[critic] Draft too short for critique; returning draft.")
        if trace is not None:
            trace.add_event(AuditEventType.PIPELINE_STEP, {
                "step": "skip_critique_short_draft",
                "chars": len(draft.content or ""),
            })
            audit_logger = get_audit_logger() if AUDIT_AVAILABLE and AUDIT_ENABLED else None
            if audit_logger:
                audit_logger.write_trace(trace)

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
    critique = await call_gemini_critic(original_task=task, draft_result=draft, job_type_str=job_type_str)

    if not critique:
        print("[critic] Critique failed; returning draft.")
        if trace is not None:
            trace.add_event(AuditEventType.PIPELINE_STEP, {"step": "critique_failed"})
            audit_logger = get_audit_logger() if AUDIT_AVAILABLE and AUDIT_ENABLED else None
            if audit_logger:
                audit_logger.write_trace(trace)

        draft.routing_decision = {
            "job_type": job_type_str,
            "provider": provider_id,
            "model": model_id,
            "reason": "High-stakes pipeline: critique failed; returning Opus draft",
        }
        return draft

    if trace is not None:
        trace.add_event(AuditEventType.PIPELINE_STEP, {
            "step": "critique_done",
            "tokens": critique.total_tokens,
            "cost": critique.cost_usd,
        })

    # -------------------------------------------------------------------------
    # Step 3: Revision (Opus)
    # -------------------------------------------------------------------------
    revision = await call_opus_revision(
        original_task=task,
        draft_result=draft,
        critique_result=critique,
        opus_model_id=model_id,
    )

    if not revision:
        print("[critic] Revision failed; returning draft.")
        if trace is not None:
            trace.add_event(AuditEventType.PIPELINE_STEP, {"step": "revision_failed"})
            audit_logger = get_audit_logger() if AUDIT_AVAILABLE and AUDIT_ENABLED else None
            if audit_logger:
                audit_logger.write_trace(trace)

        draft.routing_decision = {
            "job_type": job_type_str,
            "provider": provider_id,
            "model": model_id,
            "reason": "High-stakes pipeline: revision failed; returning Opus draft",
        }
        return draft

    if trace is not None:
        trace.add_event(AuditEventType.PIPELINE_STEP, {
            "step": "revision_done",
            "tokens": revision.total_tokens,
            "cost": revision.cost_usd,
        })
        audit_logger = get_audit_logger() if AUDIT_AVAILABLE and AUDIT_ENABLED else None
        if audit_logger:
            audit_logger.write_trace(trace)

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

    return revision
