# FILE: app/llm/routing/core.py
"""Core routing implementation (moved from app/llm/router.py).

This file intentionally preserves the existing behavior.
app/llm/router.py is now a thin compatibility wrapper.
"""

# FILE: app/llm/router.py
"""
LLM Router (PHASE 4 - v0.15.1)

Version: 0.15.1 - Simplified OVERRIDE → Frontier Model Routing

v0.15.1 Changes:
- NEW: Simplified OVERRIDE mechanism for frontier model routing
- OVERRIDE (default) → Gemini 3 Pro Preview (GEMINI_FRONTIER_MODEL_ID)
- OVERRIDE works for all jobs unless explicitly disabled
- Enhanced debug logging with audit trail

CRITICAL PIPELINE SPEC COMPLIANCE:
- §1 File Classification: MIXED_FILE detection for PDFs/DOCX with images
- §2 Stable Naming: [FILE_1], [FILE_2], etc. via build_file_map()
- §3 Relationship Detection: Pairwise modality relationships
- §7 Token Budgeting: Context allocation by content type
- §11 Fallbacks: Structured fallback chains
- §12 Audit Logging: Full routing decision trace

8-ROUTE CLASSIFICATION SYSTEM:
- CHAT_LIGHT → OpenAI (gpt-4.1-mini) - casual chat
- TEXT_HEAVY → OpenAI (gpt-4.1) - heavy text, text-only PDFs
- CODE_MEDIUM → Anthropic Sonnet - scoped code (1-3 files)
- ORCHESTRATOR → Anthropic Opus - arc
- ARCHITECTURE → Anthropic Opus - architecture design
- SECURITY → Anthropic Opus - security review
- INFRASTRUCTURE → Anthropic Opus - infra/deployment
- MIXED_FILE → OpenAI (gpt-4.1) - docs/pdfs with images
- VIDEO_CODE_DEBUG → 2-step pipeline: Gemini 3 Pro → Claude Sonnet

HIGH-STAKES CRITIQUE PIPELINE:
- Opus generates draft for ARCHITECTURE/SECURITY/INFRA jobs
- Gemini 3 Pro critiques the draft
- Opus revises based on critique
- Ensures high quality for critical system design/security decisions

ZOBIE MAP LOCAL ACTION:
- Prompt trigger: "ZOBIE MAP [url]"
- Calls external zobie_mapper controller to build architecture map
- Returns map results directly (no LLM call)

"""

import os
import logging
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen, Request
from typing import Optional, List, Dict, Any, Tuple
from uuid import uuid4

from app.llm.schemas import (
    LLMTask,
    LLMResult,
    JobType,
    Provider,
    RoutingConfig,
    RoutingOptions,
    RoutingDecision,
)

# Phase 4 imports
from app.jobs.schemas import (
    JobEnvelope,
    JobType as Phase4JobType,
    Importance,
    DataSensitivity,
    Modality,
    JobBudget,
    OutputContract,
    validate_job_envelope,
    ValidationError,
)

from app.providers.registry import llm_call as registry_llm_call

# Job classifier (8-route system with MIXED_FILE detection)
from app.llm.job_classifier import (
    classify_job,
    classify_and_route as classifier_classify_and_route,
    get_routing_for_job_type,
    get_model_config,
    is_claude_forbidden,
    is_claude_allowed,
    prepare_attachments,
    compute_modality_flags,
    detect_frontier_override,
    GEMINI_FRONTIER_MODEL_ID,
    ANTHROPIC_FRONTIER_MODEL_ID,
    OPENAI_FRONTIER_MODEL_ID,
)

# Video transcription for pipelines
from app.llm.gemini_vision import transcribe_video_for_context

# =============================================================================
# v0.15.0: CRITICAL PIPELINE SPEC MODULE IMPORTS
# =============================================================================

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

# File classification (Spec §1)
try:
    from app.llm.file_classifier import (
        classify_files,
        build_file_map,
        FileType,
        ModalityFlags,
    )

    FILE_CLASSIFIER_AVAILABLE = True
except ImportError:
    FILE_CLASSIFIER_AVAILABLE = False

# Relationship detection (Spec §3)
try:
    from app.llm.relationship_detector import (
        detect_relationships,
        validate_relationships,
        RelationshipType,
    )

    RELATIONSHIP_DETECTOR_AVAILABLE = True
except ImportError:
    RELATIONSHIP_DETECTOR_AVAILABLE = False

# Token budgeting (Spec §7)
try:
    from app.llm.token_budgeting import (
        allocate_token_budget,
        BudgetAllocation,
        ContentType,
    )

    TOKEN_BUDGETING_AVAILABLE = True
except ImportError:
    TOKEN_BUDGETING_AVAILABLE = False

# Fallback chain (Spec §11)
try:
    from app.llm.fallbacks import (
        FallbackManager,
        FailureType,
        FallbackAction,
    )

    FALLBACKS_AVAILABLE = True
except ImportError:
    FALLBACKS_AVAILABLE = False

logger = logging.getLogger(__name__)

# =============================================================================
# EXTRACTED ROUTING HELPERS (Stage 2 split)
# =============================================================================
from app.llm.pipeline.high_stakes import (
    get_environment_context,
    normalize_job_type_for_high_stakes,
    is_high_stakes_job,
    is_opus_model,
    HIGH_STAKES_JOB_TYPES,
    MIN_CRITIQUE_CHARS,
    GEMINI_CRITIC_MODEL,
    run_high_stakes_with_critique,
)
from app.llm.routing.job_routing import (
    classify_and_route,
    inject_file_map_into_messages,
)
from app.llm.routing.envelope import synthesize_envelope_from_task
from app.llm.routing.video_code_debug import run_video_code_debug_pipeline
from app.llm.routing.local_actions import _maybe_handle_zobie_map


# =============================================================================
# ROUTER DEBUG MODE
# =============================================================================
ROUTER_DEBUG = os.getenv("ORB_ROUTER_DEBUG", "0") == "1"
AUDIT_ENABLED = os.getenv("ORB_AUDIT_ENABLED", "1") == "1"


def _debug_log(msg: str):
    """Print debug message if ROUTER_DEBUG is enabled."""
    if ROUTER_DEBUG:
        print(f"[router-debug] {msg}")


# =============================================================================
# ATTACHMENT SAFETY CHECK
# =============================================================================


def _check_attachment_safety(
    task: LLMTask,
    decision: RoutingDecision,
    has_attachments: bool,
    job_type_specified: bool,
) -> Tuple[str, str, str]:
    """
    Attachment safety logic:
    - If attachments exist, and Claude is forbidden: force OpenAI.
    - If attachments exist, and job type is user-specified: honor requested type but enforce provider safety.
    """
    provider = decision.provider
    model = decision.model
    reason = decision.reason

    if not has_attachments:
        return provider, model, reason

    # If Claude is forbidden for these attachments, force OpenAI
    if is_claude_forbidden(task.attachments):
        if provider == Provider.ANTHROPIC:
            print("[router] Claude forbidden for attachments; forcing OpenAI")
            provider = Provider.OPENAI
            model = get_model_config(Provider.OPENAI, task.job_type).model
            reason = f"{reason} | Claude forbidden for attachments → OpenAI fallback"
        return provider, model, reason

    # If job type explicitly specified by user, we respect it, but keep provider safe.
    if job_type_specified:
        if provider == Provider.ANTHROPIC and not is_claude_allowed(task.attachments):
            print("[router] Job type specified but Claude not allowed; forcing OpenAI")
            provider = Provider.OPENAI
            model = get_model_config(Provider.OPENAI, task.job_type).model
            reason = f"{reason} | Claude not allowed for attachments → OpenAI fallback"
        return provider, model, reason

    return provider, model, reason


# =============================================================================
# CORE CALL FUNCTION (Async)
# =============================================================================


async def call_llm_async(task: LLMTask) -> LLMResult:
    """Primary async LLM call entry point."""
    session_id = getattr(task, "session_id", None)
    project_id = getattr(task, "project_id", 1) or 1

    job_type_specified = task.job_type is not None and task.job_type != JobType.UNKNOWN
    has_attachments = bool(task.attachments and len(task.attachments) > 0)

    # ==========================================================================
    # Extract user message text (string or multimodal list)
    # ==========================================================================
    user_messages = [m for m in task.messages if m.get("role") == "user"]
    original_message = user_messages[-1].get("content", "") if user_messages else ""

    message_text = ""
    if isinstance(original_message, str):
        message_text = original_message
    elif isinstance(original_message, list):
        # multimodal list (OpenAI style)
        for item in original_message:
            if isinstance(item, dict) and item.get("type") == "text":
                message_text += item.get("text", "")
    else:
        message_text = str(original_message or "")

    # ==========================================================================
    # Local action: ZOBIE MAP (prompt-triggered)
    # ==========================================================================
    try:
        local_result = await _maybe_handle_zobie_map(task, message_text)
        if local_result is not None:
            return local_result
    except Exception as exc:
        _debug_log(f"Local action handler failed: {exc}")

    # ==========================================================================
    # v0.15.1: Frontier override (OVERRIDE ... line)
    # ==========================================================================
    override_result = detect_frontier_override(original_message)
    cleaned_message = None
    if override_result:
        cleaned_message = override_result.cleaned_message

        print(
            f"[router] OVERRIDE detected → provider={override_result.provider_id} model={override_result.model_id} "
            f"(reason: {override_result.reason})"
        )

        # If override is requested, we still classify, but we will enforce the override routing.
        # This preserves job_type detection and downstream pipelines (e.g., high-stakes).
        task = task.model_copy(deep=True)
        if user_messages:
            user_messages[-1]["content"] = cleaned_message
            task.messages = task.messages

    # ==========================================================================
    # Step 1: Classify and route
    # ==========================================================================
    requested_type = task.job_type if job_type_specified else None
    provider, model, job_type, reason = classify_and_route(task, message_text, requested_type=requested_type)

    # Apply OVERRIDE routing if present
    if override_result:
        provider = override_result.provider_id
        model = override_result.model_id
        reason = f"{reason} | OVERRIDE: {override_result.reason}"

    # Normalize job type string for high-stakes pipeline
    job_type_str = normalize_job_type_for_high_stakes(str(job_type.value if hasattr(job_type, "value") else job_type), reason=reason)

    decision = RoutingDecision(
        job_type=job_type,
        provider=provider,
        model=model,
        reason=reason,
        confidence=0.85,
    )

    # ==========================================================================
    # Step 2: Build Phase 4 envelope + file map injection
    # ==========================================================================
    file_map = None
    if FILE_CLASSIFIER_AVAILABLE and has_attachments:
        try:
            file_map = build_file_map(task.attachments)
        except Exception as exc:
            _debug_log(f"build_file_map failed: {exc}")

    # Inject file map into messages (stable [FILE_X] naming)
    if file_map:
        task = inject_file_map_into_messages(task, file_map)

    envelope = synthesize_envelope_from_task(
        task=task,
        session_id=session_id,
        project_id=project_id,
        file_map=file_map,
        cleaned_message=cleaned_message,
    )

    # ==========================================================================
    # Step 3: Pipelines (Video+Code, High-stakes critique)
    # ==========================================================================
    if job_type == JobType.VIDEO_CODE_DEBUG:
        return await run_video_code_debug_pipeline(task=task, envelope=envelope, file_map=file_map)

    # High-stakes critique pipeline: only when Opus + high stakes type
    if is_high_stakes_job(job_type_str) and is_opus_model(model):
        result = await run_high_stakes_with_critique(
            task=task,
            provider_id=provider,
            model_id=model,
            envelope=envelope,
            job_type_str=job_type_str,
            file_map=file_map,
        )
        result.job_type = job_type
        result.routing_decision = {
            "job_type": job_type_str,
            "provider": provider,
            "model": result.model,
            "reason": "High-stakes pipeline result",
        }
        return result

    # ==========================================================================
    # Step 4: Default call (single pass)
    # ==========================================================================
    # Attachment safety (Claude restrictions)
    provider_id, model_id, reason = _check_attachment_safety(
        task=task,
        decision=decision,
        has_attachments=has_attachments,
        job_type_specified=job_type_specified,
    )

    result = await registry_llm_call(
        provider_id=provider_id,
        model_id=model_id,
        messages=envelope.messages,
        job_envelope=envelope,
    )

    # Build LLMResult
    llm_result = LLMResult(
        content=result.content,
        provider=provider_id,
        model=model_id,
        finish_reason="stop",
        error_message=None,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        total_tokens=result.usage.total_tokens,
        cost_usd=result.usage.cost_estimate,
        raw_response=result.raw_response,
        job_type=job_type,
        routing_decision={
            "job_type": job_type_str,
            "provider": provider_id,
            "model": model_id,
            "reason": reason,
        },
    )

    return llm_result


# =============================================================================
# Convenience async wrappers
# =============================================================================


async def quick_chat_async(message: str, **kwargs) -> str:
    task = LLMTask(messages=[{"role": "user", "content": message}], job_type=JobType.CHAT_LIGHT, **kwargs)
    result = await call_llm_async(task)
    return result.content


async def request_code_async(message: str, **kwargs) -> str:
    task = LLMTask(messages=[{"role": "user", "content": message}], job_type=JobType.CODE_MEDIUM, **kwargs)
    result = await call_llm_async(task)
    return result.content


async def review_work_async(message: str, **kwargs) -> str:
    task = LLMTask(messages=[{"role": "user", "content": message}], job_type=JobType.ORCHESTRATOR, **kwargs)
    result = await call_llm_async(task)
    return result.content


# =============================================================================
# Sync wrappers
# =============================================================================


def call_llm(task: LLMTask) -> LLMResult:
    return asyncio.run(call_llm_async(task))


def quick_chat(message: str, **kwargs) -> str:
    return asyncio.run(quick_chat_async(message, **kwargs))


def request_code(message: str, **kwargs) -> str:
    return asyncio.run(request_code_async(message, **kwargs))


def review_work(message: str, **kwargs) -> str:
    return asyncio.run(review_work_async(message, **kwargs))


# =============================================================================
# Compatibility helpers
# =============================================================================


def analyze_with_vision(*args, **kwargs):
    raise NotImplementedError("Vision analysis moved to dedicated modules.")


def web_search_query(*args, **kwargs):
    raise NotImplementedError("Web search moved to dedicated modules.")


def list_job_types() -> List[str]:
    return [jt.value for jt in JobType]


def get_routing_info() -> Dict[str, Any]:
    """Expose current routing configuration and environment flags."""
    return {
        "version": "0.15.1",
        "debug": ROUTER_DEBUG,
        "audit": {"available": AUDIT_AVAILABLE, "enabled": AUDIT_ENABLED},
        "policy": {"enabled": True},
        "frontier": {
            "override_enabled": True,
            "frontier_models": {
                "gemini": GEMINI_FRONTIER_MODEL_ID,
                "anthropic": ANTHROPIC_FRONTIER_MODEL_ID,
                "openai": OPENAI_FRONTIER_MODEL_ID,
            },
        },
        "high_stakes": {
            "enabled": True,
            "min_length_chars": MIN_CRITIQUE_CHARS,
            "critic_model": GEMINI_CRITIC_MODEL,
            "high_stakes_types": list(HIGH_STAKES_JOB_TYPES),
        },
        "spec_modules": {
            "file_classifier": FILE_CLASSIFIER_AVAILABLE,
            "relationship_detector": RELATIONSHIP_DETECTOR_AVAILABLE,
            "token_budgeting": TOKEN_BUDGETING_AVAILABLE,
            "fallbacks": FALLBACKS_AVAILABLE,
        },
    }


def is_policy_routing_enabled() -> bool:
    return True


def enable_policy_routing() -> None:
    return None


def disable_policy_routing() -> None:
    return None


def reload_routing_policy() -> None:
    return None


__all__ = [
    # Async API
    "call_llm_async",
    "quick_chat_async",
    "request_code_async",
    "review_work_async",
    # Sync wrappers
    "call_llm",
    "quick_chat",
    "request_code",
    "review_work",
    # Classification
    "classify_and_route",
    "normalize_job_type_for_high_stakes",
    # High-stakes pipeline
    "run_high_stakes_with_critique",
    "is_high_stakes_job",
    "is_opus_model",
    "HIGH_STAKES_JOB_TYPES",
    "get_environment_context",
    # v0.15.0: File map injection
    "inject_file_map_into_messages",
    # v0.15.1: Frontier override
    "detect_frontier_override",
    "GEMINI_FRONTIER_MODEL_ID",
    "ANTHROPIC_FRONTIER_MODEL_ID",
    "OPENAI_FRONTIER_MODEL_ID",
    # Compatibility
    "analyze_with_vision",
    "web_search_query",
    "list_job_types",
    "get_routing_info",
    "is_policy_routing_enabled",
    "enable_policy_routing",
    "disable_policy_routing",
    "reload_routing_policy",
    "synthesize_envelope_from_task",
]
