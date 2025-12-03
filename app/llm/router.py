# FILE: app/llm/router.py
"""
LLM Router (REFACTORED FOR PHASE 4)

This router uses the provider registry as the single LLM call path.

For existing /chat and /stream/chat endpoints:
- Synthesizes a minimal JobEnvelope
- Routes through provider registry
- Maintains backward-compatible behavior

For new /jobs endpoints:
- Uses real JobEnvelope from job engine
- Same provider registry path

NOTE: Streaming is NOT yet unified. Streaming endpoints continue using
the legacy clients/path. Streaming unification will be done in a
separate branch.

PHASE 4 FIXES:
- Fixed JobBudget field names (max_tokens, max_cost_estimate, max_wall_time_seconds)
- Fixed modalities → modalities_in
- Fixed needs_tools type (bool → list[str])
- Use RoutingOptions model instead of RoutingConfig class

This file exposes:

Async-first API (for FastAPI and internal usage):
- call_llm_async()
- quick_chat_async()
- request_code_async()
- review_work_async()

Sync wrappers (for CLI/testing only, NOT for FastAPI):
- call_llm()
- quick_chat()
- request_code()
- review_work()

Helpers:
- synthesize_envelope_from_task()

Compatibility helpers (for older Orb code):
- analyze_with_vision()
- web_search_query()
- list_job_types()
- get_routing_info()
- is_policy_routing_enabled()
- enable_policy_routing()
- disable_policy_routing()
- reload_routing_policy()
"""

import os
import logging
from typing import Optional, List, Dict, Any
from uuid import uuid4

from app.llm.schemas import (
    LLMTask,
    LLMResult,
    JobType as LegacyJobType,  # Old enum for backward compat
    Provider,
    RoutingConfig,
    RoutingOptions,  # NEW: for per-task routing options
)

# Phase 4 imports
from app.jobs.schemas import (
    JobEnvelope,
    JobType,
    Importance,
    DataSensitivity,
    Modality,
    JobBudget,
    OutputContract,
    validate_job_envelope,
    ValidationError,
)
from app.providers.registry import llm_call as registry_llm_call

logger = logging.getLogger(__name__)

# =============================================================================
# POLICY ROUTING CONFIG
# =============================================================================

_USE_POLICY_ROUTING = os.getenv("ORB_USE_POLICY_ROUTING", "false").lower() == "true"

try:
    from app.llm.policy import (
        load_routing_policy,
        make_routing_decision,
        resolve_job_type,
        Provider as PolicyProvider,
    )
    _policy_available = True
except ImportError:
    logger.warning("[router] Policy module not available")
    _policy_available = False


# =============================================================================
# JOB TYPE MAPPING (Legacy → Phase 4)
# =============================================================================

_LEGACY_TO_PHASE4_JOB_TYPE: Dict[str, JobType] = {
    # Simple chat / explanation / summary
    LegacyJobType.CASUAL_CHAT.value: JobType.CHAT_SIMPLE,
    LegacyJobType.QUICK_QUESTION.value: JobType.CHAT_RESEARCH,
    LegacyJobType.PROMPT_SHAPING.value: JobType.CHAT_SIMPLE,
    LegacyJobType.SUMMARY.value: JobType.CHAT_SIMPLE,
    LegacyJobType.EXPLANATION.value: JobType.CHAT_SIMPLE,

    # Code
    LegacyJobType.SIMPLE_CODE_CHANGE.value: JobType.CODE_SMALL,
    LegacyJobType.SMALL_BUGFIX.value: JobType.CODE_SMALL,
    LegacyJobType.COMPLEX_CODE_CHANGE.value: JobType.CODE_REPO,
    LegacyJobType.CODEGEN_FULL_FILE.value: JobType.CODE_REPO,

    # Architecture / critique
    LegacyJobType.ARCHITECTURE_DESIGN.value: JobType.APP_ARCHITECTURE,
    LegacyJobType.CODE_REVIEW.value: JobType.CRITIQUE_REVIEW,
    LegacyJobType.SPEC_REVIEW.value: JobType.CRITIQUE_REVIEW,
    LegacyJobType.HIGH_STAKES_INFRA.value: JobType.APP_ARCHITECTURE,
}


def _map_legacy_job_type(legacy_job_type: LegacyJobType) -> JobType:
    """
    Map legacy LLMTask job_type to Phase 4 JobType.
    """
    try:
        return _LEGACY_TO_PHASE4_JOB_TYPE[legacy_job_type.value]
    except KeyError:
        # Default to CHAT_SIMPLE for unknown legacy types
        logger.warning(
            "[router] Unknown legacy job_type '%s', defaulting to CHAT_SIMPLE",
            legacy_job_type.value,
        )
        return JobType.CHAT_SIMPLE


def _default_importance_for_job_type(job_type: JobType) -> Importance:
    """
    Determine default Importance based on JobType.
    """
    if job_type in {
        JobType.APP_ARCHITECTURE,
        JobType.CODE_REPO,
        JobType.CODE_SMALL,
    }:
        return Importance.HIGH
    if job_type in {
        JobType.DEEP_RESEARCH_TASK,
        JobType.MODEL_CAPABILITY_SYNC,
    }:
        return Importance.MEDIUM
    return Importance.LOW


def _default_modalities_for_job_type(job_type: JobType) -> List[Modality]:
    """
    Determine default modalities based on JobType.
    """
    if job_type in {
        JobType.VISION_SIMPLE,
        JobType.VISION_COMPLEX,
    }:
        return [Modality.TEXT, Modality.IMAGE]
    if job_type in {
        JobType.AUDIO_TRANSCRIPTION,
        JobType.AUDIO_MEETING,
    }:
        return [Modality.TEXT, Modality.AUDIO]
    if job_type in {
        JobType.VIDEO_SIMPLE,
        JobType.VIDEO_ADVANCED,
    }:
        return [Modality.TEXT, Modality.VIDEO]

    # Default: text-only
    return [Modality.TEXT]


# =============================================================================
# ENVELOPE SYNTHESIS (Legacy LLMTask → JobEnvelope)
# =============================================================================

def synthesize_envelope_from_task(
    task: LLMTask,
    session_id: Optional[str] = None,
    project_id: int = 1,
) -> JobEnvelope:
    """
    Synthesize a JobEnvelope from legacy LLMTask.

    This allows existing /chat and /stream/chat endpoints to use
    the unified provider registry path without breaking changes.
    """
    legacy_job_type = task.job_type
    phase4_job_type = _map_legacy_job_type(legacy_job_type)

    importance = _default_importance_for_job_type(phase4_job_type)
    modalities = _default_modalities_for_job_type(phase4_job_type)

    # Extract routing options (handle None and use defaults)
    routing = task.routing
    max_tokens = routing.max_tokens if routing else 8000
    max_cost = routing.max_cost_usd if routing else 1.0
    timeout = routing.timeout_seconds if routing else 60

    # FIXED: Use correct JobBudget field names
    budget = JobBudget(
        max_tokens=max_tokens,
        max_cost_estimate=float(max_cost),
        max_wall_time_seconds=timeout,
    )

    envelope = JobEnvelope(
        job_id=str(uuid4()),
        session_id=session_id or f"legacy-{uuid4()}",
        project_id=project_id,
        job_type=phase4_job_type,
        importance=importance,
        data_sensitivity=DataSensitivity.INTERNAL,
        modalities_in=modalities,  # FIXED: was 'modalities'
        budget=budget,
        output_contract=OutputContract.TEXT_RESPONSE,
        messages=task.messages,
        metadata={
            "legacy_provider_hint": task.provider.value if task.provider else None,
            "legacy_routing": routing.model_dump() if routing else None,
            "legacy_context": task.project_context,
        },
        allow_multi_model_review=False,
        needs_tools=[],  # FIXED: was False (bool), must be list[str]
    )

    # Validate envelope (Phase 4 rules)
    try:
        validate_job_envelope(envelope)
    except ValidationError as ve:
        logger.warning(
            "[router] Synthesized JobEnvelope failed validation: %s", ve
        )
        raise

    return envelope


# =============================================================================
# CORE CALL FUNCTION (Async)
# =============================================================================

async def call_llm_async(task: LLMTask) -> LLMResult:
    """
    Primary async LLM call entry point.

    This replaces direct calls to app.llm.clients.* and routes through
    the unified provider registry using a synthesized JobEnvelope.
    """
    # Determine session_id if present in task (optional)
    session_id = getattr(task, "session_id", None)
    project_id = getattr(task, "project_id", 1) or 1

    # Synthesize envelope
    try:
        envelope = synthesize_envelope_from_task(
            task=task,
            session_id=session_id,
            project_id=project_id,
        )
    except ValidationError as ve:
        # Convert to clean error for callers
        return LLMResult(
            content="",
            provider=task.provider.value if task.provider else Provider.OPENAI.value,
            model=task.model or "",
            finish_reason="validation_error",
            error_message=f"JobEnvelope validation failed: {ve}",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            raw_response=None,
        )
    except Exception as exc:
        # Catch Pydantic ValidationError and other exceptions
        logger.warning("[router] Envelope synthesis failed: %s", exc)
        return LLMResult(
            content="",
            provider=task.provider.value if task.provider else Provider.OPENAI.value,
            model=task.model or "",
            finish_reason="validation_error",
            error_message=f"JobEnvelope synthesis failed: {exc}",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            raw_response=None,
        )

    # Determine provider/model
    provider_id = task.provider.value if task.provider else Provider.OPENAI.value
    model_id = task.model or ""

    try:
        result = await registry_llm_call(
            provider_id=provider_id,
            model_id=model_id,
            messages=envelope.messages,
            job_envelope=envelope,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[router] llm_call failed: %s", exc)
        return LLMResult(
            content="",
            provider=provider_id,
            model=model_id,
            finish_reason="error",
            error_message=str(exc),
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            raw_response=None,
        )

    # Map provider result to LLMResult
    if not result.is_success():
        return LLMResult(
            content=result.error_message or "",
            provider=provider_id,
            model=model_id,
            finish_reason="error",
            error_message=result.error_message,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
            cost_usd=result.usage.cost_estimate,
            raw_response=result.raw_response,
        )

    return LLMResult(
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
    )


# =============================================================================
# HIGH-LEVEL HELPERS (Async)
# =============================================================================

async def quick_chat_async(
    message: str,
    context: Optional[str] = None,
) -> LLMResult:
    """
    Simple async chat helper.
    """
    messages: List[Dict[str, str]] = []
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": message})

    task = LLMTask(
        job_type=LegacyJobType.CASUAL_CHAT,
        provider=Provider.OPENAI,
        model=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4.1-mini"),
        messages=messages,
        routing=RoutingOptions(),  # FIXED: was RoutingConfig() which has no instance fields
        project_context=context,
    )
    return await call_llm_async(task)


async def request_code_async(
    message: str,
    context: Optional[str] = None,
    high_stakes: bool = False,
) -> LLMResult:
    """
    Async helper for code-related tasks.
    """
    messages: List[Dict[str, str]] = []
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": message})

    legacy_job_type = (
        LegacyJobType.HIGH_STAKES_INFRA
        if high_stakes
        else LegacyJobType.CODEGEN_FULL_FILE
    )

    task = LLMTask(
        job_type=legacy_job_type,
        provider=Provider.ANTHROPIC,
        model=os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-5-sonnet-latest"),
        messages=messages,
        routing=RoutingOptions(),  # FIXED: was RoutingConfig()
        project_context=context,
    )
    return await call_llm_async(task)


async def review_work_async(
    message: str,
    context: Optional[str] = None,
) -> LLMResult:
    """
    Async helper for review/critique work (code/spec/etc).
    """
    messages: List[Dict[str, str]] = []
    if context:
        messages.append({"role": "system", "content": context})
    messages.append({"role": "user", "content": message})

    task = LLMTask(
        job_type=LegacyJobType.CODE_REVIEW,
        provider=Provider.ANTHROPIC,
        model=os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-5-sonnet-latest"),
        messages=messages,
        routing=RoutingOptions(),  # FIXED: was RoutingConfig()
        project_context=context,
    )
    return await call_llm_async(task)


# =============================================================================
# SYNC WRAPPERS (CLI / TESTING ONLY)
# =============================================================================

def call_llm(task: LLMTask) -> LLMResult:
    """
    Sync wrapper for call_llm_async.

    WARNING: Uses asyncio.run; intended ONLY for CLI/testing.
    """
    import asyncio
    return asyncio.run(call_llm_async(task))


def quick_chat(
    message: str,
    context: Optional[str] = None,
) -> LLMResult:
    """
    Sync wrapper for quick_chat_async.

    WARNING: Uses asyncio.run; intended ONLY for CLI/testing.
    """
    import asyncio
    return asyncio.run(quick_chat_async(message=message, context=context))


def request_code(
    message: str,
    context: Optional[str] = None,
    high_stakes: bool = False,
) -> LLMResult:
    """
    Sync wrapper for request_code_async.

    WARNING: Uses asyncio.run; intended ONLY for CLI/testing.
    """
    import asyncio
    return asyncio.run(
        request_code_async(
            message=message,
            context=context,
            high_stakes=high_stakes,
        )
    )


def review_work(
    message: str,
    context: Optional[str] = None,
) -> LLMResult:
    """
    Sync wrapper for review_work_async.

    WARNING: Uses asyncio.run; intended ONLY for CLI/testing.
    """
    import asyncio
    return asyncio.run(review_work_async(message=message, context=context))


# =============================================================================
# COMPATIBILITY HELPERS
# =============================================================================

def analyze_with_vision(
    prompt: str,
    image_description: Optional[str] = None,
    context: Optional[str] = None,
) -> LLMResult:
    """
    Compatibility wrapper for legacy vision calls.

    NOTE: This does **not** currently send real image data to a vision model.
    It simply augments the prompt with an optional image description and
    optional extra context, then routes through the normal quick_chat path.
    """
    parts: List[str] = [prompt]
    if image_description:
        parts.append("\n\n[Image description]\n" + image_description)
    if context:
        parts.append("\n\n[Context]\n" + context)
    combined_prompt = "".join(parts)
    return quick_chat(message=combined_prompt, context=None)


def web_search_query(
    query: str,
    context: Optional[str] = None,
) -> LLMResult:
    """
    Compatibility wrapper for legacy web search helper.

    NOTE: This does **not** currently perform real multi-source web search.
    It simply formats the query in a "web search style" prompt and routes
    through quick_chat. Proper tool-based web search will come later.
    """
    parts: List[str] = [f"[WEB SEARCH STYLE QUERY]\n{query}"]
    if context:
        parts.append("\n\n[ADDITIONAL CONTEXT]\n" + context)
    combined_prompt = "".join(parts)
    return quick_chat(message=combined_prompt, context=None)


def list_job_types() -> List[str]:
    """
    Compatibility helper returning legacy job type values as strings.
    """
    return [jt.value for jt in LegacyJobType]


def get_routing_info() -> Dict[str, Any]:
    """
    Compatibility helper for legacy routing-info calls.

    Returns a simple dict with basic routing information so existing
    UIs and diagnostics continue to work.
    """
    return {
        "policy_routing_enabled": bool(_USE_POLICY_ROUTING and _policy_available),
        "policy_module_available": bool(_policy_available),
        "available_job_types": [jt.value for jt in LegacyJobType],
        "default_provider": Provider.OPENAI.value,
        "default_anthropic_model": os.getenv(
            "ANTHROPIC_DEFAULT_MODEL", "claude-3-5-sonnet-latest"
        ),
        "default_openai_model": os.getenv(
            "OPENAI_DEFAULT_MODEL", "gpt-4.1-mini"
        ),
    }


def is_policy_routing_enabled() -> bool:
    """
    Compatibility helper reporting if policy routing is active.
    """
    return bool(_USE_POLICY_ROUTING and _policy_available)


def enable_policy_routing() -> None:
    """
    Compatibility helper to enable policy routing at runtime.

    NOTE: This only toggles the in-process flag. You still need a
    valid policy module for it to actually be used.
    """
    global _USE_POLICY_ROUTING  # noqa: PLW0603
    if not _policy_available:
        logger.warning(
            "[router] enable_policy_routing called but policy module is not available"
        )
        _USE_POLICY_ROUTING = False
        return

    _USE_POLICY_ROUTING = True
    logger.info("[router] Policy routing ENABLED via enable_policy_routing()")


def disable_policy_routing() -> None:
    """
    Compatibility helper to disable policy routing at runtime.
    """
    global _USE_POLICY_ROUTING  # noqa: PLW0603
    _USE_POLICY_ROUTING = False
    logger.info("[router] Policy routing DISABLED via disable_policy_routing()")


def reload_routing_policy() -> None:
    """
    Compatibility helper to reload the routing policy if available.
    """
    if not _policy_available:
        logger.warning(
            "[router] reload_routing_policy called but policy module is not available"
        )
        return

    try:
        load_routing_policy(force=True)  # type: ignore[arg-type]
        logger.info("[router] Routing policy reloaded via reload_routing_policy()")
    except TypeError:
        # If load_routing_policy doesn't accept force, call without args
        load_routing_policy()  # type: ignore[call-arg]
        logger.info("[router] Routing policy reloaded via reload_routing_policy()")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Primary async functions (use in FastAPI)
    "call_llm_async",
    "quick_chat_async",
    "request_code_async",
    "review_work_async",

    # Sync wrappers (CLI/testing only)
    "call_llm",
    "quick_chat",
    "request_code",
    "review_work",

    # Compatibility helpers
    "analyze_with_vision",
    "web_search_query",
    "list_job_types",
    "get_routing_info",
    "is_policy_routing_enabled",
    "enable_policy_routing",
    "disable_policy_routing",
    "reload_routing_policy",

    # Helpers
    "synthesize_envelope_from_task",
]