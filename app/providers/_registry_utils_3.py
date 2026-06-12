# Purpose: registry utils 3
# Called-by: app.investments.chat_router, app.providers._registry_utils_4, app.providers.registry, app.rag.answerer
# Depends-on: app.cost.cost_recorder, app.llm.prompt_tiers, app.providers.registry
# Last-renovated: 2026-06-11
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


@dataclass
class LlmUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_estimate: float = 0.0

def _normalize_messages_for_openai(messages: List[dict], system_prompt: Optional[str]) -> List[dict]:
    out: List[dict] = []
    if system_prompt:
        out.append({"role": "system", "content": system_prompt})
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role in ("system", "user", "assistant", "tool"):
            if role == "tool":
                out.append(
                    {"role": "tool", "tool_call_id": m.get("tool_call_id"), "content": str(content)}
                )
            else:
                out.append({"role": role, "content": str(content)})
        else:
            out.append({"role": "user", "content": str(content)})
    return out

def _normalize_messages_for_anthropic(messages: List[dict], system_prompt: Optional[str]) -> Tuple[str, List[dict]]:
    sys_parts: List[str] = []
    if system_prompt:
        sys_parts.append(system_prompt)

    user_assistant: List[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            sys_parts.append(str(m.get("content", "")))
        elif role in ("user", "assistant"):
            user_assistant.append({"role": role, "content": str(m.get("content", ""))})

    return ("\n\n".join([p for p in sys_parts if p]).strip(), user_assistant)

def _pick_default_provider() -> Optional[str]:
    for pid in ("openai", "anthropic", "google"):
        from .registry import is_provider_available
        if is_provider_available(pid):
            return pid
    return None

def _openai_token_param_name(model_id: str) -> str:
    """
    OpenAI token-limit parameter name differs for some newer models.
    - Legacy: max_tokens
    - Newer chat models (incl gpt-5.*): max_completion_tokens
    """
    m = (model_id or "").strip().lower()
    if m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return "max_completion_tokens"
    return "max_tokens"

def _supports_temperature(model_id: str) -> bool:
    """Check if model supports non-default temperature.
    
    GPT-5.x models and o-series only accept temperature=1.
    """
    m = (model_id or "").strip().lower()
    # Models known to reject non-default temperature
    # All GPT-5 variants (gpt-5-mini, gpt-5.2-chat, gpt-5.2-pro, etc.)
    if m.startswith("gpt-5"):
        return False
    if m.startswith("o1") or m.startswith("o3") or m.startswith("o4"):
        return False
    return True

def _is_chat_model(model_id: str) -> bool:
    """Check if model supports the chat completions endpoint.
    
    Pro models (gpt-5.2-pro, gpt-5-pro, o1-pro, o3-pro) require the
    completions endpoint, not chat completions.
    """
    m = (model_id or "").strip().lower()
    # Pro models are NOT chat-capable (except preview variants)
    if "-pro" in m and "-preview" not in m:
        # gpt-5.2-pro, gpt-5-pro, o1-pro, o3-pro → completions only
        return False
    return True

async def llm_call(
    provider_id: Optional[str],
    model_id: str,
    messages: List[dict],
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    timeout_seconds: int = 180,  # v2.1: Raised from 60
    enable_web_search: bool = False,
    enable_tools: bool = False,
    tool_names: Optional[List[str]] = None,
    job_envelope: Optional[dict] = None,
    reasoning: Optional[dict] = None,  # GPT-5.x: {"effort": "none"|"low"|"medium"|"high"|"xhigh"}
    stage: Optional[str] = None,  # v2.2: Pipeline stage name for cost tracking
    **kwargs: Any,  # absorb future constraints, e.g. data_sensitivity_constraint
):
    from .registry import LlmCallResult, LlmCallStatus, get_provider_registry

    # v2.2: Auto-inject tiered system prompt when none provided
    if system_prompt is None and stage:
        try:
            from app.llm.prompt_tiers import get_system_prompt as _get_tiered_prompt
            system_prompt = _get_tiered_prompt(stage=stage)
        except Exception:
            pass

    # v2.2: Pre-call budget check
    try:
        from app.cost.cost_recorder import pre_call_budget_check
        allowed, reason = pre_call_budget_check(stage=stage)
        if not allowed:
            return LlmCallResult(
                status=LlmCallStatus.ERROR,
                provider_id=provider_id or "unknown",
                model_id=model_id,
                content="",
                error_message=f"Budget exceeded: {reason}",
            )
    except Exception:
        pass  # Budget check failure must never block calls

    result = await get_provider_registry().llm_call(
        provider_id=provider_id,
        model_id=model_id,
        messages=messages,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        enable_web_search=enable_web_search,
        enable_tools=enable_tools,
        tool_names=tool_names,
        job_envelope=job_envelope,
        reasoning=reasoning,
        **kwargs,
    )

    # v2.2: Post-call cost recording
    try:
        from app.cost.cost_recorder import record_llm_cost
        if hasattr(result, 'usage') and result.usage:
            job_id = None
            if job_envelope:
                # JobEnvelope can be a Pydantic model or a dict
                if isinstance(job_envelope, dict):
                    job_id = job_envelope.get("job_id")
                else:
                    job_id = getattr(job_envelope, "job_id", None)
            record_llm_cost(
                provider=result.provider_id if hasattr(result, 'provider_id') else (provider_id or "unknown"),
                model=result.model_id if hasattr(result, 'model_id') else model_id,
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                stage=stage,
                job_id=job_id,
            )
    except Exception:
        pass  # Cost recording failure must never break the pipeline

    return result
