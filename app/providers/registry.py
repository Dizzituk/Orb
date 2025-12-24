# FILE: app/providers/registry.py
"""
Phase 4 Provider Registry

- Single async entrypoint: llm_call(...)
- Provider-side browsing is forbidden. Any enable_web_search flag is ignored.
- Optional local tool loop (enable_tools=True):
    Model requests tools -> Orb executes locally via app.tools.registry.execute_tool_async
    -> results returned to model.

Supported (if keys + SDKs installed):
- OpenAI (AsyncOpenAI)
- Anthropic (AsyncAnthropic)
- Google Gemini (google.generativeai)

NOTE (OpenAI token param drift):
- Some newer OpenAI chat-capable models (e.g. gpt-5.*) reject `max_tokens`
  and require `max_completion_tokens` instead.
- This module routes token limits accordingly, with a retry fallback.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


class LlmCallStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    INVALID_REQUEST = "invalid_request"
    TOOL_ERROR = "tool_error"
    TOOL_LOOP_EXCEEDED = "tool_loop_exceeded"


@dataclass
class LlmUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_estimate: float = 0.0


@dataclass
class LlmCallResult:
    status: LlmCallStatus
    provider_id: str
    model_id: str
    content: str = ""
    usage: LlmUsage = field(default_factory=LlmUsage)
    raw_response: Optional[dict] = None
    error_message: Optional[str] = None

    def is_success(self) -> bool:
        return self.status == LlmCallStatus.SUCCESS


@dataclass
class ProviderConfig:
    provider_id: str
    display_name: str
    env_key_name: str


PROVIDERS: Dict[str, ProviderConfig] = {
    "openai": ProviderConfig("openai", "OpenAI", "OPENAI_API_KEY"),
    "anthropic": ProviderConfig("anthropic", "Anthropic", "ANTHROPIC_API_KEY"),
    "google": ProviderConfig("google", "Google (Gemini)", "GOOGLE_API_KEY"),
    "gemini": ProviderConfig("gemini", "Google (Gemini)", "GOOGLE_API_KEY"),
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_json(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


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
        if is_provider_available(pid):
            return pid
    return None


def _build_openai_tools(tool_defs: List[dict]) -> List[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": (t.get("description", "") or "")[:800],
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tool_defs
    ]


def _build_anthropic_tools(tool_defs: List[dict]) -> List[dict]:
    return [
        {
            "name": t["name"],
            "description": (t.get("description", "") or "")[:800],
            "input_schema": t.get("input_schema", {"type": "object", "properties": {}}),
        }
        for t in tool_defs
    ]


def _openai_token_param_name(model_id: str) -> str:
    """
    OpenAI token-limit parameter name differs for some newer models.
    - Legacy: max_tokens
    - Newer chat models (incl gpt-5.*): max_completion_tokens
    """
    m = (model_id or "").strip().lower()
    if m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3"):
        return "max_completion_tokens"
    return "max_tokens"


async def _execute_tool_by_name(name: str, args: dict, context: Optional[dict]) -> dict:
    from app.tools.registry import execute_tool_async
    resp = await execute_tool_async(name, "v1", args, context=context)
    return resp.to_dict()


class ProviderRegistry:
    def __init__(self):
        self._tool_max_iters = int(os.getenv("TOOL_LOOP_MAX_ITERS", "6"))

    def is_provider_available(self, provider_id: str) -> bool:
        cfg = PROVIDERS.get(provider_id)
        if not cfg:
            return False
        key = os.getenv(cfg.env_key_name, "").strip()
        if not key:
            return False

        try:
            if provider_id == "openai":
                from openai import AsyncOpenAI  # noqa: F401
            elif provider_id == "anthropic":
                import anthropic  # noqa: F401
            elif provider_id in ("google", "gemini"):
                import google.generativeai  # noqa: F401
        except Exception:
            return False

        return True

    async def llm_call(
        self,
        provider_id: Optional[str],
        model_id: str,
        messages: List[dict],
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        timeout_seconds: int = 60,
        enable_web_search: bool = False,  # forbidden (ignored)
        enable_tools: bool = False,
        tool_names: Optional[List[str]] = None,
        job_envelope: Optional[dict] = None,
    ) -> LlmCallResult:
        if enable_web_search:
            logger.warning("[registry] enable_web_search requested but forbidden; ignoring.")

        chosen = provider_id or _pick_default_provider()
        if not chosen:
            return LlmCallResult(
                status=LlmCallStatus.PROVIDER_UNAVAILABLE,
                provider_id=str(provider_id or "none"),
                model_id=model_id,
                error_message="No providers available (missing API keys and/or SDKs).",
            )

        if not self.is_provider_available(chosen):
            return LlmCallResult(
                status=LlmCallStatus.PROVIDER_UNAVAILABLE,
                provider_id=chosen,
                model_id=model_id,
                error_message=f"Provider unavailable: {chosen}",
            )

        context = {"job_envelope": job_envelope or {}, "provider_id": chosen, "model_id": model_id}

        tool_defs: List[dict] = []
        if enable_tools:
            from app.tools.registry import get_tool_registry
            reg = get_tool_registry()
            for td in reg.list_tools(status=None):
                if td.status.value != "enabled":
                    continue
                if tool_names and td.tool_name not in tool_names:
                    continue
                tool_defs.append(
                    {"name": td.tool_name, "description": td.description, "input_schema": td.input_schema}
                )

        try:
            if chosen == "openai":
                return await self._call_openai(
                    api_key=os.getenv(PROVIDERS["openai"].env_key_name),
                    model_id=model_id,
                    messages=messages,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                    tool_defs=tool_defs,
                    context=context,
                )
            if chosen == "anthropic":
                return await self._call_anthropic(
                    api_key=os.getenv(PROVIDERS["anthropic"].env_key_name),
                    model_id=model_id,
                    messages=messages,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                    tool_defs=tool_defs,
                    context=context,
                )
            if chosen in ("google", "gemini"):
                return await self._call_google(
                    api_key=os.getenv(PROVIDERS["google"].env_key_name),
                    model_id=model_id,
                    messages=messages,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                    tool_defs=tool_defs,
                    context=context,
                )
            return LlmCallResult(
                status=LlmCallStatus.INVALID_REQUEST,
                provider_id=chosen,
                model_id=model_id,
                error_message=f"Unknown provider: {chosen}",
            )
        except Exception as exc:
            logger.exception("[registry] llm_call failed: %s", exc)
            return LlmCallResult(
                status=LlmCallStatus.ERROR,
                provider_id=chosen,
                model_id=model_id,
                error_message=str(exc),
            )

    async def _call_openai(
        self,
        api_key: str,
        model_id: str,
        messages: List[dict],
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
        tool_defs: List[dict],
        context: Optional[dict],
    ) -> LlmCallResult:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, timeout=timeout_seconds)

        tool_iters = 0
        oai_messages = _normalize_messages_for_openai(messages, system_prompt)
        tools_param = _build_openai_tools(tool_defs) if tool_defs else None

        async def _create_chat_completion():
            base_kwargs = dict(
                model=model_id,
                messages=oai_messages,
                stream=False,
                tools=tools_param,
                tool_choice="auto" if tools_param else None,
            )

            # Tokens: map param name based on model family.
            token_param = _openai_token_param_name(model_id)
            base_kwargs[token_param] = int(max_tokens)

            # Temperature: only send if not default; some OpenAI models only accept default.
            try:
                t = float(temperature) if temperature is not None else None
            except Exception:
                t = None
            if t is not None and abs(t - 1.0) > 1e-9:
                base_kwargs["temperature"] = t

            # Robust retry: handle SDK param mismatch, server-side max_tokens mismatch, and
            # model families that reject non-default temperature.
            temp_fallback_used = False
            while True:
                try:
                    return await client.chat.completions.create(**base_kwargs)
                except TypeError as e:
                    msg = str(e)
                    # Older SDKs don't accept max_completion_tokens.
                    if "max_completion_tokens" in msg and "unexpected keyword argument" in msg:
                        base_kwargs.pop("max_completion_tokens", None)
                        base_kwargs["max_tokens"] = int(max_tokens)
                        continue
                    raise
                except Exception as e:
                    msg = str(e)

                    # Some models only support the default temperature (1).
                    if (not temp_fallback_used) and ("temperature" in msg) and (
                        "Unsupported value" in msg
                        or "does not support" in msg
                        or "unsupported_value" in msg
                        or '"param": "temperature"' in msg
                        or "'param': 'temperature'" in msg
                    ):
                        if "temperature" in base_kwargs:
                            logger.warning(
                                "[openai] temperature unsupported for model %s; retrying without temperature. err=%s",
                                model_id,
                                msg,
                            )
                            base_kwargs.pop("temperature", None)
                            temp_fallback_used = True
                            continue

                    # Server-side error: wrong tokens param name.
                    if "Unsupported parameter: 'max_tokens'" in msg and "max_completion_tokens" in msg:
                        base_kwargs.pop("max_tokens", None)
                        base_kwargs["max_completion_tokens"] = int(max_tokens)
                        continue

                    raise

        while True:
            tool_iters += 1

            resp = await _create_chat_completion()

            choice = resp.choices[0]
            msg = choice.message
            tool_calls = getattr(msg, "tool_calls", None)

            if tools_param and tool_calls:
                if tool_iters > self._tool_max_iters:
                    return LlmCallResult(
                        status=LlmCallStatus.TOOL_LOOP_EXCEEDED,
                        provider_id="openai",
                        model_id=model_id,
                        raw_response=_safe_json(resp.model_dump() if hasattr(resp, "model_dump") else {}),
                        error_message=f"Tool loop exceeded max iterations ({self._tool_max_iters}).",
                    )

                oai_messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                            }
                            for tc in tool_calls
                        ],
                    }
                )

                for tc in tool_calls:
                    tool_name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception:
                        args = {"_raw": tc.function.arguments}

                    tool_result = await _execute_tool_by_name(tool_name, args, context)
                    oai_messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(tool_result, ensure_ascii=False)}
                    )

                continue

            content = msg.content or ""
            usage = LlmUsage(
                prompt_tokens=getattr(resp.usage, "prompt_tokens", 0) if getattr(resp, "usage", None) else 0,
                completion_tokens=getattr(resp.usage, "completion_tokens", 0) if getattr(resp, "usage", None) else 0,
            )
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

            return LlmCallResult(
                status=LlmCallStatus.SUCCESS,
                provider_id="openai",
                model_id=model_id,
                content=content,
                usage=usage,
                raw_response=_safe_json(resp.model_dump() if hasattr(resp, "model_dump") else {}),
            )

    async def _call_anthropic(
        self,
        api_key: str,
        model_id: str,
        messages: List[dict],
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
        tool_defs: List[dict],
        context: Optional[dict],
    ) -> LlmCallResult:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout_seconds)

        tool_iters = 0
        final_system, user_assistant_messages = _normalize_messages_for_anthropic(messages, system_prompt)
        tools_param = _build_anthropic_tools(tool_defs) if tool_defs else None

        running_messages = list(user_assistant_messages)

        while True:
            tool_iters += 1
            resp = await client.messages.create(
                model=model_id,
                system=final_system if final_system else None,
                messages=running_messages,
                temperature=temperature,
                max_tokens=min(max_tokens, 8192),
                tools=tools_param,
            )

            content_blocks = resp.content or []
            tool_uses = [b for b in content_blocks if getattr(b, "type", None) == "tool_use"]

            if tools_param and tool_uses:
                if tool_iters > self._tool_max_iters:
                    return LlmCallResult(
                        status=LlmCallStatus.TOOL_LOOP_EXCEEDED,
                        provider_id="anthropic",
                        model_id=model_id,
                        error_message=f"Tool loop exceeded max iterations ({self._tool_max_iters}).",
                        raw_response=_safe_json(resp.model_dump() if hasattr(resp, "model_dump") else {}),
                    )

                running_messages.append(
                    {
                        "role": "assistant",
                        "content": [b.model_dump() if hasattr(b, "model_dump") else _safe_json(b) for b in content_blocks],
                    }
                )

                tool_results: List[dict] = []
                for b in tool_uses:
                    tool_out = await _execute_tool_by_name(b.name, b.input or {}, context)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": b.id,
                            "content": json.dumps(tool_out, ensure_ascii=False),
                        }
                    )

                running_messages.append({"role": "user", "content": tool_results})
                continue

            text_parts: List[str] = []
            for b in content_blocks:
                if getattr(b, "type", None) == "text":
                    text_parts.append(getattr(b, "text", ""))

            content = "\n".join([t for t in text_parts if t]).strip()

            usage = LlmUsage(
                prompt_tokens=getattr(resp.usage, "input_tokens", 0) if getattr(resp, "usage", None) else 0,
                completion_tokens=getattr(resp.usage, "output_tokens", 0) if getattr(resp, "usage", None) else 0,
            )
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

            return LlmCallResult(
                status=LlmCallStatus.SUCCESS,
                provider_id="anthropic",
                model_id=model_id,
                content=content,
                usage=usage,
                raw_response=_safe_json(resp.model_dump() if hasattr(resp, "model_dump") else {}),
            )

    async def _call_google(
        self,
        api_key: str,
        model_id: str,
        messages: List[dict],
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
        tool_defs: List[dict],
        context: Optional[dict],
    ) -> LlmCallResult:
        import google.generativeai as genai

        genai.configure(api_key=api_key)

        parts: List[str] = []
        if system_prompt:
            parts.append(f"[SYSTEM]\n{system_prompt}")
        for m in messages:
            role = m.get("role", "user")
            parts.append(f"[{role.upper()}]\n{str(m.get('content', ''))}")
        prompt = "\n\n".join(parts).strip()

        model = genai.GenerativeModel(model_id)

        resp = model.generate_content(prompt)

        # Best-effort: if SDK returns a function_call, run ONE tool and ask again.
        try:
            fc = None
            if getattr(resp, "candidates", None):
                cand = resp.candidates[0]
                content = getattr(cand, "content", None)
                if content and getattr(content, "parts", None):
                    for p in content.parts:
                        if getattr(p, "function_call", None):
                            fc = p.function_call
                            break

            if tool_defs and fc:
                name = getattr(fc, "name", "")
                args = dict(getattr(fc, "args", {}) or {})
                tool_out = await _execute_tool_by_name(name, args, context)
                prompt2 = prompt + "\n\n[TOOL_RESULT]\n" + json.dumps(tool_out, ensure_ascii=False)
                resp2 = model.generate_content(prompt2)
                return LlmCallResult(
                    status=LlmCallStatus.SUCCESS,
                    provider_id="google",
                    model_id=model_id,
                    content=getattr(resp2, "text", "") or "",
                    raw_response=_safe_json(getattr(resp2, "to_dict", lambda: {})()),
                )
        except Exception as exc:
            logger.warning("[registry] Gemini tool loop failed (best-effort): %s", exc)

        return LlmCallResult(
            status=LlmCallStatus.SUCCESS,
            provider_id="google",
            model_id=model_id,
            content=getattr(resp, "text", "") or "",
            raw_response=_safe_json(getattr(resp, "to_dict", lambda: {})()),
        )


_registry: Optional[ProviderRegistry] = None


def get_provider_registry() -> ProviderRegistry:
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
    return _registry


async def llm_call(
    provider_id: Optional[str],
    model_id: str,
    messages: List[dict],
    system_prompt: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    timeout_seconds: int = 60,
    enable_web_search: bool = False,
    enable_tools: bool = False,
    tool_names: Optional[List[str]] = None,
    job_envelope: Optional[dict] = None,
) -> LlmCallResult:
    return await get_provider_registry().llm_call(
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
    )


def is_provider_available(provider_id: str) -> bool:
    return get_provider_registry().is_provider_available(provider_id)


__all__ = [
    "LlmCallStatus",
    "LlmUsage",
    "LlmCallResult",
    "ProviderRegistry",
    "get_provider_registry",
    "llm_call",
    "is_provider_available",
]
