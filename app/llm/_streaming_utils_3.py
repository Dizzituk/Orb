from __future__ import annotations
import json
import logging
import os
from app.llm._streaming_utils_2 import _openai_text_nonstream, _should_retry_stream_error, get_default_provider
from typing import Any, AsyncGenerator, Dict, List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
HAS_ANTHROPIC = True
HAS_GEMINI = True


def _provider_key(provider: str) -> str:
    p = (provider or "").strip().lower()
    if p in ("gemini", "google"):
        return "google"
    return p

def _int_env(name: str) -> Optional[int]:
    v = os.getenv(name, "").strip()
    if not v:
        return None
    try:
        return int(v)
    except Exception:
        return None

def _openai_needs_max_completion_tokens(model: str) -> bool:
    m = (model or "").strip().lower()
    # Covers gpt-5.* and typical "o-series" reasoning models if you add them later.
    return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3")

async def stream_anthropic(
    messages: List[Dict],
    system_prompt: str = "",
    model: Optional[str] = None,
    enable_reasoning: bool = False,
    route: Optional[str] = None,
    max_tokens: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
) -> AsyncGenerator[Dict, None]:
    """Stream from Anthropic using async client.
    
    Args:
        max_tokens: Override max output tokens (default: ANTHROPIC_MAX_TOKENS env or 4096)
        timeout_seconds: Request timeout (currently logged but not enforced at HTTP level)
    """
    from .streaming import anthropic, enhance_system_prompt_with_reasoning, get_default_model, get_model_for_route
    print(f"[STREAM_ANTHROPIC] Called: model={model}, route={route}, max_tokens={max_tokens}, timeout={timeout_seconds}s")

    if not HAS_ANTHROPIC:
        yield {"type": "error", "message": "anthropic package not installed"}
        return

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        yield {"type": "error", "message": "ANTHROPIC_API_KEY not set"}
        return

    if model:
        use_model = model
    elif route:
        use_model = get_model_for_route("anthropic", route)
    else:
        use_model = get_default_model("anthropic")

    yield {"type": "metadata", "provider": "anthropic", "model": use_model}

    enhanced_prompt = enhance_system_prompt_with_reasoning(system_prompt, enable_reasoning)

    # v0.19: Apply timeout — default 300s for long architecture docs, env override available
    effective_timeout = timeout_seconds or _int_env("ANTHROPIC_TIMEOUT_SECONDS") or 300
    client = anthropic.AsyncAnthropic(
        api_key=api_key,
        timeout=effective_timeout,
    )

    prompt_tokens = None
    completion_tokens = None

    # Determine max tokens: explicit param > env var > default
    effective_max_tokens = max_tokens or _int_env("ANTHROPIC_MAX_TOKENS") or 4096
    print(f"[STREAM_ANTHROPIC] Effective max_tokens={effective_max_tokens}, timeout={effective_timeout}s")
    
    try:
        resp = await client.messages.create(
            model=use_model,
            max_tokens=effective_max_tokens,
            system=enhanced_prompt,
            messages=messages,
            stream=True,
        )

        async for event in resp:
            if event.type == "content_block_delta":
                delta_text = getattr(event.delta, "text", None)
                if delta_text:
                    yield {"type": "token", "text": delta_text}

            if event.type == "message_delta":
                usage = getattr(event, "usage", None)
                if usage:
                    prompt_tokens = getattr(usage, "input_tokens", None)
                    completion_tokens = getattr(usage, "output_tokens", None)

        if prompt_tokens is not None or completion_tokens is not None:
            total_tokens = (int(prompt_tokens or 0) + int(completion_tokens or 0))
            yield {
                "type": "done",
                "provider": "anthropic",
                "model": use_model,
                "usage": {
                    "prompt_tokens": int(prompt_tokens or 0),
                    "completion_tokens": int(completion_tokens or 0),
                    "total_tokens": int(total_tokens),
                },
            }
        else:
            yield {"type": "done", "provider": "anthropic", "model": use_model}

    except Exception as e:
        yield {"type": "error", "message": str(e)}

async def stream_gemini(
    messages: List[Dict],
    system_prompt: str = "",
    model: Optional[str] = None,
    enable_reasoning: bool = False,
    route: Optional[str] = None,
) -> AsyncGenerator[Dict, None]:
    """Stream from Gemini (Google Generative AI)."""
    from .streaming import enhance_system_prompt_with_reasoning, genai, get_default_model, get_model_for_route
    if not HAS_GEMINI:
        yield {"type": "error", "message": "google.generativeai package not installed"}
        return

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        yield {"type": "error", "message": "GOOGLE_API_KEY not set"}
        return

    if model:
        use_model = model
    elif route:
        use_model = get_model_for_route("google", route)
    else:
        use_model = get_default_model("google")

    genai.configure(api_key=api_key)

    enhanced_prompt = enhance_system_prompt_with_reasoning(system_prompt, enable_reasoning)

    yield {"type": "metadata", "provider": "gemini", "model": use_model}

    prompt_tokens = None
    completion_tokens = None
    total_tokens = None

    def _uget(obj, key: str):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    try:
        gemini_model = genai.GenerativeModel(
            model_name=use_model,
            system_instruction=enhanced_prompt,
        )

        history = []
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})

        chat = gemini_model.start_chat(history=history)

        last_msg = messages[-1]["content"] if messages else ""
        response = chat.send_message(last_msg, stream=True)

        last_chunk = None
        for chunk in response:
            last_chunk = chunk
            if getattr(chunk, "text", None):
                yield {"type": "token", "text": chunk.text}

        usage_md = _uget(response, "usage_metadata") or _uget(last_chunk, "usage_metadata")
        if usage_md:
            pt = _uget(usage_md, "prompt_token_count")
            ct = _uget(usage_md, "candidates_token_count")
            tt = _uget(usage_md, "total_token_count")
            if pt is not None:
                prompt_tokens = int(pt)
            if ct is not None:
                completion_tokens = int(ct)
            if tt is not None:
                total_tokens = int(tt)

        if prompt_tokens is not None or completion_tokens is not None or total_tokens is not None:
            if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
            usage = {
                "prompt_tokens": int(prompt_tokens or 0),
                "completion_tokens": int(completion_tokens or 0),
                "total_tokens": int(total_tokens or (int(prompt_tokens or 0) + int(completion_tokens or 0))),
            }
            yield {"type": "done", "provider": "gemini", "model": use_model, "usage": usage}
        else:
            yield {"type": "done", "provider": "gemini", "model": use_model}

    except Exception as e:
        yield {"type": "error", "message": str(e)}

async def stream_llm(
    messages: List[Dict],
    system_prompt: str = "",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    enable_reasoning: bool = False,
    route: Optional[str] = None,
    max_tokens: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
) -> AsyncGenerator[Dict, None]:
    """Stream from specified LLM provider.
    
    Args:
        max_tokens: Override max output tokens (provider-specific defaults otherwise)
        timeout_seconds: Request timeout hint (implementation varies by provider)
    """
    from .streaming import stream_openai
    print(f"[STREAM_LLM] Called: provider={provider}, model={model}, messages={len(messages)}, max_tokens={max_tokens}")

    if not provider:
        provider = get_default_provider()
        print(f"[STREAM_LLM] No provider specified, using default: {provider}")

    if not provider:
        print("[STREAM_LLM] ERROR: No LLM providers available!")
        yield {"type": "error", "message": "No LLM providers available"}
        return

    provider = provider.lower()
    print(f"[STREAM_LLM] Routing to provider: {provider}, model: {model}")

    if provider == "openai":
        async for event in stream_openai(messages, system_prompt, model, enable_reasoning, route):
            yield event
    elif provider == "anthropic":
        async for event in stream_anthropic(
            messages, system_prompt, model, enable_reasoning, route,
            max_tokens=max_tokens, timeout_seconds=timeout_seconds
        ):
            yield event
    elif provider in ("gemini", "google"):
        async for event in stream_gemini(messages, system_prompt, model, enable_reasoning, route):
            yield event
    else:
        print(f"[STREAM_LLM] ERROR: Unknown provider '{provider}'")
        yield {"type": "error", "message": f"Unknown provider '{provider}'"}

async def call_llm_text(
    provider: str,
    model: Optional[str],
    system_prompt: str,
    user_prompt: str,
    *,
    messages: Optional[List[Dict[str, str]]] = None,
    repo_snapshot: Optional[Dict[str, Any]] = None,
    constraints_hint: Optional[Any] = None,
    enable_reasoning: bool = False,
    route: Optional[str] = None,
    max_tokens: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
) -> str:
    """
    Convenience helper for callers that want a single final string.

    - Uses stream_llm under the hood.
    - Collects {"type":"token"} chunks into one string.
    - Raises RuntimeError on {"type":"error"}.
    - Retries once for transient stream disconnects; for OpenAI it falls back to a non-stream call.
    """
    from .streaming import enhance_system_prompt_with_reasoning, get_model_for_route
    if not provider:
        raise ValueError("call_llm_text: provider is required")

    if not model:
        model = get_model_for_route(provider, route)

    # Build augmented system prompt (kept small-ish and structured)
    sys = system_prompt or ""
    if constraints_hint is not None:
        try:
            ch = json.dumps(constraints_hint, ensure_ascii=False, separators=(",", ":"), default=str)
        except Exception:
            ch = str(constraints_hint)
        sys += "\n\n[CONSTRAINTS_HINT]\n" + ch

    if repo_snapshot is not None:
        try:
            rs = json.dumps(repo_snapshot, ensure_ascii=False, separators=(",", ":"), default=str)
        except Exception:
            rs = str(repo_snapshot)
        sys += "\n\n[REPO_SNAPSHOT]\n" + rs

    enhanced_sys = enhance_system_prompt_with_reasoning(sys, enable_reasoning)

    use_messages = messages if messages is not None else [{"role": "user", "content": user_prompt}]

    async def _collect_via_stream() -> str:
        out: List[str] = []
        async for event in stream_llm(
            use_messages,
            system_prompt=enhanced_sys,
            provider=provider,
            model=model,
            enable_reasoning=enable_reasoning,
            route=route,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        ):
            et = event.get("type")
            if et == "token":
                t = event.get("text", "")
                if t:
                    out.append(t)
            elif et == "error":
                raise RuntimeError(event.get("message", "LLM error"))
            else:
                pass
        return "".join(out).strip()

    try:
        return await _collect_via_stream()
    except RuntimeError as e:
        if not _should_retry_stream_error(str(e)):
            raise

        pk = _provider_key(provider)
        logger.warning(f"[call_llm_text] transient stream error, retrying via fallback: {e}")

        if pk == "openai":
            full_messages = [{"role": "system", "content": enhanced_sys}] + use_messages
            return await _openai_text_nonstream(messages=full_messages, system_prompt=enhanced_sys, model=model)

        # For other providers: one more stream attempt
        return await _collect_via_stream()
