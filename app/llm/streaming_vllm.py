# FILE: app/llm/streaming_vllm.py
# Purpose: vLLM local agent-worker streaming provider (OpenAI-compatible API) — serves "Nat".
# Called-by: app.llm._streaming_utils_3 (stream_llm dispatch)
# Depends-on: openai SDK pointed at the local vLLM endpoint
# Last-renovated: 2026-06-19
"""
vLLM local streaming provider — the serving channel for the local agent
worker "Nat" (and, later, the parallel 26B fleet).

vLLM exposes an OpenAI-compatible server, so this mirrors `stream_openai`
(same canonical event schema, same streaming tool-call delta parsing) but:

  - points the openai SDK at VLLM_BASE_URL instead of api.openai.com,
  - drops the OpenAI-only quirks (max_completion_tokens gate, the 128-tool
    cap, reasoning_effort) — vLLM takes plain `max_tokens`,
  - controls the model's NATIVE thinking via `extra_body.chat_template_kwargs`
    (the vLLM-standard toggle) instead of the <THINKING>/<ANSWER> prompt
    injection, defaulting thinking OFF (millisecond memory jobs need it off),
  - emits BOTH `text` and `content` on token events so it satisfies the
    provider-layer collector (`call_llm_text` reads `text`) AND any SSE
    consumer that reads `content`.

Config is read at CALL TIME (not import time) so an env change / .env reload
takes effect without re-importing — and so tests can repoint VLLM_BASE_URL.

This module knows NOTHING about memory or Nat's jobs — a generic provider
branch. Callers select it with provider="vllm_local".
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


# ── Config accessors (ENV-driven, read per call) ─────────────────────────

def _base_url() -> str:
    return os.getenv("VLLM_BASE_URL", "http://localhost:8003/v1")


def _api_key() -> str:
    # vLLM ignores the key by default; some deployments set --api-key.
    return os.getenv("VLLM_API_KEY", "EMPTY")


def _default_model() -> str:
    # Nat currently runs Qwen2.5-3B-Instruct-AWQ (served as "qwen2.5-3b"); the
    # real value comes from NAT_MODEL in .env — this is only a last-resort default.
    return os.getenv("NAT_MODEL") or os.getenv("VLLM_DEFAULT_MODEL") or "qwen2.5-3b"


def _max_tokens() -> int:
    try:
        return int(os.getenv("VLLM_MAX_OUTPUT_TOKENS", "1024"))
    except Exception:
        return 1024


def _timeout() -> int:
    try:
        return int(os.getenv("VLLM_TIMEOUT_SECONDS", "60"))
    except Exception:
        return 60


def _thinking_kwarg() -> str:
    # The chat-template kwarg that toggles native reasoning (gemma/qwen-style
    # templates read `enable_thinking`); override if a model differs.
    return os.getenv("NAT_THINKING_KWARG", "enable_thinking")


def _precheck() -> bool:
    # Cheap reachability probe before the real call (clean "down" error instead
    # of a cryptic SDK trace). Latency-critical callers wrap us in their own hard
    # timeout, so this never blocks a chat reply.
    return os.getenv("VLLM_PRECHECK", "1").strip().lower() in ("1", "true", "yes")


def _to_openai_tools(tools: List[Dict]) -> List[Dict]:
    """Convert internal/Anthropic-format tools to OpenAI function format."""
    out: List[Dict] = []
    for t in tools or []:
        out.append({
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", t.get("parameters", {"type": "object", "properties": {}})),
            },
        })
    return out


async def _reachable(base_url: str, timeout: float) -> Optional[str]:
    """Return None if the vLLM server answers, else a short reason string."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=2.0)) as hc:
            resp = await hc.get(f"{base_url}/models")
            if resp.status_code != 200:
                return f"HTTP {resp.status_code} from {base_url}/models"
        return None
    except Exception as e:  # noqa: BLE001 — any failure means "not reachable"
        return f"{type(e).__name__}: {e}"


async def stream_vllm(
    messages: List[Dict],
    system_prompt: str = "",
    model: Optional[str] = None,
    enable_reasoning: bool = False,
    route: Optional[str] = None,
    tools: Optional[List[Dict]] = None,
    max_tokens: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
) -> AsyncGenerator[Dict, None]:
    """Stream from a local vLLM OpenAI-compatible server (serves Nat).

    `enable_reasoning` maps to the model's NATIVE thinking: False (default)
    suppresses it for fast, direct JSON; True turns it on for deliberation.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError:
        yield {"type": "error", "message": "openai package not installed"}
        return

    base_url = _base_url()
    use_model = model or _default_model()
    effective_max_tokens = int(max_tokens or _max_tokens())
    effective_timeout = float(timeout_seconds or _timeout())
    print(f"[STREAM_VLLM] Called: model={use_model}, route={route}, thinking={enable_reasoning}")

    if _precheck():
        reason = await _reachable(base_url, effective_timeout)
        if reason:
            print(f"[STREAM_VLLM] ERROR: vLLM not reachable at {base_url}: {reason}")
            yield {"type": "error", "message": f"vLLM not reachable: {reason}"}
            return

    full_messages: List[Dict] = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    yield {"type": "metadata", "provider": "vllm_local", "model": use_model}

    client = AsyncOpenAI(
        api_key=_api_key(),
        base_url=base_url,
        timeout=httpx.Timeout(effective_timeout, connect=10.0),
    )

    create_kwargs: Dict[str, Any] = {
        "model": use_model,
        "messages": full_messages,
        "stream": True,
        "max_tokens": effective_max_tokens,
    }
    if tools:
        create_kwargs["tools"] = _to_openai_tools(tools)

    # Native-thinking control via vLLM's chat_template_kwargs. Best-effort: if the
    # server/template rejects it, fall back to a plain call so a template that
    # doesn't expose the toggle can never break Nat.
    extra_body = {"chat_template_kwargs": {_thinking_kwarg(): bool(enable_reasoning)}}

    prompt_tokens = completion_tokens = total_tokens = None

    try:
        try:
            stream = await client.chat.completions.create(
                **create_kwargs,
                extra_body=extra_body,
                stream_options={"include_usage": True},
            )
        except TypeError:
            # Older SDKs reject extra_body/stream_options kwargs entirely.
            stream = await client.chat.completions.create(**create_kwargs)
        except Exception as first_err:  # noqa: BLE001
            # Most likely the template doesn't accept the thinking kwarg — retry
            # once without extra_body before giving up.
            logger.debug("[stream_vllm] create with extra_body failed (%s); retrying plain", first_err)
            stream = await client.chat.completions.create(
                **create_kwargs,
                stream_options={"include_usage": True},
            )

        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)

            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if not delta:
                continue

            # Streaming tool-call deltas (identical shape to OpenAI).
            tool_calls_delta = getattr(delta, "tool_calls", None)
            if tool_calls_delta:
                for tc in tool_calls_delta:
                    func = getattr(tc, "function", None)
                    if not func:
                        continue
                    tc_id = getattr(tc, "id", None)
                    tc_name = getattr(func, "name", None)
                    tc_args = getattr(func, "arguments", None) or ""
                    if tc_id and tc_name:
                        yield {
                            "type": "tool_use",
                            "id": tc_id,
                            "name": tc_name,
                            "input_json": tc_args,
                            "index": getattr(tc, "index", 0),
                        }
                    elif tc_args:
                        yield {
                            "type": "tool_use_delta",
                            "arguments": tc_args,
                            "index": getattr(tc, "index", 0),
                        }

            content = getattr(delta, "content", None)
            if content:
                # Both keys: `text` for call_llm_text/provider layer, `content`
                # for SSE consumers. Additive, never wrong.
                yield {"type": "token", "text": content, "content": content}

        if any(v is not None for v in (prompt_tokens, completion_tokens, total_tokens)):
            if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
            yield {
                "type": "done",
                "provider": "vllm_local",
                "model": use_model,
                "usage": {
                    "prompt_tokens": int(prompt_tokens or 0),
                    "completion_tokens": int(completion_tokens or 0),
                    "total_tokens": int(total_tokens or (int(prompt_tokens or 0) + int(completion_tokens or 0))),
                },
            }
        else:
            yield {"type": "done", "provider": "vllm_local", "model": use_model}

    except httpx.TimeoutException:
        print(f"[STREAM_VLLM] TIMEOUT after {effective_timeout}s")
        yield {"type": "error", "message": f"vLLM timed out after {effective_timeout}s"}
    except Exception as e:  # noqa: BLE001
        print(f"[STREAM_VLLM] ERROR: {type(e).__name__}: {e}")
        yield {"type": "error", "message": f"vLLM error: {type(e).__name__}: {e}"}
