# FILE: app/llm/streaming_ollama.py
"""
Ollama local LLM streaming provider.

Uses Ollama's NATIVE streaming API at localhost:11434/api/chat.
This gives us control over the 'think' parameter to suppress
Qwen3's reasoning mode for fast, direct chat responses.

v1.1 (2026-03-23): Switched from OpenAI-compat to native Ollama API
  - Fixes empty content bug with Qwen3 thinking mode
  - Uses think=false for direct responses (63+ tok/s)
  - Proper streaming via NDJSON chunks
v1.0 (2026-03-23): Initial implementation (OpenAI-compat API).
"""

import json
import os
import logging
from typing import AsyncGenerator, Dict, List, Optional

import httpx

from app.llm.streaming import enhance_system_prompt_with_reasoning

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Strip /v1 suffix if present (we use the native API, not OpenAI-compat)
if OLLAMA_BASE_URL.endswith("/v1"):
    OLLAMA_BASE_URL = OLLAMA_BASE_URL[:-3]

OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", "qwen3:14b-split")
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_OUTPUT_TOKENS", "10000"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))


# ── Streaming function ──────────────────────────────────────────────────

async def stream_ollama(
    messages: List[Dict],
    system_prompt: str = "",
    model: Optional[str] = None,
    enable_reasoning: bool = False,
    route: Optional[str] = None,
    tools: Optional[List[Dict]] = None,
    max_tokens: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
) -> AsyncGenerator[Dict, None]:
    """
    Stream from local Ollama instance via its native /api/chat endpoint.

    Uses NDJSON streaming (each line is a JSON object with a 'message' field).
    Supports the 'think' parameter to control Qwen3's reasoning mode.
    """
    use_model = model or OLLAMA_DEFAULT_MODEL
    print(f"[STREAM_OLLAMA] Called: model={use_model}, route={route}")

    # Check Ollama is reachable
    try:
        async with httpx.AsyncClient(timeout=3.0) as hc:
            resp = await hc.get(f"{OLLAMA_BASE_URL}/api/version")
            if resp.status_code != 200:
                raise ConnectionError(f"Ollama returned {resp.status_code}")
    except Exception as e:
        print(f"[STREAM_OLLAMA] ERROR: Ollama not reachable at {OLLAMA_BASE_URL}: {e}")
        yield {"type": "error", "message": f"Ollama not reachable: {e}"}
        return

    print(f"[STREAM_OLLAMA] Using model: {use_model} via {OLLAMA_BASE_URL}")

    enhanced_prompt = enhance_system_prompt_with_reasoning(system_prompt, enable_reasoning)

    # Build Ollama-native message list
    ollama_messages = [{"role": "system", "content": enhanced_prompt}]
    ollama_messages.extend(messages)

    yield {"type": "metadata", "provider": "ollama", "model": use_model}

    effective_max_tokens = max_tokens or OLLAMA_MAX_TOKENS
    effective_timeout = timeout_seconds or OLLAMA_TIMEOUT

    # Build request body — native Ollama format
    body: Dict = {
        "model": use_model,
        "messages": ollama_messages,
        "stream": True,
        "think": enable_reasoning,  # False = direct response, True = chain-of-thought
        # keep_alive removed - let Ollama manage model lifecycle (default 5min timeout)
        "options": {
            "num_predict": effective_max_tokens,
        },
    }

    collected_content = ""
    prompt_tokens = 0
    completion_tokens = 0

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(effective_timeout, connect=10.0)) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_BASE_URL}/api/chat",
                json=body,
            ) as response:
                if response.status_code != 200:
                    error_text = ""
                    async for chunk in response.aiter_text():
                        error_text += chunk
                    print(f"[STREAM_OLLAMA] ERROR: HTTP {response.status_code}: {error_text[:300]}")
                    yield {"type": "error", "message": f"Ollama error {response.status_code}: {error_text[:200]}"}
                    return

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Extract content from the message
                    msg = chunk.get("message", {})
                    content = msg.get("content", "")

                    if content:
                        collected_content += content
                        yield {"type": "token", "content": content}

                    # Check if this is the final chunk
                    if chunk.get("done", False):
                        # Extract token counts from final chunk
                        prompt_tokens = chunk.get("prompt_eval_count", 0)
                        completion_tokens = chunk.get("eval_count", 0)

                        eval_duration = chunk.get("eval_duration", 0)
                        if eval_duration > 0 and completion_tokens > 0:
                            tok_per_sec = completion_tokens / (eval_duration / 1e9)
                            print(f"[STREAM_OLLAMA] Done: {completion_tokens} tokens at {tok_per_sec:.1f} tok/s")

        # Emit done event
        yield {
            "type": "done",
            "provider": "ollama",
            "model": use_model,
            "total_length": len(collected_content),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    except httpx.TimeoutException:
        print(f"[STREAM_OLLAMA] TIMEOUT after {effective_timeout}s")
        yield {"type": "error", "message": f"Ollama timed out after {effective_timeout}s"}

    except Exception as e:
        print(f"[STREAM_OLLAMA] ERROR: {type(e).__name__}: {e}")
        yield {"type": "error", "message": f"Ollama error: {type(e).__name__}: {e}"}