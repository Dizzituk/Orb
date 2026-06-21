# FILE: app/pipeline_v2/llm_tools.py
# Purpose: LLM Tool-Calling Loop for the Agentic Builder.
# Called-by: app.pipeline_v2.agentic_builder, app.pipeline_v2.llm_compat, app.pipeline_v2.llm_tools_anthropic, app.pipeline_v2.segment_tracker (+1 more)
# Depends-on: app.pipeline_v2, app.pipeline_v2.android_sandbox, app.pipeline_v2.llm_caller, app.pipeline_v2.llm_response_parsing (+6 more)
# Last-renovated: 2026-06-11
"""
LLM Tool-Calling Loop for the Agentic Builder.

Handles the OpenAI function-calling protocol:
  1. Send messages + tool definitions to the API
  2. If the model returns tool_calls, execute them
  3. Send tool results back as tool messages
  4. Repeat until the model returns a text response (no more tool calls)

Supports OpenAI (GPT-5.4) natively. Anthropic/Google would need
adapter logic (not implemented yet — GPT-5.4 is primary).

v1.0 (2026-03-07): Initial implementation for ASTRA v2.1.
v1.1 (2026-03-08): Added write integrity checking — pre-flight content
                    validation + post-write read-back verification.
v1.4 (2026-06-09): Tools+reasoning now routes through the Responses API
                    (llm_responses_api.py) — gpt-5.x rejects reasoning_effort
                    with function tools on /v1/chat/completions. All API calls
                    wrapped in an asyncio.wait_for hard timeout: the client's
                    own timeout was observed not firing (a wedged call froze a
                    build for 17 minutes).
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.pipeline_v2.llm_response_parsing import (
    _extract_tool_calls,
    _extract_text,
    _make_assistant_message,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions — moved to llm_tool_defs.py (2026-06-10 file-size split;
# Jobs 8-9 needed room in this file). Imported here so existing
# `from app.pipeline_v2.llm_tools import TOOLS` callers keep working.
# JOB 9: the toolset now includes edit_file (surgical unique replace).
# ---------------------------------------------------------------------------

from app.pipeline_v2.llm_tool_defs import TOOLS

# Tool execution + per-segment routing extracted 2026-06-21 (BATCH 7) to
# llm_tool_exec.py. Re-imported so run_tool_loop (below) resolves _execute_tool,
# and external importers (agentic_builder / llm_tools_anthropic / segment_tracker /
# verifier_agent) resolve set_tool_* / _execute_tool through this module unchanged.
from app.pipeline_v2.llm_tool_exec import (
    set_tool_profile,
    set_tool_segment,
    set_tool_manifest,
    _find_segment_for_path,
    _resolve_active_profile,
    _execute_tool,
)

# ---------------------------------------------------------------------------
# Main tool-calling loop
# ---------------------------------------------------------------------------

async def run_tool_loop(
    system_prompt: str,
    initial_user_message: str,
    provider: str,
    model: str,
    max_iterations: int = 50,
    max_tokens: int = 128000,
    on_tool_call: Optional[Callable[[str, Dict], None]] = None,
    on_text: Optional[Callable[[str], None]] = None,
    existing_messages: Optional[List[Dict]] = None,
    reasoning: Optional[Dict] = None,
) -> Tuple[List[Dict], int, int]:
    """Run an agentic tool-calling loop.

    Sends the initial prompt with tool definitions, then loops:
    model returns tool_calls -> we execute -> send results -> model continues.

    Args:
        system_prompt: System message.
        initial_user_message: First user message (ignored if existing_messages).
        provider: "openai" (primary).
        model: Model name (e.g. "gpt-5.4").
        max_iterations: Max tool-call rounds before stopping.
        max_tokens: Max output tokens per call.
        on_tool_call: Callback(tool_name, args) for progress tracking.
        on_text: Callback(text) for model's text responses.
        existing_messages: If provided, continues from this conversation history
            instead of starting fresh. The initial_user_message is appended as
            a new user message to this history. This keeps the builder in the
            same context window so it knows what it already built.

    Returns:
        (messages_history, total_input_tokens, total_output_tokens)
    """
    if provider == "anthropic":
        # JOB 8 (2026-06-10): Claude models run through the Anthropic adapter
        # (tool_use / tool_result blocks, thinking, image tool results).
        # NOTE: existing_messages must be Anthropic-native format when used
        # with this provider — the formats are not interchangeable.
        from app.pipeline_v2.llm_tools_anthropic import run_anthropic_tool_loop
        return await run_anthropic_tool_loop(
            system_prompt=system_prompt,
            initial_user_message=initial_user_message,
            model=model,
            max_iterations=max_iterations,
            max_tokens=max_tokens,
            on_tool_call=on_tool_call,
            on_text=on_text,
            existing_messages=existing_messages,
            thinking=(True if reasoning else None),
        )

    if existing_messages:
        messages = list(existing_messages)
        messages.append({"role": "user", "content": initial_user_message})
        logger.info(
            "[llm_tools] CONTINUATION MODE: %d existing messages + new instruction",
            len(existing_messages),
        )
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_user_message},
        ]

    total_input_tokens = 0
    total_output_tokens = 0

    for iteration in range(max_iterations):
        logger.info("[llm_tools] Iteration %d — calling %s/%s", iteration + 1, provider, model)

        response, in_tok, out_tok = await _call_api(
            provider=provider,
            model=model,
            messages=messages,
            tools=TOOLS,
            max_tokens=max_tokens,
            reasoning=reasoning,
        )
        total_input_tokens += in_tok
        total_output_tokens += out_tok

        if response is None:
            logger.error("[llm_tools] API call returned None at iteration %d", iteration + 1)
            break

        tool_calls = _extract_tool_calls(response, provider)

        if not tool_calls:
            text = _extract_text(response, provider)
            if on_text and text:
                on_text(text)
            messages.append(_make_assistant_message(response, provider))
            logger.info("[llm_tools] Text response at iteration %d (%d chars)", iteration + 1, len(text or ""))
            break

        messages.append(_make_assistant_message(response, provider))

        for tc in tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            tool_id = tc["id"]

            if on_tool_call:
                on_tool_call(tool_name, tool_args)

            result = await _execute_tool(tool_name, tool_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": result,
            })

            logger.info("[llm_tools] Tool %s -> %d chars result", tool_name, len(result))

    else:
        logger.warning("[llm_tools] Hit max iterations (%d)", max_iterations)

    return messages, total_input_tokens, total_output_tokens


# ---------------------------------------------------------------------------
# API call (OpenAI)
# ---------------------------------------------------------------------------

# Hard per-call timeout (Issue #3, 2026-06-09): the AsyncOpenAI client's own
# timeout=180 was observed NOT firing — a wedged request froze a build for
# 17 minutes with no error. asyncio.wait_for guarantees cancellation.
HARD_CALL_TIMEOUT = int(os.getenv("ASTRA_V2_BUILDER_CALL_TIMEOUT", "300"))


def _sanitize_for_chat_api(messages: List[Dict]) -> List[Dict]:
    """Strip private keys (e.g. _responses_raw_items) — chat completions
    rejects unknown message fields."""
    return [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]


async def _call_api(
    provider: str,
    model: str,
    messages: List[Dict],
    tools: List[Dict],
    max_tokens: int,
    reasoning: Optional[Dict] = None,
) -> Tuple[Optional[Any], int, int]:
    """Call the LLM API with tools. Returns (response, input_tokens, output_tokens)."""

    if provider == "openai":
        return await _call_openai(model, messages, tools, max_tokens, reasoning=reasoning)
    else:
        logger.warning("[llm_tools] Provider %s does not support tool calling — using text mode", provider)
        from app.pipeline_v2.llm_caller import call_llm
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        try:
            text = await call_llm(provider=provider, model=model,
                                  system_prompt=system, user_prompt=user,
                                  max_tokens=max_tokens)
            return {"text": text}, 0, 0
        except Exception as e:
            logger.error("[llm_tools] Fallback call failed: %s", e)
            return None, 0, 0


async def _call_openai(
    model: str,
    messages: List[Dict],
    tools: List[Dict],
    max_tokens: int,
    reasoning: Optional[Dict] = None,
) -> Tuple[Optional[Any], int, int]:
    """Call OpenAI with function calling.

    v1.4 (2026-06-09): reasoning + tools => Responses API (chat completions
    rejects that combination on gpt-5.x). On Responses failure, falls back to
    chat completions WITHOUT reasoning — same behaviour the 400-fallback gave
    before, but now logged loudly. Every call is hard-capped by
    asyncio.wait_for so a wedged request cannot freeze the build.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.error("[llm_tools] openai package not installed")
        return None, 0, 0

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.error("[llm_tools] OPENAI_API_KEY not set")
        return None, 0, 0

    client = AsyncOpenAI(api_key=api_key, timeout=180.0)

    _use_reasoning = bool(
        reasoning
        and isinstance(reasoning, dict)
        and str(reasoning.get("effort", "")).lower() in ("low", "medium", "high", "xhigh", "max")
    )

    # --- Tools + reasoning: Responses API path (v1.4) ---
    if _use_reasoning and tools:
        try:
            from app.pipeline_v2.llm_responses_api import call_openai_responses
            adapted, in_tok, out_tok = await asyncio.wait_for(
                call_openai_responses(
                    client, model, messages, tools, max_tokens,
                    str(reasoning["effort"]).lower(),
                ),
                timeout=HARD_CALL_TIMEOUT,
            )
            return adapted, in_tok, out_tok
        except asyncio.TimeoutError:
            logger.error(
                "[llm_tools] Responses API call hard-timed out after %ss — cancelled",
                HARD_CALL_TIMEOUT,
            )
            return None, 0, 0
        except Exception as e:
            logger.warning(
                "[llm_tools] Responses API path failed (%s) — falling back to "
                "chat completions WITHOUT reasoning",
                str(e)[:300],
            )

    # --- Chat Completions path (no reasoning, or Responses fallback) ---
    create_kwargs: Dict[str, Any] = dict(
        model=model,
        messages=_sanitize_for_chat_api(messages),
        tools=tools,
        max_completion_tokens=min(max_tokens, 16384),
        temperature=0,
    )

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(**create_kwargs),
            timeout=HARD_CALL_TIMEOUT,
        )
        usage = response.usage
        return response, (usage.prompt_tokens if usage else 0), (usage.completion_tokens if usage else 0)
    except asyncio.TimeoutError:
        logger.error(
            "[llm_tools] Chat completions call hard-timed out after %ss — cancelled",
            HARD_CALL_TIMEOUT,
        )
        return None, 0, 0
    except Exception as e:
        # Some gpt-5.x variants only accept the default temperature — retry once without it.
        msg = str(e)
        if "temperature" in msg:
            create_kwargs.pop("temperature", None)
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(**create_kwargs),
                    timeout=HARD_CALL_TIMEOUT,
                )
                usage = response.usage
                return response, (usage.prompt_tokens if usage else 0), (usage.completion_tokens if usage else 0)
            except Exception as e2:
                logger.error("[llm_tools] OpenAI API error (after temperature retry): %s", e2)
                return None, 0, 0
        logger.error("[llm_tools] OpenAI API error: %s", e)
        return None, 0, 0


# ---------------------------------------------------------------------------
# Response parsing — moved to llm_response_parsing.py (2026-06-09 file-size
# split; llm_tools hit the 30KB ceiling). Imported at the top of this file.
# ---------------------------------------------------------------------------

