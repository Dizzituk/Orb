# FILE: app/pipeline_v2/llm_tools_anthropic.py
"""
JOB 8 (2026-06-10) - Anthropic tool-use loop adapter.

Mirrors run_tool_loop's contract (app/pipeline_v2/llm_tools.py) for Claude
models so they can run agentic loops: tool definitions, tool_use /
tool_result blocks, extended thinking, hard per-call timeouts, token
accounting. This was the hard blocker for a Claude verifier with hands.

Key differences from the OpenAI loop, handled here:
  - Anthropic separates `system` from `messages`; tool results are sent as
    user-role messages containing tool_result blocks.
  - Thinking blocks returned by the model MUST be passed back verbatim in
    the assistant message when continuing after tool use.
  - tool_result content may contain IMAGE blocks - this is what lets the
    verifier SEE screenshots (Job 10) rather than reading descriptions.

Message history format here is Anthropic-native (list of
{"role": ..., "content": str | [blocks]}). Do not feed OpenAI-format
histories in; the verifier agent builds its history through this module
from the start.

Default tool executor is llm_tools._execute_tool (read_file / write_file /
edit_file / run_shell with profile routing + write integrity). A custom
executor can return either a plain string OR a dict:
  {"text": str}                                  -> text tool_result
  {"image": {"data": b64, "media_type": "..."}}  -> image tool_result
  {"text": ..., "image": {...}}                  -> both blocks
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

HARD_CALL_TIMEOUT = int(os.getenv("ASTRA_V2_BUILDER_CALL_TIMEOUT", "300"))
MAX_TOOL_RESULT_CHARS = 30_000
DEFAULT_THINKING_BUDGET = int(os.getenv("ASTRA_ANTHROPIC_THINKING_BUDGET", "4096"))


# ---------------------------------------------------------------------------
# Tool-schema conversion (OpenAI function format -> Anthropic tool format)
# ---------------------------------------------------------------------------

def to_anthropic_tools(openai_tools: Optional[List[Dict]]) -> List[Dict]:
    """Convert OpenAI function-calling tool defs to Anthropic tool defs.

    Accepts either OpenAI format ({"type": "function", "function": {...}})
    or already-Anthropic format ({"name", "description", "input_schema"})
    and normalises to the latter.
    """
    out: List[Dict] = []
    for t in openai_tools or []:
        if not isinstance(t, dict):
            continue
        if "input_schema" in t and "name" in t:
            out.append(t)
            continue
        fn = t.get("function", t)
        name = fn.get("name")
        if not name:
            continue
        out.append({
            "name": name,
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return out


def _tool_result_content(result: Any) -> Any:
    """Build tool_result content from an executor return value."""
    if isinstance(result, dict):
        blocks: List[Dict] = []
        text = result.get("text")
        if text:
            blocks.append({"type": "text", "text": str(text)[:MAX_TOOL_RESULT_CHARS]})
        img = result.get("image")
        if isinstance(img, dict) and img.get("data"):
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img.get("media_type", "image/png"),
                    "data": img["data"],
                },
            })
        if blocks:
            return blocks
        return [{"type": "text", "text": "(empty result)"}]
    return str(result)[:MAX_TOOL_RESULT_CHARS]


def _serialise_content_blocks(content: Any) -> List[Dict]:
    """SDK content blocks -> plain dicts safe to send back to the API.

    Thinking blocks are preserved verbatim (required by the API when
    continuing a tool-use turn with thinking enabled).
    """
    blocks: List[Dict] = []
    for b in content or []:
        try:
            if hasattr(b, "model_dump"):
                blocks.append(b.model_dump(exclude_none=True))
            elif isinstance(b, dict):
                blocks.append(b)
        except Exception as exc:
            logger.debug("[llm_anthropic] block serialise failed: %s", exc)
    return blocks


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_anthropic_tool_loop(
    system_prompt: str,
    initial_user_message: str,
    model: str,
    max_iterations: int = 50,
    max_tokens: int = 16384,
    on_tool_call: Optional[Callable[[str, Dict], None]] = None,
    on_text: Optional[Callable[[str], None]] = None,
    existing_messages: Optional[List[Dict]] = None,
    thinking: Optional[Any] = None,
    tools: Optional[List[Dict]] = None,
    tool_executor: Optional[Callable[[str, Dict], Any]] = None,
) -> Tuple[List[Dict], int, int]:
    """Run an agentic tool loop against an Anthropic model.

    Mirrors llm_tools.run_tool_loop's return contract:
        (messages_history, total_input_tokens, total_output_tokens)

    Args:
        thinking: None (off), True (enabled with default budget), or a dict
            passed straight through (e.g. {"type": "enabled",
            "budget_tokens": 8000}).
        tools: OpenAI- or Anthropic-format tool defs. Defaults to
            llm_tools.TOOLS (read/write/edit/shell).
        tool_executor: async (name, args) -> str | dict. Defaults to
            llm_tools._execute_tool.
        existing_messages: Anthropic-native history to continue from.
    """
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        logger.error("[llm_anthropic] anthropic package not installed")
        return [], 0, 0

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("[llm_anthropic] ANTHROPIC_API_KEY not set")
        return [], 0, 0

    if tools is None:
        from app.pipeline_v2.llm_tools import TOOLS as _DEFAULT_TOOLS
        tools = _DEFAULT_TOOLS
    anthropic_tools = to_anthropic_tools(tools)

    if tool_executor is None:
        from app.pipeline_v2.llm_tools import _execute_tool as _default_exec
        tool_executor = _default_exec

    if existing_messages:
        messages: List[Dict] = list(existing_messages)
        messages.append({"role": "user", "content": initial_user_message})
        logger.info(
            "[llm_anthropic] CONTINUATION MODE: %d existing messages + new instruction",
            len(existing_messages),
        )
    else:
        messages = [{"role": "user", "content": initial_user_message}]

    thinking_cfg: Optional[Dict] = None
    if thinking is True:
        thinking_cfg = {"type": "enabled", "budget_tokens": DEFAULT_THINKING_BUDGET}
    elif isinstance(thinking, dict):
        thinking_cfg = thinking

    client = AsyncAnthropic(api_key=api_key, timeout=180.0)
    total_in = 0
    total_out = 0

    for iteration in range(max_iterations):
        logger.info("[llm_anthropic] Iteration %d - calling %s", iteration + 1, model)

        resp = await _call_anthropic(
            client=client,
            model=model,
            system=system_prompt,
            messages=messages,
            tools=anthropic_tools,
            max_tokens=max_tokens,
            thinking_cfg=thinking_cfg,
        )
        if resp is None:
            logger.error("[llm_anthropic] API call returned None at iteration %d", iteration + 1)
            break

        usage = getattr(resp, "usage", None)
        total_in += int(getattr(usage, "input_tokens", 0) or 0)
        total_out += int(getattr(usage, "output_tokens", 0) or 0)

        content_blocks = list(getattr(resp, "content", []) or [])
        tool_uses = [b for b in content_blocks if getattr(b, "type", "") == "tool_use"]
        text = "".join(
            getattr(b, "text", "") for b in content_blocks if getattr(b, "type", "") == "text"
        )

        messages.append({
            "role": "assistant",
            "content": _serialise_content_blocks(content_blocks),
        })

        if text and on_text:
            try:
                on_text(text)
            except Exception:
                pass

        if not tool_uses:
            logger.info(
                "[llm_anthropic] Text response at iteration %d (%d chars, stop=%s)",
                iteration + 1, len(text or ""), getattr(resp, "stop_reason", "?"),
            )
            break

        results_content: List[Dict] = []
        for b in tool_uses:
            tool_name = getattr(b, "name", "")
            tool_args = dict(getattr(b, "input", {}) or {})
            tool_id = getattr(b, "id", "")

            if on_tool_call:
                try:
                    on_tool_call(tool_name, tool_args)
                except Exception:
                    pass

            try:
                result = await tool_executor(tool_name, tool_args)
            except Exception as exc:
                result = f"ERROR: {tool_name} crashed: {exc}"

            results_content.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": _tool_result_content(result),
            })
            _rlen = len(str(result)) if not isinstance(result, dict) else len(str(result.get("text", ""))) + (9000 if result.get("image") else 0)
            logger.info("[llm_anthropic] Tool %s -> ~%d chars result", tool_name, _rlen)

        messages.append({"role": "user", "content": results_content})

    else:
        logger.warning("[llm_anthropic] Hit max iterations (%d)", max_iterations)

    return messages, total_in, total_out


async def _call_anthropic(
    client: Any,
    model: str,
    system: str,
    messages: List[Dict],
    tools: List[Dict],
    max_tokens: int,
    thinking_cfg: Optional[Dict],
) -> Optional[Any]:
    """One messages.create call with hard timeout + graceful degradation.

    Retries once without thinking if the thinking config is rejected, and
    retries once after a short backoff on overload/rate-limit errors.
    """
    kwargs: Dict[str, Any] = dict(
        model=model,
        system=system,
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
    )
    if thinking_cfg:
        kwargs["thinking"] = thinking_cfg
        budget = int(thinking_cfg.get("budget_tokens", 0) or 0)
        if budget and kwargs["max_tokens"] <= budget:
            kwargs["max_tokens"] = budget + 4096

    for attempt in (1, 2):
        try:
            return await asyncio.wait_for(
                client.messages.create(**kwargs),
                timeout=HARD_CALL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error(
                "[llm_anthropic] Call hard-timed out after %ss - cancelled",
                HARD_CALL_TIMEOUT,
            )
            return None
        except Exception as e:
            msg = str(e)
            if "thinking" in msg.lower() and "thinking" in kwargs:
                logger.warning(
                    "[llm_anthropic] Thinking config rejected (%s) - retrying without thinking",
                    msg[:200],
                )
                kwargs.pop("thinking", None)
                continue
            if attempt == 1 and any(s in msg.lower() for s in ("overloaded", "rate", "529", "429")):
                logger.warning("[llm_anthropic] Transient API error (%s) - backing off 5s", msg[:150])
                await asyncio.sleep(5)
                continue
            logger.error("[llm_anthropic] API error: %s", msg[:400])
            return None
    return None
