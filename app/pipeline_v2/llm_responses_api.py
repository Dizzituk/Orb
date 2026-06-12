# FILE: app/pipeline_v2/llm_responses_api.py
# Purpose: OpenAI Responses API adapter for the Agentic Builder tool loop.
# Called-by: app.pipeline_v2.llm_tools
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
OpenAI Responses API adapter for the Agentic Builder tool loop.

WHY THIS EXISTS (2026-06-09): gpt-5.x rejects function tools combined with
`reasoning_effort` on /v1/chat/completions ("Please use /v1/responses
instead"). The builder loop keeps its canonical conversation history in
chat-completions format; this module translates that history + tool schema
into Responses API shape, makes the call, and adapts the result back into
a plain dict the existing extractors in llm_tools.py understand:

    {"responses_api": True, "text": str,
     "tool_calls": [{"id", "name", "arguments"}], "_raw_items": [...]}

Raw output items (including reasoning items) are preserved on the assistant
message under "_responses_raw_items" so subsequent turns replay them
faithfully — reasoning models expect their reasoning items back alongside
the function_call items they produced. Keys starting with "_" are private
and must be stripped before any chat-completions call (llm_tools does this).

Single responsibility: Responses API translation. No tool execution here.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def tools_to_responses_format(tools: List[Dict]) -> List[Dict]:
    """Chat-completions tool schema (nested under "function") -> flat Responses schema."""
    out = []
    for t in tools or []:
        fn = t.get("function", {}) if isinstance(t, dict) else {}
        if not fn:
            continue
        out.append({
            "type": "function",
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        })
    return out


def messages_to_responses_input(messages: List[Dict]) -> List[Any]:
    """Chat-completions message history -> Responses API input item list.

    - system/user messages pass through as role messages.
    - assistant messages: if they carry "_responses_raw_items" (a previous
      Responses turn), replay those raw items verbatim; otherwise reconstruct
      a message item + function_call items from the standard fields.
    - tool messages -> function_call_output items.
    """
    items: List[Any] = []
    for m in messages or []:
        role = m.get("role")
        if role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": m.get("tool_call_id", "") or "",
                "output": m.get("content", "") or "",
            })
        elif role == "assistant":
            raw = m.get("_responses_raw_items")
            if raw:
                items.extend(raw)
                continue
            content = m.get("content") or ""
            if content:
                items.append({"role": "assistant", "content": content})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {}) or {}
                items.append({
                    "type": "function_call",
                    "call_id": tc.get("id", "") or "",
                    "name": fn.get("name", "") or "",
                    "arguments": fn.get("arguments", "{}") or "{}",
                })
        else:
            # system / user / developer
            items.append({"role": role or "user", "content": m.get("content", "") or ""})
    return items


def _dump_item(item: Any) -> Dict:
    """SDK output item -> plain dict suitable for replay as an input item."""
    if hasattr(item, "model_dump"):
        return item.model_dump(exclude_none=True, exclude_unset=True)
    if isinstance(item, dict):
        return item
    return {}


async def call_openai_responses(
    client: Any,
    model: str,
    messages: List[Dict],
    tools: List[Dict],
    max_tokens: int,
    effort: str,
) -> Tuple[Dict, int, int]:
    """Make one Responses API call with function tools + reasoning.

    Raises on API failure — the caller (llm_tools._call_openai) owns the
    fallback to chat completions without reasoning.
    """
    response = await client.responses.create(
        model=model,
        input=messages_to_responses_input(messages),
        tools=tools_to_responses_format(tools),
        reasoning={"effort": effort},
        max_output_tokens=min(max_tokens, 16384),
    )

    raw_items: List[Dict] = []
    tool_calls: List[Dict] = []
    text_parts: List[str] = []

    for item in getattr(response, "output", None) or []:
        itype = getattr(item, "type", None)
        if itype == "function_call":
            tool_calls.append({
                "id": getattr(item, "call_id", "") or getattr(item, "id", "") or "",
                "name": getattr(item, "name", "") or "",
                "arguments": getattr(item, "arguments", "{}") or "{}",
            })
            raw_items.append(_dump_item(item))
        elif itype == "reasoning":
            # Replayed next turn so the model keeps its chain across tool calls.
            raw_items.append(_dump_item(item))
        elif itype == "message":
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", "") == "output_text":
                    text_parts.append(getattr(part, "text", "") or "")
            raw_items.append(_dump_item(item))
        else:
            raw_items.append(_dump_item(item))

    usage = getattr(response, "usage", None)
    in_tok = int(getattr(usage, "input_tokens", 0) or 0)
    out_tok = int(getattr(usage, "output_tokens", 0) or 0)

    adapted = {
        "responses_api": True,
        "text": "\n".join(p for p in text_parts if p),
        "tool_calls": tool_calls,
        "_raw_items": raw_items,
    }
    logger.info(
        "[llm_responses_api] %s: %d tool call(s), %d text chars, tokens %d in / %d out",
        model, len(tool_calls), len(adapted["text"]), in_tok, out_tok,
    )
    return adapted, in_tok, out_tok
