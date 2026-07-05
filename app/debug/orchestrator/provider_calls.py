# FILE: app/debug/orchestrator/provider_calls.py
# Purpose: Provider-neutral LLM call layer for the debug orchestrator — the
#          OpenAI tool-loop caller (moved verbatim from subagent_runner) plus an
#          Anthropic mirror with tool/message adapters, uniform extraction
#          helpers, and a single-shot JSON call for the decomposer/planner.
# Called-by: app.debug.orchestrator.subagent_runner, app.debug.orchestrator.decomposer,
#            app.debug.orchestrator.planner
# Depends-on: openai/anthropic SDKs, app.providers._registry_anthropic_shapes,
#             app.providers.registry (single-shot JSON only)
# Last-renovated: 2026-07-02
"""Both providers, one contract (DEBUG_PROVIDER toggle, 2026-07-02).

The subagent loop's internal history stays in the OpenAI chat shape (assistant
`tool_calls` + role:"tool" results) — the loop logic in subagent_runner is
untouched. This module owns everything provider-specific:

* _call_openai_with_tools — the exact caller subagent_runner always used
  (max_completion_tokens, reasoning_effort with unsupported-retry, 300s timeout).
* _call_anthropic_with_tools — mirrors it: AsyncAnthropic, flat debug tool defs
  converted to {name, description, input_schema}, max_tokens set, reasoning
  effort mapped to the family-correct thinking shape via
  app.providers._registry_anthropic_shapes (the same mapping pipeline_v2 /
  the registry use — no bespoke thinking config here), timeout parity.
* openai_history_to_anthropic — OpenAI-shape history -> (system, Anthropic
  content-block messages). Assistant turns produced by the Anthropic caller
  carry their raw blocks in a private "_anthropic_content" key so thinking
  blocks survive round-trips verbatim (API requirement when continuing a
  tool-use turn with thinking enabled). Consecutive role:"tool" results merge
  into ONE user message (tool_result blocks first, any trailing plain-user text
  appended after — Anthropic rejects non-alternating roles).
* extract_tool_calls / extract_assistant_message / extract_final_text —
  normalise both providers' responses to the shapes run_subagent consumes.

Adapter shapes cribbed from app/pipeline_v2/llm_tools_anthropic.py (JOB 8) —
do not invent a new mapping; extend that pattern here if a block type is added.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# LLM client timeout (seconds). Thinking models on high effort on a large
# investigation can run 60-120s per turn; give generous headroom. Shared by
# both providers (parity requirement).
_LLM_TIMEOUT_SECONDS = 300.0


# ---------------------------------------------------------------------------
# OpenAI function-calling format (our tool_definitions use flat schema;
# OpenAI expects {"type":"function","function":{...}})
# ---------------------------------------------------------------------------

def _to_openai_tool(tool_def: dict) -> dict:
    """Wrap a flat tool definition as an OpenAI function tool."""
    return {
        "type": "function",
        "function": {
            "name": tool_def["name"],
            "description": tool_def.get("description", ""),
            "parameters": tool_def.get("parameters", {"type": "object", "properties": {}}),
        },
    }


# ---------------------------------------------------------------------------
# OpenAI caller (moved verbatim from subagent_runner.py, 2026-07-02 — the
# openai path must remain the exact code path that ran before the toggle)
# ---------------------------------------------------------------------------

async def _call_openai_with_tools(
    model: str,
    messages: List[Dict],
    tools: List[dict],
    max_tokens: int = 8000,
    reasoning_effort: Optional[str] = None,
) -> Tuple[Any, int, int]:
    """Single OpenAI chat.completions call with tool support.

    If reasoning_effort is provided, passes it through to the API. If the
    model does not support it, we retry without (silent degradation).
    """
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise RuntimeError("openai package not installed")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = AsyncOpenAI(api_key=api_key, timeout=_LLM_TIMEOUT_SECONDS)
    openai_tools = [_to_openai_tool(t) for t in tools]

    call_kwargs = dict(
        model=model,
        messages=messages,
        tools=openai_tools,
        tool_choice="auto",
        max_completion_tokens=max_tokens,
    )
    # Apply reasoning effort if a role was specified; retry without if unsupported.
    if reasoning_effort:
        call_kwargs["reasoning_effort"] = reasoning_effort
    try:
        resp = await client.chat.completions.create(**call_kwargs)
    except Exception as e:
        err_str = str(e).lower()
        if reasoning_effort and (
            "reasoning_effort" in err_str or "unrecognized" in err_str or "unsupported" in err_str
        ):
            logger.info("[provider_calls] model=%s does not support reasoning_effort, retrying without", model)
            call_kwargs.pop("reasoning_effort", None)
            resp = await client.chat.completions.create(**call_kwargs)
        else:
            raise

    usage = getattr(resp, "usage", None)
    in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
    out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
    return resp, in_tok, out_tok


# ---------------------------------------------------------------------------
# Anthropic adapters
# ---------------------------------------------------------------------------

def _battr(block: Any, name: str, default: Any = None) -> Any:
    """Attribute-or-key accessor: works on SDK block objects AND plain dicts
    (dicts appear in round-tripped _anthropic_content and in tests)."""
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def to_anthropic_tools(tools: List[dict]) -> List[dict]:
    """Flat debug tool defs ({name, description, parameters}) -> Anthropic
    ({name, description, input_schema}). Already-Anthropic defs pass through."""
    out: List[dict] = []
    for t in tools or []:
        if not isinstance(t, dict) or not t.get("name"):
            continue
        if "input_schema" in t:
            out.append(t)
            continue
        out.append({
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
        })
    return out


def _serialise_content_blocks(content: Any) -> List[Dict]:
    """SDK content blocks -> plain dicts safe to send back to the API.
    Thinking blocks are preserved verbatim (required by the API when
    continuing a tool-use turn with thinking enabled)."""
    blocks: List[Dict] = []
    for b in content or []:
        try:
            if hasattr(b, "model_dump"):
                blocks.append(b.model_dump(exclude_none=True))
            elif isinstance(b, dict):
                blocks.append(b)
        except Exception as exc:
            logger.debug("[provider_calls] block serialise failed: %s", exc)
    return blocks


def openai_history_to_anthropic(messages: List[Dict]) -> Tuple[str, List[Dict]]:
    """OpenAI-shape history -> (system_prompt, Anthropic messages).

    - role:"system" -> collected into the returned system string.
    - assistant with "_anthropic_content" -> raw blocks verbatim (thinking safe).
    - assistant tool_calls -> tool_use blocks; text content -> a text block.
      An assistant turn with neither is dropped (empty content is rejected).
    - role:"tool" -> tool_result blocks, consecutive ones merged into ONE user
      message; a plain user message directly after tool results is appended to
      that same user message as a text block (tool_results stay first).
    """
    system_parts: List[str] = []
    out: List[Dict] = []

    def _append_user_block(block: Dict) -> None:
        if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
            out[-1]["content"].append(block)
        else:
            out.append({"role": "user", "content": [block]})

    for m in messages or []:
        role = m.get("role")
        if role == "system":
            text = str(m.get("content") or "")
            if text:
                system_parts.append(text)
        elif role == "assistant":
            raw = m.get("_anthropic_content")
            if raw:
                out.append({"role": "assistant", "content": raw})
                continue
            content: List[Dict] = []
            text = m.get("content")
            if text:
                content.append({"type": "text", "text": str(text)})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                content.append({
                    "type": "tool_use",
                    "id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "input": args,
                })
            if content:
                out.append({"role": "assistant", "content": content})
        elif role == "tool":
            _append_user_block({
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id", ""),
                "content": str(m.get("content") or ""),
            })
        else:  # user
            text = str(m.get("content") or "")
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append({"type": "text", "text": text})
            else:
                out.append({"role": "user", "content": text})

    return "\n\n".join(system_parts), out


# ---------------------------------------------------------------------------
# Anthropic caller (mirrors _call_openai_with_tools' contract)
# ---------------------------------------------------------------------------

async def _call_anthropic_with_tools(
    model: str,
    messages: List[Dict],
    tools: List[dict],
    max_tokens: int = 8000,
    reasoning_effort: Optional[str] = None,
) -> Tuple[Any, int, int]:
    """Single Anthropic messages.create call with tool support.

    reasoning_effort maps to the family-correct thinking shape via
    shape_anthropic_create_kwargs (adaptive thinking + output_config effort on
    Opus/Sonnet 4.6+, effort-only on Fable/Mythos, nothing when None). If the
    model rejects the thinking config, we retry once without it — the same
    silent degradation the OpenAI caller applies to reasoning_effort.
    """
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    client = AsyncAnthropic(api_key=api_key, timeout=_LLM_TIMEOUT_SECONDS)
    system, anthropic_messages = openai_history_to_anthropic(messages)

    call_kwargs: Dict[str, Any] = dict(
        model=model,
        messages=anthropic_messages,
        tools=to_anthropic_tools(tools),
        max_tokens=max_tokens,
    )
    if system:
        call_kwargs["system"] = system

    from app.providers._registry_anthropic_shapes import shape_anthropic_create_kwargs
    shape_anthropic_create_kwargs(
        call_kwargs, model,
        {"effort": reasoning_effort} if reasoning_effort else None,
    )

    try:
        resp = await client.messages.create(**call_kwargs)
    except Exception as e:
        err_str = str(e).lower()
        sent_thinking = "thinking" in call_kwargs or "extra_body" in call_kwargs
        if sent_thinking and any(s in err_str for s in ("thinking", "output_config", "effort")):
            logger.info("[provider_calls] model=%s rejected thinking config, retrying without", model)
            call_kwargs.pop("thinking", None)
            call_kwargs.pop("extra_body", None)
            resp = await client.messages.create(**call_kwargs)
        else:
            raise

    usage = getattr(resp, "usage", None)
    in_tok = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
    out_tok = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
    return resp, in_tok, out_tok


# ---------------------------------------------------------------------------
# Provider dispatch + uniform extraction (the surface run_subagent consumes)
# ---------------------------------------------------------------------------

async def call_llm_with_tools(
    provider: str,
    model: str,
    messages: List[Dict],
    tools: List[dict],
    max_tokens: int = 8000,
    reasoning_effort: Optional[str] = None,
) -> Tuple[Any, int, int]:
    """One tool-capable LLM call on the given provider. Returns
    (raw_response, input_tokens, output_tokens); pass the same provider to the
    extract_* helpers below."""
    if (provider or "openai").lower() == "anthropic":
        return await _call_anthropic_with_tools(
            model=model, messages=messages, tools=tools,
            max_tokens=max_tokens, reasoning_effort=reasoning_effort,
        )
    return await _call_openai_with_tools(
        model=model, messages=messages, tools=tools,
        max_tokens=max_tokens, reasoning_effort=reasoning_effort,
    )


def extract_assistant_message(resp: Any, provider: str = "openai") -> Dict:
    """Provider response -> OpenAI-shape assistant message for the loop history.
    Anthropic responses additionally stash their raw blocks in
    "_anthropic_content" so thinking blocks round-trip verbatim."""
    if (provider or "openai").lower() == "anthropic":
        blocks = _serialise_content_blocks(getattr(resp, "content", None))
        text = "".join(
            str(b.get("text", "")) for b in blocks if b.get("type") == "text"
        )
        out: Dict[str, Any] = {"role": "assistant", "content": text}
        tool_calls = [
            {
                "id": str(b.get("id", "")),
                "type": "function",
                "function": {
                    "name": str(b.get("name", "")),
                    "arguments": json.dumps(b.get("input") or {}),
                },
            }
            for b in blocks if b.get("type") == "tool_use"
        ]
        if tool_calls:
            out["tool_calls"] = tool_calls
        if blocks:
            out["_anthropic_content"] = blocks
        return out

    msg = resp.choices[0].message
    out = {"role": "assistant", "content": msg.content or ""}
    if getattr(msg, "tool_calls", None):
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return out


def extract_tool_calls(resp: Any, provider: str = "openai") -> List[Dict]:
    """Provider response -> [{"id", "name", "args"}] (args always a dict)."""
    if (provider or "openai").lower() == "anthropic":
        out = []
        for b in getattr(resp, "content", None) or []:
            if _battr(b, "type", "") == "tool_use":
                out.append({
                    "id": str(_battr(b, "id", "") or ""),
                    "name": str(_battr(b, "name", "") or ""),
                    "args": dict(_battr(b, "input", {}) or {}),
                })
        return out

    msg = resp.choices[0].message
    if not getattr(msg, "tool_calls", None):
        return []
    out = []
    for tc in msg.tool_calls:
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except Exception:
            args = {}
        out.append({"id": tc.id, "name": tc.function.name, "args": args})
    return out


def extract_final_text(resp: Any, provider: str = "openai") -> str:
    """Provider response -> concatenated assistant text."""
    if (provider or "openai").lower() == "anthropic":
        return "".join(
            str(_battr(b, "text", "") or "")
            for b in (getattr(resp, "content", None) or [])
            if _battr(b, "type", "") == "text"
        )
    msg = resp.choices[0].message
    return msg.content or ""


# ---------------------------------------------------------------------------
# Single-shot JSON call (decomposer / planner anthropic path)
# ---------------------------------------------------------------------------

_JSON_ONLY_SUFFIX = (
    "\n\nCRITICAL OUTPUT RULE: respond with exactly ONE JSON object and nothing "
    "else — no markdown fences, no commentary before or after the JSON."
)


async def call_anthropic_json(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
    effort: Optional[str] = "high",
) -> str:
    """Single-shot JSON-oriented Anthropic call via the provider registry.

    Anthropic has no response_format=json_object, so the JSON-only instruction
    is strengthened in the system prompt; the callers' fence/prose-tolerant
    parsers remain the safety net. Reasoning rides the registry's `reasoning`
    dict (mapped centrally by shape_anthropic_create_kwargs). Raises on any
    non-success so callers hit their existing fallback paths.
    """
    from app.providers.registry import llm_call

    result = await llm_call(
        provider_id="anthropic",
        model_id=model,
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=(system_prompt or "") + _JSON_ONLY_SUFFIX,
        max_tokens=max_tokens,
        timeout_seconds=int(_LLM_TIMEOUT_SECONDS),
        enable_tools=False,
        reasoning={"effort": effort} if effort else None,
    )
    status = getattr(getattr(result, "status", None), "value", "")
    if status != "success":
        raise RuntimeError(
            f"anthropic call failed ({status or 'unknown'}): "
            f"{getattr(result, 'error_message', '') or ''}"
        )
    return (getattr(result, "content", "") or "").strip()
