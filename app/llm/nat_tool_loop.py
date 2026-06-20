# FILE: app/llm/nat_tool_loop.py
# Purpose: Tool-execution loop for the local (vLLM) agent worker "Nat".
#          Gives Nat real agentic tool use + sequential-thinking so the future
#          26B orchestrator can delegate actual work, not just text.
# Called-by: app.llm.workers.nat_worker.run_nat_job (tools path)
# Depends-on: app.llm._streaming_utils_3.stream_llm, app.llm.chat_tool_loop.execute_chat_tool,
#             app.llm.local_tools.sequential_thinking
# Last-renovated: 2026-06-19
"""
Local-provider tool loop.

vLLM is OpenAI-shaped, so stream_vllm emits the same tool_use / tool_use_delta
events the openai branch does. This loop accumulates those deltas per round,
executes each call (sequentialthinking handled in-process; every other tool via
the shared, provider-agnostic execute_chat_tool), replays the results in OpenAI
message shape, and continues until the model stops calling tools.

The `sequentialthinking` capability is auto-attached so any agentic caller can
deliberate even if it didn't pass the tool itself.

Failure-proof: returns a dict (never raises). On a provider error it returns
whatever text it has so far with `stopped` set.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _max_rounds() -> int:
    try:
        return max(1, int(os.getenv("NAT_MAX_TOOL_ROUNDS", "8")))
    except Exception:
        return 8


def _with_sequential_thinking(tools: List[Dict]) -> List[Dict]:
    """Ensure the sequentialthinking tool is available (dedup by name)."""
    from app.llm.local_tools.sequential_thinking import SEQUENTIAL_THINKING_TOOL
    names = {t.get("name") for t in (tools or [])}
    if "sequentialthinking" in names:
        return list(tools or [])
    return list(tools or []) + [SEQUENTIAL_THINKING_TOOL]


async def _execute(name: str, params: Dict[str, Any], st_state) -> str:
    if name == "sequentialthinking":
        return st_state.record(params or {})
    from app.llm.chat_tool_loop import execute_chat_tool
    return await execute_chat_tool(name, params or {})


async def run_nat_tool_loop(
    system_prompt: str,
    user_prompt: str,
    *,
    tools: List[Dict],
    enable_thinking: bool = False,
    max_tokens: int = 512,
    timeout_seconds: int = 30,
    provider: str = "vllm_local",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Nat's agentic tool loop. Returns:
        {"text": str, "tool_calls": [{name,input,result_preview}],
         "rounds": int, "stopped": str|None}
    """
    from app.llm._streaming_utils_3 import stream_llm
    from app.llm.local_tools.sequential_thinking import SequentialThinkingState

    tools = _with_sequential_thinking(tools)
    st_state = SequentialThinkingState()
    messages: List[Dict[str, Any]] = [{"role": "user", "content": user_prompt}]
    executed: List[Dict[str, Any]] = []
    final_text = ""
    stopped: Optional[str] = None
    rounds = 0
    limit = _max_rounds()

    while rounds < limit:
        rounds += 1
        # index -> {"id","name","args"}
        calls: Dict[int, Dict[str, Any]] = {}
        order: List[int] = []
        text_chunks: List[str] = []
        errored: Optional[str] = None

        async for ev in stream_llm(
            messages=messages,
            system_prompt=system_prompt,
            provider=provider,
            model=model,
            tools=tools,
            enable_reasoning=enable_thinking,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        ):
            et = ev.get("type")
            if et == "token":
                text_chunks.append(ev.get("text", "") or ev.get("content", ""))
            elif et == "tool_use":
                idx = int(ev.get("index", len(order)))
                if idx not in calls:
                    calls[idx] = {"id": ev.get("id"), "name": ev.get("name"), "args": ev.get("input_json", "") or ""}
                    order.append(idx)
                else:
                    if ev.get("id"):
                        calls[idx]["id"] = ev["id"]
                    if ev.get("name"):
                        calls[idx]["name"] = ev["name"]
                    calls[idx]["args"] += ev.get("input_json", "") or ""
            elif et == "tool_use_delta":
                idx = int(ev.get("index", 0))
                calls.setdefault(idx, {"id": None, "name": None, "args": ""})
                if idx not in order:
                    order.append(idx)
                calls[idx]["args"] += ev.get("arguments", "") or ""
            elif et == "error":
                errored = ev.get("message", "provider error")

        round_text = "".join(text_chunks).strip()
        if round_text:
            final_text = round_text

        if errored:
            stopped = f"error: {errored}"
            break

        if not order:
            # No tool calls this round — Nat is done.
            stopped = None
            break

        # Replay assistant tool_calls + results (OpenAI message shape).
        tool_calls_payload: List[Dict[str, Any]] = []
        for idx in order:
            c = calls[idx]
            tool_calls_payload.append({
                "id": c.get("id") or f"call_{idx}",
                "type": "function",
                "function": {"name": c.get("name") or "", "arguments": c.get("args") or "{}"},
            })
        messages.append({
            "role": "assistant",
            "content": round_text or None,
            "tool_calls": tool_calls_payload,
        })

        for idx in order:
            c = calls[idx]
            name = c.get("name") or ""
            try:
                params = json.loads(c.get("args") or "{}")
                if not isinstance(params, dict):
                    params = {}
            except Exception:
                params = {}
            try:
                result = await _execute(name, params, st_state)
            except Exception as exc:  # noqa: BLE001
                result = f"Tool error: {exc}"
            executed.append({"name": name, "input": params, "result_preview": str(result)[:200]})
            messages.append({
                "role": "tool",
                "tool_call_id": c.get("id") or f"call_{idx}",
                "content": str(result),
            })
    else:
        stopped = "max_rounds"

    return {
        "text": final_text,
        "tool_calls": executed,
        "rounds": rounds,
        "stopped": stopped,
        "thoughts": st_state.thoughts,
    }
