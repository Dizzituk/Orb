from __future__ import annotations
import json
import time
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class ProviderConfig:
    provider_id: str
    display_name: str
    env_key_name: str

def _now_ms() -> int:
    return int(time.time() * 1000)

def _safe_json(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

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

def _is_reasoning_model(model_id: str) -> bool:
    """Check if model supports the reasoning parameter (GPT-5.x, o3, o4)."""
    m = (model_id or "").strip().lower()
    return m.startswith("gpt-5") or m.startswith("o3") or m.startswith("o4")

def _messages_to_prompt(messages: List[dict], system_prompt: Optional[str]) -> str:
    """Convert messages array to a single prompt string for completions API."""
    parts: List[str] = []
    if system_prompt:
        parts.append(f"System: {system_prompt}")
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    # Add prompt for assistant response
    parts.append("Assistant:")
    return "\n\n".join(parts)

async def _execute_tool_by_name(name: str, args: dict, context: Optional[dict]) -> dict:
    from app.tools.registry import execute_tool_async

    resp = await execute_tool_async(name, "v1", args, context=context)
    return resp.to_dict()
