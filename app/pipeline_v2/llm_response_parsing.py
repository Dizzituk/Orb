# FILE: app/pipeline_v2/llm_response_parsing.py
# Purpose: Response parsing for the Agentic Builder tool loop.
# Called-by: app.pipeline_v2.llm_tools
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Response parsing for the Agentic Builder tool loop.

Extracted from llm_tools.py on 2026-06-09 (file-size discipline: llm_tools
hit the 30KB ceiling when the Responses API path landed).

Handles two response shapes:
  1. OpenAI chat-completions SDK objects (hasattr "choices")
  2. The adapted dict produced by llm_responses_api.call_openai_responses:
     {"responses_api": True, "text", "tool_calls", "_raw_items"}

The canonical conversation history stays in chat-completions format;
_make_assistant_message converts either shape into that format. Responses
turns additionally carry "_responses_raw_items" (private key, stripped
before any chat-completions call) so reasoning items replay faithfully.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def _extract_tool_calls(response: Any, provider: str) -> List[Dict]:
    """Extract tool calls from API response."""
    if isinstance(response, dict) and response.get("responses_api"):
        calls = []
        for tc in response.get("tool_calls") or []:
            try:
                args = json.loads(tc.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": tc.get("id", ""), "name": tc.get("name", ""), "args": args})
        return calls
    if provider == "openai" and hasattr(response, "choices"):
        msg = response.choices[0].message
        if msg.tool_calls:
            calls = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, AttributeError):
                    args = {}
                calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": args,
                })
            return calls
    return []


def _extract_text(response: Any, provider: str) -> Optional[str]:
    """Extract text content from API response."""
    if provider == "openai" and hasattr(response, "choices"):
        return response.choices[0].message.content
    if isinstance(response, dict) and "text" in response:
        return response["text"]
    return None


def _make_assistant_message(response: Any, provider: str) -> Dict:
    """Convert API response to an assistant message for the conversation."""
    if isinstance(response, dict) and response.get("responses_api"):
        result: Dict[str, Any] = {"role": "assistant", "content": response.get("text") or ""}
        if response.get("tool_calls"):
            result["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {"name": tc.get("name", ""), "arguments": tc.get("arguments") or "{}"},
                }
                for tc in response["tool_calls"]
            ]
        if response.get("_raw_items"):
            # Replayed verbatim by llm_responses_api on the next turn so the
            # model gets its reasoning items back. Stripped for chat API.
            result["_responses_raw_items"] = response["_raw_items"]
        return result
    if provider == "openai" and hasattr(response, "choices"):
        msg = response.choices[0].message
        result = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return result
    text = _extract_text(response, provider) or ""
    return {"role": "assistant", "content": text}
