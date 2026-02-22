from __future__ import annotations
import json


def _sse(event_type: str, content: str = "", **extra) -> str:
    payload = {"type": event_type}
    if content:
        payload["content"] = content
    payload.update(extra)
    return "data: " + json.dumps(payload) + "\n\n"
