# Purpose: stream handler utils 2
# Called-by: app.llm.critical_pipeline.stream_handler
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
from __future__ import annotations
import json


def _sse(event_type: str, content: str = "", **extra) -> str:
    payload = {"type": event_type}
    if content:
        payload["content"] = content
    payload.update(extra)
    return "data: " + json.dumps(payload) + "\n\n"
