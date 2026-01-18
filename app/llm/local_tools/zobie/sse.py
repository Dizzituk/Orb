# FILE: app/llm/local_tools/zobie/sse.py
"""SSE (Server-Sent Events) helper functions for zobie tools.

Extracted from zobie_tools.py for modularity.
No logic changes - exact same SSE payload format.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


def sse_token(content: str) -> str:
    """Generate SSE token event."""
    return "data: " + json.dumps({"type": "token", "content": content}) + "\n\n"


def sse_error(error: str) -> str:
    """Generate SSE error event."""
    return "data: " + json.dumps({"type": "error", "error": error}) + "\n\n"


def sse_done(
    *,
    provider: str,
    model: str,
    total_length: int = 0,
    success: bool = True,
    error: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate SSE done event."""
    payload: Dict[str, Any] = {
        "type": "done",
        "provider": provider,
        "model": model,
        "total_length": int(total_length or 0),
        "success": bool(success),
    }
    if error:
        payload["error"] = str(error)
    if meta:
        payload["meta"] = meta
    return "data: " + json.dumps(payload) + "\n\n"
