# FILE: app/project_registry/chat_injection.py
# Purpose: Chat routing integration for project staleness detection.
# Called-by: app.llm.stream_router
# Depends-on: app.project_registry.staleness
# Last-renovated: 2026-06-11
"""
Chat routing integration for project staleness detection.

Called during normal chat flow to check if the user is discussing
a registered project with a stale architecture scan. If so, injects
a staleness warning and confirmation button into the response stream.

v1.0 (2026-03): Initial implementation.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def get_staleness_sse_events(
    db: Session,
    message: str,
) -> Optional[list[str]]:
    """
    Check if message references a project with stale scan.

    Returns a list of SSE event strings to inject before the main
    response, or None if no staleness detected.

    The events include:
    1. A token event with the staleness warning text
    2. A confirmation_required event with yes/no gate
    """
    try:
        from .staleness import check_and_suggest_rescan

        result = check_and_suggest_rescan(db, message)
        if not result:
            return None

        events = []

        # Warning text
        warning = result["message"]
        events.append(
            "data: " + json.dumps({
                "type": "token",
                "content": f"\n> **Note:** {warning}\n\n",
            }) + "\n\n"
        )

        # Confirmation gate
        events.append(
            "data: " + json.dumps({
                "type": "confirmation_required",
                "intent": "scan_project",
                "extracted_query": result["slug"],
                "message": warning,
            }) + "\n\n"
        )

        return events

    except Exception as e:
        logger.warning("[chat_injection] Staleness check failed: %s", e)
        return None


def inject_staleness_into_stream(
    original_stream,
    db: Session,
    message: str,
):
    """
    Wrap a stream generator to prepend staleness warnings.

    If a staleness warning is detected, the warning events are
    yielded before the original stream starts. Otherwise, the
    original stream passes through unchanged.
    """
    events = get_staleness_sse_events(db, message)

    async def _wrapped():
        if events:
            for event in events:
                yield event
        async for chunk in original_stream:
            yield chunk

    if events:
        return _wrapped()
    return original_stream