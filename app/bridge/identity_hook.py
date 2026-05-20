# FILE: app/bridge/identity_hook.py
"""
Thin, fire-and-forget hook called from Bridge chat handlers to scan
incoming user messages for identity facts and persist to the Tier 1
identity store.

Keeps the Bridge handlers clean — one-line call, all logic here,
all failures swallowed + logged.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def capture_from_bridge_message(message: str, source_label: str = "bridge") -> Dict[str, Any]:
    """
    Scan a Bridge user message for identity facts and persist.

    Never raises. Returns the capture result dict (possibly empty)
    for optional logging/telemetry by the caller.
    """
    if not message or len(message.strip()) < 4:
        return {"captured": {}, "written": [], "skipped": []}
    try:
        from app.self_model.identity_capture import capture_and_persist
        result = capture_and_persist(message, source=source_label)
        if result.get("written"):
            logger.info(
                "[bridge.identity_hook] captured %d fact(s): %s",
                len(result["written"]),
                ", ".join(w.get("field", "?") for w in result["written"]),
            )
        return result
    except Exception as exc:
        logger.warning("[bridge.identity_hook] failed: %s", exc)
        return {"captured": {}, "written": [], "skipped": [{"reason": str(exc)}]}


def capture_fragment_from_bridge(message: str, session_id=None, project_id=None) -> dict:
    """
    Phase 7: run the fragment-capture pipeline for a Bridge user message.
    Embeds, classifies, appends, clusters. Never raises.
    """
    if not message or len(message.strip()) < 8:
        return {"skipped": "too_short"}
    try:
        from app.self_model.fragments.capture import capture_fragment
        return capture_fragment(
            message, source="bridge",
            session_id=session_id, project_id=project_id,
        )
    except Exception as exc:
        logger.warning("[bridge.fragment_capture] failed: %s", exc)
        return {"skipped": str(exc)}