# FILE: app/llm/routing/reject_intent_handler.py
"""
Reject-intent endpoint handler.

Extracted from stream_router.py (2026-05-24) as part of the
stream_router.py refactor. The /stream/reject-intent endpoint logs
rejected-confirmation events for confidence learning — entirely
separate concern from streaming chat, just historically co-located.

Called when the user clicks 'Not what I meant' on a confirmation gate
in the frontend. The negative signal feeds the graduated confidence
system: rules that repeatedly trigger rejections eventually get
silenced via rule_suppression.py.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def handle_reject_intent(req: Any) -> dict:
    """Log a rejected confirmation event.

    Returns a small status dict the frontend can render.
    """
    try:
        from app.translation.schemas import CanonicalIntent as _CI
        from app.translation.confirmation_log import log_confirmation_event

        intent_enum = _CI(req.intent)

        # Look up the matching classifier decision so we can log rule_name
        # alongside the rejection. rule_suppression.py uses rule_name to
        # compute per-rule reject counts, which lets the system silence
        # rules that consistently misfire.
        _rejected_rule_name = _lookup_rejected_rule_name(req)

        log_confirmation_event(
            intent=intent_enum,
            user_message_excerpt=req.original_message[:200],
            confirmed=False,
            confidence=0.0,
            conversation_id=str(req.project_id),
            rule_name=_rejected_rule_name,
        )

        # Mark the matching classifier decision as rejected so the chat LLM
        # sees the negative signal in its [CLASSIFIER DECISIONS] block and
        # can answer accurately when the user asks why it fired.
        _mark_decision_rejected(req)

        logger.info(
            "[reject_intent] Rejection logged: intent=%s, message='%s'",
            req.intent, req.original_message[:50],
        )
        return {"status": "ok", "logged": True}

    except Exception as e:
        logger.warning("[reject_intent] Failed to log rejection: %s", e)
        return {"status": "error", "detail": str(e)}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _lookup_rejected_rule_name(req: Any) -> str | None:
    """Find which rule fired the now-rejected decision."""
    try:
        from app.translation.recent_decisions import get_recent_decisions
        _recent = get_recent_decisions(str(req.project_id))
        _excerpt = req.original_message[:200]
        for _d in reversed(_recent):
            if _d.message_excerpt == _excerpt:
                return _d.rule_name
        # Fallback: assume the most recent decision is the one being
        # rejected. Excerpt mismatch can happen when the frontend
        # truncates differently from us.
        if _recent:
            return _recent[-1].rule_name
        return None
    except Exception as _lookup_err:
        logger.debug("[reject_intent] rule_name lookup failed: %s", _lookup_err)
        return None


def _mark_decision_rejected(req: Any) -> None:
    """Stamp the classifier decision row as rejected so the chat LLM sees it."""
    try:
        from app.translation.recent_decisions import (
            mark_status,
            STATUS_REJECTED,
        )
        mark_status(
            conversation_id=str(req.project_id),
            status=STATUS_REJECTED,
            message_excerpt=req.original_message[:200],
        )
    except Exception as _mk_err:
        logger.debug(
            "[reject_intent] Failed to mark decision rejected: %s",
            _mk_err,
        )
