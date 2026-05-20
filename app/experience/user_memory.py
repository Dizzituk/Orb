# FILE: app/experience/user_memory.py
"""
User Memory Integration — Layer 3 of the Unified Memory System.

Bridges the existing astra_memory system with the pipeline and
conversational UI.

Section 6 of the Unified Memory System v3.0 spec.

Usage:
    # For pipeline stages (inject user preferences into prompts):
    user_context = get_user_context_for_pipeline(db, project_id)

    # For conversational recall (Weaver integration):
    user_context = get_user_context_for_conversation(db, query, project_id)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def get_user_context_for_pipeline(
    db: Session,
    project_id: int = 0,
    max_entries: int = 10,
) -> str:
    """
    Retrieve user context relevant to pipeline execution.

    Extracts from HotIndex:
    - Project context (what the user is building)
    - User preferences (code style, naming conventions)
    - Working patterns (preferred tools, languages)

    Returns formatted string for prompt injection, or empty string.
    """
    try:
        from app.astra_memory.preference_models import HotIndex

        # Get project-level context
        entries = db.query(HotIndex).filter(
            HotIndex.record_type == "project",
        ).order_by(
            HotIndex.updated_at.desc()
        ).limit(max_entries).all()

        if not entries:
            return ""

        lines = ["## USER CONTEXT"]
        for entry in entries:
            title = entry.title or ""
            one_liner = entry.one_liner or ""
            if title or one_liner:
                lines.append(f"- {title}: {one_liner}" if title else f"- {one_liner}")

        if len(lines) <= 1:
            return ""

        return "\n".join(lines)

    except Exception as e:
        logger.debug(f"[user_memory] Pipeline context retrieval failed: {e}")
        return ""


def get_user_context_for_conversation(
    db: Session,
    query: str,
    project_id: int = 0,
    max_results: int = 5,
) -> str:
    """
    Retrieve user context relevant to a conversational query.

    Uses the existing astra_memory retrieval system to find relevant
    past interactions, preferences, and project context.

    Returns formatted string for inclusion in Weaver's system prompt.

    Phase 7 audit (2026-05-02): the v3.0 spec used the wrong call signature
    for retrieve_for_query (`query=`, `project_id=`, `max_results=` — none
    of those parameters exist). The TypeError was being swallowed by the
    surrounding try/except, so this function silently returned "" forever.
    Fixed to use the actual `user_message` / `depth_override` signature, and
    to read records off `result.candidates` rather than the non-existent
    `result.records` attribute.
    """
    try:
        from app.astra_memory.retrieval import retrieve_for_query

        result = retrieve_for_query(
            db=db,
            user_message=query,
            depth_override=None,
        )

        if not result:
            return ""

        # The retrieval result exposes the selected candidates under .candidates;
        # we accept either attribute name to stay forward-compatible.
        records = (
            getattr(result, "candidates", None)
            or getattr(result, "records", None)
            or []
        )
        if not records:
            return ""

        lines = ["## RELEVANT CONTEXT FROM PAST INTERACTIONS"]
        for record in records[:max_results]:
            title = getattr(record, "title", "") or ""
            summary = (
                getattr(record, "summary", "")
                or getattr(record, "one_liner", "")
                or ""
            )
            if title and summary:
                lines.append(f"- **{title}**: {summary[:150]}")
            elif title:
                lines.append(f"- **{title}**")
            elif summary:
                lines.append(f"- {summary[:150]}")

        if len(lines) <= 1:
            return ""

        return "\n".join(lines)

    except Exception as e:
        logger.debug(f"[user_memory] Conversation context retrieval failed: {e}")
        return ""


def extract_user_facts(
    db: Session,
    message: str,
    response: str = "",
) -> List[Dict[str, Any]]:
    """
    Extract user facts from a conversation turn.

    Called after each Weaver response to identify learnable signals:
    - User preferences ("I prefer tabs over spaces")
    - Biographical facts ("I'm working on a delivery driver app")
    - Technical preferences ("Always use FastAPI, not Flask")

    Returns list of extracted facts (for storage in HotIndex).
    """
    try:
        from app.astra_memory.learning import extract_style_signals

        signals = extract_style_signals(message, response)
        return signals if signals else []

    except Exception as e:
        logger.debug(f"[user_memory] Fact extraction failed: {e}")
        return []
