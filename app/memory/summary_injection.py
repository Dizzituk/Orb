# FILE: app/memory/summary_injection.py
"""
Shared utility for injecting conversation summaries into LLM context.

This is the SINGLE place where the summary block format is defined.
All streaming paths (OpenAI, Anthropic, Gemini) and prompt builders
import from here — no copy-pasting the format.

Part of CONV-MEMORY-001: Phase 3 (Summary Injection).
"""

import logging
from typing import Optional, Dict, Any

from sqlalchemy.orm import Session as DbSession

logger = logging.getLogger(__name__)


def get_summary_for_injection(
    db: DbSession,
    project_id: int,
) -> Optional[Dict[str, Any]]:
    """
    Load the latest conversation summary for a project.

    Returns the summary_json dict, or None if no summary exists.
    Only returns summaries from active/idle sessions (not archived).
    """
    try:
        from app.memory.conversation_service import (
            get_latest_summary_for_project,
        )
        summary = get_latest_summary_for_project(db, project_id)
        if summary and summary.summary_json:
            return summary.summary_json
    except Exception as e:
        logger.debug("[summary_inject] Failed to load summary: %s", e)

    return None


def format_summary_block(summary_json: Dict[str, Any]) -> str:
    """
    Format a summary dict into the injection block for LLM context.

    This is THE canonical format. Every streaming path uses this.

    Output format:
        [CONVERSATION_SUMMARY]
        Current topic: ...
        Key decisions: ...
        Open items: ...
        Technical context: ...
        Attachments: ...
        User preferences: ...
        Models involved: ...
        [/CONVERSATION_SUMMARY]
    """
    lines = ["[CONVERSATION_SUMMARY]"]

    topic = summary_json.get("current_topic", "")
    if topic:
        lines.append(f"Current topic: {topic}")

    decisions = summary_json.get("key_decisions", [])
    if decisions:
        lines.append("Key decisions: " + "; ".join(decisions))

    open_items = summary_json.get("open_items", [])
    if open_items:
        lines.append("Open items: " + "; ".join(open_items))

    tech = summary_json.get("technical_context", "")
    if tech:
        lines.append(f"Technical context: {tech}")

    attachments = summary_json.get("attachments_described", [])
    if attachments:
        lines.append("Attachments: " + "; ".join(attachments))

    prefs = summary_json.get("user_preferences_expressed", [])
    if prefs:
        lines.append("User preferences: " + "; ".join(prefs))

    models = summary_json.get("models_involved", [])
    if models:
        lines.append("Models involved: " + ", ".join(models))

    lines.append("[/CONVERSATION_SUMMARY]")

    return "\n".join(lines)


def build_summary_context(
    db: DbSession,
    project_id: int,
) -> str:
    """
    One-call convenience: load summary + format it.

    Returns the formatted summary block, or empty string if none.
    This is the function most callers should use.
    """
    summary_json = get_summary_for_injection(db, project_id)
    if not summary_json:
        return ""

    block = format_summary_block(summary_json)
    logger.debug(
        "[summary_inject] Injecting summary for project %d (%d chars)",
        project_id, len(block),
    )
    return block
