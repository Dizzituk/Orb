# FILE: app/idle/tools_registration.py
# Purpose: Chat tool over the idle-task ledger ("what have you been working on").
# Called-by: app.tools.registry._register_defaults
# Depends-on: app.idle.ledger, app.idle.governor, app.db
# Last-renovated: 2026-07-01
"""
One voice-reachable tool, works from desktop and Bridge alike (tools execute
backend-side): reads recent completions/failures/skips plus whatever is
queued or paused, and the governor's live idle status.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _fmt_when(dt) -> str:
    try:
        return dt.isoformat(timespec="minutes") if dt else ""
    except Exception:
        return str(dt or "")


async def get_background_task_log_handler(input_data: dict, context=None) -> dict:
    from app.db import get_db_session
    from app.idle import ledger
    from app.idle.governor import get_governor

    limit = 10
    try:
        limit = max(1, min(int(input_data.get("limit", 10)), 50))
    except Exception:
        pass

    db = get_db_session()
    try:
        recent = [
            {
                "task": r.task_type,
                "status": r.status,
                "finished_at": _fmt_when(r.completed_at),
                "summary": (r.result_summary or "")[:400],
                "coverage": (r.coverage or "")[:200],
                "error": (r.error or "")[:200],
            }
            for r in ledger.recent_terminal(db, limit=limit)
        ]
        open_items = [
            {
                "task": r.task_type,
                "status": r.status,
                "queued_at": _fmt_when(r.queued_at),
            }
            for r in ledger.open_tasks(db, limit=20)
        ]
    finally:
        db.close()

    return {
        "ok": True,
        "recent": recent,
        "queued_or_paused": open_items,
        "governor": get_governor().status(),
    }


def register_idle_tools() -> None:
    """Called once from registry._register_defaults at module load."""
    from app.tools.registry import ToolDefinition, register_tool

    register_tool(ToolDefinition(
        name="get_background_task_log",
        version="v1",
        description=(
            "Read ASTRA's idle-time background work ledger: recently completed "
            "background tasks (repo map refreshes, price-watcher scrapes, deep "
            "research digs) with their result summaries, anything still queued "
            "or paused, and whether the idle governor is currently draining. "
            "Use when the user asks what you have been working on, what you did "
            "while they were away, or about background/idle task progress."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                    "description": "How many recent completions to return.",
                },
            },
            "required": [],
        },
        output_schema={"type": "object"},
        handler=get_background_task_log_handler,
    ))
