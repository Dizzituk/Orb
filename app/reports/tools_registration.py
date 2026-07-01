# FILE: app/reports/tools_registration.py
# Purpose: The surface-aware "show me the report" chat tool.
# Called-by: app.tools.registry._register_defaults
# Depends-on: app.reports.renderer, app.reports.surface, app.db
# Last-renovated: 2026-07-01
"""
One tool covers every watcher: bridge-originated turns ship the rendered
HTML to the phone as a document artifact; desktop turns open it on the
reports window. Voice-reachable on both surfaces by construction.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def show_watcher_report_handler(input_data: dict, context=None) -> dict:
    from app.db import get_db_session
    from app.reports.renderer import render_watcher_report
    from app.reports.surface import deliver_report

    watcher_id = str(input_data.get("watcher") or "").strip().lower().replace("-", "_")
    days = 30
    try:
        days = max(1, min(int(input_data.get("days", 30)), 365))
    except Exception:
        pass

    db = get_db_session()
    try:
        report = render_watcher_report(db, watcher_id, days=days)
    finally:
        db.close()

    if report is None:
        from app.watchers.framework import list_watchers

        known = sorted(s.watcher_id for s in list_watchers())
        return {"ok": False, "error": f"unknown watcher {watcher_id!r}", "known_watchers": known}

    out = await deliver_report(report["filename"], report["title"])
    out["data_through"] = report["data_through"]
    return out


def register_report_tools() -> None:
    """Called once from registry._register_defaults at module load."""
    from app.tools.registry import ToolDefinition, register_tool

    register_tool(ToolDefinition(
        name="show_watcher_report",
        version="v1",
        description=(
            "Render a styled visual report (line chart + freshness line + dated "
            "table) from a price watcher's ledger and put it in front of the "
            "user: on the desktop it opens on the reports display window; from "
            "the phone it is attached to the reply as a document. Use when the "
            "user asks to SEE or SHOW data — 'show me the Portugal land "
            "prices', 'put the hardware prices up', 'pull up the price report'. "
            "watcher is 'portugal_land' or 'hardware'. For a purely spoken "
            "answer use the watcher's own get_* tool instead."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "watcher": {
                    "type": "string",
                    "description": "Watcher id: portugal_land | hardware",
                },
                "days": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 365,
                    "default": 30,
                    "description": "History window for the chart and table.",
                },
            },
            "required": ["watcher"],
        },
        output_schema={"type": "object"},
        handler=show_watcher_report_handler,
    ))
