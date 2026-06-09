# FILE: app/tools/lifestyle_history_tools.py
"""
Health-history tool handler — gives Astra a range-based read of the whole daily
health series for long-arc trend analysis (steps / floors / calories / resting
HR / sleep / weight / nutrition over weeks and months).

Kept out of lifestyle_tools.py for the size budget; re-exported from there so
the Gemini executor adapters import it from the usual place. Session/dict
helpers come from lifestyle_tools (defined above its re-export line → no cycle).
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional

from app.tools.lifestyle_tools import _db_session

logger = logging.getLogger(__name__)


def _parse_iso(v) -> Optional[date]:
    if not v:
        return None
    try:
        return datetime.strptime(str(v).strip()[:10], "%Y-%m-%d").date()
    except (ValueError, AttributeError):
        return None


async def get_health_history_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Read the daily health series over a window. Pass `days` (default 30, max
    365) for a rolling window, or explicit `start_date`/`end_date` (YYYY-MM-DD).
    Returns the series plus a summary block of trends for the range.
    """
    start = _parse_iso(input_data.get("start_date"))
    end = _parse_iso(input_data.get("end_date"))
    try:
        days = int(input_data.get("days") or 30)
    except (TypeError, ValueError):
        days = 30

    try:
        from app.lifestyle.service import get_health_history
        with _db_session() as db:
            result = get_health_history(db, start_date=start, end_date=end, days=days)
    except Exception as exc:
        logger.exception("[lifestyle_tools] get_health_history failed")
        return {"ok": False, "error": str(exc)}

    return {"ok": True, **result}
