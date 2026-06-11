# FILE: app/tools/lifestyle_nutrition_tools.py
"""
Nutrition tool handlers that don't fit in lifestyle_tools.py (kept under the
size ceiling). Currently: meal-prep splitting — divide a batch cook's macros
across N future days and write a portion into each.

Re-exported from lifestyle_tools so the Gemini executor adapters can import it
from the usual place. Session/dict helpers come from lifestyle_tools (defined
above its re-export line, so no import cycle).
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Optional

from app.tools.lifestyle_tools import _db_session, _to_dict

logger = logging.getLogger(__name__)


def _parse_iso_date(v) -> Optional[date]:
    if not v:
        return None
    try:
        return datetime.strptime(str(v).strip()[:10], "%Y-%m-%d").date()
    except (ValueError, AttributeError):
        return None


async def prep_split_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Split a batch cook's total macros evenly across N days, writing a portion
    into each day's diary. Defaults to starting tomorrow.

    Args: description, days, total {calories,protein_g,carbs_g,fat_g[,sugar_g]}
    (or the macros passed flat), optional start_date (YYYY-MM-DD), meal_type.
    """
    description = str(input_data.get("description") or "").strip()
    try:
        days = int(input_data.get("days"))
    except (TypeError, ValueError):
        return {"ok": False, "error": "days must be an integer (how many days to split over)"}

    # Accept either a nested `total` object or flat macro keys.
    total = input_data.get("total")
    if not isinstance(total, dict):
        total = {k: input_data.get(k) for k in
                 ("calories", "protein_g", "carbs_g", "fat_g", "sugar_g")}

    start = _parse_iso_date(input_data.get("start_date"))
    meal_type = str(input_data.get("meal_type") or "meal").strip() or "meal"

    try:
        from app.lifestyle.service import prep_split
        with _db_session() as db:
            r = prep_split(db, description, total, days, start_date=start, meal_type=meal_type)
    except Exception as exc:
        logger.exception("[lifestyle_tools] prep_split failed")
        return {"ok": False, "error": str(exc)}

    if not r.get("ok"):
        miss = r.get("missing") or []
        return {
            "ok": False, "needs": miss,
            "error": (
                "NOT split — need the whole batch's " + ", ".join(miss) + ". Give the "
                "TOTAL macros for everything you cooked (calories, protein, carbs, fat) "
                "plus a description and how many days to split over. Ask the user or "
                "web_search to estimate the batch total, then call prep_split once."
            ),
        }
    return {
        "ok": True,
        "days_written": r.get("n"),
        "start_date": r.get("start_date"),
        "per_day": r.get("per_day"),
        "dates": [d.get("date") for d in r.get("days", [])],
    }


async def copy_nutrition_day_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Copy one day's food diary onto another day (2026-06-11).

    Args: source_date, target_date — each YYYY-MM-DD or
    yesterday/today/tomorrow/weekday name. Defaults: yesterday -> today.
    Idempotent: items already on the target day are skipped.
    """
    from app.lifestyle.nutrition_copy import copy_day_nutrition, resolve_day_word

    try:
        source_day = resolve_day_word(input_data.get("source_date"), default="yesterday")
        target_day = resolve_day_word(
            input_data.get("target_date"), default="today", future_bias=True,
        )
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    try:
        with _db_session() as db:
            result = copy_day_nutrition(db, source_day, target_day)
    except Exception as exc:
        logger.exception("[lifestyle_tools] copy_nutrition_day failed")
        return {"ok": False, "error": str(exc)}
    return result
