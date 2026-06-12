# FILE: app/lifestyle/history.py
# Purpose: Long-arc health history reads — the daily series Astra reads across weeks and
# Called-by: app.lifestyle.service
# Depends-on: app.lifestyle.models
# Last-renovated: 2026-06-11
"""
Long-arc health history reads — the daily series Astra reads across weeks and
months to spot trends ("resting HR drifted down this month, weight fell, here's
the training volume").

Split out of service.py to stay under the file-size budget; re-exported from
service.py for back-compat. Reads the DailySummary roll-up (one row per day),
which already carries weight, nutrition totals, activity, and the wearable
metrics (steps / floors / active calories / resting HR / sleep). Nothing is
thrown away day to day, so this is the continuous record for trend analysis.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session

from app.lifestyle.models import DailySummary

logger = logging.getLogger(__name__)


def _avg(vals: List[float]) -> Optional[float]:
    nums = [v for v in vals if v is not None]
    return round(sum(nums) / len(nums), 1) if nums else None


def _day_to_dict(s: DailySummary) -> Dict[str, Any]:
    return {
        "date": s.date.isoformat(),
        "weight_kg": s.weight_kg,
        "calories": round(s.total_calories) if s.total_calories else (0 if s.total_calories == 0 else None),
        "protein_g": round(s.total_protein_g) if s.total_protein_g else None,
        "carbs_g": round(s.total_carbs_g) if s.total_carbs_g else None,
        "fat_g": round(s.total_fat_g) if s.total_fat_g else None,
        "sugar_g": round(s.total_sugar_g) if s.total_sugar_g else None,
        "activity_mins": s.activity_mins or 0,
        "activity_types": s.activity_types,
        "steps": s.steps,
        "floors": s.floors,
        "active_calories": round(s.active_calories) if s.active_calories else None,
        "resting_hr": s.resting_hr,
        "sleep_minutes": s.sleep_minutes,
    }


def get_health_history(
    db: Session,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    days: int = 30,
) -> Dict[str, Any]:
    """Return the daily health series over a date range, plus a summary block
    of trends. Either pass start_date/end_date explicitly, or a `days` window
    counting back from today (or from end_date). days is capped at 365.

    Returns: {start, end, day_count, summary, days:[...]} where each day carries
    weight, nutrition totals, activity, steps, floors, active calories, resting
    HR and sleep. Days with no record are simply absent (gaps preserved).
    """
    days = max(1, min(365, int(days)))
    end = end_date or date.today()
    start = start_date or (end - timedelta(days=days - 1))
    if start > end:
        start, end = end, start

    rows = (
        db.query(DailySummary)
        .filter(DailySummary.date >= start, DailySummary.date <= end)
        .order_by(DailySummary.date.asc())
        .all()
    )

    series = [_day_to_dict(s) for s in rows]

    # Weight delta across the window (first vs last non-null weight)
    weights = [(s.date, s.weight_kg) for s in rows if s.weight_kg is not None]
    weight_start = weights[0][1] if weights else None
    weight_end = weights[-1][1] if weights else None
    weight_delta = round(weight_end - weight_start, 1) if (weight_start is not None and weight_end is not None) else None

    # Training days = any logged activity that day
    training_days = sum(1 for s in rows if (s.activity_mins or 0) > 0 or s.activity_types)
    total_steps = sum(s.steps for s in rows if s.steps is not None) or None

    summary = {
        "days_with_data": len(rows),
        "avg_resting_hr": _avg([s.resting_hr for s in rows]),
        "avg_sleep_minutes": _avg([s.sleep_minutes for s in rows]),
        "avg_daily_steps": _avg([float(s.steps) for s in rows if s.steps is not None]),
        "total_steps": total_steps,
        "total_floors": sum(s.floors for s in rows if s.floors is not None) or None,
        "avg_active_calories": _avg([s.active_calories for s in rows]),
        "avg_daily_calories": _avg([s.total_calories for s in rows if s.total_calories]),
        "avg_daily_protein_g": _avg([s.total_protein_g for s in rows if s.total_protein_g]),
        "weight_start_kg": weight_start,
        "weight_end_kg": weight_end,
        "weight_delta_kg": weight_delta,
        "training_days": training_days,
    }

    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "day_count": (end - start).days + 1,
        "summary": summary,
        "days": series,
    }
