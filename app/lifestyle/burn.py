# FILE: app/lifestyle/burn.py
# Purpose: Daily-burn signal for the metabolic engine.
# Called-by: app.lifestyle.service
# Depends-on: app.lifestyle.models
# Last-renovated: 2026-06-11
"""
Daily-burn signal for the metabolic engine.

Turns the wearable + workout history into a single trustworthy "active calories
per day" figure that feeds TDEE (see metabolic.build_plan). Kept separate from
service.py so the maths is isolated and that file stays within its size budget.

Design:
  • Per day, the active-calorie figure is the Garmin / Health-Connect daily
    active_calories when present. If a day has none but DOES have logged
    workouts, we fall back to the sum of those workouts' calories_burned — so a
    surf logged by hand on a day the watch missed still counts. We never
    double-count, because the workout fallback only applies when Garmin gave us
    nothing for that day.
  • We average across the days in the window that have ANY signal. A rest day
    with nothing logged is absence of evidence, not evidence of a low burn, so
    it is excluded rather than dragging the mean toward zero.
  • The average is only "trusted" once we have at least MIN_DAYS of signal.
    Below that the engine stays on the estimated activity-factor TDEE — one or
    two early days should not swing the daily burn around. As more Garmin data
    lands, the figure both switches to measured and sharpens.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.lifestyle.models import DailySummary, WorkoutSession

# Rolling look-back window, and the floor of days before a measured average is
# trusted over the estimate.
WINDOW_DAYS = 28
MIN_DAYS = 4


@dataclass
class BurnSignal:
    avg_active_calories: Optional[float]  # mean active kcal/day across days with signal
    days_used: int                        # how many days contributed
    window_days: int                      # the look-back window
    trusted: bool                         # days_used >= MIN_DAYS (and avg > 0)
    source_label: str                     # human label for MetabolicSummary.tdee_source


def _workout_calories_by_day(db: Session, start: date, end: date) -> dict:
    """Sum of logged WorkoutSession.calories_burned per day in [start, end]."""
    win_start = datetime.combine(start, datetime.min.time()).replace(tzinfo=timezone.utc)
    win_end = datetime.combine(end + timedelta(days=1), datetime.min.time()).replace(tzinfo=timezone.utc)
    rows = (
        db.query(WorkoutSession.started_at, WorkoutSession.calories_burned)
        .filter(
            WorkoutSession.started_at >= win_start,
            WorkoutSession.started_at < win_end,
            WorkoutSession.calories_burned.isnot(None),
        )
        .all()
    )
    out: dict = {}
    for started_at, cals in rows:
        if started_at is None or cals is None:
            continue
        d = started_at.date()
        out[d] = out.get(d, 0.0) + float(cals)
    return out


def compute_burn_signal(db: Session, window_days: int = WINDOW_DAYS,
                        min_days: int = MIN_DAYS) -> BurnSignal:
    """Build the trusted daily-burn signal from wearable + workout history."""
    today = date.today()
    start = today - timedelta(days=window_days)

    # Primary signal: Garmin / Health-Connect daily active calories.
    summary_rows = (
        db.query(DailySummary.date, DailySummary.active_calories)
        .filter(DailySummary.date >= start, DailySummary.active_calories.isnot(None))
        .all()
    )
    per_day: dict = {d: float(ac) for d, ac in summary_rows if ac is not None}

    # Fallback: days the watch missed but a workout was logged by hand.
    for d, cals in _workout_calories_by_day(db, start, today).items():
        if d not in per_day and cals > 0:
            per_day[d] = cals

    days_used = len(per_day)
    avg = round(sum(per_day.values()) / days_used, 0) if days_used else None
    trusted = bool(days_used >= min_days and avg and avg > 0)

    if trusted:
        label = f"measured · {days_used}-day avg"
    elif days_used > 0:
        label = f"estimated — building ({days_used}/{min_days} days)"
    else:
        label = "estimated — no watch data yet"

    return BurnSignal(
        avg_active_calories=avg,
        days_used=days_used,
        window_days=window_days,
        trusted=trusted,
        source_label=label,
    )
