# FILE: app/lifestyle/daily_view.py
# Purpose: Per-day read views for the Overview and Activity tabs.
# Called-by: app.lifestyle.router
# Depends-on: app.lifestyle, app.lifestyle.models, app.lifestyle.schemas, app.lifestyle.service
# Last-renovated: 2026-06-11
"""
Per-day read views for the Overview and Activity tabs.

Lets those tabs flick through past (and prepped future) days the same way the
Nutrition and Fitness tabs already do. Everything here is a read assembled from
existing rows — DailySummary holds each day's macro totals AND wearable metrics,
so a past day needs no recomputation.

Kept out of service.py purely for the file-size budget: this module owns the
"one specific day" reads; service.py owns "today / now".

Design note — what follows the selected day vs what stays as context:
  • day-specific (follow the chosen day): intake macros, wearable movement,
    bodyweight.
  • rolling context (always current): calorie/protein/carb/fat targets, weight
    target, streak, sessions-this-week, last activity.
  To guarantee the targets + context are identical to the live dashboard, we
  build on top of service.get_dashboard_summary() and override only the
  day-specific fields. One source of truth, no drift.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from app.lifestyle.models import WeightEntry, WorkoutSession, DailySummary
from app.lifestyle.schemas import DashboardSummary, WorkoutSessionOut
from app.lifestyle import service


def _day_bounds(target: date):
    start = datetime.combine(target, datetime.min.time()).replace(tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


def get_day_workouts(db: Session, target: date) -> List[WorkoutSessionOut]:
    """All workout sessions that started on `target` (newest first)."""
    start, end = _day_bounds(target)
    rows = (
        db.query(WorkoutSession)
        .filter(WorkoutSession.started_at >= start, WorkoutSession.started_at < end)
        .order_by(WorkoutSession.started_at.desc())
        .all()
    )
    return [service._workout_to_out(s) for s in rows]


def _weight_on_or_before(db: Session, target: date) -> Optional[float]:
    """Most recent weight recorded on or before the end of `target`."""
    _, end = _day_bounds(target)
    row = (
        db.query(WeightEntry)
        .filter(WeightEntry.recorded_at < end)
        .order_by(WeightEntry.recorded_at.desc())
        .first()
    )
    return row.weight_kg if row else None


def get_day_dashboard(db: Session, target: date) -> DashboardSummary:
    """A DashboardSummary for one specific day.

    Targets + rolling context come straight from the live dashboard (current
    values); only the day-specific fields — intake macros, wearable movement,
    bodyweight — are overridden to reflect `target`.
    """
    summary = service.get_dashboard_summary(db)

    nutr = service.get_daily_nutrition(db, target)
    summary.today_calories = nutr.total_calories
    summary.today_protein_g = nutr.total_protein_g
    summary.today_carbs_g = nutr.total_carbs_g
    summary.today_fat_g = nutr.total_fat_g
    summary.today_sugar_g = nutr.total_sugar_g

    row = db.query(DailySummary).filter(DailySummary.date == target).first()
    summary.today_steps = row.steps if row else None
    summary.today_floors = row.floors if row else None
    summary.today_active_calories = row.active_calories if row else None

    summary.current_weight_kg = _weight_on_or_before(db, target)
    summary.weight_change_7d = None  # not meaningful for a single navigated day

    return summary
