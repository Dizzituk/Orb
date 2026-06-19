# FILE: app/lifestyle/dashboard_summary.py
# Purpose: Top-level Lifestyle dashboard read model assembly.
# Called-by: app.lifestyle.service
# Depends-on: app.lifestyle.energy, app.lifestyle.energy_ledger, app.lifestyle.models, app.lifestyle.schemas
# Last-renovated: 2026-06-19
"""Top-level dashboard read assembly for the Lifestyle Engine.

This module owns the "today / now" dashboard summary. It was split out of
service.py because the old service module had grown beyond the file-size limit
and mixed weight, nutrition, workout, goals, metabolic, and dashboard concerns.
The public entrypoint remains service.get_dashboard_summary() for backwards
compatibility; service.py delegates here.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Callable, Optional

from sqlalchemy.orm import Session

from app.lifestyle.models import DailySummary, WeightEntry, WorkoutSession
from app.lifestyle.schemas import DashboardSummary


def get_dashboard_summary(
    db: Session,
    *,
    get_daily_nutrition: Callable,
    get_active_goals: Callable,
    get_metabolic_summary: Callable,
    get_weekly_activity: Callable,
    calculate_streak: Callable,
    naive_utc: Callable,
) -> DashboardSummary:
    """Assemble the live Health/Fitness overview summary."""
    latest_weight = _latest_weight(db)
    weight_7d = _weight_change_7d(db, latest_weight, naive_utc)
    today_nutr = get_daily_nutrition(db)
    engine = _energy_witness(db)

    targets = {g.goal_type: g.target_value for g in get_active_goals(db)}
    metab = _safe_metabolic_summary(get_metabolic_summary)
    weekly = get_weekly_activity(db)
    last_session = _last_workout(db)
    today_row = db.query(DailySummary).filter(DailySummary.date == date.today()).first()

    return DashboardSummary(
        current_weight_kg=latest_weight.weight_kg if latest_weight else None,
        weight_change_7d=weight_7d,
        target_weight_kg=targets.get("weight"),
        today_calories=today_nutr.total_calories,
        today_protein_g=today_nutr.total_protein_g,
        today_carbs_g=today_nutr.total_carbs_g,
        today_fat_g=today_nutr.total_fat_g,
        today_sugar_g=today_nutr.total_sugar_g,
        calories_target=engine.dynamic_target or getattr(metab, "target_calories", None) or 2200,
        protein_target=getattr(metab, "protein_g", None) or 180,
        carbs_target=getattr(metab, "carbs_g", None),
        fat_target=getattr(metab, "fat_g", None),
        sugar_limit=targets.get("sugar_limit", 25),
        activity_streak_days=calculate_streak(db),
        sessions_this_week=weekly["session_count"],
        last_activity=last_session.started_at.isoformat() if last_session else None,
        last_activity_type=last_session.activity_type if last_session else None,
        today_steps=today_row.steps if today_row else None,
        today_floors=today_row.floors if today_row else None,
        today_active_calories=_today_active_calories(today_row, engine.activity_calories),
    )


class _EnergyWitness:
    def __init__(self, activity_calories: Optional[int], dynamic_target: Optional[float]):
        self.activity_calories = activity_calories
        self.dynamic_target = dynamic_target


def _latest_weight(db: Session):
    return db.query(WeightEntry).order_by(WeightEntry.recorded_at.desc()).first()


def _weight_change_7d(db: Session, latest_weight, naive_utc: Callable) -> Optional[float]:
    if not latest_weight:
        return None
    week_ago = naive_utc(datetime.now(timezone.utc)) - timedelta(days=7)
    older = (
        db.query(WeightEntry)
        .filter(WeightEntry.recorded_at <= week_ago)
        .order_by(WeightEntry.recorded_at.desc())
        .first()
    )
    return round(latest_weight.weight_kg - older.weight_kg, 1) if older else None


def _energy_witness(db: Session) -> _EnergyWitness:
    """Read energy-engine signals without allowing dashboard 500s."""
    try:
        from app.lifestyle.energy import estimate_day_energy
        from app.lifestyle.energy_ledger import compute_today_target

        est = estimate_day_energy(db)
        activity = int(round(est["activity_kcal"])) if est.get("activity_kcal") else None
        dynamic_target = (compute_today_target(db) or {}).get("target_calories")
        return _EnergyWitness(activity, dynamic_target)
    except Exception:  # pragma: no cover - dashboard must stay available
        return _EnergyWitness(None, None)


def _safe_metabolic_summary(get_metabolic_summary: Callable):
    try:
        return get_metabolic_summary()
    except Exception:
        return None


def _last_workout(db: Session):
    return db.query(WorkoutSession).order_by(WorkoutSession.started_at.desc()).first()


def _today_active_calories(today_row, fallback: Optional[int]) -> Optional[int]:
    if today_row and today_row.active_calories:
        return today_row.active_calories
    return fallback
