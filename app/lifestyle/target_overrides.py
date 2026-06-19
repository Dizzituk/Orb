# FILE: app/lifestyle/target_overrides.py
# Purpose: Resolve dashboard macro targets from computed metabolic plan plus user overrides.
# Called-by: app.lifestyle.router
# Depends-on: app.lifestyle.metabolic, app.lifestyle.models, app.lifestyle.schemas
# Last-renovated: 2026-06-18
"""Resolve Health dashboard nutrition targets.

The metabolic engine remains the recommendation source, but active goals are
intentional user overrides. This lets chat/LLM tools and the UI set "2500 kcal"
or "220g protein" and have Today's Intake reflect that immediately.
"""
from __future__ import annotations

from typing import Any, Optional

from sqlalchemy.orm import Session

from app.lifestyle import metabolic
from app.lifestyle.models import LifestyleGoal
from app.lifestyle.schemas import GoalOut

_GOAL_FAMILY_ALIASES = {
    "calorie": "calories",
    "calories": "calories",
    "daily_calories": "calories",
    "kcal": "calories",
    "protein": "protein",
    "protein_g": "protein",
    "carb": "carbs",
    "carbs": "carbs",
    "carbohydrate": "carbs",
    "carbohydrates": "carbs",
    "carbs_g": "carbs",
    "fat": "fat",
    "fat_g": "fat",
    "sugar": "sugar_limit",
    "sugar_g": "sugar_limit",
    "sugar_limit": "sugar_limit",
    "weight": "weight",
    "target_weight": "weight",
}


def canonical_goal_type(goal_type: Optional[str]) -> Optional[str]:
    """Return the canonical family key for a lifestyle goal type."""
    key = (goal_type or "").strip().lower()
    return _GOAL_FAMILY_ALIASES.get(key, key) if key else None


def set_goal_override(
    db: Session,
    goal_type: str,
    target_value: float,
    unit: str,
    notes: Optional[str] = None,
) -> GoalOut:
    """Persist a canonical goal and deactivate old aliases in the same family."""
    canonical_type = canonical_goal_type(goal_type) or goal_type

    active_goals = (
        db.query(LifestyleGoal)
        .filter(LifestyleGoal.is_active == True)  # noqa: E712 - SQLAlchemy expression
        .all()
    )
    for goal in active_goals:
        if canonical_goal_type(goal.goal_type) == canonical_type:
            goal.is_active = False

    goal = LifestyleGoal(
        goal_type=canonical_type,
        target_value=target_value,
        unit=unit,
        notes=notes,
    )
    db.add(goal)
    db.commit()
    db.refresh(goal)
    return GoalOut(
        id=goal.id,
        goal_type=goal.goal_type,
        target_value=goal.target_value,
        unit=goal.unit,
        is_active=goal.is_active,
        notes=goal.notes,
    )


def apply_target_overrides(db: Session, summary: Any) -> Any:
    """Return a dashboard summary with active goal overrides applied."""
    targets = resolve_dashboard_targets(
        db,
        dynamic_calorie_target=getattr(summary, "calories_target", None),
        default_calories=getattr(summary, "calories_target", 2200) or 2200,
        default_protein_g=getattr(summary, "protein_target", 180) or 180,
        default_sugar_g=getattr(summary, "sugar_limit", 25) or 25,
    )
    summary.calories_target = targets["calories_target"] or summary.calories_target
    summary.protein_target = targets["protein_target"] or summary.protein_target
    summary.carbs_target = targets["carbs_target"]
    summary.fat_target = targets["fat_target"]
    summary.sugar_limit = targets["sugar_limit"] or summary.sugar_limit
    summary.target_weight_kg = targets["target_weight_kg"] or summary.target_weight_kg
    return summary


def resolve_dashboard_targets(
    db: Session,
    *,
    metabolic_summary: Any = None,
    dynamic_calorie_target: Optional[float] = None,
    default_calories: float = 2200,
    default_protein_g: float = 180,
    default_sugar_g: float = 25,
) -> dict:
    """Return final calories/protein/carbs/fat/sugar targets for the dashboard."""
    goals = _active_goal_values(db)

    metabolic_calories = _num(getattr(metabolic_summary, "target_calories", None))
    metabolic_protein = _num(getattr(metabolic_summary, "protein_g", None))
    metabolic_carbs = _num(getattr(metabolic_summary, "carbs_g", None))
    metabolic_fat = _num(getattr(metabolic_summary, "fat_g", None))

    calories = _first_number(
        goals.get("calories"),
        dynamic_calorie_target,
        metabolic_calories,
        default_calories,
    )
    protein = _first_number(goals.get("protein"), metabolic_protein, default_protein_g)

    recalculated = metabolic.recommend_macros(calories, protein)
    carbs = _first_number(goals.get("carbs"), recalculated.get("carbs_g"), metabolic_carbs)
    fat = _first_number(goals.get("fat"), recalculated.get("fat_g"), metabolic_fat)
    sugar = _first_number(goals.get("sugar_limit"), default_sugar_g)

    return {
        "goals": goals,
        "calories_target": calories,
        "protein_target": protein,
        "carbs_target": carbs,
        "fat_target": fat,
        "sugar_limit": sugar,
        "target_weight_kg": goals.get("weight"),
    }


def _active_goal_values(db: Session) -> dict[str, float]:
    rows = (
        db.query(LifestyleGoal)
        .filter(LifestyleGoal.is_active == True)  # noqa: E712 - SQLAlchemy expression
        .all()
    )
    values: dict[str, float] = {}
    for row in rows:
        key = canonical_goal_type(row.goal_type)
        value = _num(row.target_value)
        if key and value is not None:
            values[key] = value
    return values


def _first_number(*values: Any) -> Optional[float]:
    for value in values:
        num = _num(value)
        if num is not None:
            return num
    return None


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
