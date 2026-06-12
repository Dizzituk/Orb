# FILE: app/lifestyle/seed.py
# Purpose: Seed default lifestyle data on startup.
# Called-by: main
# Depends-on: app.lifestyle.models
# Last-renovated: 2026-06-11
"""
Seed default lifestyle data on startup.

Seeds:
- Default goals: weight target, sugar limit, activity minimum.

NOTE: calorie + protein targets are deliberately NOT seeded as goals. They are
computed by the metabolic engine (app/lifestyle/metabolic.py) from BMR / TDEE /
weight goal and are the single source of truth shown on BOTH the Overview and
Plan tabs. They used to be seeded as fixed goals (2200 kcal / 180 g), which then
shadowed the computed values on the dashboard and made the two tabs disagree
(Overview showed the stale 2200, Plan showed the real computed figure).

This module also runs a one-off, idempotent cleanup that deactivates any legacy
seeded calorie/protein goals left in existing databases, so they stop overriding
the engine. Both steps are safe to run on every startup.

Only creates default goals if no goals exist yet (idempotent).
"""
import logging
from sqlalchemy.orm import Session

from app.lifestyle.models import LifestyleGoal

logger = logging.getLogger(__name__)

# Default goals. Calorie + protein targets are intentionally absent — the
# metabolic engine owns those. Weight feeds that engine (goal_weight); sugar +
# activity are behavioural targets the engine does not compute.
DEFAULT_GOALS = [
    {"goal_type": "weight", "target_value": 105.0, "unit": "kg",
     "notes": "Target for bodyboarding performance — reduce load, improve paddle power"},
    {"goal_type": "sugar_limit", "target_value": 25.0, "unit": "g_per_day",
     "notes": "Sugar addiction management — strict daily cap"},
    {"goal_type": "activity", "target_value": 4.0, "unit": "sessions_per_week",
     "notes": "Minimum activity sessions — mix of surf, stretch, and gym"},
]

# Goal types the metabolic engine now owns. Any ACTIVE goal of these types is a
# legacy seed that would shadow the computed target — retire it on sight.
_ENGINE_OWNED_GOAL_TYPES = ("calories", "protein")


def cleanup_stale_targets(db: Session) -> int:
    """Deactivate any active calorie/protein goals (now engine-computed).

    Idempotent: once deactivated they are skipped on later runs. Returns the
    number deactivated. Deactivation (not deletion) keeps the history and is
    reversible.
    """
    stale = (
        db.query(LifestyleGoal)
        .filter(
            LifestyleGoal.goal_type.in_(_ENGINE_OWNED_GOAL_TYPES),
            LifestyleGoal.is_active == True,  # noqa: E712
        )
        .all()
    )
    for g in stale:
        g.is_active = False
    if stale:
        db.commit()
        logger.info(
            "[lifestyle] Retired %d legacy calorie/protein goal(s) — the "
            "metabolic engine is now the source of truth for those targets",
            len(stale),
        )
    return len(stale)


def seed_lifestyle_data(db: Session) -> dict:
    """
    Seed default goals if none exist, and always run the stale-target cleanup.
    Returns a summary of what changed.
    """
    # Always retire engine-owned goals first, even on an already-seeded DB.
    deactivated = cleanup_stale_targets(db)

    existing_count = db.query(LifestyleGoal).count()
    if existing_count > 0:
        return {
            "goals_created": 0,
            "targets_deactivated": deactivated,
            "message": "Goals already exist",
        }

    created = 0
    for goal_data in DEFAULT_GOALS:
        goal = LifestyleGoal(
            goal_type=goal_data["goal_type"],
            target_value=goal_data["target_value"],
            unit=goal_data["unit"],
            notes=goal_data["notes"],
            is_active=True,
        )
        db.add(goal)
        created += 1

    db.commit()
    logger.info(f"[lifestyle] Seeded {created} default goals")

    return {"goals_created": created, "targets_deactivated": deactivated}
