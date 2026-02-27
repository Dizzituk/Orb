# FILE: app/lifestyle/seed.py
"""
Seed default lifestyle data on startup.

Seeds:
- Default goals (weight target, macro targets, sugar limit)
- Initial daily targets based on Taz's profile

Only creates data if no goals exist yet (idempotent).
"""
import logging
from sqlalchemy.orm import Session

from app.lifestyle.models import LifestyleGoal

logger = logging.getLogger(__name__)

# Default goals based on Taz's profile:
# - 118kg, wants to optimise for bodyboarding
# - Keto preference: high protein, low carbs, moderate fat
# - Sugar addiction: needs a strict daily limit
DEFAULT_GOALS = [
    {"goal_type": "weight", "target_value": 105.0, "unit": "kg",
     "notes": "Target for bodyboarding performance — reduce load, improve paddle power"},
    {"goal_type": "calories", "target_value": 2200.0, "unit": "kcal",
     "notes": "Moderate deficit for gradual weight loss while maintaining energy for surf + deliveries"},
    {"goal_type": "protein", "target_value": 180.0, "unit": "g",
     "notes": "High protein to preserve muscle mass during cut — ~1.5g per kg target weight"},
    {"goal_type": "sugar_limit", "target_value": 25.0, "unit": "g_per_day",
     "notes": "Sugar addiction management — strict daily cap"},
    {"goal_type": "activity", "target_value": 4.0, "unit": "sessions_per_week",
     "notes": "Minimum activity sessions — mix of surf, stretch, and gym"},
]


def seed_lifestyle_data(db: Session) -> dict:
    """
    Seed default goals if none exist.
    Returns summary of what was created.
    """
    existing_count = db.query(LifestyleGoal).count()

    if existing_count > 0:
        return {"goals_created": 0, "message": "Goals already exist"}

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

    return {"goals_created": created}
