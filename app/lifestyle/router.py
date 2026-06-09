# FILE: app/lifestyle/router.py
"""
FastAPI router for the Lifestyle Engine.

All endpoints require auth via Depends(require_auth).
Follows the same pattern as app/investments/router.py.
"""
import logging
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth
from app.lifestyle import service
from app.lifestyle.fitness import get_stretch_routine, get_gym_programme, get_all_routines
from app.lifestyle.schemas import (
    WeightEntryIn, WeightEntryOut, WeightTrend,
    NutritionLogIn, NutritionLogOut, DailyNutrition,
    WorkoutSessionIn, WorkoutSessionOut,
    GoalIn, GoalOut,
    DashboardSummary, DashboardHistory,
    MetabolicSummary, FoodPreferenceIn, FoodPreferenceUpdate, FoodPreferenceOut, FoodPreferenceList,
    MealPlanIn, MealPlanOut,
    ExerciseLogIn, ExerciseSetOut, ExerciseHistory, StrengthLog,
    PrepSplitIn,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/lifestyle",
    tags=["Lifestyle"],
    dependencies=[Depends(require_auth)],
)


# ═══════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════

@router.get("/dashboard", response_model=DashboardSummary)
def get_dashboard(db: Session = Depends(get_db)):
    """Top-level lifestyle dashboard summary."""
    return service.get_dashboard_summary(db)


@router.get("/dashboard/history", response_model=DashboardHistory)
def get_history(
    range: str = Query("30d", pattern="^(7d|14d|30d|90d|all)$"),
    db: Session = Depends(get_db),
):
    """Time-series data for dashboard charts."""
    return service.get_dashboard_history(db, range)


@router.get("/dashboard/day", response_model=DashboardSummary)
def get_dashboard_day(
    target_date: str = Query(..., description="ISO date: 2026-06-03"),
    db: Session = Depends(get_db),
):
    """Per-day dashboard summary: that day's intake, movement and bodyweight,
    with current targets and rolling context. Powers Overview day-navigation."""
    from app.lifestyle import daily_view
    d = date.fromisoformat(target_date)
    return daily_view.get_day_dashboard(db, d)


# ═══════════════════════════════════════════
# WEIGHT
# ═══════════════════════════════════════════

@router.post("/weight", response_model=WeightEntryOut)
def log_weight(entry: WeightEntryIn, db: Session = Depends(get_db)):
    """Log a weight measurement."""
    return service.log_weight(db, entry.weight_kg, entry.source, entry.notes)


@router.get("/weight/trend", response_model=WeightTrend)
def get_weight_trend(
    days: int = Query(90, ge=7, le=365),
    db: Session = Depends(get_db),
):
    """Weight trend data for chart display."""
    return service.get_weight_trend(db, days)


# ═══════════════════════════════════════════
# NUTRITION
# ═══════════════════════════════════════════

@router.post("/nutrition", response_model=NutritionLogOut)
def log_nutrition(entry: NutritionLogIn, db: Session = Depends(get_db)):
    """Log a meal or food item."""
    return service.log_nutrition(
        db, entry.description, entry.meal_type,
        entry.calories, entry.protein_g, entry.carbs_g,
        entry.fat_g, entry.sugar_g,
    )


@router.get("/nutrition/today", response_model=DailyNutrition)
def get_today_nutrition(db: Session = Depends(get_db)):
    """Get today's nutrition summary."""
    return service.get_daily_nutrition(db)


@router.get("/nutrition/day", response_model=DailyNutrition)
def get_day_nutrition(
    target_date: str = Query(..., description="ISO date: 2026-02-26"),
    db: Session = Depends(get_db),
):
    """Get nutrition for a specific date."""
    d = date.fromisoformat(target_date)
    return service.get_daily_nutrition(db, d)


@router.delete("/nutrition/{log_id}")
def delete_nutrition(log_id: int, db: Session = Depends(get_db)):
    """Delete a nutrition log entry by id, then recompute that day's totals."""
    ok = service.delete_nutrition(db, log_id)
    return {"ok": ok, "deleted_id": log_id if ok else None}


@router.post("/nutrition/prep-split")
def prep_split(body: PrepSplitIn, db: Session = Depends(get_db)):
    """Divide a batch cook's total macros across N days, writing a portion into
    each upcoming day's diary."""
    from datetime import date as _d
    start = None
    if body.start_date:
        try:
            start = _d.fromisoformat(body.start_date)
        except ValueError:
            start = None
    return service.prep_split(db, body.description, body.total, body.days,
                              start_date=start, meal_type=body.meal_type)


# ═══════════════════════════════════════════
# WORKOUTS / ACTIVITY
# ═══════════════════════════════════════════

@router.post("/workout", response_model=WorkoutSessionOut)
def log_workout(session: WorkoutSessionIn, db: Session = Depends(get_db)):
    """Log an activity session."""
    return service.log_workout(
        db, session.activity_type, session.duration_mins,
        session.title, session.notes, session.calories_burned,
        session.surf_location, session.surf_conditions,
    )


@router.get("/workouts/recent")
def get_recent_workouts(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Get recent workout sessions."""
    return {"sessions": service.get_recent_workouts(db, limit)}


@router.get("/workouts/week")
def get_weekly_activity(db: Session = Depends(get_db)):
    """Get this week's activity summary."""
    return service.get_weekly_activity(db)


@router.get("/workouts/day")
def get_workouts_day(
    target_date: str = Query(..., description="ISO date: 2026-06-03"),
    db: Session = Depends(get_db),
):
    """Workout sessions for a specific day. Powers Activity day-navigation."""
    from app.lifestyle import daily_view
    d = date.fromisoformat(target_date)
    return {"sessions": daily_view.get_day_workouts(db, d)}


# ════════════════════════════
# STRENGTH (exercise sets: log / history / day-log)
# ════════════════════════════

@router.post("/exercises")
def log_exercise(entry: ExerciseLogIn, db: Session = Depends(get_db)):
    """Log one or more sets of a strength exercise."""
    sets = [{"weight_kg": s.weight_kg, "reps": s.reps} for s in (entry.sets or [])]
    rows = service.log_exercise_sets(
        db, entry.exercise_name, sets,
        unit=entry.unit or "kg", notes=entry.notes,
        display=entry.exercise_name, source="manual",
        session_id=entry.session_id,
    )
    return {"ok": True, "sets": rows}


@router.get("/exercises/log", response_model=StrengthLog)
def get_exercise_log(
    days: int = Query(21, ge=1, le=120),
    db: Session = Depends(get_db),
):
    """Day-by-day strength log across all exercises (newest day first)."""
    return service.get_strength_log(db, days=days)


@router.get("/exercises/recent")
def get_recent_exercises(db: Session = Depends(get_db)):
    """Distinct exercise names, most recently performed first (for quick-add)."""
    return {"exercises": service.list_recent_exercises(db, limit=12)}


@router.get("/exercises/history/{exercise_name}", response_model=ExerciseHistory)
def get_exercise_history(
    exercise_name: str,
    limit_sessions: int = Query(8, ge=1, le=20),
    db: Session = Depends(get_db),
):
    """Progressive-overload history for one exercise."""
    return service.get_exercise_history(db, exercise_name, limit_sessions=limit_sessions)


@router.delete("/exercises/{set_id}")
def delete_exercise_set(set_id: int, db: Session = Depends(get_db)):
    """Delete a single logged set by id."""
    ok = service.delete_exercise_set(db, set_id)
    return {"ok": ok, "deleted_id": set_id if ok else None}


# ═══════════════════════════════════════════
# FITNESS PLANS & ROUTINES
# ═══════════════════════════════════════════

@router.get("/fitness/routines")
def list_routines():
    """List all available fitness routines and programmes."""
    return {"routines": get_all_routines()}


@router.get("/fitness/stretch/{routine_type}")
def get_stretch(routine_type: str = "driver"):
    """Get a specific stretching routine."""
    return get_stretch_routine(routine_type)


@router.get("/fitness/gym")
def get_gym():
    """Get the current gym programme."""
    return get_gym_programme()


# ═══════════════════════════════════════════
# GOALS
# ═══════════════════════════════════════════

@router.post("/goals", response_model=GoalOut)
def set_goal(goal: GoalIn, db: Session = Depends(get_db)):
    """Set or update a lifestyle goal."""
    return service.set_goal(db, goal.goal_type, goal.target_value, goal.unit, goal.notes)


@router.get("/goals")
def get_goals(db: Session = Depends(get_db)):
    """Get all active goals."""
    return {"goals": service.get_active_goals(db)}


# ═════════════════════════════
# METABOLIC (BMR / TDEE / BMI / targets)
# ═════════════════════════════

@router.get("/metabolic", response_model=MetabolicSummary)
def get_metabolic(db: Session = Depends(get_db)):
    """Computed metabolic picture: BMR, daily burn, BMI, calorie + protein targets."""
    return service.get_metabolic_summary(db)


# ═════════════════════════════
# COACHING: FOOD PREFERENCES & PLAN
# ═════════════════════════════

@router.get("/preferences", response_model=FoodPreferenceList)
def get_preferences(db: Session = Depends(get_db)):
    """Everything Astra knows about how the user eats."""
    return FoodPreferenceList(preferences=service.get_food_preferences(db))


@router.post("/preferences", response_model=FoodPreferenceOut)
def add_preference(pref: FoodPreferenceIn, db: Session = Depends(get_db)):
    """Add a food preference/habit."""
    return service.add_food_preference(db, pref.category, pref.content, source="manual")


@router.delete("/preferences/{pref_id}")
def delete_preference(pref_id: int, db: Session = Depends(get_db)):
    """Delete a food preference by id."""
    ok = service.delete_food_preference(db, pref_id)
    return {"ok": ok, "deleted_id": pref_id if ok else None}


@router.put("/preferences/{pref_id}", response_model=FoodPreferenceOut)
def update_preference(pref_id: int, patch: FoodPreferenceUpdate, db: Session = Depends(get_db)):
    """Revise a food preference in place (evolve it)."""
    out = service.update_food_preference(
        db, pref_id,
        content=patch.content, category=patch.category,
        stability=patch.stability, confidence=patch.confidence,
    )
    if out is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="preference not found")
    return out


@router.get("/plan", response_model=MealPlanOut)
def get_plan(db: Session = Depends(get_db)):
    """The active written plan."""
    return service.get_meal_plan(db)


@router.put("/plan", response_model=MealPlanOut)
def put_plan(plan: MealPlanIn, db: Session = Depends(get_db)):
    """Create or update the written plan."""
    return service.set_meal_plan(db, plan.body, plan.title)
