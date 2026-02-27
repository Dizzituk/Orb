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
