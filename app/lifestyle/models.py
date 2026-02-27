# FILE: app/lifestyle/models.py
"""
SQLAlchemy models for the Lifestyle Engine.

Tables:
- weight_entries       — Daily weight logs (manual + OCR from scale screenshots)
- nutrition_logs       — Meal/food entries with macro breakdowns
- workout_sessions     — Gym, surf, stretching session records
- fitness_plans        — Active workout/stretching programmes
- lifestyle_goals      — Target weight, macro targets, activity goals
- daily_summaries      — Aggregated daily stats for dashboard charts
"""
from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, Float, String, Text, DateTime, Boolean, Date,
)
from app.db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ═══════════════════════════════════════════
# WEIGHT TRACKING
# ═══════════════════════════════════════════

class WeightEntry(Base):
    """Single weight measurement."""

    __tablename__ = "lifestyle_weight_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    recorded_at = Column(DateTime, nullable=False, default=_now)
    weight_kg = Column(Float, nullable=False)
    source = Column(String, default="manual")  # manual | scale_ocr | google_fit
    notes = Column(Text, nullable=True)


# ═══════════════════════════════════════════
# NUTRITION
# ═══════════════════════════════════════════

class NutritionLog(Base):
    """Single food/meal entry with estimated macros."""

    __tablename__ = "lifestyle_nutrition_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    logged_at = Column(DateTime, nullable=False, default=_now)
    meal_type = Column(String, default="meal")  # breakfast | lunch | dinner | snack | meal
    description = Column(Text, nullable=False)  # free text: "chicken salad, rice, bit of sauce"
    calories = Column(Float, nullable=True)
    protein_g = Column(Float, nullable=True)
    carbs_g = Column(Float, nullable=True)
    fat_g = Column(Float, nullable=True)
    fibre_g = Column(Float, nullable=True)
    sugar_g = Column(Float, nullable=True)  # tracked separately for sugar addiction management
    is_estimate = Column(Boolean, default=True)  # True = AI-estimated, False = verified/scanned
    confidence = Column(Float, default=0.6)  # 0.0-1.0 how confident the estimate is
    source = Column(String, default="text")  # text | barcode | photo_ocr | voice


# ═══════════════════════════════════════════
# WORKOUT / ACTIVITY SESSIONS
# ═══════════════════════════════════════════

class WorkoutSession(Base):
    """A logged activity session — gym, surf, stretching, or delivery walking."""

    __tablename__ = "lifestyle_workout_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    started_at = Column(DateTime, nullable=False, default=_now)
    duration_mins = Column(Integer, nullable=True)
    activity_type = Column(String, nullable=False)  # gym | surf | stretch | walk | bodyboard
    title = Column(String, nullable=True)  # e.g. "Morning stretch", "Bantham session"
    notes = Column(Text, nullable=True)
    calories_burned = Column(Float, nullable=True)  # estimated
    # Surf-specific fields
    surf_location = Column(String, nullable=True)
    surf_conditions = Column(String, nullable=True)  # e.g. "3ft, offshore, clean"
    # Gym-specific
    exercises_json = Column(Text, nullable=True)  # JSON array of exercises done
    source = Column(String, default="manual")  # manual | google_fit | voice


# ═══════════════════════════════════════════
# FITNESS PLANS
# ═══════════════════════════════════════════

class FitnessPlan(Base):
    """An active fitness programme — stretching routine, gym plan, etc."""

    __tablename__ = "lifestyle_fitness_plans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, nullable=False, default=_now)
    plan_type = Column(String, nullable=False)  # stretch | gym | bodyboard_prep | hybrid
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    schedule_json = Column(Text, nullable=True)  # JSON: weekly schedule
    exercises_json = Column(Text, nullable=True)  # JSON: exercise definitions
    is_active = Column(Boolean, default=True)
    target_weeks = Column(Integer, nullable=True)  # programme duration


# ═══════════════════════════════════════════
# GOALS & TARGETS
# ═══════════════════════════════════════════

class LifestyleGoal(Base):
    """A target — weight goal, daily macro targets, activity minimums."""

    __tablename__ = "lifestyle_goals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, nullable=False, default=_now)
    goal_type = Column(String, nullable=False)  # weight | calories | protein | activity | sugar_limit
    target_value = Column(Float, nullable=False)
    unit = Column(String, nullable=False)  # kg | kcal | g | sessions_per_week | g_per_day
    is_active = Column(Boolean, default=True)
    notes = Column(Text, nullable=True)


# ═══════════════════════════════════════════
# DAILY SUMMARIES (for charts)
# ═══════════════════════════════════════════

class DailySummary(Base):
    """Aggregated daily stats — one row per day for efficient chart rendering."""

    __tablename__ = "lifestyle_daily_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True)
    weight_kg = Column(Float, nullable=True)  # latest weight for that day
    total_calories = Column(Float, nullable=True)
    total_protein_g = Column(Float, nullable=True)
    total_carbs_g = Column(Float, nullable=True)
    total_fat_g = Column(Float, nullable=True)
    total_sugar_g = Column(Float, nullable=True)
    activity_mins = Column(Integer, default=0)
    activity_types = Column(String, nullable=True)  # comma-separated: "stretch,surf"
    streak_days = Column(Integer, default=0)  # consecutive days with activity
