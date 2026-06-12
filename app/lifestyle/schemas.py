# FILE: app/lifestyle/schemas.py
# Purpose: Pydantic schemas for the Lifestyle API request/response shapes.
# Called-by: app.bridge.dashboards, app.lifestyle.coaching, app.lifestyle.daily_view, app.lifestyle.router (+2 more)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Pydantic schemas for the Lifestyle API request/response shapes.
Mirrors the pattern in app/investments/schemas.py.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════
# WEIGHT
# ═══════════════════════════════════════════

class WeightEntryIn(BaseModel):
    """Input for logging a weight measurement."""
    weight_kg: float
    source: str = "manual"
    notes: Optional[str] = None


class WeightEntryOut(BaseModel):
    """Single weight entry for display."""
    id: int
    recorded_at: str
    weight_kg: float
    source: str
    notes: Optional[str] = None


class WeightTrend(BaseModel):
    """Weight over time for the dashboard chart."""
    points: List[WeightEntryOut] = []
    current_kg: Optional[float] = None
    change_7d_kg: Optional[float] = None
    change_30d_kg: Optional[float] = None
    target_kg: Optional[float] = None


# ═══════════════════════════════════════════
# NUTRITION
# ═══════════════════════════════════════════

class NutritionLogIn(BaseModel):
    """Input for logging a meal / food item."""
    description: str
    meal_type: str = "meal"
    # Optional overrides — if not provided, AI estimates them
    calories: Optional[float] = None
    protein_g: Optional[float] = None
    carbs_g: Optional[float] = None
    fat_g: Optional[float] = None
    sugar_g: Optional[float] = None


class NutritionLogOut(BaseModel):
    """Single nutrition entry for display."""
    id: int
    logged_at: str
    meal_type: str
    description: str
    calories: Optional[float] = None
    protein_g: Optional[float] = None
    carbs_g: Optional[float] = None
    fat_g: Optional[float] = None
    fibre_g: Optional[float] = None
    sugar_g: Optional[float] = None
    is_estimate: bool = True
    confidence: float = 0.6
    source: str = "text"


class DailyNutrition(BaseModel):
    """Aggregated nutrition for a single day."""
    date: str
    total_calories: float = 0
    total_protein_g: float = 0
    total_carbs_g: float = 0
    total_fat_g: float = 0
    total_sugar_g: float = 0
    meal_count: int = 0
    logs: List[NutritionLogOut] = []


class NutritionTargets(BaseModel):
    """Daily macro targets (keto-aware)."""
    calories: float = 2200
    protein_g: float = 180
    carbs_g: float = 30  # keto default
    fat_g: float = 160
    sugar_limit_g: float = 25  # sugar addiction management


# ═══════════════════════════════════════════
# WORKOUT / ACTIVITY
# ═══════════════════════════════════════════

class WorkoutSessionIn(BaseModel):
    """Input for logging an activity session."""
    activity_type: str  # gym | surf | stretch | walk | bodyboard
    duration_mins: Optional[int] = None
    title: Optional[str] = None
    notes: Optional[str] = None
    calories_burned: Optional[float] = None
    surf_location: Optional[str] = None
    surf_conditions: Optional[str] = None


class WorkoutSessionOut(BaseModel):
    """Single workout session for display."""
    id: int
    started_at: str
    duration_mins: Optional[int] = None
    activity_type: str
    title: Optional[str] = None
    notes: Optional[str] = None
    calories_burned: Optional[float] = None
    surf_location: Optional[str] = None
    surf_conditions: Optional[str] = None
    source: str = "manual"


# ═══════════════════════════════════════════
# GOALS
# ═══════════════════════════════════════════

class GoalIn(BaseModel):
    """Input for creating/updating a goal."""
    goal_type: str  # weight | calories | protein | activity | sugar_limit
    target_value: float
    unit: str
    notes: Optional[str] = None


class GoalOut(BaseModel):
    """Single goal for display."""
    id: int
    goal_type: str
    target_value: float
    unit: str
    is_active: bool = True
    notes: Optional[str] = None


# ═══════════════════════════════════════════
# DASHBOARD OVERVIEW
# ═══════════════════════════════════════════

class DashboardSummary(BaseModel):
    """Top-level lifestyle dashboard overview."""
    current_weight_kg: Optional[float] = None
    weight_change_7d: Optional[float] = None
    target_weight_kg: Optional[float] = None
    today_calories: float = 0
    today_protein_g: float = 0
    today_carbs_g: float = 0
    today_fat_g: float = 0
    today_sugar_g: float = 0
    calories_target: float = 2200
    protein_target: float = 180
    carbs_target: Optional[float] = None
    fat_target: Optional[float] = None
    sugar_limit: float = 25
    activity_streak_days: int = 0
    sessions_this_week: int = 0
    last_activity: Optional[str] = None
    last_activity_type: Optional[str] = None
    # Wearable headline for today (Garmin via Health Connect); None until synced
    today_steps: Optional[int] = None
    today_floors: Optional[int] = None
    today_active_calories: Optional[float] = None


class DailySummaryPoint(BaseModel):
    """Single data point for dashboard charts."""
    date: str
    weight_kg: Optional[float] = None
    total_calories: Optional[float] = None
    total_protein_g: Optional[float] = None
    total_sugar_g: Optional[float] = None
    activity_mins: int = 0


class DashboardHistory(BaseModel):
    """Time-series data for dashboard charts."""
    points: List[DailySummaryPoint] = []
    range: str = "30d"


# ═════════════════════════════
# STRENGTH / EXERCISE SETS
# ═════════════════════════════

class ExerciseSetIn(BaseModel):
    """Input for logging a single set of a strength exercise."""
    exercise_name: str
    weight_kg: Optional[float] = None
    reps: Optional[int] = None
    set_index: Optional[int] = None
    unit: str = "kg"
    notes: Optional[str] = None


class ExerciseSetOut(BaseModel):
    """A single logged set for display."""
    id: int
    performed_at: str
    exercise_name: str
    exercise_display: Optional[str] = None
    weight_kg: Optional[float] = None
    reps: Optional[int] = None
    set_index: Optional[int] = None
    unit: str = "kg"
    notes: Optional[str] = None


class ExerciseSession(BaseModel):
    """All sets of one exercise performed on a single day."""
    date: str
    sets: List[ExerciseSetOut] = []
    top_weight_kg: Optional[float] = None
    total_reps: int = 0


class ExerciseHistory(BaseModel):
    """Progressive-overload view of a single exercise over time."""
    exercise_name: str
    sessions: List[ExerciseSession] = []
    last_performed: Optional[str] = None
    last_top_weight_kg: Optional[float] = None
    best_weight_kg: Optional[float] = None        # all-time heaviest set
    best_est_1rm_kg: Optional[float] = None       # Epley estimate from best set


class StrengthDay(BaseModel):
    """One day of strength work — every exercise trained that day."""
    date: str
    exercises: List[ExerciseSession] = []


class StrengthLog(BaseModel):
    """Day-by-day strength log for the Fitness tab (newest day first)."""
    days: List[StrengthDay] = []


class ExerciseSetEntry(BaseModel):
    """One {weight, reps} pair within a bulk exercise log."""
    weight_kg: Optional[float] = None
    reps: Optional[int] = None


class ExerciseLogIn(BaseModel):
    """Log one or more sets of a single exercise in one call."""
    exercise_name: str
    sets: List[ExerciseSetEntry] = []
    unit: str = "kg"
    notes: Optional[str] = None
    session_id: Optional[int] = None


class PrepSplitIn(BaseModel):
    """Divide a batch cook's total macros across N days."""
    description: str
    days: int
    total: dict                      # {calories, protein_g, carbs_g, fat_g[, sugar_g]}
    start_date: Optional[str] = None # YYYY-MM-DD; defaults to tomorrow
    meal_type: str = "meal"


# ══════════════════════════════
# PROFILE & METABOLIC
# ══════════════════════════════

class ProfileIn(BaseModel):
    """Partial profile update — every field optional, only provided ones change."""
    sex: Optional[str] = None              # male | female
    height_cm: Optional[float] = None
    dob: Optional[str] = None              # ISO date YYYY-MM-DD
    age_years: Optional[int] = None
    activity_level: Optional[str] = None   # sedentary | light | moderate | active | very_active
    goal_pace: Optional[str] = None        # gentle | moderate | aggressive


class ProfileOut(BaseModel):
    sex: Optional[str] = None
    height_cm: Optional[float] = None
    dob: Optional[str] = None
    age_years: Optional[int] = None
    activity_level: str = "moderate"
    goal_pace: str = "moderate"


class MetabolicSummary(BaseModel):
    """Computed metabolic picture for coaching."""
    complete: bool = False
    missing: List[str] = []
    bmr: Optional[float] = None                # resting burn, kcal/day
    tdee: Optional[float] = None               # total daily burn, kcal/day
    tdee_source: Optional[str] = None          # estimated | measured
    activity_level: Optional[str] = None
    active_calories_used: Optional[float] = None
    target_calories: Optional[float] = None    # recommended sustainable intake
    deficit_kcal: Optional[int] = None
    floored_at_bmr: bool = False
    pace: Optional[str] = None
    protein_g: Optional[float] = None
    carbs_g: Optional[float] = None
    fat_g: Optional[float] = None
    current_weight_kg: Optional[float] = None
    goal_weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    bmi: Optional[float] = None
    bmi_category: Optional[str] = None


# ═════════════════════════════
# COACHING: FOOD PREFERENCES & PLAN
# ═════════════════════════════

class FoodPreferenceIn(BaseModel):
    """Add a thing Astra knows about how the user eats."""
    category: str = "note"   # like | dislike | staple | shop | timing | habit | constraint | note
    content: str
    stability: Optional[str] = None   # core | evolving | passing (default evolving)
    confidence: Optional[float] = None


class FoodPreferenceUpdate(BaseModel):
    """Revise an existing preference in place (evolve it)."""
    content: Optional[str] = None
    category: Optional[str] = None
    stability: Optional[str] = None
    confidence: Optional[float] = None


class FoodPreferenceOut(BaseModel):
    id: int
    category: str
    content: str
    stability: str = "evolving"     # core | evolving | passing
    confidence: float = 0.6
    last_seen: Optional[str] = None
    stale: bool = False             # passing + not reinforced in a while
    source: str = "chat"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class FoodPreferenceList(BaseModel):
    preferences: List[FoodPreferenceOut] = []


class MealPlanIn(BaseModel):
    """Write/update the active plan."""
    body: str
    title: Optional[str] = None


class MealPlanOut(BaseModel):
    title: Optional[str] = "My Plan"
    body: Optional[str] = None
    updated_at: Optional[str] = None
    exists: bool = False
