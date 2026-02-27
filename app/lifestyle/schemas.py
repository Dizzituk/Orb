# FILE: app/lifestyle/schemas.py
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
    sugar_limit: float = 25
    activity_streak_days: int = 0
    sessions_this_week: int = 0
    last_activity: Optional[str] = None
    last_activity_type: Optional[str] = None


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
