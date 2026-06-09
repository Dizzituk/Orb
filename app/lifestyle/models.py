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
    # --- Health Connect / wearable daily metrics (Garmin via Health Connect) ---
    steps = Column(Integer, nullable=True)          # daily step count
    floors = Column(Integer, nullable=True)         # floors/flights climbed
    active_calories = Column(Float, nullable=True)  # active kcal burned
    resting_hr = Column(Integer, nullable=True)     # resting heart rate (bpm)
    sleep_minutes = Column(Integer, nullable=True)  # total sleep minutes for the day


# ═════════════════════════════
# STRENGTH / EXERCISE SETS (per-set progressive overload)
# ═════════════════════════════

class ExerciseSet(Base):
    """A single logged set of a strength exercise — the unit of progressive-
    overload tracking.

    One row per set. Sets group by `exercise_name` (a normalised key such as
    "bench press") so "what did I lift last time?" is a cheap, indexed lookup
    independent of any day or session. `workout_session_id` can optionally tie
    a set back to a WorkoutSession but is not required — a set can be logged
    standalone straight from chat at the gym.
    """

    __tablename__ = "lifestyle_exercise_sets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    performed_at = Column(DateTime, nullable=False, default=_now)
    exercise_name = Column(String, nullable=False, index=True)  # normalised: "bench press"
    exercise_display = Column(String, nullable=True)            # original text as spoken
    weight_kg = Column(Float, nullable=True)                    # null = bodyweight
    reps = Column(Integer, nullable=True)
    set_index = Column(Integer, nullable=True)                  # 1, 2, 3 within a session
    unit = Column(String, default="kg")                         # kg | lb (value stored is kg)
    notes = Column(Text, nullable=True)
    source = Column(String, default="voice")                   # voice | manual | google_fit
    workout_session_id = Column(Integer, nullable=True)         # optional link to a session


# ═════════════════════════════
# HEALTH CONNECT INGEST LEDGER (dedup)
# ═════════════════════════════

class HealthIngestLedger(Base):
    """Dedup ledger for Health Connect ingests.

    One row per ingested Health Connect record, keyed by the record's own
    stable UID (metadata.id on the device). Lets the bridge ingest endpoint
    skip records it has already filed, so overlapping/periodic syncs never
    create duplicate weigh-ins or workout sessions.
    """

    __tablename__ = "lifestyle_health_ingest_ledger"

    client_id = Column(String, primary_key=True)   # Health Connect record UID
    kind = Column(String, nullable=False)          # weight | workout
    orb_row_id = Column(Integer, nullable=True)     # id of the created lifestyle row
    source = Column(String, nullable=True)          # garmin | health_connect | ...
    ingested_at = Column(DateTime, default=_now)


# ═════════════════════════════
# USER PROFILE (metabolic maths: BMR / TDEE / targets)
# ═════════════════════════════

class LifestyleProfile(Base):
    """Single-row body/metabolic profile.

    Holds the static-ish inputs the metabolic engine needs alongside the
    latest logged weight: height, sex, age (date of birth preferred so it
    stays current), activity level, and chosen loss pace. One row, get-or-
    create, since the system is single-user.
    """

    __tablename__ = "lifestyle_profile"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sex = Column(String, nullable=True)                   # male | female
    height_cm = Column(Float, nullable=True)
    dob = Column(Date, nullable=True)                     # date of birth (preferred)
    age_years = Column(Integer, nullable=True)            # fallback if no dob
    activity_level = Column(String, default="moderate")   # sedentary..very_active
    goal_pace = Column(String, default="moderate")        # gentle | moderate | aggressive
    updated_at = Column(DateTime, nullable=False, default=_now)


# ═════════════════════════════
# COACHING: FOOD PREFERENCES & WRITTEN PLAN
# ══════════════════════════════

class FoodPreference(Base):
    """A single thing Astra has learned about how the user eats.

    Free-text content tagged by category so the coach builds a picture over
    time — likes, dislikes, staples, where they shop, meal timing, portion
    habits, dietary constraints, and general notes. Read before giving food
    advice; shown in the Plan tab so the user can see what Astra knows.
    """

    __tablename__ = "lifestyle_food_preference"

    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String, nullable=False, default="note")
    # like | dislike | staple | shop | timing | habit | constraint | note
    content = Column(Text, nullable=False)
    # how settled this is: core (locked, rarely changes) | evolving (default) | passing (fades)
    stability = Column(String, nullable=False, default="evolving")
    confidence = Column(Float, nullable=False, default=0.6)      # 0..1, rises when reinforced
    last_seen = Column(DateTime, nullable=False, default=_now)   # last time it was stated/reinforced
    source = Column(String, default="chat")   # chat | manual | voice
    created_at = Column(DateTime, nullable=False, default=_now)
    updated_at = Column(DateTime, nullable=False, default=_now)


class MealPlan(Base):
    """The single active written nutrition/activity plan.

    One row, get-or-create. Astra authors `body` (plain text / light markdown)
    from the user's profile, metabolic numbers, goals and food preferences; the
    Plan tab renders it and the user can edit it directly.
    """

    __tablename__ = "lifestyle_meal_plan"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String, nullable=True, default="My Plan")
    body = Column(Text, nullable=True)
    updated_at = Column(DateTime, nullable=False, default=_now)
