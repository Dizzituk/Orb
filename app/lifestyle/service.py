# FILE: app/lifestyle/service.py
# Purpose: Core service layer for the Lifestyle Engine.
# Called-by: app.bridge.dashboards, app.lifestyle.daily_view, app.lifestyle.health_sync, app.lifestyle.nutrition_copy (+5 more)
# Depends-on: app.lifestyle, app.lifestyle.burn, app.lifestyle.coaching, app.lifestyle.history (+4 more)
# Last-renovated: 2026-06-11
"""
Core service layer for the Lifestyle Engine.

Handles:
- Weight tracking and trend calculation
- Daily nutrition aggregation
- Activity streak tracking
- Dashboard summary assembly
- Daily summary generation for charts
"""
import logging
from datetime import datetime, date, timedelta, timezone
from typing import Optional, List

from sqlalchemy import func, desc
from sqlalchemy.orm import Session

from app.lifestyle.models import (
    WeightEntry, NutritionLog, WorkoutSession,
    LifestyleGoal, DailySummary, LifestyleProfile,
    FoodPreference, MealPlan,
)
from app.lifestyle.schemas import (
    WeightEntryOut, WeightTrend,
    NutritionLogOut, DailyNutrition,
    WorkoutSessionOut,
    GoalOut,
    DashboardSummary, DailySummaryPoint, DashboardHistory,
    ProfileOut, MetabolicSummary,
    FoodPreferenceOut, MealPlanOut,
)

logger = logging.getLogger(__name__)


def _naive_utc(dt):
    """Coerce any datetime to naive UTC so it compares safely with DB-stored
    (naive) datetimes. SQLite drops tzinfo on round-trip, so reads come back
    naive; mixing them with a tz-aware 'now' raises TypeError."""
    if dt is None:
        return None
    if getattr(dt, "tzinfo", None) is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


# ═══════════════════════════════════════════
# WEIGHT
# ═══════════════════════════════════════════

def log_weight(db: Session, weight_kg: float, source: str = "manual",
               notes: Optional[str] = None) -> WeightEntryOut:
    """Record a weight measurement and update daily summary."""
    entry = WeightEntry(
        weight_kg=weight_kg,
        source=source,
        notes=notes,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    logger.info(f"[lifestyle] Weight logged: {weight_kg}kg ({source})")

    # Update daily summary
    _update_daily_summary(db, date.today())

    return _weight_to_out(entry)


def get_weight_trend(db: Session, days: int = 90) -> WeightTrend:
    """Get weight entries and trend data for the chart."""
    now = _naive_utc(datetime.now(timezone.utc))
    cutoff = now - timedelta(days=days)

    entries = (
        db.query(WeightEntry)
        .filter(WeightEntry.recorded_at >= cutoff)
        .order_by(WeightEntry.recorded_at.asc())
        .all()
    )

    points = [_weight_to_out(e) for e in entries]

    # Current weight = most recent entry
    current = entries[-1].weight_kg if entries else None

    # 7-day change
    change_7d = None
    if len(entries) >= 2:
        week_ago = now - timedelta(days=7)
        older = [e for e in entries if _naive_utc(e.recorded_at) <= week_ago]
        if older:
            change_7d = round(current - older[-1].weight_kg, 1) if current else None

    # 30-day change
    change_30d = None
    if len(entries) >= 2:
        month_ago = now - timedelta(days=30)
        older_30 = [e for e in entries if _naive_utc(e.recorded_at) <= month_ago]
        if older_30:
            change_30d = round(current - older_30[-1].weight_kg, 1) if current else None

    # Target weight from active goal
    target = _get_active_goal_value(db, "weight")

    return WeightTrend(
        points=points,
        current_kg=current,
        change_7d_kg=change_7d,
        change_30d_kg=change_30d,
        target_kg=target,
    )


# ═══════════════════════════════════════════
# NUTRITION
# ═══════════════════════════════════════════

def log_nutrition(db: Session, description: str, meal_type: str = "meal",
                  calories: Optional[float] = None, protein_g: Optional[float] = None,
                  carbs_g: Optional[float] = None, fat_g: Optional[float] = None,
                  sugar_g: Optional[float] = None) -> NutritionLogOut:
    """
    Log a meal/food entry.

    If macros aren't provided, they'll be None until the AI estimation
    service fills them in (future: nutrition.py estimate_macros).
    """
    entry = NutritionLog(
        description=description,
        meal_type=meal_type,
        calories=calories,
        protein_g=protein_g,
        carbs_g=carbs_g,
        fat_g=fat_g,
        sugar_g=sugar_g,
        is_estimate=calories is None,  # if user didn't provide, it's an estimate
        confidence=0.9 if calories is not None else 0.0,
        source="text",
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    logger.info(f"[lifestyle] Nutrition logged: {description} ({meal_type})")

    _update_daily_summary(db, date.today())
    return _nutrition_to_out(entry)


def log_nutrition_items(db: Session, items: List[dict], meal_type: str = "meal",
                        on_date: Optional[date] = None) -> List[NutritionLogOut]:
    """Log several distinct foods as SEPARATE rows in one transaction, then
    update the day summary once. Each item dict carries its own description +
    calories/protein_g/carbs_g/fat_g(+sugar_g). Itemising like this lets the
    user see every food with its own macros and check each estimate, rather
    than one merged blob. `on_date` (default today) stamps the rows — used by
    prep-splitting to write meals into future days. Returns the created entries.
    """
    target = on_date or date.today()
    logged_at = (
        datetime.combine(target, datetime.min.time()).replace(tzinfo=timezone.utc)
        + timedelta(hours=12)
    )  # noon, so the row sits cleanly inside the day window
    created = []
    for it in items:
        cal = it.get("calories")
        entry = NutritionLog(
            description=str(it.get("description") or "").strip(),
            meal_type=meal_type,
            calories=cal,
            protein_g=it.get("protein_g"),
            carbs_g=it.get("carbs_g"),
            fat_g=it.get("fat_g"),
            sugar_g=it.get("sugar_g"),
            is_estimate=bool(it.get("is_estimate", True)),
            confidence=0.9 if cal is not None else 0.0,
            source=it.get("source") or "text",
            logged_at=logged_at,
        )
        db.add(entry)
        created.append(entry)
    db.commit()
    for e in created:
        db.refresh(e)
    _update_daily_summary(db, target)
    logger.info(f"[lifestyle] Logged {len(created)} itemised nutrition rows for {target.isoformat()}")
    return [_nutrition_to_out(e) for e in created]


def _coerce_num(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def log_nutrition_items_checked(db: Session, raw_items: list, meal_type: str = "meal") -> dict:
    """Validate then log a list of food items as separate rows.

    Hard rule per item: each needs a description + calories/protein/carbs/fat.
    Returns {'ok': True, 'rows': [NutritionLogOut, ...]} on success, or
    {'ok': False, 'incomplete': [{'item','missing'}, ...]} if any item is
    missing required fields (nothing is logged in that case).
    """
    req = ("calories", "protein_g", "carbs_g", "fat_g")
    norm, incomplete = [], []
    for idx, it in enumerate(raw_items):
        if not isinstance(it, dict):
            continue
        desc = str(it.get("description") or "").strip()
        vals = {k: _coerce_num(it.get(k)) for k in
                ("calories", "protein_g", "carbs_g", "fat_g", "sugar_g")}
        miss = [k for k in req if vals[k] is None]
        if not desc:
            miss = ["description"] + miss
        if miss:
            incomplete.append({"item": desc or f"item {idx + 1}", "missing": miss})
        else:
            norm.append({"description": desc, **vals})
    if incomplete:
        return {"ok": False, "incomplete": incomplete}
    return {"ok": True, "rows": log_nutrition_items(db, norm, meal_type)}


def prep_split(db: Session, description: str, total: dict, days: int,
               start_date: Optional[date] = None, meal_type: str = "meal") -> dict:
    """Divide a batch cook's TOTAL macros evenly across `days` and write one
    portion row into each day, starting at `start_date` (default tomorrow).

    `total` is the whole-batch {calories, protein_g, carbs_g, fat_g[, sugar_g]}.
    Each day gets total/days, so the per-day portion is what the user actually
    eats. Rows are tagged source='prep' and the description notes the portion
    (e.g. 'Chicken curry batch (1/3)'). Future days then show the prepped meal
    in the diary. Returns {'ok', 'days': [{date, item}], 'per_day': {...}}.
    """
    n = max(1, min(14, int(days)))
    req = ("calories", "protein_g", "carbs_g", "fat_g")
    vals = {k: _coerce_num(total.get(k)) for k in
            ("calories", "protein_g", "carbs_g", "fat_g", "sugar_g")}
    missing = [k for k in req if vals[k] is None]
    if not str(description or "").strip():
        missing = ["description"] + missing
    if missing:
        return {"ok": False, "missing": missing}

    per_day = {k: (round(v / n, 1) if v is not None else None) for k, v in vals.items()}
    base = (start_date or (date.today() + timedelta(days=1)))

    written = []
    for i in range(n):
        d = base + timedelta(days=i)
        item = {
            "description": f"{description.strip()} (1/{n} prepped portion)",
            "source": "prep",
            **per_day,
        }
        rows = log_nutrition_items(db, [item], meal_type, on_date=d)
        written.append({"date": d.isoformat(), "item": _to_dict_safe(rows[0]) if rows else None})

    logger.info(f"[lifestyle] Prep-split '{description}' over {n} days from {base.isoformat()}")
    return {"ok": True, "days": written, "per_day": per_day, "n": n,
            "start_date": base.isoformat()}


def _to_dict_safe(obj):
    """Pydantic v1/v2 -> dict (local helper to avoid importing the tools layer)."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return dict(obj)


def get_daily_nutrition(db: Session, target_date: Optional[date] = None) -> DailyNutrition:
    """Get all nutrition logs for a given day with totals."""
    target = target_date or date.today()
    start = datetime.combine(target, datetime.min.time()).replace(tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    logs = (
        db.query(NutritionLog)
        .filter(NutritionLog.logged_at >= start, NutritionLog.logged_at < end)
        .order_by(NutritionLog.logged_at.asc())
        .all()
    )

    items = [_nutrition_to_out(l) for l in logs]

    return DailyNutrition(
        date=target.isoformat(),
        total_calories=sum(l.calories or 0 for l in logs),
        total_protein_g=sum(l.protein_g or 0 for l in logs),
        total_carbs_g=sum(l.carbs_g or 0 for l in logs),
        total_fat_g=sum(l.fat_g or 0 for l in logs),
        total_sugar_g=sum(l.sugar_g or 0 for l in logs),
        meal_count=len(logs),
        logs=items,
    )


def get_nutrition_history(db: Session, days: int = 30) -> List[DailyNutrition]:
    """Get daily nutrition summaries for the last N days."""
    result = []
    today = date.today()
    for i in range(days):
        d = today - timedelta(days=i)
        result.append(get_daily_nutrition(db, d))
    result.reverse()
    return result


def delete_nutrition(db: Session, log_id: int) -> bool:
    """Delete a nutrition log by id and recompute that day's summary."""
    entry = db.query(NutritionLog).filter(NutritionLog.id == log_id).first()
    if not entry:
        return False
    entry_date = entry.logged_at.date() if entry.logged_at else date.today()
    db.delete(entry)
    db.commit()
    _update_daily_summary(db, entry_date)
    logger.info(f"[lifestyle] Nutrition log {log_id} deleted")
    return True


# ═══════════════════════════════════════════
# WORKOUTS / ACTIVITY
# ═══════════════════════════════════════════

def log_workout(db: Session, activity_type: str, duration_mins: Optional[int] = None,
                title: Optional[str] = None, notes: Optional[str] = None,
                calories_burned: Optional[float] = None,
                surf_location: Optional[str] = None,
                surf_conditions: Optional[str] = None) -> WorkoutSessionOut:
    """Record an activity session."""
    session = WorkoutSession(
        activity_type=activity_type,
        duration_mins=duration_mins,
        title=title,
        notes=notes,
        calories_burned=calories_burned,
        surf_location=surf_location,
        surf_conditions=surf_conditions,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    logger.info(f"[lifestyle] Workout logged: {activity_type} ({duration_mins}min)")

    _update_daily_summary(db, date.today())
    return _workout_to_out(session)


def get_recent_workouts(db: Session, limit: int = 20) -> List[WorkoutSessionOut]:
    """Get most recent workout sessions."""
    sessions = (
        db.query(WorkoutSession)
        .order_by(WorkoutSession.started_at.desc())
        .limit(limit)
        .all()
    )
    return [_workout_to_out(s) for s in sessions]


def get_weekly_activity(db: Session) -> dict:
    """Get activity summary for the current week (Mon-Sun)."""
    today = date.today()
    monday = today - timedelta(days=today.weekday())
    start = datetime.combine(monday, datetime.min.time()).replace(tzinfo=timezone.utc)

    sessions = (
        db.query(WorkoutSession)
        .filter(WorkoutSession.started_at >= start)
        .all()
    )

    total_mins = sum(s.duration_mins or 0 for s in sessions)
    types = list(set(s.activity_type for s in sessions))

    return {
        "session_count": len(sessions),
        "total_minutes": total_mins,
        "activity_types": types,
        "sessions": [_workout_to_out(s) for s in sessions],
    }


# ═══════════════════════════════════════════
# GOALS
# ═══════════════════════════════════════════

def set_goal(db: Session, goal_type: str, target_value: float,
             unit: str, notes: Optional[str] = None) -> GoalOut:
    """Set or update a goal. Deactivates previous goal of same type."""
    # Deactivate existing goals of this type
    existing = (
        db.query(LifestyleGoal)
        .filter(LifestyleGoal.goal_type == goal_type, LifestyleGoal.is_active == True)
        .all()
    )
    for g in existing:
        g.is_active = False

    goal = LifestyleGoal(
        goal_type=goal_type,
        target_value=target_value,
        unit=unit,
        notes=notes,
    )
    db.add(goal)
    db.commit()
    db.refresh(goal)
    logger.info(f"[lifestyle] Goal set: {goal_type} = {target_value}{unit}")

    return GoalOut(
        id=goal.id,
        goal_type=goal.goal_type,
        target_value=goal.target_value,
        unit=goal.unit,
        is_active=goal.is_active,
        notes=goal.notes,
    )


def get_active_goals(db: Session) -> List[GoalOut]:
    """Get all active goals."""
    goals = (
        db.query(LifestyleGoal)
        .filter(LifestyleGoal.is_active == True)
        .all()
    )
    return [
        GoalOut(
            id=g.id, goal_type=g.goal_type, target_value=g.target_value,
            unit=g.unit, is_active=g.is_active, notes=g.notes,
        )
        for g in goals
    ]


# ═══════════════════════════════════════════
# DASHBOARD SUMMARY
# ═══════════════════════════════════════════

def get_dashboard_summary(db: Session) -> DashboardSummary:
    """Assemble the top-level dashboard overview."""
    # Latest weight
    latest_weight = (
        db.query(WeightEntry)
        .order_by(WeightEntry.recorded_at.desc())
        .first()
    )

    # 7-day weight change
    weight_7d = None
    if latest_weight:
        week_ago = _naive_utc(datetime.now(timezone.utc)) - timedelta(days=7)
        older = (
            db.query(WeightEntry)
            .filter(WeightEntry.recorded_at <= week_ago)
            .order_by(WeightEntry.recorded_at.desc())
            .first()
        )
        if older:
            weight_7d = round(latest_weight.weight_kg - older.weight_kg, 1)

    # Today's nutrition
    today_nutr = get_daily_nutrition(db)

    # Energy engine witness for movement: use its activity estimate when the
    # watch has not synced active calories yet. The calorie ring itself stays
    # on the plan target below (rolling burn average minus configured deficit),
    # not the live weekly-ledger target, so Overview and Plan never drift.
    _engine_activity = None
    try:
        from app.lifestyle.energy import estimate_day_energy
        _est = estimate_day_energy(db)
        if _est.get("activity_kcal"):
            _engine_activity = int(round(_est["activity_kcal"]))
    except Exception:  # pragma: no cover - dashboard must never 500 over this
        _engine_activity = None

    # Activity streak
    streak = _calculate_streak(db)

    # Sessions this week
    weekly = get_weekly_activity(db)

    # Last activity
    last_session = (
        db.query(WorkoutSession)
        .order_by(WorkoutSession.started_at.desc())
        .first()
    )

    # Targets: the metabolic engine is the SINGLE source of truth for the four
    # computed macro targets (calories / protein / carbs / fat) — the exact same
    # numbers the Plan tab shows, so the two tabs can never disagree. They are
    # derived from BMR / TDEE / weight goal, not user-set, so there is
    # deliberately NO goal override for them (a stale seeded calorie goal used
    # to shadow the computed value here). Weight target and the sugar limit DO
    # still come from active goals.
    targets = {g.goal_type: g.target_value for g in get_active_goals(db)}
    try:
        metab = get_metabolic_summary(db)
    except Exception:
        metab = None
    m_cal = getattr(metab, "target_calories", None) if metab else None
    m_pro = getattr(metab, "protein_g", None) if metab else None
    m_carb = getattr(metab, "carbs_g", None) if metab else None
    m_fat = getattr(metab, "fat_g", None) if metab else None

    # Today's wearable headline (steps / floors / active calories) if synced
    today_row = (
        db.query(DailySummary).filter(DailySummary.date == date.today()).first()
    )

    return DashboardSummary(
        current_weight_kg=latest_weight.weight_kg if latest_weight else None,
        weight_change_7d=weight_7d,
        target_weight_kg=targets.get("weight"),
        today_calories=today_nutr.total_calories,
        today_protein_g=today_nutr.total_protein_g,
        today_carbs_g=today_nutr.total_carbs_g,
        today_fat_g=today_nutr.total_fat_g,
        today_sugar_g=today_nutr.total_sugar_g,
        calories_target=m_cal or 2200,
        protein_target=m_pro or 180,
        carbs_target=m_carb,
        fat_target=m_fat,
        sugar_limit=targets.get("sugar_limit", 25),
        activity_streak_days=streak,
        sessions_this_week=weekly["session_count"],
        last_activity=last_session.started_at.isoformat() if last_session else None,
        last_activity_type=last_session.activity_type if last_session else None,
        today_steps=today_row.steps if today_row else None,
        today_floors=today_row.floors if today_row else None,
        today_active_calories=(
            today_row.active_calories
            if today_row and today_row.active_calories
            else _engine_activity
        ),
    )


def get_dashboard_history(db: Session, range_str: str = "30d") -> DashboardHistory:
    """Get time-series data for dashboard charts."""
    days_map = {"7d": 7, "14d": 14, "30d": 30, "90d": 90, "all": 365}
    days = days_map.get(range_str, 30)

    cutoff = date.today() - timedelta(days=days)

    summaries = (
        db.query(DailySummary)
        .filter(DailySummary.date >= cutoff)
        .order_by(DailySummary.date.asc())
        .all()
    )

    points = [
        DailySummaryPoint(
            date=s.date.isoformat(),
            weight_kg=s.weight_kg,
            total_calories=s.total_calories,
            total_protein_g=s.total_protein_g,
            total_sugar_g=s.total_sugar_g,
            activity_mins=s.activity_mins or 0,
        )
        for s in summaries
    ]

    return DashboardHistory(points=points, range=range_str)


# ═══════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════

def _weight_to_out(entry: WeightEntry) -> WeightEntryOut:
    return WeightEntryOut(
        id=entry.id,
        recorded_at=entry.recorded_at.isoformat(),
        weight_kg=entry.weight_kg,
        source=entry.source or "manual",
        notes=entry.notes,
    )


def _safe_json(s):
    """Parse a JSON text column to a dict, or None. Never raises."""
    if not s:
        return None
    try:
        import json
        return json.loads(s)
    except Exception:
        return None


def _nutrition_to_out(entry: NutritionLog) -> NutritionLogOut:
    return NutritionLogOut(
        id=entry.id,
        logged_at=entry.logged_at.isoformat(),
        meal_type=entry.meal_type or "meal",
        description=entry.description,
        calories=entry.calories,
        protein_g=entry.protein_g,
        carbs_g=entry.carbs_g,
        fat_g=entry.fat_g,
        fibre_g=entry.fibre_g,
        sugar_g=entry.sugar_g,
        is_estimate=entry.is_estimate,
        confidence=entry.confidence or 0.6,
        source=entry.source or "text",
        quantity_g=entry.quantity_g,
        per_100g=_safe_json(entry.per_100g_json),
        micros=_safe_json(entry.micros_json),
        food_product_id=entry.food_product_id,
        estimate_note=entry.estimate_note,
    )


def _workout_to_out(session: WorkoutSession) -> WorkoutSessionOut:
    return WorkoutSessionOut(
        id=session.id,
        started_at=session.started_at.isoformat(),
        duration_mins=session.duration_mins,
        activity_type=session.activity_type,
        title=session.title,
        notes=session.notes,
        calories_burned=session.calories_burned,
        surf_location=session.surf_location,
        surf_conditions=session.surf_conditions,
        source=session.source or "manual",
    )


def _get_active_goal_value(db: Session, goal_type: str) -> Optional[float]:
    """Get the target value for an active goal of the given type."""
    goal = (
        db.query(LifestyleGoal)
        .filter(LifestyleGoal.goal_type == goal_type, LifestyleGoal.is_active == True)
        .first()
    )
    return goal.target_value if goal else None


def _calculate_streak(db: Session) -> int:
    """Calculate consecutive days with at least one logged activity."""
    today = date.today()
    streak = 0

    for i in range(365):  # max lookback
        check_date = today - timedelta(days=i)
        start = datetime.combine(check_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end = start + timedelta(days=1)

        count = (
            db.query(func.count(WorkoutSession.id))
            .filter(WorkoutSession.started_at >= start, WorkoutSession.started_at < end)
            .scalar()
        )

        if count and count > 0:
            streak += 1
        else:
            # Allow today to be empty (day hasn't ended yet)
            if i == 0:
                continue
            break

    return streak


def _update_daily_summary(db: Session, target_date: date) -> None:
    """Upsert the daily summary row for a given date."""
    start = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    end = start + timedelta(days=1)

    # Latest weight for the day
    weight = (
        db.query(WeightEntry)
        .filter(WeightEntry.recorded_at >= start, WeightEntry.recorded_at < end)
        .order_by(WeightEntry.recorded_at.desc())
        .first()
    )

    # Nutrition totals
    nutr = (
        db.query(
            func.sum(NutritionLog.calories),
            func.sum(NutritionLog.protein_g),
            func.sum(NutritionLog.carbs_g),
            func.sum(NutritionLog.fat_g),
            func.sum(NutritionLog.sugar_g),
        )
        .filter(NutritionLog.logged_at >= start, NutritionLog.logged_at < end)
        .first()
    )

    # Activity totals
    activity = (
        db.query(
            func.sum(WorkoutSession.duration_mins),
            func.group_concat(WorkoutSession.activity_type),
        )
        .filter(WorkoutSession.started_at >= start, WorkoutSession.started_at < end)
        .first()
    )

    # Upsert
    existing = db.query(DailySummary).filter(DailySummary.date == target_date).first()
    if existing:
        summary = existing
    else:
        summary = DailySummary(date=target_date)
        db.add(summary)

    summary.weight_kg = weight.weight_kg if weight else summary.weight_kg
    summary.total_calories = nutr[0] if nutr and nutr[0] else 0
    summary.total_protein_g = nutr[1] if nutr and nutr[1] else 0
    summary.total_carbs_g = nutr[2] if nutr and nutr[2] else 0
    summary.total_fat_g = nutr[3] if nutr and nutr[3] else 0
    summary.total_sugar_g = nutr[4] if nutr and nutr[4] else 0
    summary.activity_mins = activity[0] if activity and activity[0] else 0
    summary.activity_types = activity[1] if activity and activity[1] else None

    db.commit()


# ═══════════════════════════════════════════
# PROFILE & METABOLIC
# ═══════════════════════════════════════════

def _profile_to_out(p: LifestyleProfile) -> ProfileOut:
    return ProfileOut(
        sex=p.sex,
        height_cm=p.height_cm,
        dob=p.dob.isoformat() if p.dob else None,
        age_years=p.age_years,
        activity_level=p.activity_level or "moderate",
        goal_pace=p.goal_pace or "moderate",
    )


def get_profile(db: Session) -> Optional[ProfileOut]:
    """Return the single profile row, or None if it hasn't been set up yet."""
    p = db.query(LifestyleProfile).order_by(LifestyleProfile.id.asc()).first()
    return _profile_to_out(p) if p else None


def upsert_profile(db: Session, *, sex: Optional[str] = None,
                   height_cm: Optional[float] = None, dob: Optional[str] = None,
                   age_years: Optional[int] = None,
                   activity_level: Optional[str] = None,
                   goal_pace: Optional[str] = None) -> ProfileOut:
    """Create or partially update the single profile row. Only set fields change."""
    p = db.query(LifestyleProfile).order_by(LifestyleProfile.id.asc()).first()
    if not p:
        p = LifestyleProfile()
        db.add(p)

    if sex is not None:
        p.sex = sex.strip().lower()
    if height_cm is not None:
        p.height_cm = float(height_cm)
    if dob is not None:
        p.dob = _parse_date(dob)
    if age_years is not None:
        p.age_years = int(age_years)
    if activity_level is not None:
        p.activity_level = activity_level.strip().lower()
    if goal_pace is not None:
        p.goal_pace = goal_pace.strip().lower()
    p.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(p)
    logger.info("[lifestyle] Profile updated")
    return _profile_to_out(p)


def _parse_date(value: str):
    """Parse an ISO date string (YYYY-MM-DD) to a date; None on failure."""
    try:
        return datetime.strptime(value.strip()[:10], "%Y-%m-%d").date()
    except (ValueError, AttributeError):
        return None


def _effective_age(p: LifestyleProfile) -> Optional[float]:
    """Age in years from dob if present, else the stored age_years."""
    if p.dob:
        today = date.today()
        years = today.year - p.dob.year - (
            (today.month, today.day) < (p.dob.month, p.dob.day)
        )
        return float(years)
    return float(p.age_years) if p.age_years else None


def get_metabolic_summary(db: Session) -> MetabolicSummary:
    """
    Gather profile + latest weight + recent wearable burn + weight goal, then
    compute BMR / TDEE / sustainable calorie + protein targets via metabolic.py.
    """
    from app.lifestyle import metabolic

    p = db.query(LifestyleProfile).order_by(LifestyleProfile.id.asc()).first()

    latest_weight = (
        db.query(WeightEntry).order_by(WeightEntry.recorded_at.desc()).first()
    )
    weight_kg = latest_weight.weight_kg if latest_weight else None
    goal_weight = _get_active_goal_value(db, "weight")

    # Daily burn from a rolling average of real active calories (Garmin +
    # hand-logged workouts), only trusted past a minimum sample. See burn.py.
    from app.lifestyle.burn import compute_burn_signal
    burn = compute_burn_signal(db)
    active_cals = burn.avg_active_calories if burn.trusted else None

    plan = metabolic.build_plan(
        weight_kg=weight_kg,
        height_cm=p.height_cm if p else None,
        age_years=_effective_age(p) if p else None,
        sex=p.sex if p else None,
        activity_level=(p.activity_level if p else None) or metabolic.DEFAULT_ACTIVITY,
        pace=(p.goal_pace if p else None) or metabolic.DEFAULT_PACE,
        active_calories=active_cals,
        goal_weight_kg=goal_weight,
    )

    height_cm = p.height_cm if p else None
    bmi = None
    bmi_category = None
    if weight_kg and height_cm:
        h_m = height_cm / 100.0
        if h_m > 0:
            bmi = round(weight_kg / (h_m * h_m), 1)
            bmi_category = _bmi_category(bmi)

    return MetabolicSummary(
        complete=plan.get("complete", False),
        missing=plan.get("missing", []),
        bmr=plan.get("bmr"),
        tdee=plan.get("tdee"),
        tdee_source=burn.source_label,
        activity_level=plan.get("activity_level"),
        active_calories_used=plan.get("active_calories_used"),
        target_calories=plan.get("target_calories"),
        deficit_kcal=plan.get("deficit_kcal"),
        floored_at_bmr=plan.get("floored_at_bmr", False),
        pace=plan.get("pace"),
        protein_g=plan.get("protein_g"),
        carbs_g=plan.get("carbs_g"),
        fat_g=plan.get("fat_g"),
        current_weight_kg=plan.get("current_weight_kg"),
        goal_weight_kg=plan.get("goal_weight_kg"),
        height_cm=height_cm,
        bmi=bmi,
        bmi_category=bmi_category,
    )


def _bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "underweight"
    if bmi < 25:
        return "healthy"
    if bmi < 30:
        return "overweight"
    return "obese"


# ══════════════════════════════════
# COACHING: food preferences + written plan
# (moved to app/lifestyle/coaching.py; re-exported here for back-compat)
# ══════════════════════════════════
from app.lifestyle.coaching import (  # noqa: E402,F401
    add_food_preference, update_food_preference, get_food_preferences,
    delete_food_preference, get_meal_plan, set_meal_plan,
    _PREF_CATEGORIES, _PREF_STABILITIES, _PREF_STALE_DAYS,
    _is_stale, _pref_to_out,
)


# ══════════════════════════════════
# STRENGTH: exercise-set logging + progressive-overload history
# (lives in app/lifestyle/strength.py; re-exported here for back-compat)
# ══════════════════════════════════
from app.lifestyle.strength import (  # noqa: E402,F401
    log_exercise_set, log_exercise_sets, delete_exercise_set,
    get_exercise_history, get_strength_log, list_recent_exercises,
    _normalise_exercise_name,
)

# Long-arc daily history reads (lives in app/lifestyle/history.py)
from app.lifestyle.history import get_health_history  # noqa: E402,F401
