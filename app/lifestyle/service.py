# FILE: app/lifestyle/service.py
# Purpose: Facade for the Lifestyle Engine service layer (split 2026-06-21, BATCH 10).
# Called-by: app.lifestyle.router, app.lifestyle.daily_view, app.bridge.dashboards (+ more)
# Depends-on: app.lifestyle.service_activity, app.lifestyle.service_nutrition, app.lifestyle.service_metabolic, app.lifestyle.coaching, app.lifestyle.strength, app.lifestyle.history
# Last-renovated: 2026-06-21
"""
Core service layer for the Lifestyle Engine (facade).

Single-responsibility implementations now live in:
- service_activity.py  - weight / workouts / goals + shared daily-summary,
                         streak and naive-UTC helpers
- service_nutrition.py - food logging, itemisation, prep-split, daily/history reads
- service_metabolic.py - profile + BMR/TDEE metabolic summary

This module keeps the cross-domain dashboard assembly (get_dashboard_summary /
get_dashboard_history) and re-exports the full public surface so existing
`from app.lifestyle.service import X` and `service.X` access keep working
unchanged. Coaching / strength / history stay re-exported here as before.
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

# --- Single-responsibility implementations, re-exported for back-compat ---
from app.lifestyle.service_activity import (  # noqa: E402,F401
    _naive_utc, log_weight, get_weight_trend, log_workout,
    get_recent_workouts, get_weekly_activity, set_goal, get_active_goals,
    _weight_to_out, _workout_to_out, _get_active_goal_value,
    _calculate_streak, _update_daily_summary,
)
from app.lifestyle.service_nutrition import (  # noqa: E402,F401
    log_nutrition, log_nutrition_items, _coerce_num,
    log_nutrition_items_checked, prep_split, _to_dict_safe,
    get_daily_nutrition, get_nutrition_history, delete_nutrition,
    _safe_json, _nutrition_to_out,
)
from app.lifestyle.service_metabolic import (  # noqa: E402,F401
    _profile_to_out, get_profile, upsert_profile, _parse_date,
    _effective_age, get_metabolic_summary, _bmi_category,
)


# ===========================================
# DASHBOARD SUMMARY (cross-domain assembly - kept in the facade)
# ===========================================


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
