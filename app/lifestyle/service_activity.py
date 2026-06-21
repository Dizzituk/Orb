# FILE: app/lifestyle/service_activity.py
# Purpose: Weight / workouts / goals + the shared daily-summary, streak and naive-UTC helpers.
# Called-by: app.lifestyle.service (re-export facade)
# Depends-on: app.lifestyle.models, app.lifestyle.schemas
# Last-renovated: 2026-06-21
"""
Activity layer for the Lifestyle Engine: weight tracking, workout/activity
sessions, goals, and the cross-domain helpers that the nutrition and metabolic
layers also depend on (the daily-summary upsert, the activity streak, the
active-goal lookup, and naive-UTC coercion). Split out of service.py; the
facade re-exports every public name here for back-compat.
"""
import logging
from datetime import datetime, date, timedelta, timezone
from typing import Optional, List

from sqlalchemy import func, desc
from sqlalchemy.orm import Session

from app.lifestyle.models import (
    WeightEntry, NutritionLog, WorkoutSession,
    LifestyleGoal, DailySummary,
)
from app.lifestyle.schemas import (
    WeightEntryOut, WeightTrend,
    WorkoutSessionOut,
    GoalOut,
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


def _weight_to_out(entry: WeightEntry) -> WeightEntryOut:





    return WeightEntryOut(





        id=entry.id,





        recorded_at=entry.recorded_at.isoformat(),





        weight_kg=entry.weight_kg,





        source=entry.source or "manual",





        notes=entry.notes,





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
