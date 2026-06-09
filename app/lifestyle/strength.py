# FILE: app/lifestyle/strength.py
"""
Strength / resistance-training layer for the Lifestyle Engine.

The unit is a single logged set (app.lifestyle.models.ExerciseSet): an
exercise name, a weight, and reps, stamped with a time. Sets group by a
normalised `exercise_name` so "what did I lift last time?" and
progressive-overload history are cheap, indexed lookups, and group by day
for the Fitness tab's day-to-day workout log.

Kept separate from service.py so that file stays small and single-purpose;
service.py re-exports the public functions for back-compat.
"""
import logging
import re
from datetime import datetime, date, timedelta, timezone
from typing import Optional, List

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.lifestyle.models import ExerciseSet
from app.lifestyle.schemas import (
    ExerciseSetOut, ExerciseSession, ExerciseHistory,
    StrengthDay, StrengthLog,
)

logger = logging.getLogger(__name__)


def _naive_utc(dt):
    if dt is None:
        return None
    if getattr(dt, "tzinfo", None) is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _normalise_exercise_name(text: str) -> str:
    """Normalise a spoken exercise name into a stable grouping key.

    Lowercase, trim, collapse internal whitespace, drop trailing
    punctuation. Conservative on purpose — no singular/plural folding —
    so distinct lifts never collide. 'Bench Press ' -> 'bench press'.
    """
    s = (text or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" .,:;!?-")
    return s


def _est_1rm(weight_kg: Optional[float], reps: Optional[int]) -> Optional[float]:
    """Epley one-rep-max estimate: w * (1 + reps/30). Needs both values."""
    if not weight_kg or not reps or reps <= 0:
        return None
    return round(weight_kg * (1 + reps / 30.0), 1)


def _set_to_out(s: ExerciseSet) -> ExerciseSetOut:
    return ExerciseSetOut(
        id=s.id,
        performed_at=s.performed_at.isoformat() if s.performed_at else "",
        exercise_name=s.exercise_name,
        exercise_display=s.exercise_display,
        weight_kg=s.weight_kg,
        reps=s.reps,
        set_index=s.set_index,
        unit=s.unit or "kg",
        notes=s.notes,
    )


def _day_of(s: ExerciseSet) -> date:
    d = _naive_utc(s.performed_at) or datetime.utcnow()
    return d.date()


def _session_from_sets(sets: List[ExerciseSet]) -> ExerciseSession:
    """Build an ExerciseSession (one exercise, one day) from its set rows."""
    ordered = sorted(sets, key=lambda x: (x.set_index or 0, x.performed_at or datetime.min))
    weights = [s.weight_kg for s in ordered if s.weight_kg is not None]
    total_reps = sum(s.reps or 0 for s in ordered)
    return ExerciseSession(
        date=_day_of(ordered[0]).isoformat() if ordered else "",
        sets=[_set_to_out(s) for s in ordered],
        top_weight_kg=max(weights) if weights else None,
        total_reps=total_reps,
    )


# ─────────────────────────────────────────────
# WRITE
# ─────────────────────────────────────────────

def log_exercise_set(db: Session, exercise_name: str, *, weight_kg: Optional[float] = None,
                     reps: Optional[int] = None, set_index: Optional[int] = None,
                     unit: str = "kg", notes: Optional[str] = None,
                     display: Optional[str] = None, source: str = "voice",
                     session_id: Optional[int] = None) -> ExerciseSetOut:
    """Log a single set. Returns the stored set."""
    key = _normalise_exercise_name(exercise_name)
    row = ExerciseSet(
        exercise_name=key,
        exercise_display=(display or exercise_name or "").strip() or None,
        weight_kg=weight_kg,
        reps=reps,
        set_index=set_index,
        unit=(unit or "kg"),
        notes=notes,
        source=source,
        workout_session_id=session_id,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    logger.info(f"[strength] Set logged: {key} {weight_kg}kg x{reps}")
    return _set_to_out(row)


def log_exercise_sets(db: Session, exercise_name: str, sets: List[dict], *,
                      unit: str = "kg", notes: Optional[str] = None,
                      display: Optional[str] = None, source: str = "voice",
                      session_id: Optional[int] = None) -> List[ExerciseSetOut]:
    """Log several sets of one exercise at once.

    `sets` is a list of {weight_kg, reps}. set_index continues from any
    sets of the same exercise already logged today, so multiple calls in
    one gym session number 1, 2, 3… in order.
    """
    key = _normalise_exercise_name(exercise_name)
    today = date.today()
    start = datetime.combine(today, datetime.min.time())
    existing_today = (
        db.query(func.count(ExerciseSet.id))
        .filter(ExerciseSet.exercise_name == key,
                ExerciseSet.performed_at >= start)
        .scalar()
    ) or 0

    out: List[ExerciseSetOut] = []
    for i, s in enumerate(sets, start=1):
        w = s.get("weight_kg")
        r = s.get("reps")
        try:
            w = float(w) if w is not None else None
        except (TypeError, ValueError):
            w = None
        try:
            r = int(r) if r is not None else None
        except (TypeError, ValueError):
            r = None
        row = ExerciseSet(
            exercise_name=key,
            exercise_display=(display or exercise_name or "").strip() or None,
            weight_kg=w,
            reps=r,
            set_index=existing_today + i,
            unit=(unit or "kg"),
            notes=notes,
            source=source,
            workout_session_id=session_id,
        )
        db.add(row)
        out.append(row)
    db.commit()
    for row in out:
        db.refresh(row)
    logger.info(f"[strength] Logged {len(out)} set(s) of {key}")
    return [_set_to_out(r) for r in out]


def delete_exercise_set(db: Session, set_id: int) -> bool:
    row = db.query(ExerciseSet).filter(ExerciseSet.id == set_id).first()
    if not row:
        return False
    db.delete(row)
    db.commit()
    logger.info(f"[strength] Set {set_id} deleted")
    return True


# ─────────────────────────────────────────────
# READ
# ─────────────────────────────────────────────

def get_exercise_history(db: Session, exercise_name: str,
                         limit_sessions: int = 8) -> ExerciseHistory:
    """Progressive-overload view of one exercise: recent sessions newest
    first, plus last/best weight and best estimated 1RM."""
    key = _normalise_exercise_name(exercise_name)
    rows = (
        db.query(ExerciseSet)
        .filter(ExerciseSet.exercise_name == key)
        .order_by(ExerciseSet.performed_at.desc())
        .all()
    )
    if not rows:
        return ExerciseHistory(exercise_name=key, sessions=[])

    # Group by day
    by_day: dict = {}
    for s in rows:
        by_day.setdefault(_day_of(s), []).append(s)

    days_sorted = sorted(by_day.keys(), reverse=True)
    sessions = [_session_from_sets(by_day[d]) for d in days_sorted[:limit_sessions]]

    all_weights = [s.weight_kg for s in rows if s.weight_kg is not None]
    best_weight = max(all_weights) if all_weights else None
    best_1rm = None
    for s in rows:
        est = _est_1rm(s.weight_kg, s.reps)
        if est is not None and (best_1rm is None or est > best_1rm):
            best_1rm = est

    return ExerciseHistory(
        exercise_name=key,
        sessions=sessions,
        last_performed=sessions[0].date if sessions else None,
        last_top_weight_kg=sessions[0].top_weight_kg if sessions else None,
        best_weight_kg=best_weight,
        best_est_1rm_kg=best_1rm,
    )


def get_strength_log(db: Session, days: int = 21) -> StrengthLog:
    """Day-by-day strength log across all exercises (Fitness-tab feed).

    Newest day first; within a day, one ExerciseSession per exercise.
    """
    cutoff = datetime.combine(date.today() - timedelta(days=days), datetime.min.time())
    rows = (
        db.query(ExerciseSet)
        .filter(ExerciseSet.performed_at >= cutoff)
        .order_by(ExerciseSet.performed_at.asc())
        .all()
    )

    # day -> exercise_name -> [sets]
    by_day: dict = {}
    for s in rows:
        d = _day_of(s)
        by_day.setdefault(d, {}).setdefault(s.exercise_name, []).append(s)

    days_out: List[StrengthDay] = []
    for d in sorted(by_day.keys(), reverse=True):
        ex_map = by_day[d]
        sessions = [_session_from_sets(ex_map[name]) for name in ex_map]
        # order exercises by their first set time that day
        sessions.sort(key=lambda s: s.sets[0].performed_at if s.sets else "")
        days_out.append(StrengthDay(date=d.isoformat(), exercises=sessions))

    return StrengthLog(days=days_out)


def list_recent_exercises(db: Session, limit: int = 12) -> List[str]:
    """Distinct exercise names, most recently performed first."""
    rows = (
        db.query(ExerciseSet.exercise_name,
                 func.max(ExerciseSet.performed_at).label("last"))
        .group_by(ExerciseSet.exercise_name)
        .order_by(func.max(ExerciseSet.performed_at).desc())
        .limit(limit)
        .all()
    )
    return [r[0] for r in rows]
