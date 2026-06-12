# FILE: app/lifestyle/energy.py
# Purpose: Context-aware daily energy engine — prices the day's burn from the
#          best available evidence (Garmin active kcal > context-priced steps >
#          logged sessions > sedentary floor), with work-day awareness.
# Called-by: app.lifestyle.energy_ledger, app.lifestyle.energy_router,
#            app.lifestyle.nudges, app.llm.stream_router (quick debrief)
# Depends-on: app.lifestyle.models, app.lifestyle.metabolic, app.finance.models_workday
"""
The watch only knows the wrist; ASTRA knows the life. This module turns steps,
logged workouts, the finance Work tab and a one-line spoken debrief into an
honest daily burn figure.

Evidence hierarchy (best wins, never mixed so nothing double-counts):
  1. garmin_active   — DailySummary.active_calories synced from the watch.
  2. context_steps   — steps priced by body weight and day context: on a
                       confirmed work day (finance WorkDay row) steps cost more
                       (loaded carries, van climbs, stop-start), scaled by the
                       day's spoken effort. Non-step workouts (surf, gym, swim)
                       are added; step-based ones (run/walk) are NOT, because
                       their energy is already inside the step count.
  3. sessions_only   — logged workouts alone (watch missed the day).
  4. sedentary_floor — BMR * 0.2, the bare-minimum NEAT allowance.

Every figure is labelled with its method and confidence. All assumptions err
UNDER, never over: a diet tool that overestimates burn quietly feeds you
maintenance while claiming a deficit, and that destroys trust. The fortnightly
scale-feedback calibration (energy_ledger) corrects upward over time.

Day effort lives in data/lifestyle/day_effort.json (same pattern as the nudge
state files): {"YYYY-MM-DD": {"effort": "heavy", "source": "voice", ...}}.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.lifestyle import metabolic
from app.lifestyle.models import (
    DailySummary, LifestyleProfile, WeightEntry, WorkoutSession,
)

logger = logging.getLogger(__name__)

EFFORT_PATH = Path("data/lifestyle/day_effort.json")

EFFORT_LEVELS = ("light", "normal", "heavy", "very_heavy")

# Net active kcal per step per kg of body weight (ordinary walking).
# 90 kg x 7,000 steps ~= 220 kcal — deliberately conservative.
NET_KCAL_PER_STEP_PER_KG = 0.00035

# Step-cost multipliers. Work days price higher than off days (carrying,
# hundreds of van entries/exits, stop-start on drives) but the floor stays
# modest: most parcels are light, and volume matters more than load.
WORK_STEP_MULT = {"light": 1.10, "normal": 1.25, "heavy": 1.45, "very_heavy": 1.60}
OFF_STEP_MULT = {"light": 0.95, "normal": 1.00, "heavy": 1.20, "very_heavy": 1.35}

# Occupational uplift for confirmed work days (2026-06-11 v2, PT-audited):
# the job is more than its steps — hours of driving, depot loading, parcels
# handled twice, van climbs. Steps alone priced a 9-hour physical shift at
# ~520 kcal, which is just the walking. So on CLOSED work days steps are
# priced at the plain walking rate and the shift's NON-walking hours get an
# incremental-MET term (above resting), scaled by spoken effort — a blend of
# driving (~1.5 MET) and handling bursts (3+), kept conservative per the
# err-under doctrine; the fortnightly scales audit arbitrates.
WORK_OCC_MET = {"light": 0.35, "normal": 0.45, "heavy": 0.55, "very_heavy": 0.65}
WALKING_STEPS_PER_HOUR = 6000.0  # ~100 steps/min, to split shift hours
DEFAULT_WORK_EFFORT = "normal"
DEFAULT_OFF_EFFORT = "normal"

# MET values for logged sessions without watch calories (kcal = MET*kg*hours).
# Conservative ends of the published ranges.
MET_VALUES: Dict[str, float] = {
    "surf": 4.5, "bodyboard": 4.0, "swim": 6.0, "gym": 4.0, "strength": 4.0,
    "weights": 4.0, "cycle": 7.0, "bike": 7.0, "hiit": 8.0, "run": 9.0,
    "jog": 7.0, "walk": 3.3, "hike": 5.3, "stretch": 2.3, "yoga": 2.5,
    "football": 7.0, "climb": 7.5,
}
DEFAULT_MET = 4.0
# Their energy already lives in the step count — never add on top of steps.
STEP_BASED_TYPES = {"run", "jog", "walk", "hike"}


# ---------------------------------------------------------------------------
# Day-effort store (JSON, schema-risk-free)
# ---------------------------------------------------------------------------

def _load_efforts() -> Dict[str, Any]:
    try:
        if EFFORT_PATH.exists():
            return json.loads(EFFORT_PATH.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[energy] effort store unreadable: %s", exc)
    return {}


def _save_efforts(data: Dict[str, Any]) -> None:
    try:
        EFFORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        EFFORT_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[energy] effort store write failed: %s", exc)


def normalise_effort(value: str) -> Optional[str]:
    v = (value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if v in EFFORT_LEVELS:
        return v
    if v in ("very_light", "easy", "quiet", "chilled", "cruisy"):
        return "light"
    if v in ("steady", "average", "usual", "standard", "ok", "okay", "medium"):
        return "normal"
    if v in ("big", "full_on", "busy", "hard", "tough"):
        return "heavy"
    if v in ("brutal", "mental", "rammed", "slammed", "savage", "massive"):
        return "very_heavy"
    return None


def get_day_effort(day: date) -> Optional[Dict[str, Any]]:
    return _load_efforts().get(day.isoformat())


def set_day_effort(day: date, effort: str, source: str = "chat",
                   notes: Optional[str] = None) -> Dict[str, Any]:
    level = normalise_effort(effort)
    if level is None:
        raise ValueError(f"unknown effort level: {effort!r}")
    data = _load_efforts()
    rec = {
        "effort": level,
        "source": source,
        "notes": notes,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    data[day.isoformat()] = rec
    _save_efforts(data)
    logger.info("[energy] day effort %s -> %s (%s)", day, level, source)
    return rec


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _profile_inputs(db: Session) -> Dict[str, Any]:
    p = db.query(LifestyleProfile).order_by(LifestyleProfile.id.asc()).first()
    latest = db.query(WeightEntry).order_by(WeightEntry.recorded_at.desc()).first()
    weight = latest.weight_kg if latest else None
    age = None
    if p is not None:
        try:
            from app.lifestyle.service import _effective_age
            age = _effective_age(p)
        except Exception:  # pragma: no cover - defensive
            age = getattr(p, "age_years", None)
    bmr = metabolic.compute_bmr(
        weight, p.height_cm if p else None, age, p.sex if p else None,
    ) if p else None
    return {
        "weight_kg": weight,
        "bmr": bmr,
        "age_years": age,
        "sex": (p.sex if p else None),
        "pace": (getattr(p, "goal_pace", None) if p else None) or metabolic.DEFAULT_PACE,
        "activity_level": (getattr(p, "activity_level", None) if p else None)
                          or metabolic.DEFAULT_ACTIVITY,
    }


def _work_day_info(db: Session, day: date) -> Optional[Dict[str, Any]]:
    """A finance WorkDay row for the date = a confirmed delivery day."""
    try:
        from app.finance.models_workday import WorkDay
        row = db.query(WorkDay).filter(WorkDay.work_date == day).first()
    except Exception as exc:  # pragma: no cover - finance optional
        logger.debug("[energy] work-day lookup unavailable: %s", exc)
        return None
    if row is None:
        return None
    return {
        "parcels": row.parcels or 0,
        "net_hours": row.net_hours,
        "status": row.status,
    }


def _workouts_for_day(db: Session, day: date) -> List[WorkoutSession]:
    start = datetime.combine(day, datetime.min.time()).replace(tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return (
        db.query(WorkoutSession)
        .filter(WorkoutSession.started_at >= start, WorkoutSession.started_at < end)
        .all()
    )


def _session_kcal(s: WorkoutSession, weight_kg: Optional[float],
                  age: Optional[float] = None, sex: Optional[str] = None) -> float:
    """Session calories, best evidence first:
      1. Watch-reported calories_burned.
      2. Heart-rate priced (Keytel et al. 2005) when the session carries an
         average HR — a hard 90-minute surf with HR pinned at 150 prices as
         the anaerobic workout it was, not a flat MET average. A small CAPPED
         afterburn (EPOC) term is added only for genuinely intense sessions
         (~8%, max 80 kcal) — real but modest, never the adverts' number.
      3. MET table fallback.
    """
    if s.calories_burned and s.calories_burned > 0:
        return float(s.calories_burned)
    if not weight_kg or not s.duration_mins:
        return 0.0

    avg_hr = getattr(s, "avg_hr", None)
    max_hr = getattr(s, "max_hr", None)
    if avg_hr and avg_hr >= 90 and age:
        sx = (sex or "").strip().lower()
        if sx in ("f", "female", "woman"):
            kcal_min = (-20.4022 + 0.4472 * avg_hr - 0.1263 * weight_kg
                        + 0.074 * age) / 4.184
        else:
            kcal_min = (-55.0969 + 0.6309 * avg_hr + 0.1988 * weight_kg
                        + 0.2017 * age) / 4.184
        kcal = max(0.0, kcal_min) * s.duration_mins
        if (avg_hr >= 140) or (max_hr and max_hr >= 160):
            kcal += min(kcal * 0.08, 80.0)  # capped EPOC
        return kcal

    met = MET_VALUES.get((s.activity_type or "").strip().lower(), DEFAULT_MET)
    return met * weight_kg * (s.duration_mins / 60.0)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

def estimate_day_energy(db: Session, day: Optional[date] = None) -> Dict[str, Any]:
    """Best-evidence burn estimate for a day. Never double-counts sources."""
    day = day or date.today()
    prof = _profile_inputs(db)
    bmr = prof["bmr"]
    weight = prof["weight_kg"]

    row = db.query(DailySummary).filter(DailySummary.date == day).first()
    work = _work_day_info(db, day)
    effort_rec = get_day_effort(day)
    effort = (effort_rec or {}).get("effort") or (
        DEFAULT_WORK_EFFORT if work else DEFAULT_OFF_EFFORT
    )

    garmin_active = None
    steps = None
    if row is not None:
        if row.active_calories and row.active_calories > 0:
            garmin_active = float(row.active_calories)
        if row.steps and row.steps > 0:
            steps = int(row.steps)

    sessions = _workouts_for_day(db, day)
    components: Dict[str, Any] = {}

    if garmin_active is not None:
        activity = garmin_active
        method, confidence = "garmin_active", "high"
        components["garmin_active"] = round(garmin_active)
    elif steps and weight:
        wkout_kcal = sum(
            _session_kcal(s, weight, prof.get("age_years"), prof.get("sex"))
            for s in sessions
            if (s.activity_type or "").strip().lower() not in STEP_BASED_TYPES
        )
        net_hours = float((work or {}).get("net_hours") or 0.0)
        if work and net_hours > 0:
            # Closed work day: plain-rate steps + occupational hours term.
            step_kcal = steps * NET_KCAL_PER_STEP_PER_KG * weight
            walking_hours = steps / WALKING_STEPS_PER_HOUR
            occ_hours = max(0.0, net_hours - walking_hours)
            occ_met = WORK_OCC_MET.get(effort, WORK_OCC_MET["normal"])
            occ_kcal = occ_met * weight * occ_hours
            activity = step_kcal + occ_kcal + wkout_kcal
            components.update({
                "step_kcal": round(step_kcal), "steps": steps,
                "occupational_kcal": round(occ_kcal),
                "occ_hours": round(occ_hours, 1), "occ_met": occ_met,
                "workout_kcal": round(wkout_kcal),
            })
        else:
            # Off day — or a work day still open (no hours yet): the old
            # carry-multiplier pricing, so the live estimate never collapses
            # mid-shift and simply firms up when the day is closed.
            mult_table = WORK_STEP_MULT if work else OFF_STEP_MULT
            mult = mult_table.get(effort, mult_table["normal"])
            step_kcal = steps * NET_KCAL_PER_STEP_PER_KG * weight * mult
            activity = step_kcal + wkout_kcal
            components.update({
                "step_kcal": round(step_kcal), "steps": steps,
                "step_multiplier": mult, "workout_kcal": round(wkout_kcal),
            })
        method, confidence = "context_steps", "medium"
    else:
        wkout_kcal = sum(_session_kcal(s, weight, prof.get("age_years"), prof.get("sex"))
                         for s in sessions)
        if wkout_kcal > 0:
            activity = wkout_kcal
            method, confidence = "sessions_only", "low"
            components["workout_kcal"] = round(wkout_kcal)
        elif bmr:
            activity = bmr * 0.2
            method, confidence = "sedentary_floor", "low"
        else:
            activity = 0.0
            method, confidence = "incomplete_profile", "none"

    total = round(bmr + activity) if bmr else None
    labels = {
        "garmin_active": "measured (watch)",
        "context_steps": "estimated from steps + context",
        "sessions_only": "estimated from logged sessions",
        "sedentary_floor": "estimated (no data — sedentary floor)",
        "incomplete_profile": "profile incomplete",
    }
    return {
        "date": day.isoformat(),
        "bmr": round(bmr) if bmr else None,
        "activity_kcal": round(activity) if bmr or activity else 0,
        "total_burn": total,
        "method": method,
        "confidence": confidence,
        "label": labels.get(method, method),
        "is_work_day": bool(work),
        "work": work,
        "effort": effort,
        "effort_source": (effort_rec or {}).get("source") or "default",
        "sessions_logged": len(sessions),
        "components": components,
        "weight_kg": weight,
    }


# ---------------------------------------------------------------------------
# Deterministic voice debrief ("heavy day today" -> logged, repriced)
# ---------------------------------------------------------------------------

_DEBRIEF_CONTEXT = re.compile(r"\b(day|round|route|shift|graft|debrief)\b", re.I)
_DEBRIEF_EFFORT = re.compile(
    r"\b(very\s+heavy|really\s+heavy|brutal|mental|rammed|slammed|massive|"
    r"heavy|full\s+on|big|busy|tough|hard|"
    r"light|easy|quiet|chilled|cruisy|"
    r"steady|normal|average|usual)\b", re.I,
)


def try_quick_debrief(db: Session, text: str) -> Optional[str]:
    """Deterministic end-of-day debrief from natural speech. Returns a reply
    string when it confidently matched, else None (caller falls through to the
    LLM). Conservative: needs both a day-context word and an effort word, and
    ignores questions so "how was my day" never self-answers.
    """
    t = (text or "").strip()
    if not t or len(t) > 280 or t.endswith("?"):
        return None
    low = t.lower()
    if low.startswith(("how", "what", "when", "did", "was", "is ", "are ")):
        return None
    # Never hijack work logging (2026-06-11 shadowing bug): figures, money or
    # work-record words mean this is an end-of-day log, not an effort remark.
    if re.search(r"\d", low) or re.search(
            r"\b(log|logged|mile|miles|odometer|parcel|parcels|collection|"
            r"pay|paid|earn|earned|fuel|expense|invoice|£|pounds?)\b", low):
        return None
    if not _DEBRIEF_CONTEXT.search(low):
        return None
    m = _DEBRIEF_EFFORT.search(low)
    if not m:
        return None
    level = normalise_effort(m.group(1))
    if level is None:
        return None

    today = date.today()
    set_day_effort(today, level, source="voice", notes=t[:200])
    est = estimate_day_energy(db, today)

    target_line = ""
    try:
        from app.lifestyle.energy_ledger import compute_today_target
        tgt = compute_today_target(db)
        if tgt.get("target_calories"):
            target_line = (
                f" Intake target for today is about {int(tgt['target_calories'])} kcal"
                f" ({tgt.get('week_note', 'weekly ledger')})."
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("[energy] target unavailable in debrief: %s", exc)

    nice = level.replace("_", " ")
    burn_bit = (
        f"burn estimate now around {est['total_burn']} kcal ({est['label']})"
        if est.get("total_burn") else "burn estimate updated"
    )
    work_bit = ""
    if est.get("is_work_day") and (est.get("work") or {}).get("parcels"):
        work_bit = f" across {est['work']['parcels']} parcels"
    return f"Logged — {nice} day{work_bit}. Today's {burn_bit}.{target_line}"
