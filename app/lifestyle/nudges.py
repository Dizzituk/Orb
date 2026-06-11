# FILE: app/lifestyle/nudges.py
"""
Proactive daily nudges — Job 5 of the memory roadmap (2026-06-10).

Evaluates today's intake/activity against the active LifestyleGoal targets
and produces at most two short, natural coaching nudges, e.g. nothing
logged by early afternoon, intake well under target late in the day,
over target with sugar running hot, protein lagging by evening.

Delivery: the scheduler (app/lifestyle/scheduler.py) writes the active
nudge to data/lifestyle/nudges_today.json; memory_injection picks it up
via get_nudge_for_injection() and weaves it into the next conversation
(with a cooldown so it appears once, not on every message). Also exposed
for the bridge/desktop to poll if wanted.

This is the template for proactivity in every other domain later.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "lifestyle"
_NUDGE_FILE = _DATA_DIR / "nudges_today.json"
_INJECTION_COOLDOWN_HOURS = 4


def _now() -> datetime:
    return datetime.now(timezone.utc)


# =============================================================================
# Today's numbers
# =============================================================================

def _today_totals(db) -> Dict[str, Any]:
    from app.lifestyle.models import NutritionLog, WorkoutSession

    start = _now().replace(hour=0, minute=0, second=0, microsecond=0)
    logs = db.query(NutritionLog).filter(NutritionLog.logged_at >= start).all()
    workouts = db.query(WorkoutSession).filter(WorkoutSession.started_at >= start).all()

    return {
        "log_count": len(logs),
        "calories": sum(l.calories or 0 for l in logs),
        "protein_g": sum(l.protein_g or 0 for l in logs),
        "sugar_g": sum(l.sugar_g or 0 for l in logs),
        "last_logged_at": max((l.logged_at for l in logs), default=None),
        "activity_mins": sum(w.duration_mins or 0 for w in workouts),
    }


def _targets(db) -> Dict[str, float]:
    from app.lifestyle.models import LifestyleGoal
    out: Dict[str, float] = {}
    for goal in db.query(LifestyleGoal).filter(LifestyleGoal.is_active == True).all():  # noqa: E712
        out[goal.goal_type] = goal.target_value
    return out


def _staple_suggestion(db) -> Optional[str]:
    """Pull a known staple so suggestions reference his real food."""
    try:
        from app.lifestyle.models import FoodPreference
        pref = (
            db.query(FoodPreference)
            .filter(FoodPreference.category == "staple")
            .order_by(FoodPreference.confidence.desc())
            .first()
        )
        if pref and pref.content:
            # content shape: "Regularly eats X — on N of the last M days"
            text = pref.content.replace("Regularly eats ", "")
            return text.split(" — ")[0].strip() or None
    except Exception:
        pass
    return None


# =============================================================================
# Rules
# =============================================================================

def evaluate_nudges(db, now: Optional[datetime] = None) -> List[str]:
    now_local = (now or datetime.now()).astimezone()
    hour = now_local.hour + now_local.minute / 60.0

    totals = _today_totals(db)
    targets = _targets(db)
    staple = _staple_suggestion(db)
    nudges: List[str] = []

    calorie_target = targets.get("calories")
    protein_target = targets.get("protein")
    sugar_limit = targets.get("sugar_limit")

    # 1. Nothing logged well into the day
    if totals["log_count"] == 0 and hour >= 13.0:
        nudges.append(
            "Nothing's been logged today yet — worth a quick voice note of whatever "
            "you've eaten so the day's picture stays honest."
        )

    # 2. Running well under target late in the day
    elif calorie_target and hour >= 15.0 and totals["calories"] < calorie_target * 0.5:
        suggestion = f" — something like {staple} would do the job" if staple else ""
        nudges.append(
            f"You're at roughly {totals['calories']:.0f} of your {calorie_target:.0f} kcal "
            f"target with most of the day gone. Worth getting a proper meal in{suggestion}."
        )

    # 3. Over target
    if calorie_target and totals["calories"] > calorie_target * 1.1:
        extra = ""
        if sugar_limit and totals["sugar_g"] > sugar_limit:
            extra = " Sugar's also over the limit today, so steer clear of the sweet stuff."
        nudges.append(
            f"Today's intake is past the {calorie_target:.0f} kcal target. If hunger bites "
            f"later, lean on protein and veg rather than carbs.{extra}"
        )

    # 4. Protein lagging by evening
    if protein_target and hour >= 19.0 and totals["protein_g"] < protein_target * 0.6:
        nudges.append(
            f"Protein's sitting at {totals['protein_g']:.0f}g against a {protein_target:.0f}g "
            f"target — a protein-forward dinner or snack would square the day off."
        )

    # 5. Sugar over limit on its own
    if sugar_limit and totals["sugar_g"] > sugar_limit and not any("Sugar" in n for n in nudges):
        nudges.append(
            f"Sugar's at {totals['sugar_g']:.0f}g today against your {sugar_limit:.0f}g limit — "
            f"one to watch for the rest of the day."
        )

    return nudges[:2]


# =============================================================================
# Persistence + injection handoff
# =============================================================================

def write_active_nudges(nudges: List[str]) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": datetime.now().astimezone().strftime("%Y-%m-%d"),
        "generated_at": _now().isoformat(),
        "items": nudges,
        "last_injected_at": None,
    }
    _NUDGE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if nudges:
        logger.info("[nudges] active: %s", nudges)


def _load() -> Optional[Dict[str, Any]]:
    if not _NUDGE_FILE.exists():
        return None
    try:
        return json.loads(_NUDGE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_active_nudges() -> List[str]:
    """All of today's nudges, no cooldown — for the bridge/desktop to poll."""
    data = _load()
    if not data or data.get("date") != datetime.now().astimezone().strftime("%Y-%m-%d"):
        return []
    return list(data.get("items") or [])


def get_nudge_for_injection() -> Optional[str]:
    """First active nudge for prompt injection, honouring a cooldown so it
    appears in one conversation turn, not every message. Stamps the read."""
    data = _load()
    if not data or data.get("date") != datetime.now().astimezone().strftime("%Y-%m-%d"):
        return None
    items = data.get("items") or []
    if not items:
        return None
    last = data.get("last_injected_at")
    if last:
        try:
            last_dt = datetime.fromisoformat(last)
            if _now() - last_dt < timedelta(hours=_INJECTION_COOLDOWN_HOURS):
                return None
        except Exception:
            pass
    data["last_injected_at"] = _now().isoformat()
    try:
        _NUDGE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass
    return " ".join(items) if len(items) > 1 else items[0]


def run_nudge_evaluation(db=None) -> List[str]:
    from app.db import SessionLocal
    close = False
    if db is None:
        db = SessionLocal()
        close = True
    try:
        nudges = evaluate_nudges(db)
        write_active_nudges(nudges)
        return nudges
    finally:
        if close:
            db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(run_nudge_evaluation())
