# FILE: app/tools/lifestyle_tools.py
# Purpose: Lifestyle tool handlers — health coaching capability for the LLM.
# Called-by: app.debug.gemini_lifestyle_tools, app.tools.lifestyle_history_tools, app.tools.lifestyle_nutrition_tools, app.tools.lifestyle_strength_tools (+1 more)
# Depends-on: app.db, app.lifestyle.models, app.lifestyle.service, app.tools.lifestyle_history_tools (+4 more)
# Last-renovated: 2026-06-11
"""
Lifestyle tool handlers — health coaching capability for the LLM.

Five tools that surface the existing app.lifestyle.service functions
to any ASTRA chat. The LLM calls these to read recent nutrition,
weight, and workouts, and to log new nutrition/workout entries.

Why a separate module: keeps the per-domain handlers self-contained
so registry.py stays focused on tool execution infrastructure.
Adding new lifestyle capabilities later means editing this file only.

Pattern: each handler opens a short-lived SQLAlchemy session, calls
the matching service function, converts the Pydantic response to a
plain dict, and closes the session. Errors are caught and returned
in {ok: false, error: ...} shape so the LLM can react rather than
the tool call failing outright.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import date as _date
from typing import Optional

logger = logging.getLogger(__name__)


@contextmanager
def _db_session():
    """Yield a short-lived DB session, always closing on exit."""
    from app.db import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _to_dict(pydantic_obj) -> dict:
    """Convert a Pydantic v1 or v2 model to a plain dict."""
    if hasattr(pydantic_obj, "model_dump"):
        return pydantic_obj.model_dump()
    if hasattr(pydantic_obj, "dict"):
        return pydantic_obj.dict()
    return dict(pydantic_obj)


# ────────────────────────────────────────────────────────────────────────
# READS
# ────────────────────────────────────────────────────────────────────────

async def get_recent_nutrition_handler(input_data: dict, context: Optional[dict]) -> dict:
    """
    Return per-day nutrition summaries for the last N days.

    days_back defaults to 1 (today only). Returns days in chronological
    order so the most recent day is last in the list.
    """
    days_back = int(input_data.get("days_back") or 1)
    days_back = max(1, min(14, days_back))

    try:
        from app.lifestyle.service import get_nutrition_history
        with _db_session() as db:
            history = get_nutrition_history(db, days=days_back)
    except Exception as exc:
        logger.exception("[lifestyle_tools] get_recent_nutrition failed")
        return {"days": [], "error": str(exc)}

    days_out = []
    for day in history:
        d = _to_dict(day)
        days_out.append({
            "date": d.get("date"),
            "totals": {
                "calories": d.get("total_calories", 0),
                "protein_g": d.get("total_protein_g", 0),
                "carbs_g": d.get("total_carbs_g", 0),
                "fat_g": d.get("total_fat_g", 0),
                "sugar_g": d.get("total_sugar_g", 0),
            },
            "meal_count": d.get("meal_count", 0),
            "logs": d.get("logs", []),
        })

    return {"days": days_out}


async def get_weight_trend_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Return the user's weight history and trend deltas."""
    days = int(input_data.get("days") or 30)
    days = max(7, min(365, days))

    try:
        from app.lifestyle.service import get_weight_trend
        with _db_session() as db:
            trend = get_weight_trend(db, days=days)
    except Exception as exc:
        logger.exception("[lifestyle_tools] get_weight_trend failed")
        return {
            "points": [],
            "current_kg": None,
            "change_7d_kg": None,
            "change_30d_kg": None,
            "target_kg": None,
            "error": str(exc),
        }

    d = _to_dict(trend)
    # Points come through as a list of WeightEntryOut dicts already.
    return {
        "points": d.get("points", []),
        "current_kg": d.get("current_kg"),
        "change_7d_kg": d.get("change_7d_kg"),
        "change_30d_kg": d.get("change_30d_kg"),
        "target_kg": d.get("target_kg"),
    }


async def get_recent_workouts_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Return the most recent workout sessions, newest first."""
    limit = int(input_data.get("limit") or 10)
    limit = max(1, min(50, limit))

    try:
        from app.lifestyle.service import get_recent_workouts
        with _db_session() as db:
            sessions = get_recent_workouts(db, limit=limit)
    except Exception as exc:
        logger.exception("[lifestyle_tools] get_recent_workouts failed")
        return {"sessions": [], "error": str(exc)}

    return {"sessions": [_to_dict(s) for s in sessions]}


# ────────────────────────────────────────────────────────────────────────
# WRITES
# ────────────────────────────────────────────────────────────────────────

async def log_nutrition_handler(input_data: dict, context: Optional[dict]) -> dict:
    """
    Log a nutrition entry. Description is required. Macro fields are
    optional — if not provided, the entry is flagged as an estimate and
    a future estimation pass can fill them in.
    """
    description = str(input_data.get("description") or "").strip()

    # Multi-item path: log each distinct food as its OWN row so the user sees
    # every food with its own macros (PT itemisation), instead of one merged
    # blob. Validation + hard-rule lives in the service layer.
    raw_items = input_data.get("items")
    if isinstance(raw_items, list) and raw_items:
        meal_type_m = str(input_data.get("meal_type") or "meal").strip() or "meal"
        try:
            from app.lifestyle.service import log_nutrition_items_checked
            with _db_session() as db:
                r = log_nutrition_items_checked(db, raw_items, meal_type_m)
        except Exception as exc:
            logger.exception("[lifestyle_tools] log_nutrition (items) failed")
            return {"ok": False, "error": str(exc)}
        if not r.get("ok"):
            return {
                "ok": False, "logged": 0, "needs_per_item": r.get("incomplete"),
                "error": (
                    "NOT logged \u2014 every food needs its own calories, protein, "
                    "carbs and fat (plus a description). Fill the missing values "
                    "per item \u2014 ask the user or web_search to estimate \u2014 "
                    "then call log_nutrition ONCE with the full items list. Sugar "
                    "too where you can."
                ),
            }
        rows = [_to_dict(x) for x in r["rows"]]
        return {"ok": True, "items": rows, "count": len(rows)}

    if not description:
        return {
            "ok": False, "log_id": None, "description": "",
            "error": "description is required",
        }

    meal_type = str(input_data.get("meal_type") or "meal").strip() or "meal"

    def _opt_float(key):
        v = input_data.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    calories = _opt_float("calories")
    protein_g = _opt_float("protein_g")
    carbs_g = _opt_float("carbs_g")
    fat_g = _opt_float("fat_g")
    sugar_g = _opt_float("sugar_g")

    # Bug 6 itemisation gate: a single description that clearly names 2+ distinct
    # foods must be logged one row PER food via items[] (the same writer
    # prep_split uses), not bunched into a merged entry. Reject so the model
    # re-issues itemised — mirroring the macro hard-rule below.
    #
    # BUT only gate macro-LESS entries. A single description that already carries
    # complete macros (calories+protein+carbs+fat) is the model committing to ONE
    # item, so it is logged as-is: re-splitting it false-rejected real single foods
    # whose text has descriptor commas (e.g. "Pork loin steaks, oven cooked, fat
    # drained, 470 g" / "Green olives, pitted in brine, about 25") even though they
    # arrived with full macros (live 2026-06-15). The gate's job is stopping
    # macro-less merged blobs, so it stays in force whenever macros are absent.
    _macros_complete = None not in (calories, protein_g, carbs_g, fat_g)
    if not _macros_complete:
        from app.lifestyle.itemisation import looks_multi_food, multi_food_rejection
        if looks_multi_food(description):
            return multi_food_rejection(description)

    # HARD RULE: never log a partial nutrition entry. Logging description-only
    # (or calories-only) and then re-logging to refine it is what creates
    # duplicate rows. Require the full set; if anything is missing, reject with
    # an actionable message so the model gathers it (ask the user, or
    # web_search to estimate) and logs ONCE, complete.
    _required = {
        "calories": calories,
        "protein_g": protein_g,
        "carbs_g": carbs_g,
        "fat_g": fat_g,
    }
    _missing = [k for k, v in _required.items() if v is None]
    if _missing:
        return {
            "ok": False,
            "log_id": None,
            "description": description,
            "needs": _missing,
            "error": (
                "NOT logged \u2014 missing " + ", ".join(_missing) + ". Do not log a "
                "partial entry. First confirm the portion/quantity, then get "
                "calories, protein, carbs and fat \u2014 ask the user, or use "
                "web_search to estimate them as closely as possible. Then call "
                "log_nutrition ONCE with every value filled. Never log and "
                "re-log to refine; that creates duplicates."
            ),
        }

    try:
        from app.lifestyle.service import log_nutrition
        with _db_session() as db:
            entry = log_nutrition(
                db,
                description=description,
                meal_type=meal_type,
                calories=calories,
                protein_g=protein_g,
                carbs_g=carbs_g,
                fat_g=fat_g,
                sugar_g=sugar_g,
            )
    except Exception as exc:
        logger.exception("[lifestyle_tools] log_nutrition failed")
        return {
            "ok": False, "log_id": None, "description": description,
            "error": str(exc),
        }

    d = _to_dict(entry)
    return {
        "ok": True,
        "log_id": d.get("id"),
        "description": d.get("description"),
        "calories": d.get("calories"),
        "protein_g": d.get("protein_g"),
        "carbs_g": d.get("carbs_g"),
        "fat_g": d.get("fat_g"),
        "sugar_g": d.get("sugar_g"),
    }


async def delete_nutrition_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Delete a single nutrition log entry by id (remove a duplicate or wrong entry)."""
    raw = input_data.get("log_id")
    try:
        log_id = int(raw)
    except (TypeError, ValueError):
        return {"ok": False, "error": "log_id must be an integer"}

    try:
        from app.lifestyle.service import delete_nutrition
        with _db_session() as db:
            ok = delete_nutrition(db, log_id)
    except Exception as exc:
        logger.exception("[lifestyle_tools] delete_nutrition failed")
        return {"ok": False, "error": str(exc)}

    if not ok:
        return {"ok": False, "log_id": log_id, "error": f"no nutrition log with id {log_id}"}
    return {"ok": True, "log_id": log_id, "deleted": True}


# Food-preference CRUD handlers live in lifestyle_preference_tools.py (kept under
# the size ceiling); re-exported here so the Gemini executor adapters that import
# them from this module keep working. Placed AFTER _db_session/_to_dict above so
# the preference module can import them without a cycle.
from app.tools.lifestyle_preference_tools import (  # noqa: E402,F401
    add_food_preference_handler, get_food_preferences_handler,
    delete_food_preference_handler, update_food_preference_handler,
)


async def set_meal_plan_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Write or update the user's visible written plan (plain text / light markdown)."""
    body = input_data.get("body")
    if not body or not str(body).strip():
        return {"ok": False, "error": "body is required (the plan text)"}
    title = input_data.get("title")
    try:
        from app.lifestyle.service import set_meal_plan
        with _db_session() as db:
            d = _to_dict(set_meal_plan(db, str(body), title))
    except Exception as exc:
        logger.exception("[lifestyle_tools] set_meal_plan failed")
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "title": d.get("title"), "updated_at": d.get("updated_at"), "chars": len(str(body))}


async def get_meal_plan_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Read the user's current written plan."""
    try:
        from app.lifestyle.service import get_meal_plan
        with _db_session() as db:
            d = _to_dict(get_meal_plan(db))
    except Exception as exc:
        logger.exception("[lifestyle_tools] get_meal_plan failed")
        return {"ok": False, "error": str(exc)}
    return {"ok": True, **d}


async def get_today_summary_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Today's intake vs targets + current weight — 'where am I at right now'."""
    try:
        from app.lifestyle.service import get_dashboard_summary
        with _db_session() as db:
            d = _to_dict(get_dashboard_summary(db))
    except Exception as exc:
        logger.exception("[lifestyle_tools] get_today_summary failed")
        return {"ok": False, "error": str(exc)}
    cal = d.get("today_calories") or 0
    pro = d.get("today_protein_g") or 0
    sug = d.get("today_sugar_g") or 0
    cal_t = d.get("calories_target") or 0
    pro_t = d.get("protein_target") or 0
    sug_t = d.get("sugar_limit") or 0
    return {
        "ok": True,
        "calories": cal, "calories_target": cal_t,
        "calories_remaining": round(cal_t - cal) if cal_t else None,
        "protein_g": pro, "protein_target": pro_t,
        "protein_remaining": round(pro_t - pro) if pro_t else None,
        "carbs_g": d.get("today_carbs_g") or 0,
        "fat_g": d.get("today_fat_g") or 0,
        "sugar_g": sug, "sugar_limit": sug_t,
        "current_weight_kg": d.get("current_weight_kg"),
    }


# ───────────────────────────────────────────────
# STRENGTH (exercise sets) — handlers live in lifestyle_strength_tools.py to
# keep this file under the size ceiling; re-exported here so the Gemini
# executor adapters that import them from this module keep working. Placed
# AFTER _db_session/_to_dict above so the strength module can import them
# without a cycle.
# ───────────────────────────────────────────────
from app.tools.lifestyle_strength_tools import (  # noqa: E402,F401
    log_exercise_handler, delete_exercise_set_handler,
    get_exercise_history_handler, get_workout_log_handler,
    _coerce_sets,
)
from app.tools.lifestyle_nutrition_tools import (  # noqa: E402,F401
    prep_split_handler,
)
from app.tools.lifestyle_history_tools import (  # noqa: E402,F401
    get_health_history_handler,
)


async def log_workout_handler(input_data: dict, context: Optional[dict]) -> dict:
    """
    Log a workout/activity session. activity_type is required and is
    a short tag like 'gym', 'surf', 'stretch', 'walk', 'bodyboard'.
    """
    activity_type = str(input_data.get("activity_type") or "").strip()
    if not activity_type:
        return {
            "ok": False, "session_id": None, "activity_type": "",
            "error": "activity_type is required",
        }

    def _opt_int(key):
        v = input_data.get(key)
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    def _opt_float(key):
        v = input_data.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _opt_str(key):
        v = input_data.get(key)
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    duration_mins = _opt_int("duration_mins")
    title = _opt_str("title")
    notes = _opt_str("notes")
    calories_burned = _opt_float("calories_burned")
    surf_location = _opt_str("surf_location")
    surf_conditions = _opt_str("surf_conditions")

    try:
        from app.lifestyle.service import log_workout
        with _db_session() as db:
            session = log_workout(
                db,
                activity_type=activity_type,
                duration_mins=duration_mins,
                title=title,
                notes=notes,
                calories_burned=calories_burned,
                surf_location=surf_location,
                surf_conditions=surf_conditions,
            )
    except Exception as exc:
        logger.exception("[lifestyle_tools] log_workout failed")
        return {
            "ok": False, "session_id": None, "activity_type": activity_type,
            "error": str(exc),
        }

    d = _to_dict(session)
    return {
        "ok": True,
        "session_id": d.get("id"),
        "activity_type": d.get("activity_type"),
        "duration_mins": d.get("duration_mins"),
        "title": d.get("title"),
    }


async def log_weight_handler(input_data: dict, context: Optional[dict]) -> dict:
    """
    Log a body-weight measurement (kg) and report the change since the
    previous entry so the LLM can coach on progress.
    """
    raw = input_data.get("weight_kg")
    try:
        weight_kg = float(raw)
    except (TypeError, ValueError):
        return {"ok": False, "entry_id": None, "weight_kg": None,
                "error": "weight_kg must be a number"}

    notes = input_data.get("notes")
    notes = str(notes).strip() if notes else None

    try:
        from app.lifestyle.service import log_weight
        from app.lifestyle.models import WeightEntry
        with _db_session() as db:
            # Capture the previous most-recent entry BEFORE inserting, for the delta.
            prev = (
                db.query(WeightEntry)
                .order_by(WeightEntry.recorded_at.desc())
                .first()
            )
            prev_weight = prev.weight_kg if prev else None
            prev_date = prev.recorded_at.isoformat() if prev else None
            entry = log_weight(db, weight_kg=weight_kg, source="voice", notes=notes)
    except Exception as exc:
        logger.exception("[lifestyle_tools] log_weight failed")
        return {"ok": False, "entry_id": None, "weight_kg": None, "error": str(exc)}

    d = _to_dict(entry)
    change = round(weight_kg - prev_weight, 1) if prev_weight is not None else None
    return {
        "ok": True,
        "entry_id": d.get("id"),
        "weight_kg": d.get("weight_kg"),
        "change_since_last_kg": change,
        "previous_weight_kg": prev_weight,
        "previous_date": prev_date,
    }


async def set_goal_handler(input_data: dict, context: Optional[dict]) -> dict:
    """
    Create or update an active lifestyle goal (weight, calories, protein,
    sugar_limit, activity). Replaces any existing active goal of the same type.
    """
    goal_type = str(input_data.get("goal_type") or "").strip().lower()
    if not goal_type:
        return {"ok": False, "goal_id": None, "goal_type": "",
                "target_value": None, "unit": "", "error": "goal_type is required"}

    try:
        target_value = float(input_data.get("target_value"))
    except (TypeError, ValueError):
        return {"ok": False, "goal_id": None, "goal_type": goal_type,
                "target_value": None, "unit": "", "error": "target_value must be a number"}

    unit = str(input_data.get("unit") or "").strip()
    notes = input_data.get("notes")
    notes = str(notes).strip() if notes else None

    try:
        from app.lifestyle.service import set_goal
        with _db_session() as db:
            goal = set_goal(db, goal_type=goal_type, target_value=target_value,
                            unit=unit, notes=notes)
    except Exception as exc:
        logger.exception("[lifestyle_tools] set_goal failed")
        return {"ok": False, "goal_id": None, "goal_type": goal_type,
                "target_value": None, "unit": unit, "error": str(exc)}

    d = _to_dict(goal)
    return {
        "ok": True,
        "goal_id": d.get("id"),
        "goal_type": d.get("goal_type"),
        "target_value": d.get("target_value"),
        "unit": d.get("unit"),
    }


async def set_deficit_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Set (or clear) the user's daily calorie DEFICIT live. Overrides the
    gentle/moderate/aggressive pace default and flows straight into today's
    dynamic intake target -- no code edit. deficit_kcal=0/null clears it and
    reverts to the pace default. Returns today's recomputed target."""
    raw = input_data.get("deficit_kcal")
    notes = input_data.get("notes")
    notes = str(notes).strip() if notes else None
    clear = raw is None or raw == "" or (isinstance(raw, (int, float)) and float(raw) <= 0)
    try:
        from app.lifestyle.service import set_goal
        from app.lifestyle.energy_ledger import compute_today_target
        from app.lifestyle.models import LifestyleGoal
        with _db_session() as db:
            if clear:
                for g in (db.query(LifestyleGoal)
                          .filter(LifestyleGoal.goal_type == "deficit",
                                  LifestyleGoal.is_active == True).all()):
                    g.is_active = False
                db.commit()
                deficit_val, source = None, "pace_default"
            else:
                deficit_val = max(0, min(int(round(float(raw))), 2000))
                set_goal(db, goal_type="deficit", target_value=deficit_val,
                         unit="kcal", notes=notes)
                source = "custom"
            try:
                tgt = compute_today_target(db).get("target_calories")
            except Exception:
                tgt = None
    except Exception as exc:
        logger.exception("[lifestyle_tools] set_deficit failed")
        return {"ok": False, "deficit_kcal": None, "error": str(exc)}
    return {
        "ok": True,
        "deficit_kcal": deficit_val,
        "today_target_calories": int(tgt) if tgt else None,
        "source": source,
    }


async def set_profile_handler(input_data: dict, context: Optional[dict]) -> dict:
    """
    Create or update the user's body profile (height, sex, age/dob, activity
    level, loss pace) used for BMR/TDEE. Only the fields provided are changed.
    """
    def _opt_str(key):
        v = input_data.get(key)
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    def _opt_float(key):
        v = input_data.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _opt_int(key):
        v = input_data.get(key)
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    fields = {
        "sex": _opt_str("sex"),
        "height_cm": _opt_float("height_cm"),
        "dob": _opt_str("dob"),
        "age_years": _opt_int("age_years"),
        "activity_level": _opt_str("activity_level"),
        "goal_pace": _opt_str("goal_pace"),
    }
    if all(v is None for v in fields.values()):
        return {"ok": False, "error": "no profile fields provided"}

    try:
        from app.lifestyle.service import upsert_profile
        with _db_session() as db:
            profile = upsert_profile(db, **fields)
    except Exception as exc:
        logger.exception("[lifestyle_tools] set_profile failed")
        return {"ok": False, "error": str(exc)}

    d = _to_dict(profile)
    d["ok"] = True
    return d


async def get_metabolic_summary_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Return BMR, daily burn (TDEE), and recommended calorie + protein targets."""
    try:
        from app.lifestyle.service import get_metabolic_summary
        with _db_session() as db:
            summary = get_metabolic_summary(db)
    except Exception as exc:
        logger.exception("[lifestyle_tools] get_metabolic_summary failed")
        return {"ok": False, "error": str(exc)}

    d = _to_dict(summary)
    d["ok"] = True
    return d


# ────────────────────────────────────────────────────────────────────────
# Registration
# ────────────────────────────────────────────────────────────────────────

# register_lifestyle_tools (the ToolDefinition registration block) lives in
# lifestyle_tools_registration.py; re-exported here so registry.py keeps importing
# it from this module. MUST stay the LAST statement of this module: the
# registration module imports the 8 handlers it registers from here, so each must
# already be defined above this line.
from app.tools.lifestyle_tools_registration import register_lifestyle_tools  # noqa: E402,F401
