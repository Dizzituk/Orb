# FILE: app/tools/strength_tools.py
"""
Strength tool handlers — per-exercise set logging + progressive-overload recall.

Two tools that let any ASTRA chat:
- log_exercise_set: record a single set (exercise, weight, reps) by voice at the gym
- get_exercise_history: pull recent sessions + last-time and best figures so the
  user can see what to beat before they load the bar.

Handlers wrap app.lifestyle.strength. Kept in their own module so
lifestyle_tools.py stays within the file-size budget.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
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


def _to_dict(obj) -> dict:
    """Convert a Pydantic v1/v2 model to a plain dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return dict(obj)


async def log_exercise_set_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Log a single set of a strength exercise. exercise_name is required;
    weight_kg, reps, set_index, unit (kg/lb) and notes are optional."""
    exercise_name = str(input_data.get("exercise_name") or "").strip()
    if not exercise_name:
        return {"ok": False, "set_id": None, "exercise_name": "",
                "error": "exercise_name is required"}

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

    weight_kg = _opt_float("weight_kg")
    reps = _opt_int("reps")
    set_index = _opt_int("set_index")
    unit = str(input_data.get("unit") or "kg").strip().lower() or "kg"
    notes = input_data.get("notes")
    notes = str(notes).strip() if notes else None

    try:
        from app.lifestyle.strength import log_exercise_set
        with _db_session() as db:
            entry = log_exercise_set(
                db, exercise_name=exercise_name, weight_kg=weight_kg,
                reps=reps, set_index=set_index, unit=unit, notes=notes,
            )
    except Exception as exc:
        logger.exception("[strength_tools] log_exercise_set failed")
        return {"ok": False, "set_id": None, "exercise_name": exercise_name,
                "error": str(exc)}

    d = _to_dict(entry)
    return {
        "ok": True,
        "set_id": d.get("id"),
        "exercise_name": d.get("exercise_name"),
        "weight_kg": d.get("weight_kg"),
        "reps": d.get("reps"),
    }


async def get_exercise_history_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Return recent sessions for one exercise plus last-time and best figures."""
    exercise_name = str(input_data.get("exercise_name") or "").strip()
    if not exercise_name:
        return {"exercise_name": "", "sessions": [],
                "error": "exercise_name is required"}

    limit_sessions = int(input_data.get("limit_sessions") or 8)
    limit_sessions = max(1, min(30, limit_sessions))

    try:
        from app.lifestyle.strength import get_exercise_history
        with _db_session() as db:
            hist = get_exercise_history(
                db, exercise_name=exercise_name, limit_sessions=limit_sessions,
            )
    except Exception as exc:
        logger.exception("[strength_tools] get_exercise_history failed")
        return {"exercise_name": exercise_name, "sessions": [], "error": str(exc)}

    d = _to_dict(hist)
    return {
        "exercise_name": d.get("exercise_name"),
        "sessions": d.get("sessions", []),
        "last_performed": d.get("last_performed"),
        "last_top_weight_kg": d.get("last_top_weight_kg"),
        "best_weight_kg": d.get("best_weight_kg"),
        "best_est_1rm_kg": d.get("best_est_1rm_kg"),
    }


def register_strength_tools() -> None:
    """Register the strength tools with the global tool registry."""
    from app.tools.registry import ToolDefinition, register_tool
    from app.tools.schemas import TOOL_SCHEMAS

    register_tool(ToolDefinition(
        name="log_exercise_set",
        version="v1",
        description=(
            "Log a single set of a strength exercise — use at the gym when the "
            "user says what they lifted, e.g. 'bench press 80 kilos for 8'. "
            "exercise_name is required; weight_kg, reps, set_index, unit "
            "(kg/lb, default kg) and notes are optional. Log each set as a "
            "separate call. Weight is stored in kg."
        ),
        input_schema=TOOL_SCHEMAS["log_exercise_set"]["input"],
        output_schema=TOOL_SCHEMAS["log_exercise_set"]["output"],
        handler=log_exercise_set_handler,
    ))

    register_tool(ToolDefinition(
        name="get_exercise_history",
        version="v1",
        description=(
            "Return the user's recent sessions for one exercise plus last-time "
            "and all-time-best figures (top weight, estimated 1RM). Use this "
            "before the user trains so you can tell them what they did last "
            "time and what to beat. exercise_name is required."
        ),
        input_schema=TOOL_SCHEMAS["get_exercise_history"]["input"],
        output_schema=TOOL_SCHEMAS["get_exercise_history"]["output"],
        handler=get_exercise_history_handler,
    ))

    logger.info("[strength_tools] registered 2 strength tools")
