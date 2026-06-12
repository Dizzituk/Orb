# FILE: app/tools/lifestyle_strength_tools.py
# Purpose: Strength tool handlers — the LLM-facing layer for exercise-set logging
# Called-by: app.tools.lifestyle_tools
# Depends-on: app.lifestyle.service, app.tools.lifestyle_tools
# Last-renovated: 2026-06-11
"""
Strength tool handlers — the LLM-facing layer for exercise-set logging
and progressive-overload history.

Split out of app/tools/lifestyle_tools.py to keep that file under the
size ceiling. lifestyle_tools.py re-exports these names, so the Gemini
executor adapters (which import `log_exercise_handler` etc. from
lifestyle_tools) keep working unchanged.

Same pattern as the rest of the lifestyle handlers: open a short-lived
DB session, call the service layer, return a plain dict in
{ok: bool, ...} shape. The session helper and dict coercer are imported
from lifestyle_tools (defined above its re-export line, so no cycle).
"""
from __future__ import annotations

import logging
from typing import Optional

from app.tools.lifestyle_tools import _db_session, _to_dict

logger = logging.getLogger(__name__)


def _coerce_sets(input_data: dict) -> list:
    """Build a [{weight_kg, reps}] list from flexible tool args.

    Accepts either an explicit `sets` array, or simple
    weight_kg/reps(+optional `sets_count`) to repeat one set N times.
    """
    raw = input_data.get("sets")
    out = []
    if isinstance(raw, list) and raw:
        for item in raw:
            if not isinstance(item, dict):
                continue
            w = item.get("weight_kg")
            r = item.get("reps")
            try:
                w = float(w) if w is not None else None
            except (TypeError, ValueError):
                w = None
            try:
                r = int(r) if r is not None else None
            except (TypeError, ValueError):
                r = None
            out.append({"weight_kg": w, "reps": r})
        return out

    # Simple form: one weight/reps, optionally repeated
    w = input_data.get("weight_kg")
    r = input_data.get("reps")
    try:
        w = float(w) if w is not None else None
    except (TypeError, ValueError):
        w = None
    try:
        r = int(r) if r is not None else None
    except (TypeError, ValueError):
        r = None
    try:
        n = int(input_data.get("sets_count") or 1)
    except (TypeError, ValueError):
        n = 1
    n = max(1, min(20, n))
    if w is not None or r is not None:
        out = [{"weight_kg": w, "reps": r} for _ in range(n)]
    return out


async def log_exercise_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Log one or more sets of a strength exercise (squats, bench press, etc)."""
    name = str(input_data.get("exercise_name") or input_data.get("exercise") or "").strip()
    if not name:
        return {"ok": False, "error": "exercise_name is required"}
    sets = _coerce_sets(input_data)
    if not sets:
        return {"ok": False, "error": "no sets provided — give weight_kg and reps, or a sets list"}
    unit = str(input_data.get("unit") or "kg").strip() or "kg"
    notes = input_data.get("notes")
    notes = str(notes).strip() if notes else None
    try:
        from app.lifestyle.service import log_exercise_sets
        with _db_session() as db:
            rows = [_to_dict(r) for r in log_exercise_sets(
                db, name, sets, unit=unit, notes=notes, display=name, source="voice",
            )]
    except Exception as exc:
        logger.exception("[lifestyle_tools] log_exercise failed")
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "exercise_name": rows[0].get("exercise_name") if rows else name,
        "sets_logged": len(rows),
        "set_ids": [r.get("id") for r in rows],
        "sets": [{"weight_kg": r.get("weight_kg"), "reps": r.get("reps")} for r in rows],
    }


async def delete_exercise_set_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Delete a single logged set by id."""
    raw = input_data.get("set_id")
    try:
        set_id = int(raw)
    except (TypeError, ValueError):
        return {"ok": False, "error": "set_id must be an integer"}
    try:
        from app.lifestyle.service import delete_exercise_set
        with _db_session() as db:
            ok = delete_exercise_set(db, set_id)
    except Exception as exc:
        logger.exception("[lifestyle_tools] delete_exercise_set failed")
        return {"ok": False, "error": str(exc)}
    if not ok:
        return {"ok": False, "set_id": set_id, "error": f"no set with id {set_id}"}
    return {"ok": True, "set_id": set_id, "deleted": True}


async def get_exercise_history_handler(input_data: dict, context: Optional[dict]) -> dict:
    """History of one exercise for progressive-overload coaching."""
    name = str(input_data.get("exercise_name") or input_data.get("exercise") or "").strip()
    if not name:
        return {"ok": False, "error": "exercise_name is required"}
    try:
        limit = int(input_data.get("limit_sessions") or 8)
    except (TypeError, ValueError):
        limit = 8
    limit = max(1, min(20, limit))
    try:
        from app.lifestyle.service import get_exercise_history
        with _db_session() as db:
            d = _to_dict(get_exercise_history(db, name, limit_sessions=limit))
    except Exception as exc:
        logger.exception("[lifestyle_tools] get_exercise_history failed")
        return {"ok": False, "error": str(exc)}
    return {"ok": True, **d}


async def get_workout_log_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Day-by-day strength log across all exercises (what they've been doing)."""
    try:
        days = int(input_data.get("days") or 21)
    except (TypeError, ValueError):
        days = 21
    days = max(1, min(120, days))
    try:
        from app.lifestyle.service import get_strength_log, list_recent_exercises
        with _db_session() as db:
            log = _to_dict(get_strength_log(db, days=days))
            recent = list_recent_exercises(db, limit=12)
    except Exception as exc:
        logger.exception("[lifestyle_tools] get_workout_log failed")
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "days": log.get("days", []), "recent_exercises": recent}
