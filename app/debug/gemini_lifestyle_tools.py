# FILE: app/debug/gemini_lifestyle_tools.py
"""
Lifestyle tool declarations and executors for the Gemini multimodal
tool loop (app.debug.gemini_tool_loop).

Why this lives here, separate from app/tools/lifestyle_tools.py:
    The Gemini multimodal path (used for image-bearing turns) has its
    own native function-calling format and its own executor map. It
    cannot call into the generic app.tools registry directly because
    that registry uses JSON-Schema input/output validation and an async
    executor wrapper this code path doesn't go through.

    Rather than duplicate the BUSINESS logic, this module's executors
    delegate to the already-tested handlers in app.tools.lifestyle_tools,
    which call the lifestyle service layer. So the path is:

        Gemini -> _lifestyle_*_executor (here)
              -> handler (app.tools.lifestyle_tools)
              -> service (app.lifestyle.service)
              -> DB

    No duplicated business logic. This module is purely the adapter
    between Gemini's function-calling shape and the existing handlers.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Function declarations for Gemini
# ---------------------------------------------------------------------------
# These follow the same dict shape as the file-tool declarations in
# gemini_tool_loop.py — Gemini converts both via _convert_schema().


from app.debug.gemini_lifestyle_decls import LIFESTYLE_TOOL_DECLARATIONS


# ---------------------------------------------------------------------------
# Executors — thin adapters that delegate to app.tools.lifestyle_tools
# ---------------------------------------------------------------------------
# Each executor takes Gemini's args dict, calls the matching handler from
# the main tools layer (which opens its own DB session), and returns a
# plain text result for Gemini. Errors come back as strings prefixed with
# "ERROR:" so they format consistently with the file-tool executors.


def _result_to_text(payload: dict, ok_summary_fn=None) -> str:
    """
    Convert a handler's dict result to a short text string Gemini can
    reason about. On success, build a one-line summary; on failure,
    surface the error message.
    """
    if payload.get("ok") is False or payload.get("error"):
        err = payload.get("error", "unknown error")
        return f"ERROR: {err}"
    if ok_summary_fn:
        try:
            return ok_summary_fn(payload)
        except Exception:
            pass
    # Fall through to compact JSON so Gemini can still parse it.
    import json
    try:
        return json.dumps(payload, default=str)[:2000]
    except Exception:
        return str(payload)[:2000]


async def _exec_log_nutrition(args: dict) -> str:
    from app.tools.lifestyle_tools import log_nutrition_handler
    result = await log_nutrition_handler(args, context=None)

    # Multi-item result: one line per logged food with its macros.
    if result.get("ok") and isinstance(result.get("items"), list):
        rows = result["items"]
        lines = [f"Logged {len(rows)} item(s):"]
        for r in rows:
            macros = []
            for label, key in (("P", "protein_g"), ("C", "carbs_g"),
                               ("F", "fat_g"), ("S", "sugar_g")):
                v = r.get(key)
                if v is not None:
                    macros.append(f"{label}={v:.0f}g")
            cal = r.get("calories")
            cal_s = f"{cal:.0f} kcal" if cal is not None else ""
            mac_s = (" (" + ", ".join(macros) + ")") if macros else ""
            lines.append(f"  • {r.get('description', '')}: {cal_s}{mac_s} [id {r.get('id')}]")
        return "\n".join(lines)[:4000]

    def _summary(p: dict) -> str:
        bits = [f"Logged: {p.get('description', '')}"]
        if p.get("calories") is not None:
            bits.append(f"{p['calories']:.0f} kcal")
        macros = []
        for label, key in (("P", "protein_g"), ("C", "carbs_g"),
                           ("F", "fat_g"), ("S", "sugar_g")):
            v = p.get(key)
            if v is not None:
                macros.append(f"{label}={v:.0f}g")
        if macros:
            bits.append("(" + ", ".join(macros) + ")")
        bits.append(f"[log_id={p.get('log_id')}]")
        return " ".join(bits)

    return _result_to_text(result, _summary)


async def _exec_log_workout(args: dict) -> str:
    from app.tools.lifestyle_tools import log_workout_handler
    result = await log_workout_handler(args, context=None)

    def _summary(p: dict) -> str:
        bits = [f"Logged workout: {p.get('activity_type', '')}"]
        if p.get("duration_mins"):
            bits.append(f"{p['duration_mins']} min")
        if p.get("title"):
            bits.append(f"— {p['title']}")
        bits.append(f"[session_id={p.get('session_id')}]")
        return " ".join(bits)

    return _result_to_text(result, _summary)


async def _exec_get_recent_nutrition(args: dict) -> str:
    from app.tools.lifestyle_tools import get_recent_nutrition_handler
    result = await get_recent_nutrition_handler(args, context=None)
    if result.get("error"):
        return f"ERROR: {result['error']}"

    days = result.get("days") or []
    if not days:
        return "No nutrition logged yet."
    lines = []
    for day in days:
        t = day.get("totals", {})
        lines.append(
            f"{day.get('date')}: "
            f"{t.get('calories', 0):.0f} kcal, "
            f"P {t.get('protein_g', 0):.0f}g, "
            f"C {t.get('carbs_g', 0):.0f}g, "
            f"F {t.get('fat_g', 0):.0f}g, "
            f"S {t.get('sugar_g', 0):.0f}g — "
            f"{day.get('meal_count', 0)} entries"
        )
        for log in day.get("logs", [])[:8]:
            desc = log.get("description", "")
            cal = log.get("calories")
            lid = log.get("id")
            line = f"   • [id {lid}] {desc}" if lid is not None else f"   • {desc}"
            if cal is not None:
                line += f" ({cal:.0f} kcal)"
            lines.append(line)
    return "\n".join(lines)[:4000]


async def _exec_get_weight_trend(args: dict) -> str:
    from app.tools.lifestyle_tools import get_weight_trend_handler
    result = await get_weight_trend_handler(args, context=None)
    if result.get("error"):
        return f"ERROR: {result['error']}"

    current = result.get("current_kg")
    if current is None:
        return "No weight entries logged yet."
    bits = [f"Current: {current:.1f} kg"]
    c7 = result.get("change_7d_kg")
    c30 = result.get("change_30d_kg")
    if c7 is not None:
        bits.append(f"7d Δ {c7:+.1f} kg")
    if c30 is not None:
        bits.append(f"30d Δ {c30:+.1f} kg")
    target = result.get("target_kg")
    if target is not None:
        bits.append(f"target {target:.1f} kg")
    points = result.get("points") or []
    bits.append(f"({len(points)} points)")
    return " | ".join(bits)


async def _exec_get_recent_workouts(args: dict) -> str:
    from app.tools.lifestyle_tools import get_recent_workouts_handler
    result = await get_recent_workouts_handler(args, context=None)
    if result.get("error"):
        return f"ERROR: {result['error']}"

    sessions = result.get("sessions") or []
    if not sessions:
        return "No workouts logged yet."
    lines = []
    for s in sessions[:20]:
        started = s.get("started_at", "")[:10]
        bits = [started, s.get("activity_type", "")]
        if s.get("duration_mins"):
            bits.append(f"{s['duration_mins']}min")
        if s.get("title"):
            bits.append(f"— {s['title']}")
        lines.append("  ".join(bits))
    return "\n".join(lines)[:4000]


async def _exec_log_weight(args: dict) -> str:
    from app.tools.lifestyle_tools import log_weight_handler
    result = await log_weight_handler(args, context=None)

    def _summary(p: dict) -> str:
        w = p.get("weight_kg")
        head = f"Logged weight: {w:.1f} kg" if isinstance(w, (int, float)) else "Logged weight"
        change = p.get("change_since_last_kg")
        if change is None:
            tail = ""
        elif change == 0:
            tail = " (no change since last)"
        else:
            direction = "down" if change < 0 else "up"
            tail = f" ({direction} {abs(change):.1f} kg since last)"
        return f"{head}{tail} [entry_id={p.get('entry_id')}]"

    return _result_to_text(result, _summary)


async def _exec_set_goal(args: dict) -> str:
    from app.tools.lifestyle_tools import set_goal_handler
    result = await set_goal_handler(args, context=None)

    def _summary(p: dict) -> str:
        gt = p.get("goal_type", "")
        tv = p.get("target_value")
        unit = p.get("unit", "") or ""
        target = (f"{tv:g} {unit}".strip()) if isinstance(tv, (int, float)) else str(tv)
        return f"Goal set: {gt} target {target} [goal_id={p.get('goal_id')}]"

    return _result_to_text(result, _summary)


async def _exec_set_profile(args: dict) -> str:
    from app.tools.lifestyle_tools import set_profile_handler
    result = await set_profile_handler(args, context=None)
    if result.get("ok") is False or result.get("error"):
        return f"ERROR: {result.get('error', 'unknown error')}"
    bits = []
    if result.get("height_cm") is not None:
        bits.append(f"height {result['height_cm']:.0f} cm")
    if result.get("sex"):
        bits.append(str(result["sex"]))
    if result.get("age_years") is not None:
        bits.append(f"age {result['age_years']}")
    elif result.get("dob"):
        bits.append(f"dob {result['dob']}")
    if result.get("activity_level"):
        bits.append(f"activity {result['activity_level']}")
    if result.get("goal_pace"):
        bits.append(f"pace {result['goal_pace']}")
    return "Profile updated: " + (", ".join(bits) if bits else "saved")


async def _exec_get_metabolic_summary(args: dict) -> str:
    from app.tools.lifestyle_tools import get_metabolic_summary_handler
    result = await get_metabolic_summary_handler(args, context=None)
    if result.get("ok") is False or result.get("error"):
        return f"ERROR: {result.get('error', 'unknown error')}"
    if not result.get("complete"):
        missing = ", ".join(result.get("missing") or []) or "some inputs"
        return f"Profile incomplete - need: {missing}. Use set_profile to add them."
    bits = [
        f"BMR {result.get('bmr'):.0f} kcal",
        f"daily burn {result.get('tdee'):.0f} kcal ({result.get('tdee_source')})",
        f"target {result.get('target_calories'):.0f} kcal/day",
        f"protein {result.get('protein_g'):.0f} g",
    ]
    if result.get("deficit_kcal"):
        bits.append(f"deficit {result['deficit_kcal']} kcal")
    if result.get("floored_at_bmr"):
        bits.append("(floored at BMR - ease pace or add movement)")
    return " | ".join(bits)


async def _exec_delete_nutrition(args: dict) -> str:
    from app.tools.lifestyle_tools import delete_nutrition_handler
    result = await delete_nutrition_handler(args, context=None)
    if result.get("ok") is False or result.get("error"):
        return f"ERROR: {result.get('error', 'unknown error')}"
    return f"Deleted nutrition log id={result.get('log_id')}."


async def _exec_add_food_preference(args: dict) -> str:
    from app.tools.lifestyle_tools import add_food_preference_handler
    result = await add_food_preference_handler(args, context=None)
    if result.get("ok") is False or result.get("error"):
        return f"ERROR: {result.get('error', 'unknown error')}"
    return f"Noted [{result.get('category')}]: {result.get('content')} [pref_id={result.get('pref_id')}]"


async def _exec_get_food_preferences(args: dict) -> str:
    from app.tools.lifestyle_tools import get_food_preferences_handler
    result = await get_food_preferences_handler(args, context=None)
    if result.get("ok") is False or result.get("error"):
        return f"ERROR: {result.get('error', 'unknown error')}"
    prefs = result.get("preferences") or []
    if not prefs:
        return "Nothing recorded yet about how they eat."
    lines = []
    for p in prefs:
        stab = p.get("stability") or "evolving"
        stale = " (stale)" if p.get("stale") else ""
        lines.append(f"   • [id {p.get('id')}] ({p.get('category')}, {stab}{stale}) {p.get('content')}")
    return "What I know about their eating:\n" + "\n".join(lines)[:4000]


async def _exec_delete_food_preference(args: dict) -> str:
    from app.tools.lifestyle_tools import delete_food_preference_handler
    result = await delete_food_preference_handler(args, context=None)
    if result.get("ok") is False or result.get("error"):
        return f"ERROR: {result.get('error', 'unknown error')}"
    return f"Removed preference id={result.get('pref_id')}."


async def _exec_update_food_preference(args: dict) -> str:
    from app.tools.lifestyle_tools import update_food_preference_handler
    result = await update_food_preference_handler(args, context=None)
    if result.get("ok") is False or result.get("error"):
        return f"ERROR: {result.get('error', 'unknown error')}"
    return f"Updated preference id={result.get('pref_id')} → ({result.get('stability')}) {result.get('content')}"


async def _exec_set_meal_plan(args: dict) -> str:
    from app.tools.lifestyle_tools import set_meal_plan_handler
    result = await set_meal_plan_handler(args, context=None)
    if result.get("ok") is False or result.get("error"):
        return f"ERROR: {result.get('error', 'unknown error')}"
    return f"Plan saved: \"{result.get('title')}\" ({result.get('chars')} chars). It's now in the Plan tab."


async def _exec_get_meal_plan(args: dict) -> str:
    from app.tools.lifestyle_tools import get_meal_plan_handler
    result = await get_meal_plan_handler(args, context=None)
    if result.get("ok") is False or result.get("error"):
        return f"ERROR: {result.get('error', 'unknown error')}"
    if not result.get("exists"):
        return "No written plan yet."
    return f"Current plan \"{result.get('title') or 'My Plan'}\":\n{result.get('body') or ''}"[:4000]


async def _exec_get_today_summary(args: dict) -> str:
    from app.tools.lifestyle_tools import get_today_summary_handler
    r = await get_today_summary_handler(args, context=None)
    if r.get("ok") is False or r.get("error"):
        return f"ERROR: {r.get('error', 'unknown error')}"

    def fld(label, val, tgt, rem, unit):
        s = f"{label} {round(val or 0)}{unit}"
        if tgt:
            s += f"/{round(tgt)}{unit}"
            if rem is not None:
                s += f" ({round(rem)} left)"
        return s

    parts = [
        fld("calories", r.get("calories"), r.get("calories_target"), r.get("calories_remaining"), ""),
        fld("protein", r.get("protein_g"), r.get("protein_target"), r.get("protein_remaining"), "g"),
        f"carbs {round(r.get('carbs_g') or 0)}g",
        f"fat {round(r.get('fat_g') or 0)}g",
        fld("sugar", r.get("sugar_g"), r.get("sugar_limit"), None, "g"),
    ]
    out = "Today so far: " + ", ".join(parts) + "."
    wt = r.get("current_weight_kg")
    if wt:
        out += f" Current weight {round(wt)} kg."
    return out


def _fmt_session(sess: dict) -> str:
    """One day of one exercise: '3x5 @ 100kg' style, with set ids."""
    sets = sess.get("sets") or []
    bits = []
    for s in sets:
        w = s.get("weight_kg")
        reps = s.get("reps")
        if w is not None and reps is not None:
            bits.append(f"{reps}×{w:g}kg [id {s.get('id')}]")
        elif reps is not None:
            bits.append(f"{reps} reps [id {s.get('id')}]")
        elif w is not None:
            bits.append(f"{w:g}kg [id {s.get('id')}]")
    return ", ".join(bits)


async def _exec_log_exercise(args: dict) -> str:
    from app.tools.lifestyle_tools import log_exercise_handler
    result = await log_exercise_handler(args, context=None)
    if result.get("ok") is False or result.get("error"):
        return f"ERROR: {result.get('error', 'unknown error')}"
    sets = result.get("sets") or []
    desc = ", ".join(
        f"{s.get('reps')}×{s.get('weight_kg'):g}kg" if s.get("weight_kg") is not None and s.get("reps") is not None
        else (f"{s.get('reps')} reps" if s.get("reps") is not None else "set")
        for s in sets
    )
    return f"Logged {result.get('exercise_name')}: {desc} ({result.get('sets_logged')} set(s))"


async def _exec_delete_exercise_set(args: dict) -> str:
    from app.tools.lifestyle_tools import delete_exercise_set_handler
    result = await delete_exercise_set_handler(args, context=None)
    if result.get("ok") is False or result.get("error"):
        return f"ERROR: {result.get('error', 'unknown error')}"
    return f"Deleted set id={result.get('set_id')}."


async def _exec_get_exercise_history(args: dict) -> str:
    from app.tools.lifestyle_tools import get_exercise_history_handler
    r = await get_exercise_history_handler(args, context=None)
    if r.get("ok") is False or r.get("error"):
        return f"ERROR: {r.get('error', 'unknown error')}"
    name = r.get("exercise_name")
    sessions = r.get("sessions") or []
    if not sessions:
        return f"No history yet for {name}."
    head_bits = [f"{name} history"]
    if r.get("last_top_weight_kg") is not None:
        head_bits.append(f"last top {r['last_top_weight_kg']:g}kg")
    if r.get("best_weight_kg") is not None:
        head_bits.append(f"best {r['best_weight_kg']:g}kg")
    if r.get("best_est_1rm_kg") is not None:
        head_bits.append(f"est 1RM {r['best_est_1rm_kg']:g}kg")
    lines = [" | ".join(head_bits)]
    for sess in sessions:
        lines.append(f"  {sess.get('date')}: {_fmt_session(sess)}")
    return "\n".join(lines)[:4000]


async def _exec_get_workout_log(args: dict) -> str:
    from app.tools.lifestyle_tools import get_workout_log_handler
    r = await get_workout_log_handler(args, context=None)
    if r.get("ok") is False or r.get("error"):
        return f"ERROR: {r.get('error', 'unknown error')}"
    days = r.get("days") or []
    if not days:
        recent = r.get("recent_exercises") or []
        if recent:
            return "No sets in that window. Recently trained: " + ", ".join(recent)
        return "No strength training logged yet."
    lines = []
    for day in days:
        lines.append(day.get("date", ""))
        for sess in day.get("exercises", []):
            lines.append(f"  {sess.get('exercise_name') or _exercise_of(sess)}: {_fmt_session(sess)}")
    return "\n".join(lines)[:4000]


def _exercise_of(sess: dict) -> str:
    """Exercise name from a session's first set (sessions group by exercise)."""
    sets = sess.get("sets") or []
    if sets:
        return sets[0].get("exercise_name") or sets[0].get("exercise_display") or "exercise"
    return "exercise"


async def _exec_prep_split(args: dict) -> str:
    from app.tools.lifestyle_tools import prep_split_handler
    r = await prep_split_handler(args, context=None)
    if r.get("ok") is False or r.get("error"):
        return f"ERROR: {r.get('error', 'unknown error')}"
    pd = r.get("per_day") or {}
    macros = []
    for label, key in (("P", "protein_g"), ("C", "carbs_g"), ("F", "fat_g"), ("S", "sugar_g")):
        v = pd.get(key)
        if v is not None:
            macros.append(f"{label}={v:.0f}g")
    cal = pd.get("calories")
    cal_s = f"{cal:.0f} kcal" if cal is not None else ""
    mac_s = (" (" + ", ".join(macros) + ")") if macros else ""
    dates = r.get("dates") or []
    span = f"{dates[0]} to {dates[-1]}" if dates else ""
    return (f"Split over {r.get('days_written')} days ({span}). "
            f"Each day: {cal_s}{mac_s}.")


async def _exec_get_health_history(args: dict) -> str:
    from app.tools.lifestyle_tools import get_health_history_handler
    r = await get_health_history_handler(args, context=None)
    if r.get("ok") is False or r.get("error"):
        return f"ERROR: {r.get('error', 'unknown error')}"
    s = r.get("summary") or {}
    days = r.get("days") or []
    head = [f"Health history {r.get('start')} \u2192 {r.get('end')} ({s.get('days_with_data', 0)} days with data):"]
    bits = []
    if s.get("weight_start_kg") is not None and s.get("weight_end_kg") is not None:
        d = s.get("weight_delta_kg")
        ds = f" ({d:+.1f})" if d is not None else ""
        bits.append(f"weight {s['weight_start_kg']:g}\u2192{s['weight_end_kg']:g}kg{ds}")
    if s.get("avg_resting_hr") is not None:
        bits.append(f"avg resting HR {s['avg_resting_hr']:g}")
    if s.get("avg_daily_steps") is not None:
        bits.append(f"avg steps {s['avg_daily_steps']:.0f}/day")
    if s.get("total_floors") is not None:
        bits.append(f"floors {s['total_floors']:g} total")
    if s.get("avg_active_calories") is not None:
        bits.append(f"avg burn {s['avg_active_calories']:.0f} kcal")
    if s.get("avg_sleep_minutes") is not None:
        hrs = s["avg_sleep_minutes"] / 60.0
        bits.append(f"avg sleep {hrs:.1f}h")
    if s.get("avg_daily_calories") is not None:
        bits.append(f"avg intake {s['avg_daily_calories']:.0f} kcal")
    if s.get("avg_daily_protein_g") is not None:
        bits.append(f"avg protein {s['avg_daily_protein_g']:.0f}g")
    bits.append(f"{s.get('training_days', 0)} training days")
    if bits:
        head.append("  " + " | ".join(bits))
    # Daily lines, most-recent first, capped to keep context tidy.
    shown = list(reversed(days))[:60]
    for d in shown:
        parts = []
        if d.get("weight_kg") is not None:
            parts.append(f"{d['weight_kg']:g}kg")
        if d.get("steps") is not None:
            parts.append(f"{d['steps']} steps")
        if d.get("floors") is not None:
            parts.append(f"{d['floors']} floors")
        if d.get("active_calories") is not None:
            parts.append(f"{d['active_calories']:g}kcal burn")
        if d.get("resting_hr") is not None:
            parts.append(f"RHR {d['resting_hr']}")
        if d.get("sleep_minutes") is not None:
            parts.append(f"sleep {d['sleep_minutes'] / 60.0:.1f}h")
        if d.get("calories") is not None:
            parts.append(f"{d['calories']:g}kcal in")
        if d.get("activity_mins"):
            parts.append(f"{d['activity_mins']}min act")
        head.append(f"  {d.get('date')}: " + (", ".join(parts) if parts else "no data"))
    if len(days) > 60:
        head.append(f"  \u2026 {len(days) - 60} earlier days omitted (summary covers the full range)")
    return "\n".join(head)[:5000]


# ---------------------------------------------------------------------------
# Public maps for gemini_tool_loop to merge with its existing tool set
# ---------------------------------------------------------------------------

async def _exec_copy_nutrition_day(args: dict) -> str:
    from app.tools.lifestyle_nutrition_tools import copy_nutrition_day_handler
    r = await copy_nutrition_day_handler(args, context=None)
    if not r.get("ok"):
        return f"ERROR: {r.get('error', 'unknown error')}"
    totals = r.get("totals") or {}
    bits = []
    if totals.get("calories") is not None:
        bits.append(f"{totals['calories']:.0f} kcal")
    for label, key in (("P", "protein_g"), ("C", "carbs_g"), ("F", "fat_g"), ("S", "sugar_g")):
        v = totals.get(key)
        if v:
            bits.append(f"{label}={v:.0f}g")
    skipped = r.get("skipped_duplicates") or 0
    skip_s = f", {skipped} already there" if skipped else ""
    return (
        f"Copied {r.get('copied')} item(s) from {r.get('source_date')} to "
        f"{r.get('target_date')} ({', '.join(bits)}){skip_s}. Target day's totals refreshed."
    )


LIFESTYLE_TOOL_EXECUTORS = {
    "log_nutrition": _exec_log_nutrition,
    "delete_nutrition": _exec_delete_nutrition,
    "add_food_preference": _exec_add_food_preference,
    "get_food_preferences": _exec_get_food_preferences,
    "delete_food_preference": _exec_delete_food_preference,
    "update_food_preference": _exec_update_food_preference,
    "set_meal_plan": _exec_set_meal_plan,
    "get_meal_plan": _exec_get_meal_plan,
    "get_today_summary": _exec_get_today_summary,
    "log_exercise": _exec_log_exercise,
    "delete_exercise_set": _exec_delete_exercise_set,
    "get_exercise_history": _exec_get_exercise_history,
    "get_workout_log": _exec_get_workout_log,
    "prep_split": _exec_prep_split,
    "copy_nutrition_day": _exec_copy_nutrition_day,
    "get_health_history": _exec_get_health_history,
    "log_workout": _exec_log_workout,
    "log_weight": _exec_log_weight,
    "set_goal": _exec_set_goal,
    "set_profile": _exec_set_profile,
    "get_metabolic_summary": _exec_get_metabolic_summary,
    "get_recent_nutrition": _exec_get_recent_nutrition,
    "get_weight_trend": _exec_get_weight_trend,
    "get_recent_workouts": _exec_get_recent_workouts,
}


def summarise_lifestyle_call(name: str, args: dict) -> str:
    """Human-readable one-liner for the chat UI's 'tool_start' event."""
    if name == "log_nutrition":
        return f"Logging nutrition: {args.get('description', '')[:60]}"
    if name == "delete_nutrition":
        return f"Deleting nutrition log {args.get('log_id', '')}"
    if name == "add_food_preference":
        return f"Noting preference: {args.get('content', '')[:50]}"
    if name == "get_food_preferences":
        return "Reading food preferences"
    if name == "delete_food_preference":
        return f"Removing preference {args.get('pref_id', '')}"
    if name == "update_food_preference":
        return f"Updating preference {args.get('pref_id', '')}"
    if name == "set_meal_plan":
        return "Writing the plan"
    if name == "get_meal_plan":
        return "Reading the plan"
    if name == "get_today_summary":
        return "Checking today's intake"
    if name == "log_exercise":
        return f"Logging {args.get('exercise_name', 'exercise')}"
    if name == "delete_exercise_set":
        return f"Deleting set {args.get('set_id', '')}"
    if name == "get_exercise_history":
        return f"Checking {args.get('exercise_name', 'exercise')} history"
    if name == "get_workout_log":
        return "Reading the workout log"
    if name == "prep_split":
        return f"Splitting {args.get('description', 'batch')[:40]} over {args.get('days', '')} days"
    if name == "copy_nutrition_day":
        return (
            f"Copying {args.get('source_date', 'yesterday')}'s food to "
            f"{args.get('target_date', 'today')}"
        )
    if name == "get_health_history":
        return "Reading health history"
    if name == "log_workout":
        return f"Logging workout: {args.get('activity_type', '')}"
    if name == "log_weight":
        return f"Logging weight: {args.get('weight_kg', '')} kg"
    if name == "set_goal":
        return f"Setting goal: {args.get('goal_type', '')} = {args.get('target_value', '')}"
    if name == "set_profile":
        return "Updating profile"
    if name == "get_metabolic_summary":
        return "Working out BMR and targets"
    if name == "get_recent_nutrition":
        return f"Reading nutrition history ({args.get('days_back', 1)}d)"
    if name == "get_weight_trend":
        return f"Reading weight trend ({args.get('days', 30)}d)"
    if name == "get_recent_workouts":
        return f"Reading recent workouts ({args.get('limit', 10)} max)"
    return f"Calling {name}"


def summarise_lifestyle_result(name: str, result: str) -> str:
    """Human-readable one-liner for the 'tool_result' event."""
    if result.startswith("ERROR:"):
        return result[:120]
    if name in ("log_nutrition", "delete_nutrition", "log_workout", "log_weight", "set_goal", "set_profile", "get_metabolic_summary", "add_food_preference", "delete_food_preference", "update_food_preference", "set_meal_plan", "get_today_summary", "log_exercise", "delete_exercise_set", "prep_split"):
        # First line of result is the success summary built above.
        return result.split("\n", 1)[0][:140]
    # Read-style tools: show line count
    lines = result.count("\n") + 1 if result else 0
    return f"{lines} line(s) returned"

