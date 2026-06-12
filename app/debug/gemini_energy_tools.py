# FILE: app/debug/gemini_energy_tools.py
# Purpose: Energy-engine tool executors for the Gemini multimodal loop —
#          log_work_debrief and get_energy_today. Lives in its own module so
#          gemini_lifestyle_tools.py stays under its size ceiling.
# Called-by: app.debug.gemini_lifestyle_tools (registry merge)
# Depends-on: app.lifestyle.energy, app.lifestyle.energy_ledger, app.db
"""Thin async executors over the energy engine (app/lifestyle/energy.py).
Declarations for these live in gemini_lifestyle_decls2.py alongside the rest
of the lifestyle tool surface."""
from __future__ import annotations

import logging
from datetime import date

logger = logging.getLogger(__name__)


async def _exec_log_work_debrief(args: dict) -> str:
    from app.db import SessionLocal
    from app.lifestyle.energy import (
        estimate_day_energy, normalise_effort, set_day_effort,
    )
    from app.lifestyle.energy_ledger import compute_today_target

    effort = normalise_effort(str(args.get("effort") or ""))
    if effort is None:
        return ("ERROR: effort must be light, normal, heavy or very_heavy "
                "(natural synonyms accepted)")
    try:
        day = date.fromisoformat(str(args["date"])) if args.get("date") else date.today()
    except ValueError:
        return "ERROR: date must be YYYY-MM-DD"

    db = SessionLocal()
    try:
        set_day_effort(day, effort, source="gemini_tool",
                       notes=(args.get("notes") or None))
        est = estimate_day_energy(db, day)
        lines = [f"Day effort recorded: {effort.replace('_', ' ')} for {day.isoformat()}."]
        # Belt-and-braces against tool shadowing (2026-06-11 bug): if the
        # user's words carried work figures, remind the model this tool
        # logged none of them.
        blob = f"{args.get('notes') or ''}".lower()
        import re as _re
        if _re.search(r"\d", blob) or _re.search(
                r"\b(mile|odometer|parcel|collection|pay|earn|fuel|expense|£|invoice)", blob):
            lines.append("NOTE: this tool logged NO work data. If the user gave "
                         "miles, parcels, pay or fuel, call finish_work_day / "
                         "log_expense now.")
        if est.get("total_burn"):
            lines.append(f"Burn estimate: ~{est['total_burn']} kcal ({est['label']}).")
        if est.get("is_work_day") and (est.get("work") or {}).get("parcels"):
            lines.append(f"Work day: {est['work']['parcels']} parcels.")
        if day == date.today():
            tgt = compute_today_target(db)
            if tgt.get("target_calories"):
                lines.append(f"Intake target today: ~{int(tgt['target_calories'])} kcal "
                             f"({tgt.get('week_note', '')}).")
        return "\n".join(lines)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("[gemini_energy] debrief failed: %s", exc)
        return f"ERROR: {exc}"
    finally:
        db.close()


async def _exec_get_energy_today(args: dict) -> str:
    from app.db import SessionLocal
    from app.lifestyle.energy import estimate_day_energy
    from app.lifestyle.energy_ledger import compute_today_target, get_week_ledger

    db = SessionLocal()
    try:
        est = estimate_day_energy(db)
        tgt = compute_today_target(db)
        led = get_week_ledger(db)
        lines = []
        if est.get("total_burn"):
            lines.append(f"Today's burn estimate: ~{est['total_burn']} kcal "
                         f"= BMR {est.get('bmr')} + activity {est.get('activity_kcal')} "
                         f"({est['label']}, confidence {est.get('confidence')}).")
        else:
            lines.append(f"Burn estimate unavailable ({est.get('label')}).")
        if est.get("is_work_day"):
            parcels = (est.get("work") or {}).get("parcels")
            lines.append(f"Work day (effort: {est.get('effort')}"
                         + (f", {parcels} parcels" if parcels else "") + ").")
        if tgt.get("target_calories"):
            lines.append(f"Dynamic intake target: ~{int(tgt['target_calories'])} kcal "
                         f"— {tgt.get('week_note', 'weekly ledger')}.")
        lines.append(f"Week ledger: {led.get('message', '')}")
        return "\n".join(lines)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("[gemini_energy] energy_today failed: %s", exc)
        return f"ERROR: {exc}"
    finally:
        db.close()


ENERGY_TOOL_EXECUTORS = {
    "set_day_effort": _exec_log_work_debrief,
    "log_work_debrief": _exec_log_work_debrief,  # legacy alias, decl renamed 2026-06-11
    "get_energy_today": _exec_get_energy_today,
}
