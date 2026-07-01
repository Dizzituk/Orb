# FILE: app/tools/finance_tools.py
# Purpose: Work-day finance tools — delivery-ledger capability for the LLM (v8.0).
# Called-by: app.debug.gemini_finance_tools, app.finance.services.work_folder_logger, app.finance.services.work_upload_logger, app.tools.registry
# Depends-on: app.db, app.finance.services, app.finance.services.work_day_service, app.tools.finance_tool_schemas (+1 more)
# Last-renovated: 2026-06-11
"""
Work-day finance tools — delivery-ledger capability for the LLM (v8.0).

Five tools that let ANY ASTRA chat (and AstraBridge, which routes through
the same chat backend) write a working day straight into the database:

  start_work_day        open today's row: start time + start odometer
  finish_work_day       close it: end time, end odometer, parcels (+ fuel)
  reconcile_work_day    close a previous day that was never signed out
  set_personal_mileage  split miles into work vs personal for a day
  get_work_day          read back a day's figures

Design: the LLM never rewrites a layout or a file — it calls a tool with a
few values and the row is written/updated; the dashboard reads the row.
Each handler opens a short-lived session, calls work_day_service (which owns
all the maths and is unit-tested), and returns a plain dict. Errors come
back as {ok: false, error: ...} so the model can react and ask the user.

Self-contained on purpose: schemas live here too, so adding a new work-day
capability later means editing this one file.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
# Session + small input coercers
# ────────────────────────────────────────────────────────────────────────

@contextmanager
def _db_session():
    """Yield a short-lived DB session, always closing on exit."""
    from app.db import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _parse_date(value) -> Optional[date]:
    """Parse 'YYYY-MM-DD' to a date. None/blank -> None (service uses today)."""
    if not value:
        return None
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value).strip()[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _opt_str(d: dict, key: str) -> Optional[str]:
    v = d.get(key)
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _opt_int(d: dict, key: str) -> Optional[int]:
    v = d.get(key)
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _opt_float(d: dict, key: str) -> Optional[float]:
    v = d.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _day_to_dict(day) -> dict:
    """Compact view of a WorkDay row for tool output."""
    return {
        "work_date": day.work_date.isoformat() if day.work_date else None,
        "status": day.status,
        "start_time": day.start_time,
        "end_time": day.end_time,
        "total_hours": day.total_hours,
        "start_odometer": day.start_odometer,
        "end_odometer": day.end_odometer,
        "total_distance": day.total_distance,
        "work_miles": day.work_miles,
        "personal_miles": day.personal_miles,
        "parcels": day.parcels,
        "collections": day.collections,
        "rate_per_parcel": day.rate_per_parcel,
        "gross_earnings": day.gross_earnings,
        "per_hour": day.per_hour,
        "per_parcel": day.per_parcel,
        "fuel_cost": day.fuel_cost,
        "route_area": day.route_area,
        "tour_id": day.tour_id,
        "tax_year": day.tax_year,
    }


# ────────────────────────────────────────────────────────────────────────
# Handlers
# ────────────────────────────────────────────────────────────────────────

async def start_work_day_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Open (or update) today's work day with start time / start odometer.
    Also surfaces a previous unclosed shift so the model can prompt the user."""
    work_date = _parse_date(input_data.get("work_date"))
    start_time = _opt_str(input_data, "start_time")
    start_odometer = _opt_float(input_data, "start_odometer")
    route_area = _opt_str(input_data, "route_area")
    tour_id = _opt_str(input_data, "tour_id")

    try:
        from app.finance.services import work_day_service as wds
        with _db_session() as db:
            day = wds.start_day(
                db, work_date=work_date, start_time=start_time,
                start_odometer=start_odometer, route_area=route_area, tour_id=tour_id,
            )
            unclosed = wds.check_unclosed(db, before=day.work_date)
            unclosed_info = None
            if unclosed is not None:
                unclosed_info = {
                    "date": unclosed.work_date.isoformat(),
                    "start_odometer": unclosed.start_odometer,
                    "suggested_end_odometer": day.start_odometer,
                    "note": (
                        "This earlier shift was never closed out. If the van wasn't "
                        "driven in between, its end odometer equals today's start "
                        f"({day.start_odometer}). Confirm the end mileage and the "
                        "parcel count with the user, then call reconcile_work_day "
                        "for that date."
                    ),
                }
            out = {
                "ok": True,
                "work_date": day.work_date.isoformat(),
                "status": day.status,
                "start_time": day.start_time,
                "start_odometer": day.start_odometer,
                "tax_year": day.tax_year,
                "unclosed_previous": unclosed_info,
            }
        return out
    except Exception as exc:
        logger.exception("[finance_tools] start_work_day failed")
        return {"ok": False, "work_date": None, "unclosed_previous": None, "error": str(exc)}


async def finish_work_day_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Close out a work day: end time, end odometer, parcel count (read from
    the end-of-day screenshot), optional fuel and personal-mileage split."""
    work_date = _parse_date(input_data.get("work_date"))
    try:
        from app.finance.services import work_day_service as wds
        with _db_session() as db:
            day = wds.finish_day(
                db,
                work_date=work_date,
                start_time=_opt_str(input_data, "start_time"),
                start_odometer=_opt_float(input_data, "start_odometer"),
                end_time=_opt_str(input_data, "end_time"),
                end_odometer=_opt_float(input_data, "end_odometer"),
                parcels=_opt_int(input_data, "parcels"),
                collections=_opt_int(input_data, "collections"),
                rate_per_parcel=_opt_float(input_data, "rate_per_parcel"),
                fuel_cost=_opt_float(input_data, "fuel_cost"),
                personal_miles=_opt_float(input_data, "personal_miles"),
                screenshot_path=_opt_str(input_data, "screenshot_path"),
            )
            out = {"ok": True, **_day_to_dict(day)}
        return out
    except Exception as exc:
        logger.exception("[finance_tools] finish_work_day failed")
        return {"ok": False, "work_date": None, "error": str(exc)}


async def reconcile_work_day_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Close a previous day the user forgot to sign out. prev_date is required;
    end_odometer is usually today's start odometer (confirmed with the user)."""
    prev_date = _parse_date(input_data.get("prev_date"))
    if prev_date is None:
        return {"ok": False, "work_date": None, "error": "prev_date (YYYY-MM-DD) is required"}
    try:
        from app.finance.services import work_day_service as wds
        with _db_session() as db:
            day = wds.reconcile_previous_day(
                db, prev_date=prev_date,
                end_odometer=_opt_float(input_data, "end_odometer"),
                parcels=_opt_int(input_data, "parcels"),
                end_time=_opt_str(input_data, "end_time"),
            )
            out = {"ok": True, **_day_to_dict(day)}
        return out
    except Exception as exc:
        logger.exception("[finance_tools] reconcile_work_day failed")
        return {"ok": False, "work_date": None, "error": str(exc)}


async def set_personal_mileage_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Record how many of a day's miles were personal, not work. Use this when
    the user confirms an unusual mileage gap was personal driving."""
    work_date = _parse_date(input_data.get("work_date")) or date.today()
    personal_miles = _opt_float(input_data, "personal_miles")
    if personal_miles is None:
        return {"ok": False, "work_date": None, "error": "personal_miles is required"}
    try:
        from app.finance.services import work_day_service as wds
        with _db_session() as db:
            day = wds.set_mileage_split(db, work_date=work_date, personal_miles=personal_miles)
            out = {
                "ok": True,
                "work_date": day.work_date.isoformat(),
                "total_distance": day.total_distance,
                "work_miles": day.work_miles,
                "personal_miles": day.personal_miles,
            }
        return out
    except Exception as exc:
        logger.exception("[finance_tools] set_personal_mileage failed")
        return {"ok": False, "work_date": None, "error": str(exc)}


async def get_work_day_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Read back a single day's figures (defaults to today).

    Returns shift_state (none | open | complete) and a plain-language note so
    the model never mistakes an open-but-started shift for an unlogged one. An
    OPEN day with a start recorded is fully logged — it just isn't closed yet,
    so its earnings/miles read zero until finish_work_day runs."""
    work_date = _parse_date(input_data.get("work_date")) or date.today()
    try:
        from app.finance.services import work_day_service as wds
        with _db_session() as db:
            day = wds.get_day(db, work_date)
            if day is None:
                return {
                    "found": False, "day": None, "shift_state": "none",
                    "note": "No work day logged for this date yet.",
                }
            has_start = day.start_odometer is not None or day.start_time is not None
            if day.status == "complete":
                state = "complete"
                note = "Shift is closed and fully logged. Report the figures as-is."
            elif has_start:
                state = "open"
                note = (
                    "Shift is OPEN with the start ALREADY recorded "
                    f"(start_time={day.start_time}, start_odometer={day.start_odometer}). "
                    "It is logged — just not closed yet, so earnings and miles stay "
                    "zero until it is. Do NOT treat this as missing/unlogged and do "
                    "NOT ask the user for the start. To close it, call finish_work_day "
                    "with the end figures only."
                )
            else:
                state = "open"
                note = "Shift is OPEN but no start time/odometer recorded yet."
            out = {"found": True, "day": _day_to_dict(day),
                   "shift_state": state, "note": note}
        return out
    except Exception as exc:
        logger.exception("[finance_tools] get_work_day failed")
        return {"found": False, "day": None, "shift_state": "error", "error": str(exc)}


# Schemas live in a sibling module to keep this file small and modular.
from app.tools.finance_tool_schemas import (  # noqa: E402
    START_WORK_DAY_INPUT, START_WORK_DAY_OUTPUT,
    FINISH_WORK_DAY_INPUT, FINISH_WORK_DAY_OUTPUT,
    RECONCILE_WORK_DAY_INPUT, RECONCILE_WORK_DAY_OUTPUT,
    SET_PERSONAL_MILEAGE_INPUT, SET_PERSONAL_MILEAGE_OUTPUT,
    GET_WORK_DAY_INPUT, GET_WORK_DAY_OUTPUT,
)


# ────────────────────────────────────────────────────────────────────────
# Registration
# ────────────────────────────────────────────────────────────────────────

def register_finance_tools() -> None:
    """Register the work-day ledger tools with the global tool registry.
    Called once from registry._register_defaults at module load."""
    from app.tools.registry import ToolDefinition, register_tool

    register_tool(ToolDefinition(
        name="start_work_day",
        version="v1",
        description=(
            "Open today's delivery work day. Call this when the user says they "
            "are starting work / clocking on. Pass start_time (HH:MM) and "
            "start_odometer (the van mileage reading) when given. If the "
            "response includes 'unclosed_previous', a past shift was never "
            "closed — tell the user and offer to set its end mileage to today's "
            "start, then call reconcile_work_day once they confirm."
        ),
        input_schema=START_WORK_DAY_INPUT,
        output_schema=START_WORK_DAY_OUTPUT,
        handler=start_work_day_handler,
    ))

    register_tool(ToolDefinition(
        name="finish_work_day",
        version="v1",
        description=(
            "Close out today's (or a given) delivery work day. Call this "
            "IMMEDIATELY when the user says they have finished work or are "
            "clocking off — do not deliberate or ask anything first. Pass "
            "whatever end figures they give: end_time (HH:MM), end_odometer and "
            "parcels (the parcel count comes from the end-of-day screenshot). "
            "You do NOT need the start time or start odometer: they were stored "
            "on today's row by start_work_day this morning, and this tool reads "
            "them back automatically to compute miles and hours. NEVER ask the "
            "user for the start mileage or start time at end of day. Earnings "
            "are computed automatically at the per-parcel rate. Pass fuel_cost "
            "or personal_miles only if the user mentions them. When it returns, "
            "the row is complete — just report the summary it gives back "
            "(parcels, gross, miles, hours); do not re-read or re-confirm the "
            "start."
        ),
        input_schema=FINISH_WORK_DAY_INPUT,
        output_schema=FINISH_WORK_DAY_OUTPUT,
        handler=finish_work_day_handler,
    ))

    register_tool(ToolDefinition(
        name="reconcile_work_day",
        version="v1",
        description=(
            "Close out a PREVIOUS day the user forgot to sign out of. prev_date "
            "(YYYY-MM-DD) is required. end_odometer is normally today's start "
            "odometer (the van didn't move overnight) — confirm with the user "
            "first. Pass the parcel count for that day if known."
        ),
        input_schema=RECONCILE_WORK_DAY_INPUT,
        output_schema=RECONCILE_WORK_DAY_OUTPUT,
        handler=reconcile_work_day_handler,
    ))

    register_tool(ToolDefinition(
        name="set_personal_mileage",
        version="v1",
        description=(
            "Record how many of a day's miles were personal rather than work. "
            "Use when the user confirms an unusual mileage gap was personal "
            "driving. Work miles are recalculated as total minus personal."
        ),
        input_schema=SET_PERSONAL_MILEAGE_INPUT,
        output_schema=SET_PERSONAL_MILEAGE_OUTPUT,
        handler=set_personal_mileage_handler,
    ))

    register_tool(ToolDefinition(
        name="get_work_day",
        version="v1",
        description=(
            "Read back a single day's work figures (defaults to today): parcels, "
            "earnings, hours, mileage split and per-hour/per-parcel rates. Use "
            "to answer 'how am I doing today' or to check what's already logged. "
            "Read shift_state and note in the result: an OPEN shift that already "
            "has a start is logged and just needs closing — it is NOT unlogged, "
            "and zero earnings on it only mean it isn't closed yet."
        ),
        input_schema=GET_WORK_DAY_INPUT,
        output_schema=GET_WORK_DAY_OUTPUT,
        handler=get_work_day_handler,
    ))

    logger.info("[finance_tools] registered 5 work-day ledger tools")
