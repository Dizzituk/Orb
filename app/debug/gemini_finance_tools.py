# FILE: app/debug/gemini_finance_tools.py
"""
Work-day (delivery ledger) tool declarations and executors for the Gemini
multimodal tool loop (app.debug.gemini_tool_loop).

Why this exists, separate from app/tools/finance_tools.py:
    Image-bearing chat turns route through the Gemini multimodal path,
    which has its OWN native function-calling format and executor map -
    it does not go through the generic app.tools registry. Without these
    declarations the loop had only file tools, so "put this screenshot in
    the work tab" got satisfied by edit_file rewriting an HTML dashboard
    instead of writing a row to the database. This module fixes that.

    Like gemini_lifestyle_tools.py, the executors here are thin adapters:
    they delegate to the already-tested handlers in app.tools.finance_tools
    (which own the maths and open their own DB session). No duplicated
    business logic.

        Gemini -> _exec_* (here) -> *_handler (app.tools.finance_tools)
              -> work_day_service -> DB
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Function declarations for Gemini (flat scalar params, like the file tools)
# ---------------------------------------------------------------------------

FINANCE_TOOL_DECLARATIONS: List[Dict[str, Any]] = [
    {
        "name": "start_work_day",
        "description": (
            "Open today's delivery WORK DAY in the work ledger (the 'Work' "
            "tab / work dashboard). USE THIS - never write_file or edit_file - "
            "when the user says they're starting work / clocking on, or to "
            "begin logging a shift. It writes a row to the work-day database "
            "that the Work tab reads; do NOT log shifts by editing HTML or "
            "dashboard files. Pass start_time (HH:MM) and start_odometer (the "
            "van mileage reading) when given. If the result includes "
            "'unclosed_previous', a past shift was never closed - tell the "
            "user, offer to set its end mileage to today's start, then call "
            "reconcile_work_day once they confirm."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "work_date": {"type": "string", "description": "YYYY-MM-DD; omit for today"},
                "start_time": {"type": "string", "description": "Clock-on time, 24h HH:MM"},
                "start_odometer": {"type": "number", "description": "Van odometer reading at clock-on"},
                "route_area": {"type": "string", "description": "Optional route/area name"},
                "tour_id": {"type": "string", "description": "Optional tour/round id"},
            },
        },
    },
    {
        "name": "finish_work_day",
        "description": (
            "Close out a delivery WORK DAY in the work ledger (the 'Work' tab). "
            "USE THIS - never write_file or edit_file - to finish/log a shift, "
            "especially from an end-of-day screenshot. Read the delivered "
            "parcel count from the attached screenshot and pass it as "
            "'parcels'; gross earnings are computed automatically at the "
            "per-parcel rate. Pass end_time (HH:MM) and end_odometer; add "
            "fuel_cost or personal_miles if the user mentions them. If the user "
            "gives BOTH start and end mileage in one go (logging a whole "
            "finished shift at once), pass start_odometer AND end_odometer "
            "together - distance is worked out for you and you do NOT need a "
            "separate start_work_day call. Writes / updates the database row "
            "the Work tab reads - do NOT write HTML."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "work_date": {"type": "string", "description": "YYYY-MM-DD; omit for today"},
                "start_time": {"type": "string", "description": "Clock-on time HH:MM - only when logging a whole shift at once"},
                "start_odometer": {"type": "number", "description": "Odometer at clock-on - only when logging a whole finished shift at once (else use start_work_day)"},
                "end_time": {"type": "string", "description": "Clock-off time, 24h HH:MM"},
                "end_odometer": {"type": "number", "description": "Van odometer reading at clock-off"},
                "parcels": {"type": "integer", "description": "Delivered parcel count, read from the end-of-day screenshot"},
                "collections": {"type": "integer", "description": "Optional collections count"},
                "rate_per_parcel": {"type": "number", "description": "Per-parcel pay rate; defaults to 2.35 if omitted"},
                "fuel_cost": {"type": "number", "description": "Optional fuel spend for the day, GBP"},
                "personal_miles": {"type": "number", "description": "Optional miles that were personal, not work"},
            },
        },
    },
    {
        "name": "reconcile_work_day",
        "description": (
            "Close out a PREVIOUS delivery day the user forgot to sign out of. "
            "prev_date (YYYY-MM-DD) is required. end_odometer is normally "
            "today's start odometer (the van didn't move overnight) - confirm "
            "with the user first. Pass the parcel count for that day if known. "
            "Writes to the work-day database, not a file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prev_date": {"type": "string", "description": "REQUIRED. YYYY-MM-DD of the day to close"},
                "end_odometer": {"type": "number", "description": "Usually today's start odometer"},
                "parcels": {"type": "integer", "description": "Parcel count for that day, if known"},
                "end_time": {"type": "string", "description": "Clock-off time, 24h HH:MM"},
            },
            "required": ["prev_date"],
        },
    },
    {
        "name": "set_personal_mileage",
        "description": (
            "Record how many of a day's miles were personal rather than work, "
            "in the work ledger. Use when the user confirms an unusual mileage "
            "gap was personal driving. Work miles are recalculated as total "
            "minus personal. Updates the work-day row, not a file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "work_date": {"type": "string", "description": "YYYY-MM-DD; omit for today"},
                "personal_miles": {"type": "number", "description": "REQUIRED. Miles that were personal, not work"},
            },
            "required": ["personal_miles"],
        },
    },
    {
        "name": "get_work_day",
        "description": (
            "Read back a single day's work-ledger figures (defaults to today): "
            "parcels, earnings, hours, mileage split and per-hour/per-parcel "
            "rates. Use to answer 'how am I doing today' or to check what's "
            "already logged before you change it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "work_date": {"type": "string", "description": "YYYY-MM-DD; omit for today"},
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Executors - thin adapters delegating to app.tools.finance_tools handlers
# ---------------------------------------------------------------------------

def _money(v) -> str:
    try:
        return "\u00a3%.2f" % float(v)
    except (TypeError, ValueError):
        return str(v)


def _num(v, dp=0) -> str:
    try:
        return ("%%.%df" % dp) % float(v)
    except (TypeError, ValueError):
        return str(v)


async def _exec_start_work_day(args: dict) -> str:
    from app.tools.finance_tools import start_work_day_handler
    r = await start_work_day_handler(args, context=None)
    if not r.get("ok"):
        return "ERROR: " + str(r.get("error", "could not start work day"))
    bits = ["Started work day " + str(r.get("work_date"))]
    if r.get("start_time"):
        bits.append("at " + str(r["start_time"]))
    if r.get("start_odometer") is not None:
        bits.append("odo " + _num(r["start_odometer"]))
    line = ", ".join(bits)
    unclosed = r.get("unclosed_previous")
    if unclosed:
        line += (". NOTE: the earlier shift on %s was never closed - suggested "
                 "end odometer is %s (today's start). Confirm end mileage + "
                 "parcels with the user, then call reconcile_work_day for %s."
                 % (unclosed.get("date"), unclosed.get("suggested_end_odometer"),
                    unclosed.get("date")))
    return line


async def _exec_finish_work_day(args: dict) -> str:
    from app.tools.finance_tools import finish_work_day_handler
    r = await finish_work_day_handler(args, context=None)
    if not r.get("ok"):
        return "ERROR: " + str(r.get("error", "could not finish work day"))
    bits = ["Closed work day " + str(r.get("work_date"))]
    if r.get("parcels") is not None:
        bits.append(_num(r["parcels"]) + " parcels")
    if r.get("gross_earnings") is not None:
        bits.append(_money(r["gross_earnings"]))
    if r.get("total_hours") is not None:
        bits.append(_num(r["total_hours"], 1) + "h")
    if r.get("work_miles") is not None:
        bits.append(_num(r["work_miles"]) + " work mi")
    if r.get("per_hour") is not None:
        bits.append(_money(r["per_hour"]) + "/h")
    return ", ".join(bits)


async def _exec_reconcile_work_day(args: dict) -> str:
    from app.tools.finance_tools import reconcile_work_day_handler
    r = await reconcile_work_day_handler(args, context=None)
    if not r.get("ok"):
        return "ERROR: " + str(r.get("error", "could not reconcile day"))
    bits = ["Reconciled " + str(r.get("work_date"))]
    if r.get("parcels") is not None:
        bits.append(_num(r["parcels"]) + " parcels")
    if r.get("gross_earnings") is not None:
        bits.append(_money(r["gross_earnings"]))
    if r.get("total_hours") is not None:
        bits.append(_num(r["total_hours"], 1) + "h")
    return ", ".join(bits)


async def _exec_set_personal_mileage(args: dict) -> str:
    from app.tools.finance_tools import set_personal_mileage_handler
    r = await set_personal_mileage_handler(args, context=None)
    if not r.get("ok"):
        return "ERROR: " + str(r.get("error", "could not set mileage split"))
    return ("Mileage split for %s: %s work / %s personal (total %s)"
            % (r.get("work_date"), _num(r.get("work_miles")),
               _num(r.get("personal_miles")), _num(r.get("total_distance"))))


async def _exec_get_work_day(args: dict) -> str:
    from app.tools.finance_tools import get_work_day_handler
    r = await get_work_day_handler(args, context=None)
    if r.get("error"):
        return "ERROR: " + str(r["error"])
    if not r.get("found"):
        return "No work day logged for that date yet."
    d = r.get("day") or {}
    bits = ["%s [%s]" % (d.get("work_date"), d.get("status"))]
    if d.get("parcels") is not None:
        bits.append(_num(d["parcels"]) + " parcels")
    if d.get("gross_earnings") is not None:
        bits.append(_money(d["gross_earnings"]))
    if d.get("total_hours") is not None:
        bits.append(_num(d["total_hours"], 1) + "h")
    if d.get("work_miles") is not None:
        bits.append(_num(d["work_miles"]) + " work mi")
    if d.get("per_hour") is not None:
        bits.append(_money(d["per_hour"]) + "/h")
    return ", ".join(bits)


FINANCE_TOOL_EXECUTORS = {
    "start_work_day": _exec_start_work_day,
    "finish_work_day": _exec_finish_work_day,
    "reconcile_work_day": _exec_reconcile_work_day,
    "set_personal_mileage": _exec_set_personal_mileage,
    "get_work_day": _exec_get_work_day,
}


def summarise_finance_call(name: str, args: dict) -> str:
    if name == "start_work_day":
        return "Starting work day" + (" at " + args["start_time"] if args.get("start_time") else "")
    if name == "finish_work_day":
        p = args.get("parcels")
        return "Closing work day" + ((" (%s parcels)" % p) if p is not None else "")
    if name == "reconcile_work_day":
        return "Reconciling day " + str(args.get("prev_date", ""))
    if name == "set_personal_mileage":
        return "Setting personal mileage"
    if name == "get_work_day":
        return "Reading work day " + str(args.get("work_date") or "today")
    return "Calling " + name


def summarise_finance_result(name: str, result: str) -> str:
    if result.startswith("ERROR:"):
        return result[:140]
    return result.split("\n", 1)[0][:140]


# ---------------------------------------------------------------------------
# Whole-folder batch logger (deterministic printer loop)
#
# v1.0 (2026-05-30): "log all the screenshots in the folder" is now ONE tool
# call backed by a deterministic loop (parse one filename -> one fast vision
# read -> write one row -> next), instead of the chat model batch-reading
# every image and then narrating without writing any rows. The model only
# triggers the pass; it does not read images or call finish_work_day per day.
# Loop lives in app/finance/services/work_folder_logger.py.
# ---------------------------------------------------------------------------

async def _exec_log_work_folder(args: dict) -> str:
    from app.finance.services.work_folder_logger import log_work_folder

    def _flt(key, default):
        v = (args or {}).get(key)
        if v is None or v == "":
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    folder = (args or {}).get("folder") or None
    start_time = (str((args or {}).get("start_time") or "").strip()) or "10:00"

    r = await log_work_folder(
        folder=folder,
        start_time=start_time,
        miles=_flt("miles", 120.0),
        rate_per_parcel=_flt("rate_per_parcel", None),
        overwrite=bool((args or {}).get("overwrite")),
    )
    if not r.get("ok"):
        return "ERROR: " + str(r.get("error", "could not log the folder"))

    logged = r.get("logged", [])
    skipped = r.get("skipped", [])
    failed = r.get("failed", [])
    parts = []
    if logged:
        total = sum((x.get("earnings") or 0) for x in logged)
        dates = [x["date"] for x in logged]
        parts.append("Logged %d day(s) %s to %s, %s total"
                     % (len(logged), dates[0], dates[-1], _money(total)))
        sample = "; ".join("%s %s %s parcels" % (x["date"], x["finish"], x["parcels"])
                           for x in logged[:3])
        parts.append("e.g. " + sample + ("..." if len(logged) > 3 else ""))
    else:
        parts.append("Logged 0 new days")
    if skipped:
        parts.append("skipped %d already logged" % len(skipped))
    if failed:
        fl = "; ".join("%s (%s)" % (f.get("date"), f.get("reason")) for f in failed[:4])
        parts.append("%d need a manual check: %s" % (len(failed), fl))
    return ". ".join(parts) + "."


FINANCE_TOOL_DECLARATIONS.append({
    "name": "log_work_folder",
    "description": (
        "Log a WHOLE FOLDER of end-of-day delivery screenshots into the work "
        "ledger in one pass. USE THIS - not read_image, not finish_work_day - "
        "whenever the user asks to log/add 'all the screenshots', 'every "
        "screenshot in the folder', 'the rest of them', or otherwise batch-log "
        "a folder of shift screenshots. It walks the folder one day at a time, "
        "reads the date and finish time from each filename, reads the parcel "
        "count off each image itself, and writes that day's row before moving "
        "on - so you do NOT read the images yourself and you do NOT call "
        "finish_work_day per day. Days already logged are skipped, so it is "
        "safe to re-run. Defaults: start_time 10:00 and 120 miles per day; "
        "pass them only to override. Returns a summary of what was logged, "
        "skipped, and anything that needs a manual check - relay it to the user."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "folder": {"type": "string", "description": "Folder path; omit to use the Documents/Work folder"},
            "start_time": {"type": "string", "description": "Clock-on time for every day, 24h HH:MM (default 10:00)"},
            "miles": {"type": "number", "description": "Miles driven per day (default 120)"},
            "rate_per_parcel": {"type": "number", "description": "Per-parcel pay rate; defaults to 2.35"},
            "overwrite": {"type": "boolean", "description": "If true, re-log days even if already logged (default false)"},
        },
    },
})

FINANCE_TOOL_EXECUTORS["log_work_folder"] = _exec_log_work_folder


async def _exec_log_uploaded_work_screenshots(args: dict) -> str:
    from app.finance.services.work_upload_logger import log_uploaded_work_screenshots

    def _flt(key, default):
        v = (args or {}).get(key)
        if v is None or v == "":
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _int(key, default):
        v = (args or {}).get(key)
        if v is None or v == "":
            return default
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default

    start_time = (str((args or {}).get("start_time") or "").strip()) or "10:00"

    r = await log_uploaded_work_screenshots(
        count=_int("count", 1),
        start_time=start_time,
        miles=_flt("miles", 120.0),
        rate_per_parcel=_flt("rate_per_parcel", None),
        overwrite=bool((args or {}).get("overwrite")),
    )
    if not r.get("ok"):
        msg = "ERROR: " + str(r.get("error", "could not log the uploaded screenshot(s)"))
        und = r.get("undated") or []
        if und:
            msg += " (no shift date in: %s)" % ", ".join(str(u) for u in und[:3])
        return msg

    logged = r.get("logged", [])
    skipped = r.get("skipped", [])
    failed = r.get("failed", [])
    parts = []
    if logged:
        total = sum((x.get("earnings") or 0) for x in logged)
        dates = [x["date"] for x in logged]
        parts.append("Logged %d uploaded day(s) %s to %s, %s total"
                     % (len(logged), dates[0], dates[-1], _money(total)))
        sample = "; ".join("%s %s %s parcels" % (x["date"], x["finish"], x["parcels"])
                           for x in logged[:3])
        parts.append("e.g. " + sample + ("..." if len(logged) > 3 else ""))
    else:
        parts.append("Logged 0 new days")
    if skipped:
        parts.append("skipped %d already logged" % len(skipped))
    if failed:
        fl = "; ".join("%s (%s)" % (f.get("date"), f.get("reason")) for f in failed[:4])
        parts.append("%d need a manual check: %s" % (len(failed), fl))
    return ". ".join(parts) + "."


FINANCE_TOOL_DECLARATIONS.append({
    "name": "log_uploaded_work_screenshots",
    "description": (
        "Log delivery screenshot(s) the user just UPLOADED from their phone "
        "into the work ledger. USE THIS - not read_image, not finish_work_day - "
        "when the user uploads one or more end-of-day shift screenshots in the "
        "chat and asks to log/add them ('log this', 'log these', 'add this "
        "shift'). It reads the date and finish time from each screenshot's "
        "original filename, reads the parcel count off the image itself, and "
        "writes the row - so you do NOT read the image yourself. Set count to "
        "how many screenshots the user just sent (default 1). Days already "
        "logged are skipped, so it is safe to re-run. Defaults: start_time "
        "10:00 and 120 miles. Returns a summary - relay it to the user, "
        "including anything that needs a manual check."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "description": "How many of the most recent uploads to log (default 1)"},
            "start_time": {"type": "string", "description": "Clock-on time, 24h HH:MM (default 10:00)"},
            "miles": {"type": "number", "description": "Miles driven per day (default 120)"},
            "rate_per_parcel": {"type": "number", "description": "Per-parcel pay rate; defaults to 2.35"},
            "overwrite": {"type": "boolean", "description": "If true, re-log days even if already logged (default false)"},
        },
    },
})

FINANCE_TOOL_EXECUTORS["log_uploaded_work_screenshots"] = _exec_log_uploaded_work_screenshots
