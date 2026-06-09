# FILE: app/tools/finance_tool_schemas.py
"""
JSON-Schemas for the work-day finance tools (v8.0).

Split out from finance_tools.py to keep each file small and single-
responsibility. These dicts are passed straight through as Gemini
function-call parameters, so they are plain JSON-Schema.
"""
from __future__ import annotations

_DATE = {"type": ["string", "null"], "description": "YYYY-MM-DD; omit for today"}
_HHMM = {"type": ["string", "null"], "description": "24h time HH:MM"}


START_WORK_DAY_INPUT = {
    "type": "object",
    "properties": {
        "work_date": _DATE,
        "start_time": _HHMM,
        "start_odometer": {"type": ["number", "null"], "description": "Van odometer at clock-on"},
        "route_area": {"type": ["string", "null"], "maxLength": 200},
        "tour_id": {"type": ["string", "null"], "maxLength": 40},
    },
}
START_WORK_DAY_OUTPUT = {
    "type": "object",
    "required": ["ok"],
    "properties": {
        "ok": {"type": "boolean"},
        "work_date": {"type": ["string", "null"]},
        "status": {"type": ["string", "null"]},
        "start_time": {"type": ["string", "null"]},
        "start_odometer": {"type": ["number", "null"]},
        "tax_year": {"type": ["string", "null"]},
        "unclosed_previous": {"type": ["object", "null"]},
        "error": {"type": "string"},
    },
}

FINISH_WORK_DAY_INPUT = {
    "type": "object",
    "properties": {
        "work_date": _DATE,
        "start_time": _HHMM,
        "start_odometer": {"type": ["number", "null"], "description": "Van odometer at clock-on (only if logging a whole shift at once)"},
        "end_time": _HHMM,
        "end_odometer": {"type": ["number", "null"], "description": "Van odometer at clock-off"},
        "parcels": {"type": ["integer", "null"], "description": "Delivered parcel count (from end-of-day screenshot)"},
        "collections": {"type": ["integer", "null"]},
        "rate_per_parcel": {"type": ["number", "null"], "description": "Defaults to 2.35"},
        "fuel_cost": {"type": ["number", "null"]},
        "personal_miles": {"type": ["number", "null"]},
        "screenshot_path": {"type": ["string", "null"]},
    },
}
FINISH_WORK_DAY_OUTPUT = {
    "type": "object",
    "required": ["ok"],
    "properties": {
        "ok": {"type": "boolean"},
        "work_date": {"type": ["string", "null"]},
        "status": {"type": ["string", "null"]},
        "parcels": {"type": ["integer", "null"]},
        "gross_earnings": {"type": ["number", "null"]},
        "total_distance": {"type": ["number", "null"]},
        "work_miles": {"type": ["number", "null"]},
        "personal_miles": {"type": ["number", "null"]},
        "total_hours": {"type": ["number", "null"]},
        "per_hour": {"type": ["number", "null"]},
        "per_parcel": {"type": ["number", "null"]},
        "tax_year": {"type": ["string", "null"]},
        "error": {"type": "string"},
    },
}

RECONCILE_WORK_DAY_INPUT = {
    "type": "object",
    "required": ["prev_date"],
    "properties": {
        "prev_date": {"type": "string", "description": "YYYY-MM-DD of the day to close"},
        "end_odometer": {"type": ["number", "null"], "description": "Usually today's start odometer"},
        "parcels": {"type": ["integer", "null"]},
        "end_time": _HHMM,
    },
}
RECONCILE_WORK_DAY_OUTPUT = FINISH_WORK_DAY_OUTPUT

SET_PERSONAL_MILEAGE_INPUT = {
    "type": "object",
    "required": ["personal_miles"],
    "properties": {
        "work_date": _DATE,
        "personal_miles": {"type": "number", "description": "Miles that were personal, not work"},
    },
}
SET_PERSONAL_MILEAGE_OUTPUT = {
    "type": "object",
    "required": ["ok"],
    "properties": {
        "ok": {"type": "boolean"},
        "work_date": {"type": ["string", "null"]},
        "total_distance": {"type": ["number", "null"]},
        "work_miles": {"type": ["number", "null"]},
        "personal_miles": {"type": ["number", "null"]},
        "error": {"type": "string"},
    },
}

GET_WORK_DAY_INPUT = {
    "type": "object",
    "properties": {"work_date": _DATE},
}
GET_WORK_DAY_OUTPUT = {
    "type": "object",
    "required": ["found"],
    "properties": {
        "found": {"type": "boolean"},
        "day": {"type": ["object", "null"]},
        "error": {"type": "string"},
    },
}
