# FILE: app/tools/vehicle_tools.py
# Purpose: Vehicle chat tools — odometer calibration, maintenance dates,
#          component fittings, status reads and trip classification by voice.
# Called-by: app.tools.registry (_register_defaults)
# Depends-on: app.db, app.vehicle.service, app.vehicle.maintenance, app.vehicle.status
# Last-renovated: 2026-06-12
"""
Vehicle chat tools (v1, OBD2 van module 2026-06-12).

Five tools that let any ASTRA chat (desktop or phone) drive the vehicle
module by voice:

  calibrate_odometer     "odometer reads 82,410" → new virtual-odo baseline
  set_maintenance_date   "MOT is due 14th March" → due date + lead warning
  log_component_fitted   "just had new front tyres" → wear counter restarts
  get_vehicle_status     summary for voice answers
  classify_trip          reclassify a recorded trip (work/personal)

Handler style mirrors finance_tools: short-lived session, the unit-tested
service owns the maths, errors return {ok: false, error} so the model can
react. Schemas are inline — this file is self-contained by design.
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


# ────────────────────────────────────────────────────────────────────────
# Handlers
# ────────────────────────────────────────────────────────────────────────

async def calibrate_odometer_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Set the virtual-odometer baseline from the dash reading Taz gives."""
    reading = input_data.get("reading")
    if reading is None:
        return {"ok": False, "error": "reading (the dash odometer miles) is required"}
    try:
        from app.vehicle import service
        with _db_session() as db:
            out = service.calibrate_odometer(db, reading)
        return out
    except Exception as exc:
        logger.exception("[vehicle_tools] calibrate_odometer failed")
        return {"ok": False, "error": str(exc)}


async def set_maintenance_date_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Record/update a maintenance item (MOT, insurance, tax, service…)."""
    kind = input_data.get("kind")
    if not kind:
        return {"ok": False, "error": "kind is required"}
    try:
        from app.vehicle import maintenance
        with _db_session() as db:
            out = maintenance.set_maintenance(
                db, kind,
                label=input_data.get("label"),
                due_date=input_data.get("due_date"),
                lead_days=input_data.get("lead_days"),
                interval_miles=input_data.get("interval_miles"),
                notes=input_data.get("notes"),
            )
        return out
    except Exception as exc:
        logger.exception("[vehicle_tools] set_maintenance_date failed")
        return {"ok": False, "error": str(exc)}


async def log_component_fitted_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Restart a wear counter at the current virtual odometer ('new front
    tyres fitted today'). Optional at_odometer overrides the fitting point."""
    kind = input_data.get("kind")
    if not kind:
        return {"ok": False, "error": "kind is required"}
    try:
        from app.vehicle import maintenance
        with _db_session() as db:
            out = maintenance.log_component_fitted(
                db, kind,
                interval_miles=input_data.get("interval_miles"),
                at_odometer=input_data.get("at_odometer"),
            )
        return out
    except Exception as exc:
        logger.exception("[vehicle_tools] log_component_fitted failed")
        return {"ok": False, "error": str(exc)}


async def get_vehicle_status_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Read the vehicle summary for voice answers ('how's the van doing')."""
    try:
        from app.vehicle import status
        with _db_session() as db:
            out = {"ok": True, "summary": status.summary(db)}
        return out
    except Exception as exc:
        logger.exception("[vehicle_tools] get_vehicle_status failed")
        return {"ok": False, "summary": None, "error": str(exc)}


async def classify_trip_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Classify a recorded trip as work or personal (or back to unclassified)."""
    trip_uuid = input_data.get("trip_uuid")
    classification = input_data.get("classification")
    if not trip_uuid or not classification:
        return {"ok": False, "error": "trip_uuid and classification are required"}
    try:
        from app.vehicle import service
        with _db_session() as db:
            out = service.classify_trip(db, trip_uuid, classification)
        return out
    except Exception as exc:
        logger.exception("[vehicle_tools] classify_trip failed")
        return {"ok": False, "error": str(exc)}


# ────────────────────────────────────────────────────────────────────────
# Schemas (inline — self-contained file)
# ────────────────────────────────────────────────────────────────────────

_OK_ERR = {
    "ok": {"type": "boolean"},
    "error": {"type": "string"},
}

CALIBRATE_ODOMETER_INPUT = {
    "type": "object",
    "required": ["reading"],
    "properties": {
        "reading": {"type": "number",
                    "description": "Dash odometer reading in miles, e.g. 82410"},
    },
}
CALIBRATE_ODOMETER_OUTPUT = {
    "type": "object",
    "required": ["ok"],
    "properties": {
        **_OK_ERR,
        "reading": {"type": ["number", "null"]},
        "previous_virtual": {"type": ["number", "null"]},
        "virtual_odometer": {"type": ["number", "null"]},
        "drift_miles": {"type": ["number", "null"]},
        "drift_pct": {"type": ["number", "null"]},
        "drift_note": {"type": "string"},
    },
}

SET_MAINTENANCE_DATE_INPUT = {
    "type": "object",
    "required": ["kind"],
    "properties": {
        "kind": {"type": "string",
                 "enum": ["mot", "insurance", "tax", "service",
                          "tyres_front", "tyres_rear", "brake_pads",
                          "brake_discs", "custom"],
                 "description": "Which obligation this is"},
        "label": {"type": ["string", "null"],
                  "description": "Display label (required for kind=custom)"},
        "due_date": {"type": ["string", "null"],
                     "description": "YYYY-MM-DD the item is due"},
        "lead_days": {"type": ["integer", "null"],
                      "description": "Warn this many days ahead (default 30)"},
        "interval_miles": {"type": ["number", "null"],
                           "description": "Wear threshold in miles (wear kinds)"},
        "notes": {"type": ["string", "null"]},
    },
}
SET_MAINTENANCE_DATE_OUTPUT = {
    "type": "object",
    "required": ["ok"],
    "properties": {
        **_OK_ERR,
        "id": {"type": ["integer", "null"]},
        "kind": {"type": ["string", "null"]},
        "label": {"type": ["string", "null"]},
        "due_date": {"type": ["string", "null"]},
        "lead_days": {"type": ["integer", "null"]},
        "interval_miles": {"type": ["number", "null"]},
        "fitted_at_odometer": {"type": ["number", "null"]},
    },
}

LOG_COMPONENT_FITTED_INPUT = {
    "type": "object",
    "required": ["kind"],
    "properties": {
        "kind": {"type": "string",
                 "enum": ["tyres_front", "tyres_rear", "brake_pads", "brake_discs"],
                 "description": "Which component was just fitted/renewed"},
        "interval_miles": {"type": ["number", "null"],
                           "description": "Override the wear threshold in miles"},
        "at_odometer": {"type": ["number", "null"],
                        "description": "Odometer at fitting if not right now"},
    },
}
LOG_COMPONENT_FITTED_OUTPUT = SET_MAINTENANCE_DATE_OUTPUT

GET_VEHICLE_STATUS_INPUT = {"type": "object", "properties": {}}
GET_VEHICLE_STATUS_OUTPUT = {
    "type": "object",
    "required": ["ok"],
    "properties": {
        **_OK_ERR,
        "summary": {"type": ["object", "null"],
                    "description": "virtual_odometer, calibration, maintenance, "
                                   "recent_trips, unclassified_count, active_dtcs, "
                                   "tax_year_split, day_open"},
    },
}

CLASSIFY_TRIP_INPUT = {
    "type": "object",
    "required": ["trip_uuid", "classification"],
    "properties": {
        "trip_uuid": {"type": "string"},
        "classification": {"type": "string",
                           "enum": ["work", "personal", "unclassified"]},
    },
}
CLASSIFY_TRIP_OUTPUT = {
    "type": "object",
    "required": ["ok"],
    "properties": {
        **_OK_ERR,
        "trip_uuid": {"type": ["string", "null"]},
        "classification": {"type": ["string", "null"]},
        "work_date": {"type": ["string", "null"]},
        "folded_into_day": {"type": ["boolean", "null"]},
        "day_status": {"type": ["string", "null"]},
    },
}


# ────────────────────────────────────────────────────────────────────────
# Registration
# ────────────────────────────────────────────────────────────────────────

def register_vehicle_tools() -> None:
    """Register the vehicle tools with the global tool registry.
    Called once from registry._register_defaults at module load."""
    from app.tools.registry import ToolDefinition, register_tool

    register_tool(ToolDefinition(
        name="calibrate_odometer",
        version="v1",
        description=(
            "Calibrate the van's virtual odometer from the dash reading. Call "
            "when the user states the odometer figure ('odometer reads 82,410'). "
            "Rejects readings lower than the current virtual odometer; if the "
            "response includes drift_note, mention the drift to the user."
        ),
        input_schema=CALIBRATE_ODOMETER_INPUT,
        output_schema=CALIBRATE_ODOMETER_OUTPUT,
        handler=calibrate_odometer_handler,
    ))

    register_tool(ToolDefinition(
        name="set_maintenance_date",
        version="v1",
        description=(
            "Record or update a van maintenance item. Call when the user gives "
            "a due date ('MOT is due 14th March', 'insurance renews 1st July') "
            "or adjusts a wear threshold. kind is one of mot/insurance/tax/"
            "service/tyres_front/tyres_rear/brake_pads/brake_discs/custom. "
            "Warnings surface in the morning briefing lead_days ahead."
        ),
        input_schema=SET_MAINTENANCE_DATE_INPUT,
        output_schema=SET_MAINTENANCE_DATE_OUTPUT,
        handler=set_maintenance_date_handler,
    ))

    register_tool(ToolDefinition(
        name="log_component_fitted",
        version="v1",
        description=(
            "Restart a wear counter when a component is renewed ('just had new "
            "front tyres', 'pads and discs done today'). Stamps the component's "
            "fitted-at point with the current virtual odometer. Needs the "
            "odometer calibrated first; pass at_odometer to backdate."
        ),
        input_schema=LOG_COMPONENT_FITTED_INPUT,
        output_schema=LOG_COMPONENT_FITTED_OUTPUT,
        handler=log_component_fitted_handler,
    ))

    register_tool(ToolDefinition(
        name="get_vehicle_status",
        version="v1",
        description=(
            "Read the van's status: virtual odometer, upcoming maintenance and "
            "wear (tyres/pads/discs), active fault codes, recent trips, and the "
            "work/personal mile split. Use for 'how's the van', 'when's the "
            "MOT', 'any fault codes', 'how many work miles this year'."
        ),
        input_schema=GET_VEHICLE_STATUS_INPUT,
        output_schema=GET_VEHICLE_STATUS_OUTPUT,
        handler=get_vehicle_status_handler,
    ))

    register_tool(ToolDefinition(
        name="classify_trip",
        version="v1",
        description=(
            "Classify a recorded van trip as work or personal. Use when the "
            "user answers a trip question ('that drive was personal') or "
            "corrects an earlier classification. trip_uuid comes from the "
            "vehicle status/trip list."
        ),
        input_schema=CLASSIFY_TRIP_INPUT,
        output_schema=CLASSIFY_TRIP_OUTPUT,
        handler=classify_trip_handler,
    ))

    logger.info("[vehicle_tools] registered 5 vehicle tools")
