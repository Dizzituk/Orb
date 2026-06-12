# FILE: app/bridge/vehicle.py
# Purpose: Bridge vehicle endpoints — phone trip-batch ingest, DTC reports, and
#          the mobile vehicle summary. Thin sub-router over app.vehicle.*.
# Called-by: main (include_router, Bridge API block)
# Depends-on: app.db, app.bridge.schemas (require_bridge_auth), app.vehicle.service, app.vehicle.status
# Last-renovated: 2026-06-12
"""
Bridge vehicle sub-router — mirrors dashboards.py: prefix /bridge,
require_bridge_auth, calls the existing service, mobile-shaped JSON, no
business logic. Kept out of bridge/router.py (near the size ceiling).

  GET  /bridge/vehicle/summary   summary incl. day_open (the phone prompter
                                 uses it to decide whether to start the day)
  POST /bridge/vehicle/trips     batch ingest; dedup by trip_uuid makes
                                 redelivery safe (offline-first phone)
  POST /bridge/vehicle/dtc       codes read at trip start; returns which are
                                 NEW so the phone can notify
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from .schemas import require_bridge_auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bridge", tags=["Bridge"])


# ── request/response shapes (kept here: one job per file) ─────────────────────

class VehicleTripIn(BaseModel):
    trip_uuid: str
    started_at: str                      # ISO 8601 (UTC)
    ended_at: Optional[str] = None
    distance_miles: float
    classification: str = "unclassified"  # work | personal | unclassified
    source: str = "obd"
    avg_speed_mph: Optional[float] = None
    max_speed_mph: Optional[float] = None
    start_voltage: Optional[float] = None
    end_voltage: Optional[float] = None
    work_date: Optional[str] = None      # YYYY-MM-DD local; server falls back


class VehicleTripBatchIn(BaseModel):
    trips: list[VehicleTripIn] = []


class VehicleTripBatchOut(BaseModel):
    ok: bool = True
    ingested: int = 0
    duplicates: int = 0
    # Redelivered trips whose server record was unclassified and arrived with
    # a real classification — the phone's post-upload answer path.
    reclassified: int = 0
    days_touched: list[str] = []
    errors: list[str] = []


class VehicleDtcIn(BaseModel):
    code: str
    description: Optional[str] = None
    severity: Optional[str] = None


class VehicleDtcReportIn(BaseModel):
    codes: list[VehicleDtcIn] = []
    trip_uuid: Optional[str] = None


class VehicleDtcReportOut(BaseModel):
    ok: bool = True
    new: list[str] = []
    known: list[str] = []


# ── routes ────────────────────────────────────────────────────────────────────

@router.get("/vehicle/summary")
async def bridge_vehicle_summary(
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_bridge_auth),
) -> dict:
    """Vehicle summary for the phone screen + trip prompter (day_open)."""
    from app.vehicle import status
    return status.summary(db)


@router.post("/vehicle/trips", response_model=VehicleTripBatchOut)
async def bridge_vehicle_trips(
    req: VehicleTripBatchIn,
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_bridge_auth),
) -> VehicleTripBatchOut:
    """Ingest a batch of phone-recorded trips. Re-posting the same batch is
    safe — trip_uuid dedup means redelivery never double-counts."""
    from app.vehicle import service
    result = service.ingest_trips(db, [t.model_dump() for t in req.trips])
    return VehicleTripBatchOut(ok=True, **result)


@router.post("/vehicle/dtc", response_model=VehicleDtcReportOut)
async def bridge_vehicle_dtc(
    req: VehicleDtcReportIn,
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_bridge_auth),
) -> VehicleDtcReportOut:
    """Record DTCs read at trip start; the phone notifies on anything new."""
    from app.vehicle import service
    result = service.record_dtcs(
        db, [c.model_dump() for c in req.codes], trip_uuid=req.trip_uuid)
    return VehicleDtcReportOut(ok=True, new=result["new"], known=result["known"])
