# FILE: app/vehicle/router.py
# Purpose: Vehicle read/manage API for the desktop (— /vehicle/*, session-auth'd
#          like the finance work router; writes also reachable by voice tools).
# Called-by: main (include_router)
# Depends-on: app.auth, app.db, app.vehicle.service, app.vehicle.maintenance, app.vehicle.status
# Last-renovated: 2026-06-12
"""
/vehicle API — mirrors app/finance/work_router.py's shape: thin handlers over
the unit-tested service/status modules, Depends(require_auth) on the router.

  GET    /vehicle/summary                      the one payload the views render
  GET    /vehicle/trips?tax_year=&limit=       trip list (newest first)
  POST   /vehicle/trips/{uuid}/classify        work|personal|unclassified
  GET    /vehicle/maintenance                  full status (dates + wear)
  POST   /vehicle/maintenance                  create/update an item by kind
  PATCH  /vehicle/maintenance/{id}             update fields on an item
  DELETE /vehicle/maintenance/{id}             soft-delete (is_active=False)
  POST   /vehicle/odometer/calibrate           dash reading → new baseline
  POST   /vehicle/dtc/{id}/clear               mark cleared in DB only
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import require_auth
from app.db import get_db
from app.vehicle import maintenance, service, status

router = APIRouter(
    prefix="/vehicle",
    tags=["Vehicle"],
    dependencies=[Depends(require_auth)],
)


class ClassifyIn(BaseModel):
    classification: str


class CalibrateIn(BaseModel):
    reading: float


class MaintenancePatchIn(BaseModel):
    label: Optional[str] = None
    due_date: Optional[str] = None          # YYYY-MM-DD
    lead_days: Optional[int] = None
    interval_miles: Optional[float] = None
    fitted_at_odometer: Optional[float] = None
    notes: Optional[str] = None


class MaintenanceIn(MaintenancePatchIn):
    kind: str


@router.get("/summary")
async def get_summary(db: Session = Depends(get_db)):
    return status.summary(db)


@router.get("/trips")
async def get_trips(tax_year: Optional[str] = Query(None),
                    limit: int = Query(100, ge=1, le=1000),
                    db: Session = Depends(get_db)):
    return {"tax_year": tax_year, "trips": status.trips_for(db, tax_year, limit)}


@router.post("/trips/{trip_uuid}/classify")
async def classify(trip_uuid: str, body: ClassifyIn,
                   db: Session = Depends(get_db)):
    result = service.classify_trip(db, trip_uuid, body.classification)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result


@router.get("/maintenance")
async def get_maintenance(db: Session = Depends(get_db)):
    return {"items": status.maintenance_status(db)}


@router.post("/maintenance")
async def post_maintenance(body: MaintenanceIn, db: Session = Depends(get_db)):
    result = maintenance.set_maintenance(
        db, body.kind, label=body.label, due_date=body.due_date,
        lead_days=body.lead_days, interval_miles=body.interval_miles,
        fitted_at_odometer=body.fitted_at_odometer, notes=body.notes)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result


@router.patch("/maintenance/{item_id}")
async def patch_maintenance(item_id: int, body: MaintenancePatchIn,
                            db: Session = Depends(get_db)):
    result = maintenance.update_maintenance(
        db, item_id, label=body.label, due_date=body.due_date,
        lead_days=body.lead_days, interval_miles=body.interval_miles,
        fitted_at_odometer=body.fitted_at_odometer, notes=body.notes)
    if not result.get("ok"):
        status_code = 404 if "No maintenance item" in str(result.get("error")) else 400
        raise HTTPException(status_code=status_code, detail=result.get("error"))
    return result


@router.delete("/maintenance/{item_id}")
async def delete_maintenance(item_id: int, db: Session = Depends(get_db)):
    result = maintenance.deactivate_maintenance(db, item_id)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error"))
    return result


@router.post("/odometer/calibrate")
async def calibrate(body: CalibrateIn, db: Session = Depends(get_db)):
    result = service.calibrate_odometer(db, body.reading)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result


@router.post("/dtc/{event_id}/clear")
async def clear_dtc(event_id: int, db: Session = Depends(get_db)):
    result = service.clear_dtc(db, event_id)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error"))
    return result
