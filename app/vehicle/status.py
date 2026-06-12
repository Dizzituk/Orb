# FILE: app/vehicle/status.py
# Purpose: Vehicle read-side — wear/maintenance status, trip queries, and the
#          summary dict that powers both the desktop Vehicle view and the phone.
# Called-by: app.vehicle.router, app.bridge.vehicle, app.tools.vehicle_tools, app.vehicle.briefing_section
# Depends-on: app.vehicle.models, app.vehicle.service, app.finance.utils.tax_year
# Last-renovated: 2026-06-12
"""
Read-side aggregation for the vehicle module. Pure functions over a Session;
no mutations here (write-side lives in service.py). Wear figures are honest
mileage accumulators ("pads at 26,000 of ~28,000 miles"), never claims of
actual physical condition.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.vehicle.models import (
    VehicleDtcEvent,
    VehicleMaintenanceItem,
    VehicleTrip,
)
from app.vehicle.service import WEAR_KINDS, get_state, virtual_odometer

# A wear item starts warning at this fraction of its interval.
WEAR_WARN_PCT = 0.90
RECENT_TRIP_COUNT = 14


# ─── serialization ────────────────────────────────────────

def serialize_trip(t: VehicleTrip) -> dict:
    return {
        "trip_uuid": t.trip_uuid,
        "started_at": t.started_at.isoformat() + "Z" if t.started_at else None,
        "ended_at": t.ended_at.isoformat() + "Z" if t.ended_at else None,
        "distance_miles": t.distance_miles,
        "classification": t.classification,
        "source": t.source,
        "avg_speed_mph": t.avg_speed_mph,
        "max_speed_mph": t.max_speed_mph,
        "start_voltage": t.start_voltage,
        "end_voltage": t.end_voltage,
        "work_date": t.work_date.isoformat() if t.work_date else None,
        "folded_into_day": t.folded_into_day,
    }


def _serialize_dtc(e: VehicleDtcEvent) -> dict:
    return {
        "id": e.id, "code": e.code, "description": e.description,
        "severity": e.severity,
        "first_seen": e.first_seen.isoformat() + "Z" if e.first_seen else None,
        "last_seen": e.last_seen.isoformat() + "Z" if e.last_seen else None,
        "occurrence_count": e.occurrence_count, "cleared": e.cleared,
        "trip_uuid": e.trip_uuid,
    }


# ─── wear / maintenance ───────────────────────────────────

def wear_status(db: Session, virtual_odo: Optional[float] = None) -> list[dict]:
    """Mileage accumulators for wear kinds with a fitted-at reading."""
    if virtual_odo is None:
        virtual_odo = virtual_odometer(db)
    items = (db.query(VehicleMaintenanceItem)
               .filter(VehicleMaintenanceItem.is_active == True,  # noqa: E712
                       VehicleMaintenanceItem.kind.in_(WEAR_KINDS))
               .all())
    out = []
    for item in items:
        miles_done = pct = None
        warning = False
        if (item.fitted_at_odometer is not None and virtual_odo is not None):
            miles_done = round(max(virtual_odo - item.fitted_at_odometer, 0.0), 0)
            if item.interval_miles:
                pct = round(miles_done / item.interval_miles, 3)
                warning = pct >= WEAR_WARN_PCT
        out.append({
            "id": item.id, "kind": item.kind, "label": item.label,
            "fitted_at_odometer": item.fitted_at_odometer,
            "interval_miles": item.interval_miles,
            "miles_done": miles_done, "pct": pct, "warning": warning,
            "notes": item.notes,
        })
    return out


def maintenance_status(db: Session, today: Optional[date] = None) -> list[dict]:
    """Date-based items (days_remaining / warning within lead_days, overdue)
    merged with the wear accumulators."""
    today = today or date.today()
    virtual_odo = virtual_odometer(db)
    out = []
    date_items = (db.query(VehicleMaintenanceItem)
                    .filter(VehicleMaintenanceItem.is_active == True,  # noqa: E712
                            VehicleMaintenanceItem.kind.notin_(WEAR_KINDS))
                    .all())
    for item in date_items:
        days_remaining = None
        warning = overdue = False
        if item.due_date is not None:
            days_remaining = (item.due_date - today).days
            overdue = days_remaining < 0
            warning = days_remaining <= (item.lead_days or 0)
        out.append({
            "id": item.id, "kind": item.kind, "label": item.label,
            "due_date": item.due_date.isoformat() if item.due_date else None,
            "lead_days": item.lead_days, "days_remaining": days_remaining,
            "warning": warning, "overdue": overdue, "notes": item.notes,
            "interval_miles": item.interval_miles,
            "fitted_at_odometer": item.fitted_at_odometer,
        })
    out.extend(wear_status(db, virtual_odo))
    return out


# ─── trips / DTCs ─────────────────────────────────────────

def trips_for(db: Session, tax_year: Optional[str] = None,
              limit: int = 100) -> list[dict]:
    """Trips newest-first, optionally bounded to a UK tax year."""
    q = db.query(VehicleTrip)
    if tax_year:
        from app.finance.utils.tax_year import tax_year_bounds
        start, end = tax_year_bounds(tax_year)
        q = q.filter(VehicleTrip.work_date >= start, VehicleTrip.work_date <= end)
    rows = q.order_by(VehicleTrip.started_at.desc()).limit(max(int(limit), 1)).all()
    return [serialize_trip(t) for t in rows]


def active_dtcs(db: Session) -> list[dict]:
    rows = (db.query(VehicleDtcEvent)
              .filter(VehicleDtcEvent.cleared == False)  # noqa: E712
              .order_by(VehicleDtcEvent.last_seen.desc())
              .all())
    return [_serialize_dtc(e) for e in rows]


def _tax_year_split(db: Session) -> dict:
    """Work vs personal miles for the current tax year, from the trip record
    (the fine-grained truth; cheap on SQLite at this volume)."""
    from app.finance.utils.tax_year import get_current_tax_year, tax_year_bounds
    ty = get_current_tax_year()
    start, end = tax_year_bounds(ty)
    rows = (db.query(VehicleTrip.classification,
                     func.coalesce(func.sum(VehicleTrip.distance_miles), 0.0),
                     func.count(VehicleTrip.id))
              .filter(VehicleTrip.work_date >= start, VehicleTrip.work_date <= end)
              .group_by(VehicleTrip.classification)
              .all())
    by_class = {r[0]: {"miles": round(float(r[1]), 1), "trips": int(r[2])} for r in rows}
    work = by_class.get("work", {"miles": 0.0, "trips": 0})
    personal = by_class.get("personal", {"miles": 0.0, "trips": 0})
    unclassified = by_class.get("unclassified", {"miles": 0.0, "trips": 0})
    classified_total = work["miles"] + personal["miles"]
    return {
        "tax_year": ty,
        "work_miles": work["miles"], "personal_miles": personal["miles"],
        "unclassified_miles": unclassified["miles"],
        "trip_count": work["trips"] + personal["trips"] + unclassified["trips"],
        "business_use_pct": (round(work["miles"] / classified_total, 4)
                             if classified_total > 0 else None),
    }


# ─── the one summary ──────────────────────────────────────

def summary(db: Session) -> dict:
    """One dict powering the desktop Vehicle view and the phone screen."""
    from app.finance.services import work_day_service as wds
    state = get_state(db)
    virtual_odo = virtual_odometer(db)
    today = date.today()
    today_day = wds.get_day(db, today)
    unclassified_count = (db.query(func.count(VehicleTrip.id))
                            .filter(VehicleTrip.classification == "unclassified")
                            .scalar() or 0)
    recent = (db.query(VehicleTrip)
                .order_by(VehicleTrip.started_at.desc())
                .limit(RECENT_TRIP_COUNT).all())
    return {
        "virtual_odometer": virtual_odo,
        "calibration": {
            "baseline_miles": state.odometer_baseline_miles,
            "baseline_at": (state.odometer_baseline_at.isoformat() + "Z"
                            if state.odometer_baseline_at else None),
            "miles_since_baseline": round(state.miles_accumulated_since_baseline or 0.0, 1),
        },
        "maintenance": maintenance_status(db, today),
        "recent_trips": [serialize_trip(t) for t in recent],
        "unclassified_count": int(unclassified_count),
        "active_dtcs": active_dtcs(db),
        "tax_year_split": _tax_year_split(db),
        "day_open": bool(today_day is not None and today_day.status == "open"),
    }
