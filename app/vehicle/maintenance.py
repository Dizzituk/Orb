# FILE: app/vehicle/maintenance.py
# Purpose: Maintenance-item write-side — upsert by kind, update by id, wear-
#          component fittings, soft-delete. Split from service.py (size).
# Called-by: app.vehicle.router, app.tools.vehicle_tools
# Depends-on: app.vehicle.models, app.vehicle.service (virtual odometer, parsing)
# Last-renovated: 2026-06-12
"""
Maintenance writes for the vehicle module. Read-side (wear %, due warnings)
lives in status.py; trip/odometer writes live in service.py. Defaults for
wear intervals and labels come from service.py so there is one source.
"""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.orm import Session

from app.vehicle.models import VehicleMaintenanceItem
from app.vehicle.service import (
    DEFAULT_INTERVAL_MILES,
    DEFAULT_LABELS,
    DEFAULT_LEAD_DAYS,
    MAINTENANCE_KINDS,
    WEAR_KINDS,
    _parse_date,
    virtual_odometer,
)

logger = logging.getLogger(__name__)


def _serialize(item: VehicleMaintenanceItem) -> dict:
    return {"ok": True, "id": item.id, "kind": item.kind, "label": item.label,
            "due_date": item.due_date.isoformat() if item.due_date else None,
            "lead_days": item.lead_days, "interval_miles": item.interval_miles,
            "fitted_at_odometer": item.fitted_at_odometer}


def _apply(item: VehicleMaintenanceItem, *, label: Optional[str],
           due_date, lead_days: Optional[int], interval_miles: Optional[float],
           fitted_at_odometer: Optional[float], notes: Optional[str]) -> Optional[str]:
    """Apply optional fields to an item; returns an error string or None."""
    if label:
        item.label = label
    parsed_due = _parse_date(due_date)
    if due_date is not None and parsed_due is None:
        return f"due_date unparseable: {due_date!r}"
    if parsed_due is not None:
        item.due_date = parsed_due
    if lead_days is not None:
        item.lead_days = int(lead_days)
    if interval_miles is not None:
        item.interval_miles = float(interval_miles)
    if fitted_at_odometer is not None:
        item.fitted_at_odometer = float(fitted_at_odometer)
    if notes is not None:
        item.notes = notes
    return None


def set_maintenance(db: Session, kind: str, *, label: Optional[str] = None,
                    due_date=None, lead_days: Optional[int] = None,
                    interval_miles: Optional[float] = None,
                    fitted_at_odometer: Optional[float] = None,
                    notes: Optional[str] = None) -> dict:
    """Create/update the active maintenance item of a kind (custom kinds are
    keyed by label, so several custom items can coexist)."""
    kind = str(kind or "").strip().lower()
    if kind not in MAINTENANCE_KINDS:
        return {"ok": False, "error": f"kind must be one of {MAINTENANCE_KINDS}"}
    query = db.query(VehicleMaintenanceItem).filter(
        VehicleMaintenanceItem.kind == kind,
        VehicleMaintenanceItem.is_active == True)  # noqa: E712
    if kind == "custom" and label:
        query = query.filter(VehicleMaintenanceItem.label == label)
    item = query.first()
    if item is None:
        item = VehicleMaintenanceItem(
            kind=kind,
            label=label or DEFAULT_LABELS.get(kind, kind),
            lead_days=DEFAULT_LEAD_DAYS,
            interval_miles=DEFAULT_INTERVAL_MILES.get(kind),
        )
        db.add(item)
    error = _apply(item, label=label, due_date=due_date, lead_days=lead_days,
                   interval_miles=interval_miles,
                   fitted_at_odometer=fitted_at_odometer, notes=notes)
    if error:
        db.rollback()
        return {"ok": False, "error": error}
    db.commit()
    return _serialize(item)


def update_maintenance(db: Session, item_id: int, *, label: Optional[str] = None,
                       due_date=None, lead_days: Optional[int] = None,
                       interval_miles: Optional[float] = None,
                       fitted_at_odometer: Optional[float] = None,
                       notes: Optional[str] = None) -> dict:
    """Update one maintenance item by id (desktop PATCH path)."""
    item = (db.query(VehicleMaintenanceItem)
              .filter(VehicleMaintenanceItem.id == item_id).one_or_none())
    if item is None:
        return {"ok": False, "error": f"No maintenance item {item_id}"}
    error = _apply(item, label=label, due_date=due_date, lead_days=lead_days,
                   interval_miles=interval_miles,
                   fitted_at_odometer=fitted_at_odometer, notes=notes)
    if error:
        db.rollback()
        return {"ok": False, "error": error}
    db.commit()
    return _serialize(item)


def log_component_fitted(db: Session, kind: str,
                         interval_miles: Optional[float] = None,
                         at_odometer: Optional[float] = None) -> dict:
    """'Just had new front tyres' — stamp fitted_at_odometer with the current
    virtual odometer (or an explicit reading) for a wear kind."""
    kind = str(kind or "").strip().lower()
    if kind not in WEAR_KINDS:
        return {"ok": False, "error": f"kind must be one of {WEAR_KINDS}"}
    odo = at_odometer if at_odometer is not None else virtual_odometer(db)
    if odo is None:
        return {"ok": False, "error": (
            "No virtual odometer yet — calibrate first "
            "(say the dash reading), then log the fitting.")}
    return set_maintenance(db, kind, fitted_at_odometer=float(odo),
                           interval_miles=interval_miles)


def deactivate_maintenance(db: Session, item_id: int) -> dict:
    """Soft-delete (is_active=False). Nothing is ever deleted."""
    item = (db.query(VehicleMaintenanceItem)
              .filter(VehicleMaintenanceItem.id == item_id).one_or_none())
    if item is None:
        return {"ok": False, "error": f"No maintenance item {item_id}"}
    item.is_active = False
    db.commit()
    return {"ok": True, "id": item.id, "is_active": False}
