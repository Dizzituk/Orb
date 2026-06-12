# FILE: app/vehicle/models.py
# Purpose: SQLAlchemy tables for the vehicle module — OBD trips, maintenance/wear
#          items, DTC events, and the single-row virtual-odometer state.
# Called-by: app.db (init_db), main (noqa registration), app.vehicle.service, app.vehicle.status, app.vehicle.router
# Depends-on: app.db (Base)
# Last-renovated: 2026-06-12
"""
Vehicle module tables (2021 Citroen Dispatch LWB, OBDLink MX+ dongle).

Four tables:
  vehicle_trips             one row per ignition-on→off trip recorded by the phone
  vehicle_maintenance_items date- and mileage-based maintenance/wear items
  vehicle_dtc_events        diagnostic trouble codes seen at trip start
  vehicle_state             single row: odometer baseline + accumulated miles

The virtual odometer = odometer_baseline_miles + miles_accumulated_since_baseline.
The dash odometer is NOT readable over standard PIDs on this van; Taz calibrates
the baseline by voice occasionally and trips accumulate distance between
calibrations. WorkDay (finance_work_days) stays the canonical daily ledger —
trips are the fine-grained record that feeds it.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    Text,
)

from app.db import Base


def _utcnow() -> datetime:
    return datetime.utcnow()


class VehicleTrip(Base):
    """One ignition-on→off trip, recorded on the phone and batch-uploaded.

    trip_uuid is generated on the phone and is the dedup key — re-posting a
    batch never double-ingests. classification drives WorkDay folding:
    work opens/extends the day, personal feeds the personal-miles split,
    unclassified waits for Taz's answer.
    """

    __tablename__ = "vehicle_trips"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trip_uuid = Column(String(64), nullable=False, unique=True, index=True)
    started_at = Column(DateTime, nullable=False)            # UTC
    ended_at = Column(DateTime, nullable=True)               # UTC
    distance_miles = Column(Float, nullable=False, default=0.0)
    classification = Column(String(12), nullable=False, default="unclassified",
                            index=True)  # work | personal | unclassified
    source = Column(String(10), nullable=False, default="obd")  # obd | manual
    avg_speed_mph = Column(Float, nullable=True)
    max_speed_mph = Column(Float, nullable=True)
    start_voltage = Column(Float, nullable=True)
    end_voltage = Column(Float, nullable=True)
    # Local date the trip belongs to (phone-stamped; UK-local fallback server-side).
    work_date = Column(Date, nullable=False, index=True)
    folded_into_day = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow)


class VehicleMaintenanceItem(Base):
    """A maintenance obligation — date-based (MOT, insurance, tax, service)
    and/or mileage-based wear (tyres, pads, discs: fitted_at + interval).

    Wear kinds use fitted_at_odometer (virtual odo when fitted) + interval_miles
    (threshold). Date kinds use due_date + lead_days. Service may use both.
    Soft-delete via is_active=False — nothing is ever deleted.
    """

    __tablename__ = "vehicle_maintenance_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    kind = Column(String(20), nullable=False, index=True)
    # mot | insurance | tax | service | tyres_front | tyres_rear |
    # brake_pads | brake_discs | custom
    label = Column(String(120), nullable=False, default="")
    due_date = Column(Date, nullable=True)
    lead_days = Column(Integer, nullable=False, default=30)
    fitted_at_odometer = Column(Float, nullable=True)   # wear kinds
    interval_miles = Column(Float, nullable=True)       # wear threshold
    notes = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    updated_at = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)


class VehicleDtcEvent(Base):
    """A diagnostic trouble code observation (mode 03/07 read at trip start).

    Upserted by code: a code seen again while uncleared bumps last_seen and
    occurrence_count; a code returning AFTER being cleared opens a new event.
    cleared marks acknowledged-in-DB only — v1 never clears codes on the van.
    """

    __tablename__ = "vehicle_dtc_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(8), nullable=False, index=True)     # e.g. P0299
    description = Column(String(300), nullable=True)
    severity = Column(String(10), nullable=False, default="warn")  # info | warn | critical
    first_seen = Column(DateTime, nullable=False, default=_utcnow)
    last_seen = Column(DateTime, nullable=False, default=_utcnow)
    occurrence_count = Column(Integer, nullable=False, default=1)
    cleared = Column(Boolean, nullable=False, default=False)
    trip_uuid = Column(String(64), nullable=True)


class VehicleState(Base):
    """Single-row state: the calibrated virtual odometer.

    virtual odometer = odometer_baseline_miles + miles_accumulated_since_baseline.
    Calibration (a dash reading by voice) sets the baseline and zeroes the
    accumulator; every ingested trip adds its distance to the accumulator.
    """

    __tablename__ = "vehicle_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    odometer_baseline_miles = Column(Float, nullable=True)
    odometer_baseline_at = Column(DateTime, nullable=True)
    miles_accumulated_since_baseline = Column(Float, nullable=False, default=0.0)
