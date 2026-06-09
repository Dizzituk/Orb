# FILE: app/finance/services/work_day_service.py
"""
Work-day ledger service (v8.0 rebuild).

Drives the open-shift workflow:
  - start_day()              open today's row: start time + start odometer
  - finish_day()             close it: end time, end odometer, parcels (+ fuel)
  - set_mileage_split()      record personal miles, recompute work miles
  - check_unclosed()         find a previous day that was never closed
  - reconcile_previous_day() close a forgotten day using today's start odo

All derived fields (distance, work miles, earnings, per-hour, per-parcel,
hours) are recomputed from raw inputs — callers never set them directly.
One row per date (WorkDay.work_date is unique). Tax year is stamped from
the canonical finance/utils/tax_year helper.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from sqlalchemy.orm import Session

from app.finance.models_workday import WorkDay
from app.finance.utils.tax_year import tax_year_for

DEFAULT_RATE = 2.35


# ─── helpers ─────────────────────────────────────────────

def _parse_hhmm(value: Optional[str]) -> Optional[int]:
    """'HH:MM' -> minutes since midnight, or None if unparseable."""
    if not value:
        return None
    try:
        h, m = value.strip().split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return None


def _hours_between(start: Optional[str], end: Optional[str], breaks: float = 0.0):
    """Return (total_hours, net_hours). (None, None) if not computable or end<start."""
    s, e = _parse_hhmm(start), _parse_hhmm(end)
    if s is None or e is None or e < s:
        return None, None
    total = round((e - s) / 60.0, 2)
    net = round(max(total - (breaks or 0.0), 0.0), 2)
    return total, net


def _recompute(day: WorkDay) -> None:
    """Recompute every derived field from the raw inputs on the row."""
    # Mileage
    if (day.start_odometer is not None and day.end_odometer is not None
            and day.end_odometer >= day.start_odometer):
        day.total_distance = round(day.end_odometer - day.start_odometer, 1)
        personal = day.personal_miles or 0.0
        day.work_miles = round(max(day.total_distance - personal, 0.0), 1)
    # Hours
    total, net = _hours_between(day.start_time, day.end_time, day.break_hours or 0.0)
    day.total_hours, day.net_hours = total, net
    # Pay
    rate = day.rate_per_parcel or DEFAULT_RATE
    parcels = day.parcels or 0
    day.gross_earnings = round(parcels * rate, 2)
    day.per_parcel = round(day.gross_earnings / parcels, 2) if parcels else 0.0
    day.per_hour = round(day.gross_earnings / total, 2) if total else 0.0


def get_day(db: Session, work_date: date) -> Optional[WorkDay]:
    return db.query(WorkDay).filter(WorkDay.work_date == work_date).one_or_none()


# ─── start of day ────────────────────────────────────────

def start_day(db: Session, *, work_date: Optional[date] = None,
              start_time: Optional[str] = None,
              start_odometer: Optional[float] = None,
              route_area: Optional[str] = None,
              tour_id: Optional[str] = None) -> WorkDay:
    """Open (or update) the row for `work_date`. Idempotent per date."""
    work_date = work_date or date.today()
    day = get_day(db, work_date)
    if day is None:
        day = WorkDay(work_date=work_date, status="open",
                      tax_year=tax_year_for(work_date),
                      rate_per_parcel=DEFAULT_RATE)
        db.add(day)
    if start_time is not None:
        day.start_time = start_time
    if start_odometer is not None:
        day.start_odometer = float(start_odometer)
    if route_area is not None:
        day.route_area = route_area
    if tour_id is not None:
        day.tour_id = tour_id
    if day.status != "complete":
        day.status = "open"
    _recompute(day)
    db.commit(); db.refresh(day)
    return day


# ─── end of day ──────────────────────────────────────────

def finish_day(db: Session, *, work_date: Optional[date] = None,
               start_time: Optional[str] = None,
               start_odometer: Optional[float] = None,
               end_time: Optional[str] = None,
               end_odometer: Optional[float] = None,
               parcels: Optional[int] = None,
               collections: Optional[int] = None,
               rate_per_parcel: Optional[float] = None,
               fuel_cost: Optional[float] = None,
               personal_miles: Optional[float] = None,
               route_area: Optional[str] = None,
               tour_id: Optional[str] = None,
               screenshot_path: Optional[str] = None) -> WorkDay:
    """Complete the row for `work_date`. Creates it first if it never opened."""
    work_date = work_date or date.today()
    day = get_day(db, work_date)
    if day is None:
        day = WorkDay(work_date=work_date, tax_year=tax_year_for(work_date),
                      rate_per_parcel=DEFAULT_RATE)
        db.add(day)
    if start_time is not None:
        day.start_time = start_time
    if start_odometer is not None:
        day.start_odometer = float(start_odometer)
    if end_time is not None:
        day.end_time = end_time
    if end_odometer is not None:
        day.end_odometer = float(end_odometer)
    if parcels is not None:
        day.parcels = int(parcels)
    if collections is not None:
        day.collections = int(collections)
    if rate_per_parcel is not None:
        day.rate_per_parcel = float(rate_per_parcel)
    if fuel_cost is not None:
        day.fuel_cost = float(fuel_cost)
    if personal_miles is not None:
        day.personal_miles = float(personal_miles)
    if route_area is not None:
        day.route_area = route_area
    if tour_id is not None:
        day.tour_id = tour_id
    if screenshot_path is not None:
        day.screenshot_path = screenshot_path
        day.ocr_processed = True
    day.status = "complete"
    _recompute(day)
    db.commit(); db.refresh(day)
    return day


def set_mileage_split(db: Session, *, work_date: date,
                      personal_miles: float) -> WorkDay:
    """Record personal miles for a day; recompute the work/personal split."""
    day = get_day(db, work_date)
    if day is None:
        raise ValueError(f"No work day for {work_date}")
    day.personal_miles = float(personal_miles)
    _recompute(day)
    db.commit(); db.refresh(day)
    return day


# ─── missing-day reconciliation ──────────────────────────

def check_unclosed(db: Session, *, before: Optional[date] = None) -> Optional[WorkDay]:
    """Most recent day left open (no end odometer) before `before`.
    Catches a shift the user forgot to close out the night before."""
    before = before or date.today()
    return (db.query(WorkDay)
              .filter(WorkDay.work_date < before)
              .filter((WorkDay.status == "open") | (WorkDay.end_odometer.is_(None)))
              .order_by(WorkDay.work_date.desc())
              .first())


def reconcile_previous_day(db: Session, *, prev_date: date,
                           end_odometer: Optional[float] = None,
                           parcels: Optional[int] = None,
                           end_time: Optional[str] = None) -> WorkDay:
    """Close out a forgotten previous day. `end_odometer` is typically today's
    start odometer (van didn't move overnight). Caller must confirm with the
    user before calling this."""
    day = get_day(db, prev_date)
    if day is None:
        raise ValueError(f"No work day for {prev_date}")
    if end_odometer is not None:
        day.end_odometer = float(end_odometer)
    if parcels is not None:
        day.parcels = int(parcels)
    if end_time is not None:
        day.end_time = end_time
    day.status = "complete"
    _recompute(day)
    db.commit(); db.refresh(day)
    return day
