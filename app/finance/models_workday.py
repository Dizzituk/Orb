# FILE: app/finance/models_workday.py
"""
WorkDay — unified daily ledger model (v8.0 rebuild).

One row per working day, holding the whole shift in a single record:
times, odometer, mileage split, parcels and pay. Opened at the start
of the day and completed at the end. Replaces the old split
DailyWorkLog + MileageLog pair.

Kept in its own module to stay small and single-responsibility.
The canonical expense ledger remains `transactions` (models.py).
"""
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Date, Text
from app.db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class WorkDay(Base):
    """A single day's work: shift, odometer, mileage split, parcels, pay."""

    __tablename__ = "finance_work_days"

    id = Column(Integer, primary_key=True, autoincrement=True)
    work_date = Column(Date, nullable=False, unique=True, index=True)
    status = Column(String(10), default="open")  # open | complete

    # Shift / hours
    start_time = Column(String(5), nullable=True)   # "HH:MM"
    end_time = Column(String(5), nullable=True)     # "HH:MM"
    total_hours = Column(Float, nullable=True)
    break_hours = Column(Float, default=0.0)
    net_hours = Column(Float, nullable=True)

    # Odometer / mileage
    start_odometer = Column(Float, nullable=True)
    end_odometer = Column(Float, nullable=True)
    total_distance = Column(Float, nullable=True)   # end - start
    work_miles = Column(Float, nullable=True)
    personal_miles = Column(Float, default=0.0)

    # Parcels / pay
    parcels = Column(Integer, default=0)            # delivery count = pay unit
    collections = Column(Integer, default=0)
    rate_per_parcel = Column(Float, default=2.35)
    gross_earnings = Column(Float, default=0.0)
    per_hour = Column(Float, default=0.0)
    per_parcel = Column(Float, default=0.0)

    # Day-level cost convenience (canonical ledger = transactions)
    fuel_cost = Column(Float, nullable=True)

    # Source / provenance
    route_area = Column(String(200), nullable=True)
    tour_id = Column(String(40), nullable=True)
    screenshot_path = Column(String(500), nullable=True)
    ocr_processed = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)

    # Tax year (UK 6 Apr-5 Apr), set on write
    tax_year = Column(String(7), nullable=False, index=True)

    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)
