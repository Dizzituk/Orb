# FILE: app/lifestyle/product_models.py
# Purpose: Personal food product library — Job 4 of the memory roadmap (2026-06-10).
# Called-by: app.lifestyle.product_router, app.lifestyle.product_service
# Depends-on: app.db
# Last-renovated: 2026-06-11
"""
Personal food product library — Job 4 of the memory roadmap (2026-06-10).

The missing memory layer for nutrition capture: previously every photo or
description was estimated from scratch, statelessly. This table gives ASTRA
a permanent catalogue of the actual products Taz buys (Aldi staples etc.) —
recognised once, remembered for life. Populated from barcode scans, Open
Food Facts lookups, photo recognition, or manual entry; reused instantly on
repeat purchases via product_service.resolve_and_save().
"""
from datetime import datetime, timezone

from sqlalchemy import Column, Integer, Float, String, Text, DateTime

from app.db import Base, engine


def _now() -> datetime:
    return datetime.now(timezone.utc)


class FoodProduct(Base):
    """One known product with per-100g nutrition. Deduped by barcode, then
    by normalised name+brand."""

    __tablename__ = "lifestyle_food_products"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, index=True)        # normalised lookup key
    display_name = Column(String, nullable=False)            # as shown to the user
    brand = Column(String, nullable=True)
    barcode = Column(String, nullable=True, index=True)

    # Per-100g values
    calories = Column(Float, nullable=True)
    protein_g = Column(Float, nullable=True)
    carbs_g = Column(Float, nullable=True)
    fat_g = Column(Float, nullable=True)
    fibre_g = Column(Float, nullable=True)
    sugar_g = Column(Float, nullable=True)

    serving_size_g = Column(Float, nullable=True)
    serving_description = Column(String, nullable=True)      # e.g. "1 pot (150g)"

    source = Column(String, default="manual")  # off_barcode | off_search | photo | manual | chat
    times_logged = Column(Integer, default=0)
    last_logged_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_now)
    updated_at = Column(DateTime, nullable=False, default=_now)


def ensure_product_table() -> None:
    """Idempotent create — db.py's global create_all runs before this module
    is imported, so the table must be created explicitly on first use."""
    Base.metadata.create_all(bind=engine, tables=[FoodProduct.__table__])
