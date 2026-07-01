# FILE: app/watchers/models.py
# Purpose: SQLAlchemy model for watcher observation ledgers (one row per key per day).
# Called-by: app.watchers.framework, app.db (init_db model import for create_all)
# Depends-on: app.db.Base
# Last-renovated: 2026-07-01
"""
One shared readings table; each watcher's ledger is its watcher_id slice.
(The spec's "ledger table per watcher" is realised as a keyed slice rather
than dynamic DDL per instance — same append-only semantics, standard ORM.)
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text

from app.db import Base


class WatcherReading(Base):
    __tablename__ = "watcher_readings"

    id = Column(Integer, primary_key=True, index=True)
    watcher_id = Column(String(64), nullable=False, index=True)
    # Observation day (YYYY-MM-DD). One row per (watcher_id, key, observed_date).
    observed_date = Column(String(10), nullable=False, index=True)
    # What was measured: a region ("central"), an item class ("rtx_5090_new").
    key = Column(String(128), nullable=False, index=True)
    value = Column(Float, nullable=True)
    unit = Column(String(32), nullable=True)  # e.g. "eur_per_m2", "gbp"
    source = Column(String(512), nullable=True)
    note = Column(Text, nullable=True)
    observed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
