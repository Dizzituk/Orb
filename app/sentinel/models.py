# FILE: app/sentinel/models.py
# Purpose: SQLAlchemy tables for Sentinel — learned baseline, connection events,
#          alerts, and key/value state (learn mode, counters).
# Called-by: app.db (init_db), app.sentinel.baseline, app.sentinel.alerts, app.sentinel.collector, app.sentinel.router
# Depends-on: app.db (Base)
# Last-renovated: 2026-06-12
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
)

from app.db import Base


def _utcnow() -> datetime:
    return datetime.utcnow()


class SentinelBaseline(Base):
    """Learned normality — one row per (process, remote, port, kind) ever observed.

    kind: "outbound" (remote_key = rDNS domain if known, else IP) or
          "listen" (remote_key = "", remote_port = local listening port).
    """

    __tablename__ = "sentinel_baseline"

    id = Column(Integer, primary_key=True, autoincrement=True)
    process_name = Column(String, nullable=False, index=True)
    remote_key = Column(String, nullable=False, default="")
    remote_port = Column(Integer, nullable=False, default=0)
    kind = Column(String, nullable=False, default="outbound")
    country = Column(String, nullable=True)
    first_seen = Column(DateTime, nullable=False, default=_utcnow)
    last_seen = Column(DateTime, nullable=False, default=_utcnow)
    hit_count = Column(Integer, nullable=False, default=1)
    trusted = Column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint(
            "process_name", "remote_key", "remote_port", "kind",
            name="uq_sentinel_baseline_pair",
        ),
    )


class SentinelConnection(Base):
    """Connection events (first sight of a pair, or a new listener) — 30-day rolling."""

    __tablename__ = "sentinel_connections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime, nullable=False, default=_utcnow, index=True)
    event = Column(String, nullable=False, default="new")  # new | listen
    pid = Column(Integer, nullable=False, default=0)
    process_name = Column(String, nullable=False, default="", index=True)
    exe_path = Column(String, nullable=False, default="")
    laddr_ip = Column(String, nullable=False, default="")
    laddr_port = Column(Integer, nullable=True)
    raddr_ip = Column(String, nullable=False, default="")
    raddr_port = Column(Integer, nullable=True)
    status = Column(String, nullable=False, default="")
    proto = Column(String, nullable=False, default="")
    rdns = Column(String, nullable=False, default="")
    country = Column(String, nullable=False, default="")


class SentinelAlert(Base):
    """Alert lifecycle — created by rules (+ optional LLM triage), acked by Taz."""

    __tablename__ = "sentinel_alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow, index=True)
    severity = Column(String, nullable=False, default="low")  # severe | medium | low
    rule_key = Column(String, nullable=False, default="")
    title = Column(String, nullable=False, default="")
    explanation = Column(Text, nullable=False, default="")
    process = Column(String, nullable=False, default="")
    remote = Column(String, nullable=False, default="")
    recommended_action = Column(String, nullable=False, default="watch")  # ignore | watch | propose_block
    confidence = Column(Float, nullable=True)
    acknowledged = Column(Boolean, nullable=False, default=False, index=True)
    action_taken = Column(String, nullable=True)
    suppressed = Column(Boolean, nullable=False, default=False)  # learn-mode medium/low


class SentinelState(Base):
    """Key/value state — learn_mode_started_at, daily connection counters, settings."""

    __tablename__ = "sentinel_state"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String, nullable=False, unique=True)
    value = Column(Text, nullable=False, default="")
    updated_at = Column(DateTime, nullable=False, default=_utcnow)
