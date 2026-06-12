# FILE: app/lifestyle/health_sync.py
# Purpose: Health Connect ingest — files batches of wearable data (Garmin via Health
# Called-by: app.bridge.dashboards
# Depends-on: app.db, app.lifestyle.models, app.lifestyle.service
# Last-renovated: 2026-06-11
"""
Health Connect ingest — files batches of wearable data (Garmin via Health
Connect, relayed by AstraBridge) into the lifestyle store.

The phone is the sensor: AstraBridge reads Health Connect on-device, aggregates
the daily metrics (steps / calories / resting HR / sleep), and posts discrete
weigh-ins and workout sessions plus those daily roll-ups here. Orb is the brain:
this module dedupes, writes into the existing lifestyle tables, and refreshes
the affected daily summaries.

Dedup: every weigh-in / workout carries the Health Connect record's own stable
UID (client_id). We keep a ledger of UIDs already filed, so periodic syncs with
overlapping windows never double-insert. Daily roll-ups dedup naturally — they
upsert the DailySummary row for their date.

All timestamps are normalised to UTC on the way in, matching the rest of the
lifestyle layer. Kept separate from service.py so neither file grows past the
size budget; this module owns everything Health-Connect-ingest specific.
"""
from __future__ import annotations

import logging
from datetime import datetime, date, timezone
from typing import Optional, List, Dict, Any, Iterable

from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from app.db import engine
from app.lifestyle.models import (
    WeightEntry, WorkoutSession, DailySummary, HealthIngestLedger,
)

logger = logging.getLogger(__name__)

_schema_ready = False


# ---------------------------------------------------------------------------
# Schema (idempotent, run-once) — adds wearable columns + ledger table.
# ---------------------------------------------------------------------------

def ensure_health_schema() -> None:
    """Create the ingest ledger table and the DailySummary wearable columns if
    missing. Idempotent and cheap; guarded so the inspect runs once per process.

    create_all also makes these on a fresh DB (the models define them); this
    handles an existing DB that predates the columns.
    """
    global _schema_ready
    if _schema_ready:
        return

    try:
        HealthIngestLedger.__table__.create(bind=engine, checkfirst=True)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[health_sync] ledger create skipped: %s", exc)

    new_cols = {
        "steps": "INTEGER",
        "floors": "INTEGER",
        "active_calories": "FLOAT",
        "total_calories_burned": "FLOAT",
        "resting_hr": "INTEGER",
        "sleep_minutes": "INTEGER",
    }
    workout_cols = {
        "avg_hr": "INTEGER",
        "max_hr": "INTEGER",
        "hr_zones_json": "TEXT",
    }
    try:
        inspector = inspect(engine)
        if "lifestyle_daily_summaries" in inspector.get_table_names():
            existing = {c["name"] for c in inspector.get_columns("lifestyle_daily_summaries")}
            with engine.connect() as conn:
                for name, sqltype in new_cols.items():
                    if name not in existing:
                        conn.execute(text(
                            f"ALTER TABLE lifestyle_daily_summaries ADD COLUMN {name} {sqltype}"
                        ))
                        conn.commit()
                        logger.info("[health_sync] added lifestyle_daily_summaries.%s", name)
        if "lifestyle_workout_sessions" in inspector.get_table_names():
            existing_w = {c["name"] for c in inspector.get_columns("lifestyle_workout_sessions")}
            with engine.connect() as conn:
                for name, sqltype in workout_cols.items():
                    if name not in existing_w:
                        conn.execute(text(
                            f"ALTER TABLE lifestyle_workout_sessions ADD COLUMN {name} {sqltype}"
                        ))
                        conn.commit()
                        logger.info("[health_sync] added lifestyle_workout_sessions.%s", name)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[health_sync] daily-summary migration skipped: %s", exc)

    _schema_ready = True


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_dt(value: Any) -> Optional[datetime]:
    """Parse an ISO 8601 string to a timezone-aware UTC datetime."""
    if not value:
        return None
    s = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_date(value: Any) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None


def _already_ingested(db: Session, client_id: str) -> bool:
    if not client_id:
        return False
    return db.query(HealthIngestLedger).get(client_id) is not None


def _record_ledger(db: Session, client_id: str, kind: str,
                   orb_row_id: Optional[int], source: Optional[str]) -> None:
    db.add(HealthIngestLedger(
        client_id=client_id, kind=kind, orb_row_id=orb_row_id, source=source,
    ))


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest_health_batch(
    db: Session,
    weights: Iterable[Dict[str, Any]] = (),
    workouts: Iterable[Dict[str, Any]] = (),
    days: Iterable[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    """File a batch of Health Connect data. Returns counts.

    Idempotent per record via the ingest ledger (weights / workouts) and per
    date (days). Back-fill safe: weigh-ins and workouts are stored with their
    real timestamps and the affected daily summaries are recomputed.
    """
    ensure_health_schema()

    errors: List[str] = []
    weights_added = weights_skipped = 0
    workouts_added = workouts_skipped = 0
    days_upserted = 0
    affected_dates = set()

    # --- Weigh-ins -> WeightEntry (feeds the weight log + trend) ---
    for w in weights or ():
        try:
            cid = str(w.get("client_id") or "").strip()
            if cid and _already_ingested(db, cid):
                weights_skipped += 1
                continue
            kg = w.get("weight_kg")
            if kg is None:
                errors.append("weight missing weight_kg")
                continue
            ts = _parse_dt(w.get("recorded_at")) or datetime.now(timezone.utc)
            entry = WeightEntry(
                weight_kg=float(kg),
                source=str(w.get("source") or "garmin"),
                recorded_at=ts,
            )
            db.add(entry)
            db.flush()
            if cid:
                _record_ledger(db, cid, "weight", entry.id, entry.source)
            weights_added += 1
            affected_dates.add(ts.date())
        except Exception as exc:
            errors.append(f"weight: {exc}")

    # --- Workouts -> WorkoutSession (surf, cardio, etc.) ---
    for wo in workouts or ():
        try:
            cid = str(wo.get("client_id") or "").strip()
            if cid and _already_ingested(db, cid):
                workouts_skipped += 1
                continue
            ts = _parse_dt(wo.get("started_at")) or datetime.now(timezone.utc)
            # Distance / avg HR have no dedicated columns yet — fold into notes.
            extras = []
            dist = wo.get("distance_m")
            if dist:
                try:
                    extras.append(f"{float(dist) / 1000:.2f} km")
                except (TypeError, ValueError):
                    pass
            hr = wo.get("avg_hr")
            if hr:
                extras.append(f"avg HR {int(hr)}")
            notes = ", ".join(extras) or None
            session = WorkoutSession(
                started_at=ts,
                duration_mins=wo.get("duration_mins"),
                activity_type=str(wo.get("activity_type") or "workout").strip().lower(),
                title=wo.get("title"),
                calories_burned=wo.get("calories_burned"),
                avg_hr=int(wo["avg_hr"]) if wo.get("avg_hr") else None,
                max_hr=int(wo["max_hr"]) if wo.get("max_hr") else None,
                hr_zones_json=wo.get("hr_zones") or None,
                notes=notes,
                source=str(wo.get("source") or "garmin"),
            )
            db.add(session)
            db.flush()
            if cid:
                _record_ledger(db, cid, "workout", session.id, session.source)
            workouts_added += 1
            affected_dates.add(ts.date())
        except Exception as exc:
            errors.append(f"workout: {exc}")

    # --- Daily metrics -> DailySummary (merge; never clobber nutrition) ---
    for d in days or ():
        try:
            day = _parse_date(d.get("date"))
            if day is None:
                errors.append("day missing/invalid date")
                continue
            row = db.query(DailySummary).filter(DailySummary.date == day).first()
            if row is None:
                row = DailySummary(date=day)
                db.add(row)
            if d.get("steps") is not None:
                row.steps = int(d["steps"])
            if d.get("floors") is not None:
                row.floors = int(d["floors"])
            if d.get("active_calories") is not None:
                row.active_calories = float(d["active_calories"])
            if d.get("total_calories") is not None:
                row.total_calories_burned = float(d["total_calories"])
            if d.get("resting_hr") is not None:
                row.resting_hr = int(d["resting_hr"])
            if d.get("sleep_minutes") is not None:
                row.sleep_minutes = int(d["sleep_minutes"])
            days_upserted += 1
        except Exception as exc:
            errors.append(f"day: {exc}")

    db.commit()

    # Recompute affected daily summaries so weight/activity roll up for the
    # real dates (handles back-filled history, not just today). Disjoint from
    # the wearable columns above, so steps/sleep survive.
    if affected_dates:
        try:
            from app.lifestyle.service import _update_daily_summary
            for d in affected_dates:
                _update_daily_summary(db, d)
        except Exception as exc:
            errors.append(f"summary recompute: {exc}")

    logger.info(
        "[health_sync] ingest: +%d/%d weights, +%d/%d workouts, %d days, %d errors",
        weights_added, weights_added + weights_skipped,
        workouts_added, workouts_added + workouts_skipped,
        days_upserted, len(errors),
    )

    return {
        "ok": not errors,
        "weights_added": weights_added,
        "weights_skipped": weights_skipped,
        "workouts_added": workouts_added,
        "workouts_skipped": workouts_skipped,
        "days_upserted": days_upserted,
        "errors": errors,
    }
