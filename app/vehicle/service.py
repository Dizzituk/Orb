# FILE: app/vehicle/service.py
# Purpose: Vehicle write-side service — virtual odometer calibration, trip batch
#          ingest with WorkDay folding, trip reclassification, DTC upserts.
#          Pure functions taking a Session (unit-testable).
# Called-by: app.vehicle.router, app.bridge.vehicle, app.tools.vehicle_tools, app.vehicle.status, app.vehicle.maintenance
# Depends-on: app.vehicle.models, app.finance.services.work_day_service, app.vehicle.dtc_codes
# Last-renovated: 2026-06-12
"""
Vehicle write-side service.

WorkDay (finance_work_days) stays canonical and its maths stay in
work_day_service — this module FEEDS that service, never duplicates it.
The one raw field written here is WorkDay.end_odometer (the rolling
"day stays open" update the spec defines); derived fields are then
recomputed by work_day_service's own _recompute, so all ledger maths
remain in exactly one place.

Folding rules (v1, deliberately simple and honest):
  - work trip, no WorkDay row yet      → open the day (start_time + virtual
    start odometer from this trip), roll end_odometer forward. The existing
    finish_work_day voice flow closes the day; we never complete it here.
  - work trip, day open                → roll end_odometer forward.
  - personal trip, day open            → add to personal_miles (split maths
    recomputed by work_day_service) and roll end_odometer (the van moved).
  - personal trip, no day / day closed → lives on the trip record only.
  - any trip, day already complete     → trip recorded, accumulator advanced,
    day untouched (late uploads never rewrite a closed ledger row; periodic
    dash calibration absorbs the drift).
Trips are processed in started_at order so the virtual odometer rolls
forward in ride order; unfolded personal trips are swept into a day that
opens later in the same date (morning errand before the first work trip).

Calibration sanity (per spec): a dash reading below the current virtual
odometer is REJECTED — values up to the current virtual odo may already be
written into WorkDay rows, and the virtual odometer must never go backwards.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Optional

from dateutil import tz as _dateutil_tz
from sqlalchemy.orm import Session

from app.vehicle.models import (
    VehicleDtcEvent,
    VehicleState,
    VehicleTrip,
)

logger = logging.getLogger(__name__)

# dateutil, not zoneinfo: this venv has no tzdata package (Windows ships no
# IANA db) but dateutil resolves Europe/London from its own data. Zero new deps.
_LOCAL_TZ = _dateutil_tz.gettz("Europe/London")

CLASSIFICATIONS = ("work", "personal", "unclassified")
MAINTENANCE_KINDS = (
    "mot", "insurance", "tax", "service",
    "tyres_front", "tyres_rear", "brake_pads", "brake_discs", "custom",
)
WEAR_KINDS = ("tyres_front", "tyres_rear", "brake_pads", "brake_discs")

# Suggested wear thresholds for a loaded LWB delivery van (spec §8 defaults;
# Taz can override per item by voice). Miles.
DEFAULT_INTERVAL_MILES = {
    "tyres_front": 20_000.0,
    "tyres_rear": 25_000.0,
    "brake_pads": 28_000.0,
    "brake_discs": 55_000.0,
}
DEFAULT_LABELS = {
    "mot": "MOT", "insurance": "Insurance", "tax": "Road tax",
    "service": "Service", "tyres_front": "Front tyres",
    "tyres_rear": "Rear tyres", "brake_pads": "Brake pads",
    "brake_discs": "Brake discs",
}
DEFAULT_LEAD_DAYS = 30
# Calibration drift worth mentioning out loud (fraction of miles driven
# since the last calibration).
DRIFT_MENTION_PCT = 0.02


def _utcnow() -> datetime:
    return datetime.utcnow()


# ─── datetime helpers ─────────────────────────────────────

def _parse_dt(value) -> Optional[datetime]:
    """ISO-8601 string/datetime → naive UTC datetime (DB convention)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _parse_date(value) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None


def _local_hhmm(dt_utc: datetime) -> str:
    """Naive-UTC datetime → local (Europe/London) HH:MM for WorkDay times."""
    return dt_utc.replace(tzinfo=timezone.utc).astimezone(_LOCAL_TZ).strftime("%H:%M")


def _local_date(dt_utc: datetime) -> date:
    return dt_utc.replace(tzinfo=timezone.utc).astimezone(_LOCAL_TZ).date()


# ─── state / virtual odometer ─────────────────────────────

def get_state(db: Session) -> VehicleState:
    """Fetch (or create) the single vehicle_state row."""
    state = db.query(VehicleState).first()
    if state is None:
        state = VehicleState(miles_accumulated_since_baseline=0.0)
        db.add(state)
        db.flush()
    return state


def virtual_odometer(db: Session) -> Optional[float]:
    """Baseline + accumulated, or None until the first calibration."""
    state = get_state(db)
    if state.odometer_baseline_miles is None:
        return None
    return round(state.odometer_baseline_miles
                 + (state.miles_accumulated_since_baseline or 0.0), 1)


def calibrate_odometer(db: Session, reading: float,
                       at: Optional[datetime] = None) -> dict:
    """Set the odometer baseline from a dash reading; zero the accumulator.

    Rejects readings that imply the virtual odometer going backwards (see
    module docstring). When the dash and the virtual odo disagree by more
    than DRIFT_MENTION_PCT of the miles driven since the last calibration,
    the drift is returned so Astra can mention it.
    """
    try:
        reading = float(reading)
    except (TypeError, ValueError):
        return {"ok": False, "error": "reading must be a number"}
    if reading <= 0:
        return {"ok": False, "error": "Odometer reading must be positive."}

    state = get_state(db)
    current = virtual_odometer(db)
    result: dict = {"ok": True, "reading": round(reading, 1),
                    "previous_virtual": current, "drift_miles": None,
                    "drift_pct": None}

    if current is not None:
        if reading < current:
            return {
                "ok": False,
                "previous_virtual": current,
                "error": (
                    f"Reading {reading:.0f} is below the current virtual "
                    f"odometer {current:.0f} — that would take the odometer "
                    "backwards. Check the reading; ledger rows may already "
                    "carry the higher figure."
                ),
            }
        drift = round(reading - current, 1)
        miles_since = state.miles_accumulated_since_baseline or 0.0
        result["drift_miles"] = drift
        if miles_since > 0:
            pct = drift / miles_since
            result["drift_pct"] = round(pct, 4)
            if pct > DRIFT_MENTION_PCT:
                result["drift_note"] = (
                    f"Virtual odometer drifted {drift:.1f} mi over "
                    f"{miles_since:.0f} mi since last calibration "
                    f"({pct * 100:.1f}% — speed-integration undercount)."
                )

    state.odometer_baseline_miles = round(reading, 1)
    state.odometer_baseline_at = _parse_dt(at) or _utcnow()
    state.miles_accumulated_since_baseline = 0.0
    db.commit()
    result["virtual_odometer"] = virtual_odometer(db)
    return result


# ─── WorkDay folding helpers ──────────────────────────────

def _advance_open_day(db: Session, day, virtual_after: Optional[float]) -> None:
    """Roll an OPEN day's end_odometer forward to the virtual odometer.

    Raw field write + work_day_service._recompute so distance/work-mile maths
    stay canonical. Never moves end_odometer backwards, never touches a
    complete day, never lets end fall below start.
    """
    from app.finance.services.work_day_service import _recompute
    if day is None or day.status == "complete" or virtual_after is None:
        return
    if day.start_odometer is not None and virtual_after < day.start_odometer:
        return
    if day.end_odometer is None or virtual_after > day.end_odometer:
        day.end_odometer = float(virtual_after)
        _recompute(day)


def _fold_work_trip(db: Session, trip: VehicleTrip,
                    virtual_before: Optional[float],
                    virtual_after: Optional[float]) -> bool:
    """First work trip opens the day; every work trip rolls end_odometer."""
    from app.finance.services import work_day_service as wds
    day = wds.get_day(db, trip.work_date)
    if day is None:
        day = wds.start_day(
            db, work_date=trip.work_date,
            start_time=_local_hhmm(trip.started_at),
            start_odometer=virtual_before,
        )
    elif day.status == "complete":
        logger.info("[vehicle] work trip %s lands on completed day %s — "
                    "day left untouched", trip.trip_uuid, trip.work_date)
        return False
    else:
        # Fill gaps without overwriting what's already on the row.
        wds.start_day(
            db, work_date=trip.work_date,
            start_time=None if day.start_time else _local_hhmm(trip.started_at),
            start_odometer=None if day.start_odometer is not None else virtual_before,
        )
    _advance_open_day(db, day, virtual_after)
    db.commit()
    return True


def _fold_personal_trip(db: Session, trip: VehicleTrip,
                        virtual_after: Optional[float]) -> bool:
    """Personal miles join the split only while the day is open."""
    from app.finance.services import work_day_service as wds
    day = wds.get_day(db, trip.work_date)
    if day is None or day.status == "complete":
        return False
    _advance_open_day(db, day, virtual_after)
    wds.set_mileage_split(
        db, work_date=trip.work_date,
        personal_miles=round((day.personal_miles or 0.0) + trip.distance_miles, 1),
    )
    return True


def _sweep_unfolded_personal(db: Session, work_date: date) -> int:
    """Fold personal trips that arrived before the day existed (morning errand,
    then work opens the day later in the batch)."""
    from app.finance.services import work_day_service as wds
    day = wds.get_day(db, work_date)
    if day is None or day.status == "complete":
        return 0
    pending = (db.query(VehicleTrip)
                 .filter(VehicleTrip.work_date == work_date,
                         VehicleTrip.classification == "personal",
                         VehicleTrip.folded_into_day == False)  # noqa: E712
                 .all())
    for trip in pending:
        if _fold_personal_trip(db, trip, None):
            trip.folded_into_day = True
    return len(pending)


# ─── trip ingest ──────────────────────────────────────────

def _normalize_trip(raw: dict) -> dict | str:
    """Validate one incoming trip dict; returns normalized dict or error str."""
    uuid = str(raw.get("trip_uuid") or "").strip()
    if not uuid:
        return "trip_uuid missing"
    started_at = _parse_dt(raw.get("started_at"))
    if started_at is None:
        return f"{uuid}: started_at missing/unparseable"
    try:
        distance = max(float(raw.get("distance_miles") or 0.0), 0.0)
    except (TypeError, ValueError):
        return f"{uuid}: distance_miles unparseable"
    classification = str(raw.get("classification") or "unclassified").strip().lower()
    if classification not in CLASSIFICATIONS:
        classification = "unclassified"
    work_date = _parse_date(raw.get("work_date")) or _local_date(started_at)

    def _f(key):
        v = raw.get(key)
        try:
            return float(v) if v is not None and v != "" else None
        except (TypeError, ValueError):
            return None

    return {
        "trip_uuid": uuid, "started_at": started_at,
        "ended_at": _parse_dt(raw.get("ended_at")),
        "distance_miles": round(distance, 2),
        "classification": classification,
        "source": str(raw.get("source") or "obd")[:10],
        "avg_speed_mph": _f("avg_speed_mph"), "max_speed_mph": _f("max_speed_mph"),
        "start_voltage": _f("start_voltage"), "end_voltage": _f("end_voltage"),
        "work_date": work_date,
    }


def ingest_trips(db: Session, trips: list[dict]) -> dict:
    """Ingest a phone batch. Dedup on trip_uuid; fold into WorkDay per the
    module rules. Redelivery of a trip the server holds as UNCLASSIFIED with
    a real classification upgrades it (the phone's post-upload answer path).
    Returns {ingested, duplicates, reclassified, days_touched, errors}."""
    normalized, errors = [], []
    for raw in trips or []:
        out = _normalize_trip(raw if isinstance(raw, dict) else {})
        (normalized if isinstance(out, dict) else errors).append(out)
    normalized.sort(key=lambda t: t["started_at"])

    state = get_state(db)
    ingested, duplicates, reclassified = 0, 0, 0
    days_touched: set[date] = set()

    for t in normalized:
        existing = (db.query(VehicleTrip)
                      .filter(VehicleTrip.trip_uuid == t["trip_uuid"]).one_or_none())
        if existing is not None:
            if (existing.classification == "unclassified"
                    and t["classification"] in ("work", "personal")):
                result = classify_trip(db, t["trip_uuid"], t["classification"])
                if result.get("ok"):
                    reclassified += 1
                    days_touched.add(existing.work_date)
                    continue
            duplicates += 1
            continue

        virtual_before = virtual_odometer(db)
        trip = VehicleTrip(**t)
        db.add(trip)
        state.miles_accumulated_since_baseline = round(
            (state.miles_accumulated_since_baseline or 0.0) + t["distance_miles"], 2)
        db.flush()
        virtual_after = virtual_odometer(db)

        if trip.classification == "work":
            trip.folded_into_day = _fold_work_trip(db, trip, virtual_before, virtual_after)
        elif trip.classification == "personal":
            trip.folded_into_day = _fold_personal_trip(db, trip, virtual_after)

        ingested += 1
        days_touched.add(t["work_date"])

    for d in days_touched:
        _sweep_unfolded_personal(db, d)
    db.commit()

    return {
        "ingested": ingested,
        "duplicates": duplicates,
        "reclassified": reclassified,
        "days_touched": sorted(d.isoformat() for d in days_touched),
        "errors": errors,
    }


# ─── reclassification ─────────────────────────────────────

def classify_trip(db: Session, trip_uuid: str, classification: str) -> dict:
    """Reclassify a trip and (re)fold its WorkDay contribution.

    Open days are adjusted both ways (personal↔work). Completed days are
    never rewritten — the trip record flips, the ledger row stays; voice
    reconciliation is the correction path for closed days.
    """
    from app.finance.services import work_day_service as wds
    classification = str(classification or "").strip().lower()
    if classification not in CLASSIFICATIONS:
        return {"ok": False, "error": f"classification must be one of {CLASSIFICATIONS}"}
    trip = (db.query(VehicleTrip)
              .filter(VehicleTrip.trip_uuid == str(trip_uuid).strip()).one_or_none())
    if trip is None:
        return {"ok": False, "error": f"No trip {trip_uuid}"}
    if trip.classification == classification:
        return {"ok": True, "trip_uuid": trip.trip_uuid,
                "classification": classification, "unchanged": True}

    old = trip.classification
    day = wds.get_day(db, trip.work_date)
    day_open = day is not None and day.status != "complete"

    # Unwind a personal fold before switching away from personal.
    if old == "personal" and trip.folded_into_day and day_open:
        wds.set_mileage_split(
            db, work_date=trip.work_date,
            personal_miles=round(max((day.personal_miles or 0.0)
                                     - trip.distance_miles, 0.0), 1))
        trip.folded_into_day = False

    trip.classification = classification

    if classification == "work":
        trip.folded_into_day = _fold_work_trip(db, trip, None, None)
    elif classification == "personal":
        trip.folded_into_day = _fold_personal_trip(db, trip, None)
    else:
        trip.folded_into_day = False

    db.commit()
    return {"ok": True, "trip_uuid": trip.trip_uuid, "classification": classification,
            "work_date": trip.work_date.isoformat(),
            "folded_into_day": trip.folded_into_day,
            "day_status": (day.status if day else None)}


# ─── DTCs ─────────────────────────────────────────────────

def record_dtcs(db: Session, codes: list[dict], trip_uuid: Optional[str] = None) -> dict:
    """Upsert DTC events from a trip-start read; return which codes are NEW
    (never seen, or returning after being cleared) so the phone can notify."""
    from app.vehicle.dtc_codes import describe
    new_codes, known_codes = [], []
    for item in codes or []:
        if isinstance(item, str):
            item = {"code": item}
        code = str(item.get("code") or "").strip().upper()
        if not code:
            continue
        event = (db.query(VehicleDtcEvent)
                   .filter(VehicleDtcEvent.code == code,
                           VehicleDtcEvent.cleared == False)  # noqa: E712
                   .one_or_none())
        if event is not None:
            event.last_seen = _utcnow()
            event.occurrence_count = (event.occurrence_count or 1) + 1
            if trip_uuid:
                event.trip_uuid = trip_uuid
            known_codes.append(code)
        else:
            db.add(VehicleDtcEvent(
                code=code,
                description=item.get("description") or describe(code),
                severity=str(item.get("severity") or "warn"),
                trip_uuid=trip_uuid,
            ))
            new_codes.append(code)
    db.commit()
    return {"new": new_codes, "known": known_codes}


def clear_dtc(db: Session, event_id: int) -> dict:
    """Mark a DTC event cleared in the DB only (v1 never touches the van)."""
    event = db.query(VehicleDtcEvent).filter(VehicleDtcEvent.id == event_id).one_or_none()
    if event is None:
        return {"ok": False, "error": f"No DTC event {event_id}"}
    event.cleared = True
    db.commit()
    return {"ok": True, "id": event.id, "code": event.code, "cleared": True}


# Maintenance write-side (set/update/log-fitted/deactivate) lives in
# app/vehicle/maintenance.py — split for the one-job-per-file size rule.
