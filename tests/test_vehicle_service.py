# FILE: tests/test_vehicle_service.py
# Purpose: Units for app/vehicle — virtual odometer maths, calibration sanity,
#          trip ingest dedup + WorkDay folding (asserting against
#          work_day_service's own recompute), wear/maintenance status, DTC upsert.
# Called-by: pytest
# Depends-on: app.vehicle.service, app.vehicle.maintenance, app.vehicle.status, app.finance.services.work_day_service
# Last-renovated: 2026-06-12
from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.finance.models_workday import WorkDay
from app.finance.services import work_day_service as wds
from app.vehicle import maintenance, service, status
from app.vehicle.models import (
    VehicleDtcEvent,
    VehicleMaintenanceItem,
    VehicleState,
    VehicleTrip,
)

_TABLES = [
    VehicleTrip.__table__,
    VehicleMaintenanceItem.__table__,
    VehicleDtcEvent.__table__,
    VehicleState.__table__,
    WorkDay.__table__,
]


@pytest.fixture()
def db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine, tables=_TABLES)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _trip(uuid: str, day: str, hhmm_utc: str, miles: float,
          classification: str = "work", **extra) -> dict:
    """Trip dict the bridge would post. hhmm_utc on the given date (UTC)."""
    started = f"{day}T{hhmm_utc}:00Z"
    out = {
        "trip_uuid": uuid, "started_at": started,
        "ended_at": None, "distance_miles": miles,
        "classification": classification, "source": "obd",
        "work_date": day,
    }
    out.update(extra)
    return out


# ── virtual odometer / calibration ───────────────────────────────────────────

def test_virtual_odometer_none_until_calibrated(db):
    assert service.virtual_odometer(db) is None
    service.calibrate_odometer(db, 82400)
    assert service.virtual_odometer(db) == 82400.0


def test_virtual_odometer_is_baseline_plus_trips(db):
    service.calibrate_odometer(db, 82400)
    service.ingest_trips(db, [_trip("t1", "2026-06-10", "08:00", 12.5),
                              _trip("t2", "2026-06-10", "10:00", 7.5)])
    assert service.virtual_odometer(db) == 82420.0


def test_calibration_rejects_backwards_reading(db):
    service.calibrate_odometer(db, 82400)
    service.ingest_trips(db, [_trip("t1", "2026-06-10", "08:00", 50.0)])
    result = service.calibrate_odometer(db, 82420)  # virtual is 82450
    assert result["ok"] is False
    assert service.virtual_odometer(db) == 82450.0  # unchanged, never backwards


def test_calibration_reports_drift_over_threshold(db):
    service.calibrate_odometer(db, 82400)
    service.ingest_trips(db, [_trip("t1", "2026-06-10", "08:00", 100.0)])
    # Dash says 82510 — virtual says 82500: 10 mi drift over 100 mi = 10%
    result = service.calibrate_odometer(db, 82510)
    assert result["ok"] is True
    assert result["drift_miles"] == 10.0
    assert result["drift_pct"] == pytest.approx(0.10)
    assert "drift_note" in result
    assert service.virtual_odometer(db) == 82510.0
    state = service.get_state(db)
    assert state.miles_accumulated_since_baseline == 0.0


def test_calibration_small_drift_no_note(db):
    service.calibrate_odometer(db, 82400)
    service.ingest_trips(db, [_trip("t1", "2026-06-10", "08:00", 100.0)])
    result = service.calibrate_odometer(db, 82501)  # 1 mi over 100 = 1%
    assert result["ok"] is True
    assert "drift_note" not in result


# ── ingest: dedup (acceptance: same batch twice ingests once) ────────────────

def test_ingest_dedup_same_batch_twice(db):
    service.calibrate_odometer(db, 82400)
    batch = [_trip("a", "2026-06-10", "08:00", 10.0),
             _trip("b", "2026-06-10", "12:00", 5.0, "personal")]
    first = service.ingest_trips(db, batch)
    assert first["ingested"] == 2 and first["duplicates"] == 0
    second = service.ingest_trips(db, batch)
    assert second["ingested"] == 0 and second["duplicates"] == 2
    assert second["reclassified"] == 0
    assert db.query(VehicleTrip).count() == 2
    assert service.virtual_odometer(db) == 82415.0  # distance counted once


def test_ingest_redelivery_upgrades_unclassified(db):
    """Phone answers AFTER upload: same UUID re-posted with a real
    classification upgrades the unclassified server record (no new row,
    no double distance)."""
    service.calibrate_odometer(db, 82400)
    service.ingest_trips(db, [_trip("u", "2026-06-10", "08:30", 12.0, "unclassified")])
    assert wds.get_day(db, date(2026, 6, 10)) is None

    result = service.ingest_trips(db, [_trip("u", "2026-06-10", "08:30", 12.0, "work")])
    assert result["ingested"] == 0 and result["duplicates"] == 0
    assert result["reclassified"] == 1
    assert db.query(VehicleTrip).count() == 1
    assert service.virtual_odometer(db) == 82412.0  # distance still counted once
    day = wds.get_day(db, date(2026, 6, 10))
    assert day is not None and day.status == "open"  # work fold ran

    # A classified record never downgrades or flips via redelivery.
    again = service.ingest_trips(db, [_trip("u", "2026-06-10", "08:30", 12.0, "personal")])
    assert again["reclassified"] == 0 and again["duplicates"] == 1
    assert db.query(VehicleTrip).filter_by(trip_uuid="u").one().classification == "work"


# ── ingest: WorkDay folding (acceptance: maths held by work_day_service) ─────

def test_work_trip_opens_day_with_time_and_virtual_odometer(db):
    service.calibrate_odometer(db, 82400)
    service.ingest_trips(db, [_trip("a", "2026-06-10", "08:30", 12.0)])
    day = wds.get_day(db, date(2026, 6, 10))
    assert day is not None and day.status == "open"
    assert day.start_time == "09:30"  # BST (Europe/London) in June
    assert day.start_odometer == 82400.0
    assert day.end_odometer == 82412.0
    assert day.total_distance == 12.0
    assert day.work_miles == 12.0


def test_existing_finish_flow_closes_day_and_maths_hold(db):
    service.calibrate_odometer(db, 82400)
    service.ingest_trips(db, [_trip("a", "2026-06-10", "08:30", 60.0),
                              _trip("b", "2026-06-10", "11:00", 40.0)])
    # Taz's voice finish flow, untouched: parcels from screenshot OCR
    day = wds.finish_day(db, work_date=date(2026, 6, 10), end_time="17:15",
                         parcels=120)
    assert day.status == "complete"
    assert day.start_odometer == 82400.0
    assert day.end_odometer == 82500.0   # rolling virtual odometer
    assert day.total_distance == 100.0
    assert day.work_miles == 100.0
    assert day.gross_earnings == pytest.approx(120 * 2.35)
    assert day.total_hours == pytest.approx(7.75)  # 09:30→17:15 local


def test_personal_trip_on_work_day_feeds_split(db):
    service.calibrate_odometer(db, 82400)
    service.ingest_trips(db, [_trip("a", "2026-06-10", "08:30", 60.0),
                              _trip("b", "2026-06-10", "13:00", 8.0, "personal"),
                              _trip("c", "2026-06-10", "14:00", 32.0)])
    day = wds.get_day(db, date(2026, 6, 10))
    assert day.personal_miles == 8.0
    assert day.total_distance == 100.0
    assert day.work_miles == 92.0       # work_day_service recompute: total - personal
    assert day.end_odometer == 82500.0  # personal miles still moved the odometer


def test_personal_trip_alone_creates_no_day(db):
    service.calibrate_odometer(db, 82400)
    result = service.ingest_trips(db, [_trip("p", "2026-06-11", "10:00", 15.0, "personal")])
    assert result["ingested"] == 1
    assert wds.get_day(db, date(2026, 6, 11)) is None
    trip = db.query(VehicleTrip).filter_by(trip_uuid="p").one()
    assert trip.folded_into_day is False
    assert service.virtual_odometer(db) == 82415.0  # still accumulates


def test_personal_before_work_same_day_is_swept_into_split(db):
    service.calibrate_odometer(db, 82400)
    # One batch: morning errand BEFORE the first work trip
    service.ingest_trips(db, [_trip("p", "2026-06-10", "07:00", 6.0, "personal"),
                              _trip("w", "2026-06-10", "08:30", 50.0)])
    day = wds.get_day(db, date(2026, 6, 10))
    assert day is not None
    assert day.personal_miles == 6.0
    trip = db.query(VehicleTrip).filter_by(trip_uuid="p").one()
    assert trip.folded_into_day is True


def test_late_work_trip_never_rewrites_completed_day(db):
    service.calibrate_odometer(db, 82400)
    service.ingest_trips(db, [_trip("a", "2026-06-10", "08:30", 100.0)])
    wds.finish_day(db, work_date=date(2026, 6, 10), end_time="17:00", parcels=100)
    day = wds.get_day(db, date(2026, 6, 10))
    end_before = day.end_odometer
    result = service.ingest_trips(db, [_trip("late", "2026-06-10", "16:00", 9.0)])
    assert result["ingested"] == 1
    db.refresh(day)
    assert day.end_odometer == end_before          # ledger row untouched
    assert day.status == "complete"
    late = db.query(VehicleTrip).filter_by(trip_uuid="late").one()
    assert late.folded_into_day is False
    assert service.virtual_odometer(db) == 82509.0  # accumulator still honest


def test_unclassified_trip_records_without_folding(db):
    service.calibrate_odometer(db, 82400)
    service.ingest_trips(db, [_trip("u", "2026-06-10", "08:30", 10.0, "unclassified")])
    assert wds.get_day(db, date(2026, 6, 10)) is None
    assert db.query(VehicleTrip).filter_by(trip_uuid="u").one().folded_into_day is False


def test_uncalibrated_ingest_opens_day_without_odometer(db):
    service.ingest_trips(db, [_trip("a", "2026-06-10", "08:30", 12.0)])
    day = wds.get_day(db, date(2026, 6, 10))
    assert day is not None and day.start_time == "09:30"
    assert day.start_odometer is None and day.end_odometer is None
    assert service.virtual_odometer(db) is None


# ── reclassification ─────────────────────────────────────────────────────────

def test_classify_unclassified_to_work_opens_day(db):
    service.calibrate_odometer(db, 82400)
    service.ingest_trips(db, [_trip("u", "2026-06-10", "08:30", 10.0, "unclassified")])
    result = service.classify_trip(db, "u", "work")
    assert result["ok"] is True and result["folded_into_day"] is True
    day = wds.get_day(db, date(2026, 6, 10))
    assert day is not None and day.status == "open"
    assert day.start_time == "09:30"


def test_classify_personal_to_work_unwinds_split(db):
    service.calibrate_odometer(db, 82400)
    service.ingest_trips(db, [_trip("w", "2026-06-10", "08:30", 50.0),
                              _trip("p", "2026-06-10", "13:00", 10.0, "personal")])
    day = wds.get_day(db, date(2026, 6, 10))
    assert day.personal_miles == 10.0 and day.work_miles == 50.0
    result = service.classify_trip(db, "p", "work")
    assert result["ok"] is True
    db.refresh(day)
    assert day.personal_miles == 0.0
    assert day.work_miles == 60.0  # recomputed by work_day_service


def test_classify_work_to_personal_adds_split(db):
    service.calibrate_odometer(db, 82400)
    service.ingest_trips(db, [_trip("w1", "2026-06-10", "08:30", 50.0),
                              _trip("w2", "2026-06-10", "13:00", 10.0)])
    result = service.classify_trip(db, "w2", "personal")
    assert result["ok"] is True
    day = wds.get_day(db, date(2026, 6, 10))
    assert day.personal_miles == 10.0
    assert day.work_miles == 50.0


def test_classify_same_value_is_noop(db):
    service.calibrate_odometer(db, 82400)
    service.ingest_trips(db, [_trip("w", "2026-06-10", "08:30", 50.0)])
    result = service.classify_trip(db, "w", "work")
    assert result["ok"] is True and result.get("unchanged") is True


def test_classify_unknown_trip_errors(db):
    result = service.classify_trip(db, "nope", "work")
    assert result["ok"] is False


# ── wear / maintenance status ────────────────────────────────────────────────

def test_wear_status_thresholds(db):
    service.calibrate_odometer(db, 100_000)
    maintenance.set_maintenance(db, "brake_pads", fitted_at_odometer=74_000.0,
                                interval_miles=28_000.0)
    service.ingest_trips(db, [_trip("t", "2026-06-10", "08:00", 0.0)])
    wear = {w["kind"]: w for w in status.wear_status(db)}
    pads = wear["brake_pads"]
    assert pads["miles_done"] == 26_000.0
    assert pads["pct"] == pytest.approx(0.929, abs=0.001)
    assert pads["warning"] is True  # ≥ 90%


def test_wear_status_under_threshold_no_warning(db):
    service.calibrate_odometer(db, 100_000)
    maintenance.set_maintenance(db, "tyres_front", fitted_at_odometer=90_000.0)
    wear = {w["kind"]: w for w in status.wear_status(db)}
    front = wear["tyres_front"]
    assert front["interval_miles"] == 20_000.0  # spec default applied
    assert front["miles_done"] == 10_000.0
    assert front["warning"] is False


def test_log_component_fitted_requires_calibration(db):
    result = maintenance.log_component_fitted(db, "tyres_front")
    assert result["ok"] is False
    service.calibrate_odometer(db, 82400)
    result = maintenance.log_component_fitted(db, "tyres_front")
    assert result["ok"] is True and result["fitted_at_odometer"] == 82400.0


def test_maintenance_date_warnings(db):
    today = date.today()
    maintenance.set_maintenance(db, "mot", due_date=(today + timedelta(days=10)).isoformat(),
                                lead_days=30)
    maintenance.set_maintenance(db, "insurance",
                                due_date=(today + timedelta(days=90)).isoformat(),
                                lead_days=30)
    maintenance.set_maintenance(db, "tax", due_date=(today - timedelta(days=2)).isoformat())
    by_kind = {m["kind"]: m for m in status.maintenance_status(db)}
    assert by_kind["mot"]["warning"] is True and by_kind["mot"]["overdue"] is False
    assert by_kind["insurance"]["warning"] is False
    assert by_kind["tax"]["overdue"] is True and by_kind["tax"]["days_remaining"] == -2


def test_maintenance_upsert_updates_not_duplicates(db):
    maintenance.set_maintenance(db, "mot", due_date="2027-03-14")
    maintenance.set_maintenance(db, "mot", due_date="2027-04-01")
    items = (db.query(VehicleMaintenanceItem)
               .filter_by(kind="mot").all())
    assert len(items) == 1
    assert items[0].due_date == date(2027, 4, 1)


def test_deactivate_maintenance_soft_deletes(db):
    created = maintenance.set_maintenance(db, "mot", due_date="2027-03-14")
    maintenance.deactivate_maintenance(db, created["id"])
    assert all(m["kind"] != "mot" for m in status.maintenance_status(db))
    assert db.query(VehicleMaintenanceItem).count() == 1  # row still there


# ── DTCs ─────────────────────────────────────────────────────────────────────

def test_dtc_new_then_known_then_cleared_reappears_new(db):
    first = service.record_dtcs(db, [{"code": "P0299"}], trip_uuid="t1")
    assert first["new"] == ["P0299"] and first["known"] == []
    event = db.query(VehicleDtcEvent).one()
    assert event.description == "Turbo underboost (boost leak / actuator)"

    second = service.record_dtcs(db, [{"code": "p0299"}], trip_uuid="t2")
    assert second["new"] == [] and second["known"] == ["P0299"]
    db.refresh(event)
    assert event.occurrence_count == 2

    service.clear_dtc(db, event.id)
    third = service.record_dtcs(db, [{"code": "P0299"}])
    assert third["new"] == ["P0299"]  # returning after clear = new event
    assert db.query(VehicleDtcEvent).count() == 2


def test_dtc_unknown_code_keeps_raw(db):
    result = service.record_dtcs(db, [{"code": "P1XYZ"}])
    assert result["new"] == ["P1XYZ"]
    event = db.query(VehicleDtcEvent).one()
    assert event.description is None  # Astra explains unknowns in chat


# ── summary ──────────────────────────────────────────────────────────────────

def test_summary_shape_and_day_open(db):
    service.calibrate_odometer(db, 82400)
    today = date.today()
    out = status.summary(db)
    assert out["virtual_odometer"] == 82400.0
    assert out["day_open"] is False
    assert out["unclassified_count"] == 0
    assert out["recent_trips"] == []

    service.ingest_trips(db, [
        _trip("w", today.isoformat(), "08:30", 40.0),
        _trip("u", today.isoformat(), "12:00", 5.0, "unclassified"),
    ])
    out = status.summary(db)
    assert out["day_open"] is True
    assert out["unclassified_count"] == 1
    assert len(out["recent_trips"]) == 2
    split = out["tax_year_split"]
    assert split["work_miles"] == 40.0
    assert split["unclassified_miles"] == 5.0
    assert split["business_use_pct"] == 1.0  # of classified miles


def test_summary_empty_db_is_safe(db):
    out = status.summary(db)
    assert out["virtual_odometer"] is None
    assert out["maintenance"] == []
    assert out["active_dtcs"] == []
    assert out["tax_year_split"]["trip_count"] == 0
