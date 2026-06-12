# FILE: app/vehicle/briefing_section.py
# Purpose: Build the "Van & Vehicle" briefing section — maintenance due within
#          lead time, wear warnings, active DTCs, unclassified trips.
# Called-by: app.briefing.briefing_compiler (compile_briefing)
# Depends-on: app.db, app.vehicle.status
# Last-renovated: 2026-06-12
"""
Vehicle section for the morning briefing. Returns plain data (no compiler
imports — the compiler owns its dataclasses) and only when there is
something worth saying; quiet days add nothing to the briefing. Opens its
own short-lived session so the compiler stays session-free. Never raises:
the briefing must not break because the vehicle module had a bad day.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

SECTION_NAME = "Van & Vehicle"


def build_vehicle_briefing_items() -> list[dict]:
    """[{headline, summary}] of vehicle warnings; [] when all quiet."""
    try:
        from app.db import SessionLocal
        from app.vehicle import status
        db = SessionLocal()
        try:
            items: list[dict] = []
            for m in status.maintenance_status(db):
                if m.get("overdue"):
                    items.append({
                        "headline": f"{m['label']} OVERDUE",
                        "summary": (f"{m['label']} was due {m['due_date']} "
                                    f"({-m['days_remaining']} days ago)."),
                    })
                elif m.get("warning") and m.get("days_remaining") is not None:
                    items.append({
                        "headline": f"{m['label']} due in {m['days_remaining']} days",
                        "summary": f"{m['label']} is due {m['due_date']}.",
                    })
                elif m.get("warning") and m.get("pct") is not None:
                    items.append({
                        "headline": f"{m['label']} at {m['pct'] * 100:.0f}% of wear interval",
                        "summary": (f"{m['label']}: {m['miles_done']:.0f} of "
                                    f"~{m['interval_miles']:.0f} miles since fitting "
                                    "— worth an eyeball, not a verdict."),
                    })
            dtcs = status.active_dtcs(db)
            if dtcs:
                codes = ", ".join(
                    f"{d['code']}" + (f" ({d['description']})" if d["description"] else "")
                    for d in dtcs[:5])
                items.append({
                    "headline": f"{len(dtcs)} active fault code{'s' if len(dtcs) != 1 else ''}",
                    "summary": f"Van is showing: {codes}.",
                })
            from sqlalchemy import func
            from app.vehicle.models import VehicleTrip
            unclassified = (db.query(func.count(VehicleTrip.id))
                              .filter(VehicleTrip.classification == "unclassified")
                              .scalar() or 0)
            if unclassified:
                items.append({
                    "headline": f"{unclassified} unclassified trip{'s' if unclassified != 1 else ''}",
                    "summary": "Tell me which were work and which were personal "
                               "to keep the mileage split exact.",
                })
            return items
        finally:
            db.close()
    except Exception as exc:  # pragma: no cover — briefing must never break
        logger.warning("[vehicle_briefing] section build failed: %s", exc)
        return []
