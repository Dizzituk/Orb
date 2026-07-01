# FILE: app/bridge/sentinel_alerts.py
# Purpose: Phone-facing Sentinel alerts feed — unacknowledged alerts for AstraBridge
#          to poll and raise as local notifications (severe=sound, medium/low=silent).
# Called-by: main (include_router), Astra-Bridge SentinelAlertsWorker
# Depends-on: app.bridge.schemas (require_bridge_auth), app.sentinel.alerts, app.db
# Last-renovated: 2026-07-01
from __future__ import annotations

import logging
import os
from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.sentinel import alerts

from .schemas import require_bridge_auth

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/bridge", tags=["Bridge"])

# Learn mode ending silently pushed every low/medium finding to the phone as a
# notification (see docs/SENTINEL_NOISE_FIX_JOBSPEC.md). The desktop Security
# tab still shows every severity via GET /sentinel/alerts — this floor only
# trims what's worth interrupting Taz's phone for.
_PHONE_MIN_SEVERITY = os.getenv("ASTRA_SENTINEL_PHONE_MIN_SEVERITY", "medium")


class SentinelAlertOut(BaseModel):
    id: int
    severity: str
    title: str
    explanation: str
    created_at: str


@router.get("/sentinel-alerts", response_model=List[SentinelAlertOut])
async def get_sentinel_alerts(
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_bridge_auth),
):
    rows = alerts.list_alerts(db, unacked_only=True, min_severity=_PHONE_MIN_SEVERITY, limit=20)
    return [
        SentinelAlertOut(
            id=a.id,
            severity=a.severity,
            title=a.title,
            explanation=a.explanation or "",
            created_at=a.created_at.isoformat() if a.created_at else "",
        )
        for a in rows
    ]
