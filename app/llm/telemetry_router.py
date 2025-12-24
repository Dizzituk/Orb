# FILE: app/llm/telemetry_router.py
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.auth import AuthResult, require_auth

# Audit logger lives alongside this router in app/llm/audit_logger.py
# Use a relative import so it works regardless of project root / PYTHONPATH.
from .audit_logger import get_audit_logger


router = APIRouter(prefix="/telemetry", tags=["telemetry"])


class TelemetryHealthResponse(BaseModel):
    ok: bool = True
    enabled: bool = True
    counts: Dict[str, int] = Field(default_factory=dict)
    by_provider: Dict[str, Any] = Field(default_factory=dict)
    avg_latency_ms: float = 0.0
    in_memory_events: int = 0


class TelemetryRecentResponse(BaseModel):
    ok: bool = True
    enabled: bool = True
    events: List[Dict[str, Any]] = Field(default_factory=list)


@router.get("/health", response_model=TelemetryHealthResponse)
def telemetry_health(auth: AuthResult = Depends(require_auth)) -> TelemetryHealthResponse:
    log = get_audit_logger()
    if log is None:
        return TelemetryHealthResponse(ok=True, enabled=False)

    try:
        snap = log.get_metrics()
    except Exception:
        # Telemetry must never crash the API.
        return TelemetryHealthResponse(ok=True, enabled=True)

    return TelemetryHealthResponse(
        ok=bool(snap.get("ok", True)),
        enabled=True,
        counts=snap.get("counts", {}) or {},
        by_provider=snap.get("by_provider", {}) or {},
        avg_latency_ms=float(snap.get("avg_latency_ms", 0.0) or 0.0),
        in_memory_events=int(snap.get("in_memory_events", 0) or 0),
    )


@router.get("/recent", response_model=TelemetryRecentResponse)
def telemetry_recent(
    limit: int = Query(50, ge=1, le=500),
    auth: AuthResult = Depends(require_auth),
) -> TelemetryRecentResponse:
    log = get_audit_logger()
    if log is None:
        return TelemetryRecentResponse(ok=True, enabled=False, events=[])

    try:
        events = log.get_recent_events(limit=limit) or []
    except Exception:
        events = []

    return TelemetryRecentResponse(ok=True, enabled=True, events=events)
