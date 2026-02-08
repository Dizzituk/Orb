# FILE: app/llm/telemetry_router.py
"""Telemetry endpoints backed by app.llm.audit_logger.

These endpoints are meant for *live health checking* (polling-friendly).
They intentionally do NOT expose prompts/outputs or any sensitive payloads.

Routes
- GET /telemetry/health  -> aggregate metrics
- GET /telemetry/recent?limit=50 -> recent sanitized events
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.auth import require_auth
from app.auth.middleware import AuthResult
from app.llm.audit_logger import get_audit_logger

router = APIRouter(prefix="/telemetry", tags=["telemetry"])


@router.get("/health")
def telemetry_health(auth: AuthResult = Depends(require_auth)):
    logger = get_audit_logger()
    if not logger or not logger.enabled:
        return {"ok": True, "enabled": False, "message": "audit/telemetry disabled"}
    return {"ok": True, "enabled": True, **logger.get_metrics()}


@router.get("/recent")
def telemetry_recent(
    limit: int = Query(default=50, ge=1, le=500),
    auth: AuthResult = Depends(require_auth),
):
    logger = get_audit_logger()
    if not logger or not logger.enabled:
        return {"ok": True, "enabled": False, "events": []}
    return {"ok": True, "enabled": True, "events": logger.get_recent_events(limit=limit)}
