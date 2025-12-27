# FILE: app/introspection/router.py
"""
Introspection Router - Read-Only Log Query Endpoints

Internal API endpoints for log introspection:
- GET /introspection/logs/last - Last completed job
- GET /introspection/logs - Time-based query
- GET /introspection/logs/{job_id} - Specific job

These endpoints are for orchestration/chat layer use only - not public.
All operations are strictly read-only.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.introspection.schemas import (
    JobLogBundle,
    LogQueryResponse,
    LogRequestType,
)
from app.introspection.service import (
    get_job_logs,
    get_jobs_in_range,
    get_last_job_logs,
    get_time_window_for_description,
)
from app.introspection.summarizer import summarize_logs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/introspection", tags=["introspection"])


# =============================================================================
# Helper
# =============================================================================

def _bundles_to_compact(bundles: list[JobLogBundle]) -> list[dict]:
    """Convert bundles to compact dict format for response."""
    result = []
    for b in bundles:
        result.append({
            "job_id": b.job_id,
            "job_type": b.job_type,
            "state": b.state,
            "created_at": b.created_at.isoformat() if b.created_at else None,
            "completed_at": b.completed_at.isoformat() if b.completed_at else None,
            "event_count": len(b.events),
            "spec_hash_verified": b.spec_hash_verified,
            "spec_hash_mismatch": b.spec_hash_mismatch,
            "events": [
                {
                    "ts": e.timestamp.isoformat(),
                    "event": e.event_type,
                    "stage": e.stage_name,
                    "status": e.status,
                    "verified": e.verified,
                    "error": e.error,
                }
                for e in b.events
            ],
        })
    return result


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/logs/last", response_model=LogQueryResponse)
async def get_last_job_logs_endpoint(
    db: Session = Depends(get_db),
):
    """
    Get logs for the most recently completed job.
    
    Returns plain-English summary plus structured log data.
    """
    start_ms = int(time.time() * 1000)
    
    try:
        result = get_last_job_logs(db)
        
        if result.error:
            return LogQueryResponse(
                request_type=LogRequestType.LAST_JOB,
                summary=result.error,
                job_bundles=[],
                total_events=0,
                query_time_ms=int(time.time() * 1000) - start_ms,
                error=result.error,
            )
        
        # Generate summary
        summary = await summarize_logs(result.bundles)
        
        return LogQueryResponse(
            request_type=LogRequestType.LAST_JOB,
            summary=summary,
            job_bundles=result.bundles,
            total_events=result.total_events,
            query_time_ms=int(time.time() * 1000) - start_ms,
        )
        
    except Exception as e:
        logger.exception("[introspection] Error in get_last_job_logs_endpoint")
        return LogQueryResponse(
            request_type=LogRequestType.LAST_JOB,
            summary=f"Error retrieving logs: {e}",
            job_bundles=[],
            total_events=0,
            query_time_ms=int(time.time() * 1000) - start_ms,
            error=str(e),
        )


@router.get("/logs", response_model=LogQueryResponse)
async def get_logs_in_range_endpoint(
    from_time: Optional[str] = Query(None, alias="from", description="Start time (ISO format)"),
    to_time: Optional[str] = Query(None, alias="to", description="End time (ISO format)"),
    window: Optional[str] = Query(None, description="Time window description (e.g., 'today', 'last hour', 'this week')"),
    db: Session = Depends(get_db),
):
    """
    Get logs for jobs in a time range.
    
    Either provide `from` and `to` ISO timestamps, or use `window` for
    natural language like "today", "last hour", "this week".
    
    Returns plain-English summary plus structured log data.
    """
    start_ms = int(time.time() * 1000)
    
    try:
        # Determine time range
        if window:
            start, end = get_time_window_for_description(window)
        elif from_time and to_time:
            try:
                start = datetime.fromisoformat(from_time.replace("Z", "+00:00"))
                end = datetime.fromisoformat(to_time.replace("Z", "+00:00"))
            except ValueError as e:
                return LogQueryResponse(
                    request_type=LogRequestType.TIME_WINDOW,
                    summary=f"Invalid time format: {e}",
                    job_bundles=[],
                    total_events=0,
                    query_time_ms=int(time.time() * 1000) - start_ms,
                    error=f"Invalid time format: {e}",
                )
        else:
            # Default: last hour
            now = datetime.now(timezone.utc)
            start = now - timedelta(hours=1)
            end = now
        
        result = get_jobs_in_range(db, start, end)
        
        if result.error:
            return LogQueryResponse(
                request_type=LogRequestType.TIME_WINDOW,
                summary=result.error,
                job_bundles=[],
                total_events=0,
                query_time_ms=int(time.time() * 1000) - start_ms,
                error=result.error,
            )
        
        # Generate summary
        summary = await summarize_logs(result.bundles)
        
        return LogQueryResponse(
            request_type=LogRequestType.TIME_WINDOW,
            summary=summary,
            job_bundles=result.bundles,
            total_events=result.total_events,
            query_time_ms=int(time.time() * 1000) - start_ms,
        )
        
    except Exception as e:
        logger.exception("[introspection] Error in get_logs_in_range_endpoint")
        return LogQueryResponse(
            request_type=LogRequestType.TIME_WINDOW,
            summary=f"Error retrieving logs: {e}",
            job_bundles=[],
            total_events=0,
            query_time_ms=int(time.time() * 1000) - start_ms,
            error=str(e),
        )


@router.get("/logs/{job_id}", response_model=LogQueryResponse)
async def get_job_logs_endpoint(
    job_id: str,
    db: Session = Depends(get_db),
):
    """
    Get logs for a specific job by ID.
    
    Returns plain-English summary plus structured log data.
    """
    start_ms = int(time.time() * 1000)
    
    try:
        result = get_job_logs(db, job_id)
        
        if result.error:
            return LogQueryResponse(
                request_type=LogRequestType.JOB_ID,
                summary=result.error,
                job_bundles=[],
                total_events=0,
                query_time_ms=int(time.time() * 1000) - start_ms,
                error=result.error,
            )
        
        # Generate summary
        summary = await summarize_logs(result.bundles)
        
        return LogQueryResponse(
            request_type=LogRequestType.JOB_ID,
            summary=summary,
            job_bundles=result.bundles,
            total_events=result.total_events,
            query_time_ms=int(time.time() * 1000) - start_ms,
        )
        
    except Exception as e:
        logger.exception("[introspection] Error in get_job_logs_endpoint")
        return LogQueryResponse(
            request_type=LogRequestType.JOB_ID,
            summary=f"Error retrieving logs: {e}",
            job_bundles=[],
            total_events=0,
            query_time_ms=int(time.time() * 1000) - start_ms,
            error=str(e),
        )


__all__ = ["router"]
