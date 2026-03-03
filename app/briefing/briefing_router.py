# FILE: app/briefing/briefing_router.py
"""
Briefing Router — FastAPI endpoints for the Morning Briefing System.

Endpoints:
  POST /briefing/generate          — Trigger on-demand briefing generation
  GET  /briefing/latest            — Get the most recent briefing
  GET  /briefing/history           — Get briefing history
  GET  /briefing/audio/{id}        — Stream briefing audio file
  GET  /briefing/text/{id}         — Get briefing text digest
  POST /briefing/scheduler/start   — Start background scheduler
  POST /briefing/scheduler/stop    — Stop background scheduler
  GET  /briefing/config            — Get current briefing configuration

v1.0 (2026-03): Initial implementation.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.auth import optional_auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/briefing", tags=["Briefing"])


# =========================================================================
# Request / response models
# =========================================================================

class GenerateRequest(BaseModel):
    frequency: str = "daily"     # "daily" or "weekly"


class BriefingResponse(BaseModel):
    ok: bool
    briefing_id: str = ""
    title: str = ""
    total_items: int = 0
    text_digest_path: str = ""
    audio_path: str = ""
    astra_alerts: list = []
    generated_at: str = ""
    error: str = ""


# =========================================================================
# Endpoints
# =========================================================================

@router.post("/generate", response_model=BriefingResponse)
async def generate_briefing_endpoint(
    req: GenerateRequest,
    auth=Depends(optional_auth),
) -> BriefingResponse:
    """Trigger on-demand briefing generation."""
    if req.frequency not in ("daily", "weekly"):
        raise HTTPException(400, "frequency must be 'daily' or 'weekly'")

    try:
        from app.briefing.briefing_scheduler import generate_briefing
        record = await generate_briefing(frequency=req.frequency)
        return BriefingResponse(
            ok=True,
            briefing_id=record.id,
            title=record.title,
            total_items=record.total_items,
            text_digest_path=record.text_digest_path,
            audio_path=record.audio_path,
            astra_alerts=record.astra_alerts,
            generated_at=record.generated_at,
        )
    except Exception as e:
        logger.error("[briefing_router] Generation failed: %s", e)
        return BriefingResponse(ok=False, error=str(e))


@router.get("/latest")
async def get_latest_briefing(auth=Depends(optional_auth)):
    """Get the most recent briefing."""
    try:
        from app.briefing.briefing_scheduler import get_latest_briefing
        record = get_latest_briefing()
        if not record:
            return {"ok": True, "briefing": None, "message": "No briefings generated yet"}

        # Read the text digest if path exists
        text_digest = ""
        digest_path = record.get("text_digest_path", "")
        if digest_path and Path(digest_path).exists():
            text_digest = Path(digest_path).read_text(encoding="utf-8")

        return {
            "ok": True,
            "briefing": record,
            "text_digest": text_digest,
        }
    except Exception as e:
        logger.error("[briefing_router] Get latest failed: %s", e)
        return {"ok": False, "error": str(e)}


@router.get("/history")
async def get_briefing_history(
    count: int = Query(10, ge=1, le=30),
    auth=Depends(optional_auth),
):
    """Get briefing history."""
    try:
        from app.briefing.briefing_scheduler import get_recent_briefings
        records = get_recent_briefings(count)
        return {"ok": True, "briefings": records, "count": len(records)}
    except Exception as e:
        logger.error("[briefing_router] History failed: %s", e)
        return {"ok": False, "error": str(e)}


@router.get("/audio/{briefing_id}")
async def get_briefing_audio(briefing_id: str, auth=Depends(optional_auth)):
    """Stream briefing audio file."""
    try:
        from app.briefing.briefing_scheduler import get_recent_briefings
        records = get_recent_briefings(30)
        record = next((r for r in records if r.get("id") == briefing_id), None)

        if not record:
            raise HTTPException(404, "Briefing not found")

        audio_path = record.get("audio_path", "")
        if not audio_path or not Path(audio_path).exists():
            raise HTTPException(404, "Audio not available for this briefing")

        return FileResponse(
            audio_path,
            media_type="audio/mpeg",
            filename=f"briefing_{briefing_id}.mp3",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[briefing_router] Audio retrieval failed: %s", e)
        raise HTTPException(500, str(e))


@router.get("/text/{briefing_id}")
async def get_briefing_text(briefing_id: str, auth=Depends(optional_auth)):
    """Get briefing text digest."""
    try:
        from app.briefing.briefing_scheduler import get_recent_briefings
        records = get_recent_briefings(30)
        record = next((r for r in records if r.get("id") == briefing_id), None)

        if not record:
            raise HTTPException(404, "Briefing not found")

        digest_path = record.get("text_digest_path", "")
        if not digest_path or not Path(digest_path).exists():
            raise HTTPException(404, "Text digest not available")

        text = Path(digest_path).read_text(encoding="utf-8")
        return {"ok": True, "briefing_id": briefing_id, "text_digest": text}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[briefing_router] Text retrieval failed: %s", e)
        raise HTTPException(500, str(e))


@router.post("/scheduler/start")
async def start_scheduler(auth=Depends(optional_auth)):
    """Start the background briefing scheduler."""
    try:
        from app.briefing.briefing_scheduler import start_scheduler
        start_scheduler()
        return {"ok": True, "message": "Scheduler started"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/scheduler/stop")
async def stop_scheduler(auth=Depends(optional_auth)):
    """Stop the background briefing scheduler."""
    try:
        from app.briefing.briefing_scheduler import stop_scheduler
        stop_scheduler()
        return {"ok": True, "message": "Scheduler stopped"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/config")
async def get_config(auth=Depends(optional_auth)):
    """Get current briefing configuration."""
    try:
        from app.briefing.briefing_config import get_topics, get_schedule, get_voice_config
        topics = get_topics()
        schedule = get_schedule()
        voice = get_voice_config()
        return {
            "ok": True,
            "topics": [
                {"name": t.name, "key": t.key, "enabled": t.enabled,
                 "priority": t.priority, "max_stories": t.max_stories,
                 "queries": t.search_queries}
                for t in topics
            ],
            "schedule": {
                "daily_hour": schedule.daily_hour,
                "daily_minute": schedule.daily_minute,
                "weekly_day": schedule.weekly_day,
                "weekly_hour": schedule.weekly_hour,
                "auto_generate": schedule.auto_generate,
                "audio_enabled": schedule.audio_enabled,
            },
            "voice": {
                "headlines": voice.voice_headlines,
                "analysis": voice.voice_analysis,
                "speed": voice.speed,
            },
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


__all__ = ["router"]
