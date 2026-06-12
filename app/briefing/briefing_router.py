# Purpose: briefing router
# Called-by: main
# Depends-on: app.auth, app.briefing.briefing_config, app.briefing.briefing_scheduler
# Last-renovated: 2026-06-11
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.auth import optional_auth

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/briefing", tags=["Briefing"])


class GenerateRequest(BaseModel):
    frequency: str = "daily"
    profile: str = "default"


class BriefingResponse(BaseModel):
    ok: bool
    briefing_id: str = ""
    title: str = ""
    total_items: int = 0
    text_digest_path: str = ""
    audio_path: str = ""
    astra_alerts: list = []
    generated_at: str = ""
    profile: str = "default"
    structure_path: str = ""
    error: str = ""


@router.post("/generate", response_model=BriefingResponse)
async def generate_briefing_endpoint(req: GenerateRequest, auth=Depends(optional_auth)) -> BriefingResponse:
    if req.frequency not in ("daily", "weekly"):
        raise HTTPException(400, "frequency must be 'daily' or 'weekly'")
    try:
        from app.briefing.briefing_scheduler import generate_briefing
        record = await generate_briefing(frequency=req.frequency, profile=req.profile)
        return BriefingResponse(
            ok=True,
            briefing_id=record.id,
            title=record.title,
            total_items=record.total_items,
            text_digest_path=record.text_digest_path,
            audio_path=record.audio_path,
            astra_alerts=record.astra_alerts,
            generated_at=record.generated_at,
            profile=record.profile,
            structure_path=record.structure_path,
        )
    except Exception as exc:
        logger.error("[briefing_router] Generation failed: %s", exc)
        return BriefingResponse(ok=False, error=str(exc))


@router.get("/latest")
async def get_latest_briefing(profile: str = Query("default"), auth=Depends(optional_auth)):
    try:
        from app.briefing.briefing_scheduler import get_latest_briefing, load_briefing_structure
        record = get_latest_briefing(profile=profile)
        if not record:
            return {"ok": True, "briefing": None, "text_digest": "", "sections": [], "astra_alerts": [], "profile": profile, "message": "No briefings generated yet"}

        text_digest = ""
        digest_path = record.get("text_digest_path", "")
        if digest_path and Path(digest_path).exists():
            text_digest = Path(digest_path).read_text(encoding="utf-8")
        structure = load_briefing_structure(record)
        return {
            "ok": True,
            "briefing": record,
            "text_digest": text_digest,
            "sections": structure.get("sections", []),
            "astra_alerts": structure.get("astra_alerts", record.get("astra_alerts", [])),
            "profile": structure.get("profile", record.get("profile", profile)),
        }
    except Exception as exc:
        logger.error("[briefing_router] Get latest failed: %s", exc)
        return {"ok": False, "error": str(exc)}


@router.get("/history")
async def get_briefing_history(count: int = Query(10, ge=1, le=30), profile: str = Query("default"), auth=Depends(optional_auth)):
    try:
        from app.briefing.briefing_scheduler import get_recent_briefings
        records = get_recent_briefings(count, profile=profile)
        return {"ok": True, "briefings": records, "count": len(records)}
    except Exception as exc:
        logger.error("[briefing_router] History failed: %s", exc)
        return {"ok": False, "error": str(exc)}


@router.get("/audio/{briefing_id}")
async def get_briefing_audio(briefing_id: str, auth=Depends(optional_auth)):
    try:
        from app.briefing.briefing_scheduler import get_recent_briefings
        records = get_recent_briefings(30)
        record = next((r for r in records if r.get("id") == briefing_id), None)
        if not record:
            raise HTTPException(404, "Briefing not found")
        audio_path = record.get("audio_path", "")
        if not audio_path or not Path(audio_path).exists():
            raise HTTPException(404, "Audio not available for this briefing")
        return FileResponse(audio_path, media_type="audio/mpeg", filename=f"briefing_{briefing_id}.mp3")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[briefing_router] Audio retrieval failed: %s", exc)
        raise HTTPException(500, str(exc))


@router.get("/text/{briefing_id}")
async def get_briefing_text(briefing_id: str, auth=Depends(optional_auth)):
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
    except Exception as exc:
        logger.error("[briefing_router] Text retrieval failed: %s", exc)
        raise HTTPException(500, str(exc))


@router.post("/scheduler/start")
async def start_scheduler(auth=Depends(optional_auth)):
    try:
        from app.briefing.briefing_scheduler import start_scheduler
        start_scheduler()
        return {"ok": True, "message": "Scheduler started"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@router.post("/scheduler/stop")
async def stop_scheduler(auth=Depends(optional_auth)):
    try:
        from app.briefing.briefing_scheduler import stop_scheduler
        stop_scheduler()
        return {"ok": True, "message": "Scheduler stopped"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@router.get("/config")
async def get_config(profile: str = Query("default"), auth=Depends(optional_auth)):
    try:
        from app.briefing.briefing_config import get_topics, get_schedule, get_voice_config
        topics = get_topics(profile=profile)
        schedule = get_schedule()
        voice = get_voice_config()
        return {
            "ok": True,
            "profile": profile,
            "topics": [
                {
                    "name": topic.name,
                    "key": topic.key,
                    "enabled": topic.enabled,
                    "priority": topic.priority,
                    "max_stories": topic.max_stories,
                    "queries": topic.search_queries,
                }
                for topic in topics
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
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


__all__ = ["router"]
