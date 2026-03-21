# FILE: app/content/video_pipeline/router.py
"""
FastAPI endpoints for the Script-to-Video Pipeline.

Provides:
- POST /generate — trigger pipeline from script
- GET /jobs — list pipeline jobs
- GET /jobs/{id} — get job status
- POST /scan-local — trigger local asset scan
- POST /extract-style — extract style from reference video
"""
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth
from app.content.video_pipeline.models import PipelineJobRequest

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/content/video-pipeline",
    tags=["Video Pipeline"],
    dependencies=[Depends(require_auth)],
)

# In-memory job tracking (jobs also persisted to disk)
_active_jobs = {}


class GenerateResponse(BaseModel):
    job_id: str
    status: str
    message: str


class ScanRequest(BaseModel):
    force_rescan: bool = False


class StyleExtractRequest(BaseModel):
    video_path: str  # Local file path OR YouTube URL
    profile_id: str
    context: str = ""


# ═══════════════════════════════════════════════════
# PIPELINE ENDPOINTS
# ═══════════════════════════════════════════════════

@router.post("/generate", response_model=GenerateResponse)
async def generate_video(
    request: PipelineJobRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Start the script-to-video pipeline.
    Runs in background, emits SSE events for progress tracking.
    """
    from app.content.video_pipeline.orchestrator import run_pipeline

    async def _run():
        job = await run_pipeline(db, request)
        _active_jobs[job.job_id] = job

    background_tasks.add_task(_run)

    return GenerateResponse(
        job_id="pending",
        status="queued",
        message="Pipeline job queued. Check /jobs for status.",
    )


@router.post("/generate-stream")
async def generate_video_stream(
    request: PipelineJobRequest,
    db: Session = Depends(get_db),
):
    """
    Start pipeline with SSE streaming for real-time progress.
    Frontend subscribes to this endpoint for live updates.
    """
    import asyncio
    from app.content.video_pipeline.orchestrator import run_pipeline
    from app.content.video_pipeline.models import PipelineStageUpdate

    event_queue = asyncio.Queue()

    async def event_callback(event: PipelineStageUpdate):
        await event_queue.put(event)

    async def run_and_signal():
        try:
            job = await run_pipeline(db, request, event_callback)
            _active_jobs[job.job_id] = job
        finally:
            await event_queue.put(None)  # Signal stream end

    asyncio.create_task(run_and_signal())

    async def event_generator():
        while True:
            event = await event_queue.get()
            if event is None:
                break
            data = event.model_dump_json()
            yield f"data: {data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/jobs")
def list_jobs():
    """List all tracked pipeline jobs."""
    from pathlib import Path
    jobs_dir = Path("data/content/video_pipeline/jobs")
    if not jobs_dir.exists():
        return {"jobs": []}

    jobs = []
    for job_dir in sorted(jobs_dir.iterdir(), reverse=True):
        state_file = job_dir / "state.json"
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
                jobs.append({
                    "job_id": state.get("job_id"),
                    "status": state.get("status"),
                    "title": state.get("request", {}).get("title", ""),
                    "created_at": state.get("created_at"),
                    "total_cost_usd": state.get("total_cost_usd", 0),
                    "output_path": state.get("output_path"),
                })
            except Exception:
                pass

    return {"jobs": jobs}


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    """Get full status for a specific pipeline job."""
    from pathlib import Path
    state_file = Path(f"data/content/video_pipeline/jobs/{job_id}/state.json")
    if not state_file.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    state = json.loads(state_file.read_text(encoding="utf-8"))
    return state


# ═══════════════════════════════════════════════════
# LOCAL SCANNER
# ═══════════════════════════════════════════════════

@router.post("/scan-local")
async def scan_local_assets(body: ScanRequest):
    """Trigger a scan of local stock footage directories."""
    from app.content.video_pipeline.local_scanner import scan_and_index
    result = await scan_and_index(force_rescan=body.force_rescan)
    return result


@router.get("/local-index")
def get_local_index():
    """Get the current local asset index."""
    from app.content.video_pipeline.local_scanner import load_metadata_index
    return load_metadata_index()


# ═══════════════════════════════════════════════════
# STYLE EXTRACTION
# ═══════════════════════════════════════════════════

@router.post("/extract-style")
async def extract_style(
    body: StyleExtractRequest,
    db: Session = Depends(get_db),
):
    """
    Extract style parameters from a reference video.

    video_path accepts either:
    - A YouTube URL (e.g. https://www.youtube.com/watch?v=abc123)
      Gemini analyses the video directly via Google ecosystem, no download.
    - A local file path (e.g. D:/Videos/reference.mp4)
      Uploaded to Gemini via File API for analysis.

    Only public YouTube videos are supported.
    """
    from app.content.video_pipeline.style_resolver import (
        extract_style_from_video,
    )

    profile = await extract_style_from_video(
        db=db,
        video_path=body.video_path,
        profile_id=body.profile_id,
        context=body.context,
    )
    return profile.model_dump()


@router.get("/heygen-voices")
async def list_heygen_voices():
    """
    List all voices available in your HeyGen account.
    Use this to find the right voice_id for your ASTRA avatar.
    """
    from app.content.video_pipeline.heygen_client import list_voices
    voices = await list_voices()
    return {
        "voices": [
            {
                "voice_id": v.get("voice_id", ""),
                "name": v.get("name", v.get("display_name", "")),
                "language": v.get("language", ""),
                "gender": v.get("gender", ""),
                "preview_audio": v.get("preview_audio", ""),
                "support_pause": v.get("support_pause", False),
            }
            for v in voices
        ]
    }


@router.get("/heygen-voices")
async def list_heygen_voices():
    """
    List all voices available in your HeyGen account.
    Use this to find voice_id for the ASTRA avatar.
    """
    import httpx
    import os
    key = os.getenv("HEYGEN_API_KEY", "")
    if not key:
        return {"error": "HEYGEN_API_KEY not set"}
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            "https://api.heygen.com/v2/voices",
            headers={"X-Api-Key": key, "Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
    voices = data.get("data", {}).get("voices", [])
    return {
        "voices": [
            {
                "voice_id": v.get("voice_id", ""),
                "name": v.get("name", ""),
                "language": v.get("language", ""),
                "gender": v.get("gender", ""),
                "preview_audio": v.get("preview_audio", ""),
            }
            for v in voices
        ]
    }


@router.get("/heygen-avatars")
async def list_heygen_avatars():
    """
    List all avatars available in your HeyGen account.
    Use this to find your avatar_id, then paste it into
    Settings > API Keys > HeyGen Avatar ID.
    """
    from app.content.video_pipeline.heygen_client import list_avatars
    avatars = await list_avatars()
    return {
        "avatars": [
            {
                "avatar_id": a.get("avatar_id", ""),
                "avatar_name": a.get("avatar_name", ""),
                "preview_image_url": a.get("preview_image_url", ""),
                "gender": a.get("gender", ""),
            }
            for a in avatars
        ]
    }


@router.get("/styles")
def list_styles(db: Session = Depends(get_db)):
    """List all stored style profiles."""
    from app.content.video_pipeline.style_resolver import StyleProfileRecord
    records = db.query(StyleProfileRecord).all()
    return {
        "profiles": [
            {
                "profile_id": r.profile_id,
                "source_filename": r.source_filename,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ]
    }
