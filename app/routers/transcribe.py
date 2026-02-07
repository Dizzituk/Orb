"""
FastAPI router for voice transcription endpoints.

Endpoints:
- GET  /model/status    — check if model is loaded
- POST /model/load      — load the whisper model
- POST /model/unload    — unload model from memory
- POST /transcribe      — transcribe audio file

All heavy work runs in a thread pool so it never blocks the async event loop.
"""
import asyncio
import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request

from app.services.model_manager import get_model_manager
from app.services.faster_whisper_service import FasterWhisperService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["transcription"])

# Module-level service instance (lazy — doesn't load model until first use)
_service = FasterWhisperService()


@router.get("/model/status")
async def get_model_status():
    """Get current transcription model status."""
    return get_model_manager().get_status()


@router.post("/model/load")
async def load_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
):
    """Load the transcription model in a background thread."""
    try:
        mm = get_model_manager()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, lambda: mm.load_model(model_name=model_name, device=device, compute_type=compute_type)
        )
        return mm.get_status()
    except Exception as e:
        logger.error("[transcribe_router] Failed to load model: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/model/unload")
async def unload_model():
    """Unload the transcription model from memory."""
    try:
        get_model_manager().unload_model()
        return {"status": "unloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcribe")
async def transcribe_audio(
    request: Request,
    file: Optional[UploadFile] = File(None),
    language: Optional[str] = Form(None),
    vad_filter: bool = Form(True),
    noise_sensitivity: float = Form(0.5),
):
    """Transcribe audio. Runs model inference in a thread pool."""
    try:
        # Get audio bytes
        if file:
            audio_bytes = await file.read()
        else:
            audio_bytes = await request.body()

        if not audio_bytes or len(audio_bytes) < 100:
            raise HTTPException(status_code=400, detail="No audio data or too short")

        # Map noise sensitivity to VAD params
        vad_params = {
            "threshold": 0.3 + (noise_sensitivity * 0.4),
            "min_speech_duration_ms": int(200 + (noise_sensitivity * 300)),
            "min_silence_duration_ms": int(400 + (noise_sensitivity * 400)),
        }

        t0 = time.time()

        # Run transcription in thread pool so we don't block the event loop
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: _service.transcribe(
                audio_bytes=audio_bytes,
                language=language,
                vad_filter=vad_filter,
                vad_parameters=vad_params,
            ),
        )
        elapsed = time.time() - t0

        logger.info("[transcribe_router] Done in %.1fs: '%s'", elapsed, result.text[:80])

        return {
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
            "processing_time": round(elapsed, 2),
            "segments": [
                {"text": s.text, "start": s.start, "end": s.end}
                for s in result.segments
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[transcribe_router] Transcription failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
