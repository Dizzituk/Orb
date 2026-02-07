from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import logging

from app.services.transcription_service import TranscriptionService
from app.services.model_manager import get_model_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/transcribe", tags=["transcribe"])


class TranscriptionResponse(BaseModel):
    """Response model for transcription results."""
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None


class ModelStatusResponse(BaseModel):
    """Response model for model status."""
    loaded: bool
    model_name: Optional[str] = None
    device: Optional[str] = None


class LoadModelRequest(BaseModel):
    """Request model for loading a transcription model."""
    model_name: str = "base"
    device: Optional[str] = None


@router.get("/status", response_model=ModelStatusResponse)
async def get_status():
    """
    Get the current status of the transcription model.
    
    Returns:
        ModelStatusResponse: Current model loading status and configuration
    """
    try:
        model_manager = get_model_manager()
        transcription_service = TranscriptionService.get_instance()
        
        is_loaded = model_manager.is_loaded("transcription")
        model_name = None
        device = None
        
        if is_loaded:
            model_name = transcription_service.model_name
            device = transcription_service.device
        
        return ModelStatusResponse(
            loaded=is_loaded,
            model_name=model_name,
            device=device
        )
    except Exception as e:
        logger.error(f"Error getting transcription status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load")
async def load_model(request: LoadModelRequest):
    """
    Load a transcription model with the specified configuration.
    
    Args:
        request: Model loading configuration
        
    Returns:
        JSON response with loading status
    """
    try:
        transcription_service = TranscriptionService.get_instance()
        
        # Initialize the model with specified configuration
        await transcription_service.initialize(
            model_name=request.model_name,
            device=request.device
        )
        
        return JSONResponse(
            content={
                "status": "success",
                "message": f"Model '{request.model_name}' loaded successfully",
                "model_name": request.model_name,
                "device": transcription_service.device
            }
        )
    except Exception as e:
        logger.error(f"Error loading transcription model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    audio: UploadFile = File(...),
    language: Optional[str] = None
):
    """
    Transcribe an audio file to text.
    
    Args:
        audio: Audio file to transcribe (WAV, MP3, etc.)
        language: Optional language code for transcription
        
    Returns:
        TranscriptionResponse: Transcription result with text and metadata
    """
    try:
        transcription_service = TranscriptionService.get_instance()
        
        # Ensure model is loaded
        if not transcription_service.is_initialized():
            await transcription_service.initialize()
        
        # Read audio file content
        audio_data = await audio.read()
        
        if not audio_data:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Perform transcription
        result = await transcription_service.transcribe(
            audio_data=audio_data,
            language=language
        )
        
        return TranscriptionResponse(
            text=result["text"],
            language=result.get("language"),
            duration=result.get("duration")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error transcribing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unload")
async def unload_model():
    """
    Unload the transcription model to free resources.
    
    Returns:
        JSON response with unload status
    """
    try:
        model_manager = get_model_manager()
        model_manager.unload_model("transcription")
        
        return JSONResponse(
            content={
                "status": "success",
                "message": "Transcription model unloaded successfully"
            }
        )
    except Exception as e:
        logger.error(f"Error unloading transcription model: {e}")
        raise HTTPException(status_code=500, detail=str(e))