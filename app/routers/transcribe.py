"""
FastAPI router for voice transcription endpoints.

This module provides HTTP endpoints for:
- Model status checking
- Model loading/unloading
- Audio transcription
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from app.services.model_manager import get_model_manager
from app.services.faster_whisper_service import FasterWhisperService
from app.services.transcription_service import TranscriptionResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model", tags=["transcription"])


class ModelStatusResponse(BaseModel):
    """Response model for model status endpoint."""
    loaded: bool = Field(..., description="Whether a transcription model is currently loaded")
    model_name: Optional[str] = Field(None, description="Name of the loaded model")
    device: Optional[str] = Field(None, description="Device the model is running on (cpu/cuda)")
    compute_type: Optional[str] = Field(None, description="Compute type (int8/float16/float32)")


class ModelLoadRequest(BaseModel):
    """Request model for loading a transcription model."""
    model_name: str = Field("base.en", description="Whisper model name to load")
    device: Optional[str] = Field(None, description="Device to load model on (cpu/cuda/auto)")
    compute_type: Optional[str] = Field(None, description="Compute type (int8/float16/float32/auto)")


class ModelLoadResponse(BaseModel):
    """Response model for model loading endpoint."""
    success: bool = Field(..., description="Whether the model was loaded successfully")
    model_name: str = Field(..., description="Name of the loaded model")
    device: str = Field(..., description="Device the model is running on")
    compute_type: str = Field(..., description="Compute type being used")
    message: Optional[str] = Field(None, description="Additional information or error message")


class TranscriptionResponse(BaseModel):
    """Response model for transcription endpoint."""
    text: str = Field(..., description="Full transcribed text")
    language: Optional[str] = Field(None, description="Detected language code")
    segments: list[Dict[str, Any]] = Field(default_factory=list, description="Detailed segment information")
    processing_time: float = Field(..., description="Time taken to process the audio in seconds")


@router.get("/status", response_model=ModelStatusResponse)
async def get_model_status() -> ModelStatusResponse:
    """
    Get the current status of the transcription model.
    
    Returns information about whether a model is loaded and its configuration.
    """
    model_manager = get_model_manager()
    transcription_service = model_manager.get_transcription_service()
    
    if transcription_service is None:
        return ModelStatusResponse(loaded=False)
    
    # Get model info from the service
    if isinstance(transcription_service, FasterWhisperService):
        model_info = transcription_service.get_model_info()
        return ModelStatusResponse(
            loaded=True,
            model_name=model_info.get("model_name"),
            device=model_info.get("device"),
            compute_type=model_info.get("compute_type")
        )
    
    return ModelStatusResponse(loaded=True)


@router.post("/load", response_model=ModelLoadResponse)
async def load_model(request: ModelLoadRequest) -> ModelLoadResponse:
    """
    Load a transcription model with specified configuration.
    
    This endpoint will download the model if not cached and load it into memory.
    Subsequent calls will unload any existing model and load the new one.
    """
    model_manager = get_model_manager()
    
    try:
        logger.info(f"Loading model: {request.model_name} on device: {request.device or 'auto'}")
        
        # Load the model
        transcription_service = await model_manager.load_transcription_model(
            model_name=request.model_name,
            device=request.device,
            compute_type=request.compute_type
        )
        
        # Get model info for response
        if isinstance(transcription_service, FasterWhisperService):
            model_info = transcription_service.get_model_info()
            return ModelLoadResponse(
                success=True,
                model_name=model_info.get("model_name", request.model_name),
                device=model_info.get("device", "unknown"),
                compute_type=model_info.get("compute_type", "unknown"),
                message="Model loaded successfully"
            )
        
        return ModelLoadResponse(
            success=True,
            model_name=request.model_name,
            device=request.device or "auto",
            compute_type=request.compute_type or "auto",
            message="Model loaded successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )


@router.post("/unload")
async def unload_model() -> Dict[str, Any]:
    """
    Unload the currently loaded transcription model from memory.
    
    This frees up GPU/CPU resources.
    """
    model_manager = get_model_manager()
    
    try:
        model_manager.unload_transcription_model()
        logger.info("Model unloaded successfully")
        return {
            "success": True,
            "message": "Model unloaded successfully"
        }
    except Exception as e:
        logger.error(f"Failed to unload model: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to unload model: {str(e)}"
        )


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe (WAV, MP3, etc.)"),
    language: Optional[str] = Form(None, description="Language code (e.g., 'en', 'es'). Auto-detect if not provided."),
    task: str = Form("transcribe", description="Task type: 'transcribe' or 'translate'"),
    temperature: float = Form(0.0, description="Sampling temperature (0.0 = greedy, higher = more random)"),
    vad_filter: bool = Form(True, description="Enable voice activity detection filter"),
    beam_size: int = Form(5, description="Beam size for beam search decoding")
) -> TranscriptionResponse:
    """
    Transcribe an audio file to text.
    
    The model will be automatically loaded if not already in memory.
    Supports various audio formats (WAV, MP3, M4A, etc.).
    """
    model_manager = get_model_manager()
    
    # Get or load transcription service
    transcription_service = model_manager.get_transcription_service()
    if transcription_service is None:
        try:
            logger.info("No model loaded, loading default model")
            transcription_service = await model_manager.load_transcription_model()
        except Exception as e:
            logger.error(f"Failed to load default model: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load transcription model: {str(e)}"
            )
    
    # Read audio file
    try:
        audio_data = await file.read()
        if not audio_data:
            raise HTTPException(status_code=400, detail="Empty audio file")
    except Exception as e:
        logger.error(f"Failed to read audio file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to read audio file: {str(e)}")
    
    # Transcribe
    try:
        logger.info(f"Transcribing audio file: {file.filename}, size: {len(audio_data)} bytes")
        
        result: TranscriptionResult = await transcription_service.transcribe(
            audio_data=audio_data,
            language=language,
            task=task,
            temperature=temperature,
            vad_filter=vad_filter,
            beam_size=beam_size
        )
        
        # Convert segments to dict format
        segments_dict = [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text
            }
            for seg in result.segments
        ]
        
        logger.info(f"Transcription successful: {len(result.text)} chars, {len(segments_dict)} segments")
        
        return TranscriptionResponse(
            text=result.text,
            language=result.language,
            segments=segments_dict,
            processing_time=result.processing_time
        )
        
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )