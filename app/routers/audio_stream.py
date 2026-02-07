from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
import asyncio
import logging
import numpy as np
from typing import Optional
import json

from app.services.wake_word_service import WakeWordService
from app.services.vad_service import VADService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audio", tags=["audio"])


class AudioStreamHandler:
    """Handles WebSocket audio streaming for wake word detection and VAD."""
    
    def __init__(self):
        self.wake_word_service: Optional[WakeWordService] = None
        self.vad_service: Optional[VADService] = None
        self.sample_rate = 16000
        
    async def initialize_services(self):
        """Initialize wake word and VAD services."""
        try:
            if self.wake_word_service is None:
                self.wake_word_service = WakeWordService()
                await asyncio.to_thread(self.wake_word_service.initialize)
                logger.info("Wake word service initialized")
                
            if self.vad_service is None:
                self.vad_service = VADService()
                await asyncio.to_thread(self.vad_service.initialize)
                logger.info("VAD service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            raise
    
    async def process_audio_chunk(self, audio_data: bytes) -> dict:
        """Process audio chunk through wake word detection and VAD.
        
        Args:
            audio_data: Raw audio bytes (PCM 16-bit, mono, 16kHz)
            
        Returns:
            dict with detection results
        """
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            results = {
                "wake_word_detected": False,
                "voice_detected": False,
                "confidence": 0.0,
                "vad_probability": 0.0
            }
            
            # Run wake word detection
            if self.wake_word_service:
                wake_word_result = await asyncio.to_thread(
                    self.wake_word_service.detect,
                    audio_float
                )
                results["wake_word_detected"] = wake_word_result["detected"]
                results["confidence"] = wake_word_result["confidence"]
            
            # Run VAD
            if self.vad_service:
                vad_result = await asyncio.to_thread(
                    self.vad_service.is_speech,
                    audio_float
                )
                results["voice_detected"] = vad_result["is_speech"]
                results["vad_probability"] = vad_result["probability"]
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return {
                "error": str(e),
                "wake_word_detected": False,
                "voice_detected": False,
                "confidence": 0.0,
                "vad_probability": 0.0
            }


@router.websocket("/stream")
async def audio_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time audio streaming.
    
    Receives audio chunks and returns wake word detection and VAD results.
    
    Expected message format (JSON):
    {
        "type": "audio",
        "data": "<base64-encoded-audio>"
    }
    
    Response format (JSON):
    {
        "type": "detection",
        "wake_word_detected": bool,
        "voice_detected": bool,
        "confidence": float,
        "vad_probability": float
    }
    """
    await websocket.accept()
    handler = AudioStreamHandler()
    
    try:
        # Initialize services
        await handler.initialize_services()
        
        # Send ready message
        await websocket.send_json({
            "type": "ready",
            "message": "Audio stream ready"
        })
        
        while True:
            try:
                # Receive message
                message = await websocket.receive()
                
                # Handle different message types
                if "text" in message:
                    data = json.loads(message["text"])
                    
                    if data.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                        continue
                    
                    if data.get("type") == "close":
                        break
                        
                elif "bytes" in message:
                    # Process audio data
                    audio_data = message["bytes"]
                    results = await handler.process_audio_chunk(audio_data)
                    
                    # Send results back
                    await websocket.send_json({
                        "type": "detection",
                        **results
                    })
                else:
                    logger.warning(f"Received unknown message type: {message}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid JSON format"
                })
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json({
                        "type": "error",
                        "error": str(e)
                    })
                    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Cleanup
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()


@router.get("/stream/status")
async def stream_status():
    """Get audio stream service status.
    
    Returns:
        dict: Service status information
    """
    handler = AudioStreamHandler()
    
    try:
        await handler.initialize_services()
        
        wake_word_ready = handler.wake_word_service is not None
        vad_ready = handler.vad_service is not None
        
        return {
            "status": "ready" if (wake_word_ready and vad_ready) else "initializing",
            "wake_word_service": "ready" if wake_word_ready else "not_initialized",
            "vad_service": "ready" if vad_ready else "not_initialized",
            "sample_rate": handler.sample_rate
        }
    except Exception as e:
        logger.error(f"Error checking stream status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "wake_word_service": "error",
            "vad_service": "error",
            "sample_rate": handler.sample_rate
        }