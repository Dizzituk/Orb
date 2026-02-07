"""
WebSocket router for continuous wake-word detection and streaming transcription.

This module provides real-time audio streaming capabilities for:
1. Wake-word detection using OpenWakeWord
2. Voice Activity Detection (VAD)
3. Streaming transcription with faster-whisper

The WebSocket endpoint accepts audio chunks, processes them for wake-word
detection and VAD, and returns transcription results when speech is detected.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Optional, Dict, Any
import asyncio
import io
import logging
import numpy as np
import wave
from pydantic import BaseModel

from app.services.wake_word_service import get_wake_word_service
from app.services.model_manager import get_model_manager
from app.services.faster_whisper_service import FasterWhisperService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audio-stream", tags=["audio-stream"])


class AudioStreamConfig(BaseModel):
    """Configuration for audio streaming session."""
    sample_rate: int = 16000
    channels: int = 1
    wake_word_enabled: bool = True
    vad_enabled: bool = True
    vad_threshold: float = 0.5


class AudioStreamMessage(BaseModel):
    """Message sent from client to server."""
    type: str  # "audio", "config", "ping"
    data: Optional[str] = None  # Base64 encoded audio data
    config: Optional[AudioStreamConfig] = None


class TranscriptionMessage(BaseModel):
    """Message sent from server to client with transcription result."""
    type: str = "transcription"
    text: str
    is_final: bool = False
    confidence: Optional[float] = None
    segments: Optional[list] = None


class WakeWordMessage(BaseModel):
    """Message sent from server to client when wake word is detected."""
    type: str = "wake_word"
    detected: bool
    confidence: float


class StatusMessage(BaseModel):
    """Status message sent from server to client."""
    type: str = "status"
    status: str
    message: Optional[str] = None


class ErrorMessage(BaseModel):
    """Error message sent from server to client."""
    type: str = "error"
    error: str
    message: str


class AudioStreamSession:
    """Manages a single audio streaming session."""
    
    def __init__(self, websocket: WebSocket, config: AudioStreamConfig):
        self.websocket = websocket
        self.config = config
        self.wake_word_service = get_wake_word_service()
        self.model_manager = get_model_manager()
        self.transcription_service: Optional[FasterWhisperService] = None
        self.audio_buffer: list = []
        self.is_recording = False
        self.last_speech_time: Optional[float] = None
        self.silence_duration = 1.0  # seconds of silence before finalizing
        
    async def initialize(self):
        """Initialize the session and load required models."""
        try:
            # Ensure transcription model is loaded
            if not self.model_manager.is_model_loaded():
                await self.send_status("loading", "Loading transcription model...")
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.model_manager.load_model,
                    "base.en"
                )
            
            # Get transcription service
            self.transcription_service = FasterWhisperService()
            
            await self.send_status("ready", "Session initialized")
            logger.info("Audio stream session initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize session: {e}")
            await self.send_error("initialization_failed", str(e))
            raise
    
    async def process_audio_chunk(self, audio_data: bytes):
        """Process incoming audio chunk for wake-word and VAD."""
        try:
            # Convert audio bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Wake-word detection (if enabled and not already recording)
            if self.config.wake_word_enabled and not self.is_recording:
                wake_word_result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.wake_word_service.process_audio,
                    audio_float
                )
                
                if wake_word_result["detected"]:
                    await self.send_wake_word(True, wake_word_result["confidence"])
                    self.is_recording = True
                    self.audio_buffer = []
                    await self.send_status("recording", "Wake word detected, recording...")
            
            # VAD and buffering (if recording)
            if self.is_recording:
                self.audio_buffer.append(audio_float)
                
                if self.config.vad_enabled:
                    vad_result = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.wake_word_service.detect_speech,
                        audio_float
                    )
                    
                    if vad_result["speech_detected"]:
                        self.last_speech_time = asyncio.get_event_loop().time()
                    else:
                        # Check if silence duration exceeded
                        if self.last_speech_time is not None:
                            current_time = asyncio.get_event_loop().time()
                            silence = current_time - self.last_speech_time
                            
                            if silence >= self.silence_duration:
                                # Finalize and transcribe
                                await self.finalize_recording()
                
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            await self.send_error("processing_failed", str(e))
    
    async def finalize_recording(self):
        """Finalize recording and transcribe buffered audio."""
        if not self.audio_buffer:
            self.is_recording = False
            return
        
        try:
            await self.send_status("transcribing", "Transcribing audio...")
            
            # Concatenate audio buffer
            audio_data = np.concatenate(self.audio_buffer)
            
            # Convert to WAV bytes
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(self.config.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.config.sample_rate)
                audio_int16 = (audio_data * 32768.0).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())
            
            wav_bytes = wav_buffer.getvalue()
            
            # Transcribe
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.transcription_service.transcribe,
                wav_bytes,
                None  # language (auto-detect)
            )
            
            # Send transcription result
            await self.send_transcription(
                text=result.text,
                is_final=True,
                confidence=result.confidence,
                segments=[
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                        "confidence": seg.confidence
                    }
                    for seg in result.segments
                ]
            )
            
            # Reset state
            self.is_recording = False
            self.audio_buffer = []
            self.last_speech_time = None
            
            await self.send_status("ready", "Ready for next input")
            
        except Exception as e:
            logger.error(f"Error finalizing recording: {e}")
            await self.send_error("transcription_failed", str(e))
            self.is_recording = False
            self.audio_buffer = []
    
    async def send_transcription(
        self,
        text: str,
        is_final: bool = False,
        confidence: Optional[float] = None,
        segments: Optional[list] = None
    ):
        """Send transcription result to client."""
        message = TranscriptionMessage(
            text=text,
            is_final=is_final,
            confidence=confidence,
            segments=segments
        )
        await self.websocket.send_json(message.dict())
    
    async def send_wake_word(self, detected: bool, confidence: float):
        """Send wake word detection result to client."""
        message = WakeWordMessage(detected=detected, confidence=confidence)
        await self.websocket.send_json(message.dict())
    
    async def send_status(self, status: str, message: Optional[str] = None):
        """Send status message to client."""
        msg = StatusMessage(status=status, message=message)
        await self.websocket.send_json(msg.dict())
    
    async def send_error(self, error: str, message: str):
        """Send error message to client."""
        msg = ErrorMessage(error=error, message=message)
        await self.websocket.send_json(msg.dict())


@router.websocket("/ws")
async def audio_stream_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for continuous audio streaming.
    
    Protocol:
    1. Client connects and sends config message
    2. Server initializes and sends ready status
    3. Client streams audio chunks
    4. Server processes audio for wake-word, VAD, and transcription
    5. Server sends transcription results when complete
    
    Message types from client:
    - config: Set session configuration
    - audio: Audio data chunk (base64 encoded)
    - ping: Keep-alive ping
    
    Message types from server:
    - status: Session status updates
    - wake_word: Wake word detection result
    - transcription: Transcription result
    - error: Error message
    """
    await websocket.accept()
    logger.info("Audio stream WebSocket connection accepted")
    
    session: Optional[AudioStreamSession] = None
    
    try:
        # Wait for initial config message
        data = await websocket.receive_json()
        
        if data.get("type") != "config":
            await websocket.send_json({
                "type": "error",
                "error": "invalid_message",
                "message": "First message must be config"
            })
            await websocket.close()
            return
        
        # Parse config
        config_data = data.get("config", {})
        config = AudioStreamConfig(**config_data)
        
        # Create and initialize session
        session = AudioStreamSession(websocket, config)
        await session.initialize()
        
        # Main message loop
        while True:
            data = await websocket.receive()
            
            if "text" in data:
                # JSON message
                message = data["text"]
                import json
                msg_data = json.loads(message)
                
                msg_type = msg_data.get("type")
                
                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                    
                elif msg_type == "config":
                    # Update config
                    config_data = msg_data.get("config", {})
                    session.config = AudioStreamConfig(**config_data)
                    await session.send_status("config_updated", "Configuration updated")
                    
                elif msg_type == "audio":
                    # Base64 encoded audio data
                    import base64
                    audio_data = base64.b64decode(msg_data.get("data", ""))
                    await session.process_audio_chunk(audio_data)
                    
            elif "bytes" in data:
                # Binary audio data
                audio_data = data["bytes"]
                await session.process_audio_chunk(audio_data)
            
    except WebSocketDisconnect:
        logger.info("Audio stream WebSocket disconnected")
        
    except Exception as e:
        logger.error(f"Error in audio stream WebSocket: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": "internal_error",
                "message": str(e)
            })
        except:
            pass
        
    finally:
        # Cleanup
        if session and session.is_recording:
            try:
                await session.finalize_recording()
            except:
                pass
        
        try:
            await websocket.close()
        except:
            pass
        
        logger.info("Audio stream session closed")