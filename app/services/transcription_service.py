"""
Transcription service implementation using Faster-Whisper.

This module provides the core transcription service interface and implementation
for converting audio to text using the Faster-Whisper model.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import tempfile
import os

from faster_whisper import WhisperModel

from app.services.model_manager import get_model_manager

logger = logging.getLogger(__name__)


class TranscriptionService(ABC):
    """Abstract base class for transcription services."""

    @abstractmethod
    async def transcribe(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio data to text.

        Args:
            audio_data: Raw audio bytes
            language: Optional language code (e.g., 'en', 'es')
            task: Either 'transcribe' or 'translate'
            **kwargs: Additional transcription options

        Returns:
            Dictionary containing transcription results with keys:
                - text: Full transcribed text
                - segments: List of segments with timestamps
                - language: Detected or specified language
        """
        pass

    @abstractmethod
    async def is_ready(self) -> bool:
        """Check if the service is ready to transcribe."""
        pass

    @abstractmethod
    async def load_model(self, model_name: str, device: Optional[str] = None) -> None:
        """Load a specific model."""
        pass

    @abstractmethod
    async def unload_model(self) -> None:
        """Unload the current model to free resources."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current service status."""
        pass


class FasterWhisperTranscriptionService(TranscriptionService):
    """
    Transcription service implementation using Faster-Whisper.
    
    This service manages audio transcription using the Faster-Whisper library,
    which provides optimized inference for Whisper models.
    """

    _instance: Optional["FasterWhisperTranscriptionService"] = None

    def __init__(self):
        """Initialize the transcription service."""
        self._model: Optional[WhisperModel] = None
        self._model_name: Optional[str] = None
        self._device: Optional[str] = None
        self._model_manager = get_model_manager()
        logger.info("FasterWhisperTranscriptionService initialized")

    @classmethod
    def get_instance(cls) -> "FasterWhisperTranscriptionService":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def transcribe(
        self,
        audio_data: bytes,
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe audio data to text using Faster-Whisper.

        Args:
            audio_data: Raw audio bytes (WAV, MP3, etc.)
            language: Optional language code (e.g., 'en', 'es')
            task: Either 'transcribe' or 'translate'
            **kwargs: Additional options:
                - beam_size: Beam search size (default: 5)
                - vad_filter: Enable VAD filtering (default: False)
                - temperature: Sampling temperature (default: 0.0)
                - word_timestamps: Include word-level timestamps (default: False)

        Returns:
            Dictionary with transcription results:
                - text: Full transcribed text
                - segments: List of segments with timing info
                - language: Detected or specified language
                - duration: Audio duration in seconds

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If audio data is invalid
        """
        if not await self.is_ready():
            raise RuntimeError("Transcription model is not loaded")

        if not audio_data:
            raise ValueError("Audio data is empty")

        # Write audio data to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_data)
            temp_path = temp_file.name

        try:
            # Extract transcription options
            beam_size = kwargs.get("beam_size", 5)
            vad_filter = kwargs.get("vad_filter", False)
            temperature = kwargs.get("temperature", 0.0)
            word_timestamps = kwargs.get("word_timestamps", False)

            # Perform transcription
            segments_iter, info = self._model.transcribe(
                temp_path,
                language=language,
                task=task,
                beam_size=beam_size,
                vad_filter=vad_filter,
                temperature=temperature,
                word_timestamps=word_timestamps,
            )

            # Convert generator to list and build response
            segments = []
            full_text = []

            for segment in segments_iter:
                segment_data = {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                }

                # Add word timestamps if requested
                if word_timestamps and hasattr(segment, "words"):
                    segment_data["words"] = [
                        {
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability,
                        }
                        for word in segment.words
                    ]

                segments.append(segment_data)
                full_text.append(segment.text.strip())

            result = {
                "text": " ".join(full_text),
                "segments": segments,
                "language": info.language,
                "duration": info.duration,
                "language_probability": info.language_probability,
            }

            logger.info(
                f"Transcribed audio: {len(audio_data)} bytes, "
                f"duration: {info.duration:.2f}s, "
                f"language: {info.language}, "
                f"segments: {len(segments)}"
            )

            return result

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_path}: {e}")

    async def is_ready(self) -> bool:
        """
        Check if the transcription service is ready.

        Returns:
            True if model is loaded and ready, False otherwise
        """
        return self._model is not None

    async def load_model(
        self, model_name: str = "base", device: Optional[str] = None
    ) -> None:
        """
        Load a Whisper model for transcription.

        Args:
            model_name: Model size (tiny, base, small, medium, large)
            device: Device to use ('cpu', 'cuda', or None for auto-detect)

        Raises:
            RuntimeError: If model loading fails
        """
        try:
            logger.info(f"Loading model: {model_name}, device: {device or 'auto'}")

            # Use model manager to load the model
            self._model = await self._model_manager.load_model(model_name, device)
            self._model_name = model_name
            self._device = device or self._model_manager.get_device()

            logger.info(
                f"Model loaded successfully: {model_name} on {self._device}"
            )

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            self._model = None
            self._model_name = None
            self._device = None
            raise RuntimeError(f"Failed to load model: {str(e)}")

    async def unload_model(self) -> None:
        """
        Unload the current model to free resources.
        """
        if self._model is not None:
            logger.info(f"Unloading model: {self._model_name}")
            await self._model_manager.unload_model()
            self._model = None
            self._model_name = None
            self._device = None
            logger.info("Model unloaded successfully")
        else:
            logger.info("No model loaded, nothing to unload")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current transcription service status.

        Returns:
            Dictionary with status information:
                - is_loaded: Whether a model is loaded
                - model_name: Current model name (if loaded)
                - device: Current device (if loaded)
                - available_models: List of available model sizes
        """
        return {
            "is_loaded": self._model is not None,
            "model_name": self._model_name,
            "device": self._device,
            "available_models": ["tiny", "base", "small", "medium", "large"],
        }


# Module-level convenience function
def get_transcription_service() -> FasterWhisperTranscriptionService:
    """
    Get the singleton transcription service instance.

    Returns:
        FasterWhisperTranscriptionService instance
    """
    return FasterWhisperTranscriptionService.get_instance()