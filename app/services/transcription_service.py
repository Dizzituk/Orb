"""
Transcription service interface and data models for voice-to-text.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """Individual segment of transcribed text with timing."""
    start: float
    end: float
    text: str
    confidence: Optional[float] = None


@dataclass
class TranscriptionResult:
    """Complete transcription result."""
    text: str
    language: str = "en"
    segments: List[TranscriptionSegment] = field(default_factory=list)
    duration: Optional[float] = None


class TranscriptionService(ABC):
    """Abstract base class for transcription implementations."""

    @abstractmethod
    def load_model(self) -> None:
        """Load the transcription model into memory."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload the transcription model from memory."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        pass

    @abstractmethod
    def transcribe(self, audio_bytes: bytes, language: Optional[str] = None,
                   vad_filter: bool = True, vad_parameters: Optional[dict] = None
                   ) -> TranscriptionResult:
        """Transcribe audio bytes to text."""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current service status."""
        pass
