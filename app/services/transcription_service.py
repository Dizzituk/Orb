from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """Individual segment of transcribed text with timing information."""
    
    start: float
    end: float
    text: str
    confidence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary format."""
        result = {
            "start": self.start,
            "end": self.end,
            "text": self.text
        }
        if self.confidence is not None:
            result["confidence"] = self.confidence
        return result


@dataclass
class TranscriptionResult:
    """Complete transcription result with metadata."""
    
    text: str
    language: str
    segments: List[TranscriptionSegment]
    duration: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        result = {
            "text": self.text,
            "language": self.language,
            "segments": [seg.to_dict() for seg in self.segments]
        }
        if self.duration is not None:
            result["duration"] = self.duration
        return result


class TranscriptionService(ABC):
    """Abstract base class for transcription service implementations."""
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the transcription model into memory.
        
        Raises:
            Exception: If model loading fails
        """
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload the transcription model from memory."""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded.
        
        Returns:
            bool: True if model is loaded, False otherwise
        """
        pass
    
    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> TranscriptionResult:
        """Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., 'en', 'es')
            task: Task type, either "transcribe" or "translate"
            
        Returns:
            TranscriptionResult: Complete transcription with segments
            
        Raises:
            Exception: If transcription fails or model not loaded
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current service status information.
        
        Returns:
            Dict containing:
                - loaded: bool, whether model is loaded
                - model_name: str, name of the model
                - device: str, device being used (cpu/cuda)
                - compute_type: str, precision type
        """
        pass