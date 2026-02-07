import os
import logging
from typing import Optional, List
import torch
import numpy as np
from faster_whisper import WhisperModel
from app.services.transcription_service import (
    TranscriptionSegment,
    TranscriptionResult,
    TranscriptionService,
)
from app.services.model_manager import get_model_manager

logger = logging.getLogger(__name__)


class FasterWhisperService(TranscriptionService):
    """
    Implementation of TranscriptionService using faster-whisper library.
    
    This service uses the faster-whisper library for efficient speech-to-text
    transcription. It supports GPU acceleration when available and provides
    configurable parameters for transcription quality and performance.
    """

    def __init__(
        self,
        model_name: str = "base",
        device: Optional[str] = None,
        compute_type: str = "float16",
        cpu_threads: int = 4,
        num_workers: int = 1,
    ):
        """
        Initialize the FasterWhisperService.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
            device: Device to use ("cuda", "cpu", or None for auto-detection)
            compute_type: Computation precision (float16, int8, float32)
            cpu_threads: Number of CPU threads to use
            num_workers: Number of parallel workers for batch processing
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers
        self._model: Optional[WhisperModel] = None
        self._is_loaded = False

    def load_model(self) -> None:
        """
        Load the Whisper model using the model manager.
        
        Raises:
            RuntimeError: If model loading fails
        """
        if self._is_loaded and self._model is not None:
            logger.info(f"Model {self.model_name} already loaded")
            return

        try:
            model_manager = get_model_manager()
            
            # Auto-detect device if not specified
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Adjust compute type for CPU
            compute_type = self.compute_type
            if self.device == "cpu" and compute_type == "float16":
                compute_type = "int8"
                logger.info("Using int8 compute type for CPU device")
            
            logger.info(
                f"Loading Whisper model '{self.model_name}' on device '{self.device}' "
                f"with compute type '{compute_type}'"
            )
            
            # Load model through model manager
            self._model = model_manager.load_model(
                model_name=self.model_name,
                device=self.device,
                compute_type=compute_type,
                cpu_threads=self.cpu_threads,
                num_workers=self.num_workers,
            )
            
            self._is_loaded = True
            logger.info(f"Model {self.model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise RuntimeError(f"Failed to load model: {e}") from e

    def unload_model(self) -> None:
        """
        Unload the Whisper model and free resources.
        """
        if self._model is not None:
            logger.info(f"Unloading model {self.model_name}")
            self._model = None
            self._is_loaded = False
            
            # Clear CUDA cache if using GPU
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache")

    def is_loaded(self) -> bool:
        """
        Check if the model is currently loaded.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self._is_loaded and self._model is not None

    def transcribe(
        self,
        audio_data: np.ndarray,
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        best_of: int = 5,
        temperature: float = 0.0,
        vad_filter: bool = True,
        vad_parameters: Optional[dict] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Audio samples as numpy array (float32, -1.0 to 1.0 range)
            language: Source language code (None for auto-detection)
            task: "transcribe" or "translate" (to English)
            beam_size: Beam size for beam search decoding
            best_of: Number of candidates to generate with temperature sampling
            temperature: Temperature for sampling (0.0 for greedy decoding)
            vad_filter: Whether to use voice activity detection
            vad_parameters: Optional VAD configuration parameters
            
        Returns:
            TranscriptionResult containing full text and segments
            
        Raises:
            RuntimeError: If model is not loaded or transcription fails
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            logger.info(f"Starting transcription (language={language}, task={task})")
            
            # Prepare VAD parameters
            vad_params = vad_parameters or {}
            if vad_filter:
                vad_params.setdefault("threshold", 0.5)
                vad_params.setdefault("min_speech_duration_ms", 250)
                vad_params.setdefault("min_silence_duration_ms", 2000)
            
            # Transcribe audio
            segments, info = self._model.transcribe(
                audio_data,
                language=language,
                task=task,
                beam_size=beam_size,
                best_of=best_of,
                temperature=temperature,
                vad_filter=vad_filter,
                vad_parameters=vad_params if vad_filter else None,
            )
            
            # Convert segments to our format
            transcription_segments: List[TranscriptionSegment] = []
            full_text_parts: List[str] = []
            
            for segment in segments:
                transcription_segments.append(
                    TranscriptionSegment(
                        start=segment.start,
                        end=segment.end,
                        text=segment.text.strip(),
                        confidence=segment.avg_logprob,
                    )
                )
                full_text_parts.append(segment.text.strip())
            
            # Combine all segments into full text
            full_text = " ".join(full_text_parts)
            
            # Determine detected language
            detected_language = info.language if info else language
            language_probability = info.language_probability if info else None
            
            result = TranscriptionResult(
                text=full_text,
                segments=transcription_segments,
                language=detected_language,
                language_probability=language_probability,
            )
            
            logger.info(
                f"Transcription completed: {len(transcription_segments)} segments, "
                f"language={detected_language}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Transcription failed: {e}") from e

    def transcribe_stream(
        self,
        audio_chunk: np.ndarray,
        language: Optional[str] = None,
        **kwargs,
    ) -> Optional[TranscriptionResult]:
        """
        Transcribe a single audio chunk for streaming scenarios.
        
        For streaming, this method processes individual audio chunks. It's a
        simplified version of transcribe() suitable for real-time processing.
        
        Args:
            audio_chunk: Audio samples as numpy array (float32)
            language: Source language code (None for auto-detection)
            **kwargs: Additional transcription parameters
            
        Returns:
            TranscriptionResult if transcription successful, None if chunk too short
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Skip very short audio chunks
        min_duration = 0.5  # seconds
        sample_rate = 16000  # Whisper expects 16kHz
        min_samples = int(min_duration * sample_rate)
        
        if len(audio_chunk) < min_samples:
            logger.debug(f"Audio chunk too short: {len(audio_chunk)} samples")
            return None

        try:
            # Use faster settings for streaming
            return self.transcribe(
                audio_data=audio_chunk,
                language=language,
                beam_size=1,  # Faster for streaming
                best_of=1,
                temperature=0.0,
                vad_filter=True,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Stream transcription failed: {e}")
            return None

    def get_model_info(self) -> dict:
        """
        Get information about the current model configuration.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "is_loaded": self._is_loaded,
            "cpu_threads": self.cpu_threads,
            "num_workers": self.num_workers,
        }