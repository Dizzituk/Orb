"""
Wake-word detection and Voice Activity Detection (VAD) service.

This service uses OpenWakeWord for lightweight wake-word detection and
provides VAD capabilities for streaming audio processing.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class WakeWordService:
    """
    Singleton service for wake-word detection and VAD using OpenWakeWord.
    
    This service provides:
    - Wake-word detection (e.g., "hey jarvis")
    - Voice Activity Detection (VAD)
    - Thread-safe operations for concurrent audio processing
    """
    
    _instance: Optional['WakeWordService'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        """Initialize the wake-word service."""
        self._model = None
        self._vad_model = None
        self._is_loaded = False
        self._model_lock = threading.Lock()
        self._threshold = 0.5
        self._vad_threshold = 0.5
        logger.info("WakeWordService instance created")
    
    @classmethod
    def get_instance(cls) -> 'WakeWordService':
        """
        Get or create the singleton instance of WakeWordService.
        
        Returns:
            WakeWordService: The singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def load_models(
        self,
        wake_word_model: Optional[str] = None,
        threshold: float = 0.5,
        vad_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Load the wake-word detection and VAD models.
        
        Args:
            wake_word_model: Optional custom wake word model path
            threshold: Detection threshold for wake words (0-1)
            vad_threshold: Detection threshold for VAD (0-1)
            
        Returns:
            Dict containing load status and model info
        """
        with self._model_lock:
            try:
                if self._is_loaded:
                    logger.info("Wake-word models already loaded")
                    return {
                        "status": "already_loaded",
                        "wake_word_model": wake_word_model or "default",
                        "threshold": self._threshold,
                        "vad_threshold": self._vad_threshold
                    }
                
                logger.info(f"Loading wake-word models (threshold: {threshold}, vad_threshold: {vad_threshold})")
                
                # Import openwakeword here to avoid loading if not needed
                from openwakeword.model import Model
                
                # Initialize OpenWakeWord model
                if wake_word_model and Path(wake_word_model).exists():
                    logger.info(f"Loading custom wake word model: {wake_word_model}")
                    self._model = Model(wakeword_models=[wake_word_model])
                else:
                    logger.info("Loading default wake word models")
                    self._model = Model()
                
                self._threshold = threshold
                self._vad_threshold = vad_threshold
                self._is_loaded = True
                
                # Get available models
                available_models = list(self._model.models.keys()) if hasattr(self._model, 'models') else []
                
                logger.info(f"Wake-word models loaded successfully. Available models: {available_models}")
                
                return {
                    "status": "loaded",
                    "wake_word_model": wake_word_model or "default",
                    "threshold": self._threshold,
                    "vad_threshold": self._vad_threshold,
                    "available_models": available_models
                }
                
            except Exception as e:
                logger.error(f"Failed to load wake-word models: {str(e)}")
                self._is_loaded = False
                raise RuntimeError(f"Failed to load wake-word models: {str(e)}")
    
    def unload_models(self) -> Dict[str, str]:
        """
        Unload the wake-word detection models to free memory.
        
        Returns:
            Dict containing unload status
        """
        with self._model_lock:
            if not self._is_loaded:
                logger.info("Wake-word models not loaded, nothing to unload")
                return {"status": "not_loaded"}
            
            try:
                logger.info("Unloading wake-word models")
                self._model = None
                self._vad_model = None
                self._is_loaded = False
                logger.info("Wake-word models unloaded successfully")
                return {"status": "unloaded"}
                
            except Exception as e:
                logger.error(f"Error unloading wake-word models: {str(e)}")
                raise RuntimeError(f"Failed to unload wake-word models: {str(e)}")
    
    def detect_wake_word(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Detect wake words in audio data.
        
        Args:
            audio_data: Audio data as numpy array (int16 or float32)
            sample_rate: Sample rate of audio data (default: 16000)
            
        Returns:
            Dict containing detection results:
                - detected: bool indicating if wake word was detected
                - scores: Dict of model names to confidence scores
                - triggered_model: Name of model that triggered (if any)
        """
        if not self._is_loaded:
            raise RuntimeError("Wake-word models not loaded. Call load_models() first.")
        
        try:
            # Convert audio to float32 if needed
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Ensure correct sample rate
            if sample_rate != 16000:
                logger.warning(f"Audio sample rate {sample_rate} != 16000, resampling may be needed")
            
            # Get predictions from model
            with self._model_lock:
                prediction = self._model.predict(audio_data)
            
            # Find highest scoring model
            max_score = 0.0
            triggered_model = None
            scores = {}
            
            for model_name, score in prediction.items():
                scores[model_name] = float(score)
                if score > max_score:
                    max_score = score
                    triggered_model = model_name
            
            detected = max_score >= self._threshold
            
            if detected:
                logger.info(f"Wake word detected: {triggered_model} (score: {max_score:.3f})")
            
            return {
                "detected": detected,
                "scores": scores,
                "triggered_model": triggered_model if detected else None,
                "max_score": max_score
            }
            
        except Exception as e:
            logger.error(f"Error detecting wake word: {str(e)}")
            raise RuntimeError(f"Wake word detection failed: {str(e)}")
    
    def detect_voice_activity(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Detect voice activity in audio data.
        
        Args:
            audio_data: Audio data as numpy array (int16 or float32)
            sample_rate: Sample rate of audio data (default: 16000)
            
        Returns:
            Dict containing VAD results:
                - voice_detected: bool indicating if voice was detected
                - confidence: Voice activity confidence score (0-1)
        """
        if not self._is_loaded:
            raise RuntimeError("Wake-word models not loaded. Call load_models() first.")
        
        try:
            # Convert audio to float32 if needed
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Simple energy-based VAD as fallback
            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            # Normalize to 0-1 range (assuming max RMS of 0.3 for speech)
            confidence = min(rms / 0.3, 1.0)
            
            voice_detected = confidence >= self._vad_threshold
            
            return {
                "voice_detected": voice_detected,
                "confidence": float(confidence)
            }
            
        except Exception as e:
            logger.error(f"Error detecting voice activity: {str(e)}")
            raise RuntimeError(f"Voice activity detection failed: {str(e)}")
    
    def process_audio_chunk(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        check_wake_word: bool = True,
        check_vad: bool = True
    ) -> Dict[str, Any]:
        """
        Process an audio chunk for both wake word and VAD.
        
        Args:
            audio_data: Audio data as numpy array (int16 or float32)
            sample_rate: Sample rate of audio data (default: 16000)
            check_wake_word: Whether to check for wake words
            check_vad: Whether to check for voice activity
            
        Returns:
            Dict containing combined detection results
        """
        results = {}
        
        if check_wake_word:
            wake_word_result = self.detect_wake_word(audio_data, sample_rate)
            results.update({
                "wake_word_detected": wake_word_result["detected"],
                "wake_word_scores": wake_word_result["scores"],
                "triggered_model": wake_word_result.get("triggered_model")
            })
        
        if check_vad:
            vad_result = self.detect_voice_activity(audio_data, sample_rate)
            results.update({
                "voice_detected": vad_result["voice_detected"],
                "voice_confidence": vad_result["confidence"]
            })
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the wake-word service.
        
        Returns:
            Dict containing service status and configuration
        """
        with self._model_lock:
            status = {
                "loaded": self._is_loaded,
                "threshold": self._threshold,
                "vad_threshold": self._vad_threshold
            }
            
            if self._is_loaded and self._model:
                available_models = list(self._model.models.keys()) if hasattr(self._model, 'models') else []
                status["available_models"] = available_models
            
            return status
    
    def set_threshold(self, threshold: float, vad_threshold: Optional[float] = None):
        """
        Update detection thresholds.
        
        Args:
            threshold: New wake word detection threshold (0-1)
            vad_threshold: Optional new VAD threshold (0-1)
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        self._threshold = threshold
        logger.info(f"Wake word threshold updated to {threshold}")
        
        if vad_threshold is not None:
            if not 0 <= vad_threshold <= 1:
                raise ValueError("VAD threshold must be between 0 and 1")
            self._vad_threshold = vad_threshold
            logger.info(f"VAD threshold updated to {vad_threshold}")


def get_wake_word_service() -> WakeWordService:
    """
    Get the singleton instance of WakeWordService.
    
    This is a convenience function for dependency injection and testing.
    
    Returns:
        WakeWordService: The singleton instance
    """
    return WakeWordService.get_instance()