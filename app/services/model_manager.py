"""
Model Manager Service

Handles model lifecycle management including:
- Lazy loading of models (load on first use)
- Model caching to avoid repeated downloads
- GPU detection and device allocation
- Model unloading and memory management
"""

import logging
import os
import threading
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages the lifecycle of Whisper models.
    
    Provides lazy loading, caching, and device management for Whisper models.
    Thread-safe singleton implementation.
    """
    
    _instance: Optional['ModelManager'] = None
    _lock = threading.Lock()
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the ModelManager.
        
        Args:
            cache_dir: Directory to cache downloaded models. Defaults to ~/.cache/whisper
        """
        self._model: Optional[WhisperModel] = None
        self._current_model_name: Optional[str] = None
        self._device: Optional[str] = None
        self._compute_type: Optional[str] = None
        self._cache_dir = cache_dir or os.path.expanduser("~/.cache/whisper")
        self._model_lock = threading.Lock()
        
        # Ensure cache directory exists
        Path(self._cache_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ModelManager initialized with cache_dir: {self._cache_dir}")
    
    @classmethod
    def get_instance(cls, cache_dir: Optional[str] = None) -> 'ModelManager':
        """
        Get or create the singleton ModelManager instance.
        
        Args:
            cache_dir: Directory to cache downloaded models (only used on first creation)
            
        Returns:
            The singleton ModelManager instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(cache_dir=cache_dir)
        return cls._instance
    
    def detect_device(self) -> tuple[str, str]:
        """
        Detect the best available device and compute type.
        
        Returns:
            Tuple of (device, compute_type) where:
            - device is one of: "cuda", "cpu"
            - compute_type is one of: "float16", "int8", "float32"
        """
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
            logger.info("CUDA available, using GPU with float16")
        else:
            device = "cpu"
            compute_type = "int8"
            logger.info("CUDA not available, using CPU with int8")
        
        return device, compute_type
    
    def load_model(
        self,
        model_name: str = "base",
        device: Optional[str] = None,
        compute_type: Optional[str] = None
    ) -> WhisperModel:
        """
        Load a Whisper model with the specified configuration.
        
        If a model is already loaded with the same configuration, returns the cached model.
        If a different model is loaded, unloads it first.
        
        Args:
            model_name: Name of the Whisper model (tiny, base, small, medium, large)
            device: Device to use ("cuda" or "cpu"). Auto-detected if None.
            compute_type: Compute type ("float16", "int8", "float32"). Auto-detected if None.
            
        Returns:
            The loaded WhisperModel instance
            
        Raises:
            Exception: If model loading fails
        """
        with self._model_lock:
            # Auto-detect device and compute type if not specified
            if device is None or compute_type is None:
                detected_device, detected_compute_type = self.detect_device()
                device = device or detected_device
                compute_type = compute_type or detected_compute_type
            
            # Check if we already have this model loaded
            if (self._model is not None and 
                self._current_model_name == model_name and
                self._device == device and
                self._compute_type == compute_type):
                logger.info(f"Model '{model_name}' already loaded, returning cached instance")
                return self._model
            
            # Unload existing model if different
            if self._model is not None:
                logger.info(f"Unloading existing model '{self._current_model_name}'")
                self.unload_model()
            
            # Load new model
            logger.info(f"Loading model '{model_name}' on device '{device}' with compute_type '{compute_type}'")
            try:
                self._model = WhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute_type,
                    download_root=self._cache_dir
                )
                self._current_model_name = model_name
                self._device = device
                self._compute_type = compute_type
                
                logger.info(f"Successfully loaded model '{model_name}'")
                return self._model
                
            except Exception as e:
                logger.error(f"Failed to load model '{model_name}': {str(e)}")
                self._model = None
                self._current_model_name = None
                self._device = None
                self._compute_type = None
                raise
    
    def get_model(self) -> Optional[WhisperModel]:
        """
        Get the currently loaded model without loading a new one.
        
        Returns:
            The currently loaded WhisperModel instance, or None if no model is loaded
        """
        return self._model
    
    def unload_model(self) -> None:
        """
        Unload the currently loaded model and free memory.
        
        Thread-safe operation that can be called even if no model is loaded.
        """
        with self._model_lock:
            if self._model is not None:
                logger.info(f"Unloading model '{self._current_model_name}'")
                
                # Delete model reference
                self._model = None
                self._current_model_name = None
                self._device = None
                self._compute_type = None
                
                # Force garbage collection and clear CUDA cache
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info("Model unloaded and memory cleared")
            else:
                logger.debug("No model loaded, nothing to unload")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the model manager.
        
        Returns:
            Dictionary containing:
            - loaded: Whether a model is currently loaded
            - model_name: Name of the loaded model (if any)
            - device: Device the model is loaded on (if any)
            - compute_type: Compute type used (if any)
            - cache_dir: Directory where models are cached
        """
        with self._model_lock:
            return {
                "loaded": self._model is not None,
                "model_name": self._current_model_name,
                "device": self._device,
                "compute_type": self._compute_type,
                "cache_dir": self._cache_dir
            }
    
    def is_loaded(self) -> bool:
        """
        Check if a model is currently loaded.
        
        Returns:
            True if a model is loaded, False otherwise
        """
        return self._model is not None


def get_model_manager(cache_dir: Optional[str] = None) -> ModelManager:
    """
    Get the singleton ModelManager instance.
    
    This is a convenience function for accessing the ModelManager singleton.
    
    Args:
        cache_dir: Directory to cache downloaded models (only used on first creation)
        
    Returns:
        The singleton ModelManager instance
    """
    return ModelManager.get_instance(cache_dir=cache_dir)