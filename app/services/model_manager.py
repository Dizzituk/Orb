"""
Model manager for Whisper models with GPU detection and lazy loading.
"""
import logging
import threading
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton manager for Whisper model lifecycle."""
    
    _instance: Optional['ModelManager'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        """Initialize the model manager."""
        self._model = None
        self._model_name: Optional[str] = None
        self._device: Optional[str] = None
        self._compute_type: Optional[str] = None
        self._model_lock = threading.Lock()
        self._is_loading = False
        
    @classmethod
    def get_instance(cls) -> 'ModelManager':
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def detect_device(self) -> tuple[str, str]:
        """
        Detect available compute device and type.
        
        Returns:
            Tuple of (device, compute_type)
        """
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("CUDA GPU detected")
                return "cuda", "float16"
        except ImportError:
            pass
        
        logger.info("Using CPU")
        return "cpu", "int8"
    
    def load_model(
        self,
        model_name: str = "base",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        download_root: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Load a Whisper model with the specified configuration.
        
        Args:
            model_name: Model size (tiny, base, small, medium, large-v2, large-v3)
            device: Device to use (cuda, cpu) - auto-detected if None
            compute_type: Compute type (float16, int8, etc.) - auto-detected if None
            download_root: Directory to cache models
            **kwargs: Additional arguments passed to WhisperModel
        """
        with self._model_lock:
            if self._is_loading:
                raise RuntimeError("Model is already being loaded")
            
            self._is_loading = True
            
            try:
                # Auto-detect device if not specified
                if device is None or compute_type is None:
                    detected_device, detected_compute = self.detect_device()
                    device = device or detected_device
                    compute_type = compute_type or detected_compute
                
                logger.info(f"Loading model '{model_name}' on {device} with {compute_type}")
                
                from faster_whisper import WhisperModel
                
                # Prepare model loading arguments
                model_args = {
                    "device": device,
                    "compute_type": compute_type,
                    **kwargs
                }
                
                if download_root:
                    model_args["download_root"] = download_root
                
                # Load the model
                self._model = WhisperModel(model_name, **model_args)
                self._model_name = model_name
                self._device = device
                self._compute_type = compute_type
                
                logger.info(f"Model '{model_name}' loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self._model = None
                self._model_name = None
                self._device = None
                self._compute_type = None
                raise
            finally:
                self._is_loading = False
    
    def get_model(self):
        """
        Get the currently loaded model.
        
        Returns:
            WhisperModel instance or None if no model is loaded
        """
        return self._model
    
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model is not None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current model status.
        
        Returns:
            Dictionary with model status information
        """
        return {
            "loaded": self.is_loaded(),
            "loading": self._is_loading,
            "model_name": self._model_name,
            "device": self._device,
            "compute_type": self._compute_type
        }
    
    def unload_model(self) -> None:
        """Unload the current model and free resources."""
        with self._model_lock:
            if self._model is not None:
                logger.info(f"Unloading model '{self._model_name}'")
                del self._model
                self._model = None
                self._model_name = None
                self._device = None
                self._compute_type = None
                
                # Force garbage collection to free memory
                import gc
                gc.collect()
                
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                
                logger.info("Model unloaded successfully")


def get_model_manager() -> ModelManager:
    """
    Convenience function to get the ModelManager singleton instance.
    
    Returns:
        ModelManager instance
    """
    return ModelManager.get_instance()