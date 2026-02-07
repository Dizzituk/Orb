"""
Model manager for Whisper models with GPU detection and lazy loading.

Reads configuration from D:\\LocalAI\\config.ini when available,
falls back to environment variables or sensible defaults.
"""
import configparser
import gc
import logging
import os
import threading
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# LocalAI config path
LOCAL_AI_CONFIG = Path(r"D:\LocalAI\config.ini")
LOCAL_AI_MODELS_DIR = Path(r"D:\LocalAI\models\whisper")


def _load_local_ai_config() -> Dict[str, str]:
    """Load whisper config from LocalAI config.ini if it exists."""
    defaults = {
        "model_size": "base.en",
        "device": "cpu",
        "compute_type": "int8",
        "vad_filter": "true",
        "language": "en",
    }
    if LOCAL_AI_CONFIG.exists():
        try:
            cfg = configparser.ConfigParser()
            cfg.read(str(LOCAL_AI_CONFIG))
            if cfg.has_section("whisper"):
                # Only read safe keys — device/compute_type are validated separately
                for key in ["model_size", "vad_filter", "language"]:
                    if cfg.has_option("whisper", key):
                        defaults[key] = cfg.get("whisper", key)
                logger.info("[model_manager] Loaded config from %s: %s", LOCAL_AI_CONFIG, defaults)
        except Exception as e:
            logger.warning("[model_manager] Failed to read %s: %s", LOCAL_AI_CONFIG, e)

    # Validate model is actually downloaded — check for .incomplete blobs
    model_dir = LOCAL_AI_MODELS_DIR / f"models--Systran--faster-whisper-{defaults['model_size']}"
    if model_dir.exists():
        incomplete = list(model_dir.rglob("*.incomplete"))
        if incomplete:
            logger.warning(
                "[model_manager] Model '%s' has incomplete downloads (%d files), "
                "falling back to base.en",
                defaults["model_size"], len(incomplete),
            )
            defaults["model_size"] = "base.en"

    return defaults


class ModelManager:
    """Singleton manager for Whisper model lifecycle."""

    _instance: Optional['ModelManager'] = None
    _lock = threading.Lock()

    def __init__(self):
        self._model = None
        self._model_name: Optional[str] = None
        self._device: Optional[str] = None
        self._compute_type: Optional[str] = None
        self._model_lock = threading.Lock()
        self._is_loading = False

        # Load config from LocalAI
        self._config = _load_local_ai_config()

    @classmethod
    def get_instance(cls) -> 'ModelManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def detect_device(self) -> tuple:
        """Detect best device. Checks LocalAI config first, then probes CUDA."""
        cfg_device = self._config.get("device", "cpu")
        cfg_compute = self._config.get("compute_type", "int8")

        if cfg_device == "cuda":
            # Verify CUDA runtime libs are actually loadable — cublas, cudnn etc.
            try:
                import ctranslate2
                ctranslate2.get_supported_compute_types("cuda")
                # Smoke test: actually do a tiny CUDA operation to catch missing cublas/cudnn
                import numpy as np
                test_model_data = np.zeros((1, 1), dtype=np.float32)
                sv = ctranslate2.StorageView.from_array(test_model_data)
                # Try to move to CUDA — this triggers cublas load
                sv_cuda = sv.to(ctranslate2.Device.cuda)
                logger.info("[model_manager] CUDA available and runtime libs OK")
                return "cuda", cfg_compute
            except Exception as e:
                logger.warning("[model_manager] CUDA runtime check failed: %s — using CPU", e)

        return "cpu", "int8"

    def load_model(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ) -> None:
        """Load a Whisper model. Uses LocalAI config as defaults."""
        with self._model_lock:
            if self._is_loading:
                raise RuntimeError("Model is already being loaded")

            # If already loaded with same config, skip
            effective_name = model_name or self._config.get("model_size", "base.en")
            if self._model is not None and self._model_name == effective_name:
                logger.info("[model_manager] Model '%s' already loaded", effective_name)
                return

            self._is_loading = True
            try:
                if device is None or compute_type is None:
                    det_device, det_compute = self.detect_device()
                    device = device or det_device
                    compute_type = compute_type or det_compute

                logger.info("[model_manager] Loading '%s' on %s (%s)", effective_name, device, compute_type)

                from faster_whisper import WhisperModel

                model_kwargs = {
                    "device": device,
                    "compute_type": compute_type,
                }

                # Use LocalAI models dir if it exists
                if LOCAL_AI_MODELS_DIR.exists():
                    model_kwargs["download_root"] = str(LOCAL_AI_MODELS_DIR)

                self._model = WhisperModel(effective_name, **model_kwargs)
                self._model_name = effective_name
                self._device = device
                self._compute_type = compute_type

                logger.info("[model_manager] Model '%s' loaded successfully on %s", effective_name, device)

            except Exception as e:
                logger.error("[model_manager] Failed to load model: %s", e)
                self._model = None
                self._model_name = None
                self._device = None
                self._compute_type = None
                raise
            finally:
                self._is_loading = False

    def get_model(self):
        """Get the loaded WhisperModel instance (or None)."""
        return self._model

    def is_loaded(self) -> bool:
        return self._model is not None

    def get_status(self) -> Dict[str, Any]:
        return {
            "loaded": self.is_loaded(),
            "loading": self._is_loading,
            "model_name": self._model_name,
            "device": self._device,
            "compute_type": self._compute_type,
            "config_source": str(LOCAL_AI_CONFIG) if LOCAL_AI_CONFIG.exists() else "defaults",
        }

    def unload_model(self) -> None:
        with self._model_lock:
            if self._model is not None:
                logger.info("[model_manager] Unloading model '%s'", self._model_name)
                del self._model
                self._model = None
                self._model_name = None
                self._device = None
                self._compute_type = None
                gc.collect()

                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

                logger.info("[model_manager] Model unloaded")


def get_model_manager() -> ModelManager:
    """Get the ModelManager singleton."""
    return ModelManager.get_instance()
