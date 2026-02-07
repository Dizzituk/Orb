import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from app.services.model_manager import ModelManager, get_model_manager


class TestModelManager:
    """Test suite for ModelManager class."""

    @pytest.fixture
    def temp_cache_dir(self, tmp_path):
        """Create a temporary cache directory."""
        cache_dir = tmp_path / "model_cache"
        cache_dir.mkdir()
        return str(cache_dir)

    @pytest.fixture
    def model_manager(self, temp_cache_dir):
        """Create a ModelManager instance with temporary cache."""
        return ModelManager(cache_dir=temp_cache_dir)

    def test_initialization_default_cache(self):
        """Test ModelManager initialization with default cache directory."""
        manager = ModelManager()
        assert manager.cache_dir is not None
        assert Path(manager.cache_dir).exists()
        assert manager.current_model is None
        assert manager.current_model_name is None

    def test_initialization_custom_cache(self, temp_cache_dir):
        """Test ModelManager initialization with custom cache directory."""
        manager = ModelManager(cache_dir=temp_cache_dir)
        assert manager.cache_dir == temp_cache_dir
        assert Path(manager.cache_dir).exists()

    def test_device_detection_cuda_available(self, model_manager):
        """Test device detection when CUDA is available."""
        with patch("torch.cuda.is_available", return_value=True):
            device = model_manager._detect_device()
            assert device == "cuda"

    def test_device_detection_cpu_only(self, model_manager):
        """Test device detection when only CPU is available."""
        with patch("torch.cuda.is_available", return_value=False):
            device = model_manager._detect_device()
            assert device == "cpu"

    def test_compute_type_detection_cuda(self, model_manager):
        """Test compute type detection for CUDA device."""
        with patch("torch.cuda.is_available", return_value=True):
            compute_type = model_manager._get_compute_type("cuda")
            assert compute_type in ["float16", "int8"]

    def test_compute_type_detection_cpu(self, model_manager):
        """Test compute type detection for CPU device."""
        compute_type = model_manager._get_compute_type("cpu")
        assert compute_type == "int8"

    @patch("app.services.model_manager.WhisperModel")
    def test_load_model_success(self, mock_whisper_model, model_manager):
        """Test successful model loading."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model

        model = model_manager.load_model("base")

        assert model == mock_model
        assert model_manager.current_model == mock_model
        assert model_manager.current_model_name == "base"
        mock_whisper_model.assert_called_once()

    @patch("app.services.model_manager.WhisperModel")
    def test_load_model_with_device(self, mock_whisper_model, model_manager):
        """Test model loading with specific device."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model

        model = model_manager.load_model("base", device="cpu")

        assert model == mock_model
        call_kwargs = mock_whisper_model.call_args[1]
        assert call_kwargs["device"] == "cpu"

    @patch("app.services.model_manager.WhisperModel")
    def test_load_model_caching(self, mock_whisper_model, model_manager):
        """Test that loading the same model twice returns cached instance."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model

        model1 = model_manager.load_model("base")
        model2 = model_manager.load_model("base")

        assert model1 == model2
        mock_whisper_model.assert_called_once()

    @patch("app.services.model_manager.WhisperModel")
    def test_load_different_model_replaces_cached(self, mock_whisper_model, model_manager):
        """Test that loading a different model replaces the cached one."""
        mock_model1 = Mock()
        mock_model2 = Mock()
        mock_whisper_model.side_effect = [mock_model1, mock_model2]

        model1 = model_manager.load_model("base")
        model2 = model_manager.load_model("small")

        assert model1 != model2
        assert model_manager.current_model == model2
        assert model_manager.current_model_name == "small"
        assert mock_whisper_model.call_count == 2

    @patch("app.services.model_manager.WhisperModel")
    def test_load_model_failure(self, mock_whisper_model, model_manager):
        """Test model loading failure handling."""
        mock_whisper_model.side_effect = Exception("Model load failed")

        with pytest.raises(Exception, match="Model load failed"):
            model_manager.load_model("base")

        assert model_manager.current_model is None
        assert model_manager.current_model_name is None

    def test_get_model_not_loaded(self, model_manager):
        """Test getting model when none is loaded."""
        model = model_manager.get_model()
        assert model is None

    @patch("app.services.model_manager.WhisperModel")
    def test_get_model_loaded(self, mock_whisper_model, model_manager):
        """Test getting model when one is loaded."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model

        model_manager.load_model("base")
        model = model_manager.get_model()

        assert model == mock_model

    def test_is_loaded_false(self, model_manager):
        """Test is_loaded returns False when no model is loaded."""
        assert model_manager.is_loaded() is False

    @patch("app.services.model_manager.WhisperModel")
    def test_is_loaded_true(self, mock_whisper_model, model_manager):
        """Test is_loaded returns True when model is loaded."""
        mock_whisper_model.return_value = Mock()
        model_manager.load_model("base")
        assert model_manager.is_loaded() is True

    def test_get_current_model_name_none(self, model_manager):
        """Test getting current model name when none is loaded."""
        assert model_manager.get_current_model_name() is None

    @patch("app.services.model_manager.WhisperModel")
    def test_get_current_model_name_loaded(self, mock_whisper_model, model_manager):
        """Test getting current model name when model is loaded."""
        mock_whisper_model.return_value = Mock()
        model_manager.load_model("base")
        assert model_manager.get_current_model_name() == "base"

    @patch("app.services.model_manager.WhisperModel")
    def test_unload_model(self, mock_whisper_model, model_manager):
        """Test model unloading."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model

        model_manager.load_model("base")
        assert model_manager.is_loaded() is True

        model_manager.unload_model()

        assert model_manager.current_model is None
        assert model_manager.current_model_name is None
        assert model_manager.is_loaded() is False

    def test_unload_model_when_none_loaded(self, model_manager):
        """Test unloading when no model is loaded."""
        model_manager.unload_model()
        assert model_manager.current_model is None
        assert model_manager.current_model_name is None

    @patch("app.services.model_manager.WhisperModel")
    def test_get_device(self, mock_whisper_model, model_manager):
        """Test getting device for loaded model."""
        with patch("torch.cuda.is_available", return_value=True):
            mock_whisper_model.return_value = Mock()
            model_manager.load_model("base")
            device = model_manager.get_device()
            assert device == "cuda"

    def test_get_device_when_not_loaded(self, model_manager):
        """Test getting device when no model is loaded."""
        device = model_manager.get_device()
        assert device in ["cpu", "cuda"]

    @patch("app.services.model_manager.WhisperModel")
    def test_model_size_validation(self, mock_whisper_model, model_manager):
        """Test loading different model sizes."""
        mock_whisper_model.return_value = Mock()

        valid_sizes = ["tiny", "base", "small", "medium", "large", "large-v2"]
        for size in valid_sizes:
            model_manager.load_model(size)
            assert model_manager.get_current_model_name() == size

    @patch("app.services.model_manager.WhisperModel")
    def test_concurrent_load_same_model(self, mock_whisper_model, model_manager):
        """Test concurrent loads of the same model use cache."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model

        model_manager.load_model("base")
        model_manager.load_model("base")
        model_manager.load_model("base")

        mock_whisper_model.assert_called_once()

    @patch("app.services.model_manager.WhisperModel")
    def test_load_model_with_custom_compute_type(self, mock_whisper_model, model_manager):
        """Test loading model with custom compute type."""
        mock_model = Mock()
        mock_whisper_model.return_value = mock_model

        model_manager.load_model("base", compute_type="float32")

        call_kwargs = mock_whisper_model.call_args[1]
        assert call_kwargs["compute_type"] == "float32"

    def test_get_model_manager_singleton(self, temp_cache_dir):
        """Test get_model_manager returns singleton instance."""
        manager1 = get_model_manager(cache_dir=temp_cache_dir)
        manager2 = get_model_manager(cache_dir=temp_cache_dir)
        assert manager1 is manager2

    def test_get_model_manager_default(self):
        """Test get_model_manager with default parameters."""
        manager = get_model_manager()
        assert isinstance(manager, ModelManager)
        assert manager.cache_dir is not None

    @patch("app.services.model_manager.WhisperModel")
    def test_model_manager_state_after_error(self, mock_whisper_model, model_manager):
        """Test ModelManager state remains consistent after load error."""
        mock_whisper_model.side_effect = Exception("Load failed")

        with pytest.raises(Exception):
            model_manager.load_model("base")

        assert model_manager.current_model is None
        assert model_manager.current_model_name is None
        assert not model_manager.is_loaded()

    @patch("app.services.model_manager.WhisperModel")
    def test_model_reload_after_unload(self, mock_whisper_model, model_manager):
        """Test reloading model after unloading."""
        mock_model1 = Mock()
        mock_model2 = Mock()
        mock_whisper_model.side_effect = [mock_model1, mock_model2]

        model_manager.load_model("base")
        model_manager.unload_model()
        model_manager.load_model("base")

        assert model_manager.is_loaded()
        assert mock_whisper_model.call_count == 2

    @patch("app.services.model_manager.WhisperModel")
    def test_cache_dir_usage(self, mock_whisper_model, temp_cache_dir):
        """Test that cache directory is used in model loading."""
        mock_whisper_model.return_value = Mock()
        manager = ModelManager(cache_dir=temp_cache_dir)

        manager.load_model("base")

        call_kwargs = mock_whisper_model.call_args[1]
        assert call_kwargs["download_root"] == temp_cache_dir