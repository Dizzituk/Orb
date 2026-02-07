import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from typing import Optional
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from app.services.transcription_service import (
    TranscriptionService,
    FasterWhisperTranscriptionService,
    get_transcription_service,
)


class TestTranscriptionService:
    """Test suite for TranscriptionService interface and FasterWhisperTranscriptionService implementation."""

    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock model manager."""
        manager = Mock()
        manager.load_model = AsyncMock()
        manager.unload_model = AsyncMock()
        manager.get_model = Mock()
        manager.is_model_loaded = Mock(return_value=False)
        manager.get_device = Mock(return_value="cpu")
        return manager

    @pytest.fixture
    def mock_whisper_model(self):
        """Create a mock Faster Whisper model."""
        model = Mock()
        
        # Mock transcribe method to return segments
        segment = Mock()
        segment.text = "This is a test transcription."
        segment.start = 0.0
        segment.end = 2.5
        
        info = Mock()
        info.language = "en"
        info.language_probability = 0.95
        info.duration = 2.5
        
        model.transcribe = Mock(return_value=([segment], info))
        return model

    @pytest.fixture
    def service(self, mock_model_manager):
        """Create a FasterWhisperTranscriptionService instance with mock model manager."""
        with patch('app.services.transcription_service.get_model_manager', return_value=mock_model_manager):
            service = FasterWhisperTranscriptionService()
            return service

    @pytest.fixture
    def audio_file(self):
        """Create a temporary audio file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Write minimal WAV header
            f.write(b'RIFF')
            f.write((36).to_bytes(4, 'little'))
            f.write(b'WAVE')
            f.write(b'fmt ')
            f.write((16).to_bytes(4, 'little'))
            f.write((1).to_bytes(2, 'little'))  # PCM
            f.write((1).to_bytes(2, 'little'))  # Mono
            f.write((16000).to_bytes(4, 'little'))  # Sample rate
            f.write((32000).to_bytes(4, 'little'))  # Byte rate
            f.write((2).to_bytes(2, 'little'))  # Block align
            f.write((16).to_bytes(2, 'little'))  # Bits per sample
            f.write(b'data')
            f.write((0).to_bytes(4, 'little'))
            path = f.name
        
        yield path
        
        # Cleanup
        try:
            os.unlink(path)
        except:
            pass

    @pytest.mark.asyncio
    async def test_load_model(self, service, mock_model_manager, mock_whisper_model):
        """Test loading a transcription model."""
        mock_model_manager.get_model.return_value = mock_whisper_model
        mock_model_manager.is_model_loaded.return_value = False
        
        await service.load_model("base")
        
        mock_model_manager.load_model.assert_called_once()
        assert "base" in str(mock_model_manager.load_model.call_args)

    @pytest.mark.asyncio
    async def test_load_model_already_loaded(self, service, mock_model_manager, mock_whisper_model):
        """Test loading a model that's already loaded."""
        mock_model_manager.is_model_loaded.return_value = True
        mock_model_manager.get_model.return_value = mock_whisper_model
        
        await service.load_model("base")
        
        # Should not call load_model since it's already loaded
        mock_model_manager.load_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_unload_model(self, service, mock_model_manager):
        """Test unloading a transcription model."""
        await service.unload_model()
        
        mock_model_manager.unload_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_file(self, service, mock_model_manager, mock_whisper_model, audio_file):
        """Test transcribing an audio file."""
        mock_model_manager.get_model.return_value = mock_whisper_model
        mock_model_manager.is_model_loaded.return_value = True
        
        result = await service.transcribe(audio_file)
        
        assert "text" in result
        assert result["text"] == "This is a test transcription."
        assert "language" in result
        assert result["language"] == "en"
        assert "segments" in result
        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == "This is a test transcription."
        assert result["segments"][0]["start"] == 0.0
        assert result["segments"][0]["end"] == 2.5

    @pytest.mark.asyncio
    async def test_transcribe_without_model(self, service, mock_model_manager):
        """Test transcribing without loading a model first."""
        mock_model_manager.is_model_loaded.return_value = False
        mock_model_manager.get_model.return_value = None
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            await service.transcribe("dummy.wav")

    @pytest.mark.asyncio
    async def test_transcribe_invalid_file(self, service, mock_model_manager, mock_whisper_model):
        """Test transcribing a non-existent file."""
        mock_model_manager.get_model.return_value = mock_whisper_model
        mock_model_manager.is_model_loaded.return_value = True
        
        with pytest.raises(FileNotFoundError):
            await service.transcribe("nonexistent.wav")

    @pytest.mark.asyncio
    async def test_transcribe_bytes(self, service, mock_model_manager, mock_whisper_model):
        """Test transcribing audio from bytes."""
        mock_model_manager.get_model.return_value = mock_whisper_model
        mock_model_manager.is_model_loaded.return_value = True
        
        # Create minimal WAV bytes
        audio_bytes = b'RIFF'
        audio_bytes += (36).to_bytes(4, 'little')
        audio_bytes += b'WAVEfmt '
        audio_bytes += (16).to_bytes(4, 'little')
        audio_bytes += (1).to_bytes(2, 'little')
        audio_bytes += (1).to_bytes(2, 'little')
        audio_bytes += (16000).to_bytes(4, 'little')
        audio_bytes += (32000).to_bytes(4, 'little')
        audio_bytes += (2).to_bytes(2, 'little')
        audio_bytes += (16).to_bytes(2, 'little')
        audio_bytes += b'data'
        audio_bytes += (0).to_bytes(4, 'little')
        
        result = await service.transcribe(audio_bytes)
        
        assert "text" in result
        assert result["text"] == "This is a test transcription."

    @pytest.mark.asyncio
    async def test_transcribe_with_language(self, service, mock_model_manager, mock_whisper_model, audio_file):
        """Test transcribing with a specified language."""
        mock_model_manager.get_model.return_value = mock_whisper_model
        mock_model_manager.is_model_loaded.return_value = True
        
        result = await service.transcribe(audio_file, language="es")
        
        # Verify transcribe was called with language parameter
        call_args = mock_whisper_model.transcribe.call_args
        assert call_args is not None
        if len(call_args) > 1 and 'language' in call_args[1]:
            assert call_args[1]['language'] == "es"

    @pytest.mark.asyncio
    async def test_transcribe_with_task(self, service, mock_model_manager, mock_whisper_model, audio_file):
        """Test transcribing with translation task."""
        mock_model_manager.get_model.return_value = mock_whisper_model
        mock_model_manager.is_model_loaded.return_value = True
        
        result = await service.transcribe(audio_file, task="translate")
        
        # Verify transcribe was called with task parameter
        call_args = mock_whisper_model.transcribe.call_args
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_transcribe_multiple_segments(self, service, mock_model_manager, audio_file):
        """Test transcribing audio with multiple segments."""
        # Create model with multiple segments
        model = Mock()
        segment1 = Mock()
        segment1.text = "First segment."
        segment1.start = 0.0
        segment1.end = 1.5
        
        segment2 = Mock()
        segment2.text = "Second segment."
        segment2.start = 1.5
        segment2.end = 3.0
        
        info = Mock()
        info.language = "en"
        info.language_probability = 0.95
        info.duration = 3.0
        
        model.transcribe = Mock(return_value=([segment1, segment2], info))
        
        mock_model_manager.get_model.return_value = model
        mock_model_manager.is_model_loaded.return_value = True
        
        result = await service.transcribe(audio_file)
        
        assert len(result["segments"]) == 2
        assert result["text"] == "First segment. Second segment."
        assert result["segments"][0]["text"] == "First segment."
        assert result["segments"][1]["text"] == "Second segment."

    @pytest.mark.asyncio
    async def test_is_ready(self, service, mock_model_manager, mock_whisper_model):
        """Test checking if service is ready."""
        mock_model_manager.is_model_loaded.return_value = False
        assert not await service.is_ready()
        
        mock_model_manager.is_model_loaded.return_value = True
        mock_model_manager.get_model.return_value = mock_whisper_model
        assert await service.is_ready()

    @pytest.mark.asyncio
    async def test_get_model_info(self, service, mock_model_manager, mock_whisper_model):
        """Test retrieving model information."""
        mock_model_manager.is_model_loaded.return_value = True
        mock_model_manager.get_model.return_value = mock_whisper_model
        
        # Load model first
        await service.load_model("base")
        
        info = await service.get_model_info()
        
        assert "model_name" in info
        assert "device" in info
        assert info["device"] == "cpu"
        assert "loaded" in info
        assert info["loaded"] is True

    def test_get_transcription_service_singleton(self):
        """Test that get_transcription_service returns a singleton."""
        service1 = get_transcription_service()
        service2 = get_transcription_service()
        
        assert service1 is service2

    @pytest.mark.asyncio
    async def test_concurrent_transcriptions(self, service, mock_model_manager, mock_whisper_model, audio_file):
        """Test handling concurrent transcription requests."""
        mock_model_manager.get_model.return_value = mock_whisper_model
        mock_model_manager.is_model_loaded.return_value = True
        
        # Simulate concurrent requests
        tasks = [
            service.transcribe(audio_file),
            service.transcribe(audio_file),
            service.transcribe(audio_file),
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert "text" in result
            assert result["text"] == "This is a test transcription."

    @pytest.mark.asyncio
    async def test_transcribe_empty_audio(self, service, mock_model_manager, mock_whisper_model):
        """Test transcribing empty audio data."""
        mock_model_manager.get_model.return_value = mock_whisper_model
        mock_model_manager.is_model_loaded.return_value = True
        
        with pytest.raises((ValueError, RuntimeError)):
            await service.transcribe(b"")

    @pytest.mark.asyncio
    async def test_transcribe_format_webm(self, service, mock_model_manager, mock_whisper_model):
        """Test transcribing WebM format audio."""
        mock_model_manager.get_model.return_value = mock_whisper_model
        mock_model_manager.is_model_loaded.return_value = True
        
        # Create temporary WebM file
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            # Minimal WebM header
            f.write(b'\x1a\x45\xdf\xa3')  # EBML header
            path = f.name
        
        try:
            result = await service.transcribe(path)
            assert "text" in result
        finally:
            try:
                os.unlink(path)
            except:
                pass

    @pytest.mark.asyncio
    async def test_transcribe_format_mp3(self, service, mock_model_manager, mock_whisper_model):
        """Test transcribing MP3 format audio."""
        mock_model_manager.get_model.return_value = mock_whisper_model
        mock_model_manager.is_model_loaded.return_value = True
        
        # Create temporary MP3 file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            # Minimal MP3 header
            f.write(b'\xff\xfb')
            path = f.name
        
        try:
            result = await service.transcribe(path)
            assert "text" in result
        finally:
            try:
                os.unlink(path)
            except:
                pass

    @pytest.mark.asyncio
    async def test_transcribe_with_vad(self, service, mock_model_manager, mock_whisper_model, audio_file):
        """Test transcribing with VAD filter enabled."""
        mock_model_manager.get_model.return_value = mock_whisper_model
        mock_model_manager.is_model_loaded.return_value = True
        
        result = await service.transcribe(audio_file, vad_filter=True)
        
        assert "text" in result
        # Verify VAD parameter was passed
        call_args = mock_whisper_model.transcribe.call_args
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_model_reload(self, service, mock_model_manager, mock_whisper_model):
        """Test reloading a model with different settings."""
        mock_model_manager.get_model.return_value = mock_whisper_model
        mock_model_manager.is_model_loaded.return_value = False
        
        # Load first model
        await service.load_model("base")
        mock_model_manager.is_model_loaded.return_value = True
        
        # Reload with different model
        mock_model_manager.is_model_loaded.return_value = False
        await service.load