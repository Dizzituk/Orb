import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from io import BytesIO
import wave
import struct
from unittest.mock import Mock, patch, AsyncMock
from app.routers.transcribe import router, TranscriptionResponse, ModelStatusResponse, LoadModelRequest


@pytest.fixture
def app():
    """Create a test FastAPI application."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_transcription_service():
    """Create a mock transcription service."""
    service = Mock()
    service.is_loaded = Mock(return_value=False)
    service.get_model_name = Mock(return_value=None)
    service.load_model = AsyncMock()
    service.unload_model = AsyncMock()
    service.transcribe = AsyncMock(return_value="test transcription")
    return service


@pytest.fixture
def sample_audio_bytes():
    """Generate sample WAV audio bytes for testing."""
    # Create a 1-second, 16kHz, mono WAV file in memory
    sample_rate = 16000
    duration = 1  # seconds
    frequency = 440  # Hz (A4 note)
    
    buffer = BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        # Generate sine wave samples
        for i in range(sample_rate * duration):
            value = int(32767 * 0.3 * (i % (sample_rate // frequency)) / (sample_rate // frequency))
            data = struct.pack('<h', value)
            wav_file.writeframes(data)
    
    buffer.seek(0)
    return buffer.read()


class TestGetStatus:
    """Test cases for GET /transcribe/status endpoint."""
    
    def test_status_model_not_loaded(self, client, mock_transcription_service):
        """Test status endpoint when model is not loaded."""
        with patch('app.routers.transcribe.get_transcription_service', return_value=mock_transcription_service):
            response = client.get("/transcribe/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["is_loaded"] is False
            assert data["model_name"] is None
            assert "device" in data
    
    def test_status_model_loaded(self, client, mock_transcription_service):
        """Test status endpoint when model is loaded."""
        mock_transcription_service.is_loaded.return_value = True
        mock_transcription_service.get_model_name.return_value = "base.en"
        
        with patch('app.routers.transcribe.get_transcription_service', return_value=mock_transcription_service):
            response = client.get("/transcribe/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["is_loaded"] is True
            assert data["model_name"] == "base.en"
            assert "device" in data


class TestLoadModel:
    """Test cases for POST /transcribe/load endpoint."""
    
    @pytest.mark.asyncio
    async def test_load_model_success(self, client, mock_transcription_service):
        """Test successful model loading."""
        with patch('app.routers.transcribe.get_transcription_service', return_value=mock_transcription_service):
            response = client.post(
                "/transcribe/load",
                json={"model_name": "base.en"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "loaded"
            assert data["model_name"] == "base.en"
            mock_transcription_service.load_model.assert_called_once_with("base.en")
    
    @pytest.mark.asyncio
    async def test_load_model_already_loaded(self, client, mock_transcription_service):
        """Test loading model when already loaded."""
        mock_transcription_service.is_loaded.return_value = True
        mock_transcription_service.get_model_name.return_value = "base.en"
        
        with patch('app.routers.transcribe.get_transcription_service', return_value=mock_transcription_service):
            response = client.post(
                "/transcribe/load",
                json={"model_name": "base.en"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "already_loaded"
            mock_transcription_service.load_model.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_load_model_invalid_name(self, client, mock_transcription_service):
        """Test loading model with invalid name."""
        mock_transcription_service.load_model.side_effect = ValueError("Invalid model name")
        
        with patch('app.routers.transcribe.get_transcription_service', return_value=mock_transcription_service):
            response = client.post(
                "/transcribe/load",
                json={"model_name": "invalid_model"}
            )
            
            assert response.status_code == 400
            assert "Invalid model name" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_load_model_error(self, client, mock_transcription_service):
        """Test model loading with unexpected error."""
        mock_transcription_service.load_model.side_effect = RuntimeError("Model loading failed")
        
        with patch('app.routers.transcribe.get_transcription_service', return_value=mock_transcription_service):
            response = client.post(
                "/transcribe/load",
                json={"model_name": "base.en"}
            )
            
            assert response.status_code == 500
            assert "Model loading failed" in response.json()["detail"]


class TestTranscribeAudio:
    """Test cases for POST /transcribe/transcribe endpoint."""
    
    @pytest.mark.asyncio
    async def test_transcribe_success(self, client, mock_transcription_service, sample_audio_bytes):
        """Test successful audio transcription."""
        mock_transcription_service.is_loaded.return_value = True
        mock_transcription_service.transcribe.return_value = "hello world"
        
        with patch('app.routers.transcribe.get_transcription_service', return_value=mock_transcription_service):
            response = client.post(
                "/transcribe/transcribe",
                files={"file": ("test.wav", BytesIO(sample_audio_bytes), "audio/wav")}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["text"] == "hello world"
            assert data["language"] is not None
            mock_transcription_service.transcribe.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_transcribe_model_not_loaded(self, client, mock_transcription_service, sample_audio_bytes):
        """Test transcription when model is not loaded."""
        mock_transcription_service.is_loaded.return_value = False
        
        with patch('app.routers.transcribe.get_transcription_service', return_value=mock_transcription_service):
            response = client.post(
                "/transcribe/transcribe",
                files={"file": ("test.wav", BytesIO(sample_audio_bytes), "audio/wav")}
            )
            
            assert response.status_code == 400
            assert "Model not loaded" in response.json()["detail"]
            mock_transcription_service.transcribe.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_transcribe_no_file(self, client, mock_transcription_service):
        """Test transcription without providing a file."""
        mock_transcription_service.is_loaded.return_value = True
        
        with patch('app.routers.transcribe.get_transcription_service', return_value=mock_transcription_service):
            response = client.post("/transcribe/transcribe")
            
            assert response.status_code == 422  # Unprocessable Entity
    
    @pytest.mark.asyncio
    async def test_transcribe_invalid_audio(self, client, mock_transcription_service):
        """Test transcription with invalid audio data."""
        mock_transcription_service.is_loaded.return_value = True
        mock_transcription_service.transcribe.side_effect = ValueError("Invalid audio format")
        
        invalid_data = b"not valid audio data"
        
        with patch('app.routers.transcribe.get_transcription_service', return_value=mock_transcription_service):
            response = client.post(
                "/transcribe/transcribe",
                files={"file": ("test.wav", BytesIO(invalid_data), "audio/wav")}
            )
            
            assert response.status_code == 400
            assert "Invalid audio format" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_transcribe_with_language(self, client, mock_transcription_service, sample_audio_bytes):
        """Test transcription with specified language."""
        mock_transcription_service.is_loaded.return_value = True
        mock_transcription_service.transcribe.return_value = "bonjour"
        
        with patch('app.routers.transcribe.get_transcription_service', return_value=mock_transcription_service):
            response = client.post(
                "/transcribe/transcribe",
                files={"file": ("test.wav", BytesIO(sample_audio_bytes), "audio/wav")},
                data={"language": "fr"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["text"] == "bonjour"
            # Verify language parameter was passed to transcribe
            call_kwargs = mock_transcription_service.transcribe.call_args[1]
            assert call_kwargs.get("language") == "fr"
    
    @pytest.mark.asyncio
    async def test_transcribe_error(self, client, mock_transcription_service, sample_audio_bytes):
        """Test transcription with unexpected error."""
        mock_transcription_service.is_loaded.return_value = True
        mock_transcription_service.transcribe.side_effect = RuntimeError("Transcription failed")
        
        with patch('app.routers.transcribe.get_transcription_service', return_value=mock_transcription_service):
            response = client.post(
                "/transcribe/transcribe",
                files={"file": ("test.wav", BytesIO(sample_audio_bytes), "audio/wav")}
            )
            
            assert response.status_code == 500
            assert "Transcription failed" in response.json()["detail"]


class TestUnloadModel:
    """Test cases for POST /transcribe/unload endpoint."""
    
    @pytest.mark.asyncio
    async def test_unload_model_success(self, client, mock_transcription_service):
        """Test successful model unloading."""
        mock_transcription_service.is_loaded.return_value = True
        
        with patch('app.routers.transcribe.get_transcription_service', return_value=mock_transcription_service):
            response = client.post("/transcribe/unload")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unloaded"
            mock_transcription_service.unload_model.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_unload_model_not_loaded(self, client, mock_transcription_service):
        """Test unloading when model is not loaded."""
        mock_transcription_service.is_loaded.return_value = False
        
        with patch('app.routers.transcribe.get_transcription_service', return_value=mock_transcription_service):
            response = client.post("/transcribe/unload")
            
            assert response.status_code == 400
            assert "No model loaded" in response.json()["detail"]
            mock_transcription_service.unload_model.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_unload_model_error(self, client, mock_transcription_service):
        """Test model unloading with unexpected error."""
        mock_transcription_service.is_loaded.return_value = True
        mock_transcription_service.unload_model.side_effect = RuntimeError("Unload failed")
        
        with patch('app.routers.transcribe.get_transcription_service', return_value=mock_transcription_service):
            response = client.post("/transcribe/unload")
            
            assert response.status_code == 500
            assert "Unload failed" in response.json()["detail"]


class TestErrorHandling:
    """Test cases for error handling across endpoints."""
    
    def test_invalid_endpoint(self, client):
        """Test accessing invalid endpoint."""
        response = client.get("/transcribe/invalid")
        assert response.status_code == 404
    
    def test_wrong_method(self, client):
        """Test using wrong HTTP method."""
        response = client.get("/transcribe/load")
        assert response.status_code == 405
    
    @pytest.mark.asyncio
    async def test_malformed_json(self, client, mock_transcription_service):
        """Test sending malformed JSON to load endpoint."""
        with patch('app.routers.transcribe.get_transcription_service', return_value=mock_transcription_service):
            response = client.post(
                "/transcribe/load",
                data="not valid json",
                headers={"Content-Type": "application/json"}
            )
            assert response.status_code == 422


class TestResponseModels:
    """Test cases for response model validation."""
    
    def test_transcription_response_model(self):
        """Test TranscriptionResponse model validation."""
        response = TranscriptionResponse(text="hello", language="en")
        assert response.text == "hello"
        assert response.language == "en"
    
    def test_model_status_response_model(self):
        """Test ModelStatusResponse model validation."""
        response = ModelStatusResponse(
            is_loaded=True,
            model_name="base.en",
            device="cuda"
        )
        assert response.is_loaded is True
        assert response.model_name == "base.en"
        assert response.device == "cuda"
    
    def test_load_model_request_model(self):
        """Test LoadModelRequest model validation."""
        request = LoadModelRequest(model_name="base.en")
        assert request.model_name == "base.en"