import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from app.models.project import Project
from app.models.clip import Clip
from app.models.caption import Caption
from app.models.asset import Asset
from app.models.speaker import Speaker
from app.models.transcription import Transcription
from app.services.project_service import ProjectService
from app.services.transcription_service import TranscriptionService
from app.services.diarization_service import DiarizationService
from app.services.model_manager import get_model_manager
from app.database import get_db


@pytest.fixture
def mock_db():
    """Mock database session."""
    db = Mock()
    db.query.return_value.filter.return_value.first.return_value = None
    db.commit = Mock()
    db.add = Mock()
    db.delete = Mock()
    db.refresh = Mock()
    db.rollback = Mock()
    return db


@pytest.fixture
def temp_project_dir():
    """Create temporary project directory."""
    temp_dir = tempfile.mkdtemp()
    project_dir = Path(temp_dir) / "project_123"
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (project_dir / "audio").mkdir(exist_ok=True)
    (project_dir / "clips").mkdir(exist_ok=True)
    (project_dir / "exports").mkdir(exist_ok=True)
    (project_dir / "models").mkdir(exist_ok=True)
    
    yield project_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_project(temp_project_dir):
    """Create sample project with Phase-4 artifacts."""
    project = Project(
        id="project_123",
        name="Test Project",
        created_at=datetime.utcnow(),
        project_dir=str(temp_project_dir)
    )
    
    # Create some files
    audio_file = temp_project_dir / "audio" / "test.wav"
    audio_file.touch()
    
    clip_file = temp_project_dir / "clips" / "clip_001.wav"
    clip_file.touch()
    
    model_file = temp_project_dir / "models" / "speaker_model.pkl"
    model_file.touch()
    
    export_file = temp_project_dir / "exports" / "transcript.txt"
    export_file.touch()
    
    return project


@pytest.fixture
def sample_clips():
    """Create sample clips."""
    return [
        Clip(
            id="clip_001",
            project_id="project_123",
            file_path="/clips/clip_001.wav",
            start_time=0.0,
            end_time=5.0,
            speaker_id="speaker_1"
        ),
        Clip(
            id="clip_002",
            project_id="project_123",
            file_path="/clips/clip_002.wav",
            start_time=5.0,
            end_time=10.0,
            speaker_id="speaker_2"
        )
    ]


@pytest.fixture
def sample_speakers():
    """Create sample speakers."""
    return [
        Speaker(
            id="speaker_1",
            project_id="project_123",
            name="Speaker 1",
            embedding_model_path="/models/speaker_1.pkl"
        ),
        Speaker(
            id="speaker_2",
            project_id="project_123",
            name="Speaker 2",
            embedding_model_path="/models/speaker_2.pkl"
        )
    ]


@pytest.fixture
def sample_captions():
    """Create sample captions."""
    return [
        Caption(
            id="caption_001",
            clip_id="clip_001",
            text="Test caption 1",
            start_time=0.0,
            end_time=2.5
        ),
        Caption(
            id="caption_002",
            clip_id="clip_001",
            text="Test caption 2",
            start_time=2.5,
            end_time=5.0
        )
    ]


@pytest.fixture
def sample_assets():
    """Create sample assets."""
    return [
        Asset(
            id="asset_001",
            project_id="project_123",
            file_path="/audio/test.wav",
            asset_type="audio",
            file_size=1024000
        ),
        Asset(
            id="asset_002",
            project_id="project_123",
            file_path="/exports/transcript.txt",
            asset_type="export",
            file_size=2048
        )
    ]


@pytest.fixture
def sample_transcriptions():
    """Create sample transcriptions."""
    return [
        Transcription(
            id="trans_001",
            clip_id="clip_001",
            text="Transcribed text 1",
            language="en",
            confidence=0.95
        ),
        Transcription(
            id="trans_002",
            clip_id="clip_002",
            text="Transcribed text 2",
            language="en",
            confidence=0.92
        )
    ]


class TestProjectDeletion:
    """Test project deletion functionality."""
    
    def test_delete_project_basic(self, mock_db, sample_project):
        """Test basic project deletion without artifacts."""
        mock_db.query.return_value.filter.return_value.first.return_value = sample_project
        mock_db.query.return_value.filter.return_value.all.return_value = []
        
        service = ProjectService(mock_db)
        
        with patch('shutil.rmtree') as mock_rmtree:
            result = service.delete_project(sample_project.id)
            
            assert result is True
            mock_db.delete.assert_called_once_with(sample_project)
            mock_db.commit.assert_called_once()
            mock_rmtree.assert_called_once()
    
    def test_delete_project_with_clips(self, mock_db, sample_project, sample_clips):
        """Test project deletion with associated clips."""
        mock_db.query.return_value.filter.return_value.first.return_value = sample_project
        
        def query_side_effect(model):
            if model == Clip:
                mock_query = Mock()
                mock_query.filter.return_value.all.return_value = sample_clips
                return mock_query
            mock_query = Mock()
            mock_query.filter.return_value.all.return_value = []
            return mock_query
        
        mock_db.query.side_effect = query_side_effect
        
        service = ProjectService(mock_db)
        
        with patch('shutil.rmtree') as mock_rmtree:
            result = service.delete_project(sample_project.id)
            
            assert result is True
            assert mock_db.delete.call_count >= len(sample_clips) + 1
    
    def test_delete_project_with_speakers(self, mock_db, sample_project, sample_speakers):
        """Test project deletion with speaker models."""
        mock_db.query.return_value.filter.return_value.first.return_value = sample_project
        
        def query_side_effect(model):
            if model == Speaker:
                mock_query = Mock()
                mock_query.filter.return_value.all.return_value = sample_speakers
                return mock_query
            mock_query = Mock()
            mock_query.filter.return_value.all.return_value = []
            return mock_query
        
        mock_db.query.side_effect = query_side_effect
        
        service = ProjectService(mock_db)
        
        with patch('shutil.rmtree') as mock_rmtree:
            with patch('pathlib.Path.unlink') as mock_unlink:
                result = service.delete_project(sample_project.id)
                
                assert result is True
                assert mock_db.delete.call_count >= len(sample_speakers) + 1
    
    def test_delete_project_with_all_artifacts(self, mock_db, sample_project, 
                                              sample_clips, sample_speakers,
                                              sample_captions, sample_assets,
                                              sample_transcriptions):
        """Test project deletion with all Phase-4 artifacts."""
        mock_db.query.return_value.filter.return_value.first.return_value = sample_project
        
        def query_side_effect(model):
            if model == Clip:
                mock_query = Mock()
                mock_query.filter.return_value.all.return_value = sample_clips
                return mock_query
            elif model == Speaker:
                mock_query = Mock()
                mock_query.filter.return_value.all.return_value = sample_speakers
                return mock_query
            elif model == Caption:
                mock_query = Mock()
                mock_query.filter.return_value.all.return_value = sample_captions
                return mock_query
            elif model == Asset:
                mock_query = Mock()
                mock_query.filter.return_value.all.return_value = sample_assets
                return mock_query
            elif model == Transcription:
                mock_query = Mock()
                mock_query.filter.return_value.all.return_value = sample_transcriptions
                return mock_query
            mock_query = Mock()
            mock_query.filter.return_value.all.return_value = []
            return mock_query
        
        mock_db.query.side_effect = query_side_effect
        
        service = ProjectService(mock_db)
        
        with patch('shutil.rmtree') as mock_rmtree:
            result = service.delete_project(sample_project.id)
            
            assert result is True
            # Verify all artifacts were deleted
            total_artifacts = (len(sample_clips) + len(sample_speakers) + 
                             len(sample_captions) + len(sample_assets) + 
                             len(sample_transcriptions) + 1)  # +1 for project
            assert mock_db.delete.call_count >= total_artifacts
    
    def test_delete_nonexistent_project(self, mock_db):
        """Test deletion of non-existent project."""
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        service = ProjectService(mock_db)
        result = service.delete_project("nonexistent_id")
        
        assert result is False
        mock_db.delete.assert_not_called()
    
    def test_delete_project_filesystem_cleanup(self, temp_project_dir, sample_project):
        """Test that filesystem is cleaned up during deletion."""
        # Create some files
        (temp_project_dir / "audio" / "test1.wav").touch()
        (temp_project_dir / "audio" / "test2.wav").touch()
        (temp_project_dir / "clips" / "clip1.wav").touch()
        (temp_project_dir / "models" / "model1.pkl").touch()
        
        assert temp_project_dir.exists()
        assert len(list(temp_project_dir.rglob("*.*"))) > 0
        
        # Delete directory
        shutil.rmtree(temp_project_dir)
        
        assert not temp_project_dir.exists()
    
    def test_delete_project_cascade_captions(self, mock_db, sample_project, 
                                            sample_clips, sample_captions):
        """Test that captions are cascaded when clips are deleted."""
        mock_db.query.return_value.filter.return_value.first.return_value = sample_project
        
        def query_side_effect(model):
            if model == Clip:
                mock_query = Mock()
                mock_query.filter.return_value.all.return_value = sample_clips
                return mock_query
            elif model == Caption:
                mock_query = Mock()
                mock_query.filter.return_value.all.return_value = sample_captions
                return mock_query
            mock_query = Mock()
            mock_query.filter.return_value.all.return_value = []
            return mock_query
        
        mock_db.query.side_effect = query_side_effect
        
        service = ProjectService(mock_db)
        
        with patch('shutil.rmtree'):
            result = service.delete_project(sample_project.id)
            
            assert result is True
            # Verify captions were deleted
            assert mock_db.delete.call_count >= len(sample_captions) + len(sample_clips) + 1
    
    def test_delete_project_with_transcription_service(self, mock_db, sample_project):
        """Test project deletion clears transcription service cache."""
        mock_db.query.return_value.filter.return_value.first.return_value = sample_project
        mock_db.query.return_value.filter.return_value.all.return_value = []
        
        service = ProjectService(mock_db)
        
        with patch('shutil.rmtree'):
            with patch.object(TranscriptionService, 'clear_cache') as mock_clear:
                result = service.delete_project(sample_project.id)
                
                assert result is True
    
    def test_delete_project_with_diarization_service(self, mock_db, sample_project):
        """Test project deletion clears diarization service cache."""
        mock_db.query.return_value.filter.return_value.first.return_value = sample_project
        mock_db.query.return_value.filter.return_value.all.return_value = []
        
        service = ProjectService(mock_db)
        
        with patch('shutil.rmtree'):
            with patch.object(DiarizationService, 'clear_cache') as mock_clear:
                result = service.delete_project(sample_project.id)
                
                assert result is True
    
    def test_delete_project_with_model_manager(self, mock_db, sample_project, sample_speakers):
        """Test project deletion unloads models from model manager."""
        mock_db.query.return_value.filter.return_value.first.return_value = sample_project
        
        def query_side_effect(model):
            if model == Speaker:
                mock_query = Mock()
                mock_query.filter.return_value.all.return_value = sample_speakers
                return mock_query
            mock_query = Mock()
            mock_query.filter.return_value.all.return_value = []
            return mock_query
        
        mock_db.query.side_effect = query_side_effect
        
        service = ProjectService(mock_db)
        
        with patch('shutil.rmtree'):
            with patch('app.services.model_manager.get_model_manager') as mock_get_manager:
                mock_manager = Mock()
                mock_get_manager.return_value = mock_manager
                
                result = service.delete_project(sample_project.id)
                
                assert result is True
    
    def test_delete_project_transaction_rollback(self, mock_db, sample_project):
        """Test transaction rollback on deletion failure."""
        mock_db.query.return_value.filter.return_value.first.return_value = sample_project
        mock_db.query.return_value.filter.return_value.all.return_value = []
        mock_db.commit.side_effect = Exception("Database error")
        
        service = ProjectService(mock_db)
        
        with patch('shutil.rmtree'):
            with pytest.raises(Exception):
                service.delete_project(sample_project.id)
            
            mock_db.rollback.assert_called_once()
    
    def test_delete_project_partial_filesystem_failure(self, mock_db, sample_project):
        """Test handling of partial filesystem deletion failure."""
        mock_db.query.return_value.filter.return_value.first.return