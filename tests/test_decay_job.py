# FILE: tests/test_decay_job.py
"""
Tests for ASTRA Memory confidence decay job.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

from app.astra_memory.decay_job import (
    DecayJobConfig,
    run_confidence_decay,
    get_decay_metrics,
    DecayJobScheduler,
    _recompute_all_preferences,
    _expire_low_confidence,
    _clean_stale_hot_index,
)
from app.astra_memory.preference_models import RecordStatus


class TestDecayJobConfig:
    """Tests for decay job configuration."""
    
    def test_default_config(self):
        """Default config has reasonable values."""
        config = DecayJobConfig()
        
        assert config.expire_below_confidence == 0.15
        assert config.dispute_threshold == 0.3
        assert config.hot_index_stale_days == 90
        assert config.max_preferences_per_run == 1000
    
    def test_custom_config(self):
        """Custom config values work."""
        config = DecayJobConfig()
        config.expire_below_confidence = 0.1
        config.dispute_threshold = 0.25
        
        assert config.expire_below_confidence == 0.1
        assert config.dispute_threshold == 0.25


class TestRecomputePreferences:
    """Tests for preference recomputation."""
    
    def test_empty_preferences(self):
        """Handles empty preference set."""
        db = MagicMock()
        db.query.return_value.filter.return_value.limit.return_value.all.return_value = []
        
        config = DecayJobConfig()
        result = _recompute_all_preferences(db, config)
        
        assert result["recomputed"] == 0
        assert result["errors"] == []
    
    def test_recompute_multiple(self):
        """Recomputes multiple preferences."""
        db = MagicMock()
        
        # Mock preferences
        mock_prefs = [MagicMock(id=1), MagicMock(id=2), MagicMock(id=3)]
        db.query.return_value.filter.return_value.limit.return_value.all.return_value = mock_prefs
        
        config = DecayJobConfig()
        
        with patch("app.astra_memory.decay_job.recompute_preference_confidence"):
            result = _recompute_all_preferences(db, config)
        
        assert result["recomputed"] == 3


class TestExpireLowConfidence:
    """Tests for expiring low confidence preferences."""
    
    def test_expire_very_low(self):
        """Very low confidence preferences get expired."""
        db = MagicMock()
        
        # Mock preference with very low confidence
        mock_pref = MagicMock()
        mock_pref.confidence = 0.1  # Below 0.15 threshold
        mock_pref.namespace = "test"
        mock_pref.key = "pref"
        db.query.return_value.filter.return_value.limit.return_value.all.return_value = [mock_pref]
        
        config = DecayJobConfig()
        result = _expire_low_confidence(db, config)
        
        assert result["expired"] == 1
        assert mock_pref.status == RecordStatus.EXPIRED
    
    def test_dispute_low(self):
        """Low confidence (but not very low) preferences get disputed."""
        db = MagicMock()
        
        # Mock preference with low confidence (between thresholds)
        mock_pref = MagicMock()
        mock_pref.confidence = 0.25  # Below 0.3 but above 0.15
        mock_pref.namespace = "test"
        mock_pref.key = "pref"
        db.query.return_value.filter.return_value.limit.return_value.all.return_value = [mock_pref]
        
        config = DecayJobConfig()
        result = _expire_low_confidence(db, config)
        
        assert result["disputed"] == 1
        assert mock_pref.status == RecordStatus.DISPUTED


class TestCleanStaleHotIndex:
    """Tests for cleaning stale hot index entries."""
    
    def test_no_stale_entries(self):
        """No cleanup when no stale entries."""
        db = MagicMock()
        db.query.return_value.filter.return_value.limit.return_value.all.return_value = []
        
        config = DecayJobConfig()
        result = _clean_stale_hot_index(db, config)
        
        assert result["cleaned"] == 0
    
    def test_clean_stale(self):
        """Stale entries get cleaned."""
        db = MagicMock()
        
        # Mock stale entries
        mock_entries = [MagicMock(), MagicMock()]
        db.query.return_value.filter.return_value.limit.return_value.all.return_value = mock_entries
        
        config = DecayJobConfig()
        result = _clean_stale_hot_index(db, config)
        
        assert result["cleaned"] == 2
        assert db.delete.call_count == 2


class TestRunConfidenceDecay:
    """Tests for full decay job run."""
    
    def test_full_run(self):
        """Full decay job runs all steps."""
        db = MagicMock()
        
        # Mock all queries to return empty
        db.query.return_value.filter.return_value.limit.return_value.all.return_value = []
        
        with patch("app.astra_memory.decay_job._recompute_all_preferences") as mock_recompute, \
             patch("app.astra_memory.decay_job._expire_low_confidence") as mock_expire, \
             patch("app.astra_memory.decay_job._clean_stale_hot_index") as mock_clean:
            
            mock_recompute.return_value = {"recomputed": 5, "errors": []}
            mock_expire.return_value = {"expired": 1, "disputed": 2}
            mock_clean.return_value = {"cleaned": 3}
            
            result = run_confidence_decay(db)
        
        assert result["preferences_recomputed"] == 5
        assert result["preferences_expired"] == 1
        assert result["preferences_disputed"] == 2
        assert result["hot_index_cleaned"] == 3
        assert "started_at" in result
        assert "completed_at" in result
        assert "duration_seconds" in result


class TestDecayMetrics:
    """Tests for decay metrics."""
    
    def test_empty_metrics(self):
        """Handles empty database."""
        db = MagicMock()
        
        # Mock empty queries
        db.query.return_value.group_by.return_value.all.return_value = []
        db.query.return_value.filter.return_value.all.return_value = []
        db.query.return_value.scalar.return_value = 0
        
        metrics = get_decay_metrics(db)
        
        assert "preferences_by_status" in metrics
        assert "confidence_distribution" in metrics
        assert "total_evidence_records" in metrics
        assert "hot_index_entries" in metrics


class TestDecayJobScheduler:
    """Tests for scheduler."""
    
    def test_initial_status(self):
        """Scheduler starts in stopped state."""
        scheduler = DecayJobScheduler(interval_hours=12)
        
        status = scheduler.get_status()
        
        assert status["running"] == False
        assert status["interval_hours"] == 12
        assert status["last_run"] is None
    
    def test_run_now(self):
        """Manual run works."""
        scheduler = DecayJobScheduler()
        
        with patch("app.astra_memory.decay_job.run_confidence_decay") as mock_run, \
             patch("app.db.SessionLocal") as mock_session:
            
            mock_db = MagicMock()
            mock_session.return_value = mock_db
            mock_run.return_value = {
                "preferences_recomputed": 10,
                "preferences_expired": 0,
                "preferences_disputed": 0,
                "hot_index_cleaned": 0,
                "errors": [],
                "started_at": "2025-01-01T00:00:00Z",
                "completed_at": "2025-01-01T00:00:01Z",
                "duration_seconds": 1.0,
            }
            
            result = scheduler.run_now()
            
            assert result["preferences_recomputed"] == 10
            assert scheduler._last_run is not None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestDecayJobIntegration:
    """Integration tests requiring database."""
    
    @pytest.fixture
    def db_session(self):
        """Get database session."""
        from app.db import SessionLocal
        db = SessionLocal()
        yield db
        db.close()
    
    def test_metrics_query(self, db_session):
        """Can query metrics from real database."""
        metrics = get_decay_metrics(db_session)
        
        assert isinstance(metrics["preferences_by_status"], dict)
        assert isinstance(metrics["confidence_distribution"], dict)
        assert isinstance(metrics["total_evidence_records"], int)
        assert isinstance(metrics["hot_index_entries"], int)
    
    def test_full_decay_run(self, db_session):
        """Can run full decay job on real database."""
        result = run_confidence_decay(db_session)
        
        assert "preferences_recomputed" in result
        assert "preferences_expired" in result
        assert "duration_seconds" in result
        assert result["errors"] == [] or isinstance(result["errors"], list)
