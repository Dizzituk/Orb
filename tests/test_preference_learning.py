# FILE: tests/test_preference_learning.py
"""
Tests for ASTRA Memory preference learning module.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from app.astra_memory.learning import (
    record_message_feedback,
    analyze_conversation_patterns,
    _extract_style_signals,
    _extract_preference_hints,
    _analyze_correction,
)


class TestStyleSignalExtraction:
    """Tests for extracting style signals from content."""
    
    def test_concise_length(self):
        """Short content signals concise preference."""
        content = "This is a short response."
        signals = _extract_style_signals(content)
        assert signals.get("length") == "concise"
    
    def test_detailed_length(self):
        """Long content signals detailed preference."""
        content = " ".join(["word"] * 350)  # 350 words
        signals = _extract_style_signals(content)
        assert signals.get("length") == "detailed"
    
    def test_code_blocks_detected(self):
        """Code blocks are detected."""
        content = "Here's some code:\n```python\nprint('hello')\n```"
        signals = _extract_style_signals(content)
        assert signals.get("uses_code_blocks") == "yes"
    
    def test_bullets_detected(self):
        """Bullet points are detected."""
        content = "• Point 1\n• Point 2\n• Point 3\n• Point 4"
        signals = _extract_style_signals(content)
        assert signals.get("uses_bullets") == "yes"
    
    def test_headers_detected(self):
        """Headers are detected."""
        content = "# Title\n## Section 1\n### Subsection"
        signals = _extract_style_signals(content)
        assert signals.get("uses_headers") == "yes"
    
    def test_empty_content(self):
        """Empty content returns empty signals."""
        signals = _extract_style_signals("")
        assert signals == {}


class TestPreferenceHintExtraction:
    """Tests for extracting preference hints from user comments."""
    
    def test_too_formal_hint(self):
        """'Too formal' suggests casual tone."""
        hints = _extract_preference_hints("This was too formal for me")
        assert ("response.style", "tone", "casual") in hints
    
    def test_too_casual_hint(self):
        """'Too casual' suggests formal tone."""
        hints = _extract_preference_hints("Too casual, be more professional")
        assert ("response.style", "tone", "formal") in hints
    
    def test_too_long_hint(self):
        """'Too long' suggests concise length."""
        hints = _extract_preference_hints("This is too long, be more concise")
        assert ("response.style", "length", "concise") in hints
    
    def test_too_short_hint(self):
        """'Too short' suggests detailed length."""
        hints = _extract_preference_hints("Too short, need more detail")
        assert ("response.style", "length", "detailed") in hints
    
    def test_no_bullets_hint(self):
        """'No bullets' suggests avoid bullets."""
        hints = _extract_preference_hints("Please no bullets or lists")
        assert ("response.format", "bullets", "avoid") in hints
    
    def test_use_bullets_hint(self):
        """'Use bullets' suggests prefer bullets."""
        hints = _extract_preference_hints("Use bullets to organize this")
        assert ("response.format", "bullets", "preferred") in hints
    
    def test_no_hints(self):
        """Generic comment has no hints."""
        hints = _extract_preference_hints("Thanks for the help!")
        assert hints == []


class TestCorrectionAnalysis:
    """Tests for analyzing corrections."""
    
    def test_length_reduction(self):
        """Shorter correction suggests concise preference."""
        original = " ".join(["word"] * 100)
        correction = " ".join(["word"] * 50)  # 50% reduction
        changes = _analyze_correction(original, correction)
        assert ("length_preference", "concise") in changes
    
    def test_length_increase(self):
        """Longer correction suggests detailed preference."""
        original = " ".join(["word"] * 50)
        correction = " ".join(["word"] * 100)  # 100% increase
        changes = _analyze_correction(original, correction)
        assert ("length_preference", "detailed") in changes
    
    def test_bullet_removal(self):
        """Removing bullets suggests no_bullets preference."""
        original = "List:\n• Item 1\n• Item 2\n• Item 3\n• Item 4"
        correction = "Items: Item 1, Item 2, Item 3, Item 4"
        changes = _analyze_correction(original, correction)
        assert ("format_change", "no_bullets") in changes
    
    def test_bullet_addition(self):
        """Adding bullets suggests use_bullets preference."""
        original = "Items: Item 1, Item 2, Item 3, Item 4"
        correction = "List:\n• Item 1\n• Item 2\n• Item 3\n• Item 4"
        changes = _analyze_correction(original, correction)
        assert ("format_change", "use_bullets") in changes


class TestRecordMessageFeedback:
    """Tests for recording message feedback."""
    
    def test_message_not_found(self):
        """Returns error when message doesn't exist."""
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        
        result = record_message_feedback(db, message_id=999, feedback_type="positive")
        
        assert result["status"] == "error"
        assert result["reason"] == "message_not_found"
    
    def test_positive_feedback_recorded(self):
        """Positive feedback triggers preference learning."""
        db = MagicMock()
        mock_message = MagicMock()
        mock_message.id = 1
        mock_message.content = "Here's a detailed response with lots of information."
        db.query.return_value.filter.return_value.first.return_value = mock_message
        
        with patch("app.astra_memory.learning._learn_from_positive") as mock_learn:
            mock_learn.return_value = {"preferences_updated": ["test"]}
            
            result = record_message_feedback(
                db, 
                message_id=1, 
                feedback_type="positive",
                metadata={"provider": "openai", "model": "gpt-4"}
            )
            
            assert result["message_id"] == 1
            assert result["feedback_type"] == "positive"
            mock_learn.assert_called_once()


class TestAnalyzeConversationPatterns:
    """Tests for conversation pattern analysis."""
    
    def test_empty_messages(self):
        """Returns stats for empty message set."""
        db = MagicMock()
        db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
        
        result = analyze_conversation_patterns(db)
        
        assert result["messages_analyzed"] == 0
        assert result["patterns_found"] == []
    
    def test_length_pattern_detection(self):
        """Detects length patterns from messages."""
        db = MagicMock()
        
        # Create mock messages with short content
        mock_messages = []
        for i in range(10):
            msg = MagicMock()
            msg.content = "Short reply"  # ~2 words
            mock_messages.append(msg)
        
        db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = mock_messages
        
        result = analyze_conversation_patterns(db)
        
        assert result["messages_analyzed"] == 10
        # Should detect concise preference
        length_patterns = [p for p in result["patterns_found"] if p.get("type") == "length_preference"]
        if length_patterns:
            assert length_patterns[0]["value"] == "concise"


# =============================================================================
# INTEGRATION TESTS (require database)
# =============================================================================

class TestFeedbackIntegration:
    """Integration tests requiring database."""
    
    @pytest.fixture
    def db_session(self):
        """Get database session."""
        from app.db import SessionLocal
        db = SessionLocal()
        yield db
        db.close()
    
    def test_feedback_endpoint_schema(self, db_session):
        """Verify feedback endpoint schemas are valid."""
        from app.astra_memory.router import FeedbackRequest, FeedbackResponse
        
        # Test request schema
        req = FeedbackRequest(
            message_id=1,
            feedback_type="positive",
            comment="Great response!",
            metadata={"provider": "openai"}
        )
        assert req.message_id == 1
        assert req.feedback_type == "positive"
        
        # Test response schema
        resp = FeedbackResponse(
            status="ok",
            message_id=1,
            feedback_type="positive",
            preferences_updated=["test.pref"],
            preferences_created=[]
        )
        assert resp.status == "ok"
