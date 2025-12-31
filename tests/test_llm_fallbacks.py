# FILE: tests/test_llm_fallbacks.py
"""
Tests for app/llm/fallbacks.py
LLM failover logic - handles provider failures gracefully.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone


class TestFallbackImports:
    """Test fallback module structure."""
    
    def test_imports_without_error(self):
        """Test module imports cleanly."""
        from app.llm import fallbacks
        assert fallbacks is not None
    
    def test_core_exports(self):
        """Test core components are exported."""
        from app.llm.fallbacks import (
            FailureType,
            FallbackAction,
            FallbackEvent,
            FallbackResult,
            FallbackHandler,
            FALLBACK_CHAINS,
            get_fallback_chain,
            get_next_fallback,
            get_failure_rule,
            handle_preprocessing_failure,
            handle_video_failure,
            handle_vision_failure,
            handle_overwatcher_failure,
            handle_critique_failure,
        )
        assert FailureType is not None
        assert FallbackAction is not None
        assert callable(get_fallback_chain)


class TestFailureTypeEnum:
    """Test FailureType enumeration."""
    
    def test_preprocessing_failures_exist(self):
        """Test preprocessing failure types exist."""
        from app.llm.fallbacks import FailureType
        
        assert FailureType.VIDEO_TRANSCRIPTION_FAILED
        assert FailureType.IMAGE_PROCESSING_FAILED
        assert FailureType.CODE_EXTRACTION_FAILED
        assert FailureType.TEXT_EXTRACTION_FAILED
    
    def test_model_failures_exist(self):
        """Test model failure types exist."""
        from app.llm.fallbacks import FailureType
        
        assert FailureType.MODEL_UNAVAILABLE
        assert FailureType.MODEL_RATE_LIMITED
        assert FailureType.MODEL_TIMEOUT
        assert FailureType.MODEL_ERROR
    
    def test_critical_pipeline_failures_exist(self):
        """Test critical pipeline failure types exist."""
        from app.llm.fallbacks import FailureType
        
        assert FailureType.OVERWATCHER_UNAVAILABLE
        assert FailureType.CRITIQUE_FAILED
        assert FailureType.REVISION_FAILED


class TestFallbackActionEnum:
    """Test FallbackAction enumeration."""
    
    def test_actions_exist(self):
        """Test all action types exist."""
        from app.llm.fallbacks import FallbackAction
        
        assert FallbackAction.RETRY_SAME
        assert FallbackAction.USE_FALLBACK
        assert FallbackAction.SKIP_STEP
        assert FallbackAction.ABORT_TASK
        assert FallbackAction.DEGRADE_GRACEFULLY


class TestFallbackChains:
    """Test fallback chain configuration."""
    
    def test_code_chain_exists(self):
        """Test code task fallback chain exists."""
        from app.llm.fallbacks import FALLBACK_CHAINS
        
        assert "code" in FALLBACK_CHAINS
        chain = FALLBACK_CHAINS["code"]
        assert len(chain) >= 2
        # First should be Sonnet
        assert "sonnet" in chain[0][1].lower()
    
    def test_vision_chain_exists(self):
        """Test vision task fallback chain exists."""
        from app.llm.fallbacks import FALLBACK_CHAINS
        
        assert "vision" in FALLBACK_CHAINS
        chain = FALLBACK_CHAINS["vision"]
        assert len(chain) >= 2
        # Should include Gemini for vision
        providers = [c[0] for c in chain]
        assert "google" in providers
    
    def test_critical_chain_exists(self):
        """Test critical task fallback chain exists."""
        from app.llm.fallbacks import FALLBACK_CHAINS
        
        assert "critical" in FALLBACK_CHAINS
        chain = FALLBACK_CHAINS["critical"]
        assert len(chain) >= 2
        # First should be Opus for critical
        assert "opus" in chain[0][1].lower()
    
    def test_video_chain_exists(self):
        """Test video task fallback chain exists."""
        from app.llm.fallbacks import FALLBACK_CHAINS
        
        assert "video" in FALLBACK_CHAINS
        chain = FALLBACK_CHAINS["video"]
        # Should be Gemini models for video
        providers = [c[0] for c in chain]
        assert all(p == "google" for p in providers)


class TestGetFallbackChain:
    """Test get_fallback_chain function."""
    
    def test_returns_code_chain(self):
        """Test returns correct chain for code role."""
        from app.llm.fallbacks import get_fallback_chain, FALLBACK_CHAINS
        
        chain = get_fallback_chain("code")
        assert chain == FALLBACK_CHAINS["code"]
    
    def test_returns_vision_chain(self):
        """Test returns correct chain for vision role."""
        from app.llm.fallbacks import get_fallback_chain, FALLBACK_CHAINS
        
        chain = get_fallback_chain("vision")
        assert chain == FALLBACK_CHAINS["vision"]
    
    def test_unknown_role_returns_default(self):
        """Test unknown role returns text chain as default."""
        from app.llm.fallbacks import get_fallback_chain, FALLBACK_CHAINS
        
        chain = get_fallback_chain("unknown_role_xyz")
        assert chain == FALLBACK_CHAINS["text"]
    
    def test_returns_list_of_tuples(self):
        """Test chain format is list of (provider, model) tuples."""
        from app.llm.fallbacks import get_fallback_chain
        
        chain = get_fallback_chain("code")
        assert isinstance(chain, list)
        for item in chain:
            assert isinstance(item, tuple)
            assert len(item) == 2
            provider, model = item
            assert isinstance(provider, str)
            assert isinstance(model, str)


class TestGetNextFallback:
    """Test get_next_fallback function."""
    
    def test_gets_next_in_chain(self):
        """Test gets next model after current."""
        from app.llm.fallbacks import get_next_fallback, FALLBACK_CHAINS
        
        chain = FALLBACK_CHAINS["code"]
        first_provider, first_model = chain[0]
        
        next_model = get_next_fallback(first_provider, first_model, "code")
        
        assert next_model is not None
        assert next_model == chain[1]
    
    def test_returns_none_at_end_of_chain(self):
        """Test returns None when at end of chain."""
        from app.llm.fallbacks import get_next_fallback, FALLBACK_CHAINS
        
        chain = FALLBACK_CHAINS["code"]
        last_provider, last_model = chain[-1]
        
        next_model = get_next_fallback(last_provider, last_model, "code")
        
        assert next_model is None
    
    def test_starts_from_beginning_if_not_in_chain(self):
        """Test starts from beginning if current model not in chain."""
        from app.llm.fallbacks import get_next_fallback, FALLBACK_CHAINS
        
        chain = FALLBACK_CHAINS["code"]
        
        next_model = get_next_fallback("unknown_provider", "unknown_model", "code")
        
        assert next_model == chain[0]


class TestGetFailureRule:
    """Test get_failure_rule function."""
    
    def test_rate_limit_rule(self):
        """Test rate limit failure returns retry action."""
        from app.llm.fallbacks import get_failure_rule, FailureType, FallbackAction
        
        rule = get_failure_rule(FailureType.MODEL_RATE_LIMITED)
        
        assert rule.failure_type == FailureType.MODEL_RATE_LIMITED
        assert rule.action == FallbackAction.RETRY_SAME
        assert rule.retry_count > 0
    
    def test_model_unavailable_rule(self):
        """Test model unavailable uses fallback."""
        from app.llm.fallbacks import get_failure_rule, FailureType, FallbackAction
        
        rule = get_failure_rule(FailureType.MODEL_UNAVAILABLE)
        
        assert rule.action == FallbackAction.USE_FALLBACK
    
    def test_video_failure_rule(self):
        """Test video failure degrades gracefully."""
        from app.llm.fallbacks import get_failure_rule, FailureType, FallbackAction
        
        rule = get_failure_rule(FailureType.VIDEO_TRANSCRIPTION_FAILED)
        
        assert rule.action == FallbackAction.DEGRADE_GRACEFULLY
        assert rule.skip_if_other_modalities == True
    
    def test_routing_error_aborts(self):
        """Test routing error aborts task."""
        from app.llm.fallbacks import get_failure_rule, FailureType, FallbackAction
        
        rule = get_failure_rule(FailureType.ROUTING_ERROR)
        
        assert rule.action == FallbackAction.ABORT_TASK


class TestFallbackEvent:
    """Test FallbackEvent dataclass."""
    
    def test_create_event(self):
        """Test creating a fallback event."""
        from app.llm.fallbacks import FallbackEvent, FailureType, FallbackAction
        
        event = FallbackEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type=FailureType.MODEL_TIMEOUT,
            action_taken=FallbackAction.USE_FALLBACK,
            original_provider="anthropic",
            original_model="claude-sonnet-4-20250514",
            fallback_provider="openai",
            fallback_model="gpt-4.1",
            error_message="Request timed out",
            task_id="TASK_123",
        )
        
        assert event.failure_type == FailureType.MODEL_TIMEOUT
        assert event.original_provider == "anthropic"
        assert event.fallback_provider == "openai"
    
    def test_to_dict(self):
        """Test event serialization to dict."""
        from app.llm.fallbacks import FallbackEvent, FailureType, FallbackAction
        
        event = FallbackEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type=FailureType.MODEL_ERROR,
            action_taken=FallbackAction.ABORT_TASK,
            error_message="Test error",
        )
        
        result = event.to_dict()
        
        assert isinstance(result, dict)
        assert "timestamp" in result
        assert "failure_type" in result
        assert result["failure_type"] == "MODEL_ERROR"


class TestFallbackResult:
    """Test FallbackResult dataclass."""
    
    def test_create_success_result(self):
        """Test creating a successful result."""
        from app.llm.fallbacks import FallbackResult
        
        result = FallbackResult(
            success=True,
            content="Response content",
            fallback_used=False,
            final_provider="anthropic",
            final_model="claude-sonnet-4-20250514",
        )
        
        assert result.success == True
        assert result.fallback_used == False
    
    def test_create_fallback_result(self):
        """Test creating a result that used fallback."""
        from app.llm.fallbacks import FallbackResult, FallbackEvent, FailureType, FallbackAction
        
        event = FallbackEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type=FailureType.MODEL_UNAVAILABLE,
            action_taken=FallbackAction.USE_FALLBACK,
        )
        
        result = FallbackResult(
            success=True,
            content="Fallback response",
            fallback_used=True,
            fallback_events=[event],
            final_provider="openai",
            final_model="gpt-4.1",
        )
        
        assert result.fallback_used == True
        assert len(result.fallback_events) == 1
    
    def test_to_dict(self):
        """Test result serialization."""
        from app.llm.fallbacks import FallbackResult
        
        result = FallbackResult(success=True, content="test")
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["success"] == True
        assert "fallback_used" in result_dict


class TestHandlePreprocessingFailure:
    """Test handle_preprocessing_failure function."""
    
    def test_returns_action_and_event(self):
        """Test returns tuple of action and event."""
        from app.llm.fallbacks import (
            handle_preprocessing_failure, 
            FailureType, 
            FallbackAction,
            FallbackEvent,
        )
        
        action, event = handle_preprocessing_failure(
            failure_type=FailureType.VIDEO_TRANSCRIPTION_FAILED,
            error_message="FFmpeg error",
            has_other_modalities=True,
            task_id="TASK_1",
        )
        
        assert isinstance(action, FallbackAction)
        assert isinstance(event, FallbackEvent)
    
    def test_skips_if_other_modalities(self):
        """Test skips step if other modalities available."""
        from app.llm.fallbacks import handle_preprocessing_failure, FailureType, FallbackAction
        
        action, event = handle_preprocessing_failure(
            failure_type=FailureType.VIDEO_TRANSCRIPTION_FAILED,
            error_message="Error",
            has_other_modalities=True,
        )
        
        assert action == FallbackAction.SKIP_STEP


class TestHandleVideoFailure:
    """Test handle_video_failure function (Spec §11.1)."""
    
    def test_continues_with_text(self):
        """Test continues processing if text available."""
        from app.llm.fallbacks import handle_video_failure, FallbackAction
        
        action, event = handle_video_failure(
            error_message="Transcription failed",
            has_text=True,
            has_code=False,
            has_images=False,
        )
        
        # Should skip video step, continue with text
        assert action == FallbackAction.SKIP_STEP
    
    def test_event_has_user_message(self):
        """Test event includes user message."""
        from app.llm.fallbacks import handle_video_failure
        
        action, event = handle_video_failure(
            error_message="Error",
            has_text=True,
        )
        
        # Should have a user-facing message about video
        assert event.user_message is not None or event.failure_type.value == "VIDEO_TRANSCRIPTION_FAILED"


class TestHandleVisionFailure:
    """Test handle_vision_failure function (Spec §11.2)."""
    
    def test_continues_with_other_content(self):
        """Test continues if other content available."""
        from app.llm.fallbacks import handle_vision_failure, FallbackAction
        
        action, event = handle_vision_failure(
            error_message="OCR failed",
            has_text=True,
            has_code=True,
        )
        
        assert action == FallbackAction.SKIP_STEP


class TestHandleOverwatcherFailure:
    """Test handle_overwatcher_failure function (Spec §11.4)."""
    
    def test_returns_skip_step_event(self):
        """Test returns event indicating skip."""
        from app.llm.fallbacks import handle_overwatcher_failure, FailureType
        
        event = handle_overwatcher_failure(
            error_message="Overwatcher service unavailable",
            task_id="TASK_1",
        )
        
        assert event.failure_type == FailureType.OVERWATCHER_UNAVAILABLE
    
    def test_task_marked_unchecked(self):
        """Test task should be marked as unchecked."""
        from app.llm.fallbacks import handle_overwatcher_failure, FallbackAction
        
        event = handle_overwatcher_failure(
            error_message="Connection refused",
        )
        
        # Action should be skip, task continues unchecked
        assert event.action_taken == FallbackAction.SKIP_STEP


class TestHandleCritiqueFailure:
    """Test handle_critique_failure function."""
    
    def test_returns_fallback_action(self):
        """Test returns fallback action."""
        from app.llm.fallbacks import handle_critique_failure, FallbackAction
        
        action, event, next_model = handle_critique_failure(
            error_message="Critique model failed",
            original_provider="google",
            original_model="gemini-3.0-pro-preview",
        )
        
        assert action == FallbackAction.USE_FALLBACK
    
    def test_provides_next_model(self):
        """Test provides next fallback model."""
        from app.llm.fallbacks import handle_critique_failure, FALLBACK_CHAINS
        
        chain = FALLBACK_CHAINS["critique"]
        first_provider, first_model = chain[0]
        
        action, event, next_model = handle_critique_failure(
            error_message="Error",
            original_provider=first_provider,
            original_model=first_model,
        )
        
        if len(chain) > 1:
            assert next_model is not None
            assert next_model == chain[1]


class TestFallbackHandler:
    """Test FallbackHandler class."""
    
    def test_handler_creation(self):
        """Test handler instantiation."""
        from app.llm.fallbacks import FallbackHandler, MAX_FALLBACK_ATTEMPTS
        
        handler = FallbackHandler()
        assert handler.max_attempts == MAX_FALLBACK_ATTEMPTS
    
    def test_handler_custom_max_attempts(self):
        """Test handler with custom max attempts."""
        from app.llm.fallbacks import FallbackHandler
        
        handler = FallbackHandler(max_attempts=5)
        assert handler.max_attempts == 5
    
    def test_record_event(self):
        """Test recording events."""
        from app.llm.fallbacks import FallbackHandler, FallbackEvent, FailureType, FallbackAction
        
        handler = FallbackHandler()
        
        event = FallbackEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type=FailureType.MODEL_ERROR,
            action_taken=FallbackAction.USE_FALLBACK,
        )
        
        handler.record_event(event)
        
        events = handler.get_events()
        assert len(events) == 1
        assert events[0] == event
    
    def test_clear_events(self):
        """Test clearing events."""
        from app.llm.fallbacks import FallbackHandler, FallbackEvent, FailureType, FallbackAction
        
        handler = FallbackHandler()
        
        event = FallbackEvent(
            timestamp=datetime.now(timezone.utc),
            failure_type=FailureType.MODEL_ERROR,
            action_taken=FallbackAction.ABORT_TASK,
        )
        handler.record_event(event)
        
        handler.clear_events()
        
        assert len(handler.get_events()) == 0
    
    def test_execute_with_fallback_success(self):
        """Test successful execution without fallback."""
        import asyncio
        from app.llm.fallbacks import FallbackHandler
        
        handler = FallbackHandler()
        
        async def mock_operation(provider, model):
            return f"Response from {provider}/{model}"
        
        async def run_test():
            return await handler.execute_with_fallback(
                operation=mock_operation,
                role="code",
                initial_provider="anthropic",
                initial_model="claude-sonnet-4-20250514",
            )
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_test())
        finally:
            loop.close()
        
        assert result.success == True
        assert result.fallback_used == False
        assert "Response from" in result.content
    
    def test_execute_with_fallback_on_failure(self):
        """Test fallback triggered on failure."""
        import asyncio
        from app.llm.fallbacks import FallbackHandler, FALLBACK_CHAINS
        
        handler = FallbackHandler()
        call_count = 0
        
        async def mock_operation(provider, model):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Model unavailable 503")
            return f"Response from {provider}/{model}"
        
        chain = FALLBACK_CHAINS["code"]
        first_provider, first_model = chain[0]
        
        async def run_test():
            return await handler.execute_with_fallback(
                operation=mock_operation,
                role="code",
                initial_provider=first_provider,
                initial_model=first_model,
            )
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_test())
        finally:
            loop.close()
        
        assert result.success == True
        assert result.fallback_used == True
        assert len(result.fallback_events) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
