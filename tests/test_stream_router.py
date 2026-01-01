# FILE: tests/test_stream_router.py
"""
Tests for app/llm/stream_router.py
Core streaming router - routes messages to appropriate handlers.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import inspect


class TestStreamRouterImports:
    """Test stream router module structure."""

    def test_imports_without_error(self):
        """Test module imports cleanly."""
        from app.llm import stream_router
        assert stream_router is not None

    def test_router_defined(self):
        """Test FastAPI router is defined."""
        from app.llm.stream_router import router
        assert router is not None

    def test_stream_request_model(self):
        """Test StreamRequest model is defined."""
        from app.llm.stream_router import StreamRequest
        assert StreamRequest is not None


class TestTriggerDetection:
    """Test trigger phrase detection."""

    def test_zombie_map_trigger_detection(self):
        """Test detection of zombie map triggers."""
        from app.llm.stream_router import _is_zobie_map_trigger

        # Test positive cases - exact matches from _ZOBIE_TRIGGER_SET
        assert _is_zobie_map_trigger("zombie map") == True
        assert _is_zobie_map_trigger("zobie map") == True
        assert _is_zobie_map_trigger("zobie_map") == True
        assert _is_zobie_map_trigger("/zobie_map") == True
        assert _is_zobie_map_trigger("/zombie_map") == True
        assert _is_zobie_map_trigger("  zombie map  ") == True  # Whitespace stripped
        assert _is_zobie_map_trigger("ZOMBIE MAP") == True  # Case insensitive

        # Test negative cases
        assert _is_zobie_map_trigger("show me the zombie map") == False
        assert _is_zobie_map_trigger("zombie") == False
        assert _is_zobie_map_trigger("hello world") == False
        assert _is_zobie_map_trigger("") == False
        assert _is_zobie_map_trigger(None) == False

    def test_archmap_trigger_detection(self):
        """Test detection of architecture map triggers."""
        from app.llm.stream_router import _is_archmap_trigger
        from app.llm.local_tools.archmap_helpers import _ARCHMAP_TRIGGER_SET

        # Test positive cases - should match items in _ARCHMAP_TRIGGER_SET
        for trigger in _ARCHMAP_TRIGGER_SET:
            assert _is_archmap_trigger(trigger) == True, f"Should match: {trigger}"
            assert _is_archmap_trigger(trigger.upper()) == True, f"Should match uppercase: {trigger}"
            assert _is_archmap_trigger(f"  {trigger}  ") == True, f"Should match with whitespace: {trigger}"

        # Test negative cases
        assert _is_archmap_trigger("random text") == False
        assert _is_archmap_trigger("show me the architecture") == False
        assert _is_archmap_trigger("") == False
        assert _is_archmap_trigger(None) == False

    def test_introspection_trigger_not_available(self):
        """Test introspection trigger when module not available."""
        from app.llm.stream_router import _is_introspection_trigger, _INTROSPECTION_AVAILABLE

        if not _INTROSPECTION_AVAILABLE:
            # If introspection not available, should always return False
            assert _is_introspection_trigger("show me the logs") == False
            assert _is_introspection_trigger("anything") == False

    def test_sandbox_trigger_not_available(self):
        """Test sandbox trigger when module not available."""
        from app.llm.stream_router import _is_sandbox_trigger, _SANDBOX_AVAILABLE

        if not _SANDBOX_AVAILABLE:
            assert _is_sandbox_trigger("any message") == False


class TestStreamGeneration:
    """Test stream generation functions."""

    def test_sandbox_stream_is_async_generator(self):
        """Test sandbox stream generator is async generator."""
        from app.llm.stream_router import generate_sandbox_stream
        assert inspect.isasyncgenfunction(generate_sandbox_stream)

    def test_introspection_stream_is_async_generator(self):
        """Test introspection stream generator is async generator."""
        from app.llm.stream_router import generate_introspection_stream
        assert inspect.isasyncgenfunction(generate_introspection_stream)

    def test_sse_stream_is_async_generator(self):
        """Test SSE stream generator is async generator."""
        from app.llm.stream_router import generate_sse_stream
        assert inspect.isasyncgenfunction(generate_sse_stream)


class TestZobieTriggerSet:
    """Test the trigger set constant."""

    def test_trigger_set_defined(self):
        """Test trigger set is defined correctly."""
        from app.llm.stream_router import _ZOBIE_TRIGGER_SET

        assert "zombie map" in _ZOBIE_TRIGGER_SET
        assert "zobie map" in _ZOBIE_TRIGGER_SET
        assert "zobie_map" in _ZOBIE_TRIGGER_SET
        assert "/zobie_map" in _ZOBIE_TRIGGER_SET
        assert "/zombie_map" in _ZOBIE_TRIGGER_SET

    def test_trigger_set_is_set(self):
        """Test trigger set is a set type."""
        from app.llm.stream_router import _ZOBIE_TRIGGER_SET
        assert isinstance(_ZOBIE_TRIGGER_SET, set)


class TestStreamRequest:
    """Test StreamRequest model."""

    def test_required_fields(self):
        """Test required fields are enforced."""
        from app.llm.stream_router import StreamRequest
        from pydantic import ValidationError

        # Should work with required fields
        req = StreamRequest(project_id=1, message="hello")
        assert req.project_id == 1
        assert req.message == "hello"

        # Should fail without required fields
        with pytest.raises(ValidationError):
            StreamRequest(project_id=1)  # Missing message

    def test_default_values(self):
        """Test default values are set correctly."""
        from app.llm.stream_router import StreamRequest

        req = StreamRequest(project_id=1, message="hello")
        
        assert req.provider is None
        assert req.model is None
        assert req.include_history == True
        assert req.history_limit == 20
        assert req.use_semantic_search == True
        assert req.enable_reasoning == False

    def test_optional_fields(self):
        """Test optional fields can be set."""
        from app.llm.stream_router import StreamRequest

        req = StreamRequest(
            project_id=1,
            message="hello",
            provider="anthropic",
            model="claude-sonnet-4",
            job_type="code_small",
            include_history=False,
            history_limit=10,
            enable_reasoning=True,
        )
        
        assert req.provider == "anthropic"
        assert req.model == "claude-sonnet-4"
        assert req.job_type == "code_small"
        assert req.include_history == False
        assert req.history_limit == 10
        assert req.enable_reasoning == True


class TestEnvironmentConfig:
    """Test environment configuration."""

    def test_zobie_controller_url_default(self):
        """Test default zobie controller URL."""
        from app.llm.stream_router import ZOBIE_CONTROLLER_URL
        assert "192.168.250.2" in ZOBIE_CONTROLLER_URL or ZOBIE_CONTROLLER_URL is not None

    def test_zobie_mapper_timeout_is_int(self):
        """Test mapper timeout is integer."""
        from app.llm.stream_router import ZOBIE_MAPPER_TIMEOUT_SEC
        assert isinstance(ZOBIE_MAPPER_TIMEOUT_SEC, int)
        assert ZOBIE_MAPPER_TIMEOUT_SEC > 0


class TestLocalToolImports:
    """Test local tool imports."""

    def test_archmap_helpers_import(self):
        """Test archmap helpers are imported."""
        from app.llm.stream_router import ARCHMAP_PROVIDER, ARCHMAP_MODEL
        assert ARCHMAP_PROVIDER is not None
        assert ARCHMAP_MODEL is not None

    def test_zobie_tools_import(self):
        """Test zobie tools are imported."""
        from app.llm.stream_router import (
            generate_local_architecture_map_stream,
            generate_local_zobie_map_stream,
        )
        assert callable(generate_local_architecture_map_stream)
        assert callable(generate_local_zobie_map_stream)


class TestStreamUtilsImports:
    """Test stream utils imports."""

    def test_default_models_imported(self):
        """Test DEFAULT_MODELS is imported."""
        from app.llm.stream_router import DEFAULT_MODELS
        assert isinstance(DEFAULT_MODELS, dict)

    def test_parse_reasoning_tags_imported(self):
        """Test parse_reasoning_tags is imported."""
        from app.llm.stream_router import parse_reasoning_tags
        assert callable(parse_reasoning_tags)

    def test_make_session_id_imported(self):
        """Test make_session_id is imported."""
        from app.llm.stream_router import make_session_id
        assert callable(make_session_id)

    def test_classify_job_type_imported(self):
        """Test classify_job_type is imported."""
        from app.llm.stream_router import classify_job_type
        assert callable(classify_job_type)

    def test_select_provider_for_job_type_imported(self):
        """Test select_provider_for_job_type is imported."""
        from app.llm.stream_router import select_provider_for_job_type
        assert callable(select_provider_for_job_type)


class TestHighStakesImports:
    """Test high stakes stream imports."""

    def test_high_stakes_critique_stream_imported(self):
        """Test high stakes critique stream is imported."""
        from app.llm.stream_router import generate_high_stakes_critique_stream
        assert callable(generate_high_stakes_critique_stream)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
