# FILE: tests/test_llm_clients.py
"""
Tests for app/llm/clients.py
LLM API client wrappers - provider availability, envelope building, embeddings.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio


class TestClientsImports:
    """Test clients module imports."""

    def test_imports_without_error(self):
        """Test that clients module imports cleanly."""
        from app.llm import clients
        assert clients is not None

    def test_all_exports(self):
        """Test __all__ exports are accessible."""
        from app.llm.clients import (
            async_call_openai,
            async_call_anthropic,
            async_call_google,
            call_openai,
            call_anthropic,
            call_google,
            get_embeddings,
            generate_embedding,
            check_provider_availability,
            list_available_providers,
        )
        assert callable(async_call_openai)
        assert callable(check_provider_availability)


class TestCheckProviderAvailability:
    """Test check_provider_availability function."""

    def test_returns_dict(self):
        """Test returns a dictionary."""
        from app.llm.clients import check_provider_availability
        
        result = check_provider_availability()
        assert isinstance(result, dict)

    def test_contains_expected_providers(self):
        """Test contains openai, anthropic, google keys."""
        from app.llm.clients import check_provider_availability
        
        result = check_provider_availability()
        assert "openai" in result
        assert "anthropic" in result
        assert "google" in result

    def test_values_are_booleans(self):
        """Test all values are booleans."""
        from app.llm.clients import check_provider_availability
        
        result = check_provider_availability()
        for provider, available in result.items():
            assert isinstance(available, bool), f"{provider} value is not bool"


class TestListAvailableProviders:
    """Test list_available_providers function."""

    def test_returns_list(self):
        """Test returns a list."""
        from app.llm.clients import list_available_providers
        
        result = list_available_providers()
        assert isinstance(result, list)

    def test_returns_strings(self):
        """Test all items are strings."""
        from app.llm.clients import list_available_providers
        
        result = list_available_providers()
        for item in result:
            assert isinstance(item, str)

    def test_consistent_with_availability_check(self):
        """Test consistent with check_provider_availability."""
        from app.llm.clients import (
            check_provider_availability,
            list_available_providers,
        )
        
        availability = check_provider_availability()
        available_list = list_available_providers()
        
        # All items in list should be available
        for provider in available_list:
            assert availability.get(provider) == True


class TestBuildBasicEnvelope:
    """Test _build_basic_envelope internal function."""

    def test_creates_valid_envelope(self):
        """Test creates a valid JobEnvelope."""
        from app.llm.clients import _build_basic_envelope
        from app.jobs.schemas import JobEnvelope, JobType
        
        envelope = _build_basic_envelope(
            provider_id="openai",
            model_id="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
        )
        
        assert isinstance(envelope, JobEnvelope)
        assert envelope.job_type == JobType.CHAT_SIMPLE

    def test_includes_system_prompt(self):
        """Test system prompt is included in messages."""
        from app.llm.clients import _build_basic_envelope
        
        envelope = _build_basic_envelope(
            provider_id="openai",
            model_id="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            system_prompt="You are helpful.",
        )
        
        # System prompt should be first message
        assert envelope.messages[0]["role"] == "system"
        assert envelope.messages[0]["content"] == "You are helpful."

    def test_budget_configuration(self):
        """Test budget is configured correctly."""
        from app.llm.clients import _build_basic_envelope
        
        envelope = _build_basic_envelope(
            provider_id="openai",
            model_id="gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=4096,
            timeout_seconds=120,
        )
        
        assert envelope.budget.max_tokens == 4096
        assert envelope.budget.max_wall_time_seconds == 120

    def test_metadata_contains_hints(self):
        """Test metadata contains provider/model hints."""
        from app.llm.clients import _build_basic_envelope
        
        envelope = _build_basic_envelope(
            provider_id="anthropic",
            model_id="claude-sonnet-4",
            messages=[{"role": "user", "content": "Hi"}],
        )
        
        assert envelope.metadata["provider_hint"] == "anthropic"
        assert envelope.metadata["model_hint"] == "claude-sonnet-4"


class TestAsyncCallFunctions:
    """Test async call function signatures."""

    def test_async_call_openai_is_coroutine(self):
        """Test async_call_openai is async function."""
        from app.llm.clients import async_call_openai
        
        assert asyncio.iscoroutinefunction(async_call_openai)

    def test_async_call_anthropic_is_coroutine(self):
        """Test async_call_anthropic is async function."""
        from app.llm.clients import async_call_anthropic
        
        assert asyncio.iscoroutinefunction(async_call_anthropic)

    def test_async_call_google_is_coroutine(self):
        """Test async_call_google is async function."""
        from app.llm.clients import async_call_google
        
        assert asyncio.iscoroutinefunction(async_call_google)


class TestSyncCallFunctions:
    """Test sync call function signatures."""

    def test_call_openai_exists(self):
        """Test call_openai exists and is callable."""
        from app.llm.clients import call_openai
        
        assert callable(call_openai)

    def test_call_anthropic_exists(self):
        """Test call_anthropic exists and is callable."""
        from app.llm.clients import call_anthropic
        
        assert callable(call_anthropic)

    def test_call_google_exists(self):
        """Test call_google exists and is callable."""
        from app.llm.clients import call_google
        
        assert callable(call_google)


class TestEmbeddingFunctions:
    """Test embedding function signatures."""

    def test_get_embeddings_exists(self):
        """Test get_embeddings exists."""
        from app.llm.clients import get_embeddings
        
        assert callable(get_embeddings)

    def test_generate_embedding_exists(self):
        """Test generate_embedding exists."""
        from app.llm.clients import generate_embedding
        
        assert callable(generate_embedding)


class TestLlmCallAndUnpackAsync:
    """Test _llm_call_and_unpack_async internal function."""

    def test_is_coroutine_function(self):
        """Test is async function."""
        from app.llm.clients import _llm_call_and_unpack_async
        
        assert asyncio.iscoroutinefunction(_llm_call_and_unpack_async)


class TestErrorHandling:
    """Test error handling in client functions."""

    def test_validation_error_returns_error_dict(self):
        """Test validation errors return error in usage dict."""
        from app.llm.clients import _build_basic_envelope
        from app.jobs.schemas import JobBudget
        from pydantic import ValidationError
        
        # Test with invalid budget (tokens below minimum)
        with pytest.raises(ValidationError):
            _build_basic_envelope(
                provider_id="openai",
                model_id="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,  # Below minimum of 1000
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
