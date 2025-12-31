# FILE: tests/test_routing_core.py
"""
Tests for app/llm/routing/core.py
Core routing logic - model selection and dispatch.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from unittest.mock import Mock, patch, AsyncMock


class TestRoutingCoreImports:
    """Test core routing module."""
    
    def test_imports_without_error(self):
        """Test module imports cleanly."""
        from app.llm.routing import core
        assert core is not None
    
    def test_core_exports(self):
        """Test core functions are exported."""
        from app.llm.routing.core import (
            call_llm_async,
            call_llm,
            quick_chat_async,
            quick_chat,
            request_code_async,
            request_code,
            review_work_async,
            review_work,
            list_job_types,
            get_routing_info,
            is_policy_routing_enabled,
        )
        assert callable(call_llm_async)
        assert callable(list_job_types)
    
    def test_high_stakes_exports(self):
        """Test high-stakes pipeline exports."""
        from app.llm.routing.core import (
            is_high_stakes_job,
            is_opus_model,
            run_high_stakes_with_critique,
            HIGH_STAKES_JOB_TYPES,
        )
        assert callable(is_high_stakes_job)
        assert callable(is_opus_model)
    
    def test_frontier_exports(self):
        """Test frontier override exports."""
        from app.llm.routing.core import (
            GEMINI_FRONTIER_MODEL_ID,
            ANTHROPIC_FRONTIER_MODEL_ID,
            OPENAI_FRONTIER_MODEL_ID,
        )
        assert GEMINI_FRONTIER_MODEL_ID is not None
        assert ANTHROPIC_FRONTIER_MODEL_ID is not None
        assert OPENAI_FRONTIER_MODEL_ID is not None


class TestListJobTypes:
    """Test list_job_types function."""
    
    def test_returns_list(self):
        """Test returns a list."""
        from app.llm.routing.core import list_job_types
        
        result = list_job_types()
        assert isinstance(result, list)
    
    def test_list_not_empty(self):
        """Test list is not empty."""
        from app.llm.routing.core import list_job_types
        
        result = list_job_types()
        assert len(result) > 0
    
    def test_contains_expected_types(self):
        """Test list contains expected job types."""
        from app.llm.routing.core import list_job_types
        
        result = list_job_types()
        
        # Should contain some common job types
        # Check for at least one code-related and one chat-related
        result_lower = [r.lower() for r in result]
        
        has_code = any("code" in r for r in result_lower)
        has_chat = any("chat" in r for r in result_lower)
        
        assert has_code or has_chat, f"Expected code or chat types, got: {result}"
    
    def test_returns_strings(self):
        """Test all items are strings."""
        from app.llm.routing.core import list_job_types
        
        result = list_job_types()
        
        for item in result:
            assert isinstance(item, str)


class TestGetRoutingInfo:
    """Test get_routing_info function."""
    
    def test_returns_dict(self):
        """Test returns a dictionary."""
        from app.llm.routing.core import get_routing_info
        
        result = get_routing_info()
        assert isinstance(result, dict)
    
    def test_contains_version(self):
        """Test contains version info."""
        from app.llm.routing.core import get_routing_info
        
        result = get_routing_info()
        assert "version" in result
    
    def test_contains_debug_flag(self):
        """Test contains debug flag."""
        from app.llm.routing.core import get_routing_info
        
        result = get_routing_info()
        assert "debug" in result
    
    def test_contains_audit_info(self):
        """Test contains audit info."""
        from app.llm.routing.core import get_routing_info
        
        result = get_routing_info()
        assert "audit" in result
        assert "available" in result["audit"]
        assert "enabled" in result["audit"]
    
    def test_contains_frontier_info(self):
        """Test contains frontier model info."""
        from app.llm.routing.core import get_routing_info
        
        result = get_routing_info()
        assert "frontier" in result
        assert "frontier_models" in result["frontier"]
    
    def test_contains_high_stakes_info(self):
        """Test contains high-stakes pipeline info."""
        from app.llm.routing.core import get_routing_info
        
        result = get_routing_info()
        assert "high_stakes" in result
        assert "enabled" in result["high_stakes"]
        assert "high_stakes_types" in result["high_stakes"]
    
    def test_contains_spec_modules(self):
        """Test contains spec module availability."""
        from app.llm.routing.core import get_routing_info
        
        result = get_routing_info()
        assert "spec_modules" in result
        
        modules = result["spec_modules"]
        assert "file_classifier" in modules
        assert "token_budgeting" in modules
        assert "fallbacks" in modules


class TestIsPolicyRoutingEnabled:
    """Test is_policy_routing_enabled function."""
    
    def test_returns_bool(self):
        """Test returns a boolean."""
        from app.llm.routing.core import is_policy_routing_enabled
        
        result = is_policy_routing_enabled()
        assert isinstance(result, bool)
    
    def test_default_enabled(self):
        """Test policy routing is enabled by default."""
        from app.llm.routing.core import is_policy_routing_enabled
        
        result = is_policy_routing_enabled()
        assert result == True


class TestIsHighStakesJob:
    """Test is_high_stakes_job function."""
    
    def test_architecture_is_high_stakes(self):
        """Test architecture job is high stakes."""
        from app.llm.routing.core import is_high_stakes_job
        
        result = is_high_stakes_job("architecture")
        assert result == True
    
    def test_security_is_high_stakes(self):
        """Test security job is high stakes."""
        from app.llm.routing.core import is_high_stakes_job
        
        result = is_high_stakes_job("security")
        assert result == True
    
    def test_chat_not_high_stakes(self):
        """Test chat job is not high stakes."""
        from app.llm.routing.core import is_high_stakes_job
        
        result = is_high_stakes_job("chat_light")
        assert result == False
    
    def test_unknown_not_high_stakes(self):
        """Test unknown job is not high stakes."""
        from app.llm.routing.core import is_high_stakes_job
        
        result = is_high_stakes_job("random_unknown_job")
        assert result == False


class TestIsOpusModel:
    """Test is_opus_model function."""
    
    def test_opus_model_detected(self):
        """Test Opus model is correctly detected."""
        from app.llm.routing.core import is_opus_model
        
        result = is_opus_model("claude-opus-4-20250514")
        assert result == True
    
    def test_sonnet_not_opus(self):
        """Test Sonnet is not Opus."""
        from app.llm.routing.core import is_opus_model
        
        result = is_opus_model("claude-sonnet-4-20250514")
        assert result == False
    
    def test_gpt_not_opus(self):
        """Test GPT is not Opus."""
        from app.llm.routing.core import is_opus_model
        
        result = is_opus_model("gpt-4.1")
        assert result == False
    
    def test_gemini_not_opus(self):
        """Test Gemini is not Opus."""
        from app.llm.routing.core import is_opus_model
        
        result = is_opus_model("gemini-3.0-pro-preview")
        assert result == False


class TestHighStakesJobTypes:
    """Test HIGH_STAKES_JOB_TYPES constant."""
    
    def test_is_set_or_list(self):
        """Test HIGH_STAKES_JOB_TYPES is iterable."""
        from app.llm.routing.core import HIGH_STAKES_JOB_TYPES
        
        assert hasattr(HIGH_STAKES_JOB_TYPES, '__iter__')
    
    def test_contains_architecture(self):
        """Test contains architecture."""
        from app.llm.routing.core import HIGH_STAKES_JOB_TYPES
        
        types_lower = [t.lower() for t in HIGH_STAKES_JOB_TYPES]
        assert any("architecture" in t for t in types_lower)
    
    def test_contains_security(self):
        """Test contains security."""
        from app.llm.routing.core import HIGH_STAKES_JOB_TYPES
        
        types_lower = [t.lower() for t in HIGH_STAKES_JOB_TYPES]
        assert any("security" in t for t in types_lower)


class TestFrontierModelIds:
    """Test frontier model ID constants."""
    
    def test_gemini_frontier_format(self):
        """Test Gemini frontier model ID format."""
        from app.llm.routing.core import GEMINI_FRONTIER_MODEL_ID
        
        assert isinstance(GEMINI_FRONTIER_MODEL_ID, str)
        assert "gemini" in GEMINI_FRONTIER_MODEL_ID.lower()
    
    def test_anthropic_frontier_format(self):
        """Test Anthropic frontier model ID format."""
        from app.llm.routing.core import ANTHROPIC_FRONTIER_MODEL_ID
        
        assert isinstance(ANTHROPIC_FRONTIER_MODEL_ID, str)
        assert "claude" in ANTHROPIC_FRONTIER_MODEL_ID.lower()
    
    def test_openai_frontier_format(self):
        """Test OpenAI frontier model ID format."""
        from app.llm.routing.core import OPENAI_FRONTIER_MODEL_ID
        
        assert isinstance(OPENAI_FRONTIER_MODEL_ID, str)
        assert "gpt" in OPENAI_FRONTIER_MODEL_ID.lower() or "o1" in OPENAI_FRONTIER_MODEL_ID.lower() or "4" in OPENAI_FRONTIER_MODEL_ID


class TestCompatibilityHelpers:
    """Test compatibility helper functions."""
    
    def test_analyze_with_vision_raises(self):
        """Test analyze_with_vision raises NotImplementedError."""
        from app.llm.routing.core import analyze_with_vision
        
        with pytest.raises(NotImplementedError):
            analyze_with_vision()
    
    def test_web_search_query_raises(self):
        """Test web_search_query raises NotImplementedError."""
        from app.llm.routing.core import web_search_query
        
        with pytest.raises(NotImplementedError):
            web_search_query()
    
    def test_enable_policy_routing(self):
        """Test enable_policy_routing exists and is callable."""
        from app.llm.routing.core import enable_policy_routing
        
        # Should not raise
        result = enable_policy_routing()
        assert result is None


class TestSyncWrappers:
    """Test sync wrapper functions exist."""
    
    def test_call_llm_exists(self):
        """Test call_llm sync wrapper exists."""
        from app.llm.routing.core import call_llm
        
        assert callable(call_llm)
    
    def test_quick_chat_exists(self):
        """Test quick_chat sync wrapper exists."""
        from app.llm.routing.core import quick_chat
        
        assert callable(quick_chat)
    
    def test_request_code_exists(self):
        """Test request_code sync wrapper exists."""
        from app.llm.routing.core import request_code
        
        assert callable(request_code)
    
    def test_review_work_exists(self):
        """Test review_work sync wrapper exists."""
        from app.llm.routing.core import review_work
        
        assert callable(review_work)


class TestAsyncFunctions:
    """Test async function signatures."""
    
    def test_call_llm_async_is_coroutine(self):
        """Test call_llm_async is async function."""
        from app.llm.routing.core import call_llm_async
        import asyncio
        
        assert asyncio.iscoroutinefunction(call_llm_async)
    
    def test_quick_chat_async_is_coroutine(self):
        """Test quick_chat_async is async function."""
        from app.llm.routing.core import quick_chat_async
        import asyncio
        
        assert asyncio.iscoroutinefunction(quick_chat_async)
    
    def test_request_code_async_is_coroutine(self):
        """Test request_code_async is async function."""
        from app.llm.routing.core import request_code_async
        import asyncio
        
        assert asyncio.iscoroutinefunction(request_code_async)
    
    def test_review_work_async_is_coroutine(self):
        """Test review_work_async is async function."""
        from app.llm.routing.core import review_work_async
        import asyncio
        
        assert asyncio.iscoroutinefunction(review_work_async)


class TestEnvelopeSynthesis:
    """Test envelope synthesis export."""
    
    def test_synthesize_envelope_exported(self):
        """Test synthesize_envelope_from_task is exported."""
        from app.llm.routing.core import synthesize_envelope_from_task
        
        assert callable(synthesize_envelope_from_task)


class TestClassifyAndRoute:
    """Test classify_and_route export."""
    
    def test_classify_and_route_exported(self):
        """Test classify_and_route is exported."""
        from app.llm.routing.core import classify_and_route
        
        assert callable(classify_and_route)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
