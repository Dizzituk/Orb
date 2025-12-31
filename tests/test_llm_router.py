# FILE: tests/test_llm_router.py
"""
Tests for app/llm/router.py
LLM routing API - thin wrapper that re-exports from routing/core.py
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestLLMRouterImports:
    """Test router module structure."""
    
    def test_imports_without_error(self):
        """Test module imports cleanly."""
        from app.llm import router
        assert router is not None
    
    def test_primary_api_exports(self):
        """Test primary API functions are exported."""
        from app.llm.router import (
            call_llm,
            quick_chat,
            request_code,
            review_work,
        )
        assert callable(call_llm)
        assert callable(quick_chat)
        assert callable(request_code)
        assert callable(review_work)
    
    def test_async_equivalents_exported(self):
        """Test async function variants are exported."""
        from app.llm.router import (
            call_llm_async,
            quick_chat_async,
            request_code_async,
            review_work_async,
        )
        assert callable(call_llm_async)
        assert callable(quick_chat_async)
        assert callable(request_code_async)
        assert callable(review_work_async)
    
    def test_sync_and_async_are_same(self):
        """Test sync names are aliases to async functions."""
        from app.llm.router import (
            call_llm, call_llm_async,
            quick_chat, quick_chat_async,
        )
        assert call_llm is call_llm_async
        assert quick_chat is quick_chat_async


class TestToolingExports:
    """Test tooling and utility exports."""
    
    def test_vision_export(self):
        """Test analyze_with_vision is exported."""
        from app.llm.router import analyze_with_vision
        assert callable(analyze_with_vision)
    
    def test_web_search_export(self):
        """Test web_search_query is exported."""
        from app.llm.router import web_search_query
        assert callable(web_search_query)


class TestRoutingMetaExports:
    """Test routing metadata exports."""
    
    def test_list_job_types_export(self):
        """Test list_job_types is exported."""
        from app.llm.router import list_job_types
        assert callable(list_job_types)
    
    def test_get_routing_info_export(self):
        """Test get_routing_info is exported."""
        from app.llm.router import get_routing_info
        assert callable(get_routing_info)
    
    def test_policy_routing_exports(self):
        """Test policy routing functions are exported."""
        from app.llm.router import is_policy_routing_enabled, enable_policy_routing
        assert callable(is_policy_routing_enabled)
        assert callable(enable_policy_routing)


class TestPipelineHelperExports:
    """Test pipeline helper exports."""
    
    def test_high_stakes_helper(self):
        """Test run_high_stakes_with_critique is exported."""
        from app.llm.router import run_high_stakes_with_critique
        assert callable(run_high_stakes_with_critique)
    
    def test_envelope_synthesizer(self):
        """Test synthesize_envelope_from_task is exported."""
        from app.llm.router import synthesize_envelope_from_task
        assert callable(synthesize_envelope_from_task)
    
    def test_job_type_helpers(self):
        """Test is_high_stakes_job and is_opus_model are exported."""
        from app.llm.router import is_high_stakes_job, is_opus_model
        assert callable(is_high_stakes_job)
        assert callable(is_opus_model)


class TestAllExports:
    """Test __all__ is complete."""
    
    def test_all_defined(self):
        """Test __all__ is defined."""
        from app.llm import router
        assert hasattr(router, '__all__')
        assert len(router.__all__) > 0
    
    def test_all_exports_exist(self):
        """Test all items in __all__ actually exist."""
        from app.llm import router
        for name in router.__all__:
            assert hasattr(router, name), f"Missing export: {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
