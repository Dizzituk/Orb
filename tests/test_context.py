# FILE: tests/test_context.py
"""
Tests for app/llm/context.py
Context utilities for LLM calls - datetime and system context.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from datetime import datetime
import re


class TestContextImports:
    """Test context module structure."""
    
    def test_imports_without_error(self):
        """Test module imports cleanly."""
        from app.llm import context
        assert context is not None
    
    def test_core_functions_exist(self):
        """Test core functions are exported."""
        from app.llm.context import (
            get_current_datetime_context,
            get_system_context,
            enhance_system_prompt,
        )
        assert callable(get_current_datetime_context)
        assert callable(get_system_context)
        assert callable(enhance_system_prompt)


class TestGetCurrentDatetimeContext:
    """Test get_current_datetime_context function."""
    
    def test_returns_string(self):
        """Test function returns a string."""
        from app.llm.context import get_current_datetime_context
        
        result = get_current_datetime_context()
        assert isinstance(result, str)
    
    def test_contains_day_of_week(self):
        """Test result contains day of week."""
        from app.llm.context import get_current_datetime_context
        
        result = get_current_datetime_context()
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        assert any(day in result for day in days)
    
    def test_contains_month(self):
        """Test result contains month name."""
        from app.llm.context import get_current_datetime_context
        
        result = get_current_datetime_context()
        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
        assert any(month in result for month in months)
    
    def test_contains_year(self):
        """Test result contains current year."""
        from app.llm.context import get_current_datetime_context
        
        result = get_current_datetime_context()
        current_year = str(datetime.now().year)
        assert current_year in result
    
    def test_contains_time(self):
        """Test result contains time with AM/PM."""
        from app.llm.context import get_current_datetime_context
        
        result = get_current_datetime_context()
        assert "AM" in result or "PM" in result
    
    def test_format_structure(self):
        """Test result matches expected format structure."""
        from app.llm.context import get_current_datetime_context
        
        result = get_current_datetime_context()
        # Should contain " at " separating date and time
        assert " at " in result


class TestGetSystemContext:
    """Test get_system_context function."""
    
    def test_returns_string(self):
        """Test function returns a string."""
        from app.llm.context import get_system_context
        
        result = get_system_context()
        assert isinstance(result, str)
    
    def test_contains_datetime_prefix(self):
        """Test result starts with expected prefix."""
        from app.llm.context import get_system_context
        
        result = get_system_context()
        assert result.startswith("Current date and time:")
    
    def test_contains_datetime_context(self):
        """Test result contains actual datetime info."""
        from app.llm.context import get_system_context, get_current_datetime_context
        
        result = get_system_context()
        datetime_part = get_current_datetime_context()
        
        # The datetime context should be included
        assert datetime_part in result


class TestEnhanceSystemPrompt:
    """Test enhance_system_prompt function."""
    
    def test_adds_context_to_prompt(self):
        """Test context is prepended to prompt."""
        from app.llm.context import enhance_system_prompt, get_system_context
        
        base_prompt = "You are a helpful assistant."
        result = enhance_system_prompt(base_prompt)
        
        # Should contain both context and original prompt
        assert "Current date and time:" in result
        assert base_prompt in result
    
    def test_context_before_prompt(self):
        """Test context appears before the base prompt."""
        from app.llm.context import enhance_system_prompt
        
        base_prompt = "You are a helpful assistant."
        result = enhance_system_prompt(base_prompt)
        
        # Context should be at the start
        assert result.startswith("Current date and time:")
        
        # Base prompt should come after
        context_end = result.index("Current date and time:") + len("Current date and time:")
        prompt_start = result.index(base_prompt)
        assert prompt_start > context_end
    
    def test_empty_prompt(self):
        """Test behavior with empty base prompt."""
        from app.llm.context import enhance_system_prompt, get_system_context
        
        result = enhance_system_prompt("")
        expected = get_system_context()
        
        assert result == expected
    
    def test_none_like_empty(self):
        """Test behavior with falsy base prompt."""
        from app.llm.context import enhance_system_prompt, get_system_context
        
        # Empty string is falsy
        result = enhance_system_prompt("")
        assert result == get_system_context()
    
    def test_separator_between_context_and_prompt(self):
        """Test proper separation between context and prompt."""
        from app.llm.context import enhance_system_prompt
        
        base_prompt = "You are a helpful assistant."
        result = enhance_system_prompt(base_prompt)
        
        # Should have newlines separating context from prompt
        assert "\n\n" in result
    
    def test_preserves_multiline_prompt(self):
        """Test multiline prompts are preserved."""
        from app.llm.context import enhance_system_prompt
        
        base_prompt = "Line 1\nLine 2\nLine 3"
        result = enhance_system_prompt(base_prompt)
        
        assert "Line 1\nLine 2\nLine 3" in result


class TestContextIntegration:
    """Integration tests for context utilities."""
    
    def test_datetime_is_current(self):
        """Test datetime reflects actual current time."""
        from app.llm.context import get_current_datetime_context
        
        now = datetime.now()
        result = get_current_datetime_context()
        
        # Current day should be in result
        current_day = now.strftime("%A")
        assert current_day in result
        
        # Current month should be in result
        current_month = now.strftime("%B")
        assert current_month in result
    
    def test_full_workflow(self):
        """Test typical usage workflow."""
        from app.llm.context import enhance_system_prompt
        
        # Typical system prompt
        system_prompt = """You are Orb, an AI assistant.
You help users with coding and technical tasks.
Be helpful, accurate, and concise."""
        
        enhanced = enhance_system_prompt(system_prompt)
        
        # Should have datetime context
        assert "Current date and time:" in enhanced
        
        # Should preserve original prompt
        assert "You are Orb" in enhanced
        assert "coding and technical tasks" in enhanced


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
