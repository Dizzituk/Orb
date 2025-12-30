# FILE: tests/test_sandbox_tools.py
"""Tests for sandbox tools and intent detection."""

import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest


class TestIntentDetection:
    """Test intent detection for sandbox commands."""
    
    def test_detect_start_zombie(self):
        """Should detect start zombie intent."""
        from app.sandbox.tools import detect_sandbox_intent
        
        prompts = [
            "start your zombie",
            "Start your zombie please",
            "Can you start your clone?",
            "boot sandbox orb",
            "spin up the sandbox",
            "wake up your clone",
        ]
        
        for prompt in prompts:
            tool, params = detect_sandbox_intent(prompt)
            assert tool == "start_sandbox_clone", f"Failed for: {prompt}"
            assert params.get("full_mode", True) is True
    
    def test_detect_start_backend_only(self):
        """Should detect backend-only start."""
        from app.sandbox.tools import detect_sandbox_intent
        
        prompt = "start your zombie backend only"
        tool, params = detect_sandbox_intent(prompt)
        
        assert tool == "start_sandbox_clone"
        assert params.get("full_mode") is False
    
    def test_detect_stop_zombie(self):
        """Should detect stop zombie intent."""
        from app.sandbox.tools import detect_sandbox_intent
        
        prompts = [
            "stop your zombie",
            "Stop the zombie",
            "shutdown sandbox",
            "kill the sandbox",
            "stop sandbox orb",
        ]
        
        for prompt in prompts:
            tool, params = detect_sandbox_intent(prompt)
            assert tool == "stop_sandbox_clone", f"Failed for: {prompt}"
    
    def test_detect_status_check(self):
        """Should detect status check intent."""
        from app.sandbox.tools import detect_sandbox_intent
        
        prompts = [
            "is your zombie running?",
            "check sandbox status",
            "is your clone up?",
            "zombie status",
        ]
        
        for prompt in prompts:
            tool, params = detect_sandbox_intent(prompt)
            assert tool == "check_sandbox_status", f"Failed for: {prompt}"
    
    def test_no_intent_for_normal_prompts(self):
        """Should return None for non-sandbox prompts."""
        from app.sandbox.tools import detect_sandbox_intent
        
        prompts = [
            "Hello, how are you?",
            "Write me a Python function",
            "What is the weather like?",
            "Help me debug this code",
        ]
        
        for prompt in prompts:
            tool, params = detect_sandbox_intent(prompt)
            assert tool is None, f"Wrongly detected for: {prompt}"


class TestSandboxTools:
    """Test sandbox tool definitions."""
    
    def test_tools_defined(self):
        """Should have all required tools defined."""
        from app.sandbox.tools import SANDBOX_TOOLS
        
        tool_names = [t["name"] for t in SANDBOX_TOOLS]
        
        assert "start_sandbox_clone" in tool_names
        assert "stop_sandbox_clone" in tool_names
        assert "check_sandbox_status" in tool_names
    
    def test_tool_has_description(self):
        """Each tool should have a description."""
        from app.sandbox.tools import SANDBOX_TOOLS
        
        for tool in SANDBOX_TOOLS:
            assert "description" in tool
            assert len(tool["description"]) > 20


class TestSandboxManager:
    """Test sandbox manager (without actual sandbox connection)."""
    
    def test_manager_singleton(self):
        """Should return same instance."""
        from app.sandbox.manager import get_sandbox_manager
        
        m1 = get_sandbox_manager()
        m2 = get_sandbox_manager()
        
        assert m1 is m2
    
    def test_status_enum(self):
        """Status enum should have expected values."""
        from app.sandbox.manager import SandboxStatus
        
        assert SandboxStatus.DISCONNECTED.value == "disconnected"
        assert SandboxStatus.STOPPED.value == "stopped"
        assert SandboxStatus.RUNNING.value == "running"


class TestHandlePrompt:
    """Test the handle_sandbox_prompt convenience function."""
    
    def test_returns_none_for_non_sandbox(self):
        """Should return None for non-sandbox prompts."""
        from app.sandbox.tools import handle_sandbox_prompt
        
        result = handle_sandbox_prompt("Hello world")
        assert result is None
    
    def test_returns_string_for_status_check(self):
        """Should return status string for status check."""
        from app.sandbox.tools import handle_sandbox_prompt
        
        # This will try to connect to sandbox, but should return a message
        # even if sandbox is not available
        result = handle_sandbox_prompt("check zombie status")
        
        assert result is not None
        assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
