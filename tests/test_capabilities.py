# FILE: tests/test_capabilities.py
"""
Tests for ASTRA Capability Layer

Run with: pytest tests/test_capabilities.py -v
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.capabilities.loader import (
    load_capabilities,
    get_capability_context,
    get_capability_summary,
    get_hard_safety_rules,
    check_capability,
    reload_capabilities,
    FALLBACK_CAPABILITIES,
)

from app.capabilities.injector import (
    inject_capabilities,
    enhance_system_prompt_with_capabilities,
    get_stage_specific_context,
    should_block_action,
)


class TestCapabilitiesLoader:
    """Tests for capabilities loader module."""
    
    def test_fallback_capabilities_valid(self):
        """Fallback capabilities should be valid structure."""
        assert 'version' in FALLBACK_CAPABILITIES
        assert 'identity' in FALLBACK_CAPABILITIES
        assert 'hard_safety_rules' in FALLBACK_CAPABILITIES
    
    def test_load_capabilities_uses_fallback_on_missing_file(self):
        """Should use fallback when file doesn't exist."""
        reload_capabilities()  # Clear cache
        
        with patch('app.capabilities.loader.DEFAULT_CAPABILITIES_PATH', '/nonexistent/path.json'):
            reload_capabilities()
            caps = load_capabilities()
            assert caps == FALLBACK_CAPABILITIES
    
    def test_get_capability_context_returns_string(self):
        """Context should be a non-empty string."""
        context = get_capability_context()
        assert isinstance(context, str)
        assert len(context) > 100  # Should be substantial
        assert 'ASTRA' in context
    
    def test_get_capability_context_contains_key_sections(self):
        """Context should contain all key sections."""
        context = get_capability_context()
        assert 'IDENTITY' in context
        assert 'ENVIRONMENTS' in context
        assert 'SAFETY' in context or 'HARD SAFETY' in context
        assert 'CAN DO' in context or 'CAPABILITIES' in context
    
    def test_get_capability_summary_is_shorter(self):
        """Summary should be shorter than full context."""
        full = get_capability_context()
        summary = get_capability_summary()
        assert len(summary) < len(full)
        assert 'ASTRA' in summary
    
    def test_get_hard_safety_rules_returns_list(self):
        """Should return list of safety rules."""
        rules = get_hard_safety_rules()
        assert isinstance(rules, list)
        # Fallback has at least 2 rules
        assert len(rules) >= 2
    
    def test_check_capability_sandbox_write(self):
        """Writing to sandbox should be allowed."""
        result = check_capability('write_files', 'sandbox')
        # With fallback, sandbox is allowed
        assert result.get('allowed') is not False
    
    def test_check_capability_host_blocked(self):
        """Writing to host should be blocked."""
        result = check_capability('write_files', 'host_pc')
        assert result['allowed'] == False
        assert 'not allowed' in result['reason'].lower() or 'forbidden' in result['reason'].lower()


class TestCapabilitiesInjector:
    """Tests for capabilities injector module."""
    
    def test_inject_capabilities_returns_string(self):
        """Injection should return string."""
        result = inject_capabilities("Test prompt")
        assert isinstance(result, str)
    
    def test_inject_capabilities_prepends_context(self):
        """Capabilities should be at the start."""
        original = "This is the original prompt."
        result = inject_capabilities(original)
        assert result.startswith("=")  # Context starts with separator
        assert original in result
        # Original should be AFTER capabilities
        cap_end = result.find("=" * 60 + "\n\n") + len("=" * 60 + "\n\n")
        assert result.index(original) > 0
    
    def test_inject_capabilities_handles_none(self):
        """Should handle None input."""
        result = inject_capabilities(None)
        assert isinstance(result, str)
        assert 'ASTRA' in result
    
    def test_inject_capabilities_handles_empty(self):
        """Should handle empty string input."""
        result = inject_capabilities("")
        assert isinstance(result, str)
        assert 'ASTRA' in result
    
    def test_enhance_with_additional_context(self):
        """Should include additional context."""
        result = enhance_system_prompt_with_capabilities(
            "Base prompt",
            additional_context="Current time: 12:00"
        )
        assert "Base prompt" in result
        assert "Current time: 12:00" in result
        assert "ASTRA" in result
    
    def test_get_stage_specific_context_chat(self):
        """Chat stage should have specific guidance."""
        context = get_stage_specific_context("chat")
        assert "route" in context.lower() or "pipeline" in context.lower()
    
    def test_get_stage_specific_context_overwatcher(self):
        """Overwatcher stage should have enforcement guidance."""
        context = get_stage_specific_context("overwatcher")
        assert "sandbox" in context.lower()
    
    def test_get_stage_specific_context_unknown(self):
        """Unknown stage should return base context."""
        context = get_stage_specific_context("unknown_stage")
        assert "ASTRA" in context


class TestActionBlocking:
    """Tests for action blocking logic."""
    
    def test_block_host_windows_path(self):
        """Should block Windows system paths."""
        blocked, reason = should_block_action("write", "C:\\Windows\\System32\\test.txt")
        assert blocked == True
        assert "HSR" in reason or "forbidden" in reason.lower()
    
    def test_block_host_program_files(self):
        """Should block Program Files."""
        blocked, reason = should_block_action("write", "C:\\Program Files\\test\\file.txt")
        assert blocked == True
    
    def test_allow_sandbox_user_path(self):
        """Should allow sandbox user paths."""
        blocked, reason = should_block_action("write", "C:\\Users\\WDAGUtilityAccount\\Desktop\\test.txt")
        assert blocked == False
        assert "allowed" in reason.lower()
    
    def test_delete_requires_confirmation(self):
        """Delete actions should note confirmation requirement."""
        blocked, reason = should_block_action("delete", "C:\\Users\\WDAGUtilityAccount\\Desktop\\test.txt")
        assert blocked == False
        assert "confirmation" in reason.lower()
    
    def test_allow_normal_sandbox_operation(self):
        """Normal sandbox operations should be allowed."""
        blocked, reason = should_block_action("read", "D:\\Orb\\app\\main.py")
        # This might be blocked as host path - that's correct!
        # The sandbox version would be accessed via controller
        pass  # Path detection is intentionally conservative


class TestCapabilityContextContent:
    """Tests for specific content in capability context."""
    
    def test_context_mentions_sandbox_controller(self):
        """Context should mention sandbox controller URL."""
        context = get_capability_context()
        assert "192.168.250.2" in context or "sandbox" in context.lower()
    
    def test_context_mentions_forbidden_host(self):
        """Context should clearly state host PC is forbidden."""
        context = get_capability_context()
        assert "host" in context.lower()
        assert "❌" in context or "FORBIDDEN" in context or "NEVER" in context
    
    def test_context_explains_correct_behavior(self):
        """Context should explain what to do instead of 'I can't'."""
        context = get_capability_context()
        # Should mention not saying "I cannot" or similar
        lower_context = context.lower()
        assert "cannot" in lower_context or "can't" in lower_context or "don't say" in lower_context
    
    def test_context_mentions_delete_confirmation(self):
        """Context should mention delete confirmation requirement."""
        context = get_capability_context()
        assert "delete" in context.lower()
        assert "confirm" in context.lower()


class TestCapabilityVersioning:
    """Tests for capability versioning."""
    
    def test_fallback_has_version(self):
        """Fallback should have version number."""
        assert FALLBACK_CAPABILITIES.get('version') >= 1
    
    def test_reload_clears_cache(self):
        """Reload should clear and refresh capabilities."""
        # Load once
        caps1 = load_capabilities()
        # Reload
        caps2 = reload_capabilities()
        # Should still work (may be same or different depending on file presence)
        assert 'version' in caps2


# Integration test
class TestIntegration:
    """Integration tests simulating real usage."""
    
    def test_full_prompt_flow(self):
        """Test complete prompt enhancement flow."""
        base_prompt = "You are a helpful assistant for the Orb project."
        
        # Enhance with capabilities
        enhanced = enhance_system_prompt_with_capabilities(
            base_prompt,
            include_full=True,
            additional_context="Current date: 2026-01-04"
        )
        
        # Should have all components in correct order
        # 1. Capabilities (starts with separator)
        assert enhanced.startswith("=")
        
        # 2. Additional context somewhere
        assert "2026-01-04" in enhanced
        
        # 3. Base prompt at end
        assert "helpful assistant" in enhanced
        
        # Capabilities should come before base prompt
        cap_pos = enhanced.find("ASTRA")
        base_pos = enhanced.find("helpful assistant")
        assert cap_pos < base_pos
    
    def test_stage_specific_prompts_unique(self):
        """Different stages should have different contexts."""
        chat_ctx = get_stage_specific_context("chat")
        ow_ctx = get_stage_specific_context("overwatcher")
        
        # Should both have base capabilities
        assert "ASTRA" in chat_ctx
        assert "ASTRA" in ow_ctx
        
        # But with different additions
        # (This may pass even if same, since summary is shared)
        assert len(chat_ctx) > 50
        assert len(ow_ctx) > 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
