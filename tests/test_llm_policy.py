# FILE: tests/test_llm_policy.py
"""
Tests for app/llm/policy.py
Model selection policy - rules for choosing models.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from unittest.mock import Mock, patch


class TestPolicyImports:
    """Test policy module structure."""
    
    def test_imports_without_error(self):
        """Test module imports cleanly."""
        from app.llm import policy
        assert policy is not None
    
    def test_core_exports(self):
        """Test core components are exported."""
        from app.llm.policy import (
            Provider,
            AttachmentMode,
            DataType,
            JobType,
            JobPolicy,
            RoutingPolicy,
            RoutingDecision,
            PolicyError,
            load_routing_policy,
            get_policy_for_job,
            get_provider_for_job,
            validate_task_data,
            detect_data_types,
            make_routing_decision,
        )
        assert Provider is not None
        assert callable(load_routing_policy)


class TestProviderEnum:
    """Test Provider enumeration."""
    
    def test_providers_exist(self):
        """Test all providers are defined."""
        from app.llm.policy import Provider
        
        assert Provider.OPENAI
        assert Provider.ANTHROPIC
        assert Provider.GEMINI
    
    def test_provider_values(self):
        """Test provider string values."""
        from app.llm.policy import Provider
        
        assert Provider.OPENAI.value == "openai"
        assert Provider.ANTHROPIC.value == "anthropic"
        assert Provider.GEMINI.value == "gemini"


class TestAttachmentModeEnum:
    """Test AttachmentMode enumeration."""
    
    def test_modes_exist(self):
        """Test all attachment modes exist."""
        from app.llm.policy import AttachmentMode
        
        assert AttachmentMode.NONE
        assert AttachmentMode.TEXT_ONLY
        assert AttachmentMode.ALL


class TestJobTypeEnum:
    """Test JobType enumeration."""
    
    def test_gpt_primary_jobs(self):
        """Test GPT-primary job types exist."""
        from app.llm.policy import JobType
        
        assert JobType.CASUAL_CHAT
        assert JobType.QUICK_QUESTION
        assert JobType.SUMMARIZATION
        assert JobType.RESEARCH
    
    def test_claude_primary_jobs(self):
        """Test Claude-primary job types exist."""
        from app.llm.policy import JobType
        
        assert JobType.ARCHITECTURE
        assert JobType.DEEP_PLANNING
        assert JobType.SECURITY_REVIEW
        assert JobType.COMPLEX_CODE
    
    def test_gemini_primary_jobs(self):
        """Test Gemini-primary job types exist."""
        from app.llm.policy import JobType
        
        assert JobType.VISION
        assert JobType.VIDEO_ANALYSIS
        assert JobType.WEB_SEARCH


class TestPolicyExceptions:
    """Test policy exception classes."""
    
    def test_policy_error_base(self):
        """Test PolicyError is base exception."""
        from app.llm.policy import PolicyError
        
        with pytest.raises(PolicyError):
            raise PolicyError("Test error")
    
    def test_unknown_job_type_error(self):
        """Test UnknownJobTypeError."""
        from app.llm.policy import UnknownJobTypeError
        
        error = UnknownJobTypeError("bad_job", ["job1", "job2"])
        assert "bad_job" in str(error)
        assert error.job_type == "bad_job"
    
    def test_unknown_provider_error(self):
        """Test UnknownProviderError."""
        from app.llm.policy import UnknownProviderError
        
        error = UnknownProviderError("bad_provider")
        assert "bad_provider" in str(error)
    
    def test_data_validation_error(self):
        """Test DataValidationError."""
        from app.llm.policy import DataValidationError
        
        error = DataValidationError(
            job_type="test_job",
            forbidden=["videos"],
            found=["videos"]
        )
        assert "test_job" in str(error)


class TestLoadRoutingPolicy:
    """Test load_routing_policy function."""
    
    def test_loads_default_policy(self):
        """Test loading default policy file."""
        from app.llm.policy import load_routing_policy, RoutingPolicy
        
        policy = load_routing_policy()
        
        assert isinstance(policy, RoutingPolicy)
    
    def test_policy_has_providers(self):
        """Test loaded policy has provider configs."""
        from app.llm.policy import load_routing_policy
        
        policy = load_routing_policy()
        
        assert "openai" in policy.providers
        assert "anthropic" in policy.providers
        assert "gemini" in policy.providers
    
    def test_policy_has_entries(self):
        """Test loaded policy has job entries."""
        from app.llm.policy import load_routing_policy
        
        policy = load_routing_policy()
        
        assert len(policy.entries) > 0
    
    def test_policy_caching(self):
        """Test policy is cached on repeat loads."""
        from app.llm.policy import load_routing_policy
        
        policy1 = load_routing_policy()
        policy2 = load_routing_policy()
        
        # Should be same instance (cached)
        assert policy1 is policy2
    
    def test_force_reload(self):
        """Test force reload bypasses cache."""
        from app.llm.policy import load_routing_policy
        
        policy1 = load_routing_policy()
        policy2 = load_routing_policy(force_reload=True)
        
        # Should be new instance
        assert policy2 is not None
    
    def test_invalid_path_raises(self):
        """Test invalid path raises PolicyError."""
        from app.llm.policy import load_routing_policy, PolicyError
        
        with pytest.raises(PolicyError):
            load_routing_policy(path="/nonexistent/path.json")


class TestGetPolicyForJob:
    """Test get_policy_for_job function."""
    
    def test_gets_known_job(self):
        """Test getting policy for known job type."""
        from app.llm.policy import get_policy_for_job, JobPolicy
        
        policy = get_policy_for_job("architecture")
        
        assert isinstance(policy, JobPolicy)
        assert policy.job_type == "architecture"
    
    def test_casual_chat_job(self):
        """Test casual_chat job policy."""
        from app.llm.policy import get_policy_for_job, Provider
        
        policy = get_policy_for_job("casual_chat")
        
        assert policy.primary_provider == Provider.OPENAI
    
    def test_architecture_job(self):
        """Test architecture job policy."""
        from app.llm.policy import get_policy_for_job, Provider
        
        policy = get_policy_for_job("architecture")
        
        assert policy.primary_provider == Provider.ANTHROPIC
    
    def test_vision_job(self):
        """Test vision job policy."""
        from app.llm.policy import get_policy_for_job, Provider
        
        policy = get_policy_for_job("vision")
        
        assert policy.primary_provider == Provider.GEMINI
    
    def test_unknown_job_raises(self):
        """Test unknown job type raises error."""
        from app.llm.policy import get_policy_for_job, UnknownJobTypeError
        
        with pytest.raises(UnknownJobTypeError):
            get_policy_for_job("totally_unknown_job_xyz")


class TestGetProviderForJob:
    """Test get_provider_for_job function."""
    
    def test_returns_provider_tuple(self):
        """Test returns primary and review providers."""
        from app.llm.policy import get_provider_for_job, Provider
        
        primary, review = get_provider_for_job("architecture")
        
        assert isinstance(primary, Provider)
        # Review may be None or Provider
    
    def test_gpt_primary_for_chat(self):
        """Test GPT is primary for casual chat."""
        from app.llm.policy import get_provider_for_job, Provider
        
        primary, _ = get_provider_for_job("casual_chat")
        
        assert primary == Provider.OPENAI


class TestValidateTaskData:
    """Test validate_task_data function."""
    
    def test_valid_data_passes(self):
        """Test valid data passes validation."""
        from app.llm.policy import validate_task_data, get_policy_for_job
        
        policy = get_policy_for_job("casual_chat")
        
        is_valid, violations = validate_task_data(
            policy, 
            ["text"], 
            raise_on_error=False
        )
        
        assert is_valid == True
        assert len(violations) == 0
    
    def test_returns_violations(self):
        """Test returns list of violations."""
        from app.llm.policy import validate_task_data, get_policy_for_job
        
        policy = get_policy_for_job("casual_chat")
        
        # Check what's forbidden and test with that
        if policy.forbidden_data:
            forbidden_type = policy.forbidden_data[0]
            is_valid, violations = validate_task_data(
                policy,
                [forbidden_type],
                raise_on_error=False
            )
            
            assert is_valid == False
            assert forbidden_type in violations


class TestDetectDataTypes:
    """Test detect_data_types function."""
    
    def test_always_includes_text(self):
        """Test text is always detected."""
        from app.llm.policy import detect_data_types
        
        result = detect_data_types("Hello world")
        
        assert "text" in result
    
    def test_detects_code(self):
        """Test code detection in content."""
        from app.llm.policy import detect_data_types
        
        code_content = """
def hello():
    print("Hello World")
    
import os
class MyClass:
    pass
"""
        result = detect_data_types(code_content)
        
        assert "code" in result
    
    def test_detects_logs(self):
        """Test log detection in content."""
        from app.llm.policy import detect_data_types
        
        log_content = """
Traceback (most recent call last):
  File "test.py", line 10
    raise ValueError
ValueError: invalid value
"""
        result = detect_data_types(log_content)
        
        assert "logs" in result
    
    def test_detects_json(self):
        """Test JSON detection."""
        from app.llm.policy import detect_data_types
        
        json_content = '{"key": "value", "items": [1, 2, 3]}'
        
        result = detect_data_types(json_content)
        
        assert "structured_json" in result
    
    def test_detects_images_from_attachments(self):
        """Test image detection from attachments."""
        from app.llm.policy import detect_data_types
        
        attachments = [
            {"mime_type": "image/png", "filename": "screenshot.png", "size": 1024}
        ]
        
        result = detect_data_types("Some text", attachments)
        
        assert "images" in result
    
    def test_detects_pdfs_from_attachments(self):
        """Test PDF detection from attachments."""
        from app.llm.policy import detect_data_types
        
        attachments = [
            {"mime_type": "application/pdf", "filename": "document.pdf", "size": 5000}
        ]
        
        result = detect_data_types("Some text", attachments)
        
        assert "pdfs" in result
    
    def test_detects_video_from_attachments(self):
        """Test video detection from attachments."""
        from app.llm.policy import detect_data_types
        
        attachments = [
            {"mime_type": "video/mp4", "filename": "recording.mp4", "size": 100000}
        ]
        
        result = detect_data_types("Some text", attachments)
        
        assert "videos" in result


class TestMakeRoutingDecision:
    """Test make_routing_decision function."""
    
    def test_returns_routing_decision(self):
        """Test returns RoutingDecision object."""
        from app.llm.policy import make_routing_decision, RoutingDecision
        
        decision = make_routing_decision(
            job_type="casual_chat",
            content="Hello, how are you?"
        )
        
        assert isinstance(decision, RoutingDecision)
    
    def test_decision_has_required_fields(self):
        """Test decision has all required fields."""
        from app.llm.policy import make_routing_decision
        
        decision = make_routing_decision(
            job_type="casual_chat",
            content="Test content"
        )
        
        assert hasattr(decision, "job_type")
        assert hasattr(decision, "primary_provider")
        assert hasattr(decision, "primary_model")
        assert hasattr(decision, "temperature")
        assert hasattr(decision, "timeout_seconds")
    
    def test_decision_for_code_job(self):
        """Test routing decision for code job."""
        from app.llm.policy import make_routing_decision, Provider
        
        decision = make_routing_decision(
            job_type="complex_code",
            content="Write a Python function"
        )
        
        assert decision.primary_provider == Provider.ANTHROPIC
    
    def test_detects_data_types(self):
        """Test decision includes detected data types."""
        from app.llm.policy import make_routing_decision
        
        decision = make_routing_decision(
            job_type="casual_chat",
            content='{"test": "json"}'
        )
        
        assert "text" in decision.detected_data_types


class TestRoutingPolicyMethods:
    """Test RoutingPolicy class methods."""
    
    def test_list_job_types(self):
        """Test listing all job types."""
        from app.llm.policy import load_routing_policy
        
        policy = load_routing_policy()
        job_types = policy.list_job_types()
        
        assert isinstance(job_types, list)
        assert len(job_types) > 0
        assert "architecture" in job_types or "casual_chat" in job_types
    
    def test_get_provider_config(self):
        """Test getting provider configuration."""
        from app.llm.policy import load_routing_policy, Provider
        
        policy = load_routing_policy()
        config = policy.get_provider_config(Provider.OPENAI)
        
        assert config is not None
        assert hasattr(config, "model")
        assert hasattr(config, "max_context_tokens")
    
    def test_is_high_stakes(self):
        """Test high stakes job detection."""
        from app.llm.policy import load_routing_policy
        
        policy = load_routing_policy()
        
        # These are typically high stakes
        high_stakes_jobs = ["architecture", "security_review", "migration"]
        
        for job in high_stakes_jobs:
            if job in policy.list_job_types():
                result = policy.is_high_stakes(job)
                # Result depends on policy config, just verify it returns bool
                assert isinstance(result, bool)


class TestJobPolicy:
    """Test JobPolicy model."""
    
    def test_job_policy_fields(self):
        """Test JobPolicy has expected fields."""
        from app.llm.policy import get_policy_for_job
        
        policy = get_policy_for_job("casual_chat")
        
        assert hasattr(policy, "job_type")
        assert hasattr(policy, "primary_provider")
        assert hasattr(policy, "allowed_data")
        assert hasattr(policy, "forbidden_data")
        assert hasattr(policy, "temperature")
    
    def test_job_policy_has_provider(self):
        """Test JobPolicy always has a provider."""
        from app.llm.policy import get_policy_for_job, Provider
        
        policy = get_policy_for_job("research")
        
        assert isinstance(policy.primary_provider, Provider)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
