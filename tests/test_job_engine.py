# FILE: tests/test_job_engine.py
"""
Tests for app/jobs/engine.py
Phase 4 Job Engine - routing helpers and envelope conversion.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from datetime import datetime


class TestJobEngineImports:
    """Test engine module imports."""
    
    def test_imports_without_error(self):
        """Test module imports cleanly."""
        from app.jobs import engine
        assert engine is not None
    
    def test_exports(self):
        """Test __all__ exports."""
        from app.jobs.engine import execute_job, create_and_run_job
        
        assert callable(execute_job)
        assert callable(create_and_run_job)


class TestDetermineRouting:
    """Test _determine_routing function."""
    
    def test_architecture_routes_to_claude(self):
        """Test APP_ARCHITECTURE routes to Claude."""
        from app.jobs.engine import _determine_routing
        from app.jobs.schemas import JobEnvelope, JobType, JobBudget
        
        envelope = JobEnvelope(
            session_id="test",
            project_id=1,
            job_type=JobType.APP_ARCHITECTURE,
            messages=[{"role": "user", "content": "Design a system"}],
            budget=JobBudget(),
        )
        
        provider, model, temp = _determine_routing(envelope)
        
        assert provider == "anthropic"
        assert "claude" in model.lower()
    
    def test_code_repo_routes_to_claude(self):
        """Test CODE_REPO routes to Claude."""
        from app.jobs.engine import _determine_routing
        from app.jobs.schemas import JobEnvelope, JobType, JobBudget
        
        envelope = JobEnvelope(
            session_id="test",
            project_id=1,
            job_type=JobType.CODE_REPO,
            messages=[{"role": "user", "content": "Refactor this"}],
            budget=JobBudget(),
        )
        
        provider, model, temp = _determine_routing(envelope)
        
        assert provider == "anthropic"
    
    def test_code_small_routes_to_claude(self):
        """Test CODE_SMALL routes to Claude."""
        from app.jobs.engine import _determine_routing
        from app.jobs.schemas import JobEnvelope, JobType, JobBudget
        
        envelope = JobEnvelope(
            session_id="test",
            project_id=1,
            job_type=JobType.CODE_SMALL,
            messages=[{"role": "user", "content": "Write a function"}],
            budget=JobBudget(),
        )
        
        provider, model, temp = _determine_routing(envelope)
        
        assert provider == "anthropic"
    
    def test_chat_simple_routes_to_gpt(self):
        """Test CHAT_SIMPLE routes to GPT."""
        from app.jobs.engine import _determine_routing
        from app.jobs.schemas import JobEnvelope, JobType, JobBudget
        
        envelope = JobEnvelope(
            session_id="test",
            project_id=1,
            job_type=JobType.CHAT_SIMPLE,
            messages=[{"role": "user", "content": "Hello"}],
            budget=JobBudget(),
        )
        
        provider, model, temp = _determine_routing(envelope)
        
        assert provider == "openai"
        assert "gpt" in model.lower()
    
    def test_returns_temperature(self):
        """Test returns temperature value."""
        from app.jobs.engine import _determine_routing
        from app.jobs.schemas import JobEnvelope, JobType, JobBudget
        
        envelope = JobEnvelope(
            session_id="test",
            project_id=1,
            job_type=JobType.CHAT_SIMPLE,
            messages=[{"role": "user", "content": "Hi"}],
            budget=JobBudget(),
        )
        
        provider, model, temp = _determine_routing(envelope)
        
        assert isinstance(temp, float)
        assert 0.0 <= temp <= 1.0


class TestExtractUserIntent:
    """Test _extract_user_intent function."""
    
    def test_extracts_last_user_message(self):
        """Test extracts last user message content."""
        from app.jobs.engine import _extract_user_intent
        
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Last message"},
        ]
        
        intent = _extract_user_intent(messages)
        
        assert intent == "Last message"
    
    def test_empty_messages_returns_empty(self):
        """Test empty messages returns empty string or JSON."""
        from app.jobs.engine import _extract_user_intent
        
        intent = _extract_user_intent([])
        
        # Returns either empty string or "[]"
        assert intent in ["", "[]"]
    
    def test_none_messages_returns_empty(self):
        """Test None messages returns empty string or JSON."""
        from app.jobs.engine import _extract_user_intent
        
        intent = _extract_user_intent(None)
        
        assert intent in ["", "[]"]
    
    def test_no_user_messages_returns_json(self):
        """Test no user messages returns JSON dump."""
        from app.jobs.engine import _extract_user_intent
        
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "assistant", "content": "Response"},
        ]
        
        intent = _extract_user_intent(messages)
        
        # Should return JSON since no user message found
        assert "system" in intent.lower() or intent == ""


class TestBuildRoutingDecision:
    """Test _build_routing_decision function."""
    
    def test_creates_routing_decision(self):
        """Test creates valid RoutingDecision."""
        from app.jobs.engine import _build_routing_decision
        from app.jobs.schemas import JobEnvelope, JobType, JobBudget, RoutingDecision
        
        envelope = JobEnvelope(
            job_id="test-job",
            session_id="test",
            project_id=1,
            job_type=JobType.CHAT_SIMPLE,
            messages=[{"role": "user", "content": "Hi"}],
            budget=JobBudget(),
        )
        
        decision = _build_routing_decision(
            envelope=envelope,
            provider_id="openai",
            model_id="gpt-4o",
            temperature=0.7,
        )
        
        assert isinstance(decision, RoutingDecision)
        assert decision.architect.provider == "openai"
        assert decision.architect.model_id == "gpt-4o"
    
    def test_tier_mapping_claude(self):
        """Test Claude models get tier S."""
        from app.jobs.engine import _build_routing_decision
        from app.jobs.schemas import JobEnvelope, JobType, JobBudget
        
        envelope = JobEnvelope(
            job_id="test",
            session_id="test",
            project_id=1,
            job_type=JobType.CODE_SMALL,
            messages=[{"role": "user", "content": "Code"}],
            budget=JobBudget(),
        )
        
        decision = _build_routing_decision(
            envelope=envelope,
            provider_id="anthropic",
            model_id="claude-sonnet-4-20250514",
            temperature=0.7,
        )
        
        assert decision.architect.tier == "S"
    
    def test_tier_mapping_gpt(self):
        """Test GPT models get tier A."""
        from app.jobs.engine import _build_routing_decision
        from app.jobs.schemas import JobEnvelope, JobType, JobBudget
        
        envelope = JobEnvelope(
            job_id="test",
            session_id="test",
            project_id=1,
            job_type=JobType.CHAT_SIMPLE,
            messages=[{"role": "user", "content": "Chat"}],
            budget=JobBudget(),
        )
        
        decision = _build_routing_decision(
            envelope=envelope,
            provider_id="openai",
            model_id="gpt-4o",
            temperature=0.7,
        )
        
        assert decision.architect.tier == "A"
    
    def test_unknown_model_gets_tier_b(self):
        """Test unknown models get tier B."""
        from app.jobs.engine import _build_routing_decision
        from app.jobs.schemas import JobEnvelope, JobType, JobBudget
        
        envelope = JobEnvelope(
            job_id="test",
            session_id="test",
            project_id=1,
            job_type=JobType.CHAT_SIMPLE,
            messages=[{"role": "user", "content": "Chat"}],
            budget=JobBudget(),
        )
        
        decision = _build_routing_decision(
            envelope=envelope,
            provider_id="unknown",
            model_id="unknown-model",
            temperature=0.7,
        )
        
        assert decision.architect.tier == "B"


class TestBuildSystemPrompt:
    """Test _build_system_prompt function."""
    
    def test_openai_prompt(self):
        """Test OpenAI gets appropriate prompt."""
        from app.jobs.engine import _build_system_prompt
        from app.jobs.schemas import JobEnvelope, JobType, JobBudget
        
        envelope = JobEnvelope(
            session_id="test",
            project_id=1,
            job_type=JobType.CHAT_SIMPLE,
            messages=[{"role": "user", "content": "Hi"}],
            budget=JobBudget(),
        )
        
        prompt = _build_system_prompt(envelope, "openai")
        
        assert "Orb" in prompt
        assert "concise" in prompt.lower() or "direct" in prompt.lower()
    
    def test_anthropic_architecture_prompt(self):
        """Test Anthropic architecture gets architect prompt."""
        from app.jobs.engine import _build_system_prompt
        from app.jobs.schemas import JobEnvelope, JobType, JobBudget
        
        envelope = JobEnvelope(
            session_id="test",
            project_id=1,
            job_type=JobType.APP_ARCHITECTURE,
            messages=[{"role": "user", "content": "Design"}],
            budget=JobBudget(),
        )
        
        prompt = _build_system_prompt(envelope, "anthropic")
        
        assert "architect" in prompt.lower()
        assert "design" in prompt.lower() or "architecture" in prompt.lower()
    
    def test_anthropic_code_prompt(self):
        """Test Anthropic code job gets code-focused prompt."""
        from app.jobs.engine import _build_system_prompt
        from app.jobs.schemas import JobEnvelope, JobType, JobBudget
        
        envelope = JobEnvelope(
            session_id="test",
            project_id=1,
            job_type=JobType.CODE_SMALL,
            messages=[{"role": "user", "content": "Write code"}],
            budget=JobBudget(),
        )
        
        prompt = _build_system_prompt(envelope, "anthropic")
        
        assert "code" in prompt.lower()
    
    def test_appends_user_system_prompt(self):
        """Test appends user's system prompt."""
        from app.jobs.engine import _build_system_prompt
        from app.jobs.schemas import JobEnvelope, JobType, JobBudget
        
        envelope = JobEnvelope(
            session_id="test",
            project_id=1,
            job_type=JobType.CHAT_SIMPLE,
            messages=[{"role": "user", "content": "Hi"}],
            budget=JobBudget(),
            system_prompt="Custom instructions here",
        )
        
        prompt = _build_system_prompt(envelope, "openai")
        
        assert "Custom instructions here" in prompt
    
    def test_gemini_prompt(self):
        """Test Gemini/other gets analyst prompt."""
        from app.jobs.engine import _build_system_prompt
        from app.jobs.schemas import JobEnvelope, JobType, JobBudget
        
        envelope = JobEnvelope(
            session_id="test",
            project_id=1,
            job_type=JobType.VISION_SIMPLE,
            messages=[{"role": "user", "content": "Analyze"}],
            budget=JobBudget(),
        )
        
        prompt = _build_system_prompt(envelope, "gemini")
        
        assert "analyst" in prompt.lower() or "review" in prompt.lower()


class TestPlaceholderRoutingDecision:
    """Test _placeholder_routing_decision function."""
    
    def test_creates_placeholder(self):
        """Test creates placeholder routing decision."""
        from app.jobs.engine import _placeholder_routing_decision
        from app.jobs.schemas import JobEnvelope, JobType, JobBudget, RoutingDecision
        
        envelope = JobEnvelope(
            job_id="test",
            session_id="test",
            project_id=1,
            job_type=JobType.CHAT_SIMPLE,
            messages=[{"role": "user", "content": "Hi"}],
            budget=JobBudget(),
        )
        
        decision = _placeholder_routing_decision(envelope)
        
        assert isinstance(decision, RoutingDecision)
        assert decision.architect.provider == "unknown"
        assert decision.architect.model_id == "unknown"
    
    def test_uses_budget_values(self):
        """Test uses envelope budget values."""
        from app.jobs.engine import _placeholder_routing_decision
        from app.jobs.schemas import JobEnvelope, JobType, JobBudget
        
        envelope = JobEnvelope(
            job_id="test",
            session_id="test",
            project_id=1,
            job_type=JobType.CHAT_SIMPLE,
            messages=[{"role": "user", "content": "Hi"}],
            budget=JobBudget(max_tokens=4096, max_wall_time_seconds=120),
        )
        
        decision = _placeholder_routing_decision(envelope)
        
        assert decision.max_tokens == 4096
        assert decision.timeout_seconds == 120


class TestRequestToEnvelope:
    """Test _request_to_envelope function."""
    
    def test_converts_request(self):
        """Test converts CreateJobRequest to JobEnvelope."""
        from app.jobs.engine import _request_to_envelope
        from app.jobs.schemas import CreateJobRequest, JobEnvelope, JobType
        
        request = CreateJobRequest(
            project_id=1,
            job_type="chat_simple",
            messages=[{"role": "user", "content": "Hello"}],
        )
        
        envelope = _request_to_envelope(request, "job-123", "session-456")
        
        assert isinstance(envelope, JobEnvelope)
        assert envelope.job_id == "job-123"
        assert envelope.session_id == "session-456"
        assert envelope.job_type == JobType.CHAT_SIMPLE
    
    def test_sets_default_importance(self):
        """Test sets default importance when not provided."""
        from app.jobs.engine import _request_to_envelope
        from app.jobs.schemas import CreateJobRequest, Importance
        
        request = CreateJobRequest(
            project_id=1,
            job_type="chat_simple",
            messages=[{"role": "user", "content": "Hi"}],
        )
        
        envelope = _request_to_envelope(request, "job", "session")
        
        assert envelope.importance == Importance.MEDIUM
    
    def test_preserves_importance(self):
        """Test preserves provided importance."""
        from app.jobs.engine import _request_to_envelope
        from app.jobs.schemas import CreateJobRequest, Importance
        
        request = CreateJobRequest(
            project_id=1,
            job_type="chat_simple",
            messages=[{"role": "user", "content": "Hi"}],
            importance=Importance.HIGH,
        )
        
        envelope = _request_to_envelope(request, "job", "session")
        
        assert envelope.importance == Importance.HIGH
    
    def test_output_contract_mapping_chat(self):
        """Test CHAT_SIMPLE gets TEXT_RESPONSE contract."""
        from app.jobs.engine import _request_to_envelope
        from app.jobs.schemas import CreateJobRequest, OutputContract
        
        request = CreateJobRequest(
            project_id=1,
            job_type="chat_simple",
            messages=[{"role": "user", "content": "Hi"}],
        )
        
        envelope = _request_to_envelope(request, "job", "session")
        
        assert envelope.output_contract == OutputContract.TEXT_RESPONSE
    
    def test_output_contract_mapping_architecture(self):
        """Test APP_ARCHITECTURE gets ARCHITECTURE_DOC contract."""
        from app.jobs.engine import _request_to_envelope
        from app.jobs.schemas import CreateJobRequest, OutputContract
        
        request = CreateJobRequest(
            project_id=1,
            job_type="app_architecture",
            messages=[{"role": "user", "content": "Design"}],
        )
        
        envelope = _request_to_envelope(request, "job", "session")
        
        assert envelope.output_contract == OutputContract.ARCHITECTURE_DOC
    
    def test_output_contract_mapping_code(self):
        """Test CODE_SMALL gets CODE_PATCH_PROPOSAL contract."""
        from app.jobs.engine import _request_to_envelope
        from app.jobs.schemas import CreateJobRequest, OutputContract
        
        request = CreateJobRequest(
            project_id=1,
            job_type="code_small",
            messages=[{"role": "user", "content": "Code"}],
        )
        
        envelope = _request_to_envelope(request, "job", "session")
        
        assert envelope.output_contract == OutputContract.CODE_PATCH_PROPOSAL
    
    def test_invalid_job_type_raises(self):
        """Test invalid job_type raises ValueError."""
        from app.jobs.engine import _request_to_envelope
        from app.jobs.schemas import CreateJobRequest
        from pydantic import ValidationError
        
        # CreateJobRequest validates job_type, so this should fail at request creation
        with pytest.raises(ValidationError):
            CreateJobRequest(
                project_id=1,
                job_type="invalid_type",
                messages=[{"role": "user", "content": "Hi"}],
            )
    
    def test_adds_image_modality_with_attachments(self):
        """Test adds IMAGE modality when attachments present."""
        from app.jobs.engine import _request_to_envelope
        from app.jobs.schemas import CreateJobRequest, Modality
        
        request = CreateJobRequest(
            project_id=1,
            job_type="vision_simple",
            messages=[{"role": "user", "content": "Analyze"}],
            attachments=[{"type": "image", "data": "base64..."}],
        )
        
        envelope = _request_to_envelope(request, "job", "session")
        
        assert Modality.TEXT in envelope.modalities_in
        assert Modality.IMAGE in envelope.modalities_in


class TestAsyncFunctions:
    """Test async function signatures."""
    
    def test_execute_job_is_async(self):
        """Test execute_job is async function."""
        import asyncio
        from app.jobs.engine import execute_job
        
        assert asyncio.iscoroutinefunction(execute_job)
    
    def test_create_and_run_job_is_async(self):
        """Test create_and_run_job is async function."""
        import asyncio
        from app.jobs.engine import create_and_run_job
        
        assert asyncio.iscoroutinefunction(create_and_run_job)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
