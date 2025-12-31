# FILE: tests/test_job_schemas.py
"""
Tests for app/jobs/schemas.py
Phase 4 Job System schemas, enums, and validation.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from datetime import datetime


class TestJobSchemasImports:
    """Test schema module imports."""
    
    def test_imports_without_error(self):
        """Test module imports cleanly."""
        from app.jobs import schemas
        assert schemas is not None
    
    def test_all_exports(self):
        """Test __all__ exports are accessible."""
        from app.jobs.schemas import (
            JobType,
            JobState,
            ErrorType,
            Importance,
            DataSensitivity,
            Modality,
            OutputContract,
            JobBudget,
            JobEnvelope,
            ValidationError,
            validate_job_envelope,
            ModelSelection,
            RoutingDecision,
            ToolInvocation,
            CritiqueIssue,
            UsageMetrics,
            JobResult,
            CreateJobRequest,
        )
        assert JobType is not None
        assert validate_job_envelope is not None


class TestJobTypeEnum:
    """Test JobType enum."""
    
    def test_chat_types_exist(self):
        """Test chat job types exist."""
        from app.jobs.schemas import JobType
        
        assert JobType.CHAT_SIMPLE.value == "chat_simple"
        assert JobType.CHAT_RESEARCH.value == "chat_research"
    
    def test_code_types_exist(self):
        """Test code job types exist."""
        from app.jobs.schemas import JobType
        
        assert JobType.CODE_SMALL.value == "code_small"
        assert JobType.CODE_REPO.value == "code_repo"
    
    def test_architecture_type_exists(self):
        """Test architecture job type exists."""
        from app.jobs.schemas import JobType
        
        assert JobType.APP_ARCHITECTURE.value == "app_architecture"
    
    def test_vision_types_exist(self):
        """Test vision job types exist."""
        from app.jobs.schemas import JobType
        
        assert JobType.VISION_SIMPLE.value == "vision_simple"
        assert JobType.VISION_COMPLEX.value == "vision_complex"
    
    def test_video_types_exist(self):
        """Test video job types exist."""
        from app.jobs.schemas import JobType
        
        assert JobType.VIDEO_SIMPLE.value == "video_simple"
        assert JobType.VIDEO_ADVANCED.value == "video_advanced"
    
    def test_critique_type_exists(self):
        """Test critique job type exists."""
        from app.jobs.schemas import JobType
        
        assert JobType.CRITIQUE_REVIEW.value == "critique_review"


class TestJobStateEnum:
    """Test JobState enum."""
    
    def test_lifecycle_states(self):
        """Test all lifecycle states exist."""
        from app.jobs.schemas import JobState
        
        assert JobState.PENDING.value == "pending"
        assert JobState.RUNNING.value == "running"
        assert JobState.SUCCEEDED.value == "succeeded"
        assert JobState.FAILED.value == "failed"
        assert JobState.CANCELLED.value == "cancelled"
    
    def test_spec_clarification_state(self):
        """Test spec clarification state exists."""
        from app.jobs.schemas import JobState
        
        assert JobState.NEEDS_SPEC_CLARIFICATION.value == "needs_spec_clarification"


class TestErrorTypeEnum:
    """Test ErrorType enum."""
    
    def test_validation_error(self):
        """Test VALIDATION_ERROR exists."""
        from app.jobs.schemas import ErrorType
        
        assert ErrorType.VALIDATION_ERROR.value == "VALIDATION_ERROR"
    
    def test_routing_error(self):
        """Test ROUTING_ERROR exists."""
        from app.jobs.schemas import ErrorType
        
        assert ErrorType.ROUTING_ERROR.value == "ROUTING_ERROR"
    
    def test_model_error(self):
        """Test MODEL_ERROR exists."""
        from app.jobs.schemas import ErrorType
        
        assert ErrorType.MODEL_ERROR.value == "MODEL_ERROR"
    
    def test_tool_error(self):
        """Test TOOL_ERROR exists."""
        from app.jobs.schemas import ErrorType
        
        assert ErrorType.TOOL_ERROR.value == "TOOL_ERROR"
    
    def test_timeout(self):
        """Test TIMEOUT exists."""
        from app.jobs.schemas import ErrorType
        
        assert ErrorType.TIMEOUT.value == "TIMEOUT"
    
    def test_internal_error(self):
        """Test INTERNAL_ERROR exists."""
        from app.jobs.schemas import ErrorType
        
        assert ErrorType.INTERNAL_ERROR.value == "INTERNAL_ERROR"


class TestImportanceEnum:
    """Test Importance enum."""
    
    def test_all_levels(self):
        """Test all importance levels exist."""
        from app.jobs.schemas import Importance
        
        assert Importance.LOW.value == "low"
        assert Importance.MEDIUM.value == "medium"
        assert Importance.HIGH.value == "high"
        assert Importance.CRITICAL.value == "critical"


class TestDataSensitivityEnum:
    """Test DataSensitivity enum."""
    
    def test_all_levels(self):
        """Test all sensitivity levels exist."""
        from app.jobs.schemas import DataSensitivity
        
        assert DataSensitivity.PUBLIC.value == "public"
        assert DataSensitivity.INTERNAL.value == "internal"
        assert DataSensitivity.CONFIDENTIAL.value == "confidential"
        assert DataSensitivity.HIGHLY_CONFIDENTIAL.value == "highly_confidential"


class TestModalityEnum:
    """Test Modality enum."""
    
    def test_all_modalities(self):
        """Test all modalities exist."""
        from app.jobs.schemas import Modality
        
        assert Modality.TEXT.value == "text"
        assert Modality.IMAGE.value == "image"
        assert Modality.AUDIO.value == "audio"
        assert Modality.VIDEO.value == "video"
        assert Modality.CODE.value == "code"


class TestOutputContractEnum:
    """Test OutputContract enum."""
    
    def test_text_response(self):
        """Test TEXT_RESPONSE exists."""
        from app.jobs.schemas import OutputContract
        
        assert OutputContract.TEXT_RESPONSE.value == "text_response"
    
    def test_architecture_doc(self):
        """Test ARCHITECTURE_DOC exists."""
        from app.jobs.schemas import OutputContract
        
        assert OutputContract.ARCHITECTURE_DOC.value == "architecture_doc"
    
    def test_code_patch_proposal(self):
        """Test CODE_PATCH_PROPOSAL exists."""
        from app.jobs.schemas import OutputContract
        
        assert OutputContract.CODE_PATCH_PROPOSAL.value == "code_patch_proposal"
    
    def test_critique_review(self):
        """Test CRITIQUE_REVIEW exists."""
        from app.jobs.schemas import OutputContract
        
        assert OutputContract.CRITIQUE_REVIEW.value == "critique_review"


class TestJobBudget:
    """Test JobBudget model."""
    
    def test_default_values(self):
        """Test default budget values."""
        from app.jobs.schemas import JobBudget
        
        budget = JobBudget()
        
        assert budget.max_tokens == 8192
        assert budget.max_cost_estimate == 1.0
        assert budget.max_wall_time_seconds == 300
    
    def test_custom_values(self):
        """Test custom budget values."""
        from app.jobs.schemas import JobBudget
        
        budget = JobBudget(
            max_tokens=4096,
            max_cost_estimate=0.5,
            max_wall_time_seconds=120,
        )
        
        assert budget.max_tokens == 4096
        assert budget.max_cost_estimate == 0.5
        assert budget.max_wall_time_seconds == 120
    
    def test_validation_min_tokens(self):
        """Test minimum tokens validation."""
        from app.jobs.schemas import JobBudget
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            JobBudget(max_tokens=500)  # Below minimum of 1000


class TestJobEnvelope:
    """Test JobEnvelope model."""
    
    def test_create_minimal(self):
        """Test creating minimal envelope."""
        from app.jobs.schemas import JobEnvelope, JobType
        
        envelope = JobEnvelope(
            session_id="test-session",
            project_id=1,
            job_type=JobType.CHAT_SIMPLE,
            messages=[{"role": "user", "content": "Hello"}],
        )
        
        assert envelope.session_id == "test-session"
        assert envelope.project_id == 1
        assert envelope.job_type == JobType.CHAT_SIMPLE
    
    def test_default_importance(self):
        """Test default importance is MEDIUM."""
        from app.jobs.schemas import JobEnvelope, JobType, Importance
        
        envelope = JobEnvelope(
            session_id="test",
            project_id=1,
            job_type=JobType.CHAT_SIMPLE,
            messages=[{"role": "user", "content": "Hi"}],
        )
        
        assert envelope.importance == Importance.MEDIUM
    
    def test_default_sensitivity(self):
        """Test default sensitivity is INTERNAL."""
        from app.jobs.schemas import JobEnvelope, JobType, DataSensitivity
        
        envelope = JobEnvelope(
            session_id="test",
            project_id=1,
            job_type=JobType.CHAT_SIMPLE,
            messages=[{"role": "user", "content": "Hi"}],
        )
        
        assert envelope.data_sensitivity == DataSensitivity.INTERNAL
    
    def test_job_type_string_validation(self):
        """Test job_type accepts string and converts to enum."""
        from app.jobs.schemas import JobEnvelope, JobType
        
        envelope = JobEnvelope(
            session_id="test",
            project_id=1,
            job_type="chat_simple",  # String input
            messages=[{"role": "user", "content": "Hi"}],
        )
        
        assert envelope.job_type == JobType.CHAT_SIMPLE
    
    def test_invalid_job_type_raises(self):
        """Test invalid job_type raises error."""
        from app.jobs.schemas import JobEnvelope
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            JobEnvelope(
                session_id="test",
                project_id=1,
                job_type="invalid_type",
                messages=[{"role": "user", "content": "Hi"}],
            )


class TestValidateJobEnvelope:
    """Test validate_job_envelope function."""
    
    def test_valid_envelope_passes(self):
        """Test valid envelope passes validation."""
        from app.jobs.schemas import (
            JobEnvelope,
            JobType,
            validate_job_envelope,
        )
        
        envelope = JobEnvelope(
            session_id="test",
            project_id=1,
            job_type=JobType.CHAT_SIMPLE,
            messages=[{"role": "user", "content": "Hello"}],
        )
        
        # Should not raise
        validate_job_envelope(envelope)
    
    def test_empty_messages_raises(self):
        """Test empty messages raises ValidationError."""
        from app.jobs.schemas import (
            JobEnvelope,
            JobType,
            ValidationError,
            validate_job_envelope,
        )
        
        envelope = JobEnvelope(
            session_id="test",
            project_id=1,
            job_type=JobType.CHAT_SIMPLE,
            messages=[],
        )
        
        with pytest.raises(ValidationError) as exc_info:
            validate_job_envelope(envelope)
        
        assert "messages list cannot be empty" in exc_info.value.errors
    
    def test_tool_conflict_raises(self):
        """Test overlapping allowed/forbidden tools raises error."""
        from app.jobs.schemas import (
            JobEnvelope,
            JobType,
            ValidationError,
            validate_job_envelope,
        )
        
        envelope = JobEnvelope(
            session_id="test",
            project_id=1,
            job_type=JobType.CHAT_SIMPLE,
            messages=[{"role": "user", "content": "Hi"}],
            allowed_tools=["web_search"],
            forbidden_tools=["web_search"],
        )
        
        with pytest.raises(ValidationError) as exc_info:
            validate_job_envelope(envelope)
        
        assert any("both allowed and forbidden" in e for e in exc_info.value.errors)
    
    def test_excessive_review_rounds_raises(self):
        """Test max_review_rounds > 3 raises error at model creation."""
        from app.jobs.schemas import JobEnvelope, JobType
        from pydantic import ValidationError
        
        # Pydantic validates max_review_rounds <= 3 at model creation
        with pytest.raises(ValidationError) as exc_info:
            JobEnvelope(
                session_id="test",
                project_id=1,
                job_type=JobType.CHAT_SIMPLE,
                messages=[{"role": "user", "content": "Hi"}],
                allow_multi_model_review=True,
                max_review_rounds=5,
            )
        
        assert "max_review_rounds" in str(exc_info.value)


class TestValidationError:
    """Test ValidationError exception."""
    
    def test_stores_errors(self):
        """Test ValidationError stores error list."""
        from app.jobs.schemas import ValidationError
        
        errors = ["Error 1", "Error 2"]
        exc = ValidationError(errors)
        
        assert exc.errors == errors
    
    def test_message_contains_errors(self):
        """Test exception message contains errors."""
        from app.jobs.schemas import ValidationError
        
        errors = ["Error 1", "Error 2"]
        exc = ValidationError(errors)
        
        assert "Error 1" in str(exc)
        assert "Error 2" in str(exc)


class TestModelSelection:
    """Test ModelSelection model."""
    
    def test_create(self):
        """Test creating ModelSelection."""
        from app.jobs.schemas import ModelSelection
        
        selection = ModelSelection(
            provider="anthropic",
            model_id="claude-sonnet-4-20250514",
            tier="S",
            role="architect",
        )
        
        assert selection.provider == "anthropic"
        assert selection.tier == "S"
        assert selection.role == "architect"


class TestRoutingDecision:
    """Test RoutingDecision model."""
    
    def test_create(self):
        """Test creating RoutingDecision."""
        from app.jobs.schemas import RoutingDecision, ModelSelection
        
        decision = RoutingDecision(
            job_id="test-job",
            job_type="chat_simple",
            resolved_job_type="chat_simple",
            architect=ModelSelection(
                provider="openai",
                model_id="gpt-4o",
                tier="A",
                role="architect",
            ),
            data_sensitivity_constraint="internal",
            allowed_tools=[],
            forbidden_tools=[],
        )
        
        assert decision.job_id == "test-job"
        assert decision.architect.provider == "openai"
    
    def test_fallback_tracking(self):
        """Test fallback tracking fields."""
        from app.jobs.schemas import RoutingDecision, ModelSelection
        
        decision = RoutingDecision(
            job_id="test",
            job_type="chat_simple",
            resolved_job_type="chat_simple",
            architect=ModelSelection(
                provider="openai",
                model_id="gpt-4o",
                tier="A",
                role="architect",
            ),
            data_sensitivity_constraint="internal",
            allowed_tools=[],
            forbidden_tools=[],
            fallback_occurred=True,
            fallback_reason="Rate limited",
        )
        
        assert decision.fallback_occurred == True
        assert decision.fallback_reason == "Rate limited"


class TestUsageMetrics:
    """Test UsageMetrics model."""
    
    def test_create(self):
        """Test creating UsageMetrics."""
        from app.jobs.schemas import UsageMetrics
        
        metrics = UsageMetrics(
            model_id="gpt-4o",
            provider="openai",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_estimate=0.01,
        )
        
        assert metrics.total_tokens == 150
        assert metrics.cost_estimate == 0.01
    
    def test_default_zeros(self):
        """Test default token counts are zero."""
        from app.jobs.schemas import UsageMetrics
        
        metrics = UsageMetrics(
            model_id="test",
            provider="test",
        )
        
        assert metrics.prompt_tokens == 0
        assert metrics.completion_tokens == 0
        assert metrics.total_tokens == 0
        assert metrics.cost_estimate == 0.0


class TestCritiqueIssue:
    """Test CritiqueIssue model."""
    
    def test_create(self):
        """Test creating CritiqueIssue."""
        from app.jobs.schemas import CritiqueIssue
        
        issue = CritiqueIssue(
            severity="blocker",
            issue_type="security",
            description="SQL injection vulnerability",
            fix_hint="Use parameterized queries",
        )
        
        assert issue.severity == "blocker"
        assert issue.issue_type == "security"
        assert issue.resolved == False


class TestJobResult:
    """Test JobResult model."""
    
    def test_create_success(self):
        """Test creating successful JobResult."""
        from app.jobs.schemas import (
            JobResult,
            JobState,
            OutputContract,
            RoutingDecision,
            ModelSelection,
        )
        
        now = datetime.utcnow()
        
        result = JobResult(
            job_id="test-job",
            session_id="test-session",
            project_id=1,
            job_type="chat_simple",
            state=JobState.SUCCEEDED,
            content="Hello!",
            output_contract=OutputContract.TEXT_RESPONSE,
            routing_decision=RoutingDecision(
                job_id="test-job",
                job_type="chat_simple",
                resolved_job_type="chat_simple",
                architect=ModelSelection(
                    provider="openai",
                    model_id="gpt-4o",
                    tier="A",
                    role="architect",
                ),
                data_sensitivity_constraint="internal",
                allowed_tools=[],
                forbidden_tools=[],
            ),
            started_at=now,
            completed_at=now,
        )
        
        assert result.state == JobState.SUCCEEDED
        assert result.content == "Hello!"
    
    def test_create_failed(self):
        """Test creating failed JobResult."""
        from app.jobs.schemas import (
            JobResult,
            JobState,
            ErrorType,
            OutputContract,
            RoutingDecision,
            ModelSelection,
        )
        
        now = datetime.utcnow()
        
        result = JobResult(
            job_id="test-job",
            session_id="test-session",
            project_id=1,
            job_type="chat_simple",
            state=JobState.FAILED,
            content="",
            output_contract=OutputContract.TEXT_RESPONSE,
            routing_decision=RoutingDecision(
                job_id="test-job",
                job_type="chat_simple",
                resolved_job_type="chat_simple",
                architect=ModelSelection(
                    provider="openai",
                    model_id="gpt-4o",
                    tier="A",
                    role="architect",
                ),
                data_sensitivity_constraint="internal",
                allowed_tools=[],
                forbidden_tools=[],
            ),
            started_at=now,
            completed_at=now,
            error_type=ErrorType.MODEL_ERROR,
            error_message="Provider unavailable",
        )
        
        assert result.state == JobState.FAILED
        assert result.error_type == ErrorType.MODEL_ERROR


class TestCreateJobRequest:
    """Test CreateJobRequest model."""
    
    def test_create_valid(self):
        """Test creating valid request."""
        from app.jobs.schemas import CreateJobRequest
        
        request = CreateJobRequest(
            project_id=1,
            job_type="chat_simple",
            messages=[{"role": "user", "content": "Hello"}],
        )
        
        assert request.job_type == "chat_simple"
        assert request.project_id == 1
    
    def test_invalid_job_type_raises(self):
        """Test invalid job_type raises error."""
        from app.jobs.schemas import CreateJobRequest
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError):
            CreateJobRequest(
                project_id=1,
                job_type="invalid_type",
                messages=[{"role": "user", "content": "Hi"}],
            )
    
    def test_optional_fields(self):
        """Test optional fields have correct defaults."""
        from app.jobs.schemas import CreateJobRequest
        
        request = CreateJobRequest(
            project_id=1,
            job_type="chat_simple",
            messages=[{"role": "user", "content": "Hi"}],
        )
        
        assert request.session_id is None
        assert request.importance is None
        assert request.needs_internet == False
        assert request.allow_multi_model_review == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
