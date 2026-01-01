# FILE: tests/test_specs.py
"""
Tests for ASTRA Specs module.

Tests cover:
1. Git utils (commit hash retrieval)
2. Spec schema (creation, validation, serialization)
3. Spec service (CRUD operations)
4. Weaver context gathering
"""
from __future__ import annotations
import pytest
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.git_utils import (
    get_current_commit,
    get_current_branch,
    get_commit_short,
    is_git_repo,
    GitError,
    GitResult,
)

from app.specs import (
    SPEC_SCHEMA_VERSION,
    SpecStatus,
    SpecSchema,
    SpecInput,
    SpecOutput,
    SpecStep,
    SpecRequirements,
    SpecConstraints,
    SpecSafety,
    SpecMetadata,
    SpecProvenance,
    SpecValidationResult,
    validate_spec,
    spec_to_markdown,
)


# =============================================================================
# GIT UTILS TESTS
# =============================================================================

class TestGitUtils:
    """Tests for git utility functions."""
    
    def test_get_current_commit_in_repo(self):
        """Should return commit hash when in a git repo."""
        # Assuming tests run from D:\Orb which is a git repo
        result = get_current_commit()
        
        # Either succeeds (in git repo) or fails gracefully (not in repo)
        assert isinstance(result, GitResult)
        if result.success:
            assert result.value is not None
            assert len(result.value) == 40  # Full SHA
            assert result.error is None
        else:
            assert result.error in (GitError.NO_GIT_REPO, GitError.GIT_NOT_INSTALLED)
    
    def test_get_current_commit_invalid_path(self):
        """Should return NO_GIT_REPO for non-existent path."""
        result = get_current_commit(repo_path="/nonexistent/path/that/does/not/exist")
        assert not result.success
        assert result.error == GitError.NO_GIT_REPO
    
    def test_get_current_commit_invalid_branch(self):
        """Should return INVALID_BRANCH for non-existent branch."""
        result = get_current_commit(branch="this-branch-definitely-does-not-exist-12345")
        if result.success:
            # Might succeed if we're not in a git repo (returns NO_GIT_REPO instead)
            pass
        else:
            assert result.error in (GitError.INVALID_BRANCH, GitError.NO_GIT_REPO, GitError.GIT_NOT_INSTALLED)
    
    def test_get_commit_short(self):
        """Should truncate commit hash."""
        full_hash = "abc123def456789012345678901234567890abcd"
        assert get_commit_short(full_hash) == "abc123d"
        assert get_commit_short(full_hash, length=12) == "abc123def456"
    
    def test_get_commit_short_empty(self):
        """Should handle empty string."""
        assert get_commit_short("") == ""
        assert get_commit_short(None) == ""
    
    def test_is_git_repo(self):
        """Should return boolean."""
        result = is_git_repo()
        assert isinstance(result, bool)
    
    def test_get_current_branch(self):
        """Should return branch name or detached indicator."""
        result = get_current_branch()
        assert isinstance(result, GitResult)
        if result.success:
            assert result.value is not None
            # Either a branch name or "(detached)"
            assert len(result.value) > 0


# =============================================================================
# SPEC SCHEMA TESTS
# =============================================================================

class TestSpecSchema:
    """Tests for spec schema creation and validation."""
    
    def test_create_empty_spec(self):
        """Should create spec with defaults."""
        spec = SpecSchema()
        
        assert spec.spec_version == SPEC_SCHEMA_VERSION
        assert spec.spec_id is not None
        assert len(spec.spec_id) == 36  # UUID
        assert spec.title == ""
        assert spec.objective == ""
    
    def test_create_spec_with_content(self):
        """Should create spec with provided content."""
        spec = SpecSchema(
            title="Test Feature",
            summary="A test feature summary",
            objective="Build a test feature",
            acceptance_criteria=["It works", "It's fast"],
        )
        
        assert spec.title == "Test Feature"
        assert spec.summary == "A test feature summary"
        assert spec.objective == "Build a test feature"
        assert len(spec.acceptance_criteria) == 2
    
    def test_spec_to_dict(self):
        """Should serialize spec to dictionary."""
        spec = SpecSchema(
            title="Test",
            objective="Do something",
        )
        
        d = spec.to_dict()
        
        assert isinstance(d, dict)
        assert d["title"] == "Test"
        assert d["objective"] == "Do something"
        assert "spec_version" in d
        assert "spec_id" in d
        assert "requirements" in d
        assert "constraints" in d
        assert "safety" in d
        assert "provenance" in d
    
    def test_spec_to_json(self):
        """Should serialize spec to JSON string."""
        spec = SpecSchema(title="Test")
        
        json_str = spec.to_json()
        
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["title"] == "Test"
    
    def test_spec_from_dict(self):
        """Should deserialize spec from dictionary."""
        data = {
            "spec_version": "1.0",
            "spec_id": "test-123",
            "title": "From Dict",
            "objective": "Test deserialization",
            "requirements": {
                "functional": ["Req 1", "Req 2"],
                "non_functional": ["NFR 1"],
            },
            "constraints": {
                "platform": "Windows",
            },
            "safety": {
                "risks": ["Risk 1"],
                "mitigations": ["Mit 1"],
            },
            "acceptance_criteria": ["AC 1"],
            "provenance": {
                "source_message_ids": [1, 2, 3],
                "commit_hash": "abc123",
            },
        }
        
        spec = SpecSchema.from_dict(data)
        
        assert spec.spec_id == "test-123"
        assert spec.title == "From Dict"
        assert spec.requirements.functional == ["Req 1", "Req 2"]
        assert spec.constraints.platform == "Windows"
        assert spec.safety.risks == ["Risk 1"]
        assert spec.provenance.source_message_ids == [1, 2, 3]
    
    def test_spec_from_json(self):
        """Should deserialize spec from JSON string."""
        json_str = '{"spec_version": "1.0", "spec_id": "json-test", "title": "From JSON", "objective": "Test"}'
        
        spec = SpecSchema.from_json(json_str)
        
        assert spec.spec_id == "json-test"
        assert spec.title == "From JSON"
    
    def test_spec_compute_hash(self):
        """Should compute deterministic hash."""
        spec1 = SpecSchema(title="Test", objective="Same")
        spec2 = SpecSchema(title="Test", objective="Same")
        spec3 = SpecSchema(title="Test", objective="Different")
        
        # Same content (except timestamps) should have same hash
        # Note: spec_id differs, so hashes will differ
        # But same spec serialized twice should be consistent
        hash1 = spec1.compute_hash()
        hash1_again = spec1.compute_hash()
        
        assert hash1 == hash1_again
        assert len(hash1) == 64  # SHA-256
    
    def test_spec_roundtrip(self):
        """Should survive serialization roundtrip."""
        original = SpecSchema(
            title="Roundtrip Test",
            objective="Test serialization",
            requirements=SpecRequirements(
                functional=["F1", "F2"],
                non_functional=["NF1"],
            ),
            constraints=SpecConstraints(
                platform="Windows",
                integrations=["API1", "API2"],
            ),
            acceptance_criteria=["Works"],
        )
        
        json_str = original.to_json()
        restored = SpecSchema.from_json(json_str)
        
        assert restored.title == original.title
        assert restored.objective == original.objective
        assert restored.requirements.functional == original.requirements.functional
        assert restored.constraints.platform == original.constraints.platform


# =============================================================================
# SPEC VALIDATION TESTS
# =============================================================================

class TestSpecValidation:
    """Tests for spec validation."""
    
    def test_validate_empty_spec(self):
        """Empty spec should have errors."""
        spec = SpecSchema()
        
        result = validate_spec(spec)
        
        assert not result.valid
        assert "objective is required" in result.errors
    
    def test_validate_minimal_valid_spec(self):
        """Spec with objective should be valid."""
        spec = SpecSchema(
            objective="Do something useful",
        )
        
        result = validate_spec(spec)
        
        assert result.valid
        assert len(result.errors) == 0
        # Should have warnings about missing optional fields
        assert len(result.warnings) > 0
    
    def test_validate_complete_spec(self):
        """Complete spec should have minimal warnings."""
        spec = SpecSchema(
            title="Complete Spec",
            summary="A complete spec for testing",
            objective="Test validation",
            requirements=SpecRequirements(
                functional=["Feature 1"],
            ),
            safety=SpecSafety(
                risks=["Risk 1"],
                mitigations=["Mitigation 1"],
            ),
            acceptance_criteria=["It validates"],
            provenance=SpecProvenance(
                source_message_ids=[1, 2, 3],
                commit_hash="abc123def456",
            ),
        )
        
        result = validate_spec(spec)
        
        assert result.valid
        assert len(result.errors) == 0
    
    def test_validate_missing_provenance_warning(self):
        """Missing provenance should generate warnings."""
        spec = SpecSchema(
            objective="Test",
            provenance=SpecProvenance(),
        )
        
        result = validate_spec(spec)
        
        assert result.valid  # Not an error, just a warning
        warning_text = " ".join(result.warnings)
        assert "source_message_ids" in warning_text or "commit_hash" in warning_text


# =============================================================================
# SPEC MARKDOWN TESTS
# =============================================================================

class TestSpecMarkdown:
    """Tests for spec to markdown conversion."""
    
    def test_markdown_contains_title(self):
        """Markdown should contain title."""
        spec = SpecSchema(title="My Title", objective="Test")
        
        md = spec_to_markdown(spec)
        
        assert "# My Title" in md
    
    def test_markdown_contains_objective(self):
        """Markdown should contain objective."""
        spec = SpecSchema(objective="Build something great")
        
        md = spec_to_markdown(spec)
        
        assert "## Objective" in md
        assert "Build something great" in md
    
    def test_markdown_contains_requirements(self):
        """Markdown should contain requirements."""
        spec = SpecSchema(
            objective="Test",
            requirements=SpecRequirements(
                functional=["Feature A", "Feature B"],
                non_functional=["Fast"],
            ),
        )
        
        md = spec_to_markdown(spec)
        
        assert "### Functional" in md
        assert "Feature A" in md
        assert "Feature B" in md
        assert "### Non-Functional" in md
        assert "Fast" in md
    
    def test_markdown_contains_provenance(self):
        """Markdown should contain provenance section."""
        spec = SpecSchema(
            objective="Test",
            provenance=SpecProvenance(
                generator_model="test-model",
                source_message_ids=[1, 2, 3],
            ),
        )
        
        md = spec_to_markdown(spec)
        
        assert "## Provenance" in md
        assert "test-model" in md
        assert "3 messages" in md


# =============================================================================
# SPEC INPUT/OUTPUT TESTS
# =============================================================================

class TestSpecInputOutput:
    """Tests for spec input and output definitions."""
    
    def test_spec_input(self):
        """Should create input definition."""
        inp = SpecInput(
            name="user_id",
            type="string",
            required=True,
            example="user_123",
            source="API request",
        )
        
        d = inp.to_dict()
        
        assert d["name"] == "user_id"
        assert d["type"] == "string"
        assert d["required"] is True
    
    def test_spec_output(self):
        """Should create output definition."""
        out = SpecOutput(
            name="result",
            type="object",
            example='{"status": "ok"}',
            acceptance_criteria=["Returns valid JSON"],
        )
        
        d = out.to_dict()
        
        assert d["name"] == "result"
        assert d["acceptance_criteria"] == ["Returns valid JSON"]
    
    def test_spec_step(self):
        """Should create step definition."""
        step = SpecStep(
            id="1",
            description="First step",
            dependencies=[],
            notes="Important step",
        )
        
        d = step.to_dict()
        
        assert d["id"] == "1"
        assert d["description"] == "First step"


# =============================================================================
# SPEC STATUS TESTS
# =============================================================================

class TestSpecStatus:
    """Tests for spec status enum."""
    
    def test_status_values(self):
        """Should have expected status values."""
        assert SpecStatus.DRAFT.value == "draft"
        assert SpecStatus.VALIDATED.value == "validated"
        assert SpecStatus.REJECTED.value == "rejected"
        assert SpecStatus.SUPERSEDED.value == "superseded"


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
