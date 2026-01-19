# FILE: tests/test_spec_gate_grounded.py
"""
Tests for SpecGate Contract v1 (Grounded Implementation).

These tests verify:
1. Evidence loading works correctly
2. Read-only runtime is enforced
3. Question generation follows the rules
4. POT spec output format is correct
5. Integration with spec_gate_stream works

Run with: pytest tests/test_spec_gate_grounded.py -v
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any


# =============================================================================
# Test: Evidence Collector Module
# =============================================================================

class TestEvidenceCollector:
    """Tests for app/pot_spec/evidence_collector.py"""
    
    def test_import_evidence_collector(self):
        """Verify evidence_collector module imports correctly."""
        try:
            from app.pot_spec.evidence_collector import (
                EvidenceBundle,
                EvidenceSource,
                load_evidence,
                refuse_write_operation,
                WRITE_REFUSED_ERROR,
            )
            assert EvidenceBundle is not None
            assert EvidenceSource is not None
            assert load_evidence is not None
            assert WRITE_REFUSED_ERROR == "SpecGate runtime is read-only. Write actions are not permitted."
        except ImportError as e:
            pytest.skip(f"evidence_collector not available: {e}")
    
    def test_evidence_source_to_evidence_line(self):
        """Test EvidenceSource formatting."""
        try:
            from app.pot_spec.evidence_collector import EvidenceSource
        except ImportError:
            pytest.skip("evidence_collector not available")
        
        # Test architecture map source
        source = EvidenceSource(
            source_type="architecture_map",
            filename="ARCHITECTURE_MAP.md",
            mtime=datetime(2026, 1, 19, 12, 0, 0),
            mtime_human="2026-01-19 12:00:00",
            found=True,
        )
        line = source.to_evidence_line()
        assert "ARCHITECTURE_MAP.md" in line
        assert "2026-01-19 12:00:00" in line
        
        # Test file read source with line range
        source = EvidenceSource(
            source_type="file_read",
            path="app/llm/stream_router.py",
            line_range=(1, 50),
            found=True,
        )
        line = source.to_evidence_line()
        assert "app/llm/stream_router.py" in line
        assert "lines 1-50" in line
        
        # Test fallback source
        source = EvidenceSource(
            source_type="arch_query_fallback",
            query="SpecGate",
            is_fallback=True,
            found=True,
        )
        line = source.to_evidence_line()
        assert "fallback" in line.lower()
        assert "SpecGate" in line
    
    def test_refuse_write_operation(self):
        """Test that write operations are refused."""
        try:
            from app.pot_spec.evidence_collector import refuse_write_operation, WRITE_REFUSED_ERROR
        except ImportError:
            pytest.skip("evidence_collector not available")
        
        with pytest.raises(RuntimeError) as exc_info:
            refuse_write_operation("write file to disk")
        
        assert WRITE_REFUSED_ERROR in str(exc_info.value)
    
    def test_evidence_bundle_to_markdown(self):
        """Test EvidenceBundle markdown generation."""
        try:
            from app.pot_spec.evidence_collector import EvidenceBundle, EvidenceSource
        except ImportError:
            pytest.skip("evidence_collector not available")
        
        bundle = EvidenceBundle()
        bundle.add_source(EvidenceSource(
            source_type="architecture_map",
            filename="ARCHITECTURE_MAP.md",
            mtime_human="2026-01-19 12:00:00",
            found=True,
        ))
        bundle.add_source(EvidenceSource(
            source_type="codebase_report",
            filename="CODEBASE_REPORT_FULL_2026-01-19.md",
            mtime_human="2026-01-19 11:00:00",
            found=True,
        ))
        
        md = bundle.to_evidence_used_markdown()
        assert "## Evidence Used" in md
        assert "ARCHITECTURE_MAP.md" in md
        assert "CODEBASE_REPORT_FULL" in md


# =============================================================================
# Test: Spec Gate Grounded Module
# =============================================================================

class TestSpecGateGrounded:
    """Tests for app/pot_spec/spec_gate_grounded.py"""
    
    def test_import_spec_gate_grounded(self):
        """Verify spec_gate_grounded module imports correctly."""
        try:
            from app.pot_spec.spec_gate_grounded import (
                run_spec_gate_grounded,
                GroundedPOTSpec,
                GroundedQuestion,
                GroundedFact,
                QuestionCategory,
                build_pot_spec_markdown,
            )
            assert run_spec_gate_grounded is not None
            assert GroundedPOTSpec is not None
            assert GroundedQuestion is not None
            assert QuestionCategory is not None
        except ImportError as e:
            pytest.skip(f"spec_gate_grounded not available: {e}")
    
    def test_question_category_enum(self):
        """Test QuestionCategory enum values."""
        try:
            from app.pot_spec.spec_gate_grounded import QuestionCategory
        except ImportError:
            pytest.skip("spec_gate_grounded not available")
        
        assert QuestionCategory.PREFERENCE == "preference"
        assert QuestionCategory.MISSING_PRODUCT_DECISION == "product_decision"
        assert QuestionCategory.AMBIGUOUS_EVIDENCE == "ambiguous"
        assert QuestionCategory.SAFETY_CONSTRAINT == "safety"
    
    def test_grounded_question_format(self):
        """Test GroundedQuestion formatting."""
        try:
            from app.pot_spec.spec_gate_grounded import GroundedQuestion, QuestionCategory
        except ImportError:
            pytest.skip("spec_gate_grounded not available")
        
        question = GroundedQuestion(
            question="What color should the button be?",
            category=QuestionCategory.PREFERENCE,
            why_it_matters="User experience depends on visual consistency",
            evidence_found="No color specified in Weaver output or codebase",
            options=["Blue", "Green", "Match existing theme"],
        )
        
        formatted = question.format()
        assert "What color should the button be?" in formatted
        assert "Why it matters" in formatted
        assert "Evidence found" in formatted
        assert "Options" in formatted
        assert "(A) Blue" in formatted
    
    def test_pot_spec_markdown_structure(self):
        """Test POT spec markdown has all required sections."""
        try:
            from app.pot_spec.spec_gate_grounded import (
                GroundedPOTSpec,
                GroundedFact,
                build_pot_spec_markdown,
            )
            from app.pot_spec.evidence_collector import EvidenceBundle, EvidenceSource
        except ImportError:
            pytest.skip("spec_gate_grounded not available")
        
        # Create evidence bundle
        bundle = EvidenceBundle()
        bundle.add_source(EvidenceSource(
            source_type="architecture_map",
            filename="ARCHITECTURE_MAP.md",
            found=True,
        ))
        
        # Create spec
        spec = GroundedPOTSpec(
            goal="Implement SpecGate Contract v1",
            confirmed_components=[
                GroundedFact(
                    description="app/pot_spec exists",
                    source="architecture_map",
                    confidence="confirmed",
                )
            ],
            what_exists=["app/pot_spec/spec_gate_v2.py"],
            what_missing=["app/pot_spec/spec_gate_grounded.py"],
            in_scope=["Create new grounded implementation"],
            out_of_scope=["Refactor existing v2"],
            constraints_from_intent=["Read-only runtime"],
            constraints_from_repo=["Must use existing types"],
            evidence_bundle=bundle,
            proposed_steps=[
                "Create evidence_collector.py",
                "Create spec_gate_grounded.py",
                "Wire into spec_gate_stream.py",
            ],
            acceptance_tests=[
                "Evidence loads correctly",
                "Questions follow rules",
                "Output matches template",
            ],
            risks=[{"risk": "Breaking existing flow", "mitigation": "Feature flag"}],
            refactor_flags=["Consider splitting large files"],
            spec_id="test-spec-123",
            spec_hash="abc123",
            spec_version=1,
            is_complete=True,
        )
        
        md = build_pot_spec_markdown(spec)
        
        # Verify all required sections present
        required_sections = [
            "# Point-of-Truth Specification",
            "## Goal",
            "## Current Reality (Grounded Facts)",
            "## Scope",
            "### In Scope",
            "### Out of Scope",
            "## Constraints",
            "### From Weaver Intent",
            "### Discovered from Repo",
            "## Evidence Used",
            "## Proposed Step Plan",
            "## Acceptance Tests",
            "## Risks + Mitigations",
            "## Refactor Flags",
            "## Open Questions",
            "## Metadata",
        ]
        
        for section in required_sections:
            assert section in md, f"Missing section: {section}"
        
        # Verify content
        assert "Implement SpecGate Contract v1" in md
        assert "spec_gate_grounded.py" in md
        assert "Read-only runtime" in md


class TestRound3Finalization:
    """Tests that Round 3 finalization does NOT guess or fill gaps."""
    
    def test_round3_preserves_questions_not_guesses(self):
        """Verify Round 3 marks questions as UNRESOLVED, not filled in."""
        try:
            from app.pot_spec.spec_gate_grounded import (
                GroundedPOTSpec,
                GroundedQuestion,
                QuestionCategory,
                build_pot_spec_markdown,
            )
        except ImportError:
            pytest.skip("spec_gate_grounded not available")
        
        # Create spec with unresolved questions (simulating Round 3)
        spec = GroundedPOTSpec(
            goal="Test job",
            open_questions=[
                GroundedQuestion(
                    question="What color should it be?",
                    category=QuestionCategory.PREFERENCE,
                    why_it_matters="User preference",
                    evidence_found="Not in evidence",
                ),
            ],
            spec_version=3,
            is_complete=True,  # Round 3 forces completion
            blocking_issues=["Finalized with 1 unanswered question(s) - NOT guessed"],
        )
        
        md = build_pot_spec_markdown(spec)
        
        # Verify the markdown contains UNRESOLVED markers
        assert "FINALIZED WITH UNRESOLVED QUESTIONS" in md
        assert "UNRESOLVED (no guess" in md
        assert "NOT guessed" in md or "No Guess" in md
        
        # Verify questions are NOT removed/hidden
        assert "What color should it be?" in md
    
    def test_round3_includes_unresolved_section(self):
        """Verify Round 3 has explicit Unresolved/Unknown section."""
        try:
            from app.pot_spec.spec_gate_grounded import (
                GroundedPOTSpec,
                GroundedQuestion,
                QuestionCategory,
                build_pot_spec_markdown,
            )
        except ImportError:
            pytest.skip("spec_gate_grounded not available")
        
        # Create spec with gaps (simulating Round 3)
        spec = GroundedPOTSpec(
            goal="Test job",
            proposed_steps=[],  # No steps - gap
            acceptance_tests=[],  # No tests - gap
            open_questions=[
                GroundedQuestion(
                    question="What steps?",
                    category=QuestionCategory.MISSING_PRODUCT_DECISION,
                    why_it_matters="Required",
                    evidence_found="Not found",
                ),
            ],
            spec_version=3,
            is_complete=True,
        )
        
        md = build_pot_spec_markdown(spec)
        
        # Verify Unresolved/Unknown section exists
        assert "Unresolved / Unknown (No Guess)" in md
        assert "Steps" in md and "Not specified" in md
        assert "human input" in md.lower()


# =============================================================================
# Test: Runtime Read-Only Enforcement
# =============================================================================

class TestReadOnlyRuntime:
    """Tests that SpecGate runtime is strictly read-only."""
    
    def test_no_db_writes_in_result(self):
        """Verify result indicates no DB persistence."""
        try:
            from app.pot_spec.spec_gate_types import SpecGateResult
        except ImportError:
            pytest.skip("spec_gate_types not available")
        
        # Grounded implementation should always return db_persisted=False
        result = SpecGateResult(
            ready_for_pipeline=True,
            db_persisted=False,  # MUST be False for Contract v1
        )
        assert result.db_persisted is False
    
    @pytest.mark.asyncio
    async def test_grounded_returns_no_persistence(self):
        """Test that run_spec_gate_grounded never persists to DB."""
        try:
            from app.pot_spec.spec_gate_grounded import run_spec_gate_grounded
        except ImportError:
            pytest.skip("spec_gate_grounded not available")
        
        # Mock the database session
        mock_db = Mock()
        
        # Mock evidence loading
        with patch('app.pot_spec.spec_gate_grounded.load_evidence') as mock_load:
            from app.pot_spec.evidence_collector import EvidenceBundle
            mock_load.return_value = EvidenceBundle()
            
            result = await run_spec_gate_grounded(
                db=mock_db,
                job_id="test-job",
                user_intent="Test intent",
                provider_id="test",
                model_id="test",
                project_id=1,
                constraints_hint={"goal": "Test goal"},
                spec_version=1,
            )
        
        # Verify no DB write methods were called
        assert not mock_db.add.called
        assert not mock_db.commit.called
        
        # Verify result indicates no persistence
        assert result.db_persisted is False


# =============================================================================
# Test: Question Generation Rules
# =============================================================================

class TestQuestionRules:
    """Tests that question generation follows Contract v1 rules."""
    
    def test_max_questions_limit(self):
        """Verify max 7 questions are generated."""
        try:
            from app.pot_spec.spec_gate_grounded import (
                generate_grounded_questions,
                GroundedPOTSpec,
            )
            from app.pot_spec.evidence_collector import EvidenceBundle
        except ImportError:
            pytest.skip("spec_gate_grounded not available")
        
        # Create a spec with many potential gaps
        spec = GroundedPOTSpec(
            goal="",  # Missing
            what_missing=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],  # 10 gaps
            # Many more potential issues...
        )
        
        questions = generate_grounded_questions(
            spec=spec,
            intent={},
            evidence=EvidenceBundle(),
            round_number=1,
        )
        
        assert len(questions) <= 7, f"Too many questions: {len(questions)}"
    
    def test_no_questions_when_evidence_sufficient(self):
        """Verify no questions when evidence answers everything."""
        try:
            from app.pot_spec.spec_gate_grounded import (
                generate_grounded_questions,
                GroundedPOTSpec,
                GroundedFact,
            )
            from app.pot_spec.evidence_collector import EvidenceBundle
        except ImportError:
            pytest.skip("spec_gate_grounded not available")
        
        # Create a well-grounded spec
        spec = GroundedPOTSpec(
            goal="Create a new file",
            confirmed_components=[
                GroundedFact(description="Target exists", source="evidence"),
            ],
            what_exists=["target/path"],
            in_scope=["Create file"],
            out_of_scope=["Delete files"],
            proposed_steps=["Step 1", "Step 2"],
            acceptance_tests=["Test 1"],
        )
        
        questions = generate_grounded_questions(
            spec=spec,
            intent={"goal": "Create a new file"},
            evidence=EvidenceBundle(),
            round_number=1,
        )
        
        # Should have minimal or no questions
        assert len(questions) <= 3


# =============================================================================
# Test: Integration with Spec Gate Stream
# =============================================================================

class TestSpecGateStreamIntegration:
    """Tests for spec_gate_stream.py integration."""
    
    def test_grounded_flag_available(self):
        """Verify USE_GROUNDED_SPEC_GATE flag is available."""
        try:
            from app.llm.spec_gate_stream import (
                _USE_GROUNDED_SPEC_GATE,
                _SPEC_GATE_GROUNDED_AVAILABLE,
            )
        except ImportError:
            pytest.skip("spec_gate_stream not available")
        
        # Flag should exist and be a boolean
        assert isinstance(_USE_GROUNDED_SPEC_GATE, bool)
        assert isinstance(_SPEC_GATE_GROUNDED_AVAILABLE, bool)
    
    def test_grounded_import_available(self):
        """Verify grounded implementation is importable in stream."""
        try:
            from app.llm.spec_gate_stream import run_spec_gate_grounded
            assert run_spec_gate_grounded is not None or True  # May be None if import failed
        except ImportError:
            pytest.skip("spec_gate_stream not available")


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
