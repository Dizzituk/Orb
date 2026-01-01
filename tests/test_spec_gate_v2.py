# FILE: tests/test_spec_gate_v2.py
"""
Tests for Spec Gate v2 with clarification state integration.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
import json
import tempfile
from unittest.mock import patch

from app.pot_spec.spec_gate_v2 import (
    SpecGateResult,
    _filter_questions_with_clarification,
)
from app.pot_spec.clarification_state import (
    ClarificationState,
    ClarificationDecision,
    save_clarification_state,
)


class TestSpecGateResult:
    def test_basic_creation(self):
        result = SpecGateResult(
            spec_id="spec-123",
            spec_hash="hash456",
            open_questions=["Q1?", "Q2?"],
            spec_version=1,
            ready_for_pipeline=False,
        )
        assert result.spec_id == "spec-123"
        assert result.ready_for_pipeline is False

    def test_backward_compatible_unpacking(self):
        result = SpecGateResult(
            spec_id="spec-abc",
            spec_hash="hash-xyz",
            open_questions=["Question?"],
            spec_version=2,
            ready_for_pipeline=True,
        )
        sid, shash, questions = result
        assert sid == "spec-abc"
        assert shash == "hash-xyz"
        assert questions == ["Question?"]

    def test_indexing(self):
        result = SpecGateResult(
            spec_id="s1",
            spec_hash="h1",
            open_questions=["Q"],
            spec_version=1,
            ready_for_pipeline=False,
        )
        assert result[0] == "s1"
        assert result[1] == "h1"
        assert result[2] == ["Q"]

    def test_hard_stopped(self):
        result = SpecGateResult(
            spec_id="spec-stop",
            spec_hash="hash-stop",
            open_questions=[],
            spec_version=1,
            ready_for_pipeline=False,
            hard_stopped=True,
            hard_stop_reason="Exceeded 3-round cap",
        )
        assert result.hard_stopped is True
        assert "3-round" in result.hard_stop_reason


class TestFilterQuestions:
    def test_empty_questions_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filtered, decision, state = _filter_questions_with_clarification(
                questions=[],
                job_artifact_root=tmpdir,
                job_id="job-empty",
                spec_version=1,
            )
            assert decision == ClarificationDecision.READY_FOR_CONFIRM
            assert filtered == []

    def test_dedupe_across_rounds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Round 1
            filtered1, decision1, _ = _filter_questions_with_clarification(
                questions=["What database?"],
                job_artifact_root=tmpdir,
                job_id="job-dedupe",
                spec_version=1,
            )
            assert decision1 == ClarificationDecision.CONTINUE
            assert "What database?" in filtered1

            # Round 2 - same question should be deduped
            filtered2, decision2, _ = _filter_questions_with_clarification(
                questions=["What database?"],
                job_artifact_root=tmpdir,
                job_id="job-dedupe",
                spec_version=2,
            )
            assert decision2 == ClarificationDecision.READY_FOR_CONFIRM
            assert filtered2 == []


class TestClarificationFlow:
    def test_result_iteration_in_loop(self):
        result = SpecGateResult(
            spec_id="s",
            spec_hash="h",
            open_questions=["q"],
            spec_version=1,
            ready_for_pipeline=False,
        )
        items = list(result)
        assert items == ["s", "h", ["q"]]

    def test_decision_flow_continue_to_ready(self):
        state = ClarificationState(job_id="flow-test")
        
        # Round 1: Has questions
        d1 = state.record_questions(["Q1?"], spec_version=1)
        assert d1 == ClarificationDecision.CONTINUE
        assert not state.ready_for_pipeline
        
        # Round 2: No questions
        d2 = state.record_questions([], spec_version=2)
        assert d2 == ClarificationDecision.READY_FOR_CONFIRM
        assert state.ready_for_pipeline
        
        # Confirm
        confirmed = state.confirm_for_pipeline(spec_version=2, spec_hash="final")
        assert confirmed is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
