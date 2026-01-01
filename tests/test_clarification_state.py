# FILE: tests/test_clarification_state.py
"""
Tests for Spec Gate clarification state management (Job 4).
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
import json
import tempfile

from app.pot_spec.clarification_state import (
    ClarificationState,
    ClarificationDecision,
    HoleType,
    compute_question_signature,
    compute_question_signature_simple,
    load_clarification_state,
    save_clarification_state,
    format_spec_summary_markdown,
    MAX_ROUNDS_PER_HOLE,
)


class TestQuestionSignature:
    def test_simple_signature_deterministic(self):
        q = "What database should we use?"
        sig1 = compute_question_signature_simple(q)
        sig2 = compute_question_signature_simple(q)
        assert sig1 == sig2
        assert len(sig1) == 16

    def test_whitespace_normalization(self):
        q1 = "What database should we use?"
        q2 = "  What   database  should   we  use?  "
        assert compute_question_signature_simple(q1) == compute_question_signature_simple(q2)

    def test_different_questions_different_sigs(self):
        sig1 = compute_question_signature_simple("What database?")
        sig2 = compute_question_signature_simple("What framework?")
        assert sig1 != sig2

    def test_hole_type_affects_signature(self):
        sig1 = compute_question_signature(HoleType.MISSING_INFO, "What?")
        sig2 = compute_question_signature(HoleType.AMBIGUITY, "What?")
        assert sig1 != sig2


class TestClarificationState:
    def test_initial_state(self):
        state = ClarificationState(job_id="job-123")
        assert state.current_round == 0
        assert not state.hard_stopped
        assert not state.ready_for_pipeline

    def test_record_questions_increments_round(self):
        state = ClarificationState(job_id="job-123")
        decision = state.record_questions(["Q1?", "Q2?"], spec_version=1)
        assert decision == ClarificationDecision.CONTINUE
        assert state.current_round == 1
        assert state.total_questions_asked == 2

    def test_empty_questions_ready_for_confirm(self):
        state = ClarificationState(job_id="job-123")
        decision = state.record_questions([], spec_version=1)
        assert decision == ClarificationDecision.READY_FOR_CONFIRM
        assert state.ready_for_pipeline is True

    def test_dedupe_skips_duplicates(self):
        state = ClarificationState(job_id="job-123")
        state.record_questions(["What database?"], spec_version=1)
        decision = state.record_questions(["What database?"], spec_version=2)
        assert decision == ClarificationDecision.ALREADY_ASKED
        assert state.total_questions_asked == 1

    def test_hard_stop_blocks_questions(self):
        state = ClarificationState(job_id="job-123")
        state.hard_stopped = True
        decision = state.record_questions(["New?"], spec_version=1)
        assert decision == ClarificationDecision.HARD_STOP

    def test_would_exceed_cap(self):
        state = ClarificationState(job_id="job-123")
        state.rounds_by_sig["hole-x"] = MAX_ROUNDS_PER_HOLE
        assert state.would_exceed_cap("hole-x") is True
        assert state.would_exceed_cap("hole-y") is False


class TestConfirmationGate:
    def test_confirm_when_ready(self):
        state = ClarificationState(job_id="job-123")
        state.ready_for_pipeline = True
        result = state.confirm_for_pipeline(spec_version=2, spec_hash="abc")
        assert result is True
        assert state.user_confirmed is True

    def test_cannot_confirm_not_ready(self):
        state = ClarificationState(job_id="job-123")
        result = state.confirm_for_pipeline(spec_version=1, spec_hash="x")
        assert result is False

    def test_cannot_confirm_hard_stopped(self):
        state = ClarificationState(job_id="job-123")
        state.ready_for_pipeline = True
        state.hard_stopped = True
        result = state.confirm_for_pipeline(spec_version=1, spec_hash="x")
        assert result is False


class TestPersistence:
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = ClarificationState(job_id="job-456")
            state.record_questions(["Q?"], spec_version=1)
            save_clarification_state(tmpdir, "job-456", state)
            loaded = load_clarification_state(tmpdir, "job-456")
            assert loaded.current_round == 1
            assert loaded.total_questions_asked == 1

    def test_load_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = load_clarification_state(tmpdir, "nope")
            assert loaded.job_id == "nope"
            assert loaded.current_round == 0


class TestSpecSummary:
    def test_basic_summary(self):
        spec = {
            "job_id": "job-123",
            "spec_id": "spec-abc",
            "spec_version": 2,
            "goal": "Build API",
            "requirements": {"must": ["Auth"], "should": [], "can": []},
        }
        md = format_spec_summary_markdown(spec, "spec.json")
        assert "job-123" in md
        assert "Build API" in md
        assert "Reply **Yes**" in md


class TestClarificationFlow:
    def test_full_flow(self):
        state = ClarificationState(job_id="job-flow")
        d1 = state.record_questions(["Q1?", "Q2?"], spec_version=1)
        assert d1 == ClarificationDecision.CONTINUE
        state.record_user_response("A1 A2")
        d2 = state.record_questions(["Q1?", "Q3?"], spec_version=2)  # Q1 is dupe
        assert d2 == ClarificationDecision.CONTINUE
        assert state.total_questions_asked == 3
        d3 = state.record_questions([], spec_version=3)
        assert d3 == ClarificationDecision.READY_FOR_CONFIRM
        assert state.confirm_for_pipeline(spec_version=3, spec_hash="final")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
