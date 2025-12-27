# FILE: tests/test_spec_gate_parse.py
"""Tests for Spec Gate parsing and validation logic.

These tests verify:
1. JSON extraction from model output
2. Draft validation (meaningful content check)
3. Edge cases (empty, malformed, minimal content)
"""

import pytest

from app.pot_spec.spec_gate import (
    parse_spec_gate_output,
    _extract_json_object,
    _is_draft_meaningful,
    _SpecGateDraft,
    _safe_excerpt,
)


class TestExtractJsonObject:
    """Tests for _extract_json_object helper."""

    def test_extracts_clean_json(self):
        text = '{"goal": "test", "requirements": {}}'
        result = _extract_json_object(text)
        assert result == text

    def test_extracts_json_with_leading_text(self):
        text = 'Here is the spec:\n{"goal": "test"}'
        result = _extract_json_object(text)
        assert result == '{"goal": "test"}'

    def test_extracts_json_with_trailing_text(self):
        text = '{"goal": "test"}\nLet me know if you need changes.'
        result = _extract_json_object(text)
        assert result == '{"goal": "test"}'

    def test_extracts_json_with_code_fences(self):
        text = '```json\n{"goal": "test"}\n```'
        result = _extract_json_object(text)
        assert result is not None
        assert "goal" in result

    def test_returns_none_for_no_json(self):
        text = "This is just plain text with no JSON"
        result = _extract_json_object(text)
        assert result is None

    def test_returns_none_for_empty_string(self):
        result = _extract_json_object("")
        assert result is None

    def test_returns_none_for_none(self):
        result = _extract_json_object(None)
        assert result is None

    def test_handles_nested_braces(self):
        text = '{"goal": "test", "constraints": {"nested": {"deep": true}}}'
        result = _extract_json_object(text)
        assert result == text


class TestParseSpecGateOutput:
    """Tests for parse_spec_gate_output function."""

    def test_parses_valid_minimal_json(self):
        text = '{"goal": "Build a REST API", "requirements": {"must": [], "should": [], "can": []}}'
        draft = parse_spec_gate_output(text)
        assert draft.goal == "Build a REST API"
        assert draft.requirements == {"must": [], "should": [], "can": []}

    def test_parses_full_spec(self):
        text = '''
        {
            "goal": "Create a user authentication system",
            "requirements": {
                "must": ["Support email/password login"],
                "should": ["Include password reset"],
                "can": ["Add social login later"]
            },
            "constraints": {"database": "PostgreSQL"},
            "acceptance_tests": ["User can log in with valid credentials"],
            "open_questions": [],
            "recommendations": ["Use bcrypt for hashing"]
        }
        '''
        draft = parse_spec_gate_output(text)
        assert draft.goal == "Create a user authentication system"
        assert draft.requirements["must"] == ["Support email/password login"]
        assert draft.constraints == {"database": "PostgreSQL"}
        assert len(draft.acceptance_tests) == 1
        assert draft.open_questions == []
        assert len(draft.recommendations) == 1

    def test_extracts_json_from_noisy_output(self):
        text = """
        noise before
        ```json
        {
          "goal": "x",
          "requirements": {"must": [], "should": [], "can": []},
          "constraints": {},
          "acceptance_tests": [],
          "open_questions": [],
          "recommendations": [],
          "repo_snapshot": null
        }
        ```
        trailing noise
        """
        draft = parse_spec_gate_output(text)
        assert draft.goal == "x"

    def test_raises_on_empty_string(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_spec_gate_output("")

    def test_raises_on_non_json(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_spec_gate_output("This is not JSON at all")

    def test_raises_on_malformed_json(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            parse_spec_gate_output('{"goal": "missing closing brace"')

    def test_handles_empty_goal(self):
        # Empty goal is now allowed - validation happens separately
        text = '{"goal": "", "requirements": {"must": ["something"], "should": [], "can": []}}'
        draft = parse_spec_gate_output(text)
        assert draft.goal == ""
        assert draft.requirements["must"] == ["something"]

    def test_handles_missing_optional_fields(self):
        text = '{"goal": "test"}'
        draft = parse_spec_gate_output(text)
        assert draft.goal == "test"
        assert draft.requirements == {"must": [], "should": [], "can": []}
        assert draft.constraints == {}
        assert draft.acceptance_tests == []
        assert draft.open_questions == []


class TestIsDraftMeaningful:
    """Tests for _is_draft_meaningful validation."""

    def test_meaningful_with_good_goal(self):
        draft = _SpecGateDraft(
            goal="Build a comprehensive REST API with authentication and authorization",
            requirements={"must": [], "should": [], "can": []},
            constraints={},
            acceptance_tests=[],
            open_questions=[],
            recommendations=[],
        )
        assert _is_draft_meaningful(draft) is True

    def test_meaningful_with_requirements(self):
        draft = _SpecGateDraft(
            goal="",
            requirements={"must": ["Implement user login"], "should": [], "can": []},
            constraints={},
            acceptance_tests=[],
            open_questions=[],
            recommendations=[],
        )
        assert _is_draft_meaningful(draft) is True

    def test_meaningful_with_questions(self):
        draft = _SpecGateDraft(
            goal="",
            requirements={"must": [], "should": [], "can": []},
            constraints={},
            acceptance_tests=[],
            open_questions=["What database should we use?"],
            recommendations=[],
        )
        assert _is_draft_meaningful(draft) is True

    def test_not_meaningful_when_empty(self):
        draft = _SpecGateDraft(
            goal="",
            requirements={"must": [], "should": [], "can": []},
            constraints={},
            acceptance_tests=[],
            open_questions=[],
            recommendations=[],
        )
        assert _is_draft_meaningful(draft) is False

    def test_not_meaningful_with_too_short_goal(self):
        draft = _SpecGateDraft(
            goal="Do it",  # Less than 10 chars
            requirements={"must": [], "should": [], "can": []},
            constraints={},
            acceptance_tests=[],
            open_questions=[],
            recommendations=[],
        )
        assert _is_draft_meaningful(draft) is False

    def test_not_meaningful_with_empty_questions(self):
        draft = _SpecGateDraft(
            goal="",
            requirements={"must": [], "should": [], "can": []},
            constraints={},
            acceptance_tests=[],
            open_questions=["", "   "],  # Empty/whitespace questions don't count
            recommendations=[],
        )
        assert _is_draft_meaningful(draft) is False


class TestSafeExcerpt:
    """Tests for _safe_excerpt helper."""

    def test_returns_short_text_unchanged(self):
        text = "Short text"
        assert _safe_excerpt(text, 100) == text

    def test_truncates_long_text(self):
        text = "A" * 100
        result = _safe_excerpt(text, 50)
        assert len(result) < 100
        assert "truncated" in result
        assert "100 total chars" in result

    def test_handles_empty_string(self):
        assert _safe_excerpt("") == ""

    def test_handles_none(self):
        assert _safe_excerpt(None) == ""


# Legacy test (from original file)
def test_parse_spec_gate_output_extracts_json():
    """Original test case - kept for backward compatibility."""
    text = "noise\n{\n  \"goal\": \"x\",\n  \"requirements\": {\"must\": [], \"should\": [], \"can\": []},\n  \"constraints\": {},\n  \"acceptance_tests\": [],\n  \"open_questions\": [],\n  \"recommendations\": [],\n  \"repo_snapshot\": null\n}\ntrailing"
    draft = parse_spec_gate_output(text)
    assert draft.goal == "x"
