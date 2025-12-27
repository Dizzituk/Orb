# FILE: tests/test_critique_schemas.py
"""Unit tests for critique schemas (Block 5)."""

import json
import pytest
from app.llm.pipeline.critique_schemas import (
    CritiqueIssue,
    CritiqueResult,
    extract_json_from_llm_output,
    parse_critique_output,
    build_json_critique_prompt,
    build_json_revision_prompt,
)


class TestCritiqueIssue:
    def test_to_dict(self):
        issue = CritiqueIssue(
            id="ISSUE-001",
            spec_ref="MUST-1",
            arch_ref="Section 2",
            category="security",
            severity="blocking",
            description="Missing auth",
            fix_suggestion="Add JWT",
        )
        d = issue.to_dict()
        assert d["id"] == "ISSUE-001"
        assert d["spec_ref"] == "MUST-1"
        assert d["severity"] == "blocking"

    def test_from_dict(self):
        data = {
            "id": "ISSUE-002",
            "spec_ref": "SHOULD-1",
            "arch_ref": None,
            "category": "clarity",
            "severity": "non_blocking",
            "description": "Unclear",
            "fix_suggestion": "Add docs",
        }
        issue = CritiqueIssue.from_dict(data)
        assert issue.id == "ISSUE-002"
        assert issue.severity == "non_blocking"


class TestCritiqueResult:
    def test_overall_pass_computed(self):
        # No blocking issues = pass
        result = CritiqueResult(blocking_issues=[], non_blocking_issues=[])
        assert result.overall_pass is True

    def test_overall_fail_when_blocking(self):
        issue = CritiqueIssue(
            id="ISSUE-001", spec_ref=None, arch_ref=None,
            category="security", severity="blocking",
            description="Bad", fix_suggestion="Fix",
        )
        result = CritiqueResult(blocking_issues=[issue])
        assert result.overall_pass is False

    def test_to_json_and_back(self):
        issue = CritiqueIssue(
            id="ISSUE-001", spec_ref="MUST-1", arch_ref="Sec 2",
            category="correctness", severity="blocking",
            description="Wrong", fix_suggestion="Fix it",
        )
        result = CritiqueResult(
            blocking_issues=[issue],
            summary="Needs work",
            critique_model="gemini-2.0-flash",
        )
        
        json_str = result.to_json()
        parsed = CritiqueResult.from_json(json_str)
        
        assert len(parsed.blocking_issues) == 1
        assert parsed.blocking_issues[0].id == "ISSUE-001"
        assert parsed.overall_pass is False
        assert parsed.summary == "Needs work"

    def test_to_markdown(self):
        issue = CritiqueIssue(
            id="ISSUE-001", spec_ref="MUST-1", arch_ref="Sec 2",
            category="security", severity="blocking",
            description="Missing auth", fix_suggestion="Add JWT",
        )
        result = CritiqueResult(
            blocking_issues=[issue],
            summary="Security gaps",
            critique_model="gemini",
        )
        
        md = result.to_markdown()
        assert "# Architecture Critique Report" in md
        assert "FAILED" in md
        assert "ISSUE-001" in md
        assert "Missing auth" in md


class TestExtractJson:
    def test_clean_json(self):
        data = extract_json_from_llm_output('{"foo": "bar"}')
        assert data == {"foo": "bar"}

    def test_json_in_code_fence(self):
        raw = '''Here is the result:
```json
{"blocking_issues": [], "overall_pass": true}
```
Done.'''
        data = extract_json_from_llm_output(raw)
        assert data["overall_pass"] is True

    def test_json_with_prose(self):
        raw = '''I found some issues:
{"blocking_issues": [{"id": "ISSUE-001"}]}
Let me know if you need more.'''
        data = extract_json_from_llm_output(raw)
        assert len(data["blocking_issues"]) == 1

    def test_no_json(self):
        raw = "This is just prose with no JSON at all."
        data = extract_json_from_llm_output(raw)
        assert data is None

    def test_nested_braces(self):
        raw = '''{"outer": {"inner": {"deep": true}}}'''
        data = extract_json_from_llm_output(raw)
        assert data["outer"]["inner"]["deep"] is True


class TestParseCritiqueOutput:
    def test_valid_json(self):
        raw = json.dumps({
            "blocking_issues": [
                {"id": "ISSUE-001", "category": "security", "severity": "blocking",
                 "description": "Bad", "fix_suggestion": "Fix"}
            ],
            "non_blocking_issues": [],
            "summary": "Issues found",
            "spec_coverage": {"MUST-1": "partial"},
        })
        
        result = parse_critique_output(raw, model="test-model")
        assert len(result.blocking_issues) == 1
        assert result.overall_pass is False
        assert result.critique_model == "test-model"

    def test_invalid_json(self):
        raw = "This is not JSON at all"
        result = parse_critique_output(raw, model="test")
        assert "Failed to parse" in result.summary


class TestPromptBuilders:
    def test_json_critique_prompt(self):
        prompt = build_json_critique_prompt(
            draft_text="# My Architecture\n...",
            original_request="Build a todo app",
            spec_json='{"goal": "todo app"}',
        )
        
        assert "blocking_issues" in prompt
        assert "Build a todo app" in prompt
        assert "ONLY valid JSON" in prompt

    def test_json_revision_prompt(self):
        issue = CritiqueIssue(
            id="ISSUE-001", spec_ref="MUST-1", arch_ref=None,
            category="security", severity="blocking",
            description="No auth", fix_suggestion="Add auth",
        )
        critique = CritiqueResult(blocking_issues=[issue])
        
        prompt = build_json_revision_prompt(
            draft_text="# Architecture",
            original_request="Build app",
            critique=critique,
        )
        
        assert "ISSUE-001" in prompt
        assert "No auth" in prompt
        assert "Add auth" in prompt
