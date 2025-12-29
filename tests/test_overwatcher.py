# FILE: tests/test_overwatcher.py
"""Tests for Block 9: Overwatcher/Verification.

Tests:
1. ErrorSignature computation
2. Evidence bundle building
3. Overwatcher output parsing (no-code validation)
4. Strike tracking
5. Deep research integration
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pytest


class TestErrorSignature:
    """Test ErrorSignature computation."""
    
    def test_compute_from_traceback(self):
        """ErrorSignature extracts components from traceback."""
        from app.overwatcher.error_signature import compute_error_signature
        
        traceback = '''
Traceback (most recent call last):
  File "app/pot_spec/service.py", line 42, in process_job
    result = self._run_stage(stage)
  File "app/pot_spec/service.py", line 78, in _run_stage
    raise ValueError("Invalid stage config")
ValueError: Invalid stage config

FAILED tests/test_service.py::test_process_job
'''
        
        sig = compute_error_signature(traceback)
        
        assert sig.exception_type == "ValueError"
        assert sig.failing_test_name == "tests/test_service.py::test_process_job"
        assert len(sig.top_stack_frames) > 0
        assert sig.signature_hash  # Has a hash
    
    def test_same_error_matches(self):
        """Same traceback produces same signature."""
        from app.overwatcher.error_signature import compute_error_signature, signatures_match
        
        traceback = '''
  File "app/foo.py", line 10, in bar
    raise KeyError("missing")
KeyError: missing
FAILED tests/test_foo.py::test_bar
'''
        
        sig1 = compute_error_signature(traceback)
        sig2 = compute_error_signature(traceback)
        
        assert signatures_match(sig1, sig2)
        assert sig1.signature_hash == sig2.signature_hash
    
    def test_different_error_no_match(self):
        """Different errors produce different signatures."""
        from app.overwatcher.error_signature import compute_error_signature, signatures_match
        
        traceback1 = '''
  File "app/foo.py", line 10, in bar
    raise KeyError("missing")
KeyError: missing
'''
        
        traceback2 = '''
  File "app/baz.py", line 20, in qux
    raise TypeError("wrong type")
TypeError: wrong type
'''
        
        sig1 = compute_error_signature(traceback1)
        sig2 = compute_error_signature(traceback2)
        
        assert not signatures_match(sig1, sig2)
    
    def test_extract_exception_type(self):
        """Extract various exception types."""
        from app.overwatcher.error_signature import extract_exception_type
        
        assert extract_exception_type("ValueError: bad value") == "ValueError"
        assert extract_exception_type("TypeError: wrong type") == "TypeError"
        assert extract_exception_type("AssertionError") == "AssertionError"
        assert extract_exception_type("E   KeyError: 'foo'") == "KeyError"


class TestEvidenceBundle:
    """Test EvidenceBundle building."""
    
    def test_bundle_to_prompt_text(self):
        """Bundle converts to prompt text."""
        from app.overwatcher.evidence import (
            EvidenceBundle,
            FileChange,
            TestResult,
        )
        
        bundle = EvidenceBundle(
            job_id="job-001",
            chunk_id="CHUNK-001",
            stage_run_id="run-001",
            spec_id="SPEC-001",
            spec_hash="abc123def456",
            strike_number=1,
            file_changes=[
                FileChange(path="app/foo.py", action="add", intent="New module"),
                FileChange(path="app/bar.py", action="modify", intent="Fix bug"),
            ],
            test_result=TestResult(passed=5, failed=1, failing_tests=["test_foo"]),
            chunk_title="Add foo module",
            chunk_objective="Implement foo functionality",
        )
        
        text = bundle.to_prompt_text()
        
        assert "EVIDENCE BUNDLE" in text
        assert "job-001" in text
        assert "CHUNK-001" in text
        assert "Strike: 1/3" in text
        assert "app/foo.py" in text
        assert "[add]" in text
        assert "Passed: 5" in text
        assert "Failed: 1" in text
    
    def test_bundle_token_estimate(self):
        """Bundle provides token estimate."""
        from app.overwatcher.evidence import EvidenceBundle
        
        bundle = EvidenceBundle(
            job_id="job-001",
            chunk_id="CHUNK-001",
            stage_run_id="run-001",
            spec_id="SPEC-001",
            spec_hash="abc123",
            strike_number=1,
        )
        
        tokens = bundle.estimate_tokens()
        
        assert tokens > 0
        assert tokens < 1000  # Minimal bundle should be small
    
    def test_truncate_output(self):
        """Output truncation keeps head and tail."""
        from app.overwatcher.evidence import truncate_output
        
        lines = "\n".join([f"line {i}" for i in range(100)])
        
        truncated = truncate_output(lines, max_lines=20)
        
        assert "line 0" in truncated
        assert "line 99" in truncated
        assert "truncated" in truncated.lower()
        assert len(truncated.split("\n")) <= 21  # 20 + separator


class TestOverwatcherOutput:
    """Test Overwatcher output parsing and validation."""
    
    def test_parse_valid_json(self):
        """Parse valid JSON output."""
        from app.overwatcher.overwatcher import parse_overwatcher_output, Decision
        
        raw = '''
{
  "decision": "FAIL",
  "diagnosis": "Missing import statement",
  "fix_actions": [
    {
      "order": 1,
      "target_file": "app/foo.py",
      "action_type": "fix_import",
      "description": "Add missing import for datetime module",
      "rationale": "The datetime.now() call requires datetime import"
    }
  ],
  "constraints": ["Do not modify tests"],
  "verification": [
    {"command": "pytest tests/", "expected_outcome": "all pass", "timeout_seconds": 60}
  ],
  "blockers": ["Import error"],
  "nonblockers": [],
  "confidence": 0.9,
  "needs_deep_research": false
}
'''
        
        output = parse_overwatcher_output(raw)
        
        assert output.decision == Decision.FAIL
        assert "import" in output.diagnosis.lower()
        assert len(output.fix_actions) == 1
        assert output.fix_actions[0].target_file == "app/foo.py"
        assert output.confidence == 0.9
    
    def test_parse_json_in_code_fence(self):
        """Parse JSON wrapped in code fence."""
        from app.overwatcher.overwatcher import parse_overwatcher_output, Decision
        
        raw = '''
Here is my analysis:

```json
{
  "decision": "PASS",
  "diagnosis": "All tests passing",
  "fix_actions": [],
  "constraints": [],
  "verification": [],
  "blockers": [],
  "nonblockers": [],
  "confidence": 1.0,
  "needs_deep_research": false
}
```
'''
        
        output = parse_overwatcher_output(raw)
        
        assert output.decision == Decision.PASS
    
    def test_validate_no_code(self):
        """Detect code in output."""
        from app.overwatcher.overwatcher import OverwatcherOutput, Decision, FixAction
        
        # Valid - no code
        valid_output = OverwatcherOutput(
            decision=Decision.FAIL,
            diagnosis="Missing function",
            fix_actions=[
                FixAction(
                    order=1,
                    target_file="app/foo.py",
                    action_type="add_function",
                    description="Add a helper function to parse input",
                    rationale="Needed for input validation",
                )
            ],
        )
        
        violations = valid_output.validate_no_code()
        assert len(violations) == 0
        
        # Invalid - has code
        invalid_output = OverwatcherOutput(
            decision=Decision.FAIL,
            diagnosis="Missing function",
            fix_actions=[
                FixAction(
                    order=1,
                    target_file="app/foo.py",
                    action_type="add_function",
                    description="Add this code:\n```python\ndef foo():\n    pass\n```",
                    rationale="Needed",
                )
            ],
        )
        
        violations = invalid_output.validate_no_code()
        assert len(violations) > 0
    
    def test_contains_code_detection(self):
        """Detect various code patterns."""
        from app.overwatcher.overwatcher import contains_code
        
        # Should detect
        assert contains_code("```python\ndef foo():\n    pass\n```")
        assert contains_code("    def my_function():")
        assert contains_code("    class MyClass:")
        assert contains_code("+    def added_function():")
        
        # Should not detect
        assert not contains_code("Add a function called foo")
        assert not contains_code("The class needs a new method")
        assert not contains_code("import statement is missing")  # Natural language


class TestStrikeState:
    """Test strike tracking."""
    
    def test_same_error_increments(self):
        """Same error signature increments strike count."""
        from app.overwatcher.orchestrator import StrikeState
        from app.overwatcher.error_signature import ErrorSignature
        from app.overwatcher.overwatcher import OverwatcherOutput, Decision
        
        state = StrikeState(chunk_id="CHUNK-001")
        
        sig = ErrorSignature(
            exception_type="ValueError",
            failing_test_name="test_foo",
            top_stack_frames=["foo.py:bar"],
            module_path="app/foo.py",
        )
        
        output = OverwatcherOutput(decision=Decision.FAIL, diagnosis="error")
        
        # First strike
        count = state.record_strike(sig, output)
        assert count == 1
        
        # Same error - second strike
        count = state.record_strike(sig, output)
        assert count == 2
        
        # Same error - third strike
        count = state.record_strike(sig, output)
        assert count == 3
        assert state.is_exhausted()
    
    def test_different_error_resets(self):
        """Different error signature resets to 1."""
        from app.overwatcher.orchestrator import StrikeState
        from app.overwatcher.error_signature import ErrorSignature
        from app.overwatcher.overwatcher import OverwatcherOutput, Decision
        
        state = StrikeState(chunk_id="CHUNK-001")
        
        sig1 = ErrorSignature(
            exception_type="ValueError",
            failing_test_name="test_foo",
            top_stack_frames=["foo.py:bar"],
            module_path="app/foo.py",
        )
        
        sig2 = ErrorSignature(
            exception_type="TypeError",  # Different error
            failing_test_name="test_baz",
            top_stack_frames=["baz.py:qux"],
            module_path="app/baz.py",
        )
        
        output = OverwatcherOutput(decision=Decision.FAIL, diagnosis="error")
        
        # First error - strike 1
        count = state.record_strike(sig1, output)
        assert count == 1
        
        # Same error - strike 2
        count = state.record_strike(sig1, output)
        assert count == 2
        
        # Different error - reset to 1
        count = state.record_strike(sig2, output)
        assert count == 1
        assert not state.is_exhausted()


class TestDeepResearch:
    """Test Deep Research components."""
    
    def test_generate_search_queries(self):
        """Generate search queries from error signature."""
        from app.overwatcher.deep_research import generate_search_queries
        from app.overwatcher.error_signature import ErrorSignature
        
        sig = ErrorSignature(
            exception_type="ValueError",
            failing_test_name="tests/test_foo.py::test_bar",
            top_stack_frames=["foo.py:bar"],
            module_path="app/foo.py",
        )
        
        queries = generate_search_queries(sig)
        
        assert len(queries) > 0
        assert len(queries) <= 3
        assert any("ValueError" in q for q in queries)
    
    def test_parse_research_result(self):
        """Parse research result JSON."""
        from app.overwatcher.deep_research import parse_research_result
        
        raw = '''
{
  "explanation": "This is a common Python error",
  "likely_cause": "Missing type conversion",
  "suggested_fix": "Add explicit int() conversion before comparison",
  "sources": [
    {"title": "Stack Overflow", "url": "https://so.com/q/123", "excerpt": "Convert to int", "relevance": 0.9}
  ],
  "confidence": 0.85
}
'''
        
        result = parse_research_result(raw)
        
        assert "Python error" in result.explanation
        assert "conversion" in result.likely_cause.lower()
        assert len(result.sources) == 1
        assert result.confidence == 0.85
    
    def test_result_to_context_string(self):
        """Research result converts to context string."""
        from app.overwatcher.deep_research import DeepResearchResult, ResearchSource
        
        result = DeepResearchResult(
            explanation="This is a type error",
            likely_cause="Incompatible types",
            suggested_fix="Add type conversion",
            sources=[
                ResearchSource(
                    title="Python Docs",
                    url="https://docs.python.org",
                    excerpt="Type handling",
                    relevance=0.9,
                )
            ],
            confidence=0.8,
            search_calls_used=2,
        )
        
        context = result.to_context_string()
        
        assert "Deep Research" in context
        assert "type error" in context
        assert "Python Docs" in context
        assert "80%" in context


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
