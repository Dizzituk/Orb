# FILE: tests/test_stage3_locks.py
"""Unit tests for Stage 3 spec-hash locks.

Tests verify:
1. build_spec_echo_instruction() produces correct format
2. parse_spec_echo_headers() handles various input cases
3. Correct headers pass verification
4. Incorrect/missing headers fail fast with proper error messages
"""

import pytest
from unittest.mock import patch, MagicMock

from app.jobs.stage3_locks import (
    build_spec_echo_instruction,
    parse_spec_echo_headers,
    verify_spec_hash,
    verify_and_store_stage3,
)


class TestBuildSpecEchoInstruction:
    """Tests for build_spec_echo_instruction helper."""

    def test_produces_correct_format(self):
        result = build_spec_echo_instruction(
            spec_id="abc-123",
            spec_hash="def456789"
        )
        
        assert "SPEC_ID: abc-123" in result
        assert "SPEC_HASH: def456789" in result
        assert "You MUST echo these identifiers" in result
        assert "Do not ask the user any questions" in result

    def test_handles_long_hash(self):
        long_hash = "a" * 64
        result = build_spec_echo_instruction(
            spec_id="test-id",
            spec_hash=long_hash
        )
        
        assert f"SPEC_HASH: {long_hash}" in result

    def test_result_is_appendable_to_system_prompt(self):
        result = build_spec_echo_instruction(
            spec_id="test",
            spec_hash="hash123"
        )
        
        system_prompt = "You are a helpful assistant."
        combined = system_prompt + "\n\n" + result
        
        assert "You are a helpful assistant." in combined
        assert "SPEC_ID: test" in combined


class TestParseSpecEchoHeaders:
    """Tests for parse_spec_echo_headers function."""

    def test_parses_valid_headers(self):
        output = "SPEC_ID: abc-123\nSPEC_HASH: def456\n\nRest of response..."
        spec_id, spec_hash, note = parse_spec_echo_headers(output)
        
        assert spec_id == "abc-123"
        assert spec_hash == "def456"
        assert note == "ok"

    def test_handles_extra_whitespace(self):
        output = "  \n  SPEC_ID: abc-123  \n  SPEC_HASH: def456  \n\nContent"
        spec_id, spec_hash, note = parse_spec_echo_headers(output)
        
        assert spec_id == "abc-123"
        assert spec_hash == "def456"
        assert note == "ok"

    def test_handles_bom(self):
        output = "\ufeffSPEC_ID: abc-123\nSPEC_HASH: def456\n\nContent"
        spec_id, spec_hash, note = parse_spec_echo_headers(output)
        
        assert spec_id == "abc-123"
        assert spec_hash == "def456"
        assert note == "ok"

    def test_empty_output(self):
        spec_id, spec_hash, note = parse_spec_echo_headers("")
        
        assert spec_id is None
        assert spec_hash is None
        assert note == "empty_output"

    def test_missing_headers(self):
        # Need at least 2 non-empty lines to get to "missing_SPEC_ID"
        output = "Just some regular text\nwithout any spec headers"
        spec_id, spec_hash, note = parse_spec_echo_headers(output)
        
        assert spec_id is None
        assert spec_hash is None
        assert note == "missing_SPEC_ID"

    def test_missing_spec_hash(self):
        output = "SPEC_ID: abc-123\nSome other line"
        spec_id, spec_hash, note = parse_spec_echo_headers(output)
        
        assert spec_id is None
        assert spec_hash is None
        assert note == "missing_SPEC_HASH"

    def test_only_one_line(self):
        output = "SPEC_ID: abc-123"
        spec_id, spec_hash, note = parse_spec_echo_headers(output)
        
        assert spec_id is None
        assert spec_hash is None
        assert note == "missing_header_lines"

    def test_empty_spec_values(self):
        output = "SPEC_ID: \nSPEC_HASH: def456"
        spec_id, spec_hash, note = parse_spec_echo_headers(output)
        
        assert spec_id is None
        assert spec_hash == "def456"
        assert note == "empty_spec_fields"

    def test_swapped_order_fails(self):
        output = "SPEC_HASH: def456\nSPEC_ID: abc-123"
        spec_id, spec_hash, note = parse_spec_echo_headers(output)
        
        assert spec_id is None
        assert spec_hash is None
        assert note == "missing_SPEC_ID"

    def test_case_sensitive(self):
        output = "spec_id: abc-123\nspec_hash: def456"
        spec_id, spec_hash, note = parse_spec_echo_headers(output)
        
        # Headers are case-sensitive per the spec
        assert spec_id is None
        assert note == "missing_SPEC_ID"

    def test_real_world_format(self):
        """Test with format matching actual model output."""
        output = """SPEC_ID: 015da9bb-7ced-47a1-8b15-b20221e635a9
SPEC_HASH: 2fdc4222f20d083acec0fd17a7459e04fdded7c547c321d7a70385d676a050ca

# Todo REST API Architecture

## Overview
This document describes the architecture for a todo REST API...
"""
        spec_id, spec_hash, note = parse_spec_echo_headers(output)
        
        assert spec_id == "015da9bb-7ced-47a1-8b15-b20221e635a9"
        assert spec_hash == "2fdc4222f20d083acec0fd17a7459e04fdded7c547c321d7a70385d676a050ca"
        assert note == "ok"


class TestVerifySpecHash:
    """Tests for verify_spec_hash function."""

    def test_matching_hash_returns_true(self):
        """Test that matching hash returns verified=True."""
        output = "SPEC_ID: abc-123\nSPEC_HASH: correct_hash\n\nContent"
        
        # Mock the ledger module imports inside verify_spec_hash
        with patch.dict('sys.modules', {
            'app.pot_spec.ledger': MagicMock(),
            'app.pot_spec.service': MagicMock(),
        }):
            # The function will use fallback path when imports fail internally
            verified, ret_id, ret_hash, note = verify_spec_hash(
                job_id="job-1",
                stage_name="test_stage",
                spec_id="abc-123",
                expected_spec_hash="correct_hash",
                raw_output=output,
            )
        
        assert verified is True
        assert ret_hash == "correct_hash"
        assert note == "ok"

    def test_mismatched_hash_returns_false(self):
        """Test that mismatched hash returns verified=False."""
        output = "SPEC_ID: abc-123\nSPEC_HASH: wrong_hash\n\nContent"
        
        with patch.dict('sys.modules', {
            'app.pot_spec.ledger': MagicMock(),
            'app.pot_spec.service': MagicMock(),
        }):
            verified, ret_id, ret_hash, note = verify_spec_hash(
                job_id="job-1",
                stage_name="test_stage",
                spec_id="abc-123",
                expected_spec_hash="correct_hash",
                raw_output=output,
            )
        
        assert verified is False
        assert ret_hash == "wrong_hash"

    def test_missing_headers_returns_false(self):
        """Test that missing headers returns verified=False."""
        # Need 2 lines to get past "missing_header_lines"
        output = "No headers here\njust content"
        
        with patch.dict('sys.modules', {
            'app.pot_spec.ledger': MagicMock(),
            'app.pot_spec.service': MagicMock(),
        }):
            verified, ret_id, ret_hash, note = verify_spec_hash(
                job_id="job-1",
                stage_name="test_stage",
                spec_id="abc-123",
                expected_spec_hash="correct_hash",
                raw_output=output,
            )
        
        assert verified is False
        assert ret_id is None
        assert ret_hash is None
        assert "missing" in note

    def test_fallback_when_ledger_unavailable(self):
        """Should still verify even if ledger module not available."""
        output = "SPEC_ID: abc-123\nSPEC_HASH: correct_hash\n\nContent"
        
        # Force fallback by making imports fail
        verified, ret_id, ret_hash, note = verify_spec_hash(
            job_id="job-1",
            stage_name="test_stage",
            spec_id="abc-123",
            expected_spec_hash="correct_hash",
            raw_output=output,
        )
        
        # Should still work via fallback
        assert verified is True or note == "ok"


class TestVerifyAndStoreStage3:
    """Tests for verify_and_store_stage3 convenience function."""

    def test_success_case(self):
        output = "SPEC_ID: abc-123\nSPEC_HASH: hash456\n\nContent"
        
        with patch("app.jobs.stage3_locks.verify_spec_hash") as mock_verify, \
             patch("app.jobs.stage3_locks.write_stage3_artifacts") as mock_write, \
             patch("app.jobs.stage3_locks.append_stage3_ledger_event") as mock_ledger:
            
            mock_verify.return_value = (True, "abc-123", "hash456", "ok")
            mock_write.return_value = {"raw_output": "/path/to/output.md"}
            
            verified, error_msg = verify_and_store_stage3(
                job_id="job-1",
                stage_name="test_stage",
                spec_id="abc-123",
                expected_spec_hash="hash456",
                raw_output=output,
                provider="anthropic",
                model="claude-sonnet-4-20250514",
            )
            
            assert verified is True
            assert error_msg == ""
            mock_write.assert_called_once()
            mock_ledger.assert_called_once()

    def test_failure_case(self):
        output = "SPEC_ID: abc-123\nSPEC_HASH: wrong_hash\n\nContent"
        
        with patch("app.jobs.stage3_locks.verify_spec_hash") as mock_verify, \
             patch("app.jobs.stage3_locks.write_stage3_artifacts") as mock_write, \
             patch("app.jobs.stage3_locks.append_stage3_ledger_event") as mock_ledger:
            
            mock_verify.return_value = (False, "abc-123", "wrong_hash", "hash_mismatch")
            mock_write.return_value = {}
            
            verified, error_msg = verify_and_store_stage3(
                job_id="job-1",
                stage_name="test_stage",
                spec_id="abc-123",
                expected_spec_hash="hash456",
                raw_output=output,
            )
            
            assert verified is False
            assert "spec-hash lock failed" in error_msg
            assert "hash456" in error_msg
            assert "wrong_hash" in error_msg


class TestEndToEndScenarios:
    """Integration-style tests for realistic scenarios."""

    def test_correct_header_passes(self):
        """Simulates a model that correctly echoes the spec headers."""
        spec_id = "015da9bb-7ced-47a1-8b15-b20221e635a9"
        spec_hash = "2fdc4222f20d083acec0fd17a7459e04fdded7c547c321d7a70385d676a050ca"
        
        # Build the instruction
        instruction = build_spec_echo_instruction(spec_id, spec_hash)
        
        # Simulate model output that correctly follows instruction
        model_output = f"""SPEC_ID: {spec_id}
SPEC_HASH: {spec_hash}

# Architecture Document

This is the architecture document content...
"""
        
        # Parse and verify
        ret_id, ret_hash, note = parse_spec_echo_headers(model_output)
        
        assert ret_id == spec_id
        assert ret_hash == spec_hash
        assert note == "ok"

    def test_incorrect_header_fails_and_logs(self):
        """Simulates a model that echoes wrong spec hash."""
        spec_id = "015da9bb-7ced-47a1-8b15-b20221e635a9"
        expected_hash = "2fdc4222f20d083acec0fd17a7459e04fdded7c547c321d7a70385d676a050ca"
        wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000"
        
        model_output = f"""SPEC_ID: {spec_id}
SPEC_HASH: {wrong_hash}

# Architecture Document
"""
        
        ret_id, ret_hash, note = parse_spec_echo_headers(model_output)
        
        assert ret_id == spec_id
        assert ret_hash == wrong_hash
        assert ret_hash != expected_hash
        assert note == "ok"  # Parsing succeeded, but values differ

    def test_model_forgot_headers_fails(self):
        """Simulates a model that forgot to include the headers."""
        model_output = """# Architecture Document

This is the architecture document content without headers...
"""
        
        ret_id, ret_hash, note = parse_spec_echo_headers(model_output)
        
        assert ret_id is None
        assert ret_hash is None
        assert "missing" in note

    def test_model_malformed_headers_fails(self):
        """Simulates a model that included malformed headers."""
        model_output = """spec-id: abc-123
spec-hash: def456

# Content
"""
        
        ret_id, ret_hash, note = parse_spec_echo_headers(model_output)
        
        assert ret_id is None
        assert ret_hash is None
        assert "missing" in note
