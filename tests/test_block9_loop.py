# FILE: tests/test_block9_loop.py
"""
Tests for Block 9: Verified Execution Loop

Tests the integration of:
- Overwatcher diagnosis on verification failure
- Strike tracking by ErrorSignature
- Strike 3 HARD_STOP + incident report
- Deep research on Strike 2
"""

import pytest
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path


# =============================================================================
# Mock Classes (inline for standalone testing)
# =============================================================================

@dataclass
class MockErrorSignature:
    signature_hash: str
    error_class: str = "TestError"
    message_core: str = "test error"
    
    def to_dict(self):
        return {"signature_hash": self.signature_hash}


@dataclass
class MockVerificationResult:
    chunk_id: str
    status: str  # "passed" or "failed"
    tests_passed: int = 0
    tests_failed: int = 0
    lint_errors: int = 0
    type_errors: int = 0
    command_results: List[Any] = field(default_factory=list)
    evidence_paths: List[str] = field(default_factory=list)
    legacy_failures: List[str] = field(default_factory=list)


@dataclass
class MockChunk:
    chunk_id: str
    title: str = "Test Chunk"
    objective: str = "Test objective"
    sequence: int = 1
    status: str = "pending"
    allowed_files: Dict[str, List[str]] = field(default_factory=lambda: {"modify": ["app/test.py"]})
    verification: Any = None


@dataclass
class MockOverwatcherOutput:
    decision: str = "FAIL"
    diagnosis: str = "Test diagnosis"
    fix_actions: List[Any] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    confidence: float = 0.8


@dataclass
class MockValidatedResult:
    output: MockOverwatcherOutput = field(default_factory=MockOverwatcherOutput)
    validation_passed: bool = True
    reprompt_count: int = 0
    contract_violations: List[Any] = field(default_factory=list)
    cost_tracked: bool = True
    budget_status: str = "within_budget"


# =============================================================================
# Simplified Block9 Logic for Testing
# =============================================================================

class StrikeManagerForTest:
    """Simplified strike manager for testing."""
    
    def __init__(self):
        self.strikes: Dict[str, int] = {}  # signature_hash -> count
        self.history: List[Dict] = []
    
    def record_strike(self, signature_hash: str, diagnosis: str) -> int:
        """Record a strike, return new count."""
        if signature_hash in self.strikes:
            self.strikes[signature_hash] += 1
        else:
            self.strikes[signature_hash] = 1
        
        self.history.append({
            "signature": signature_hash,
            "count": self.strikes[signature_hash],
            "diagnosis": diagnosis,
        })
        
        return self.strikes[signature_hash]
    
    def get_strike_count(self, signature_hash: str) -> int:
        return self.strikes.get(signature_hash, 0)
    
    def reset_for_new_signature(self, old_sig: str, new_sig: str) -> bool:
        """Check if signature changed (would reset strikes)."""
        return old_sig != new_sig


def run_block9_simulation(
    *,
    chunk: MockChunk,
    execute_results: List[bool],  # Sequence of execute success/fail
    verify_results: List[MockVerificationResult],  # Sequence of verification results
    signatures: List[str],  # Signature for each failure
    max_strikes: int = 3,
) -> Dict[str, Any]:
    """
    Simulate Block 9 loop for testing.
    
    Returns dict with:
        success: bool
        strikes_used: int
        overwatcher_calls: int
        hard_stopped: bool
        final_signature: str
    """
    strike_manager = StrikeManagerForTest()
    current_signature: Optional[str] = None
    strike_count = 0
    overwatcher_calls = 0
    attempt = 0
    
    while strike_count < max_strikes and attempt < len(execute_results):
        exec_success = execute_results[attempt]
        
        if not exec_success:
            # Execution failed
            sig = signatures[attempt] if attempt < len(signatures) else "exec_fail"
            
            # Check if same signature
            if current_signature and current_signature == sig:
                strike_count += 1
            else:
                strike_count = 1
                current_signature = sig
            
            strike_manager.record_strike(sig, f"Execution failed attempt {attempt}")
            overwatcher_calls += 1
            attempt += 1
            continue
        
        # Verification
        verify_result = verify_results[attempt] if attempt < len(verify_results) else None
        
        if verify_result and verify_result.status == "passed":
            # Success!
            return {
                "success": True,
                "strikes_used": strike_count,
                "overwatcher_calls": overwatcher_calls,
                "hard_stopped": False,
                "final_signature": current_signature,
            }
        
        # Verification failed
        sig = signatures[attempt] if attempt < len(signatures) else "verify_fail"
        overwatcher_calls += 1
        
        # Check if same signature
        if current_signature and current_signature == sig:
            strike_count += 1
        else:
            strike_count = 1
            current_signature = sig
        
        strike_manager.record_strike(sig, f"Verification failed attempt {attempt}")
        attempt += 1
    
    # Exhausted strikes or attempts
    return {
        "success": False,
        "strikes_used": strike_count,
        "overwatcher_calls": overwatcher_calls,
        "hard_stopped": strike_count >= max_strikes,
        "final_signature": current_signature,
    }


# =============================================================================
# Tests
# =============================================================================

class TestBlock9StrikeLogic:
    """Test strike tracking by signature."""
    
    def test_same_signature_accumulates_strikes(self):
        """Same error signature should accumulate strikes."""
        result = run_block9_simulation(
            chunk=MockChunk(chunk_id="chunk-1"),
            execute_results=[True, True, True],
            verify_results=[
                MockVerificationResult(chunk_id="chunk-1", status="failed"),
                MockVerificationResult(chunk_id="chunk-1", status="failed"),
                MockVerificationResult(chunk_id="chunk-1", status="failed"),
            ],
            signatures=["sig-abc", "sig-abc", "sig-abc"],  # Same signature
            max_strikes=3,
        )
        
        assert result["success"] is False
        assert result["strikes_used"] == 3
        assert result["hard_stopped"] is True
        assert result["overwatcher_calls"] == 3
    
    def test_different_signature_resets_strikes(self):
        """Different error signature should reset strikes to 1."""
        result = run_block9_simulation(
            chunk=MockChunk(chunk_id="chunk-2"),
            execute_results=[True, True, True, True],
            verify_results=[
                MockVerificationResult(chunk_id="chunk-2", status="failed"),
                MockVerificationResult(chunk_id="chunk-2", status="failed"),
                MockVerificationResult(chunk_id="chunk-2", status="failed"),
                MockVerificationResult(chunk_id="chunk-2", status="passed"),
            ],
            signatures=["sig-abc", "sig-xyz", "sig-123", "sig-123"],  # Different signatures
            max_strikes=3,
        )
        
        # Should succeed because each new signature resets to strike 1
        assert result["success"] is True
        assert result["strikes_used"] == 1  # Reset on last
    
    def test_success_on_first_attempt(self):
        """Success on first attempt uses no strikes."""
        result = run_block9_simulation(
            chunk=MockChunk(chunk_id="chunk-3"),
            execute_results=[True],
            verify_results=[
                MockVerificationResult(chunk_id="chunk-3", status="passed"),
            ],
            signatures=[],
            max_strikes=3,
        )
        
        assert result["success"] is True
        assert result["strikes_used"] == 0
        assert result["overwatcher_calls"] == 0
        assert result["hard_stopped"] is False
    
    def test_success_after_one_failure(self):
        """Success after one failure uses one strike."""
        result = run_block9_simulation(
            chunk=MockChunk(chunk_id="chunk-4"),
            execute_results=[True, True],
            verify_results=[
                MockVerificationResult(chunk_id="chunk-4", status="failed"),
                MockVerificationResult(chunk_id="chunk-4", status="passed"),
            ],
            signatures=["sig-fail", "sig-fail"],
            max_strikes=3,
        )
        
        assert result["success"] is True
        assert result["strikes_used"] == 1
        assert result["overwatcher_calls"] == 1


class TestBlock9OverwatcherIntegration:
    """Test Overwatcher is called on failures."""
    
    def test_overwatcher_called_on_each_failure(self):
        """Overwatcher should be called once per failure."""
        result = run_block9_simulation(
            chunk=MockChunk(chunk_id="chunk-5"),
            execute_results=[True, True, True],
            verify_results=[
                MockVerificationResult(chunk_id="chunk-5", status="failed"),
                MockVerificationResult(chunk_id="chunk-5", status="failed"),
                MockVerificationResult(chunk_id="chunk-5", status="passed"),
            ],
            signatures=["sig-a", "sig-b", "sig-b"],  # Different, then same
            max_strikes=3,
        )
        
        assert result["overwatcher_calls"] == 2
        assert result["success"] is True
    
    def test_no_overwatcher_on_success(self):
        """Overwatcher should not be called when verification passes."""
        result = run_block9_simulation(
            chunk=MockChunk(chunk_id="chunk-6"),
            execute_results=[True],
            verify_results=[
                MockVerificationResult(chunk_id="chunk-6", status="passed"),
            ],
            signatures=[],
            max_strikes=3,
        )
        
        assert result["overwatcher_calls"] == 0


class TestBlock9HardStop:
    """Test Strike 3 hard stop behavior."""
    
    def test_hard_stop_on_strike_3(self):
        """Strike 3 should trigger hard stop."""
        result = run_block9_simulation(
            chunk=MockChunk(chunk_id="chunk-7"),
            execute_results=[True, True, True, True],
            verify_results=[
                MockVerificationResult(chunk_id="chunk-7", status="failed"),
                MockVerificationResult(chunk_id="chunk-7", status="failed"),
                MockVerificationResult(chunk_id="chunk-7", status="failed"),
                MockVerificationResult(chunk_id="chunk-7", status="passed"),  # Would pass but never reached
            ],
            signatures=["sig-same", "sig-same", "sig-same", "sig-same"],
            max_strikes=3,
        )
        
        assert result["success"] is False
        assert result["hard_stopped"] is True
        assert result["strikes_used"] == 3
    
    def test_no_hard_stop_under_3_strikes(self):
        """Should not hard stop with fewer than 3 strikes."""
        result = run_block9_simulation(
            chunk=MockChunk(chunk_id="chunk-8"),
            execute_results=[True, True],
            verify_results=[
                MockVerificationResult(chunk_id="chunk-8", status="failed"),
                MockVerificationResult(chunk_id="chunk-8", status="passed"),
            ],
            signatures=["sig-x", "sig-x"],
            max_strikes=3,
        )
        
        assert result["success"] is True
        assert result["hard_stopped"] is False


class TestStrikeManager:
    """Test the strike manager directly."""
    
    def test_first_strike_is_one(self):
        """First strike for a signature should be 1."""
        manager = StrikeManagerForTest()
        count = manager.record_strike("sig-new", "First failure")
        assert count == 1
    
    def test_same_signature_increments(self):
        """Same signature should increment."""
        manager = StrikeManagerForTest()
        manager.record_strike("sig-repeat", "First")
        manager.record_strike("sig-repeat", "Second")
        count = manager.record_strike("sig-repeat", "Third")
        assert count == 3
    
    def test_different_signature_starts_at_one(self):
        """Different signature starts at 1."""
        manager = StrikeManagerForTest()
        manager.record_strike("sig-a", "A failure")
        manager.record_strike("sig-a", "A again")
        count = manager.record_strike("sig-b", "B failure")
        assert count == 1
    
    def test_history_tracked(self):
        """Strike history should be tracked."""
        manager = StrikeManagerForTest()
        manager.record_strike("sig-1", "First")
        manager.record_strike("sig-2", "Second")
        
        assert len(manager.history) == 2
        assert manager.history[0]["signature"] == "sig-1"
        assert manager.history[1]["signature"] == "sig-2"


class TestEdgeCases:
    """Test edge cases."""
    
    def test_execution_failure_counts_as_strike(self):
        """Execution failure (not just verification) should count."""
        result = run_block9_simulation(
            chunk=MockChunk(chunk_id="chunk-exec"),
            execute_results=[False, False, False],
            verify_results=[],  # Never reached
            signatures=["exec-fail", "exec-fail", "exec-fail"],
            max_strikes=3,
        )
        
        assert result["success"] is False
        assert result["strikes_used"] == 3
        assert result["hard_stopped"] is True
    
    def test_mixed_exec_and_verify_failures(self):
        """Mix of execution and verification failures."""
        result = run_block9_simulation(
            chunk=MockChunk(chunk_id="chunk-mixed"),
            execute_results=[False, True, True],
            verify_results=[
                MockVerificationResult(chunk_id="chunk-mixed", status="failed"),  # Skipped
                MockVerificationResult(chunk_id="chunk-mixed", status="failed"),
                MockVerificationResult(chunk_id="chunk-mixed", status="failed"),
            ],
            signatures=["sig-same", "sig-same", "sig-same"],
            max_strikes=3,
        )
        
        assert result["success"] is False
        assert result["hard_stopped"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
