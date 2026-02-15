# FILE: app/orchestrator/strike_tracker.py
"""
Strike Tracker: Universal error-resolution logic for ASTRA pipeline.

Implements the three-strike rule across all checkout and verification stages:

  Strike 1: Error occurs → attempt fix → if same error returns, escalate.
  Strike 2: Same error → MUST take a different approach → if still same, escalate.
  Strike 3: Same error a third time → hard stop, write review for human.

If the error CHANGES (different error signature), strikes reset to 1 for the
new error. This prevents perpetual loops on the same problem while allowing
the system to work through cascading issues.

Every strike attempt is recorded with full context for downstream RAG ingestion:
- What error was seen
- What fix strategy was tried
- Whether it resolved, changed the error, or failed
- Duration and metadata

This record becomes training data for the RAG memory system, allowing future
runs to skip failed strategies and go straight to what worked before.

v1.0 (2026-02-15): Initial implementation.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class StrikeVerdict(Enum):
    """What the StrikeTracker tells the caller to do."""
    PROCEED = "proceed"              # Strike 1: try to fix it
    CHANGE_APPROACH = "change_approach"  # Strike 2: same error, different strategy needed
    HARD_STOP = "hard_stop"          # Strike 3: give up, write review


class FixOutcome(Enum):
    """Result of a fix attempt."""
    RESOLVED = "resolved"            # Error gone entirely
    NEW_ERROR = "new_error"          # Different error appeared (strikes reset)
    SAME_ERROR = "same_error"        # Exact same error came back
    FIX_FAILED = "fix_failed"        # Couldn't even generate/apply a fix


@dataclass
class StrikeAttempt:
    """Record of a single fix attempt — this is the RAG training data."""
    strike_number: int
    error_signature: str
    error_detail: str
    fix_strategy: str
    fix_description: str
    outcome: FixOutcome
    new_error_signature: Optional[str] = None
    new_error_detail: Optional[str] = None
    duration_ms: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrikeRecord:
    """Complete record for one check/stage — all attempts across all errors."""
    check_name: str
    stage: str
    job_id: str
    attempts: List[StrikeAttempt] = field(default_factory=list)
    final_verdict: Optional[str] = None  # "resolved" | "hard_stop"
    total_duration_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for RAG storage."""
        return {
            "check_name": self.check_name,
            "stage": self.stage,
            "job_id": self.job_id,
            "attempts": [
                {
                    "strike": a.strike_number,
                    "error_sig": a.error_signature,
                    "error_detail": a.error_detail[:500],
                    "strategy": a.fix_strategy,
                    "description": a.fix_description,
                    "outcome": a.outcome.value,
                    "new_error_sig": a.new_error_signature,
                    "duration_ms": a.duration_ms,
                    "timestamp": a.timestamp,
                    "metadata": a.metadata,
                }
                for a in self.attempts
            ],
            "final_verdict": self.final_verdict,
            "total_duration_ms": self.total_duration_ms,
        }


def _error_signature(error_text: str) -> str:
    """
    Compute a stable signature for an error so we can detect "same error."

    Strips variable parts (line numbers, hex addresses, timestamps) and
    hashes the normalized text. Two errors with the same root cause but
    different line numbers will get the same signature.
    """
    import re

    if not error_text:
        return "empty_error"

    # Normalize: strip line numbers, hex addresses, timestamps, file paths
    normalized = error_text.strip()

    # Remove line numbers like "line 42" or ":42:"
    normalized = re.sub(r'\bline \d+\b', 'line N', normalized)
    normalized = re.sub(r':\d+:', ':N:', normalized)

    # Remove hex addresses like 0x7fff1234
    normalized = re.sub(r'0x[0-9a-fA-F]+', '0xADDR', normalized)

    # Remove timestamps
    normalized = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', 'TIMESTAMP', normalized)

    # Remove Windows paths but keep the filename
    normalized = re.sub(r'[A-Z]:\\[\w\\.-]+\\', 'PATH\\', normalized)

    # Keep only the last line (the actual error message) for signature
    lines = [l.strip() for l in normalized.splitlines() if l.strip()]
    # Find the actual error line (usually starts with ErrorType:)
    error_line = normalized
    for line in reversed(lines):
        if ':' in line and not line.startswith('File') and not line.startswith('PATH'):
            error_line = line
            break

    return hashlib.sha256(error_line.encode()).hexdigest()[:16]


class StrikeTracker:
    """
    Tracks errors and strikes for a single check within a pipeline stage.

    Usage:
        tracker = StrikeTracker("boot_test", "phase_checkout", "sg-abc123")

        while True:
            error = run_check()
            if not error:
                tracker.record_resolution()
                break

            verdict = tracker.report_error(error)

            if verdict == StrikeVerdict.HARD_STOP:
                tracker.record_hard_stop(error)
                break

            # Get what strategy to use
            strategy = "reconcile_import" if verdict == StrikeVerdict.PROCEED else "comment_out_import"

            fix_result = apply_fix(strategy)
            tracker.record_attempt(
                error_detail=error,
                fix_strategy=strategy,
                fix_description="Reconciled import path from X to Y",
                outcome=FixOutcome.RESOLVED if fix_result else FixOutcome.SAME_ERROR,
            )

        record = tracker.get_record()  # → StrikeRecord for RAG
    """

    MAX_STRIKES = 3

    def __init__(self, check_name: str, stage: str, job_id: str) -> None:
        self.check_name = check_name
        self.stage = stage
        self.job_id = job_id

        self._current_error_sig: Optional[str] = None
        self._current_error_text: Optional[str] = None
        self._strike_count: int = 0
        self._attempts: List[StrikeAttempt] = []
        self._start_time: float = time.time()
        self._resolved: bool = False
        self._strategies_tried: List[str] = []  # For current error signature

    def report_error(self, error_text: str) -> StrikeVerdict:
        """
        Report an error. Returns what the caller should do.

        If error is same as last time → increment strikes.
        If error is different → reset strikes to 1 for new error.
        """
        sig = _error_signature(error_text)

        if sig == self._current_error_sig:
            # Same error
            self._strike_count += 1
        else:
            # New/different error — reset
            self._current_error_sig = sig
            self._current_error_text = error_text
            self._strike_count = 1
            self._strategies_tried = []

        if self._strike_count >= self.MAX_STRIKES:
            return StrikeVerdict.HARD_STOP
        elif self._strike_count == 2:
            return StrikeVerdict.CHANGE_APPROACH
        else:
            return StrikeVerdict.PROCEED

    def record_attempt(
        self,
        error_detail: str,
        fix_strategy: str,
        fix_description: str,
        outcome: FixOutcome,
        new_error_detail: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: int = 0,
    ) -> None:
        """Record a fix attempt for RAG training data."""
        attempt = StrikeAttempt(
            strike_number=self._strike_count,
            error_signature=self._current_error_sig or "",
            error_detail=error_detail,
            fix_strategy=fix_strategy,
            fix_description=fix_description,
            outcome=outcome,
            new_error_signature=(
                _error_signature(new_error_detail) if new_error_detail else None
            ),
            new_error_detail=new_error_detail,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        self._attempts.append(attempt)
        self._strategies_tried.append(fix_strategy)

    def get_strategies_tried(self) -> List[str]:
        """
        Return strategies already tried for the current error.
        Used by strike 2 logic to pick a DIFFERENT approach.
        """
        return list(self._strategies_tried)

    def record_resolution(self) -> None:
        """Mark the check as resolved (no more errors)."""
        self._resolved = True

    def record_hard_stop(self, final_error: str) -> None:
        """Mark the check as hard-stopped after max strikes."""
        self._resolved = False
        self._current_error_text = final_error

    @property
    def strike_count(self) -> int:
        return self._strike_count

    @property
    def current_error(self) -> Optional[str]:
        return self._current_error_text

    @property
    def is_resolved(self) -> bool:
        return self._resolved

    def get_record(self) -> StrikeRecord:
        """Get the complete record for RAG storage."""
        elapsed = int((time.time() - self._start_time) * 1000)
        return StrikeRecord(
            check_name=self.check_name,
            stage=self.stage,
            job_id=self.job_id,
            attempts=list(self._attempts),
            final_verdict="resolved" if self._resolved else "hard_stop",
            total_duration_ms=elapsed,
        )
