# FILE: app/orchestrator/refactor_state.py
"""
Refactor State — Persistent tracking of refactor attempts across runs.

Tracks per-file strike counts, failure reasons, and flags files that need
pipeline-level (LLM) decomposition after exhausting AST-based extraction.

State file: D:\\Orb\\data\\refactor_state.json

Three-strikes rule:
  - Strike 1-2: File will be retried on next refactor run
  - Strike 3:   File marked as 'needs_pipeline' and skipped by scanner
  - Reset:      Strikes reset if the file's size changes (someone edited it)

Usage:
    from app.orchestrator.refactor_state import RefactorState

    state = RefactorState.load()
    if state.should_skip("path/to/file.py"):
        ...  # Skip this file
    state.record_failure("path/to/file.py", 25.3, "Boot failed after extraction")
    state.record_success("path/to/file.py", 25.3, 18.1)
    state.save()
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

STATE_FILE = os.path.join(r"D:\Orb", "data", "refactor_state.json")
MAX_STRIKES = 3


@dataclass
class FileRecord:
    """Tracking record for a single file."""
    path: str
    strikes: int = 0
    last_size_kb: float = 0.0
    last_failure: str = ""
    last_attempt: str = ""
    needs_pipeline: bool = False
    successes: int = 0
    total_kb_reduced: float = 0.0

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "strikes": self.strikes,
            "last_size_kb": self.last_size_kb,
            "last_failure": self.last_failure,
            "last_attempt": self.last_attempt,
            "needs_pipeline": self.needs_pipeline,
            "successes": self.successes,
            "total_kb_reduced": self.total_kb_reduced,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FileRecord":
        return cls(
            path=d.get("path", ""),
            strikes=d.get("strikes", 0),
            last_size_kb=d.get("last_size_kb", 0.0),
            last_failure=d.get("last_failure", ""),
            last_attempt=d.get("last_attempt", ""),
            needs_pipeline=d.get("needs_pipeline", False),
            successes=d.get("successes", 0),
            total_kb_reduced=d.get("total_kb_reduced", 0.0),
        )


class RefactorState:
    """Persistent refactor state across runs."""

    def __init__(self) -> None:
        self.files: Dict[str, FileRecord] = {}
        self.total_runs: int = 0
        self.total_successes: int = 0
        self.total_failures: int = 0
        self.last_run: str = ""

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    @classmethod
    def load(cls) -> "RefactorState":
        """Load state from disk, or return fresh state if missing."""
        state = cls()
        if not os.path.exists(STATE_FILE):
            return state
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            state.total_runs = data.get("total_runs", 0)
            state.total_successes = data.get("total_successes", 0)
            state.total_failures = data.get("total_failures", 0)
            state.last_run = data.get("last_run", "")
            for fd in data.get("files", []):
                rec = FileRecord.from_dict(fd)
                state.files[rec.path] = rec
            logger.info(
                f"[refactor_state] Loaded state: {len(state.files)} tracked files, "
                f"{sum(1 for r in state.files.values() if r.needs_pipeline)} need pipeline"
            )
        except Exception as e:
            logger.warning(f"[refactor_state] Failed to load state: {e}")
        return state

    def save(self) -> None:
        """Persist state to disk."""
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        data = {
            "total_runs": self.total_runs,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "last_run": self.last_run,
            "files": [rec.to_dict() for rec in self.files.values()],
        }
        try:
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"[refactor_state] Failed to save state: {e}")

    # -----------------------------------------------------------------
    # Query
    # -----------------------------------------------------------------

    def should_skip(self, path: str, current_size_kb: float = 0.0) -> bool:
        """
        Should the scanner skip this file?

        Returns True if the file has 3+ strikes AND its size hasn't changed
        (size change = someone edited it, so reset strikes and retry).
        """
        rec = self.files.get(path)
        if rec is None:
            return False

        # If the file size changed significantly, reset strikes — someone worked on it
        if current_size_kb > 0 and abs(current_size_kb - rec.last_size_kb) > 0.5:
            logger.info(
                f"[refactor_state] Size changed for {os.path.basename(path)}: "
                f"{rec.last_size_kb:.1f}KB → {current_size_kb:.1f}KB — resetting strikes"
            )
            rec.strikes = 0
            rec.needs_pipeline = False
            return False

        return rec.needs_pipeline or rec.strikes >= MAX_STRIKES

    def get_skip_reason(self, path: str) -> str:
        """Human-readable reason why a file is skipped."""
        rec = self.files.get(path)
        if rec is None:
            return ""
        if rec.needs_pipeline:
            return f"Needs pipeline decomposition ({rec.strikes} strikes: {rec.last_failure})"
        if rec.strikes >= MAX_STRIKES:
            return f"{rec.strikes} strikes: {rec.last_failure}"
        return ""

    @property
    def pipeline_queue(self) -> List[FileRecord]:
        """Files that need pipeline-level decomposition."""
        return [r for r in self.files.values() if r.needs_pipeline]

    @property
    def stats(self) -> dict:
        """Summary statistics."""
        tracked = len(self.files)
        pipeline = sum(1 for r in self.files.values() if r.needs_pipeline)
        struck_out = sum(1 for r in self.files.values() if r.strikes >= MAX_STRIKES)
        return {
            "tracked_files": tracked,
            "needs_pipeline": pipeline,
            "struck_out": struck_out,
            "total_runs": self.total_runs,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
        }

    # -----------------------------------------------------------------
    # Recording
    # -----------------------------------------------------------------

    def record_failure(self, path: str, size_kb: float, reason: str) -> None:
        """Record a failed extraction attempt."""
        rec = self.files.get(path)
        if rec is None:
            rec = FileRecord(path=path)
            self.files[path] = rec

        rec.strikes += 1
        rec.last_size_kb = size_kb
        rec.last_failure = reason[:200]
        rec.last_attempt = datetime.utcnow().isoformat()
        self.total_failures += 1

        if rec.strikes >= MAX_STRIKES:
            rec.needs_pipeline = True
            logger.info(
                f"[refactor_state] ⚠ {os.path.basename(path)} struck out "
                f"({rec.strikes} strikes) — flagged for pipeline decomposition"
            )

    def record_success(self, path: str, old_kb: float, new_kb: float) -> None:
        """Record a successful extraction."""
        rec = self.files.get(path)
        if rec is None:
            rec = FileRecord(path=path)
            self.files[path] = rec

        rec.successes += 1
        rec.last_size_kb = new_kb
        rec.total_kb_reduced += (old_kb - new_kb)
        rec.last_attempt = datetime.utcnow().isoformat()
        # Success resets strikes (the file is cooperating)
        rec.strikes = 0
        rec.needs_pipeline = False
        self.total_successes += 1

    def mark_run_start(self) -> None:
        """Record the start of a refactor run."""
        self.total_runs += 1
        self.last_run = datetime.utcnow().isoformat()

    def reset_file(self, path: str) -> None:
        """Manually reset a file's strikes (e.g. after pipeline fixes it)."""
        if path in self.files:
            self.files[path].strikes = 0
            self.files[path].needs_pipeline = False

    def reset_all(self) -> None:
        """Reset all state (nuclear option)."""
        self.files.clear()
        logger.info("[refactor_state] All state reset")
