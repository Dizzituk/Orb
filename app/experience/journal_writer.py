# FILE: app/experience/journal_writer.py
# Purpose: Build Journal writer — append-only NDJSON journal per job.
# Called-by: app.experience, app.experience.context, app.experience.distillation
# Depends-on: app.experience.schemas
# Last-renovated: 2026-06-11
"""
Build Journal writer — append-only NDJSON journal per job.

Each pipeline job produces a journal at: jobs/{job_id}/journal.ndjson
One JSON object per line, one line per event. Append-only, never modified.

Usage:
    # Get/create writer for a job
    writer = get_journal_writer(job_id, job_dir)

    # Emit an entry (fire-and-forget, never raises)
    writer.emit(BuildJournalEntry(
        job_id=job_id,
        stage="critical_pipeline",
        event_type="architecture_decision",
        description="Chose package structure over monolith",
    ))

    # Or use the convenience function (auto-creates writer):
    emit_journal_entry(
        job_id=job_id,
        job_dir=job_dir,
        stage="implementer",
        event_type="implementation_strike",
        severity="error",
        description="SyntaxError in generated code",
        error_signature="abc123",
    )

    # Hook into existing add_trace() calls:
    emit_from_trace(
        job_id=job_id,
        job_dir=job_dir,
        trace_stage="BOOT_CHECK_FAIL",
        trace_status="strike_1",
        trace_details={"error": "ImportError: no module named app.utils"},
    )

Design:
- Non-blocking: write failures are logged, never raised
- Thread-safe: uses a lock per writer instance
- Lazy: journal file created on first write
- Singleton per job_id: get_journal_writer() returns same instance
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .schemas import (
    BuildJournalEntry,
    EventSeverity,
    TRACE_STAGE_TO_EVENT_TYPE,
    TRACE_STATUS_TO_SEVERITY,
)

logger = logging.getLogger(__name__)

# Singleton registry: job_id → JournalWriter
_writers: Dict[str, "JournalWriter"] = {}
_registry_lock = threading.Lock()

# Default job directory root
JOBS_ROOT = os.environ.get("ASTRA_JOBS_ROOT", os.path.join("D:\\Orb", "jobs", "jobs"))


class JournalWriter:
    """
    Append-only NDJSON journal writer for a single job.

    Thread-safe. Write failures are caught and logged — the journal
    must never crash the pipeline.
    """

    def __init__(self, job_id: str, job_dir: str):
        self.job_id = job_id
        self.job_dir = job_dir
        self.journal_path = os.path.join(job_dir, "journal.ndjson")
        self._lock = threading.Lock()
        self._entry_count = 0

    def emit(self, entry: BuildJournalEntry) -> bool:
        """
        Append a journal entry. Returns True on success, False on failure.
        Never raises.
        """
        try:
            line = entry.to_json() + "\n"
            with self._lock:
                # Ensure directory exists (lazy creation)
                os.makedirs(self.job_dir, exist_ok=True)
                with open(self.journal_path, "a", encoding="utf-8") as f:
                    f.write(line)
                self._entry_count += 1
            return True
        except Exception as e:
            logger.warning(
                f"[journal] Failed to write entry for job {self.job_id}: {e}"
            )
            return False

    @property
    def entry_count(self) -> int:
        return self._entry_count

    def read_all(self) -> list[BuildJournalEntry]:
        """Read all journal entries. Used by distillation."""
        entries = []
        try:
            if not os.path.exists(self.journal_path):
                return entries
            with open(self.journal_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(BuildJournalEntry.from_json(line))
                        except Exception as e:
                            logger.warning(f"[journal] Bad line in {self.journal_path}: {e}")
        except Exception as e:
            logger.warning(f"[journal] Failed to read journal for {self.job_id}: {e}")
        return entries


def get_journal_writer(job_id: str, job_dir: Optional[str] = None) -> JournalWriter:
    """
    Get or create a JournalWriter for a job. Singleton per job_id.

    Args:
        job_id: The job identifier (e.g. "sg-f0f5f67e")
        job_dir: Job directory path. If None, derived from JOBS_ROOT/job_id.
    """
    with _registry_lock:
        if job_id in _writers:
            return _writers[job_id]

        if not job_dir:
            job_dir = os.path.join(JOBS_ROOT, job_id)

        writer = JournalWriter(job_id, job_dir)
        _writers[job_id] = writer
        return writer


def close_journal(job_id: str) -> None:
    """Remove a writer from the registry (e.g. when a job completes)."""
    with _registry_lock:
        _writers.pop(job_id, None)


def emit_journal_entry(
    job_id: str,
    job_dir: Optional[str] = None,
    *,
    stage: str,
    event_type: str,
    severity: str = "info",
    description: str = "",
    root_cause: str = "",
    resolution: str = "",
    error_signature: str = "",
    file_scope: str = "",
    segment_id: str = "",
    job_type: str = "",
    language: str = "",
    strategy_used: str = "",
    strike_number: int = 0,
    duration_ms: int = 0,
    details: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Convenience function: create and emit a journal entry in one call.
    Fire-and-forget — never raises.
    """
    try:
        writer = get_journal_writer(job_id, job_dir)
        entry = BuildJournalEntry(
            job_id=job_id,
            stage=stage,
            event_type=event_type,
            severity=severity,
            description=description,
            root_cause=root_cause,
            resolution=resolution,
            error_signature=error_signature,
            file_scope=file_scope,
            segment_id=segment_id,
            job_type=job_type,
            language=language,
            strategy_used=strategy_used,
            strike_number=strike_number,
            duration_ms=duration_ms,
            details=details or {},
        )
        return writer.emit(entry)
    except Exception as e:
        logger.warning(f"[journal] emit_journal_entry failed: {e}")
        return False


def emit_from_trace(
    job_id: str,
    job_dir: Optional[str] = None,
    *,
    trace_stage: str,
    trace_status: str,
    trace_details: Optional[Dict[str, Any]] = None,
    segment_id: str = "",
    file_scope: str = "",
    job_type: str = "",
) -> bool:
    """
    Convert an existing add_trace() call into a journal entry.

    This is the bridge between the current trace system and the journal.
    Maps trace_stage → event_type and trace_status → severity using
    the lookup tables in schemas.py.

    Usage in architecture_executor.py:
        # Existing:
        add_trace("BOOT_CHECK_FAIL", "strike_1", {"error": "ImportError..."})

        # Add alongside:
        emit_from_trace(
            job_id=job_id, job_dir=job_dir,
            trace_stage="BOOT_CHECK_FAIL",
            trace_status="strike_1",
            trace_details={"error": "ImportError..."},
        )
    """
    try:
        details = trace_details or {}

        # Map trace stage to event type
        event_type = TRACE_STAGE_TO_EVENT_TYPE.get(
            trace_stage, "error_generic"
        )

        # Map trace status to severity
        severity = TRACE_STATUS_TO_SEVERITY.get(
            trace_status, EventSeverity.INFO
        )

        # Extract useful fields from details
        description = details.get("error", "") or details.get("reason", "") or ""
        error_sig = details.get("error_sig", "") or details.get("error_signature", "")
        file_path = details.get("file_path", "") or details.get("file", "") or file_scope
        strategy = details.get("strategy", "") or details.get("approach", "")
        resolution = details.get("resolution", "") or details.get("fix", "")
        root_cause = details.get("root_cause", "") or details.get("cause", "")
        strike = details.get("strike", 0)
        duration = details.get("duration_ms", 0)

        # If description is empty, build one from stage + status
        if not description:
            description = f"{trace_stage}: {trace_status}"

        return emit_journal_entry(
            job_id=job_id,
            job_dir=job_dir,
            stage=_infer_pipeline_stage(trace_stage),
            event_type=event_type,
            severity=severity,
            description=description,
            root_cause=root_cause,
            resolution=resolution,
            error_signature=error_sig,
            file_scope=file_path,
            segment_id=segment_id,
            job_type=job_type,
            strategy_used=strategy,
            strike_number=strike,
            duration_ms=duration,
            details=details,
        )
    except Exception as e:
        logger.warning(f"[journal] emit_from_trace failed: {e}")
        return False


def _infer_pipeline_stage(trace_stage: str) -> str:
    """Infer the pipeline stage name from a trace stage string."""
    ts = trace_stage.upper()

    if ts.startswith("BOOT_") or ts.startswith("BUILD_"):
        return "phase_checkout"
    if ts.startswith("FILE_TASK_") or ts.startswith("VERBATIM") or ts.startswith("MODIFY_EDIT"):
        return "implementer"
    if ts.startswith("EDIT_"):
        return "implementer"
    if ts.startswith("ARCHITECTURE_"):
        return "architecture_executor"
    if ts.startswith("SANDBOX_"):
        return "sandbox"
    if ts.startswith("OVERWATCHER_"):
        return "overwatcher"
    if ts.startswith("IMPLEMENTER_"):
        return "implementer"
    if ts.startswith("VERIFICATION_"):
        return "verification"
    if ts.startswith("SPEC_"):
        return "spec_gate"
    if ts.startswith("POT_"):
        return "pot_executor"
    if ts.startswith("JOB_CHECK"):
        return "phase_checkout"
    if "QUARANTINE" in ts:
        return "implementer"
    if "MODULE_SHADOW" in ts:
        return "implementer"
    if "AUTO_INIT" in ts:
        return "implementer"
    if "TWO_PASS" in ts:
        return "implementer"

    return "unknown"
