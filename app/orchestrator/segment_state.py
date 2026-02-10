# FILE: app/orchestrator/segment_state.py
"""
Segment execution state tracking and persistence.

Provides SegmentState and JobState dataclass models for tracking segment
execution progress, plus atomic load/save operations for crash recovery.

Design:
    - Dataclass-based (matching segment_schemas.py conventions, NOT Pydantic)
    - JSON round-trip via to_dict()/from_dict()
    - Atomic writes (write to temp file, then rename) to prevent corruption
    - Uses spec_gate_persistence path patterns for consistency

Phase 2 of Pipeline Segmentation.

v1.0 (2026-02-08): Initial implementation
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SEGMENT_STATE_BUILD_ID = "2026-02-08-v1.0-initial"
print(f"[SEGMENT_STATE_LOADED] BUILD_ID={SEGMENT_STATE_BUILD_ID}")

# Re-use the SegmentStatus enum from Phase 1 schemas
from app.pot_spec.grounded.segment_schemas import SegmentStatus, SegmentManifest

# Re-use path construction from spec_gate_persistence
try:
    from app.pot_spec.spec_gate_persistence import (
        artifact_root as _artifact_root,
        job_dir as _job_dir,
    )
    _PERSISTENCE_AVAILABLE = True
except ImportError:
    _PERSISTENCE_AVAILABLE = False

    def _artifact_root() -> str:
        root = os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs")
        return os.path.abspath(root)

    def _job_dir(job_root: str, job_id: str) -> str:
        return os.path.join(job_root, "jobs", job_id)


STATE_FILENAME = "state.json"


# =============================================================================
# STATE MODELS
# =============================================================================


@dataclass
class SegmentState:
    """Execution state of a single segment."""

    segment_id: str
    status: str = SegmentStatus.PENDING.value  # Store as string for JSON compat
    started_at: Optional[str] = None           # ISO timestamp
    completed_at: Optional[str] = None         # ISO timestamp
    error: Optional[str] = None                # Error message if FAILED
    output_files: List[str] = field(default_factory=list)  # Files created/modified
    evidence_provided_to: List[str] = field(default_factory=list)  # Downstream recipients

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "output_files": self.output_files,
            "evidence_provided_to": self.evidence_provided_to,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SegmentState":
        return cls(
            segment_id=data.get("segment_id", ""),
            status=data.get("status", SegmentStatus.PENDING.value),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error=data.get("error"),
            output_files=data.get("output_files", []),
            evidence_provided_to=data.get("evidence_provided_to", []),
        )


@dataclass
class JobState:
    """Execution state of a segmented job (all segments)."""

    job_id: str
    manifest_version: str = "1.0"
    total_segments: int = 0
    segments: Dict[str, SegmentState] = field(default_factory=dict)
    started_at: str = ""
    last_updated: str = ""
    overall_status: str = "running"  # running | complete | partial | failed
    integration_check: Optional[Dict[str, Any]] = None  # Phase 3: cross-segment integration check result

    def __post_init__(self):
        if not self.started_at:
            self.started_at = _now_iso()
        if not self.last_updated:
            self.last_updated = _now_iso()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "manifest_version": self.manifest_version,
            "total_segments": self.total_segments,
            "segments": {
                sid: seg.to_dict() for sid, seg in self.segments.items()
            },
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "overall_status": self.overall_status,
            "integration_check": self.integration_check,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobState":
        segments_raw = data.get("segments", {})
        segments = {
            sid: SegmentState.from_dict(seg_data)
            for sid, seg_data in segments_raw.items()
        }
        return cls(
            job_id=data.get("job_id", ""),
            manifest_version=data.get("manifest_version", "1.0"),
            total_segments=data.get("total_segments", 0),
            segments=segments,
            started_at=data.get("started_at", ""),
            last_updated=data.get("last_updated", ""),
            overall_status=data.get("overall_status", "running"),
            integration_check=data.get("integration_check"),
        )

    # --- Convenience accessors ---

    def count_by_status(self) -> Dict[str, int]:
        """Count segments by status."""
        counts: Dict[str, int] = {}
        for seg in self.segments.values():
            counts[seg.status] = counts.get(seg.status, 0) + 1
        return counts

    def compute_overall_status(self) -> str:
        """Derive overall_status from individual segment states."""
        counts = self.count_by_status()
        total = self.total_segments

        if counts.get(SegmentStatus.COMPLETE.value, 0) == total:
            return "complete"
        # v3.0: All segments approved = awaiting execution
        if counts.get(SegmentStatus.APPROVED.value, 0) == total:
            return "approved"
        if counts.get(SegmentStatus.FAILED.value, 0) == total:
            return "failed"
        if counts.get(SegmentStatus.FAILED.value, 0) > 0 or counts.get(SegmentStatus.BLOCKED.value, 0) > 0:
            if counts.get(SegmentStatus.COMPLETE.value, 0) > 0:
                return "partial"
            if counts.get(SegmentStatus.IN_PROGRESS.value, 0) > 0:
                return "running"
            return "failed"
        if counts.get(SegmentStatus.IN_PROGRESS.value, 0) > 0:
            return "running"
        if counts.get(SegmentStatus.PENDING.value, 0) == total:
            return "running"  # Not started yet, but will run
        return "running"

    def summary(self) -> str:
        """Human-readable summary of job state."""
        counts = self.count_by_status()
        parts = []
        for status_val in [
            SegmentStatus.COMPLETE.value,
            SegmentStatus.FAILED.value,
            SegmentStatus.BLOCKED.value,
            SegmentStatus.IN_PROGRESS.value,
            SegmentStatus.PENDING.value,
        ]:
            count = counts.get(status_val, 0)
            if count > 0:
                parts.append(f"{count} {status_val}")
        return f"JobState({self.job_id}: {', '.join(parts)} — {self.overall_status})"


# =============================================================================
# PERSISTENCE
# =============================================================================


def _now_iso() -> str:
    """Current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def _get_state_path(job_dir_path: str) -> str:
    """Get the path to state.json within a job directory."""
    return os.path.join(job_dir_path, STATE_FILENAME)


def get_job_dir(job_id: str) -> str:
    """Get the job directory path using canonical path construction."""
    return _job_dir(_artifact_root(), job_id)


def init_state(job_id: str, manifest: SegmentManifest) -> JobState:
    """
    Create a fresh JobState from a manifest.

    Initialises all segments as PENDING in the manifest's execution order.
    """
    segments: Dict[str, SegmentState] = {}
    for seg in manifest.segments:
        segments[seg.segment_id] = SegmentState(segment_id=seg.segment_id)

    state = JobState(
        job_id=job_id,
        manifest_version=manifest.manifest_version,
        total_segments=manifest.total_segments,
        segments=segments,
    )

    logger.info("[SEGMENT_LOOP] Initialised state for job %s: %d segments", job_id, manifest.total_segments)
    print(f"[SEGMENT_LOOP] State initialised: {state.summary()}")
    return state


def load_state(job_dir_path: str) -> Optional[JobState]:
    """
    Load state.json from a job directory.

    Returns None if the file doesn't exist (first run).
    """
    state_path = _get_state_path(job_dir_path)

    if not os.path.isfile(state_path):
        return None

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        state = JobState.from_dict(data)
        logger.info("[SEGMENT_LOOP] Loaded state from %s: %s", state_path, state.summary())
        return state
    except Exception as e:
        logger.error("[SEGMENT_LOOP] Failed to load state from %s: %s", state_path, e)
        return None


def save_state(state: JobState, job_dir_path: str) -> None:
    """
    Persist state.json to disk atomically.

    Uses write-to-temp-then-rename to prevent corruption on crash.
    Updates last_updated timestamp before writing.
    """
    state.last_updated = _now_iso()
    state.overall_status = state.compute_overall_status()

    state_path = _get_state_path(job_dir_path)
    os.makedirs(job_dir_path, exist_ok=True)

    data = state.to_dict()
    json_str = json.dumps(data, indent=2, ensure_ascii=False)

    # Atomic write: temp file in same directory, then rename
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=job_dir_path, prefix=".state_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json_str)
            # Atomic rename (same filesystem guaranteed — same directory)
            os.replace(tmp_path, state_path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as e:
        logger.error("[SEGMENT_LOOP] Failed to save state to %s: %s", state_path, e)
        raise


def load_or_init_state(job_id: str, manifest: SegmentManifest) -> JobState:
    """
    Load existing state for crash recovery, or create fresh state.

    If a state.json exists and has IN_PROGRESS segments, reset them to
    PENDING (the segment will be re-run — sandbox execution is stateless).
    """
    job_dir_path = get_job_dir(job_id)
    existing = load_state(job_dir_path)

    if existing is not None:
        # Crash recovery: reset any IN_PROGRESS segments to PENDING
        reset_count = 0
        for seg_state in existing.segments.values():
            if seg_state.status == SegmentStatus.IN_PROGRESS.value:
                seg_state.status = SegmentStatus.PENDING.value
                seg_state.started_at = None
                reset_count += 1

        if reset_count > 0:
            logger.info(
                "[SEGMENT_LOOP] Crash recovery: reset %d IN_PROGRESS segment(s) to PENDING",
                reset_count,
            )
            print(f"[SEGMENT_LOOP] CRASH RECOVERY: reset {reset_count} in-progress segment(s)")
            save_state(existing, job_dir_path)

        return existing

    # First run — create fresh state
    state = init_state(job_id, manifest)
    save_state(state, job_dir_path)
    return state
