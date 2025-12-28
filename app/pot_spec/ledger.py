# FILE: app/pot_spec/ledger.py
"""Append-only ledger for deterministic replay.

Each event is a single JSON object written as one line (ndjson).

v3 (2025-12):
- Added Block 4 events: ARCH_CREATED, ARCH_MIRROR_WRITTEN
- Added Block 5 events: CRITIQUE_CREATED, CRITIQUE_PASS, CRITIQUE_FAIL
- Added Block 6 events: ARCH_REVISED, REVISION_LOOP_STARTED, REVISION_LOOP_TERMINATED
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================

def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime."""
    if not ts_str:
        return None
    try:
        ts = ts_str.replace("Z", "+00:00")
        if "+" not in ts and "-" not in ts[10:]:
            ts = ts + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception:
        try:
            return datetime.strptime(ts_str[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        except Exception:
            return None


# =============================================================================
# Write Operations
# =============================================================================

def append_event(job_artifact_root: str, job_id: str, event: dict[str, Any]) -> str:
    """Append event to jobs/<job_id>/ledger/events.ndjson.

    Returns the absolute path written.
    """
    ledger_dir = os.path.join(job_artifact_root, job_id, "ledger")
    os.makedirs(ledger_dir, exist_ok=True)
    path = os.path.join(ledger_dir, "events.ndjson")

    record = dict(event)
    record.setdefault("ts", _utc_iso())
    record.setdefault("job_id", job_id)

    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")

    return path


# =============================================================================
# Read Operations
# =============================================================================

def read_events(job_artifact_root: str, job_id: str) -> list[dict[str, Any]]:
    """Read all events from a job's ledger.
    
    Returns list of event dicts, or empty list if ledger doesn't exist.
    """
    ledger_path = os.path.join(job_artifact_root, job_id, "ledger", "events.ndjson")
    
    if not os.path.exists(ledger_path):
        return []
    
    events = []
    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"[ledger] Invalid JSON in {job_id}: {e}")
                    continue
    except Exception as e:
        logger.warning(f"[ledger] Failed to read ledger for {job_id}: {e}")
    
    return events


def read_events_in_range(
    job_artifact_root: str,
    job_id: str,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    """Read events from a job's ledger within a time range.
    
    Args:
        job_artifact_root: Root folder for job artifacts
        job_id: Job UUID
        start: Start of time window (inclusive)
        end: End of time window (inclusive)
    
    Returns list of event dicts within the time range.
    """
    all_events = read_events(job_artifact_root, job_id)
    
    filtered = []
    for event in all_events:
        ts = _parse_timestamp(event.get("ts", ""))
        if ts and start <= ts <= end:
            filtered.append(event)
    
    return filtered


# =============================================================================
# Spec-Gate Hash Events (Block 3)
# =============================================================================

def emit_spec_hash_computed(
    job_artifact_root: str,
    job_id: str,
    stage_name: str,
    spec_id: str,
    expected_spec_hash: str,
) -> str:
    """Emit STAGE_SPEC_HASH_COMPUTED event."""
    event = {
        "event": "STAGE_SPEC_HASH_COMPUTED",
        "stage_name": stage_name,
        "spec_id": spec_id,
        "expected_spec_hash": expected_spec_hash,
        "status": "ok",
    }
    return append_event(job_artifact_root, job_id, event)


def emit_spec_hash_verified(
    job_artifact_root: str,
    job_id: str,
    stage_name: str,
    spec_id: str,
    expected_spec_hash: str,
    observed_spec_hash: str,
) -> str:
    """Emit STAGE_SPEC_HASH_VERIFIED event."""
    event = {
        "event": "STAGE_SPEC_HASH_VERIFIED",
        "stage_name": stage_name,
        "spec_id": spec_id,
        "expected_spec_hash": expected_spec_hash,
        "observed_spec_hash": observed_spec_hash,
        "verified": True,
        "status": "ok",
    }
    return append_event(job_artifact_root, job_id, event)


def emit_spec_hash_mismatch(
    job_artifact_root: str,
    job_id: str,
    stage_name: str,
    spec_id: str,
    expected_spec_hash: str,
    observed_spec_hash: Optional[str],
    reason: Optional[str] = None,
) -> str:
    """Emit STAGE_SPEC_HASH_MISMATCH event."""
    message = "spec hash mismatch — pipeline aborted before applying output"
    if reason:
        message = f"{message} ({reason})"
    
    event = {
        "event": "STAGE_SPEC_HASH_MISMATCH",
        "stage_name": stage_name,
        "spec_id": spec_id,
        "expected_spec_hash": expected_spec_hash,
        "observed_spec_hash": observed_spec_hash,
        "verified": False,
        "severity": "ERROR",
        "message": message,
        "status": "error",
    }
    
    logger.warning(
        f"[ledger] SPEC_HASH_MISMATCH job={job_id} stage={stage_name} "
        f"expected={expected_spec_hash[:16]}... observed={observed_spec_hash[:16] if observed_spec_hash else 'None'}..."
    )
    
    return append_event(job_artifact_root, job_id, event)


# =============================================================================
# Architecture Events (Block 4)
# =============================================================================

def emit_arch_created(
    job_artifact_root: str,
    job_id: str,
    arch_id: str,
    arch_version: int,
    arch_hash: str,
    spec_id: str,
    spec_hash: str,
    artefact_id: Optional[str] = None,
    mirror_path: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Emit ARCH_CREATED event when architecture document is stored.
    
    Args:
        job_artifact_root: Root folder for job artifacts
        job_id: Job UUID
        arch_id: Architecture document ID
        arch_version: Version number (1, 2, 3...)
        arch_hash: SHA-256 hash of canonical architecture content
        spec_id: Spec ID this architecture implements
        spec_hash: Spec hash this architecture implements
        artefact_id: DB artefact ID (if stored in DB)
        mirror_path: Filesystem mirror path (if mirrored)
        model: Model that generated this architecture
    """
    event = {
        "event": "ARCH_CREATED",
        "arch_id": arch_id,
        "arch_version": arch_version,
        "arch_hash": arch_hash,
        "spec_id": spec_id,
        "spec_hash": spec_hash,
        "artefact_id": artefact_id,
        "mirror_path": mirror_path,
        "model": model,
        "status": "ok",
    }
    
    logger.info(f"[ledger] ARCH_CREATED job={job_id} arch={arch_id} v{arch_version}")
    
    return append_event(job_artifact_root, job_id, event)


def emit_arch_mirror_written(
    job_artifact_root: str,
    job_id: str,
    arch_id: str,
    arch_version: int,
    mirror_path: str,
    checksum: str,
) -> str:
    """Emit ARCH_MIRROR_WRITTEN event when arch is mirrored to filesystem."""
    event = {
        "event": "ARCH_MIRROR_WRITTEN",
        "arch_id": arch_id,
        "arch_version": arch_version,
        "mirror_path": mirror_path,
        "checksum": checksum,
        "status": "ok",
    }
    return append_event(job_artifact_root, job_id, event)


# =============================================================================
# Critique Events (Block 5)
# =============================================================================

def emit_critique_created(
    job_artifact_root: str,
    job_id: str,
    critique_id: str,
    arch_id: str,
    arch_version: int,
    blocking_count: int,
    non_blocking_count: int,
    overall_pass: bool,
    model: Optional[str] = None,
    json_path: Optional[str] = None,
    md_path: Optional[str] = None,
) -> str:
    """Emit CRITIQUE_CREATED event when critique is generated."""
    event = {
        "event": "CRITIQUE_CREATED",
        "critique_id": critique_id,
        "arch_id": arch_id,
        "arch_version": arch_version,
        "blocking_count": blocking_count,
        "non_blocking_count": non_blocking_count,
        "overall_pass": overall_pass,
        "model": model,
        "json_path": json_path,
        "md_path": md_path,
        "status": "ok",
    }
    
    logger.info(
        f"[ledger] CRITIQUE_CREATED job={job_id} arch={arch_id} "
        f"blocking={blocking_count} pass={overall_pass}"
    )
    
    return append_event(job_artifact_root, job_id, event)


def emit_critique_pass(
    job_artifact_root: str,
    job_id: str,
    critique_id: str,
    arch_id: str,
    arch_version: int,
) -> str:
    """Emit CRITIQUE_PASS event when critique passes (no blocking issues)."""
    event = {
        "event": "CRITIQUE_PASS",
        "critique_id": critique_id,
        "arch_id": arch_id,
        "arch_version": arch_version,
        "status": "ok",
    }
    
    logger.info(f"[ledger] CRITIQUE_PASS job={job_id} arch={arch_id} v{arch_version}")
    
    return append_event(job_artifact_root, job_id, event)


def emit_critique_fail(
    job_artifact_root: str,
    job_id: str,
    critique_id: str,
    arch_id: str,
    arch_version: int,
    blocking_issues: List[str],
) -> str:
    """Emit CRITIQUE_FAIL event when critique fails (has blocking issues)."""
    event = {
        "event": "CRITIQUE_FAIL",
        "critique_id": critique_id,
        "arch_id": arch_id,
        "arch_version": arch_version,
        "blocking_issue_ids": blocking_issues,
        "status": "fail",
    }
    
    logger.info(
        f"[ledger] CRITIQUE_FAIL job={job_id} arch={arch_id} v{arch_version} "
        f"issues={blocking_issues}"
    )
    
    return append_event(job_artifact_root, job_id, event)


# =============================================================================
# Revision Loop Events (Block 6)
# =============================================================================

def emit_revision_loop_started(
    job_artifact_root: str,
    job_id: str,
    arch_id: str,
    max_iterations: int,
) -> str:
    """Emit REVISION_LOOP_STARTED event when entering revision loop."""
    event = {
        "event": "REVISION_LOOP_STARTED",
        "arch_id": arch_id,
        "max_iterations": max_iterations,
        "status": "ok",
    }
    return append_event(job_artifact_root, job_id, event)


def emit_arch_revised(
    job_artifact_root: str,
    job_id: str,
    arch_id: str,
    old_version: int,
    new_version: int,
    new_hash: str,
    addressed_issues: List[str],
    model: Optional[str] = None,
) -> str:
    """Emit ARCH_REVISED event when architecture is revised to address issues."""
    event = {
        "event": "ARCH_REVISED",
        "arch_id": arch_id,
        "old_version": old_version,
        "new_version": new_version,
        "new_hash": new_hash,
        "addressed_issues": addressed_issues,
        "model": model,
        "status": "ok",
    }
    
    logger.info(
        f"[ledger] ARCH_REVISED job={job_id} arch={arch_id} "
        f"v{old_version}→v{new_version} addressed={addressed_issues}"
    )
    
    return append_event(job_artifact_root, job_id, event)


def emit_revision_loop_terminated(
    job_artifact_root: str,
    job_id: str,
    arch_id: str,
    final_version: int,
    reason: str,  # "pass" | "max_iterations" | "error"
    iterations_used: int,
    final_pass: bool,
) -> str:
    """Emit REVISION_LOOP_TERMINATED event when revision loop ends."""
    event = {
        "event": "REVISION_LOOP_TERMINATED",
        "arch_id": arch_id,
        "final_version": final_version,
        "reason": reason,
        "iterations_used": iterations_used,
        "final_pass": final_pass,
        "status": "ok" if final_pass else "exhausted",
    }
    
    logger.info(
        f"[ledger] REVISION_LOOP_TERMINATED job={job_id} arch={arch_id} "
        f"v{final_version} reason={reason} pass={final_pass}"
    )
    
    return append_event(job_artifact_root, job_id, event)


__all__ = [
    # Write
    "append_event",
    # Read
    "read_events",
    "read_events_in_range",
    # Spec-hash events (Block 3)
    "emit_spec_hash_computed",
    "emit_spec_hash_verified",
    "emit_spec_hash_mismatch",
    # Architecture events (Block 4)
    "emit_arch_created",
    "emit_arch_mirror_written",
    # Critique events (Block 5)
    "emit_critique_created",
    "emit_critique_pass",
    "emit_critique_fail",
    # Revision loop events (Block 6)
    "emit_revision_loop_started",
    "emit_arch_revised",
    "emit_revision_loop_terminated",
    # Chunk events (Block 7)
    "emit_chunk_plan_created",
    # Implementation events (Block 8)
    "emit_chunk_implemented",
    "emit_boundary_violation",
    # Verification events (Block 9)
    "emit_verify_pass",
    "emit_verify_fail",
    # Quarantine events (Block 10)
    "emit_quarantine_applied",
    # Deletion events (Block 11)
    "emit_deletion_complete",
    # Replay events (Block 12)
    "emit_replay_pack_created",
]


# =============================================================================
# Block 7: Chunk Plan Events
# =============================================================================

def emit_chunk_plan_created(
    job_artifact_root: str,
    job_id: str,
    plan_id: str,
    arch_id: str,
    arch_version: int,
    chunk_count: int,
    plan_path: Optional[str] = None,
) -> str:
    """Emit CHUNK_PLAN_CREATED event."""
    event = {
        "event": "CHUNK_PLAN_CREATED",
        "plan_id": plan_id,
        "arch_id": arch_id,
        "arch_version": arch_version,
        "chunk_count": chunk_count,
        "plan_path": plan_path,
        "status": "ok",
    }
    logger.info(f"[ledger] CHUNK_PLAN_CREATED job={job_id} chunks={chunk_count}")
    return append_event(job_artifact_root, job_id, event)


# =============================================================================
# Block 8: Implementation Events
# =============================================================================

def emit_chunk_implemented(
    job_artifact_root: str,
    job_id: str,
    chunk_id: str,
    files_added: List[str],
    files_modified: List[str],
    model: Optional[str] = None,
) -> str:
    """Emit CHUNK_IMPLEMENTED event."""
    event = {
        "event": "CHUNK_IMPLEMENTED",
        "chunk_id": chunk_id,
        "files_added": files_added,
        "files_modified": files_modified,
        "file_count": len(files_added) + len(files_modified),
        "model": model,
        "status": "ok",
    }
    logger.info(f"[ledger] CHUNK_IMPLEMENTED job={job_id} chunk={chunk_id}")
    return append_event(job_artifact_root, job_id, event)


def emit_boundary_violation(
    job_artifact_root: str,
    job_id: str,
    chunk_id: str,
    violations: List[Dict[str, Any]],
) -> str:
    """Emit BOUNDARY_VIOLATION event."""
    event = {
        "event": "BOUNDARY_VIOLATION",
        "chunk_id": chunk_id,
        "violations": violations,
        "violation_count": len(violations),
        "severity": "ERROR",
        "status": "rejected",
    }
    logger.warning(f"[ledger] BOUNDARY_VIOLATION job={job_id} chunk={chunk_id} count={len(violations)}")
    return append_event(job_artifact_root, job_id, event)


# =============================================================================
# Block 9: Verification Events
# =============================================================================

def emit_verify_pass(
    job_artifact_root: str,
    job_id: str,
    chunk_id: str,
    tests_passed: int,
    lint_errors: int,
    type_errors: int,
    evidence_paths: Optional[List[str]] = None,
) -> str:
    """Emit VERIFY_PASS event."""
    event = {
        "event": "VERIFY_PASS",
        "chunk_id": chunk_id,
        "tests_passed": tests_passed,
        "lint_errors": lint_errors,
        "type_errors": type_errors,
        "evidence_paths": evidence_paths or [],
        "status": "ok",
    }
    logger.info(f"[ledger] VERIFY_PASS job={job_id} chunk={chunk_id}")
    return append_event(job_artifact_root, job_id, event)


def emit_verify_fail(
    job_artifact_root: str,
    job_id: str,
    chunk_id: str,
    tests_failed: int,
    lint_errors: int,
    type_errors: int,
    failure_summary: str,
    evidence_paths: Optional[List[str]] = None,
) -> str:
    """Emit VERIFY_FAIL event."""
    event = {
        "event": "VERIFY_FAIL",
        "chunk_id": chunk_id,
        "tests_failed": tests_failed,
        "lint_errors": lint_errors,
        "type_errors": type_errors,
        "failure_summary": failure_summary,
        "evidence_paths": evidence_paths or [],
        "severity": "ERROR",
        "status": "fail",
    }
    logger.warning(f"[ledger] VERIFY_FAIL job={job_id} chunk={chunk_id}")
    return append_event(job_artifact_root, job_id, event)


# =============================================================================
# Block 10: Quarantine Events
# =============================================================================

def emit_quarantine_applied(
    job_artifact_root: str,
    job_id: str,
    report_id: str,
    quarantined_files: List[str],
    repo_still_passes: bool,
) -> str:
    """Emit QUARANTINE_APPLIED event."""
    event = {
        "event": "QUARANTINE_APPLIED",
        "report_id": report_id,
        "quarantined_files": quarantined_files,
        "quarantined_count": len(quarantined_files),
        "repo_still_passes": repo_still_passes,
        "status": "ok" if repo_still_passes else "warning",
    }
    logger.info(f"[ledger] QUARANTINE_APPLIED job={job_id} files={len(quarantined_files)}")
    return append_event(job_artifact_root, job_id, event)


# =============================================================================
# Block 11: Deletion Events
# =============================================================================

def emit_deletion_complete(
    job_artifact_root: str,
    job_id: str,
    report_id: str,
    deleted_files: List[str],
    approved_by: str,
    repo_still_passes: bool,
) -> str:
    """Emit DELETION_COMPLETE event."""
    event = {
        "event": "DELETION_COMPLETE",
        "report_id": report_id,
        "deleted_files": deleted_files,
        "deleted_count": len(deleted_files),
        "approved_by": approved_by,
        "repo_still_passes": repo_still_passes,
        "status": "ok" if repo_still_passes else "warning",
    }
    logger.info(f"[ledger] DELETION_COMPLETE job={job_id} deleted={len(deleted_files)}")
    return append_event(job_artifact_root, job_id, event)


# =============================================================================
# Block 12: Replay Pack Events
# =============================================================================

def emit_replay_pack_created(
    job_artifact_root: str,
    job_id: str,
    pack_id: str,
    pack_path: str,
) -> str:
    """Emit REPLAY_PACK_CREATED event."""
    event = {
        "event": "REPLAY_PACK_CREATED",
        "pack_id": pack_id,
        "pack_path": pack_path,
        "status": "ok",
    }
    logger.info(f"[ledger] REPLAY_PACK_CREATED job={job_id} pack={pack_id}")
    return append_event(job_artifact_root, job_id, event)