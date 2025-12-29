# FILE: app/pot_spec/ledger_overwatcher.py
"""Ledger events for Blocks 7-12 (Overwatcher Implementation Loop).

Block 7: Chunk planning
Block 8: Implementation
Block 9: Verification
Block 10: Deep Research (on Strike 2)
Block 11: Promotion Gate
Block 12: Quarantine and Replay
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.pot_spec.ledger_core import append_event

logger = logging.getLogger(__name__)


# =============================================================================
# Block 7: Chunk Planning Events
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
    """Emit CHUNK_PLAN_CREATED event when planner generates chunks.
    
    Args:
        job_artifact_root: Root for artifacts
        job_id: Job UUID
        plan_id: Plan UUID
        arch_id: Source architecture ID
        arch_version: Architecture version
        chunk_count: Number of chunks in plan
        plan_path: Path to stored plan JSON
    """
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
    """Emit CHUNK_IMPLEMENTED event when Sonnet completes a chunk.
    
    Args:
        job_artifact_root: Root for artifacts
        job_id: Job UUID
        chunk_id: Chunk being implemented
        files_added: List of new files created
        files_modified: List of existing files modified
        model: Model that generated the code
    """
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
    """Emit BOUNDARY_VIOLATION event when diff exceeds allowed files.
    
    This triggers automatic rollback.
    
    Args:
        job_artifact_root: Root for artifacts
        job_id: Job UUID
        chunk_id: Chunk that violated boundaries
        violations: List of violation dicts with file_path, action, reason
    """
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
    """Emit VERIFY_PASS event when chunk passes verification gate.
    
    Args:
        job_artifact_root: Root for artifacts
        job_id: Job UUID
        chunk_id: Chunk that was verified
        tests_passed: Number of tests passed
        lint_errors: Number of lint errors (should be 0)
        type_errors: Number of type errors (should be 0)
        evidence_paths: Paths to verification output files
    """
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
    """Emit VERIFY_FAIL event when chunk fails verification gate.
    
    This triggers rollback and may trigger Strike escalation.
    
    Args:
        job_artifact_root: Root for artifacts
        job_id: Job UUID
        chunk_id: Chunk that failed
        tests_failed: Number of tests failed
        lint_errors: Number of lint errors
        type_errors: Number of type errors
        failure_summary: Human-readable summary
        evidence_paths: Paths to verification output files
    """
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
# Block 10: Deep Research Events
# =============================================================================

def emit_deep_research_triggered(
    job_artifact_root: str,
    job_id: str,
    chunk_id: str,
    strike_number: int,
    research_query: str,
) -> str:
    """Emit DEEP_RESEARCH_TRIGGERED event on Strike 2.
    
    Args:
        job_artifact_root: Root for artifacts
        job_id: Job UUID
        chunk_id: Chunk requiring research
        strike_number: Current strike count (should be 2)
        research_query: Query sent to research model
    """
    event = {
        "event": "DEEP_RESEARCH_TRIGGERED",
        "chunk_id": chunk_id,
        "strike_number": strike_number,
        "research_query": research_query[:500],
        "status": "ok",
    }
    logger.info(f"[ledger] DEEP_RESEARCH_TRIGGERED job={job_id} chunk={chunk_id} strike={strike_number}")
    return append_event(job_artifact_root, job_id, event)


def emit_deep_research_completed(
    job_artifact_root: str,
    job_id: str,
    chunk_id: str,
    guidance_summary: str,
    sources_count: int,
) -> str:
    """Emit DEEP_RESEARCH_COMPLETED event when research returns.
    
    Args:
        job_artifact_root: Root for artifacts
        job_id: Job UUID
        chunk_id: Chunk that received research
        guidance_summary: Summary of research findings
        sources_count: Number of sources consulted
    """
    event = {
        "event": "DEEP_RESEARCH_COMPLETED",
        "chunk_id": chunk_id,
        "guidance_summary": guidance_summary[:500],
        "sources_count": sources_count,
        "status": "ok",
    }
    logger.info(f"[ledger] DEEP_RESEARCH_COMPLETED job={job_id} chunk={chunk_id}")
    return append_event(job_artifact_root, job_id, event)


# =============================================================================
# Block 11: Promotion Gate Events
# =============================================================================

def emit_promotion_requested(
    job_artifact_root: str,
    job_id: str,
    chunk_id: str,
    files_to_promote: List[str],
) -> str:
    """Emit PROMOTION_REQUESTED event when chunk ready for Main.
    
    Args:
        job_artifact_root: Root for artifacts
        job_id: Job UUID
        chunk_id: Chunk to promote
        files_to_promote: List of files to copy to Main
    """
    event = {
        "event": "PROMOTION_REQUESTED",
        "chunk_id": chunk_id,
        "files_to_promote": files_to_promote,
        "file_count": len(files_to_promote),
        "status": "pending",
    }
    logger.info(f"[ledger] PROMOTION_REQUESTED job={job_id} chunk={chunk_id}")
    return append_event(job_artifact_root, job_id, event)


def emit_promotion_approved(
    job_artifact_root: str,
    job_id: str,
    chunk_id: str,
    files_promoted: List[str],
) -> str:
    """Emit PROMOTION_APPROVED event when chunk promoted to Main.
    
    Args:
        job_artifact_root: Root for artifacts
        job_id: Job UUID
        chunk_id: Chunk promoted
        files_promoted: List of files copied to Main
    """
    event = {
        "event": "PROMOTION_APPROVED",
        "chunk_id": chunk_id,
        "files_promoted": files_promoted,
        "file_count": len(files_promoted),
        "status": "ok",
    }
    logger.info(f"[ledger] PROMOTION_APPROVED job={job_id} chunk={chunk_id}")
    return append_event(job_artifact_root, job_id, event)


def emit_promotion_rejected(
    job_artifact_root: str,
    job_id: str,
    chunk_id: str,
    reason: str,
) -> str:
    """Emit PROMOTION_REJECTED event when promotion gate fails.
    
    Args:
        job_artifact_root: Root for artifacts
        job_id: Job UUID
        chunk_id: Chunk rejected
        reason: Reason for rejection
    """
    event = {
        "event": "PROMOTION_REJECTED",
        "chunk_id": chunk_id,
        "reason": reason,
        "severity": "ERROR",
        "status": "rejected",
    }
    logger.warning(f"[ledger] PROMOTION_REJECTED job={job_id} chunk={chunk_id} reason={reason}")
    return append_event(job_artifact_root, job_id, event)


# =============================================================================
# Block 12: Quarantine and Replay Events
# =============================================================================

def emit_quarantine_created(
    job_artifact_root: str,
    job_id: str,
    stage_id: str,
    reason: str,
    quarantine_path: str,
) -> str:
    """Emit QUARANTINE_CREATED event when output is quarantined.
    
    Spec §7.2: stage_id, reason, quarantine_path
    
    Args:
        job_artifact_root: Root for artifacts
        job_id: Job UUID
        stage_id: Stage that produced quarantined output
        reason: Reason for quarantine
        quarantine_path: Path where quarantined files stored
    """
    event = {
        "event": "QUARANTINE_CREATED",
        "stage_id": stage_id,
        "reason": reason,
        "quarantine_path": quarantine_path,
        "status": "quarantined",
    }
    logger.warning(f"[ledger] QUARANTINE_CREATED job={job_id} stage={stage_id} reason={reason}")
    return append_event(job_artifact_root, job_id, event)


def emit_quarantine_applied(
    job_artifact_root: str,
    job_id: str,
    report_id: str,
    quarantined_files: List[str],
    repo_still_passes: bool,
) -> str:
    """Emit QUARANTINE_APPLIED event (legacy, for dead code removal).
    
    Args:
        job_artifact_root: Root for artifacts
        job_id: Job UUID
        report_id: Quarantine report ID
        quarantined_files: Files moved to quarantine
        repo_still_passes: Whether repo passes after quarantine
    """
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


def emit_deletion_complete(
    job_artifact_root: str,
    job_id: str,
    report_id: str,
    deleted_files: List[str],
    approved_by: str,
    repo_still_passes: bool,
) -> str:
    """Emit DELETION_COMPLETE event when quarantined files are deleted.
    
    Args:
        job_artifact_root: Root for artifacts
        job_id: Job UUID
        report_id: Deletion report ID
        deleted_files: Files permanently deleted
        approved_by: Who approved deletion ("user" or "auto")
        repo_still_passes: Whether repo passes after deletion
    """
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


def emit_replay_pack_created(
    job_artifact_root: str,
    job_id: str,
    pack_id: str,
    pack_path: str,
) -> str:
    """Emit REPLAY_PACK_CREATED event when replay bundle is generated.
    
    Args:
        job_artifact_root: Root for artifacts
        job_id: Job UUID
        pack_id: Replay pack UUID
        pack_path: Path to replay pack JSON
    """
    event = {
        "event": "REPLAY_PACK_CREATED",
        "pack_id": pack_id,
        "pack_path": pack_path,
        "status": "ok",
    }
    logger.info(f"[ledger] REPLAY_PACK_CREATED job={job_id} pack={pack_id}")
    return append_event(job_artifact_root, job_id, event)


# =============================================================================
# Strike Tracking Events
# =============================================================================

def emit_strike_recorded(
    job_artifact_root: str,
    job_id: str,
    chunk_id: str,
    strike_number: int,
    reason: str,
) -> str:
    """Emit STRIKE_RECORDED event when chunk fails verification.
    
    Args:
        job_artifact_root: Root for artifacts
        job_id: Job UUID
        chunk_id: Chunk that failed
        strike_number: Current strike count (1, 2, or 3)
        reason: Reason for strike
    """
    event = {
        "event": "STRIKE_RECORDED",
        "chunk_id": chunk_id,
        "strike_number": strike_number,
        "reason": reason,
        "status": "warning" if strike_number < 3 else "error",
    }
    logger.warning(f"[ledger] STRIKE_RECORDED job={job_id} chunk={chunk_id} strike={strike_number}")
    return append_event(job_artifact_root, job_id, event)


def emit_chunk_abandoned(
    job_artifact_root: str,
    job_id: str,
    chunk_id: str,
    strike_count: int,
    final_error: str,
) -> str:
    """Emit CHUNK_ABANDONED event on Strike 3 (chunk exhausted).
    
    Args:
        job_artifact_root: Root for artifacts
        job_id: Job UUID
        chunk_id: Chunk being abandoned
        strike_count: Final strike count (should be 3)
        final_error: Last error message
    """
    event = {
        "event": "CHUNK_ABANDONED",
        "chunk_id": chunk_id,
        "strike_count": strike_count,
        "final_error": final_error,
        "severity": "ERROR",
        "status": "abandoned",
    }
    logger.error(f"[ledger] CHUNK_ABANDONED job={job_id} chunk={chunk_id} strikes={strike_count}")
    return append_event(job_artifact_root, job_id, event)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Block 7
    "emit_chunk_plan_created",
    # Block 8
    "emit_chunk_implemented",
    "emit_boundary_violation",
    # Block 9
    "emit_verify_pass",
    "emit_verify_fail",
    # Block 10
    "emit_deep_research_triggered",
    "emit_deep_research_completed",
    # Block 11
    "emit_promotion_requested",
    "emit_promotion_approved",
    "emit_promotion_rejected",
    # Block 12
    "emit_quarantine_created",
    "emit_quarantine_applied",
    "emit_deletion_complete",
    "emit_replay_pack_created",
    # Strike tracking
    "emit_strike_recorded",
    "emit_chunk_abandoned",
]
