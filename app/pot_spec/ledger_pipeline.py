# FILE: app/pot_spec/ledger_pipeline.py
"""Ledger events for Blocks 1-6 (Design Loop).

Block 1: Job creation
Block 2: Spec Gate
Block 3: Spec hash verification
Block 4: Architecture generation
Block 5: Critique
Block 6: Revision loop

Plus core pipeline events (status changes, failures, fallbacks).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.pot_spec.ledger_core import append_event

logger = logging.getLogger(__name__)


# =============================================================================
# Block 1: Job Creation Events (Spec §7.2)
# =============================================================================

def emit_job_created(
    job_artifact_root: str,
    job_id: str,
    job_type: str,
    user_request_excerpt: str,
) -> str:
    """Emit JOB_CREATED event when job is initialized.
    
    Spec §7.2: job_id, job_type, user_request_excerpt
    """
    event = {
        "event": "JOB_CREATED",
        "job_type": job_type,
        "user_request_excerpt": user_request_excerpt[:500] if user_request_excerpt else "",
        "status": "ok",
    }
    logger.info(f"[ledger] JOB_CREATED job={job_id} type={job_type}")
    return append_event(job_artifact_root, job_id, event)


# =============================================================================
# Block 2: Spec Gate Events (Spec §7.2)
# =============================================================================

def emit_spec_created(
    job_artifact_root: str,
    job_id: str,
    spec_id: str,
    spec_hash: str,
    spec_version: int,
    parent_spec_id: Optional[str] = None,
) -> str:
    """Emit SPEC_CREATED event when Spec Gate produces spec.
    
    Spec §7.2: spec_id, spec_hash, spec_version, parent_spec_id
    """
    event = {
        "event": "SPEC_CREATED",
        "spec_id": spec_id,
        "spec_hash": spec_hash,
        "spec_version": spec_version,
        "parent_spec_id": parent_spec_id,
        "status": "ok",
    }
    logger.info(f"[ledger] SPEC_CREATED job={job_id} spec={spec_id} v{spec_version}")
    return append_event(job_artifact_root, job_id, event)


def emit_spec_questions_generated(
    job_artifact_root: str,
    job_id: str,
    question_count: int,
    blocking_count: int = 0,
) -> str:
    """Emit SPEC_QUESTIONS_GENERATED event when Spec Gate needs clarification.
    
    Spec §7.2: question_count, blocking_count
    """
    event = {
        "event": "SPEC_QUESTIONS_GENERATED",
        "question_count": question_count,
        "blocking_count": blocking_count,
        "status": "waiting_for_user",
    }
    logger.info(f"[ledger] SPEC_QUESTIONS_GENERATED job={job_id} questions={question_count}")
    return append_event(job_artifact_root, job_id, event)


# =============================================================================
# Block 3: Spec Hash Verification Events (Spec §7.2)
# =============================================================================

def emit_spec_hash_computed(
    job_artifact_root: str,
    job_id: str,
    stage_name: str,
    spec_id: str,
    expected_spec_hash: str,
) -> str:
    """Emit internal tracking event for hash computation."""
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
    spec_hash: str,
) -> str:
    """Emit STAGE_SPEC_HASH_VERIFIED event when header check passes.
    
    Spec §7.2: spec_hash (confirmed)
    """
    event = {
        "event": "STAGE_SPEC_HASH_VERIFIED",
        "stage_id": stage_name,
        "spec_id": spec_id,
        "spec_hash": spec_hash,
        "status": "ok",
    }
    logger.info(f"[ledger] STAGE_SPEC_HASH_VERIFIED job={job_id} stage={stage_name}")
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
    """Emit STAGE_SPEC_HASH_MISMATCH event when header check fails.
    
    Spec §7.2: expected, observed, reason
    """
    message = "spec hash mismatch — pipeline aborted before applying output"
    if reason:
        message = f"{message} ({reason})"
    
    event = {
        "event": "STAGE_SPEC_HASH_MISMATCH",
        "stage_id": stage_name,
        "spec_id": spec_id,
        "expected": expected_spec_hash,
        "observed": observed_spec_hash,
        "reason": reason or "hash_mismatch",
        "severity": "ERROR",
        "message": message,
        "status": "error",
    }
    
    logger.warning(
        f"[ledger] STAGE_SPEC_HASH_MISMATCH job={job_id} stage={stage_name} "
        f"expected={expected_spec_hash[:16]}... observed={observed_spec_hash[:16] if observed_spec_hash else 'None'}..."
    )
    
    return append_event(job_artifact_root, job_id, event)


def emit_spec_hash_missing(
    job_artifact_root: str,
    job_id: str,
    stage_name: str,
    parse_note: str,
) -> str:
    """Emit STAGE_SPEC_HASH_MISSING event when no header found.
    
    Spec §7.2: parse_note
    """
    event = {
        "event": "STAGE_SPEC_HASH_MISSING",
        "stage_id": stage_name,
        "parse_note": parse_note,
        "severity": "ERROR",
        "status": "error",
    }
    logger.warning(f"[ledger] STAGE_SPEC_HASH_MISSING job={job_id} stage={stage_name} note={parse_note}")
    return append_event(job_artifact_root, job_id, event)


# =============================================================================
# Core Pipeline Events (Spec §7.2)
# =============================================================================

def emit_job_status_changed(
    job_artifact_root: str,
    job_id: str,
    old_status: str,
    new_status: str,
    reason: str = "",
) -> str:
    """Emit JOB_STATUS_CHANGED event on any status transition.
    
    Spec §7.2: old_status, new_status, reason
    """
    event = {
        "event": "JOB_STATUS_CHANGED",
        "old_status": old_status,
        "new_status": new_status,
        "reason": reason,
        "status": "ok",
    }
    logger.info(f"[ledger] JOB_STATUS_CHANGED job={job_id} {old_status} -> {new_status}")
    return append_event(job_artifact_root, job_id, event)


def emit_stage_started(
    job_artifact_root: str,
    job_id: str,
    stage_id: str,
    stage_run_id: str,
) -> str:
    """Emit STAGE_STARTED event when any stage begins.
    
    Spec §7.2: stage_id, stage_run_id
    """
    event = {
        "event": "STAGE_STARTED",
        "stage_id": stage_id,
        "stage_run_id": stage_run_id,
        "status": "ok",
    }
    logger.debug(f"[ledger] STAGE_STARTED job={job_id} stage={stage_id} run={stage_run_id}")
    return append_event(job_artifact_root, job_id, event)


def emit_stage_output_stored(
    job_artifact_root: str,
    job_id: str,
    stage_id: str,
    artifact_path: str,
    content_hash: str,
) -> str:
    """Emit STAGE_OUTPUT_STORED event when output is accepted.
    
    Spec §7.2: artifact_path, content_hash
    """
    event = {
        "event": "STAGE_OUTPUT_STORED",
        "stage_id": stage_id,
        "artifact_path": artifact_path,
        "content_hash": content_hash,
        "status": "ok",
    }
    logger.info(f"[ledger] STAGE_OUTPUT_STORED job={job_id} stage={stage_id} path={artifact_path}")
    return append_event(job_artifact_root, job_id, event)


def emit_stage_failed(
    job_artifact_root: str,
    job_id: str,
    stage_id: str,
    error_type: str,
    error_message: str,
    attempted_path: Optional[str] = None,
) -> str:
    """Emit STAGE_FAILED event on stage error.
    
    Spec §7.2: error_type, error_message, attempted_path (if artifact)
    """
    event = {
        "event": "STAGE_FAILED",
        "stage_id": stage_id,
        "error_type": error_type,
        "error_message": error_message,
        "severity": "ERROR",
        "status": "failed",
    }
    if attempted_path:
        event["attempted_path"] = attempted_path
    
    logger.error(f"[ledger] STAGE_FAILED job={job_id} stage={stage_id} error={error_type}: {error_message}")
    return append_event(job_artifact_root, job_id, event)


def emit_provider_fallback(
    job_artifact_root: str,
    job_id: str,
    from_provider: str,
    from_model: str,
    to_provider: str,
    to_model: str,
    reason: str,
) -> str:
    """Emit PROVIDER_FALLBACK event when primary unavailable.
    
    Spec §7.2: from_provider, from_model, to_provider, to_model, reason
    """
    event = {
        "event": "PROVIDER_FALLBACK",
        "from_provider": from_provider,
        "from_model": from_model,
        "to_provider": to_provider,
        "to_model": to_model,
        "reason": reason,
        "status": "ok",
    }
    logger.warning(f"[ledger] PROVIDER_FALLBACK job={job_id} {from_provider}/{from_model} -> {to_provider}/{to_model}")
    return append_event(job_artifact_root, job_id, event)


# =============================================================================
# Terminal State Events (Spec §7.2)
# =============================================================================

def emit_job_completed(
    job_artifact_root: str,
    job_id: str,
    final_arch_version: int,
    total_iterations: int,
) -> str:
    """Emit JOB_COMPLETED event on pipeline success.
    
    Spec §7.2: final_arch_version, total_iterations
    """
    event = {
        "event": "JOB_COMPLETED",
        "final_arch_version": final_arch_version,
        "total_iterations": total_iterations,
        "status": "completed",
    }
    logger.info(f"[ledger] JOB_COMPLETED job={job_id} arch_v{final_arch_version} iterations={total_iterations}")
    return append_event(job_artifact_root, job_id, event)


def emit_job_failed(
    job_artifact_root: str,
    job_id: str,
    error_type: str,
    error_message: str,
    failed_stage_id: Optional[str] = None,
) -> str:
    """Emit JOB_FAILED event on pipeline failure.
    
    Spec §7.2: error_type, error_message, failed_stage_id
    """
    event = {
        "event": "JOB_FAILED",
        "error_type": error_type,
        "error_message": error_message,
        "failed_stage_id": failed_stage_id,
        "severity": "ERROR",
        "status": "failed",
    }
    logger.error(f"[ledger] JOB_FAILED job={job_id} stage={failed_stage_id} error={error_type}")
    return append_event(job_artifact_root, job_id, event)


def emit_job_aborted(
    job_artifact_root: str,
    job_id: str,
    reason: str,
) -> str:
    """Emit JOB_ABORTED event when user cancels.
    
    Spec §7.2: reason
    """
    event = {
        "event": "JOB_ABORTED",
        "reason": reason,
        "status": "aborted",
    }
    logger.info(f"[ledger] JOB_ABORTED job={job_id} reason={reason}")
    return append_event(job_artifact_root, job_id, event)


# =============================================================================
# Block 4: Architecture Events (Spec §7.2)
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
    """Emit ARCH_CREATED event when architecture document is generated.
    
    Spec §7.2: arch_id, arch_hash, arch_version
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
# Block 5: Critique Events (Spec §7.2)
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
    """Emit CRITIQUE_CREATED event when critique is generated.
    
    Spec §7.2: critique_id, blocking_count, non_blocking_count, overall_pass
    """
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
    """Emit CRITIQUE_PASS event when critique passes (no blocking issues).
    
    Spec §7.2: arch_version
    """
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
    """Emit CRITIQUE_FAIL event when critique fails (has blocking issues).
    
    Spec §7.2: blocking_issue_ids
    """
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
# Block 6: Revision Loop Events (Spec §7.2)
# =============================================================================

def emit_revision_loop_started(
    job_artifact_root: str,
    job_id: str,
    arch_id: str,
    max_iterations: int,
) -> str:
    """Emit REVISION_LOOP_STARTED event when entering revision loop.
    
    Spec §7.2: max_iterations
    """
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
    """Emit ARCH_REVISED event when architecture is revised to address issues.
    
    Spec §7.2: old_version, new_version, addressed_issues
    """
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
    """Emit REVISION_LOOP_TERMINATED event when revision loop ends.
    
    Spec §7.2: reason (pass/max_iterations/error), final_version, final_pass
    """
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


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Block 1
    "emit_job_created",
    # Block 2
    "emit_spec_created",
    "emit_spec_questions_generated",
    # Block 3
    "emit_spec_hash_computed",
    "emit_spec_hash_verified",
    "emit_spec_hash_mismatch",
    "emit_spec_hash_missing",
    # Core
    "emit_job_status_changed",
    "emit_stage_started",
    "emit_stage_output_stored",
    "emit_stage_failed",
    "emit_provider_fallback",
    # Terminal
    "emit_job_completed",
    "emit_job_failed",
    "emit_job_aborted",
    # Block 4
    "emit_arch_created",
    "emit_arch_mirror_written",
    # Block 5
    "emit_critique_created",
    "emit_critique_pass",
    "emit_critique_fail",
    # Block 6
    "emit_revision_loop_started",
    "emit_arch_revised",
    "emit_revision_loop_terminated",
]
