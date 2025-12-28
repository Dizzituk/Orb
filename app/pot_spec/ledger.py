# FILE: app/pot_spec/ledger.py
"""Append-only ledger for deterministic replay.

Each event is a single JSON object written as one line (ndjson).

Spec v2.3 §7: Required ledger events for full audit trail.

This file contains core read/write operations. Events are split across:
- ledger_pipeline.py: Blocks 1-6 (spec, arch, critique, revision)
- ledger_overwatcher.py: Blocks 7-12 (chunk, implementation, verification, quarantine)
"""

from __future__ import annotations

# =============================================================================
# Core functions from ledger_core (avoids circular imports)
# =============================================================================

from app.pot_spec.ledger_core import (
    append_event,
    read_events,
    read_events_in_range,
)

# =============================================================================
# Re-exports from submodules
# =============================================================================

from app.pot_spec.ledger_pipeline import (
    # Block 1: Job creation
    emit_job_created,
    # Block 2: Spec Gate
    emit_spec_created,
    emit_spec_questions_generated,
    # Block 3: Spec hash verification
    emit_spec_hash_computed,
    emit_spec_hash_verified,
    emit_spec_hash_mismatch,
    emit_spec_hash_missing,
    # Core pipeline events
    emit_job_status_changed,
    emit_stage_started,
    emit_stage_output_stored,
    emit_stage_failed,
    emit_provider_fallback,
    # Terminal states
    emit_job_completed,
    emit_job_failed,
    emit_job_aborted,
    # Block 4: Architecture
    emit_arch_created,
    emit_arch_mirror_written,
    # Block 5: Critique
    emit_critique_created,
    emit_critique_pass,
    emit_critique_fail,
    # Block 6: Revision loop
    emit_revision_loop_started,
    emit_arch_revised,
    emit_revision_loop_terminated,
)

from app.pot_spec.ledger_overwatcher import (
    # Block 7: Chunk planning
    emit_chunk_plan_created,
    # Block 8: Implementation
    emit_chunk_implemented,
    emit_boundary_violation,
    # Block 9: Verification
    emit_verify_pass,
    emit_verify_fail,
    # Block 10-12: Quarantine/Replay
    emit_quarantine_created,
    emit_quarantine_applied,
    emit_deletion_complete,
    emit_replay_pack_created,
)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core
    "append_event",
    "read_events",
    "read_events_in_range",
    # Block 1: Job creation
    "emit_job_created",
    # Block 2: Spec Gate
    "emit_spec_created",
    "emit_spec_questions_generated",
    # Block 3: Spec hash verification
    "emit_spec_hash_computed",
    "emit_spec_hash_verified",
    "emit_spec_hash_mismatch",
    "emit_spec_hash_missing",
    # Core pipeline events
    "emit_job_status_changed",
    "emit_stage_started",
    "emit_stage_output_stored",
    "emit_stage_failed",
    "emit_provider_fallback",
    # Terminal states
    "emit_job_completed",
    "emit_job_failed",
    "emit_job_aborted",
    # Block 4: Architecture
    "emit_arch_created",
    "emit_arch_mirror_written",
    # Block 5: Critique
    "emit_critique_created",
    "emit_critique_pass",
    "emit_critique_fail",
    # Block 6: Revision loop
    "emit_revision_loop_started",
    "emit_arch_revised",
    "emit_revision_loop_terminated",
    # Block 7: Chunk planning
    "emit_chunk_plan_created",
    # Block 8: Implementation
    "emit_chunk_implemented",
    "emit_boundary_violation",
    # Block 9: Verification
    "emit_verify_pass",
    "emit_verify_fail",
    # Block 10-12: Quarantine/Replay
    "emit_quarantine_created",
    "emit_quarantine_applied",
    "emit_deletion_complete",
    "emit_replay_pack_created",
]
