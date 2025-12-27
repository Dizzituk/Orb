# FILE: app/overwatcher/__init__.py
"""Overwatcher: Implementation controller with safety gates.

Blocks 7-12 implementation:
- Block 7: Chunk planner (arch -> bounded implementation units)
- Block 8: Implementation executor with diff boundary enforcement
- Block 9: Verification gate (tests/lint/types on touched files)
- Block 10: Quarantine workflow (static + dynamic evidence)
- Block 11: Deletion workflow (approval-gated)
- Block 12: Replay pack (deterministic bundle)
"""

from app.overwatcher.schemas import (
    # Enums
    ChunkStatus,
    VerificationStatus,
    FileAction,
    QuarantineReason,
    # Block 7
    ChunkStep,
    ChunkVerification,
    Chunk,
    ChunkPlan,
    # Block 8
    BoundaryViolation,
    DiffCheckResult,
    # Block 9
    CommandResult,
    VerificationResult,
    # Block 10-11
    StaticEvidence,
    DynamicEvidence,
    QuarantineCandidate,
    QuarantineReport,
    DeletionReport,
    # Block 12
    ReplayPack,
)

from app.overwatcher.planner import (
    generate_chunk_plan,
    store_chunk_plan,
    load_chunk_plan,
    topological_sort_chunks,
    get_next_chunk,
)

from app.overwatcher.executor import (
    check_diff_boundaries,
    execute_chunk,
    create_backup,
    rollback_chunk,
)

from app.overwatcher.verifier import (
    verify_chunk,
    run_full_verification,
    run_smoke_boot,
)

from app.overwatcher.quarantine import (
    run_quarantine_workflow,
    run_deletion_workflow,
)

from app.overwatcher.replay import (
    generate_replay_pack,
    load_replay_pack,
    compare_replay_packs,
)

__all__ = [
    # Schemas
    "ChunkStatus",
    "VerificationStatus",
    "FileAction",
    "QuarantineReason",
    "ChunkStep",
    "ChunkVerification",
    "Chunk",
    "ChunkPlan",
    "BoundaryViolation",
    "DiffCheckResult",
    "CommandResult",
    "VerificationResult",
    "StaticEvidence",
    "DynamicEvidence",
    "QuarantineCandidate",
    "QuarantineReport",
    "DeletionReport",
    "ReplayPack",
    # Planner (Block 7)
    "generate_chunk_plan",
    "store_chunk_plan",
    "load_chunk_plan",
    "topological_sort_chunks",
    "get_next_chunk",
    # Executor (Block 8)
    "check_diff_boundaries",
    "execute_chunk",
    "create_backup",
    "rollback_chunk",
    # Verifier (Block 9)
    "verify_chunk",
    "run_full_verification",
    "run_smoke_boot",
    # Quarantine (Block 10-11)
    "run_quarantine_workflow",
    "run_deletion_workflow",
    # Replay (Block 12)
    "generate_replay_pack",
    "load_replay_pack",
    "compare_replay_packs",
]
