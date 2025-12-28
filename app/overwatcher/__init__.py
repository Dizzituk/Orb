# FILE: app/overwatcher/__init__.py
"""Overwatcher: Implementation controller with safety gates.

Blocks 7-12 implementation:
- Block 7: Chunk planner (arch -> bounded implementation units)
- Block 8: Implementation executor with diff boundary enforcement
- Block 9: Verification gate (tests/lint/types) + Overwatcher supervisor
- Block 10: Quarantine workflow (static + dynamic evidence)
- Block 11: Deletion workflow (approval-gated)
- Block 12: Replay pack (deterministic bundle)

Spec §9.1-9.6 additions:
- ErrorSignature for three-strike tracking
- Evidence bundle for cost-controlled Overwatcher input
- Overwatcher (GPT-5.2 Pro) supervisor - diagnoses failures, no code output
- Deep Research (Strike 2 only) via web search

Sandbox Bridge:
- SandboxClient: HTTP client for Windows Sandbox controller
- SandboxVerifier: Block 9 verification via isolated sandbox
- SandboxExecutor: Block 8 file writes via sandbox
- EvidenceLoader: Load zobie_map evidence packs
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
    SamplingParams,
    StageConfig,
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

# Block 9: Overwatcher supervisor (Spec §9.1-9.6)
from app.overwatcher.error_signature import (
    ErrorSignature,
    compute_error_signature,
    signatures_match,
)

from app.overwatcher.evidence import (
    FileChange,
    TestResult,
    LintResult,
    EnvironmentContext,
    EvidenceBundle,
    build_evidence_bundle,
)

from app.overwatcher.overwatcher import (
    Decision,
    FixAction,
    VerificationStep,
    OverwatcherOutput,
    run_overwatcher,
)

from app.overwatcher.deep_research import (
    ResearchSource,
    DeepResearchResult,
    run_deep_research,
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

from app.overwatcher.orchestrator import (
    PipelineState,
    StrikeState,
    HashVerificationError,
    BoundaryViolationError,
    VerificationFailedError,
    StrikeThreeError,
    verify_stage_hash,
    execute_chunk_with_rollback,
    run_chunk_with_strikes,
    run_implementation_loop,
    build_hash_header,
)

# Sandbox Bridge
from app.overwatcher.sandbox_client import (
    SandboxClient,
    SandboxError,
    ShellResult,
    get_sandbox_client,
)

from app.overwatcher.sandbox_verifier import (
    verify_chunk_sandbox,
    run_full_verification_sandbox,
    run_smoke_boot_sandbox,
    sandbox_run_pytest,
    sandbox_run_ruff,
    sandbox_run_mypy,
)

from app.overwatcher.sandbox_executor import (
    execute_chunk_sandbox,
    execute_chunk_with_sandbox_fallback,
    create_sandbox_backup,
    rollback_sandbox_changes,
)

from app.overwatcher.evidence_loader import (
    EvidencePack,
    Symbol,
    Route,
    EnumDef,
    DictConstant,
    load_evidence_pack,
    find_latest_evidence_dir,
    check_modification_safety,
)

# Job Runner
from app.overwatcher.job_runner import (
    SimpleJob,
    JobResult,
    run_simple_job,
    run_shell_job,
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
    "SamplingParams",
    "StageConfig",
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
    # Verifier (Block 9 - command execution)
    "verify_chunk",
    "run_full_verification",
    "run_smoke_boot",
    # Error Signature (Spec §9.4)
    "ErrorSignature",
    "compute_error_signature",
    "signatures_match",
    # Evidence Bundle (Spec §9.3)
    "FileChange",
    "TestResult",
    "LintResult",
    "EnvironmentContext",
    "EvidenceBundle",
    "build_evidence_bundle",
    # Overwatcher Supervisor (Spec §9.1-9.2)
    "Decision",
    "FixAction",
    "VerificationStep",
    "OverwatcherOutput",
    "run_overwatcher",
    # Deep Research (Spec §9.5)
    "ResearchSource",
    "DeepResearchResult",
    "run_deep_research",
    # Quarantine (Block 10-11)
    "run_quarantine_workflow",
    "run_deletion_workflow",
    # Replay (Block 12)
    "generate_replay_pack",
    "load_replay_pack",
    "compare_replay_packs",
    # Orchestrator
    "PipelineState",
    "StrikeState",
    "HashVerificationError",
    "BoundaryViolationError",
    "VerificationFailedError",
    "StrikeThreeError",
    "verify_stage_hash",
    "execute_chunk_with_rollback",
    "run_chunk_with_strikes",
    "run_implementation_loop",
    "build_hash_header",
    # Sandbox Client
    "SandboxClient",
    "SandboxError",
    "ShellResult",
    "get_sandbox_client",
    # Sandbox Verifier
    "verify_chunk_sandbox",
    "run_full_verification_sandbox",
    "run_smoke_boot_sandbox",
    "sandbox_run_pytest",
    "sandbox_run_ruff",
    "sandbox_run_mypy",
    # Sandbox Executor
    "execute_chunk_sandbox",
    "execute_chunk_with_sandbox_fallback",
    "create_sandbox_backup",
    "rollback_sandbox_changes",
    # Evidence Loader
    "EvidencePack",
    "Symbol",
    "Route",
    "EnumDef",
    "DictConstant",
    "load_evidence_pack",
    "find_latest_evidence_dir",
    "check_modification_safety",
    # Job Runner
    "SimpleJob",
    "JobResult",
    "run_simple_job",
    "run_shell_job",
]
