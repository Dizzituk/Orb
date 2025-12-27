# FILE: app/overwatcher/schemas.py
"""Overwatcher schemas for chunk planning, execution, and verification.

Block 7-12 data structures:
- Chunk: Bounded implementation unit with file permissions
- VerificationResult: Gate pass/fail with evidence
- QuarantineCandidate: File with static/dynamic evidence
- ReplayPack: Deterministic replay bundle
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# Enums
# =============================================================================

class ChunkStatus(str, Enum):
    """Status of a chunk in the implementation pipeline."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class VerificationStatus(str, Enum):
    """Status of a verification gate."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class FileAction(str, Enum):
    """Allowed file actions in a chunk."""
    ADD = "add"
    MODIFY = "modify"
    DELETE = "delete"


class QuarantineReason(str, Enum):
    """Reason for quarantining a file."""
    NO_REFERENCES = "no_references"
    NO_IMPORTS = "no_imports"
    UNUSED_EXPORT = "unused_export"
    DEAD_CODE = "dead_code"
    DEPRECATED = "deprecated"
    MANUAL_FLAG = "manual_flag"


# =============================================================================
# Block 7: Chunk Schema
# =============================================================================

@dataclass
class ChunkStep:
    """A single step within a chunk implementation."""
    step_id: str
    description: str
    file_path: str
    action: FileAction
    details: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "file_path": self.file_path,
            "action": self.action.value if isinstance(self.action, FileAction) else self.action,
            "details": self.details,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkStep":
        return cls(
            step_id=data.get("step_id", ""),
            description=data.get("description", ""),
            file_path=data.get("file_path", ""),
            action=FileAction(data.get("action", "modify")),
            details=data.get("details", ""),
        )


@dataclass
class ChunkVerification:
    """Verification commands for a chunk."""
    commands: List[str] = field(default_factory=list)
    expected_outcomes: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "commands": self.commands,
            "expected_outcomes": self.expected_outcomes,
            "timeout_seconds": self.timeout_seconds,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkVerification":
        return cls(
            commands=data.get("commands", []),
            expected_outcomes=data.get("expected_outcomes", {}),
            timeout_seconds=data.get("timeout_seconds", 60),
        )


@dataclass
class Chunk:
    """A bounded implementation unit for Sonnet coding.
    
    Each chunk is context-sized and self-contained:
    - Clear objective and scope
    - Explicit file permissions (add/modify/delete)
    - Spec and arch traceability
    - Verification commands
    - Rollback plan
    - Stop conditions
    """
    chunk_id: str
    title: str
    objective: str
    
    # Traceability
    spec_refs: List[str] = field(default_factory=list)  # ["MUST-1", "SHOULD-2"]
    arch_refs: List[str] = field(default_factory=list)  # ["Section 2.1", "Module: Auth"]
    
    # File permissions (Block 8 enforcement)
    allowed_files: Dict[str, List[str]] = field(default_factory=lambda: {
        "add": [],
        "modify": [],
        "delete_candidates": [],
    })
    
    # Implementation steps
    steps: List[ChunkStep] = field(default_factory=list)
    
    # Verification (Block 9)
    verification: ChunkVerification = field(default_factory=ChunkVerification)
    
    # Rollback
    rollback_plan: str = ""
    
    # Stop conditions
    stop_conditions: List[str] = field(default_factory=list)
    
    # Metadata
    status: ChunkStatus = ChunkStatus.PENDING
    priority: int = 0  # Lower = higher priority
    estimated_tokens: int = 0
    dependencies: List[str] = field(default_factory=list)  # chunk_ids this depends on
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "title": self.title,
            "objective": self.objective,
            "spec_refs": self.spec_refs,
            "arch_refs": self.arch_refs,
            "allowed_files": self.allowed_files,
            "steps": [s.to_dict() for s in self.steps],
            "verification": self.verification.to_dict(),
            "rollback_plan": self.rollback_plan,
            "stop_conditions": self.stop_conditions,
            "status": self.status.value if isinstance(self.status, ChunkStatus) else self.status,
            "priority": self.priority,
            "estimated_tokens": self.estimated_tokens,
            "dependencies": self.dependencies,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=data.get("chunk_id", ""),
            title=data.get("title", ""),
            objective=data.get("objective", ""),
            spec_refs=data.get("spec_refs", []),
            arch_refs=data.get("arch_refs", []),
            allowed_files=data.get("allowed_files", {"add": [], "modify": [], "delete_candidates": []}),
            steps=[ChunkStep.from_dict(s) for s in data.get("steps", [])],
            verification=ChunkVerification.from_dict(data.get("verification", {})),
            rollback_plan=data.get("rollback_plan", ""),
            stop_conditions=data.get("stop_conditions", []),
            status=ChunkStatus(data.get("status", "pending")),
            priority=data.get("priority", 0),
            estimated_tokens=data.get("estimated_tokens", 0),
            dependencies=data.get("dependencies", []),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "Chunk":
        return cls.from_dict(json.loads(json_str))
    
    def get_all_allowed_paths(self) -> List[str]:
        """Get all paths this chunk is allowed to touch."""
        paths = []
        paths.extend(self.allowed_files.get("add", []))
        paths.extend(self.allowed_files.get("modify", []))
        paths.extend(self.allowed_files.get("delete_candidates", []))
        return paths


@dataclass
class ChunkPlan:
    """Complete chunk plan for an architecture implementation."""
    plan_id: str
    job_id: str
    arch_id: str
    arch_version: int
    spec_id: str
    spec_hash: str
    
    chunks: List[Chunk] = field(default_factory=list)
    
    # Metadata
    created_at: str = ""
    total_estimated_tokens: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "job_id": self.job_id,
            "arch_id": self.arch_id,
            "arch_version": self.arch_version,
            "spec_id": self.spec_id,
            "spec_hash": self.spec_hash,
            "chunks": [c.to_dict() for c in self.chunks],
            "created_at": self.created_at,
            "total_estimated_tokens": self.total_estimated_tokens,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkPlan":
        return cls(
            plan_id=data.get("plan_id", ""),
            job_id=data.get("job_id", ""),
            arch_id=data.get("arch_id", ""),
            arch_version=data.get("arch_version", 1),
            spec_id=data.get("spec_id", ""),
            spec_hash=data.get("spec_hash", ""),
            chunks=[Chunk.from_dict(c) for c in data.get("chunks", [])],
            created_at=data.get("created_at", ""),
            total_estimated_tokens=data.get("total_estimated_tokens", 0),
        )


# =============================================================================
# Block 8: Diff Boundary Result
# =============================================================================

@dataclass
class BoundaryViolation:
    """A file touched outside allowed boundaries."""
    file_path: str
    action: str  # "added", "modified", "deleted"
    reason: str  # Why it's a violation
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiffCheckResult:
    """Result of checking implementation diff against chunk boundaries."""
    passed: bool
    violations: List[BoundaryViolation] = field(default_factory=list)
    files_added: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    files_deleted: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": [v.to_dict() for v in self.violations],
            "files_added": self.files_added,
            "files_modified": self.files_modified,
            "files_deleted": self.files_deleted,
        }


# =============================================================================
# Block 9: Verification Result
# =============================================================================

@dataclass
class CommandResult:
    """Result of running a verification command."""
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    passed: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    """Result of verification gate for a chunk."""
    chunk_id: str
    status: VerificationStatus
    
    # Individual command results
    command_results: List[CommandResult] = field(default_factory=list)
    
    # Summary
    tests_passed: int = 0
    tests_failed: int = 0
    lint_errors: int = 0
    type_errors: int = 0
    
    # Evidence paths
    evidence_paths: List[str] = field(default_factory=list)
    
    # Legacy tracking (existing failures not caused by this chunk)
    legacy_failures: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "status": self.status.value if isinstance(self.status, VerificationStatus) else self.status,
            "command_results": [r.to_dict() for r in self.command_results],
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "lint_errors": self.lint_errors,
            "type_errors": self.type_errors,
            "evidence_paths": self.evidence_paths,
            "legacy_failures": self.legacy_failures,
        }


# =============================================================================
# Block 10-11: Quarantine Schema
# =============================================================================

@dataclass
class StaticEvidence:
    """Static analysis evidence for quarantine decision."""
    rg_references: int = 0  # Count of references found by ripgrep
    import_count: int = 0   # Count of imports of this file
    config_references: int = 0  # References in config files
    last_modified: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DynamicEvidence:
    """Dynamic analysis evidence for quarantine decision."""
    tests_passed: bool = False
    smoke_boot_passed: bool = False
    import_walk_passed: bool = False
    coverage_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QuarantineCandidate:
    """A file candidate for quarantine."""
    file_path: str
    reason: QuarantineReason
    confidence: float  # 0.0 to 1.0
    
    static_evidence: StaticEvidence = field(default_factory=StaticEvidence)
    dynamic_evidence: DynamicEvidence = field(default_factory=DynamicEvidence)
    
    # Status
    quarantined: bool = False
    quarantine_path: Optional[str] = None  # Where it was moved
    deleted: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "reason": self.reason.value if isinstance(self.reason, QuarantineReason) else self.reason,
            "confidence": self.confidence,
            "static_evidence": self.static_evidence.to_dict(),
            "dynamic_evidence": self.dynamic_evidence.to_dict(),
            "quarantined": self.quarantined,
            "quarantine_path": self.quarantine_path,
            "deleted": self.deleted,
        }


@dataclass
class QuarantineReport:
    """Report of quarantine analysis and actions."""
    report_id: str
    job_id: str
    
    candidates: List[QuarantineCandidate] = field(default_factory=list)
    
    # Verification after quarantine
    repo_still_passes: bool = False
    verification_evidence: Optional[VerificationResult] = None
    
    # Metadata
    created_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "job_id": self.job_id,
            "candidates": [c.to_dict() for c in self.candidates],
            "repo_still_passes": self.repo_still_passes,
            "verification_evidence": self.verification_evidence.to_dict() if self.verification_evidence else None,
            "created_at": self.created_at,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass
class DeletionReport:
    """Report of file deletions after quarantine approval."""
    report_id: str
    job_id: str
    quarantine_report_id: str
    
    deleted_files: List[str] = field(default_factory=list)
    deletion_evidence: Dict[str, str] = field(default_factory=dict)  # file -> reason
    
    # Verification after deletion
    repo_still_passes: bool = False
    verification_evidence: Optional[VerificationResult] = None
    
    # Approval
    approved_by: str = ""  # "user" or "auto"
    approved_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "job_id": self.job_id,
            "quarantine_report_id": self.quarantine_report_id,
            "deleted_files": self.deleted_files,
            "deletion_evidence": self.deletion_evidence,
            "repo_still_passes": self.repo_still_passes,
            "verification_evidence": self.verification_evidence.to_dict() if self.verification_evidence else None,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
        }


# =============================================================================
# Block 12: Replay Pack Schema
# =============================================================================

@dataclass
class SamplingParams:
    """LLM sampling parameters for deterministic replay."""
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    max_tokens: int = 4096
    seed: Optional[int] = None  # For providers that support it
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SamplingParams":
        return cls(
            temperature=data.get("temperature", 0.0),
            top_p=data.get("top_p", 1.0),
            top_k=data.get("top_k"),
            max_tokens=data.get("max_tokens", 4096),
            seed=data.get("seed"),
        )


@dataclass
class StageConfig:
    """Configuration for a single stage in the pipeline."""
    model_id: str
    provider_id: str
    sampling: SamplingParams = field(default_factory=SamplingParams)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "sampling": self.sampling.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageConfig":
        return cls(
            model_id=data.get("model_id", ""),
            provider_id=data.get("provider_id", ""),
            sampling=SamplingParams.from_dict(data.get("sampling", {})),
        )


@dataclass
class ReplayPack:
    """Deterministic replay bundle for a job.
    
    Contains everything needed to replay decisions stage-by-stage.
    Includes full model IDs AND sampling parameters for true determinism.
    """
    pack_id: str
    job_id: str
    created_at: str
    
    # Artifact references
    spec_path: str = ""
    arch_path: str = ""
    critique_paths: List[str] = field(default_factory=list)
    plan_path: str = ""
    
    # Ledger
    ledger_path: str = ""
    
    # Model identifiers per stage (legacy, kept for compatibility)
    model_versions: Dict[str, str] = field(default_factory=dict)
    
    # Full stage configs with sampling params (new)
    stage_configs: Dict[str, StageConfig] = field(default_factory=dict)
    
    # Verification outputs
    verification_paths: List[str] = field(default_factory=list)
    
    # Tool commands executed
    commands_log_path: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "job_id": self.job_id,
            "created_at": self.created_at,
            "spec_path": self.spec_path,
            "arch_path": self.arch_path,
            "critique_paths": self.critique_paths,
            "plan_path": self.plan_path,
            "ledger_path": self.ledger_path,
            "model_versions": self.model_versions,
            "stage_configs": {k: v.to_dict() for k, v in self.stage_configs.items()},
            "verification_paths": self.verification_paths,
            "commands_log_path": self.commands_log_path,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


__all__ = [
    # Enums
    "ChunkStatus",
    "VerificationStatus",
    "FileAction",
    "QuarantineReason",
    # Block 7: Chunks
    "ChunkStep",
    "ChunkVerification",
    "Chunk",
    "ChunkPlan",
    # Block 8: Diff
    "BoundaryViolation",
    "DiffCheckResult",
    # Block 9: Verification
    "CommandResult",
    "VerificationResult",
    # Block 10-11: Quarantine
    "StaticEvidence",
    "DynamicEvidence",
    "QuarantineCandidate",
    "QuarantineReport",
    "DeletionReport",
    # Block 12: Replay
    "SamplingParams",
    "StageConfig",
    "ReplayPack",
]
