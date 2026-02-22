from __future__ import annotations
import json
from app.overwatcher.schemas import BoundaryViolation, Chunk, OverrideType, QuarantineCandidate, StageConfig, StrikeEvent, VerificationResult
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class HoleType(str, Enum):
    """Spec Gate hole classification."""
    MISSING_INFO = "missing_info"
    CONTRADICTION = "contradiction"
    AMBIGUITY = "ambiguity"
    SAFETY_GAP = "safety_gap"

@dataclass
class StrikeState:
    """Per-job strike tracking state.
    
    Persisted to: jobs/<job_id>/governance/strike_state.json
    """
    job_id: str
    created_at: str = ""
    updated_at: str = ""
    strikes_by_error_sig: Dict[str, int] = field(default_factory=dict)
    strikes_by_spec_hole_sig: Dict[str, int] = field(default_factory=dict)
    history: List[StrikeEvent] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "strikes_by_error_sig": self.strikes_by_error_sig,
            "strikes_by_spec_hole_sig": self.strikes_by_spec_hole_sig,
            "history": [e.to_dict() for e in self.history],
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrikeState":
        return cls(
            job_id=data.get("job_id", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            strikes_by_error_sig=data.get("strikes_by_error_sig", {}),
            strikes_by_spec_hole_sig=data.get("strikes_by_spec_hole_sig", {}),
            history=[StrikeEvent.from_dict(e) for e in data.get("history", [])],
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "StrikeState":
        return cls.from_dict(json.loads(json_str))
    
    def save(self, jobs_dir: Path) -> Path:
        """Save strike state to jobs/<job_id>/governance/strike_state.json"""
        governance_dir = jobs_dir / self.job_id / "governance"
        governance_dir.mkdir(parents=True, exist_ok=True)
        
        state_file = governance_dir / "strike_state.json"
        with open(state_file, "w") as f:
            f.write(self.to_json())
        
        return state_file
    
    @classmethod
    def load(cls, jobs_dir: Path, job_id: str) -> Optional["StrikeState"]:
        """Load strike state from jobs/<job_id>/governance/strike_state.json"""
        state_file = jobs_dir / job_id / "governance" / "strike_state.json"
        if not state_file.exists():
            return None
        
        with open(state_file) as f:
            return cls.from_json(f.read())

@dataclass
class HumanOverride:
    """Human override token for governance decisions."""
    override_type: OverrideType
    reason: str
    user_id: str
    timestamp: str
    target_signature: Optional[str] = None
    acknowledged_incident_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "override_type": self.override_type.value if isinstance(self.override_type, OverrideType) else self.override_type,
            "reason": self.reason,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "target_signature": self.target_signature,
            "acknowledged_incident_id": self.acknowledged_incident_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HumanOverride":
        return cls(
            override_type=OverrideType(data.get("override_type", "force_continue")),
            reason=data.get("reason", ""),
            user_id=data.get("user_id", ""),
            timestamp=data.get("timestamp", ""),
            target_signature=data.get("target_signature"),
            acknowledged_incident_id=data.get("acknowledged_incident_id"),
        )

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
            arch_version=data.get("arch_version", 0),
            spec_id=data.get("spec_id", ""),
            spec_hash=data.get("spec_hash", ""),
            chunks=[Chunk.from_dict(c) for c in data.get("chunks", [])],
            created_at=data.get("created_at", ""),
            total_estimated_tokens=data.get("total_estimated_tokens", 0),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "ChunkPlan":
        return cls.from_dict(json.loads(json_str))

@dataclass
class DiffCheckResult:
    """Result of checking a diff against chunk boundaries."""
    allowed: bool
    violations: List[BoundaryViolation] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "violations": [v.to_dict() for v in self.violations],
        }

@dataclass
class QuarantineReport:
    """Report of quarantine analysis and actions."""
    report_id: str
    job_id: str
    
    candidates: List[QuarantineCandidate] = field(default_factory=list)
    
    repo_still_passes: bool = False
    verification_evidence: Optional[VerificationResult] = None
    
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
    deletion_evidence: Dict[str, str] = field(default_factory=dict)
    
    repo_still_passes: bool = False
    verification_evidence: Optional[VerificationResult] = None
    
    approved_by: str = ""
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

@dataclass
class ReplayPack:
    """Deterministic replay bundle for a job."""
    pack_id: str
    job_id: str
    created_at: str
    
    spec_path: str = ""
    arch_path: str = ""
    critique_paths: List[str] = field(default_factory=list)
    plan_path: str = ""
    
    ledger_path: str = ""
    
    model_versions: Dict[str, str] = field(default_factory=dict)
    stage_configs: Dict[str, StageConfig] = field(default_factory=dict)
    
    verification_paths: List[str] = field(default_factory=list)
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
