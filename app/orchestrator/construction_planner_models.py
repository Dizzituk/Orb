# FILE: app/orchestrator/construction_planner_models.py
"""
Construction Planner — Data Models (Stage 3).

Defines the phase decomposition data structures used when a project
is too large to build in a single pass. Each phase contains segments
that execute in the segment loop, gated by Phase Checkout.

v1.0 (2026-02-14): Initial implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class PhaseContract:
    """What a phase promises to deliver for downstream phases."""
    phase_id: str
    exports: List[str] = field(default_factory=list)  # file paths produced
    interfaces: List[str] = field(default_factory=list)  # public APIs exposed
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_id": self.phase_id,
            "exports": self.exports,
            "interfaces": self.interfaces,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhaseContract":
        return cls(
            phase_id=data.get("phase_id", ""),
            exports=data.get("exports", []),
            interfaces=data.get("interfaces", []),
            description=data.get("description", ""),
        )


@dataclass
class PhaseDefinition:
    """A single phase within a multi-phase construction plan."""
    phase_id: str
    phase_number: int
    title: str = ""
    description: str = ""
    file_scope: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)  # phase_ids this needs
    contract: Optional[PhaseContract] = None  # what this delivers downstream
    spec_section: str = ""  # the spec subset for this phase
    estimated_segments: int = 1

    # Execution state
    status: str = "pending"  # pending, running, complete, failed
    manifest_path: Optional[str] = None
    checkout_status: Optional[str] = None  # pass, fail, error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase_id": self.phase_id,
            "phase_number": self.phase_number,
            "title": self.title,
            "description": self.description,
            "file_scope": self.file_scope,
            "depends_on": self.depends_on,
            "contract": self.contract.to_dict() if self.contract else None,
            "spec_section": self.spec_section[:200] + "..." if len(self.spec_section) > 200 else self.spec_section,
            "estimated_segments": self.estimated_segments,
            "status": self.status,
            "manifest_path": self.manifest_path,
            "checkout_status": self.checkout_status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PhaseDefinition":
        contract = None
        if data.get("contract"):
            contract = PhaseContract.from_dict(data["contract"])
        return cls(
            phase_id=data.get("phase_id", ""),
            phase_number=data.get("phase_number", 0),
            title=data.get("title", ""),
            description=data.get("description", ""),
            file_scope=data.get("file_scope", []),
            depends_on=data.get("depends_on", []),
            contract=contract,
            spec_section=data.get("spec_section", ""),
            estimated_segments=data.get("estimated_segments", 1),
            status=data.get("status", "pending"),
            manifest_path=data.get("manifest_path"),
            checkout_status=data.get("checkout_status"),
        )


@dataclass
class ConstructionPlan:
    """Complete multi-phase construction plan for a project."""
    job_id: str
    total_phases: int = 1
    phases: List[PhaseDefinition] = field(default_factory=list)
    is_multi_phase: bool = False
    reasoning: str = ""  # why multi-phase was chosen

    # Execution state
    current_phase: int = 0  # 0 = not started
    status: str = "pending"  # pending, running, complete, failed, partial

    # Metadata
    created_at: str = ""
    total_files: int = 0
    estimated_total_segments: int = 0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "total_phases": self.total_phases,
            "phases": [p.to_dict() for p in self.phases],
            "is_multi_phase": self.is_multi_phase,
            "reasoning": self.reasoning,
            "current_phase": self.current_phase,
            "status": self.status,
            "created_at": self.created_at,
            "total_files": self.total_files,
            "estimated_total_segments": self.estimated_total_segments,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstructionPlan":
        phases = [PhaseDefinition.from_dict(p) for p in data.get("phases", [])]
        return cls(
            job_id=data.get("job_id", ""),
            total_phases=data.get("total_phases", 1),
            phases=phases,
            is_multi_phase=data.get("is_multi_phase", False),
            reasoning=data.get("reasoning", ""),
            current_phase=data.get("current_phase", 0),
            status=data.get("status", "pending"),
            created_at=data.get("created_at", ""),
            total_files=data.get("total_files", 0),
            estimated_total_segments=data.get("estimated_total_segments", 0),
        )

    @property
    def next_phase(self) -> Optional[PhaseDefinition]:
        """Get the next pending phase, or None if all done."""
        for p in self.phases:
            if p.status == "pending":
                return p
        return None

    @property
    def all_complete(self) -> bool:
        return all(p.status == "complete" for p in self.phases)
