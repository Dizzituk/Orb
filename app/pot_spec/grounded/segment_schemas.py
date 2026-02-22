# FILE: app/pot_spec/grounded/segment_schemas.py
"""
Pipeline Segmentation Data Models (v1.0)

Pydantic-style dataclasses for the segmentation system. These models define
the structure of segment specs, manifests, interface contracts, and grounding
data used when SpecGate decomposes large jobs into independent segments.

Design principle: Every model is JSON-serialisable via to_dict()/from_dict()
for disk persistence, consistent with spec_models.py conventions.

Classes:
--------
- SegmentStatus: Enum for segment execution state (Phase 2 — defined here for schema completeness)
- InterfaceContract: What a segment exposes or consumes
- VerifiedFile: A file confirmed to exist via host filesystem check
- InterfaceRead: Signature data read from a boundary file
- StaleEntry: Architecture map entry that doesn't match reality
- CreateTarget: File that needs to be created by a segment
- GroundingData: Aggregate verification results for a segment
- SegmentSpec: Self-contained segment specification
- SegmentManifest: Top-level manifest with segment ordering and contracts

Version Notes:
-------------
v1.0 (2026-02-08): Initial implementation — Phase 1 of Pipeline Segmentation
    - All models for manifest + segment spec + grounding data
    - SegmentStatus enum defined but NOT used until Phase 2 (orchestrator)
    - JSON round-trip via to_dict()/from_dict()
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from app.pot_spec.grounded._segment_schemas_utils_1 import CreateTarget, InterfaceContract, InterfaceRead, SEGMENT_SCHEMAS_BUILD_ID, SegmentManifest, SegmentStatus, StaleEntry, VerifiedFile

logger = logging.getLogger(__name__)
print(f"[SEGMENT_SCHEMAS_LOADED] BUILD_ID={SEGMENT_SCHEMAS_BUILD_ID}")


# =============================================================================
# ENUMS
# =============================================================================


# =============================================================================
# INTERFACE CONTRACTS
# =============================================================================


# =============================================================================
# GROUNDING DATA — File Verification Results
# =============================================================================


@dataclass
class GroundingData:
    """
    Aggregate verification results for a segment.
    
    Produced by the file verifier after checking the architecture map
    against the host filesystem. Each segment spec carries its own
    grounding data so downstream stages know exactly what's real.
    """
    verified_files: List[VerifiedFile] = field(default_factory=list)
    interface_reads: List[InterfaceRead] = field(default_factory=list)
    stale_entries: List[StaleEntry] = field(default_factory=list)
    create_targets: List[CreateTarget] = field(default_factory=list)
    new_files: List[str] = field(default_factory=list)  # Files on disk but not in arch map
    verification_errors: List[str] = field(default_factory=list)  # Files that couldn't be read

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verified_files": [v.to_dict() for v in self.verified_files],
            "interface_reads": [i.to_dict() for i in self.interface_reads],
            "stale_entries": [s.to_dict() for s in self.stale_entries],
            "create_targets": [c.to_dict() for c in self.create_targets],
            "new_files": self.new_files,
            "verification_errors": self.verification_errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroundingData":
        return cls(
            verified_files=[VerifiedFile.from_dict(v) for v in data.get("verified_files", [])],
            interface_reads=[InterfaceRead.from_dict(i) for i in data.get("interface_reads", [])],
            stale_entries=[StaleEntry.from_dict(s) for s in data.get("stale_entries", [])],
            create_targets=[CreateTarget.from_dict(c) for c in data.get("create_targets", [])],
            new_files=data.get("new_files", []),
            verification_errors=data.get("verification_errors", []),
        )

    def has_errors(self) -> bool:
        """True if any verification errors were encountered."""
        return bool(self.verification_errors)

    def summary(self) -> str:
        """Human-readable summary of verification results."""
        parts = [
            f"{len(self.verified_files)} verified",
            f"{len(self.interface_reads)} interfaces read",
            f"{len(self.stale_entries)} stale",
            f"{len(self.create_targets)} to create",
            f"{len(self.new_files)} new (not in map)",
            f"{len(self.verification_errors)} errors",
        ]
        return f"GroundingData({', '.join(parts)})"


# =============================================================================
# SEGMENT SPEC
# =============================================================================

@dataclass
class SegmentSpec:
    """
    Self-contained segment specification.
    
    Each segment spec is a complete job description that can be processed
    through the existing pipeline (Critical Pipeline → Critique → Overwatcher
    → Implementer) without any knowledge of sibling segments. The only
    cross-segment references are via exposes/consumes interface contracts.
    
    Design Spec Section 3.3 — Per-Segment Spec Files (seg-XX-spec.json).
    """
    # Identity
    segment_id: str                     # e.g. "seg-01-backend-services"
    title: str                          # Human-readable description
    parent_spec_id: Optional[str] = None  # Reference to the full SPoT spec

    # Requirements — subset of parent spec requirements this segment fulfils
    requirements: List[str] = field(default_factory=list)

    # File scope — what this segment creates or modifies (explicit list)
    file_scope: List[str] = field(default_factory=list)

    # Evidence files — files this segment needs to READ for context
    evidence_files: List[str] = field(default_factory=list)

    # Dependencies — segment_ids that must complete first
    dependencies: List[str] = field(default_factory=list)

    # Interface contracts
    exposes: Optional[InterfaceContract] = None   # What this segment creates for downstream
    consumes: Optional[InterfaceContract] = None   # What this segment needs from upstream

    # Acceptance criteria — subset of parent spec criteria for this segment
    acceptance_criteria: List[str] = field(default_factory=list)

    # Complexity estimate
    estimated_files: int = 0  # Count of files to create/modify

    # Grounding — verified evidence from file_verifier
    grounding_data: Optional[GroundingData] = None

    # v6.1: Deterministic refactor flag — set by refactor_pipeline
    deterministic_refactor: bool = False

    # v6.1 FIX 13: Per-segment deterministic source.
    # When this segment is part of a deterministic refactor, this is the
    # source monolith path that THIS segment's files came from.
    deterministic_source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "title": self.title,
            "parent_spec_id": self.parent_spec_id,
            "requirements": self.requirements,
            "file_scope": self.file_scope,
            "evidence_files": self.evidence_files,
            "dependencies": self.dependencies,
            "exposes": self.exposes.to_dict() if self.exposes else None,
            "consumes": self.consumes.to_dict() if self.consumes else None,
            "acceptance_criteria": self.acceptance_criteria,
            "estimated_files": self.estimated_files,
            "grounding_data": self.grounding_data.to_dict() if self.grounding_data else None,
            "deterministic_refactor": self.deterministic_refactor,
            "deterministic_source": self.deterministic_source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SegmentSpec":
        exposes_data = data.get("exposes")
        consumes_data = data.get("consumes")
        grounding_data = data.get("grounding_data")
        return cls(
            segment_id=data.get("segment_id", ""),
            title=data.get("title", ""),
            parent_spec_id=data.get("parent_spec_id"),
            requirements=data.get("requirements", []),
            file_scope=data.get("file_scope", []),
            evidence_files=data.get("evidence_files", []),
            dependencies=data.get("dependencies", []),
            exposes=InterfaceContract.from_dict(exposes_data) if exposes_data else None,
            consumes=InterfaceContract.from_dict(consumes_data) if consumes_data else None,
            acceptance_criteria=data.get("acceptance_criteria", []),
            estimated_files=data.get("estimated_files", 0),
            grounding_data=GroundingData.from_dict(grounding_data) if grounding_data else None,
            deterministic_refactor=data.get("deterministic_refactor", False),
            deterministic_source=data.get("deterministic_source"),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "SegmentSpec":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# SEGMENT MANIFEST
# =============================================================================
