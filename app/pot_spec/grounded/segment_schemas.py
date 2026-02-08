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

logger = logging.getLogger(__name__)

SEGMENT_SCHEMAS_BUILD_ID = "2026-02-08-v1.0-initial"
print(f"[SEGMENT_SCHEMAS_LOADED] BUILD_ID={SEGMENT_SCHEMAS_BUILD_ID}")


# =============================================================================
# ENUMS
# =============================================================================

class SegmentStatus(str, Enum):
    """
    Execution state of a segment.
    
    NOTE: This enum is defined for schema completeness but is NOT used in
    Phase 1. It will be consumed by the orchestrator in Phase 2 when
    segment execution state is tracked via state.json.
    """
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"
    BLOCKED = "blocked"


# =============================================================================
# INTERFACE CONTRACTS
# =============================================================================

@dataclass
class InterfaceContract:
    """
    What a segment exposes to or consumes from sibling segments.
    
    Interface contracts define the boundary between segments. A segment
    that has dependents MUST declare what it exposes. A segment that
    depends on others declares what it consumes.
    
    Examples:
        - class_names: ["TranscriptionService", "LocalAIConfig"]
        - method_signatures: ["async def transcribe(audio: bytes) -> str"]
        - endpoint_paths: ["POST /voice/transcribe", "GET /voice/status"]
        - export_names: ["VoiceInput", "useVoiceRecorder"]
    """
    class_names: List[str] = field(default_factory=list)
    method_signatures: List[str] = field(default_factory=list)
    endpoint_paths: List[str] = field(default_factory=list)
    export_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_names": self.class_names,
            "method_signatures": self.method_signatures,
            "endpoint_paths": self.endpoint_paths,
            "export_names": self.export_names,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InterfaceContract":
        return cls(
            class_names=data.get("class_names", []),
            method_signatures=data.get("method_signatures", []),
            endpoint_paths=data.get("endpoint_paths", []),
            export_names=data.get("export_names", []),
        )

    def is_empty(self) -> bool:
        """True if no contracts are declared."""
        return (
            not self.class_names
            and not self.method_signatures
            and not self.endpoint_paths
            and not self.export_names
        )


# =============================================================================
# GROUNDING DATA — File Verification Results
# =============================================================================

@dataclass
class VerifiedFile:
    """A file confirmed to exist on the host filesystem."""
    path: str
    last_modified: Optional[str] = None  # ISO format timestamp
    size_bytes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "last_modified": self.last_modified,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerifiedFile":
        return cls(
            path=data.get("path", ""),
            last_modified=data.get("last_modified"),
            size_bytes=data.get("size_bytes"),
        )


@dataclass
class InterfaceRead:
    """Signature data read from a boundary file (class/function signatures)."""
    path: str
    signatures: List[str] = field(default_factory=list)  # e.g. ["class TranscriptionService:", "def transcribe(...)"]
    raw_excerpt: Optional[str] = None  # First ~500 chars of the file
    read_ok: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "signatures": self.signatures,
            "raw_excerpt": self.raw_excerpt,
            "read_ok": self.read_ok,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InterfaceRead":
        return cls(
            path=data.get("path", ""),
            signatures=data.get("signatures", []),
            raw_excerpt=data.get("raw_excerpt"),
            read_ok=data.get("read_ok", True),
        )


@dataclass
class StaleEntry:
    """Architecture map entry that doesn't match the actual file on disk."""
    path: str
    map_claimed: str    # What the architecture map said (e.g. "class FooService")
    actual_found: str   # What was actually found (e.g. "class FooHandler")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "map_claimed": self.map_claimed,
            "actual_found": self.actual_found,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StaleEntry":
        return cls(
            path=data.get("path", ""),
            map_claimed=data.get("map_claimed", ""),
            actual_found=data.get("actual_found", ""),
        )


@dataclass
class CreateTarget:
    """A file that needs to be created by a segment."""
    path: str
    must_expose: List[str] = field(default_factory=list)  # Interfaces the new file must provide
    reason: str = ""  # Why this file needs to be created

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "must_expose": self.must_expose,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CreateTarget":
        return cls(
            path=data.get("path", ""),
            must_expose=data.get("must_expose", []),
            reason=data.get("reason", ""),
        )


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

@dataclass
class SegmentManifest:
    """
    Top-level manifest for a segmented job.
    
    Contains the segment ordering, dependency graph, interface contracts,
    and requirement-to-segment mapping. The orchestrator (Phase 2) reads
    this to process segments in the correct order.
    
    Design Spec Section 3.3 — The Manifest (manifest.json).
    """
    # Parent spec reference
    parent_spec_id: Optional[str] = None
    parent_spec_hash: Optional[str] = None

    # Segment list — topologically sorted by dependency order
    segments: List[SegmentSpec] = field(default_factory=list)

    # Requirement-to-segment mapping
    # Key: requirement string, Value: list of segment_ids that address it
    requirement_map: Dict[str, List[str]] = field(default_factory=dict)

    # Metadata
    total_segments: int = 0
    total_files: int = 0
    generated_at: str = ""
    manifest_version: str = "1.0"

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()
        if not self.total_segments:
            self.total_segments = len(self.segments)
        if not self.total_files:
            self.total_files = sum(s.estimated_files for s in self.segments)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_spec_id": self.parent_spec_id,
            "parent_spec_hash": self.parent_spec_hash,
            "segments": [s.to_dict() for s in self.segments],
            "requirement_map": self.requirement_map,
            "total_segments": self.total_segments,
            "total_files": self.total_files,
            "generated_at": self.generated_at,
            "manifest_version": self.manifest_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SegmentManifest":
        return cls(
            parent_spec_id=data.get("parent_spec_id"),
            parent_spec_hash=data.get("parent_spec_hash"),
            segments=[SegmentSpec.from_dict(s) for s in data.get("segments", [])],
            requirement_map=data.get("requirement_map", {}),
            total_segments=data.get("total_segments", 0),
            total_files=data.get("total_files", 0),
            generated_at=data.get("generated_at", ""),
            manifest_version=data.get("manifest_version", "1.0"),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "SegmentManifest":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def get_segment(self, segment_id: str) -> Optional[SegmentSpec]:
        """Look up a segment by ID."""
        for s in self.segments:
            if s.segment_id == segment_id:
                return s
        return None

    def get_execution_order(self) -> List[str]:
        """Return segment IDs in dependency-safe execution order (already sorted)."""
        return [s.segment_id for s in self.segments]

    def get_independent_segments(self) -> List[List[str]]:
        """
        Return groups of segments that can execute in parallel.
        
        Segments with the same set of completed dependencies can run
        concurrently. This is a future optimisation — Phase 1 runs
        segments sequentially.
        """
        completed: set = set()
        groups: List[List[str]] = []

        remaining = list(self.segments)
        while remaining:
            # Find all segments whose dependencies are fully satisfied
            ready = [
                s for s in remaining
                if all(dep in completed for dep in s.dependencies)
            ]
            if not ready:
                # Should not happen if DAG validation passed, but safety check
                logger.error(
                    "[segment_schemas] get_independent_segments: deadlock — "
                    "remaining segments have unsatisfied deps: %s",
                    [s.segment_id for s in remaining]
                )
                break

            group_ids = [s.segment_id for s in ready]
            groups.append(group_ids)
            completed.update(group_ids)
            remaining = [s for s in remaining if s.segment_id not in completed]

        return groups

    def summary(self) -> str:
        """Human-readable manifest summary."""
        seg_list = ", ".join(s.segment_id for s in self.segments)
        return (
            f"SegmentManifest(v{self.manifest_version}, "
            f"{self.total_segments} segments, "
            f"{self.total_files} files, "
            f"order=[{seg_list}])"
        )
