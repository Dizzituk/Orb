from __future__ import annotations
import json
import logging
from app.pot_spec.grounded.segment_schemas import SegmentSpec, logger
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


SEGMENT_SCHEMAS_BUILD_ID = "2026-02-08-v1.0-initial"

class SegmentStatus(str, Enum):
    """
    Execution state of a segment.
    
    NOTE: This enum is defined for schema completeness but is NOT used in
    Phase 1. It will be consumed by the orchestrator in Phase 2 when
    segment execution state is tracked via state.json.
    """
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"       # v3.0: Architecture generated + critique passed, awaiting execution
    COMPLETE = "complete"
    FAILED = "failed"
    BLOCKED = "blocked"

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

    # v5.18: External consumer files deferred to post-recon
    # Files that only need import-path updates after a file->package refactor.
    # These are NOT included in any segment's file_scope.
    deferred_consumer_files: List[str] = field(default_factory=list)

    # v6.1 FIX 13: Deterministic refactor source files (multi-file support).
    # When set, indicates this manifest was produced by the deterministic
    # refactor pipeline. Each entry is a source monolith path.
    deterministic_sources: List[str] = field(default_factory=list)

    @property
    def deterministic_source(self) -> Optional[str]:
        """Backward-compat: return first source or None."""
        return self.deterministic_sources[0] if self.deterministic_sources else None

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
            "deferred_consumer_files": self.deferred_consumer_files,
            "deterministic_sources": self.deterministic_sources,
            # Backward compat for consumers reading manifest.json
            "deterministic_source": self.deterministic_source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SegmentManifest":
        # v6.1 FIX 13: Read list form, fall back to single-string form
        _sources = data.get("deterministic_sources", [])
        if not _sources:
            _single = data.get("deterministic_source")
            if _single:
                _sources = [_single]
        return cls(
            parent_spec_id=data.get("parent_spec_id"),
            parent_spec_hash=data.get("parent_spec_hash"),
            deferred_consumer_files=data.get("deferred_consumer_files", []),
            segments=[SegmentSpec.from_dict(s) for s in data.get("segments", [])],
            requirement_map=data.get("requirement_map", {}),
            total_segments=data.get("total_segments", 0),
            total_files=data.get("total_files", 0),
            generated_at=data.get("generated_at", ""),
            manifest_version=data.get("manifest_version", "1.0"),
            deterministic_sources=_sources,
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
