"""
Skeleton Contracts — Deterministic Interface Binding for Segments.

v5.6 of Pipeline Evolution.

Generates interface contracts DETERMINISTICALLY from the manifest alone.
Zero LLM calls. Runs between segmentation and architecture generation.

For each segment, the skeleton defines:
  - File scope constraint (ONLY these files may be touched)
  - Export contracts (files that downstream segments depend on)
  - Import contracts (files from upstream segments this segment needs)
  - Cross-segment bindings (the dependency graph edges)

The contract markdown is injected into each segment's Critical Pipeline
prompt as a hard constraint, preventing:
  - Scope creep (touching files outside the segment's scope)
  - Phantom segments (referencing segments that don't exist)
  - Interface drift (inventing alternative imports)

v1.0 (2026-02-12): Initial implementation — deterministic skeleton.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SKELETON_CONTRACTS_BUILD_ID = "2026-02-14-v1.1-peer-segment-imports"
print(f"[SKELETON_CONTRACTS_LOADED] BUILD_ID={SKELETON_CONTRACTS_BUILD_ID}")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExportBinding:
    """A file that this segment creates and downstream segments depend on."""
    file_path: str
    consumed_by: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"file_path": self.file_path, "consumed_by": self.consumed_by}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExportBinding":
        return cls(
            file_path=data.get("file_path", ""),
            consumed_by=data.get("consumed_by", []),
        )


@dataclass
class SegmentSkeleton:
    """Skeleton contract for a single segment."""
    segment_id: str
    title: str = ""
    file_scope: List[str] = field(default_factory=list)
    exports: List[ExportBinding] = field(default_factory=list)
    imports_from: Dict[str, List[str]] = field(default_factory=dict)  # seg_id -> [file_paths]
    dependencies: List[str] = field(default_factory=list)
    peer_imports_from: Dict[str, List[str]] = field(default_factory=dict)  # v1.1: peer seg_id -> [file_paths]
    total_segments_in_job: int = 0
    all_segment_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "title": self.title,
            "file_scope": self.file_scope,
            "exports": [e.to_dict() for e in self.exports],
            "imports_from": self.imports_from,
            "dependencies": self.dependencies,
            "peer_imports_from": self.peer_imports_from,
            "total_segments_in_job": self.total_segments_in_job,
            "all_segment_ids": self.all_segment_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SegmentSkeleton":
        return cls(
            segment_id=data.get("segment_id", ""),
            title=data.get("title", ""),
            file_scope=data.get("file_scope", []),
            exports=[ExportBinding.from_dict(e) for e in data.get("exports", [])],
            imports_from=data.get("imports_from", {}),
            dependencies=data.get("dependencies", []),
            peer_imports_from=data.get("peer_imports_from", {}),
            total_segments_in_job=data.get("total_segments_in_job", 0),
            all_segment_ids=data.get("all_segment_ids", []),
        )


@dataclass
class SkeletonContractSet:
    """Complete skeleton contract set for a segmented job."""
    job_id: str
    total_segments: int = 0
    skeletons: List[SegmentSkeleton] = field(default_factory=list)
    cross_segment_bindings: List[Dict[str, str]] = field(default_factory=list)
    generated_at: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "total_segments": self.total_segments,
            "skeletons": [s.to_dict() for s in self.skeletons],
            "cross_segment_bindings": self.cross_segment_bindings,
            "generated_at": self.generated_at,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkeletonContractSet":
        return cls(
            job_id=data.get("job_id", ""),
            total_segments=data.get("total_segments", 0),
            skeletons=[SegmentSkeleton.from_dict(s) for s in data.get("skeletons", [])],
            cross_segment_bindings=data.get("cross_segment_bindings", []),
            generated_at=data.get("generated_at", ""),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "SkeletonContractSet":
        return cls.from_dict(json.loads(json_str))

    def format_contract_for_segment(self, segment_id: str) -> str:
        """
        Format the skeleton contract as markdown for injection into
        a segment's architecture generation prompt.

        This is the key output — it tells the Critical Pipeline LLM
        exactly what files this segment owns, what it exports, what
        it imports, and what the overall segment structure looks like.
        """
        skeleton = None
        for s in self.skeletons:
            if s.segment_id == segment_id:
                skeleton = s
                break

        if skeleton is None:
            return ""

        parts = []
        parts.append("## Skeleton Contract (BINDING — DO NOT VIOLATE)\n")

        # --- Job structure awareness ---
        parts.append(f"**This job has exactly {skeleton.total_segments_in_job} segments.** "
                     f"Do NOT reference any other segment numbers.\n")
        parts.append("Segment IDs in this job:")
        for sid in skeleton.all_segment_ids:
            marker = " ← (this segment)" if sid == segment_id else ""
            parts.append(f"  - `{sid}`{marker}")
        parts.append("")

        # --- File scope constraint ---
        parts.append("### File Scope Constraint\n")
        parts.append("**You may ONLY design architecture for these files. "
                     "Do NOT add, modify, or reference any other files in your file inventory.**\n")
        for fp in skeleton.file_scope:
            parts.append(f"  - `{fp}`")
        parts.append("")

        # --- Exports ---
        if skeleton.exports:
            parts.append("### This Segment EXPORTS\n")
            parts.append("The following files are consumed by downstream segments. "
                        "You MUST create/modify them with stable, importable interfaces.\n")
            for exp in skeleton.exports:
                consumers = ", ".join(f"`{c}`" for c in exp.consumed_by)
                parts.append(f"  - `{exp.file_path}` → consumed by {consumers}")
            parts.append("")

        # --- Imports ---
        if skeleton.imports_from:
            parts.append("### This Segment IMPORTS FROM\n")
            parts.append("These files are created by upstream segments. "
                        "When you need functionality from them, import from these exact paths.\n")
            for upstream_seg, files in skeleton.imports_from.items():
                parts.append(f"  From `{upstream_seg}`:")
                for fp in files:
                    parts.append(f"    - `{fp}`")
            parts.append("")

        # --- Peer imports (v1.1) ---
        if skeleton.peer_imports_from:
            parts.append("### Peer Segment Imports (OPTIONAL)\n")
            parts.append("These segments build before yours and are NOT your direct dependencies, ")
            parts.append("but their exported files are available for import if needed. ")
            parts.append("Using these can avoid unnecessary workarounds like callable injection ")
            parts.append("when a direct import would be simpler and preserve original signatures.\n")
            for peer_seg, files in skeleton.peer_imports_from.items():
                parts.append(f"  From `{peer_seg}`:")
                for fp in files:
                    parts.append(f"    - `{fp}`")
            parts.append("")

        # --- Dependencies ---
        if skeleton.dependencies:
            parts.append("### Dependencies\n")
            parts.append("This segment depends on these segments completing first:")
            for dep in skeleton.dependencies:
                parts.append(f"  - `{dep}`")
            parts.append("")

        # --- Package structure for imports ---
        _packages = set()
        _parent_files = []
        for fp in skeleton.file_scope:
            fp_norm = fp.replace("\\", "/")
            parts_list = fp_norm.split("/")
            if len(parts_list) >= 2:
                _pkg = "/".join(parts_list[:-1])
                _packages.add(_pkg)
            # Detect files in parent package vs sub-package
            if len(parts_list) >= 3:
                _parent_pkg = "/".join(parts_list[:-2])
                _packages.add(_parent_pkg)

        if len(_packages) > 1:
            parts.append("### Import Path Rules\n")
            parts.append("Files in this segment span multiple directory levels:")
            for _pkg in sorted(_packages):
                _pkg_files = [fp for fp in skeleton.file_scope if fp.replace('\\', '/').startswith(_pkg + '/')]
                if _pkg_files:
                    parts.append(f"  - `{_pkg}/`: {len(_pkg_files)} file(s)")
            parts.append("")
            parts.append("**Import rules**:")
            parts.append("- Files in the SAME directory use single-dot: `from .module import ...`")
            parts.append("- Files importing from a PARENT directory use double-dot: `from ..module import ...`")
            parts.append("- Files importing from a SIBLING sub-package use: `from ..subpkg.module import ...`")
            parts.append("")

        # --- Rules ---
        parts.append("### Rules\n")
        parts.append("1. Your file inventory MUST only contain files listed in File Scope Constraint above.")
        parts.append("2. Do NOT invent files, test files, or helper files outside the scope.")
        parts.append("3. Do NOT reference segment numbers that don't exist in this job.")
        parts.append(f"4. This job has {skeleton.total_segments_in_job} segments total — "
                     f"not more, not fewer.")
        parts.append("5. If you need to import from upstream segments, use the exact file paths listed above.")
        parts.append("6. Use correct relative import depth — see Import Path Rules above if present.")
        parts.append("")

        return "\n".join(parts)


# =============================================================================
# GENERATOR — Pure logic, no LLM calls
# =============================================================================

def generate_skeleton_contract(
    manifest_dict: Dict[str, Any],
    job_id: str,
) -> SkeletonContractSet:
    """
    Generate skeleton contracts deterministically from a segment manifest.

    Reads the manifest's segments, file_scopes, evidence_files, and
    dependencies to produce binding contracts for each segment.

    Zero LLM calls. Pure Python logic.
    """
    segments_raw = manifest_dict.get("segments", [])
    if not segments_raw:
        return SkeletonContractSet(job_id=job_id, total_segments=0)

    total_segments = len(segments_raw)
    all_seg_ids = [s.get("segment_id", "") for s in segments_raw]

    # Build a map: file_path -> owning segment_id
    file_to_segment: Dict[str, str] = {}
    for seg in segments_raw:
        seg_id = seg.get("segment_id", "")
        for fp in seg.get("file_scope", []):
            file_to_segment[fp] = seg_id

    # Build a map: segment_id -> evidence_files
    seg_evidence: Dict[str, List[str]] = {}
    for seg in segments_raw:
        seg_id = seg.get("segment_id", "")
        seg_evidence[seg_id] = seg.get("evidence_files", [])

    skeletons = []
    all_bindings = []

    for seg in segments_raw:
        seg_id = seg.get("segment_id", "")
        title = seg.get("title", "")
        file_scope = seg.get("file_scope", [])
        dependencies = seg.get("dependencies", [])

        # --- Determine exports ---
        # A file is "exported" if it appears in another segment's evidence_files
        exports = []
        for fp in file_scope:
            consumers = []
            for other_seg in segments_raw:
                other_id = other_seg.get("segment_id", "")
                if other_id == seg_id:
                    continue
                if fp in other_seg.get("evidence_files", []):
                    consumers.append(other_id)
            if consumers:
                exports.append(ExportBinding(file_path=fp, consumed_by=consumers))
                for consumer_id in consumers:
                    all_bindings.append({
                        "from_segment": seg_id,
                        "to_segment": consumer_id,
                        "file_path": fp,
                        "binding_type": "evidence_dependency",
                    })

        # --- Determine imports ---
        # Group evidence_files by which segment owns them
        imports_from: Dict[str, List[str]] = {}
        for ev_file in seg_evidence.get(seg_id, []):
            owning_seg = file_to_segment.get(ev_file)
            if owning_seg and owning_seg != seg_id:
                if owning_seg not in imports_from:
                    imports_from[owning_seg] = []
                if ev_file not in imports_from[owning_seg]:
                    imports_from[owning_seg].append(ev_file)

        # --- Determine peer imports (v1.1) ---
        # Peer = segments that build before this one (earlier in order) but
        # are NOT listed as direct dependencies. They share a common consumer
        # but don't depend on each other. Their exports are available for import.
        seg_index = all_seg_ids.index(seg_id) if seg_id in all_seg_ids else -1
        peer_imports_from: Dict[str, List[str]] = {}
        if seg_index > 0:
            for earlier_id in all_seg_ids[:seg_index]:
                if earlier_id in dependencies:
                    continue  # already in imports_from, not a peer
                if earlier_id in imports_from:
                    continue  # already imported via evidence
                # Find what this earlier segment exports
                for other_seg in segments_raw:
                    if other_seg.get("segment_id") == earlier_id:
                        for fp in other_seg.get("file_scope", []):
                            # Check if this file is consumed by anyone downstream
                            for consumer_seg in segments_raw:
                                if fp in consumer_seg.get("evidence_files", []):
                                    if earlier_id not in peer_imports_from:
                                        peer_imports_from[earlier_id] = []
                                    if fp not in peer_imports_from[earlier_id]:
                                        peer_imports_from[earlier_id].append(fp)
                                    break
                        break

        skeleton = SegmentSkeleton(
            segment_id=seg_id,
            title=title,
            file_scope=file_scope,
            exports=exports,
            imports_from=imports_from,
            dependencies=dependencies,
            peer_imports_from=peer_imports_from,
            total_segments_in_job=total_segments,
            all_segment_ids=all_seg_ids,
        )
        skeletons.append(skeleton)

    # Deduplicate bindings
    seen_bindings = set()
    unique_bindings = []
    for b in all_bindings:
        key = (b["from_segment"], b["to_segment"], b["file_path"])
        if key not in seen_bindings:
            seen_bindings.add(key)
            unique_bindings.append(b)

    contract_set = SkeletonContractSet(
        job_id=job_id,
        total_segments=total_segments,
        skeletons=skeletons,
        cross_segment_bindings=unique_bindings,
    )

    logger.info(
        "[skeleton_contracts] Generated: %d segments, %d bindings for job %s",
        total_segments, len(unique_bindings), job_id,
    )

    return contract_set


# =============================================================================
# PERSISTENCE
# =============================================================================

def save_skeleton_contract(contract_set: SkeletonContractSet, job_dir: str) -> str:
    """Save skeleton contracts to disk alongside the segment manifest."""
    segments_dir = os.path.join(job_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)
    path = os.path.join(segments_dir, "skeleton_contract.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(contract_set.to_json(indent=2))
    logger.info("[skeleton_contracts] Saved: %s", path)
    return path


def load_skeleton_contract(job_dir: str) -> Optional[SkeletonContractSet]:
    """Load skeleton contracts from disk. Returns None if not found."""
    path = os.path.join(job_dir, "segments", "skeleton_contract.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return SkeletonContractSet.from_json(f.read())
    except Exception as e:
        logger.warning("[skeleton_contracts] Failed to load: %s", e)
        return None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ExportBinding",
    "SegmentSkeleton",
    "SkeletonContractSet",
    "generate_skeleton_contract",
    "save_skeleton_contract",
    "load_skeleton_contract",
    "SKELETON_CONTRACTS_BUILD_ID",
]
