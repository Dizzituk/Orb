from __future__ import annotations
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


SKELETON_CONTRACTS_BUILD_ID = "2026-02-18-v2.4-do-not-define-prohibition"

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


# Auto-generated re-exports for symbols in numbered _utils files
_REEXPORT_MAP = {
    "ExportBinding": "_skeleton_contracts_utils_3",
    "SkeletonContractSet": "_skeleton_contracts_utils_3",
}

def __getattr__(name):
    if name in _REEXPORT_MAP:
        import importlib
        mod = importlib.import_module(f"app.orchestrator.{_REEXPORT_MAP[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
