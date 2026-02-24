# FILE: app/pot_spec/grounded/segmentation.py
"""
Pipeline Segmentation Engine (v1.0)

Core segmentation logic: determines if a job needs decomposition, identifies
segment boundaries, generates segment specs, builds the manifest, and
validates the result deterministically.

v2.0 (2026-02): Extracted _build_manifest_from_concepts to _manifest_builder.py
v1.0 (2026-02-08): Initial implementation
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from .segment_schemas import SegmentManifest, SegmentSpec
from .file_verifier import verify_segment_files
from app.pot_spec.grounded._segmentation_utils_2 import (
    BACKEND_PATH_INDICATORS, FILE_COUNT_THRESHOLD, FRONTEND_PATH_INDICATORS,
    MAX_FILES_PER_SEGMENT, MIN_FILES_PER_SEGMENT, SEGMENTATION_BUILD_ID,
    _generate_segment_id, _resolve_to_absolute,
)
from app.pot_spec.grounded._segmentation_utils_3 import (
    _distribute_requirements, _infer_layer_dependencies,
    _load_architecture_file_list, _merge_small_segments,
    classify_file_layer, group_files_by_layer, needs_segmentation,
)
from app.pot_spec.grounded._segmentation_utils_4 import ArchLayer, generate_segments
from app.pot_spec.grounded._manifest_builder import (
    _build_manifest_from_concepts as _build_manifest_from_concepts_impl,
)

logger = logging.getLogger(__name__)
print(f"[SEGMENTATION_LOADED] BUILD_ID={SEGMENTATION_BUILD_ID}")


# =============================================================================
# v5.5 PHASE 3B: Build manifest from concept-aware groupings
# =============================================================================

def _build_manifest_from_concepts(
    concept_groups: List[Dict[str, Any]],
    file_scope: List[str],
    requirements: List[str],
    acceptance_criteria: List[str],
    parent_spec_id: Optional[str],
    parent_spec_hash: Optional[str],
    arch_paths: List[str],
) -> Optional[SegmentManifest]:
    """Delegate to _manifest_builder with injected dependencies."""
    return _build_manifest_from_concepts_impl(
        concept_groups,
        file_scope,
        requirements,
        acceptance_criteria,
        parent_spec_id,
        parent_spec_hash,
        arch_paths,
        _topological_sort_fn=_topological_sort,
        _validate_manifest_fn=validate_manifest,
    )


# =============================================================================
# TOPOLOGICAL SORT (Kahn's algorithm)
# =============================================================================

def _topological_sort(
    nodes: Any,
    edges: Dict[str, List[str]],
) -> Optional[List[str]]:
    """
    Topological sort using Kahn's algorithm.

    Args:
        nodes: All node names
        edges: Dict mapping each node to its dependencies (predecessors).

    Returns:
        Sorted list in dependency order, or None if a cycle is detected.
    """
    node_list = list(nodes)

    in_degree: Dict[str, int] = {n: 0 for n in node_list}
    successors: Dict[str, List[str]] = {n: [] for n in node_list}

    for node in node_list:
        deps = edges.get(node, [])
        in_degree[node] = len(deps)
        for dep in deps:
            if dep in successors:
                successors[dep].append(node)

    queue = [n for n in node_list if in_degree[n] == 0]
    result: List[str] = []

    while queue:
        queue.sort()
        node = queue.pop(0)
        result.append(node)
        for successor in successors.get(node, []):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)

    if len(result) != len(node_list):
        unprocessed = set(node_list) - set(result)
        logger.error("[segmentation] DAG cycle detected! Unprocessed: %s", unprocessed)
        return None

    return result


# =============================================================================
# MANIFEST VALIDATION
# =============================================================================

def validate_manifest(manifest: SegmentManifest) -> Tuple[bool, List[str]]:
    """
    Deterministic validation of a segment manifest.

    Checks:
    1. Dependency graph is a DAG (no cycles)
    2. Every requirement maps to at least one segment
    3. No file is a CREATE/MODIFY target in more than one segment
    """
    errors: List[str] = []

    # Check 1: DAG
    edges: Dict[str, List[str]] = {}
    segment_ids = {s.segment_id for s in manifest.segments}
    for seg in manifest.segments:
        for dep in seg.dependencies:
            if dep not in segment_ids:
                errors.append(f"Segment {seg.segment_id} depends on unknown segment: {dep}")
        edges[seg.segment_id] = seg.dependencies

    sorted_order = _topological_sort(segment_ids, edges)
    if sorted_order is None:
        errors.append("CYCLE DETECTED: Segment dependency graph contains circular dependencies")

    # Check 2: Requirement coverage
    for req, seg_ids in manifest.requirement_map.items():
        if not seg_ids:
            errors.append(f"Unmapped requirement: {req}")
        else:
            for sid in seg_ids:
                if sid not in segment_ids:
                    errors.append(f"Requirement '{req}' maps to non-existent segment: {sid}")

    # Check 3: File ownership uniqueness
    file_owners: Dict[str, str] = {}
    for seg in manifest.segments:
        for path in seg.file_scope:
            if path in file_owners:
                errors.append(
                    f"File ownership conflict: '{path}' is in both "
                    f"{file_owners[path]} and {seg.segment_id}"
                )
            else:
                file_owners[path] = seg.segment_id

    # Check 4: Interface contracts (warning only)
    segments_with_dependents: Set[str] = set()
    for seg in manifest.segments:
        for dep in seg.dependencies:
            segments_with_dependents.add(dep)

    for seg in manifest.segments:
        if seg.segment_id in segments_with_dependents:
            if seg.exposes is None or seg.exposes.is_empty():
                logger.warning(
                    "[segmentation] Segment %s has dependents but no interface contracts declared",
                    seg.segment_id,
                )

    is_valid = len(errors) == 0
    if is_valid:
        logger.info("[segmentation] Manifest validation PASSED")
    else:
        logger.warning("[segmentation] Manifest validation FAILED: %d errors", len(errors))
        for err in errors:
            logger.warning("[segmentation]   - %s", err)

    return is_valid, errors
