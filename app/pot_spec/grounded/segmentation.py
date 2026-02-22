# FILE: app/pot_spec/grounded/segmentation.py
"""
Pipeline Segmentation Engine (v1.0)

Core segmentation logic: determines if a job needs decomposition, identifies
segment boundaries, generates segment specs, builds the manifest, and
validates the result deterministically.

Design Spec Sections 3-4:
- Structural analysis from architecture map
- Segmentation criteria (file count >15, backend+frontend span)
- Boundary detection (architectural layer, dependency direction, file clustering)
- Manifest building with topological sort
- Deterministic validation (DAG, requirement coverage, file ownership)

Failure behaviour:
    If segmentation validation fails, the job falls back gracefully to
    single-pass. Validation failure does NOT block or abort the job.
    See validate_manifest() for details.

Version Notes:
-------------
v1.0 (2026-02-08): Initial implementation — Phase 1 of Pipeline Segmentation
    - Deterministic triggers: file count >15, backend+frontend span
    - "Multiple independent features" criterion deferred (requires LLM call)
    - DAG cycle detection, requirement coverage, file ownership validation
    - Graceful fallback to single pass on any validation failure
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from .segment_schemas import (
    CreateTarget,
    GroundingData,
    InterfaceContract,
    SegmentManifest,
    SegmentSpec,
)
from .file_verifier import verify_segment_files
from app.pot_spec.grounded._segmentation_utils_2 import BACKEND_PATH_INDICATORS, FILE_COUNT_THRESHOLD, FRONTEND_PATH_INDICATORS, MAX_FILES_PER_SEGMENT, MIN_FILES_PER_SEGMENT, SEGMENTATION_BUILD_ID, _generate_segment_id, _resolve_to_absolute
from app.pot_spec.grounded._segmentation_utils_3 import _distribute_requirements, _infer_layer_dependencies, _load_architecture_file_list, _merge_small_segments, classify_file_layer, group_files_by_layer, needs_segmentation
from app.pot_spec.grounded._segmentation_utils_4 import ArchLayer, generate_segments

logger = logging.getLogger(__name__)
print(f"[SEGMENTATION_LOADED] BUILD_ID={SEGMENTATION_BUILD_ID}")


# =============================================================================
# CONSTANTS
# =============================================================================

# File count threshold above which segmentation is triggered

# Minimum files per segment (merge tiny segments into adjacent ones)

# Maximum files per segment (split large clusters)

# Backend path indicators

# Frontend path indicators


# =============================================================================
# SEGMENTATION CRITERIA — Deterministic triggers
# =============================================================================


# =============================================================================
# BOUNDARY DETECTION — Classify files into architectural layers
# =============================================================================


# =============================================================================
# SEGMENT GENERATION
# =============================================================================


# =============================================================================
# ARCHITECTURE MAP LOADING (v1.1 — Gap 2 fix)
# =============================================================================


# =============================================================================
# REQUIREMENT DISTRIBUTION (v1.1 — Gap 3 fix)
# =============================================================================


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
) -> Optional["SegmentManifest"]:
    """
    Build a SegmentManifest from concept-aware groupings (Phase 3B).

    Takes the output of smart_segmentation.generate_concept_segments() and
    converts it into the same SegmentManifest format that the legacy
    layer-based path produces.

    Args:
        concept_groups: List of dicts with keys: title, files, concepts, depends_on
        file_scope: Full file scope (for validation)
        requirements: Job requirements
        acceptance_criteria: Job acceptance criteria
        parent_spec_id/parent_spec_hash: Parent spec identifiers
        arch_paths: Known architecture file paths for verification
    """
    segments: List[SegmentSpec] = []
    index_to_seg_id: Dict[int, str] = {}

    for idx, group in enumerate(concept_groups):
        title = group.get("title", f"Segment {idx + 1}")
        files = group.get("files", [])
        concepts = group.get("concepts", [])

        # Generate segment ID from title
        # e.g. "Voice Transcription" → "seg-01-voice-transcription"
        slug = title.lower().replace(" ", "-").replace("_", "-")
        slug = re.sub(r'[^a-z0-9\-]', '', slug)[:30]
        seg_id = f"seg-{idx + 1:02d}-{slug}" if slug else f"seg-{idx + 1:02d}"
        index_to_seg_id[idx] = seg_id

        # Evidence files: files from dependency groups that this segment reads
        dep_indices = group.get("depends_on", [])
        evidence_files: List[str] = []
        for dep_idx in dep_indices:
            if 0 <= dep_idx < len(concept_groups):
                for dep_file in concept_groups[dep_idx].get("files", []):
                    if dep_file not in files:
                        evidence_files.append(dep_file)

        # File verification (same as legacy path)
        grounding = None
        try:
            grounding = verify_segment_files(
                file_scope=files,
                evidence_files=evidence_files,
                known_arch_paths=set(arch_paths) if arch_paths else None,
            )
        except Exception as verify_err:
            logger.warning("[segmentation] v5.5 File verification failed for %s: %s",
                           seg_id, verify_err)

        # v3.0: Safety net — deduplicate file_scope
        files = list(dict.fromkeys(files))  # preserves order, removes dupes

        segment = SegmentSpec(
            segment_id=seg_id,
            title=f"{title} — {len(files)} file(s)",
            parent_spec_id=parent_spec_id,
            requirements=[],  # Populated below
            acceptance_criteria=[],
            file_scope=files,
            evidence_files=evidence_files,
            dependencies=[],  # Resolved below after all IDs are known
            estimated_files=len(files),
            grounding_data=grounding,
        )
        segments.append(segment)

    # Resolve dependency indices → segment IDs
    for idx, group in enumerate(concept_groups):
        dep_indices = group.get("depends_on", [])
        dep_ids = [index_to_seg_id[d] for d in dep_indices
                    if d in index_to_seg_id and d != idx]
        segments[idx].dependencies = dep_ids

    # =========================================================================
    # v1.5 FIX: Deterministic cycle breaking
    # =========================================================================
    # The LLM segmenter sometimes produces circular dependencies, most commonly
    # between facade/__init__.py segments and the modules they re-export.
    # (e.g. facade depends on executor, executor depends on facade)
    #
    # Strategy:
    # 1. Detect facade/init segments — these contain __init__.py files and
    #    exist to re-export from other modules. They are ALWAYS terminal
    #    (nothing should depend on them; they depend on everything else).
    # 2. Remove any edges where non-facade segments depend on facade segments.
    # 3. Ensure facade segments depend on ALL non-facade segments.
    # 4. Run a general Kahn’s-based cycle breaker for any remaining cycles:
    #    find back-edges and remove them, preferring to cut the edge FROM
    #    the segment with fewer dependencies (the "earlier" node).
    # =========================================================================
    if len(segments) >= 2:
        # Step 1: Identify facade segments
        # v1.6: Also detect the original monolith file in file→package refactors.
        # When segment_loop.py is being converted to segment_loop/__init__.py,
        # the segment containing segment_loop.py IS the facade even though it
        # doesn't have __init__.py in its file_scope.
        _facade_ids: set = set()

        # Collect all subpackage dirs: if any segment has dir/file.py, record 'dir'
        _all_files_normalised = []
        for seg in segments:
            for f in seg.file_scope:
                _all_files_normalised.append(f.replace("\\", "/"))
        _subpackage_dirs = set()
        for _fnorm in _all_files_normalised:
            _parts = _fnorm.rsplit("/", 1)
            if len(_parts) == 2:
                _subpackage_dirs.add(_parts[0])

        for seg in segments:
            _has_init = any(
                f.replace("\\", "/").endswith("__init__.py")
                for f in seg.file_scope
            )

            # v1.6: Detect original monolith being replaced by subpackage.
            # If this segment has 'foo.py' and another segment has 'foo/__init__.py'
            # or 'foo/bar.py', then 'foo.py' is the monolith being replaced.
            _has_monolith_being_replaced = False
            for f in seg.file_scope:
                _fn = f.replace("\\", "/")
                if _fn.endswith(".py") and not _fn.endswith("__init__.py"):
                    # Check if there's a subpackage with the same stem
                    _stem = _fn[:-3]  # strip .py
                    if _stem in _subpackage_dirs:
                        _has_monolith_being_replaced = True
                        logger.info(
                            "[segmentation] v1.6 Monolith replacement detected: "
                            "%s → %s/ subpackage",
                            _fn, _stem,
                        )
                        break

            _title_lower = (seg.title or "").lower()
            _title_has_facade_keyword = any(
                kw in _title_lower
                for kw in ["facade", "fa\u00e7ade", "init", "package init", "re-export",
                           "package initialization", "package entry",
                           "stream", "integration"]
            )
            _is_facade = (
                (_has_init and _title_has_facade_keyword)
                or _has_monolith_being_replaced
            )
            if _is_facade:
                _facade_ids.add(seg.segment_id)
                logger.info(
                    "[segmentation] v1.6 Facade segment detected: %s — will be terminal"
                    " (init=%s, monolith_replace=%s)",
                    seg.segment_id, _has_init, _has_monolith_being_replaced,
                )

        # Step 2: Remove edges where non-facade depends on facade
        if _facade_ids:
            for seg in segments:
                if seg.segment_id in _facade_ids:
                    continue
                _before = len(seg.dependencies)
                seg.dependencies = [
                    d for d in seg.dependencies if d not in _facade_ids
                ]
                if len(seg.dependencies) < _before:
                    logger.info(
                        "[segmentation] v1.5 Removed facade dependency from %s "
                        "(facades are terminal, nothing depends on them)",
                        seg.segment_id,
                    )

            # Step 3: Ensure facade segments depend on ALL non-facade segments
            _non_facade_ids = [
                s.segment_id for s in segments
                if s.segment_id not in _facade_ids
            ]
            for seg in segments:
                if seg.segment_id in _facade_ids:
                    _existing = set(seg.dependencies)
                    for nf_id in _non_facade_ids:
                        if nf_id not in _existing:
                            seg.dependencies.append(nf_id)
                    seg.dependencies = sorted(set(seg.dependencies))
                    logger.info(
                        "[segmentation] v1.5 Facade %s now depends on %d segments (terminal)",
                        seg.segment_id, len(seg.dependencies),
                    )

        # Step 4: General cycle breaker for any remaining cycles
        # Uses iterative Kahn’s: find nodes with no unresolved predecessors,
        # process them. Any nodes left over are in cycles — break cycles by
        # removing the dependency edge from the node with fewer deps.
        _seg_by_id = {s.segment_id: s for s in segments}
        _max_rounds = len(segments)  # Worst case: one cycle break per segment
        for _cycle_round in range(_max_rounds):
            # Check for cycles using Kahn’s
            _edges = {s.segment_id: list(s.dependencies) for s in segments}
            _sorted = _topological_sort(set(_edges.keys()), _edges)
            if _sorted is not None:
                break  # No cycles remain

            # Find nodes in cycle (unprocessed by Kahn’s)
            _in_deg = {sid: 0 for sid in _edges}
            _succs = {sid: [] for sid in _edges}
            for sid, deps in _edges.items():
                _in_deg[sid] = len(deps)
                for d in deps:
                    if d in _succs:
                        _succs[d].append(sid)
            _queue = [sid for sid, deg in _in_deg.items() if deg == 0]
            _processed = set()
            while _queue:
                n = _queue.pop(0)
                _processed.add(n)
                for s in _succs.get(n, []):
                    _in_deg[s] -= 1
                    if _in_deg[s] == 0:
                        _queue.append(s)
            _in_cycle = set(_edges.keys()) - _processed

            if not _in_cycle:
                break  # Safety check

            # Break cycle: find the edge between two cycle nodes where the
            # dependency target has the fewest own dependencies (most upstream)
            _best_edge = None
            _best_score = float('inf')
            for sid in _in_cycle:
                _seg_obj = _seg_by_id[sid]
                for dep in _seg_obj.dependencies:
                    if dep in _in_cycle:
                        # Score = how many deps the target has; lower = more upstream
                        score = len(_seg_by_id[dep].dependencies)
                        if score < _best_score:
                            _best_score = score
                            _best_edge = (sid, dep)

            if _best_edge:
                _from_seg, _to_seg = _best_edge
                _seg_by_id[_from_seg].dependencies = [
                    d for d in _seg_by_id[_from_seg].dependencies
                    if d != _to_seg
                ]
                logger.warning(
                    "[segmentation] v1.5 CYCLE BREAK: removed edge %s → %s "
                    "(breaking cycle, round %d)",
                    _from_seg, _to_seg, _cycle_round + 1,
                )
            else:
                logger.error(
                    "[segmentation] v1.5 Cannot find edge to break cycle in: %s",
                    _in_cycle,
                )
                break  # Give up — validation will catch it

    # =========================================================================
    # v1.4 FIX: Refactor source files belong in integration segment only
    # =========================================================================
    # In refactor-to-package jobs, the LLM grouper may put the original monolith
    # file (and its neighbours like __init__.py) into a helper segment's
    # file_scope. This causes Python file/directory naming conflicts: you can't
    # have architecture_executor.py AND architecture_executor/ at the same level.
    #
    # Fix: Identify existing source files (from grounding_data.verified_files)
    # and ensure they only appear in the integration segment (the one with the
    # most dependencies — it runs last). In all other segments, move these files
    # from file_scope → evidence_files so they're available as read-only context.
    # =========================================================================
    if len(segments) >= 2:
        # Find the integration segment — most dependencies = runs last
        _integration_seg = max(segments, key=lambda s: len(s.dependencies))
        _integration_id = _integration_seg.segment_id

        # Collect all verified (existing) files across all segments
        _all_verified: set = set()
        for seg in segments:
            if seg.grounding_data and isinstance(seg.grounding_data, dict):
                for vf in seg.grounding_data.get("verified_files", []):
                    _vf_path = vf.get("path", "") if isinstance(vf, dict) else str(vf)
                    if _vf_path:
                        _all_verified.add(_vf_path.replace("\\", "/").lower())

        if _all_verified:
            _moved_count = 0
            for seg in segments:
                if seg.segment_id == _integration_id:
                    continue  # Integration segment keeps everything

                _new_scope = []
                for f in seg.file_scope:
                    _f_norm = f.replace("\\", "/").lower()
                    if _f_norm in _all_verified:
                        # Move to evidence_files instead
                        if f not in seg.evidence_files:
                            seg.evidence_files.append(f)
                        _moved_count += 1
                        logger.info(
                            "[segmentation] v1.4 Moved existing file %s from %s file_scope → evidence_files",
                            f, seg.segment_id,
                        )
                    else:
                        _new_scope.append(f)

                if len(_new_scope) < len(seg.file_scope):
                    seg.file_scope = _new_scope
                    seg.estimated_files = len(_new_scope)

            if _moved_count > 0:
                logger.info(
                    "[segmentation] v1.4 Relocated %d existing file(s) to evidence — "
                    "integration segment: %s",
                    _moved_count, _integration_id,
                )

    # =========================================================================
    # v1.2 FIX #1: Infer missing cross-segment dependencies for same-package files
    # =========================================================================
    # When all new files share a common package directory (refactor-to-package),
    # the LLM segmenter often misses import dependencies between sub-modules.
    # E.g., source_context.py imports from sandbox_helpers.py but the segmenter
    # put them in different segments without declaring the dependency.
    #
    # Strategy: detect common package prefix, then for each segment that depends
    # on ANY other segment, also add transitive dependencies (if A->B and B->C,
    # then A should depend on C). For refactor-to-package jobs this ensures
    # utility modules are always available before modules that use them.
    # =========================================================================
    _all_files = [f for seg in segments for f in seg.file_scope]
    _normalised = [f.replace("\\", "/") for f in _all_files]
    if len(_normalised) >= 2:
        # Detect common package prefix (e.g. "app/overwatcher/architecture_executor/")
        _parts_list = [f.rsplit("/", 1) for f in _normalised if "/" in f]
        if _parts_list:
            _dirs = [p[0] for p in _parts_list]
            # Use majority directory (most files share it) rather than requiring ALL
            # This handles refactor-to-package where the original monolith file sits
            # in the parent dir while all new sub-modules sit in the package dir.
            from collections import Counter as _Counter
            _dir_counts = _Counter(_dirs)
            _most_common_dir, _most_common_count = _dir_counts.most_common(1)[0]
            # Trigger if ≥60% of files share the same directory
            _common_dir = _most_common_dir if _most_common_count >= len(_dirs) * 0.6 else None
            if _common_dir:
                logger.info("[segmentation] v1.2 Common package detected: %s — applying transitive deps", _common_dir)
                # Build file->segment index
                _file_to_seg_idx: Dict[int, int] = {}  # file_index -> seg_index
                for seg_idx, seg in enumerate(segments):
                    for fp in seg.file_scope:
                        _file_to_seg_idx[id(fp)] = seg_idx

                # Apply transitive closure: if seg A depends on seg B,
                # and seg B depends on seg C, then seg A should also depend on C.
                _changed = True
                _rounds = 0
                while _changed and _rounds < 10:
                    _changed = False
                    _rounds += 1
                    for seg in segments:
                        _new_deps = set(seg.dependencies)
                        for dep_id in list(seg.dependencies):
                            _dep_seg = next((s for s in segments if s.segment_id == dep_id), None)
                            if _dep_seg:
                                for transitive_dep in _dep_seg.dependencies:
                                    if transitive_dep not in _new_deps and transitive_dep != seg.segment_id:
                                        _new_deps.add(transitive_dep)
                                        _changed = True
                        if len(_new_deps) > len(seg.dependencies):
                            _added = _new_deps - set(seg.dependencies)
                            logger.info("[segmentation] v1.2 Added transitive deps to %s: %s", seg.segment_id, list(_added))
                            seg.dependencies = sorted(_new_deps)
                            # Also update evidence_files for the new dependencies
                            for added_dep_id in _added:
                                _added_seg = next((s for s in segments if s.segment_id == added_dep_id), None)
                                if _added_seg:
                                    for dep_file in _added_seg.file_scope:
                                        if dep_file not in seg.evidence_files and dep_file not in seg.file_scope:
                                            seg.evidence_files.append(dep_file)
                logger.info("[segmentation] v1.2 Transitive dependency closure complete (%d rounds)", _rounds)

    # Distribute requirements
    requirement_map = _distribute_requirements(requirements, segments)
    seg_id_to_seg = {s.segment_id: s for s in segments}
    for req, seg_ids in requirement_map.items():
        for sid in seg_ids:
            if sid in seg_id_to_seg:
                seg_id_to_seg[sid].requirements.append(req)

    # Build manifest
    manifest = SegmentManifest(
        parent_spec_id=parent_spec_id,
        parent_spec_hash=parent_spec_hash,
        segments=segments,
        requirement_map=requirement_map,
    )

    # Validate
    valid, errors = validate_manifest(manifest)
    if not valid:
        logger.warning(
            "[segmentation] v5.5 Concept manifest validation failed: %s — "
            "falling back to legacy layer-based segmentation",
            errors,
        )
        return None  # Caller falls through to legacy path

    logger.info("[segmentation] v5.5 Concept manifest generated: %s", manifest.summary())
    return manifest


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


# =============================================================================
# TOPOLOGICAL SORT (Kahn's algorithm)
# =============================================================================

def _topological_sort(
    nodes: Any,  # Iterable of node names
    edges: Dict[str, List[str]],  # node → list of dependencies (predecessors)
) -> Optional[List[str]]:
    """
    Topological sort using Kahn's algorithm.
    
    Args:
        nodes: All node names
        edges: Dict mapping each node to its dependencies (predecessors).
               edges[A] = [B, C] means A depends on B and C (B, C come before A).
    
    Returns:
        Sorted list of nodes in dependency order, or None if a cycle is detected.
    """
    node_list = list(nodes)
    
    # Build in-degree map and adjacency list (reverse direction for Kahn's)
    in_degree: Dict[str, int] = {n: 0 for n in node_list}
    successors: Dict[str, List[str]] = {n: [] for n in node_list}

    for node in node_list:
        deps = edges.get(node, [])
        in_degree[node] = len(deps)
        for dep in deps:
            if dep in successors:
                successors[dep].append(node)

    # Start with nodes that have no dependencies
    queue = [n for n in node_list if in_degree[n] == 0]
    result: List[str] = []

    while queue:
        # Sort for deterministic ordering
        queue.sort()
        node = queue.pop(0)
        result.append(node)

        for successor in successors.get(node, []):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)

    if len(result) != len(node_list):
        # Cycle detected — not all nodes were processed
        unprocessed = set(node_list) - set(result)
        logger.error(
            "[segmentation] DAG cycle detected! Unprocessed nodes: %s",
            unprocessed,
        )
        return None

    return result


# =============================================================================
# MANIFEST VALIDATION
# =============================================================================

def validate_manifest(manifest: SegmentManifest) -> Tuple[bool, List[str]]:
    """
    Deterministic validation of a segment manifest.
    
    Checks (Design Spec Section 13 — Hard Rules):
    1. Dependency graph is a DAG (no cycles)
    2. Every requirement maps to at least one segment
    3. No file is a CREATE/MODIFY target in more than one segment
    
    If ANY check fails, the manifest is invalid and the job should
    fall back to single-pass processing. This does NOT abort the job.
    
    Returns:
        (is_valid: bool, errors: List[str])
    """
    errors: List[str] = []

    # Check 1: DAG — no circular dependencies
    edges: Dict[str, List[str]] = {}
    segment_ids = {s.segment_id for s in manifest.segments}
    for seg in manifest.segments:
        # Validate that all dependencies reference real segments
        for dep in seg.dependencies:
            if dep not in segment_ids:
                errors.append(
                    f"Segment {seg.segment_id} depends on unknown segment: {dep}"
                )
        edges[seg.segment_id] = seg.dependencies

    sorted_order = _topological_sort(segment_ids, edges)
    if sorted_order is None:
        errors.append("CYCLE DETECTED: Segment dependency graph contains circular dependencies")

    # Check 2: Requirement coverage
    for req, seg_ids in manifest.requirement_map.items():
        if not seg_ids:
            errors.append(f"Unmapped requirement: {req}")
        else:
            # Verify the segment IDs actually exist
            for sid in seg_ids:
                if sid not in segment_ids:
                    errors.append(
                        f"Requirement '{req}' maps to non-existent segment: {sid}"
                    )

    # Check 3: File ownership uniqueness
    file_owners: Dict[str, str] = {}  # path → segment_id
    for seg in manifest.segments:
        for path in seg.file_scope:
            if path in file_owners:
                errors.append(
                    f"File ownership conflict: '{path}' is in both "
                    f"{file_owners[path]} and {seg.segment_id}"
                )
            else:
                file_owners[path] = seg.segment_id

    # Check 4: Interface contracts — segments with dependents must expose
    segments_with_dependents: Set[str] = set()
    for seg in manifest.segments:
        for dep in seg.dependencies:
            segments_with_dependents.add(dep)

    for seg in manifest.segments:
        if seg.segment_id in segments_with_dependents:
            if seg.exposes is None or seg.exposes.is_empty():
                # This is a warning, not a hard error in Phase 1
                # Interface contracts will be populated by the LLM during
                # segment spec generation — they may not be known at boundary
                # detection time.
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
