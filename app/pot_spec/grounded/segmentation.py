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

logger = logging.getLogger(__name__)

SEGMENTATION_BUILD_ID = "2026-02-08-v1.1-gap-fixes"
print(f"[SEGMENTATION_LOADED] BUILD_ID={SEGMENTATION_BUILD_ID}")


# =============================================================================
# CONSTANTS
# =============================================================================

# File count threshold above which segmentation is triggered
FILE_COUNT_THRESHOLD = 15

# Minimum files per segment (merge tiny segments into adjacent ones)
MIN_FILES_PER_SEGMENT = 2

# Maximum files per segment (split large clusters)
MAX_FILES_PER_SEGMENT = 15

# Backend path indicators
BACKEND_PATH_INDICATORS = {
    "app/", "app\\",
    "routers/", "routers\\",
    "services/", "services\\",
    "endpoints/", "endpoints\\",
    "models/", "models\\",
    "main.py",
}

# Frontend path indicators
FRONTEND_PATH_INDICATORS = {
    "src/components/", "src\\components\\",
    "src/hooks/", "src\\hooks\\",
    "src/services/", "src\\services\\",
    "src/types/", "src\\types\\",
    "src/styles/", "src\\styles\\",
    ".tsx", ".jsx",
    "main.js",
    "main.tsx",
    "App.tsx",
}


# =============================================================================
# SEGMENTATION CRITERIA — Deterministic triggers
# =============================================================================

def needs_segmentation(
    file_scope: List[str],
    requirements: Optional[List[str]] = None,
) -> Tuple[bool, str]:
    """
    Determine if a job needs segmentation.
    
    Phase 1 triggers (deterministic only):
    1. Total files in spec > FILE_COUNT_THRESHOLD (15)
    2. Spec spans both backend and frontend
    
    "Multiple independent features" is deferred — it requires an LLM
    classification call which contradicts the threshold check being a
    simple deterministic function.
    
    Args:
        file_scope: All files the job creates or modifies
        requirements: Job requirements (reserved for future use)
    
    Returns:
        (should_segment: bool, reason: str)
    """
    total_files = len(file_scope)

    # Trigger 1: File count
    if total_files > FILE_COUNT_THRESHOLD:
        return True, f"File count ({total_files}) exceeds threshold ({FILE_COUNT_THRESHOLD})"

    # Trigger 2: Backend + Frontend span
    has_backend = False
    has_frontend = False
    for path in file_scope:
        normalised = path.replace("\\", "/").lower()
        if any(indicator.replace("\\", "/").lower() in normalised for indicator in BACKEND_PATH_INDICATORS):
            has_backend = True
        if any(indicator.replace("\\", "/").lower() in normalised for indicator in FRONTEND_PATH_INDICATORS):
            has_frontend = True
        if has_backend and has_frontend:
            return True, "Spec spans both backend and frontend"

    return False, f"No segmentation needed ({total_files} files, single domain)"


# =============================================================================
# BOUNDARY DETECTION — Classify files into architectural layers
# =============================================================================

class ArchLayer:
    """Architectural layer names used for boundary detection."""
    BACKEND_SERVICES = "backend-services"
    BACKEND_API = "backend-api"
    BACKEND_PIPELINE = "backend-pipeline"
    BACKEND_CONFIG = "backend-config"
    FRONTEND_COMPONENTS = "frontend-components"
    FRONTEND_HOOKS = "frontend-hooks"
    FRONTEND_SERVICES = "frontend-services"
    FRONTEND_INTEGRATION = "frontend-integration"
    INFRASTRUCTURE = "infrastructure"
    TESTS = "tests"


def classify_file_layer(path: str) -> str:
    """
    Classify a file into an architectural layer based on its path.
    
    Priority order (Design Spec Section 4.2):
    1. Architectural layer (path-based)
    2. File extension as tiebreaker
    """
    normalised = path.replace("\\", "/").lower()

    # Tests
    if "/tests/" in normalised or "/test_" in normalised or normalised.startswith("tests/"):
        return ArchLayer.TESTS

    # Frontend layers
    if "src/components/" in normalised:
        return ArchLayer.FRONTEND_COMPONENTS
    if "src/hooks/" in normalised:
        return ArchLayer.FRONTEND_HOOKS
    if "src/services/" in normalised:
        return ArchLayer.FRONTEND_SERVICES
    if "src/" in normalised and normalised.endswith((".tsx", ".jsx", ".ts", ".js")):
        # Catch-all for frontend files not in a specific sub-directory
        # (e.g. src/App.tsx, src/main.tsx, main.js at electron root)
        return ArchLayer.FRONTEND_INTEGRATION

    # Electron main process
    if normalised.endswith("main.js") and "src/" not in normalised:
        return ArchLayer.FRONTEND_INTEGRATION

    # Backend layers
    if "/routers/" in normalised or "/endpoints/" in normalised:
        return ArchLayer.BACKEND_API
    if "/services/" in normalised:
        return ArchLayer.BACKEND_SERVICES
    if "/pipeline/" in normalised or "/overwatcher/" in normalised or "/pot_spec/" in normalised:
        return ArchLayer.BACKEND_PIPELINE
    if "/config/" in normalised or "config.py" in normalised:
        return ArchLayer.BACKEND_CONFIG

    # Backend catch-all
    if normalised.endswith(".py") and "app/" in normalised:
        return ArchLayer.BACKEND_SERVICES

    # Infrastructure / config
    if normalised.endswith((".json", ".yaml", ".yml", ".toml", ".cfg", ".ini")):
        return ArchLayer.INFRASTRUCTURE

    # Default
    return ArchLayer.INFRASTRUCTURE


def group_files_by_layer(file_scope: List[str]) -> Dict[str, List[str]]:
    """
    Group files into architectural layers.
    
    Returns dict mapping layer name → list of file paths.
    """
    groups: Dict[str, List[str]] = defaultdict(list)
    for path in file_scope:
        layer = classify_file_layer(path)
        groups[layer].append(path)
    return dict(groups)


# =============================================================================
# SEGMENT GENERATION
# =============================================================================

def _generate_segment_id(index: int, layer: str) -> str:
    """Generate a segment ID like 'seg-01-backend-services'."""
    return f"seg-{index:02d}-{layer}"


def _infer_layer_dependencies(layers_present: List[str]) -> Dict[str, List[str]]:
    """
    Infer dependency ordering between architectural layers.
    
    Dependency direction (Design Spec Section 4.2):
    - Config/infrastructure has no dependencies (foundation)
    - Backend services depend on config
    - Backend API depends on services
    - Backend pipeline depends on services
    - Frontend components depend on backend API (for endpoint contracts)
    - Frontend hooks/services depend on components
    - Frontend integration depends on all frontend layers
    - Tests depend on everything
    """
    # Define the dependency graph between layers
    _LAYER_DEPS: Dict[str, List[str]] = {
        ArchLayer.INFRASTRUCTURE: [],
        ArchLayer.BACKEND_CONFIG: [],
        ArchLayer.BACKEND_SERVICES: [ArchLayer.BACKEND_CONFIG, ArchLayer.INFRASTRUCTURE],
        ArchLayer.BACKEND_API: [ArchLayer.BACKEND_SERVICES],
        ArchLayer.BACKEND_PIPELINE: [ArchLayer.BACKEND_SERVICES],
        ArchLayer.FRONTEND_SERVICES: [ArchLayer.BACKEND_API],
        ArchLayer.FRONTEND_HOOKS: [ArchLayer.BACKEND_API, ArchLayer.FRONTEND_SERVICES],
        ArchLayer.FRONTEND_COMPONENTS: [ArchLayer.FRONTEND_HOOKS, ArchLayer.FRONTEND_SERVICES],
        ArchLayer.FRONTEND_INTEGRATION: [
            ArchLayer.FRONTEND_COMPONENTS,
            ArchLayer.FRONTEND_HOOKS,
            ArchLayer.FRONTEND_SERVICES,
        ],
        ArchLayer.TESTS: [],  # Tests depend on everything — resolved at segment level
    }

    present_set = set(layers_present)
    result: Dict[str, List[str]] = {}

    for layer in layers_present:
        raw_deps = _LAYER_DEPS.get(layer, [])
        # Only include dependencies that are actually present in this job
        result[layer] = [d for d in raw_deps if d in present_set]

        # Tests depend on all non-test layers present
        if layer == ArchLayer.TESTS:
            result[layer] = [l for l in layers_present if l != ArchLayer.TESTS]

    return result


def _merge_small_segments(
    layer_groups: Dict[str, List[str]],
    layer_deps: Dict[str, List[str]],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Merge layers with fewer than MIN_FILES_PER_SEGMENT files into
    adjacent layers to avoid trivially small segments.
    """
    # Find layers that are too small
    small_layers = [
        layer for layer, files in layer_groups.items()
        if len(files) < MIN_FILES_PER_SEGMENT and layer != ArchLayer.TESTS
    ]

    if not small_layers:
        return layer_groups, layer_deps

    merged_groups = dict(layer_groups)
    merged_deps = dict(layer_deps)

    for small in small_layers:
        if small not in merged_groups:
            continue  # Already merged

        files = merged_groups[small]
        deps = merged_deps.get(small, [])

        # Find a merge target: prefer a dependency (merge into upstream)
        merge_target = None
        if deps:
            # Merge into the first dependency that still exists
            for dep in deps:
                if dep in merged_groups:
                    merge_target = dep
                    break

        if not merge_target:
            # No dependency — merge into the first layer that depends on us
            for layer, layer_dep_list in merged_deps.items():
                if small in layer_dep_list and layer in merged_groups:
                    merge_target = layer
                    break

        if merge_target:
            merged_groups[merge_target].extend(files)
            del merged_groups[small]

            # Update deps: anything that depended on small now depends on merge_target
            for layer in merged_deps:
                if small in merged_deps[layer]:
                    merged_deps[layer] = [
                        merge_target if d == small else d
                        for d in merged_deps[layer]
                    ]
                    # Deduplicate
                    merged_deps[layer] = list(dict.fromkeys(merged_deps[layer]))
                    # Remove self-dependency
                    merged_deps[layer] = [d for d in merged_deps[layer] if d != layer]

            if small in merged_deps:
                del merged_deps[small]

            logger.info(
                "[segmentation] Merged small layer %s (%d files) into %s",
                small, len(files), merge_target,
            )

    return merged_groups, merged_deps


# =============================================================================
# ARCHITECTURE MAP LOADING (v1.1 — Gap 2 fix)
# =============================================================================

def _load_architecture_file_list() -> List[str]:
    """
    v1.1: Load known file paths from the architecture index (INDEX.json).
    
    Returns a list of absolute file paths known to the architecture map.
    Used for:
    - Resolving relative paths in file_scope to absolute paths
    - Providing the 'known paths' set for new-file detection in file_verifier
    - Identifying boundary files (files at layer edges)
    
    Returns empty list if INDEX.json is unavailable (non-fatal).
    """
    import json as _json
    
    arch_index_dir = os.getenv(
        "ASTRA_ARCH_INDEX_DIR",
        os.path.join("D:\\", "Orb", ".architecture"),
    )
    index_path = os.path.join(arch_index_dir, "INDEX.json")
    
    if not os.path.isfile(index_path):
        logger.debug("[segmentation] INDEX.json not found at %s", index_path)
        return []
    
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = _json.load(f)
        
        paths = []
        for entry in index_data.get("files", []):
            file_path = entry.get("path") or entry.get("file")
            if file_path:
                paths.append(file_path)
        
        logger.info("[segmentation] v1.1 Loaded %d paths from architecture index", len(paths))
        return paths
    except Exception as e:
        logger.warning("[segmentation] v1.1 Failed to load INDEX.json: %s", e)
        return []


def _resolve_to_absolute(path: str, known_paths: List[str]) -> Optional[str]:
    """
    Resolve a relative path to its absolute form using the architecture index.
    
    If the path is already absolute and exists, return it directly.
    Otherwise, search known_paths for a match on the relative suffix.
    """
    # Already absolute?
    if os.path.isabs(path):
        return path if os.path.exists(path) else path  # Return as-is even if missing
    
    # Normalise for matching
    path_norm = path.replace('/', '\\').lower()
    
    for known in known_paths:
        known_norm = known.replace('/', '\\').lower()
        if known_norm.endswith(path_norm):
            return known
    
    return None


# =============================================================================
# REQUIREMENT DISTRIBUTION (v1.1 — Gap 3 fix)
# =============================================================================

def _distribute_requirements(
    requirements: List[str],
    segments: List[SegmentSpec],
) -> Dict[str, List[str]]:
    """
    v1.1: Distribute requirements to segments using keyword/path overlap.
    
    Heuristic (Phase 1 — no LLM call):
    1. For each requirement, extract keywords (file paths, layer names,
       technology terms).
    2. Score each segment by overlap: file path matches score highest,
       layer keyword matches score medium.
    3. Assign the requirement to the top-scoring segment(s).
    4. If no segment matches, assign to all segments (fallback).
    
    This doesn't need to be perfect — it just needs to be better than
    "every requirement maps to every segment". Phase 2 can refine with
    LLM-powered assignment.
    """
    requirement_map: Dict[str, List[str]] = {}
    
    # Build a lookup: normalised file path → segment_id
    file_to_segment: Dict[str, str] = {}
    for seg in segments:
        for f in seg.file_scope:
            file_to_segment[f.replace('/', '\\').lower()] = seg.segment_id
    
    # Layer keywords → segment_id
    layer_keywords: Dict[str, List[str]] = {
        "backend": [],
        "frontend": [],
        "api": [],
        "endpoint": [],
        "service": [],
        "component": [],
        "hook": [],
        "config": [],
        "test": [],
        "ui": [],
        "router": [],
    }
    for seg in segments:
        seg_layer = seg.segment_id.split('-', 2)[-1] if '-' in seg.segment_id else ""
        if "backend" in seg_layer:
            layer_keywords["backend"].append(seg.segment_id)
        if "frontend" in seg_layer or "component" in seg_layer or "hook" in seg_layer:
            layer_keywords["frontend"].append(seg.segment_id)
            layer_keywords["ui"].append(seg.segment_id)
        if "api" in seg_layer:
            layer_keywords["api"].append(seg.segment_id)
            layer_keywords["endpoint"].append(seg.segment_id)
            layer_keywords["router"].append(seg.segment_id)
        if "service" in seg_layer:
            layer_keywords["service"].append(seg.segment_id)
        if "component" in seg_layer:
            layer_keywords["component"].append(seg.segment_id)
        if "hook" in seg_layer:
            layer_keywords["hook"].append(seg.segment_id)
        if "config" in seg_layer:
            layer_keywords["config"].append(seg.segment_id)
        if "test" in seg_layer:
            layer_keywords["test"].append(seg.segment_id)
    
    all_segment_ids = [s.segment_id for s in segments]
    
    for req in requirements:
        matched_segments: Dict[str, int] = {}  # segment_id → score
        req_lower = req.lower()
        
        # Score 1: Direct file path references (highest signal)
        # Look for path-like strings in the requirement
        path_fragments = re.findall(
            r'[\w/\\.-]+\.(?:py|ts|tsx|js|jsx)',
            req_lower,
        )
        for frag in path_fragments:
            frag_norm = frag.replace('/', '\\')
            for file_key, seg_id in file_to_segment.items():
                if frag_norm in file_key or file_key.endswith(frag_norm):
                    matched_segments[seg_id] = matched_segments.get(seg_id, 0) + 10
        
        # Score 2: Layer keyword overlap (medium signal)
        for keyword, seg_ids in layer_keywords.items():
            if keyword in req_lower:
                for sid in seg_ids:
                    matched_segments[sid] = matched_segments.get(sid, 0) + 3
        
        if matched_segments:
            # Take segments with scores above half the max score
            max_score = max(matched_segments.values())
            threshold = max_score // 2 if max_score > 3 else 0
            assigned = [
                sid for sid, score in matched_segments.items()
                if score > threshold
            ]
            requirement_map[req] = assigned if assigned else all_segment_ids
        else:
            # No match — assign to all segments (fallback)
            requirement_map[req] = list(all_segment_ids)
    
    # Log distribution quality
    total = len(requirements)
    specific = sum(1 for sids in requirement_map.values() if len(sids) < len(all_segment_ids))
    logger.info(
        "[segmentation] v1.1 Requirement distribution: %d/%d requirements mapped to specific segments",
        specific, total,
    )
    
    return requirement_map


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def generate_segments(
    file_scope: List[str],
    requirements: List[str],
    acceptance_criteria: List[str],
    parent_spec_id: Optional[str] = None,
    parent_spec_hash: Optional[str] = None,
) -> Optional[SegmentManifest]:
    """
    Generate a segmented manifest from a job's file scope.
    
    This is the main segmentation entry point. It:
    1. Checks if segmentation is needed
    2. Loads architecture map for path resolution (v1.1)
    3. Groups files by architectural layer
    4. Infers dependency ordering
    5. Merges small segments
    6. Generates SegmentSpec objects
    7. Runs file verification per segment (v1.1)
    8. Distributes requirements to segments (v1.1)
    9. Builds and validates the manifest
    
    Returns None if segmentation is not needed (job runs single-pass).
    Returns None if validation fails (graceful fallback to single-pass).
    
    Args:
        file_scope: All files the job creates or modifies
        requirements: Job requirements
        acceptance_criteria: Job acceptance criteria
        parent_spec_id: ID of the parent SPoT spec
        parent_spec_hash: Hash of the parent SPoT spec
    """
    # Check if segmentation is needed
    should_segment, reason = needs_segmentation(file_scope, requirements)
    if not should_segment:
        logger.info("[segmentation] Skipping segmentation: %s", reason)
        return None

    logger.info("[segmentation] Segmentation triggered: %s", reason)

    # v1.1: Load architecture map for path resolution and boundary detection
    arch_paths = _load_architecture_file_list()

    # Group files by architectural layer
    layer_groups = group_files_by_layer(file_scope)
    layers_present = list(layer_groups.keys())

    logger.info(
        "[segmentation] File distribution: %s",
        {layer: len(files) for layer, files in layer_groups.items()},
    )

    # Infer dependency ordering
    layer_deps = _infer_layer_dependencies(layers_present)

    # Merge small segments
    layer_groups, layer_deps = _merge_small_segments(layer_groups, layer_deps)

    # Topological sort to determine segment order
    sorted_layers = _topological_sort(layer_groups.keys(), layer_deps)
    if sorted_layers is None:
        logger.error("[segmentation] Topological sort failed — cycle detected in layer dependencies")
        return None  # Fallback to single pass

    # Build a set of all files in scope for evidence_files calculation
    all_scope_files: Set[str] = set()
    for files in layer_groups.values():
        all_scope_files.update(files)

    # Generate segment specs
    segments: List[SegmentSpec] = []
    layer_to_segment_id: Dict[str, str] = {}

    for index, layer in enumerate(sorted_layers, start=1):
        seg_id = _generate_segment_id(index, layer)
        layer_to_segment_id[layer] = seg_id

        files = layer_groups[layer]
        dep_layers = layer_deps.get(layer, [])
        dep_segment_ids = [layer_to_segment_id[dl] for dl in dep_layers if dl in layer_to_segment_id]

        # v1.1: Compute evidence_files — files from dependency layers that
        # this segment needs to READ for context (boundary files)
        evidence_files: List[str] = []
        for dep_layer in dep_layers:
            if dep_layer in layer_groups:
                for dep_file in layer_groups[dep_layer]:
                    if dep_file not in files:  # Don't include own files
                        evidence_files.append(dep_file)

        # v1.1: Run file verification for this segment's files
        grounding = None
        try:
            grounding = verify_segment_files(
                file_scope=files,
                evidence_files=evidence_files,
                known_arch_paths=set(arch_paths) if arch_paths else None,
            )
            logger.info(
                "[segmentation] v1.1 File verification for %s: "
                "%d verified, %d stale, %d create_targets",
                seg_id,
                len(grounding.verified_files),
                len(grounding.stale_entries),
                len(grounding.create_targets),
            )
        except Exception as verify_err:
            logger.warning(
                "[segmentation] v1.1 File verification failed for %s (non-fatal): %s",
                seg_id, verify_err,
            )

        segment = SegmentSpec(
            segment_id=seg_id,
            title=f"{layer.replace('-', ' ').title()} — {len(files)} files",
            parent_spec_id=parent_spec_id,
            requirements=[],  # Populated by _distribute_requirements below
            file_scope=files,
            evidence_files=evidence_files,
            dependencies=dep_segment_ids,
            estimated_files=len(files),
            grounding_data=grounding,
        )
        segments.append(segment)

    # v1.1: Distribute requirements to segments (heuristic, not trivial broadcast)
    requirement_map = _distribute_requirements(requirements, segments)

    # Assign requirements to each segment based on the map
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
            "[segmentation] Manifest validation failed — falling back to single pass. Errors: %s",
            errors,
        )
        return None  # Graceful fallback

    logger.info("[segmentation] Manifest generated successfully: %s", manifest.summary())
    return manifest


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
