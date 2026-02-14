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

SEGMENTATION_BUILD_ID = "2026-02-13-v1.4-refactor-source-scope-fix"
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

def generate_segments(
    file_scope: List[str],
    requirements: List[str],
    acceptance_criteria: List[str],
    parent_spec_id: Optional[str] = None,
    parent_spec_hash: Optional[str] = None,
    concept_groups: Optional[List[Dict[str, Any]]] = None,
) -> Optional[SegmentManifest]:
    """
    Generate a segmented manifest from a job's file scope.
    
    This is the main segmentation entry point. It:
    1. Checks if segmentation is needed
    2. (v5.5) If concept_groups provided, uses concept-aware grouping (Phase 3B)
    3. Otherwise loads architecture map for path resolution (v1.1)
    4. Groups files by architectural layer (legacy fallback)
    5. Infers dependency ordering
    6. Merges small segments
    7. Generates SegmentSpec objects
    8. Runs file verification per segment (v1.1)
    9. Distributes requirements to segments (v1.1)
    10. Builds and validates the manifest
    
    Returns None if segmentation is not needed (job runs single-pass).
    Returns None if validation fails (graceful fallback to single-pass).
    
    Args:
        file_scope: All files the job creates or modifies
        requirements: Job requirements
        acceptance_criteria: Job acceptance criteria
        parent_spec_id: ID of the parent SPoT spec
        parent_spec_hash: Hash of the parent SPoT spec
        concept_groups: (v5.5 Phase 3B) Pre-computed concept segments from
                        smart_segmentation.py. List of dicts with keys:
                        title, files, concepts, depends_on.
    """
    # Check if segmentation is needed (skipped if concept_groups already provided)
    if concept_groups is None:
        should_segment, reason = needs_segmentation(file_scope, requirements)
        if not should_segment:
            logger.info("[segmentation] Skipping segmentation: %s", reason)
            return None
        logger.info("[segmentation] Segmentation triggered (legacy): %s", reason)
    else:
        logger.info("[segmentation] Using concept-aware grouping (Phase 3B): %d groups",
                     len(concept_groups))

    # v1.1: Load architecture map for path resolution and boundary detection
    arch_paths = _load_architecture_file_list()

    # =====================================================================
    # v5.5 PHASE 3B: Concept-aware grouping (preferred path)
    # =====================================================================
    if concept_groups and len(concept_groups) >= 2:
        logger.info("[segmentation] v5.5 Building segments from concept groups")
        return _build_manifest_from_concepts(
            concept_groups=concept_groups,
            file_scope=file_scope,
            requirements=requirements,
            acceptance_criteria=acceptance_criteria,
            parent_spec_id=parent_spec_id,
            parent_spec_hash=parent_spec_hash,
            arch_paths=arch_paths,
        )

    # =====================================================================
    # Legacy path: layer-based grouping
    # =====================================================================
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
