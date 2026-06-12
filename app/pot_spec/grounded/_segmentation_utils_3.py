# Purpose: segmentation utils 3
# Called-by: app.pot_spec.grounded._manifest_builder, app.pot_spec.grounded._segmentation_utils_4, app.pot_spec.grounded._spec_runner_segmentation, app.pot_spec.grounded.segmentation
# Depends-on: app.pot_spec.grounded._segmentation_utils_2, app.pot_spec.grounded.segment_schemas, app.pot_spec.grounded.segmentation
# Last-renovated: 2026-06-11
from __future__ import annotations
import logging
import os
import re
from .segment_schemas import SegmentSpec
from app.pot_spec.grounded._segmentation_utils_2 import BACKEND_PATH_INDICATORS, FILE_COUNT_THRESHOLD, FRONTEND_PATH_INDICATORS, MIN_FILES_PER_SEGMENT
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


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

def classify_file_layer(path: str) -> str:
    """
    Classify a file into an architectural layer based on its path.
    
    Priority order (Design Spec Section 4.2):
    1. Architectural layer (path-based)
    2. File extension as tiebreaker
    """
    from .segmentation import ArchLayer
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
    from .segmentation import ArchLayer
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
    from .segmentation import ArchLayer
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
