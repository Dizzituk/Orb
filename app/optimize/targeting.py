from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from app.optimize.models import DependencyEdge
from app.optimize.target_registry import OptimizeTargetDefinition, path_matches_target


SOURCE_EXTENSIONS = {".py", ".ts", ".tsx", ".kt", ".js", ".jsx", ".css"}


def file_is_in_scope(target: OptimizeTargetDefinition, relative_path: str) -> bool:
    return path_matches_target(target, relative_path)


def filter_paths_for_target(paths: List[str], target: OptimizeTargetDefinition) -> List[str]:
    scoped: List[str] = []
    for path in paths:
        normalised = path.replace('\\', '/')
        if file_is_in_scope(target, normalised):
            scoped.append(normalised)
    return scoped


def filter_index_for_target(index_data: Dict[str, Any], target: OptimizeTargetDefinition) -> Dict[str, Any]:
    if not isinstance(index_data, dict):
        return {}
    return {
        key: value
        for key, value in index_data.items()
        if file_is_in_scope(target, str(key))
    }


def filter_import_graph_for_target(import_data: Dict[str, Any], target: OptimizeTargetDefinition) -> Dict[str, Any]:
    graph = import_data.get("graph", {}) if isinstance(import_data, dict) else {}
    filtered_graph: Dict[str, List[str]] = {}
    edge_count = 0
    for source, targets in graph.items():
        source_norm = str(source).replace('\\', '/')
        if not file_is_in_scope(target, source_norm):
            continue
        filtered_targets = [
            str(candidate).replace('\\', '/')
            for candidate in targets
            if file_is_in_scope(target, str(candidate))
        ]
        filtered_graph[source_norm] = filtered_targets
        edge_count += len(filtered_targets)
    return {
        "graph": filtered_graph,
        "stats": {
            "total_modules": len(filtered_graph),
            "total_edges": edge_count,
        },
    }


def filter_dependency_edges_for_target(edges: List[DependencyEdge], target: OptimizeTargetDefinition) -> List[DependencyEdge]:
    return [
        edge
        for edge in edges
        if file_is_in_scope(target, edge.source) and file_is_in_scope(target, edge.target)
    ]


def should_scan_file(relative_path: str) -> bool:
    return Path(relative_path).suffix.lower() in SOURCE_EXTENSIONS
