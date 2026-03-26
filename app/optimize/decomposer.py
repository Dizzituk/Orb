from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.optimize.architecture_paths import get_architecture_paths
from app.optimize.config import (
    DEAD_CODE_IGNORE_PATTERNS,
    FILE_SIZE_OVERSIZED_BYTES,
    HIGH_COMPLEXITY_LINES,
)
from app.optimize.models import ChunkManifest, CodeChunk, DeadCodeCandidate, DependencyEdge
from app.optimize.target_registry import OptimizeTargetDefinition
from app.optimize.targeting import (
    filter_import_graph_for_target,
    filter_index_for_target,
    filter_paths_for_target,
    should_scan_file,
)

logger = logging.getLogger(__name__)


async def decompose(
    target: OptimizeTargetDefinition,
    emit: Optional[callable] = None,
) -> ChunkManifest:
    emit = emit or (lambda msg: None)
    t_start = time.time()

    emit(f"🔍 Decompose: Analysing {target.display_label}...")
    emit(f"   User outcome: {target.user_outcome}")

    import_graph = _load_import_graph(target)
    index_data = _load_index(target)

    emit("   Scanning scoped files...")
    chunks = _scan_files(target, index_data)
    emit(f"   Found {len(chunks)} scoped source files")

    edges = _build_dependency_edges(import_graph)
    _apply_dependency_counts(chunks, edges)
    emit(f"   Mapped {len(edges)} in-scope dependency edges")

    emit("   Detecting dead code...")
    dead_code = _detect_dead_code(chunks, edges, import_graph)
    emit(f"   Found {len(dead_code)} dead code candidates")

    oversized = sum(1 for chunk in chunks if chunk.is_oversized)
    manifest = ChunkManifest(
        target=target.target_id,
        chunks=chunks,
        dependency_edges=edges,
        dead_code=dead_code,
        total_files=len(chunks),
        total_lines=sum(chunk.lines for chunk in chunks),
        total_size_bytes=sum(chunk.size_bytes for chunk in chunks),
        oversized_files=oversized,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    duration = time.time() - t_start
    emit(f"✅ Decompose complete ({duration:.1f}s): {manifest.summary()}")
    return manifest


def _load_import_graph(target: OptimizeTargetDefinition) -> Dict[str, Any]:
    try:
        path = get_architecture_paths(target)["import_graph"]
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            filtered = filter_import_graph_for_target(data, target)
            logger.info("[decompose] Loaded scoped import graph for %s: %d modules", target.target_id, len(filtered.get("graph", {})))
            return filtered
    except Exception as exc:
        logger.warning("[decompose] Failed to load import graph for %s: %s", target.target_id, exc)
    return {"graph": {}, "stats": {}}


def _load_index(target: OptimizeTargetDefinition) -> Dict[str, Any]:
    try:
        path = get_architecture_paths(target)["index"]
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            filtered = filter_index_for_target(data, target)
            logger.info("[decompose] Loaded scoped index for %s: %d entries", target.target_id, len(filtered))
            return filtered
    except Exception as exc:
        logger.warning("[decompose] Failed to load index for %s: %s", target.target_id, exc)
    return {}


def _scan_files(target: OptimizeTargetDefinition, index_data: Dict[str, Any]) -> List[CodeChunk]:
    """Scan files in scope via the sandbox filesystem."""
    from app.sandbox_walk import sandbox_walk, sandbox_read_python

    chunks: List[CodeChunk] = []
    root_path = target.root_path.replace("\\", "/").rstrip("/")
    skip_dirs = {"__pycache__", ".git", "node_modules", ".venv", "venv", ".pytest_cache", "build", ".gradle", "dist"}

    for dirpath, dirnames, filenames in sandbox_walk(root_path, skip_dirs):
        for filename in filenames:
            # Build relative path
            full_path = dirpath.rstrip("\\").rstrip("/") + "/" + filename
            norm_full = full_path.replace("\\", "/")
            if norm_full.startswith(root_path + "/"):
                rel = norm_full[len(root_path) + 1:]
            else:
                rel = norm_full

            if not should_scan_file(rel):
                continue
            if rel not in filter_paths_for_target([rel], target):
                continue

            # Read content from sandbox
            content_text = sandbox_read_python(full_path)
            if content_text is None:
                continue

            lines = content_text.count("\n") + 1
            size = len(content_text.encode("utf-8", errors="ignore"))

            chunk = CodeChunk(
                path=rel,
                name=Path(filename).stem,
                lines=lines,
                size_bytes=size,
                complexity_estimate=_estimate_complexity(lines),
                is_oversized=size > FILE_SIZE_OVERSIZED_BYTES,
                tags=_infer_tags(rel),
                responsibility=_infer_responsibility(rel, target),
            )
            if rel in index_data and isinstance(index_data[rel], dict):
                chunk.responsibility = index_data[rel].get("summary", chunk.responsibility)
            chunks.append(chunk)

    return chunks


def _estimate_complexity(lines: int) -> float:
    if lines < 50:
        return 0.1
    if lines < 200:
        return 0.3
    if lines < HIGH_COMPLEXITY_LINES:
        return 0.5
    if lines < 1000:
        return 0.7
    return 0.9


def _infer_tags(path: str) -> List[str]:
    tags: List[str] = []
    path_lower = path.lower()
    if "router" in path_lower:
        tags.append("router")
    if "model" in path_lower:
        tags.append("model")
    if "service" in path_lower:
        tags.append("service")
    if "test" in path_lower:
        tags.append("test")
    if "schema" in path_lower:
        tags.append("schema")
    if "navigation" in path_lower:
        tags.append("navigation")
    if "viewmodel" in path_lower:
        tags.append("viewmodel")
    if "component" in path_lower:
        tags.append("component")
    if "page" in path_lower:
        tags.append("page")
    return tags


def _infer_responsibility(path: str, target: OptimizeTargetDefinition) -> str:
    return f"{target.subsystem_label} file: {path}"


def _build_dependency_edges(import_data: Dict[str, Any]) -> List[DependencyEdge]:
    edges: List[DependencyEdge] = []
    graph = import_data.get("graph", {})
    for source, targets in graph.items():
        if not isinstance(targets, list):
            continue
        for target in targets:
            edges.append(DependencyEdge(source=source, target=target, edge_type="import"))
    return edges


def _apply_dependency_counts(chunks: List[CodeChunk], edges: List[DependencyEdge]) -> None:
    dependents_count: Dict[str, int] = {}
    dependencies_count: Dict[str, int] = {}
    for edge in edges:
        dependencies_count[edge.source] = dependencies_count.get(edge.source, 0) + 1
        dependents_count[edge.target] = dependents_count.get(edge.target, 0) + 1
    for chunk in chunks:
        chunk.dependencies = dependencies_count.get(chunk.path, 0)
        chunk.dependents = dependents_count.get(chunk.path, 0)


def _detect_dead_code(
    chunks: List[CodeChunk],
    edges: List[DependencyEdge],
    import_graph: Dict[str, Any],
) -> List[DeadCodeCandidate]:
    del edges, import_graph
    dead_code: List[DeadCodeCandidate] = []
    for chunk in chunks:
        if any(pattern in chunk.path for pattern in DEAD_CODE_IGNORE_PATTERNS):
            continue
        if chunk.dependents == 0 and "test" not in chunk.tags and "page" not in chunk.tags:
            dead_code.append(
                DeadCodeCandidate(
                    path=chunk.path,
                    item_type="file",
                    name=chunk.name,
                    reason="No in-scope dependents found for this scoped optimisation target",
                    confidence=0.55,
                )
            )
    return dead_code
