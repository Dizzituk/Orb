from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

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
_BOUNDARY_TAGS = {"router", "page", "test", "startup", "service"}
_TAG_KEYWORDS = (
    ("router", "router"),
    ("model", "model"),
    ("service", "service"),
    ("test", "test"),
    ("schema", "schema"),
    ("navigation", "navigation"),
    ("viewmodel", "viewmodel"),
    ("component", "component"),
    ("page", "page"),
)
_COMPLEXITY_BANDS = (
    (50, 0.1),
    (200, 0.3),
    (HIGH_COMPLEXITY_LINES, 0.5),
    (1000, 0.7),
)
_SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "venv", ".pytest_cache", "build", ".gradle", "dist"}

async def decompose(
    target: OptimizeTargetDefinition,
    emit: Optional[callable] = None,
    protected_paths: Optional[Set[str]] = None,
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
    dead_code = _detect_dead_code(chunks, edges, import_graph, protected_paths=protected_paths)
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

    for dirpath, _, filenames in sandbox_walk(root_path, _SKIP_DIRS):
        for filename in filenames:
            rel = _build_relative_path(root_path, dirpath, filename)
            if not _is_scoped_file(rel, target):
                continue

            content_text = _read_scoped_file(dirpath, filename, sandbox_read_python)
            if content_text is None:
                continue

            chunks.append(_build_chunk(rel, filename, content_text, target, index_data))

    return chunks


def _read_scoped_file(dirpath: str, filename: str, sandbox_read_python: callable) -> Optional[str]:
    full_path = _join_path(dirpath, filename)
    return sandbox_read_python(full_path)


def _join_path(dirpath: str, filename: str) -> str:
    return dirpath.rstrip("\\").rstrip("/") + "/" + filename


def _build_relative_path(root_path: str, dirpath: str, filename: str) -> str:
    norm_full = _join_path(dirpath, filename).replace("\\", "/")
    if norm_full.startswith(root_path + "/"):
        return norm_full[len(root_path) + 1:]
    return norm_full


def _is_scoped_file(rel_path: str, target: OptimizeTargetDefinition) -> bool:
    if not should_scan_file(rel_path):
        return False
    return rel_path in filter_paths_for_target([rel_path], target)


def _build_chunk(
    rel_path: str,
    filename: str,
    content_text: str,
    target: OptimizeTargetDefinition,
    index_data: Dict[str, Any],
) -> CodeChunk:
    lines = content_text.count("\n") + 1
    size = len(content_text.encode("utf-8", errors="ignore"))
    responsibility = _resolve_responsibility(rel_path, target, index_data)

    return CodeChunk(
        path=rel_path,
        name=Path(filename).stem,
        lines=lines,
        size_bytes=size,
        complexity_estimate=_estimate_complexity(lines),
        is_oversized=size > FILE_SIZE_OVERSIZED_BYTES,
        tags=_infer_tags(rel_path),
        responsibility=responsibility,
    )


def _resolve_responsibility(
    rel_path: str,
    target: OptimizeTargetDefinition,
    index_data: Dict[str, Any],
) -> str:
    responsibility = _infer_responsibility(rel_path, target)
    index_entry = index_data.get(rel_path)
    if isinstance(index_entry, dict):
        return index_entry.get("summary", responsibility)
    return responsibility


def _estimate_complexity(lines: int) -> float:
    for threshold, score in _COMPLEXITY_BANDS:
        if lines < threshold:
            return score
    return 0.9


def _infer_tags(path: str) -> List[str]:
    path_lower = path.lower()
    return [tag for keyword, tag in _TAG_KEYWORDS if keyword in path_lower]


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
    protected_paths: Optional[Set[str]] = None,
) -> List[DeadCodeCandidate]:
    """Detect genuinely dead files within the optimisation scope.

    A file is NOT dead code if:
    - It matches an ignore pattern (config, test, init, etc.)
    - It has a boundary role tag (router, startup, service)
    - It has in-scope dependents
    - It has dependents OUTSIDE the scope (checked via import graph)
    - It was created or modified during a previous pass in this run
    """
    del edges
    scoped_paths = {chunk.path for chunk in chunks}
    externally_referenced = _find_externally_referenced_paths(scoped_paths, import_graph)
    _protected = protected_paths or set()

    dead_code: List[DeadCodeCandidate] = []
    for chunk in chunks:
        if chunk.path in _protected:
            continue
        if _should_skip_dead_code_candidate(chunk, externally_referenced):
            continue
        if _is_dead_code_candidate(chunk):
            dead_code.append(_build_dead_code_candidate(chunk))
    return dead_code


def _is_dead_code_candidate(chunk: CodeChunk) -> bool:
    return chunk.dependents == 0


def _find_externally_referenced_paths(scoped_paths: Set[str], import_graph: Dict[str, Any]) -> Set[str]:
    externally_referenced: Set[str] = set()
    for mod_info in _iter_external_module_infos(import_graph, scoped_paths):
        for imported_path in _iter_imported_module_paths(mod_info):
            matched_path = _match_scoped_import(imported_path, scoped_paths)
            if matched_path:
                externally_referenced.add(matched_path)
    return externally_referenced


def _iter_external_module_infos(
    import_graph: Dict[str, Any],
    scoped_paths: Set[str],
) -> Iterable[Any]:
    if not import_graph:
        return ()

    graph_modules = import_graph.get("modules", {})
    if not isinstance(graph_modules, dict):
        return ()

    return (
        mod_info
        for mod_path, mod_info in graph_modules.items()
        if mod_path not in scoped_paths
    )


def _iter_imported_module_paths(mod_info: Any) -> List[str]:
    if not isinstance(mod_info, dict):
        return []

    imported_paths: List[str] = []
    for imp in mod_info.get("imports", []):
        module_path = _extract_import_module_path(imp)
        if module_path:
            imported_paths.append(module_path)
    return imported_paths


def _extract_import_module_path(import_entry: Any) -> str:
    if isinstance(import_entry, str):
        return import_entry
    if isinstance(import_entry, dict):
        return import_entry.get("module", "")
    return ""


def _match_scoped_import(imported_path: str, scoped_paths: Set[str]) -> Optional[str]:
    import_path, import_name = _normalise_import_path(imported_path)
    for scoped_path in scoped_paths:
        if _paths_match_import(scoped_path, import_path, import_name):
            return scoped_path
    return None


def _normalise_import_path(imported_path: str) -> tuple[str, str]:
    import_path = imported_path.replace(".", "/") + ".py"
    import_name = import_path.rsplit('/', 1)[-1]
    return import_path, import_name


def _paths_match_import(scoped_path: str, import_path: str, import_name: str) -> bool:
    scoped_name = scoped_path.rsplit('/', 1)[-1]
    return (
        scoped_path.endswith(import_path)
        or import_path.endswith(scoped_name)
        or scoped_name == import_name
    )


def _should_skip_dead_code_candidate(chunk: CodeChunk, externally_referenced: Set[str]) -> bool:
    return any(
        predicate(chunk, externally_referenced)
        for predicate in _DEAD_CODE_SKIP_RULES
    )


def _skip_ignored_pattern(chunk: CodeChunk, externally_referenced: Set[str]) -> bool:
    del externally_referenced
    return any(pattern in chunk.path for pattern in DEAD_CODE_IGNORE_PATTERNS)


def _skip_boundary_tag(chunk: CodeChunk, externally_referenced: Set[str]) -> bool:
    del externally_referenced
    return bool(chunk.tags and _BOUNDARY_TAGS.intersection(chunk.tags))


def _skip_external_reference(chunk: CodeChunk, externally_referenced: Set[str]) -> bool:
    return chunk.path in externally_referenced


_DEAD_CODE_SKIP_RULES = (
    _skip_ignored_pattern,
    _skip_boundary_tag,
    _skip_external_reference,
)


def _build_dead_code_candidate(chunk: CodeChunk) -> DeadCodeCandidate:
    return DeadCodeCandidate(
        path=chunk.path,
        item_type="file",
        name=chunk.name,
        reason="No in-scope or external dependents found for this file",
        confidence=0.55,
    )
