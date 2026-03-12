# FILE: app/optimize/decomposer.py
"""
Phase A: Decomposer.

Ingests a target system and breaks it into discrete functional chunks.
Each chunk is mapped to a single responsibility with defined inputs,
outputs, and dependencies.

Leverages existing architecture data:
  - IMPORT_GRAPH.json for dependency map
  - INDEX.json for file metadata
  - TECH_DEBT.md for known large files

Outputs: ChunkManifest with chunks, dependency edges, dead code
candidates, and size audit.

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-OPT-001.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from app.optimize.config import (
    DEAD_CODE_IGNORE_PATTERNS,
    FILE_SIZE_OVERSIZED_BYTES,
    FILE_SIZE_TARGET_KB,
    HIGH_COMPLEXITY_LINES,
    IMPORT_GRAPH_PATH,
    INDEX_PATH,
)
from app.optimize.models import (
    ChunkManifest,
    CodeChunk,
    DeadCodeCandidate,
    DependencyEdge,
)

logger = logging.getLogger(__name__)


async def decompose(
    target_root: str,
    target_id: str = "astra-backend",
    emit: Optional[callable] = None,
) -> ChunkManifest:
    """Run Phase A: Decompose a target into functional chunks.

    Args:
        target_root: Root directory of the target (e.g. D:/Orb).
        target_id: Target identifier for reporting.
        emit: Progress callback.

    Returns:
        ChunkManifest with full decomposition.
    """
    emit = emit or (lambda msg: None)
    t_start = time.time()

    emit(f"🔍 Decompose: Analysing {target_id}...")

    # Load existing architecture data
    import_graph = _load_import_graph()
    index_data = _load_index()

    # Build chunks from filesystem + index
    emit("   Scanning files...")
    chunks = _scan_files(target_root, index_data)
    emit(f"   Found {len(chunks)} source files")

    # Apply dependency data from import graph
    edges = _build_dependency_edges(import_graph)
    _apply_dependency_counts(chunks, edges)
    emit(f"   Mapped {len(edges)} dependency edges")

    # Detect dead code candidates
    emit("   Detecting dead code...")
    dead_code = _detect_dead_code(chunks, edges, import_graph)
    emit(f"   Found {len(dead_code)} dead code candidates")

    # Flag oversized files
    oversized = sum(1 for c in chunks if c.is_oversized)

    # Build manifest
    manifest = ChunkManifest(
        target=target_id,
        chunks=chunks,
        dependency_edges=edges,
        dead_code=dead_code,
        total_files=len(chunks),
        total_lines=sum(c.lines for c in chunks),
        total_size_bytes=sum(c.size_bytes for c in chunks),
        oversized_files=oversized,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    duration = time.time() - t_start
    emit(f"✅ Decompose complete ({duration:.1f}s): {manifest.summary()}")

    return manifest


# ═══════════════════════════════════════════════════════════════════
# Data loaders
# ═══════════════════════════════════════════════════════════════════

def _load_import_graph() -> Dict[str, Any]:
    """Load the pre-built import graph from .architecture/."""
    try:
        path = Path(IMPORT_GRAPH_PATH)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            graph = data.get("graph", {})
            logger.info("[decompose] Loaded import graph: %d modules", len(graph))
            return data
    except Exception as e:
        logger.warning("[decompose] Failed to load import graph: %s", e)
    return {"graph": {}, "stats": {}}


def _load_index() -> Dict[str, Any]:
    """Load the architecture INDEX.json."""
    try:
        path = Path(INDEX_PATH)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            logger.info("[decompose] Loaded index: %d entries", len(data))
            return data
    except Exception as e:
        logger.warning("[decompose] Failed to load index: %s", e)
    return {}


# ═══════════════════════════════════════════════════════════════════
# File scanning
# ═══════════════════════════════════════════════════════════════════

def _scan_files(
    root: str,
    index_data: Dict[str, Any],
) -> List[CodeChunk]:
    """Scan the target directory for source files."""
    chunks = []
    root_path = Path(root)

    extensions = {".py", ".ts", ".tsx", ".kt", ".js", ".jsx"}
    skip_dirs = {"__pycache__", ".git", "node_modules", ".venv", "venv", ".pytest_cache"}

    for dirpath, dirnames, filenames in os.walk(root_path):
        # Skip excluded directories
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]

        for fname in filenames:
            fpath = Path(dirpath) / fname
            if fpath.suffix not in extensions:
                continue

            rel = str(fpath.relative_to(root_path)).replace("\\", "/")
            try:
                stat = fpath.stat()
                size = stat.st_size

                # Count lines
                try:
                    text = fpath.read_text(encoding="utf-8", errors="ignore")
                    lines = text.count("\n") + 1
                except Exception:
                    lines = 0

                # Estimate complexity from line count and nesting
                complexity = _estimate_complexity(lines, size)

                chunk = CodeChunk(
                    path=rel,
                    name=fpath.stem,
                    lines=lines,
                    size_bytes=size,
                    complexity_estimate=complexity,
                    is_oversized=size > FILE_SIZE_OVERSIZED_BYTES,
                    tags=_infer_tags(rel),
                )
                chunks.append(chunk)

            except OSError:
                continue

    return chunks


def _estimate_complexity(lines: int, size: int) -> float:
    """Rough complexity estimate from file metrics. Returns 0-1."""
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
    """Infer tags from file path."""
    tags = []
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
    if "__init__" in path_lower:
        tags.append("init")
    if "util" in path_lower or "helper" in path_lower:
        tags.append("utility")
    return tags


# ═══════════════════════════════════════════════════════════════════
# Dependency mapping
# ═══════════════════════════════════════════════════════════════════

def _build_dependency_edges(
    import_data: Dict[str, Any],
) -> List[DependencyEdge]:
    """Convert the import graph into DependencyEdge objects."""
    edges = []
    graph = import_data.get("graph", {})

    for source, targets in graph.items():
        if not isinstance(targets, list):
            continue
        for target in targets:
            edges.append(DependencyEdge(
                source=source,
                target=target,
                edge_type="import",
            ))

    return edges


def _apply_dependency_counts(
    chunks: List[CodeChunk],
    edges: List[DependencyEdge],
) -> None:
    """Apply dependency counts from edges to chunks."""
    # Build lookups
    dependents_count: Dict[str, int] = {}
    dependencies_count: Dict[str, int] = {}
    imports_map: Dict[str, List[str]] = {}

    for edge in edges:
        dependents_count[edge.target] = dependents_count.get(edge.target, 0) + 1
        dependencies_count[edge.source] = dependencies_count.get(edge.source, 0) + 1
        if edge.source not in imports_map:
            imports_map[edge.source] = []
        imports_map[edge.source].append(edge.target)

    for chunk in chunks:
        chunk.dependents = dependents_count.get(chunk.path, 0)
        chunk.dependencies = dependencies_count.get(chunk.path, 0)
        chunk.imports = imports_map.get(chunk.path, [])


# ═══════════════════════════════════════════════════════════════════
# Dead code detection
# ═══════════════════════════════════════════════════════════════════

def _detect_dead_code(
    chunks: List[CodeChunk],
    edges: List[DependencyEdge],
    import_data: Dict[str, Any],
) -> List[DeadCodeCandidate]:
    """Detect files that appear to have no dependents."""
    candidates = []

    # All files that are imported by something
    imported_files: Set[str] = set()
    for edge in edges:
        imported_files.add(edge.target)

    # Entry points that are legitimate even without dependents
    entry_patterns = [
        "main.py", "router.py", "api_router.py", "seed.py",
        "startup.py", "scheduler.py", "conftest.py",
    ]

    for chunk in chunks:
        # Skip ignored patterns
        if any(p in chunk.path for p in DEAD_CODE_IGNORE_PATTERNS):
            continue

        # Skip entry points
        if any(chunk.path.endswith(ep) for ep in entry_patterns):
            continue

        # If nobody imports this file, it's a dead code candidate
        if chunk.path not in imported_files and chunk.dependents == 0:
            candidates.append(DeadCodeCandidate(
                path=chunk.path,
                item_type="file",
                name=chunk.name,
                reason="No other module imports this file",
                confidence=0.7,
            ))

    return candidates
