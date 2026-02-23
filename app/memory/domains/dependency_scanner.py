# FILE: app/memory/domains/dependency_scanner.py
"""
Tier 3 — Dependency Graph.

Scans Python files in app/ for import statements and builds a
machine-readable graph of {module → [depends_on_modules]}.

Filters out internal _*_utils_*.py edges to keep the graph clean —
dependencies point to public interfaces only.

Output stored in rag_entries and also written to disk at
.architecture/IMPORT_GRAPH.json for inspection.
"""

import ast
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.db import get_db_session
from app.memory.rag_entries_model import RAGEntry

logger = logging.getLogger(__name__)

DOMAIN = "architecture"
PROJECT = "astra-core"
TIER = "T3"


# =========================================================================
# Import scanning
# =========================================================================

def scan_imports(root_dir: str) -> dict[str, list[str]]:
    """
    Scan Python files under root_dir and extract import relationships.

    Returns a dict of {relative_path → [imported_relative_paths]}.
    Only includes imports that resolve to files within root_dir.

    Filters:
    - Skips _*_utils_*.py targets (internal implementation files).
      Dependencies point to the public interface module instead.
    - Skips __pycache__, .git, node_modules, data, jobs, .venv
    """
    root = Path(root_dir).resolve()
    skip_dirs = {"__pycache__", ".git", "node_modules", "data", "jobs", ".venv", ".architecture"}

    # Build map of module paths → relative file paths
    module_map = _build_module_map(root, skip_dirs)

    # Scan each Python file for imports
    graph: dict[str, list[str]] = {}

    for py_file in _iter_python_files(root, skip_dirs):
        rel_path = str(py_file.relative_to(root)).replace("\\", "/")
        imports = _extract_imports(py_file)

        resolved = []
        for imp in imports:
            target = module_map.get(imp)
            if target and target != rel_path:
                # Filter out internal utils targets
                if not _is_internal_utils(target):
                    resolved.append(target)

        if resolved:
            graph[rel_path] = sorted(set(resolved))

    return graph


def _build_module_map(root: Path, skip_dirs: set) -> dict[str, str]:
    """
    Map Python module dotted paths to relative file paths.

    e.g. "app.memory.router" → "app/memory/router.py"
         "app.rag" → "app/rag/__init__.py"
    """
    module_map = {}

    for py_file in _iter_python_files(root, skip_dirs):
        rel = py_file.relative_to(root)
        parts = list(rel.parts)

        # Convert path to module dotted name
        if parts[-1] == "__init__.py":
            # Package: app/memory/__init__.py → app.memory
            mod_parts = parts[:-1]
        else:
            # Module: app/memory/router.py → app.memory.router
            mod_parts = parts[:-1] + [py_file.stem]

        dotted = ".".join(mod_parts)
        rel_str = str(rel).replace("\\", "/")
        module_map[dotted] = rel_str

    return module_map


def _iter_python_files(root: Path, skip_dirs: set):
    """Yield all .py files under root, skipping excluded dirs."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for f in filenames:
            if f.endswith(".py"):
                yield Path(dirpath) / f


def _extract_imports(filepath: Path) -> list[str]:
    """
    Parse a Python file and extract all import module paths.

    Returns dotted module names like ["app.memory.router", "app.rag.models"].
    Ignores stdlib and third-party imports (only keeps app.* imports).
    """
    try:
        source = filepath.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("app."):
                    imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("app."):
                imports.append(node.module)

    return imports


def _is_internal_utils(rel_path: str) -> bool:
    """
    Check if a file is an internal utils implementation file.

    Matches patterns like _foo_utils_1.py, _service_utils.py etc.
    These are internal implementation details — dependencies should
    point to the public interface module instead.
    """
    filename = rel_path.rsplit("/", 1)[-1] if "/" in rel_path else rel_path
    # Pattern: starts with _ and contains _utils
    return filename.startswith("_") and "_utils" in filename


# =========================================================================
# Graph statistics
# =========================================================================

def summarise_graph(graph: dict[str, list[str]]) -> dict:
    """Generate summary statistics for the dependency graph."""
    all_targets = set()
    for targets in graph.values():
        all_targets.update(targets)

    # Find most-depended-on modules
    dep_counts: dict[str, int] = {}
    for targets in graph.values():
        for t in targets:
            dep_counts[t] = dep_counts.get(t, 0) + 1

    top_deps = sorted(dep_counts.items(), key=lambda x: x[1], reverse=True)[:20]

    # Find modules with most dependencies
    most_imports = sorted(graph.items(), key=lambda x: len(x[1]), reverse=True)[:20]

    return {
        "total_modules": len(graph),
        "total_edges": sum(len(v) for v in graph.values()),
        "unique_targets": len(all_targets),
        "most_depended_on": [{"module": m, "dependents": c} for m, c in top_deps],
        "most_dependencies": [{"module": m, "count": len(deps)} for m, deps in most_imports],
    }


# =========================================================================
# Storage
# =========================================================================

def store_dependency_graph(
    root_dir: str = r"D:\Orb",
    save_to_disk: bool = True,
) -> int:
    """
    Scan imports, store graph in rag_entries, optionally save to disk.

    Returns the rag_entries row ID.
    """
    logger.info("[dependency_scanner] Scanning imports...")
    graph = scan_imports(root_dir)
    stats = summarise_graph(graph)

    logger.info(
        f"[dependency_scanner] Found {stats['total_modules']} modules, "
        f"{stats['total_edges']} edges"
    )

    # Save to disk
    if save_to_disk:
        disk_path = os.path.join(root_dir, ".architecture", "IMPORT_GRAPH.json")
        output = {
            "generated_at": datetime.utcnow().isoformat(),
            "stats": stats,
            "graph": graph,
        }
        os.makedirs(os.path.dirname(disk_path), exist_ok=True)
        with open(disk_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        logger.info(f"[dependency_scanner] Saved to {disk_path}")

    # Store in rag_entries
    text = _format_graph_text(graph, stats)

    db = get_db_session()
    try:
        # Remove any existing T3 entry (replace, not append)
        existing = db.query(RAGEntry).filter(
            RAGEntry.domain == DOMAIN,
            RAGEntry.project_id == PROJECT,
            RAGEntry.chunk_text.like(f"[{TIER}:%"),
        ).all()
        for e in existing:
            db.delete(e)

        entry = RAGEntry(
            project_id=PROJECT,
            domain=DOMAIN,
            chunk_text=text,
            status="ACTIVE",
            ingest_source="dependency_scan",
            indexed_at=datetime.utcnow(),
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)
        logger.info(f"[dependency_scanner] Stored graph in rag_entries (id={entry.id})")
        return entry.id
    finally:
        db.close()


def _format_graph_text(graph: dict, stats: dict) -> str:
    """Format graph as searchable text for RAG retrieval."""
    parts = [
        f"[{TIER}:dependency_graph] IMPORT DEPENDENCY GRAPH",
        f"Modules: {stats['total_modules']}, Edges: {stats['total_edges']}",
        "",
        "MOST DEPENDED ON:",
    ]
    for item in stats["most_depended_on"][:10]:
        parts.append(f"  {item['module']} ({item['dependents']} dependents)")

    parts.append("")
    parts.append("MOST DEPENDENCIES:")
    for item in stats["most_dependencies"][:10]:
        parts.append(f"  {item['module']} ({item['count']} imports)")

    return "\n".join(parts)
