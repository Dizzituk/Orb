# FILE: app/memory/domains/dependency_scanner.py
"""
Tier 3 — Dependency Graph.

Scans Python files in app/ for import statements and builds a
machine-readable graph of {module → [depends_on_modules]}.

CRITICAL RULE: All code scanning reads from the SANDBOX environment,
not the host filesystem. ASTRA's understanding of its own code must
come from the sandbox clone. The host D:\\Orb is only used for:
  - Writing the IMPORT_GRAPH.json output file (read-only reference)
  - The .architecture directory (static reference data)

v0.14.0: Migrated from os.walk (host) to sandbox_walk (sandbox).
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
# Import scanning (via sandbox)
# =========================================================================

def scan_imports(root_dir: str) -> dict[str, list[str]]:
    """
    Scan Python files under root_dir via the sandbox and extract
    import relationships.

    Returns a dict of {relative_path → [imported_relative_paths]}.
    Only includes imports that resolve to files within root_dir.
    """
    skip_dirs = {
        "__pycache__", ".git", "node_modules", "data", "jobs",
        ".venv", ".architecture",
    }

    # Build map of module paths → relative file paths
    module_map = _build_module_map(root_dir, skip_dirs)

    # Scan each Python file for imports
    graph: dict[str, list[str]] = {}

    for filepath, content in _iter_python_files_with_content(root_dir, skip_dirs):
        rel_path = _make_relative(filepath, root_dir)
        imports = _extract_imports_from_source(content, filepath)

        resolved = []
        for imp in imports:
            target = module_map.get(imp)
            if target and target != rel_path:
                if not _is_internal_utils(target):
                    resolved.append(target)

        if resolved:
            graph[rel_path] = sorted(set(resolved))

    return graph


def _build_module_map(root_dir: str, skip_dirs: set) -> dict[str, str]:
    """
    Map Python module dotted paths to relative file paths.

    e.g. "app.memory.router" → "app/memory/router.py"
    """
    from app.sandbox_walk import sandbox_walk

    module_map = {}

    for dirpath, dirnames, filenames in sandbox_walk(root_dir, skip_dirs):
        for fname in filenames:
            if not fname.endswith(".py"):
                continue

            filepath = dirpath.rstrip("\\") + "\\" + fname
            rel = _make_relative(filepath, root_dir)
            parts = rel.replace("\\", "/").split("/")

            if parts[-1] == "__init__.py":
                mod_parts = parts[:-1]
            else:
                mod_parts = parts[:-1] + [parts[-1].replace(".py", "")]

            dotted = ".".join(mod_parts)
            module_map[dotted] = rel.replace("\\", "/")

    return module_map


def _iter_python_files_with_content(
    root_dir: str,
    skip_dirs: set,
):
    """
    Yield (filepath, content) for all Python files via sandbox.

    Reads each file's content from the sandbox controller.
    """
    from app.sandbox_walk import sandbox_walk, sandbox_read_python

    for dirpath, dirnames, filenames in sandbox_walk(root_dir, skip_dirs):
        for fname in filenames:
            if not fname.endswith(".py"):
                continue

            filepath = dirpath.rstrip("\\") + "\\" + fname
            content = sandbox_read_python(filepath)
            if content is not None:
                yield filepath, content


def _make_relative(filepath: str, root_dir: str) -> str:
    """Make a filepath relative to root_dir."""
    # Normalise both paths
    fp = filepath.replace("/", "\\").rstrip("\\")
    rd = root_dir.replace("/", "\\").rstrip("\\")

    if fp.startswith(rd + "\\"):
        return fp[len(rd) + 1:]
    return fp


def _extract_imports_from_source(source: str, filepath: str = "") -> list[str]:
    """
    Parse Python source and extract all import module paths.

    Returns dotted module names like ["app.memory.router", "app.rag.models"].
    Ignores stdlib and third-party imports (only keeps app.* imports).
    """
    try:
        tree = ast.parse(source, filename=filepath)
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
    """Check if a file is an internal utils implementation file."""
    filename = rel_path.rsplit("/", 1)[-1] if "/" in rel_path else rel_path
    return filename.startswith("_") and "_utils" in filename


# =========================================================================
# Graph statistics
# =========================================================================

def summarise_graph(graph: dict[str, list[str]]) -> dict:
    """Generate summary statistics for the dependency graph."""
    all_targets = set()
    for targets in graph.values():
        all_targets.update(targets)

    dep_counts: dict[str, int] = {}
    for targets in graph.values():
        for t in targets:
            dep_counts[t] = dep_counts.get(t, 0) + 1

    top_deps = sorted(dep_counts.items(), key=lambda x: x[1], reverse=True)[:20]
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
    Scan imports via SANDBOX, store graph in rag_entries.

    The root_dir is used as the path prefix — the actual file reads
    go through the sandbox controller, not the host filesystem.

    Optionally saves IMPORT_GRAPH.json to the host .architecture dir
    (this is a read-only reference file, not used for code operations).
    """
    logger.info("[dependency_scanner] Scanning imports (via sandbox)...")
    graph = scan_imports(root_dir)
    stats = summarise_graph(graph)

    logger.info(
        f"[dependency_scanner] Found {stats['total_modules']} modules, "
        f"{stats['total_edges']} edges"
    )

    # Save to disk (host .architecture dir — read-only reference)
    if save_to_disk:
        disk_path = os.path.join(root_dir, ".architecture", "IMPORT_GRAPH.json")
        output = {
            "generated_at": datetime.utcnow().isoformat(),
            "source": "sandbox",
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
        f"[{TIER}:dependency_graph] IMPORT DEPENDENCY GRAPH (source: sandbox)",
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
