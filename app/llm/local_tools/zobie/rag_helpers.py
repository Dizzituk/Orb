# FILE: app/llm/local_tools/zobie/rag_helpers.py
# Purpose: RAG-related helpers for generating INDEX and SIGNATURES JSON, and CODEBASE.md.
# Called-by: app.llm.local_tools.zobie, app.llm.local_tools.zobie.streams.archmap_full, app.llm.local_tools.zobie.streams.scan_sandbox
# Depends-on: app.llm.local_tools.zobie.signature_extract, app.rag.jobs.embedding_job, app.rag.models
# Last-renovated: 2026-06-12
"""RAG-related helpers for generating INDEX and SIGNATURES JSON, and CODEBASE.md.

Extracted from zobie_tools.py for modularity.
No logic changes - exact same generation behavior.
"""

from __future__ import annotations

import ast
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .signature_extract import (
    extract_python_signatures,
    extract_js_signatures,
    strip_line_numbers,
    map_kind_to_chunk_type,
)

logger = logging.getLogger(__name__)


def generate_signatures_json(
    contents_data: List[Dict[str, Any]],
    repo_root: str,
) -> Dict[str, Any]:
    """
    Generate SIGNATURES JSON in format expected by RAG pipeline.
    
    Expected format:
    {
        "scan_repo_root": "D:\\Orb",
        "by_file": {
            "path/to/file.py": [
                {"name": ..., "kind": ..., "signature": ..., ...}
            ]
        }
    }
    """
    by_file: Dict[str, List[Dict]] = {}
    
    for content_info in contents_data:
        path = content_info.get("path", "")
        content = content_info.get("content", "")
        language = content_info.get("language", "")
        
        if not path or not content:
            continue
        
        if content_info.get("error"):
            continue
        
        # Strip line numbers from content before signature extraction
        # (sandbox_controller returns content with line numbers when include_line_numbers=True)
        raw_content = strip_line_numbers(content)
        
        # Extract signatures based on language
        signatures = []
        ext = os.path.splitext(path)[1].lower()
        
        if ext in (".py", ".pyw", ".pyi") or language == "python":
            signatures = extract_python_signatures(raw_content, path)
        elif ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs") or language in ("javascript", "typescript"):
            signatures = extract_js_signatures(raw_content, path)
        
        if signatures:
            by_file[path] = signatures
    
    return {
        "scan_repo_root": repo_root,
        "by_file": by_file,
        "total_files": len(by_file),
        "total_signatures": sum(len(sigs) for sigs in by_file.values()),
    }


def generate_index_for_rag(
    files_data: List[Dict[str, Any]],
    contents_data: List[Dict[str, Any]],
    repo_root: str,
) -> Dict[str, Any]:
    """
    Generate INDEX JSON in format expected by RAG pipeline.
    
    Expected format:
    {
        "scanned_files": [
            {"path": "...", "lines": N, "bytes": M}
        ]
    }
    """
    # Build content lookup
    content_by_path = {c.get("path", ""): c for c in contents_data if c.get("path")}
    
    scanned_files = []
    for f in files_data:
        path = f.get("path", "")
        content_info = content_by_path.get(path, {})
        
        scanned_files.append({
            "path": path,
            "lines": content_info.get("line_count", 0),
            "bytes": content_info.get("size_bytes") or f.get("size_bytes", 0),
            "language": content_info.get("language", ""),
        })
    
    return {
        "scan_repo_root": repo_root,
        "scanned_files": scanned_files,
        "total_files": len(scanned_files),
    }


def generate_codebase_md(
    files_data: List[Dict[str, Any]],
    contents_data: List[Dict[str, Any]],
) -> str:
    """
    Generate CODEBASE.md with all source code and line numbers.
    
    Output format:
    # CODEBASE SNAPSHOT
    Generated: 2026-01-07T22:00:00Z
    Files: 420 | Lines: 50000 | Size: 2.5MB
    
    ---
    
    ## D:/Orb/main.py
    **Language:** python | **Lines:** 245 | **Size:** 7.5KB
    ```python
      1: from fastapi import FastAPI
      2: import logging
      ...
    ```
    
    ---
    """
    # Build content lookup
    content_by_path: Dict[str, Dict[str, Any]] = {}
    for c in contents_data:
        path = c.get("path", "")
        if path:
            content_by_path[path] = c
    
    # Stats
    total_files = len(files_data)
    total_lines = sum(c.get("line_count", 0) for c in contents_data if not c.get("error"))
    total_bytes = sum(c.get("size_bytes", 0) for c in contents_data if not c.get("error"))
    files_with_content = sum(1 for c in contents_data if c.get("content") and not c.get("error"))
    
    lines = [
        "# CODEBASE SNAPSHOT",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Files:** {total_files} | **With Content:** {files_with_content} | **Lines:** {total_lines:,} | **Size:** {total_bytes / 1024 / 1024:.2f}MB",
        "",
        "---",
        "",
    ]
    
    # Group by root then sort by path
    by_root: Dict[str, List[Dict]] = {}
    for f in files_data:
        root = f.get("root", "")
        if root not in by_root:
            by_root[root] = []
        by_root[root].append(f)
    
    for root in sorted(by_root.keys()):
        root_name = os.path.basename(root) or root
        lines.append(f"# {root_name}")
        lines.append("")
        
        # Sort files by path
        files_in_root = sorted(by_root[root], key=lambda x: x.get("path", ""))
        
        for f in files_in_root:
            path = f.get("path", "")
            name = f.get("name", "")
            
            # Get content
            content_info = content_by_path.get(path, {})
            content = content_info.get("content", "")
            language = content_info.get("language", "text")
            line_count = content_info.get("line_count", 0)
            size_bytes = content_info.get("size_bytes", 0) or f.get("size_bytes", 0)
            error = content_info.get("error")
            
            # Calculate relative path
            try:
                rel_path = path.replace(root, "").lstrip("\\/").replace("\\", "/")
            except:
                rel_path = name
            
            lines.append(f"## {rel_path}")
            
            if error:
                lines.append(f"**Error:** {error}")
                lines.append("")
            elif content:
                size_kb = size_bytes / 1024 if size_bytes else 0
                lines.append(f"**Language:** {language} | **Lines:** {line_count} | **Size:** {size_kb:.1f}KB")
                lines.append(f"```{language}")
                lines.append(content)
                lines.append("```")
                lines.append("")
            else:
                lines.append("*No content captured*")
                lines.append("")
            
            lines.append("---")
            lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# IMPORT GRAPH (host-derived — written by CREATE ARCHITECTURE MAP)
# =============================================================================

def _import_graph_relpath(abs_path: str, repo_root: str) -> str:
    """Relative, forward-slash path of abs_path under repo_root."""
    rp = abs_path.replace("/", "\\")
    rr = repo_root.replace("/", "\\").rstrip("\\")
    if rp.lower().startswith((rr + "\\").lower()):
        return rp[len(rr) + 1:].replace("\\", "/")
    return rp.replace("\\", "/")


def _import_graph_is_internal_utils(rel_path: str) -> bool:
    """Internal `_*_utils*.py` split files are noise in a dependency graph."""
    filename = rel_path.rsplit("/", 1)[-1] if "/" in rel_path else rel_path
    return filename.startswith("_") and "_utils" in filename


def generate_import_graph(
    contents_data: List[Dict[str, Any]],
    repo_root: str,
) -> Dict[str, Any]:
    """
    Build an `app.*` import dependency graph from already-fetched HOST file
    contents (no sandbox dependency).

    Mirrors the schema written by app.memory.domains.dependency_scanner so the
    same readers (RAG structured context, optimizer) consume it unchanged:
        {"generated_at", "source": "host", "stats": {...}, "graph": {...}}

    Only Python files under repo_root are considered; only intra-repo `app.*`
    imports that resolve to a scanned file become edges.
    """
    # 1) Map dotted module name -> relative file path for all .py files.
    module_map: Dict[str, str] = {}
    py_files: List[tuple] = []  # (rel_path, raw_source)

    for info in contents_data:
        path = info.get("path", "")
        content = info.get("content", "")
        if not path or not content or info.get("error"):
            continue
        if os.path.splitext(path)[1].lower() != ".py":
            continue

        rel = _import_graph_relpath(path, repo_root)
        if not rel.endswith(".py"):
            continue

        parts = rel.split("/")
        if parts[-1] == "__init__.py":
            mod_parts = parts[:-1]
        else:
            mod_parts = parts[:-1] + [parts[-1][:-3]]
        dotted = ".".join(mod_parts)
        if dotted:
            module_map[dotted] = rel

        py_files.append((rel, strip_line_numbers(content)))

    # 2) Parse imports per file and resolve to scanned files.
    graph: Dict[str, List[str]] = {}
    for rel_path, source in py_files:
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            continue

        imports: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("app."):
                        imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("app."):
                    imports.append(node.module)

        resolved = []
        for imp in imports:
            target = module_map.get(imp)
            if target and target != rel_path and not _import_graph_is_internal_utils(target):
                resolved.append(target)

        if resolved:
            graph[rel_path] = sorted(set(resolved))

    # 3) Summary stats (same shape as dependency_scanner.summarise_graph).
    all_targets: set = set()
    dep_counts: Dict[str, int] = {}
    for targets in graph.values():
        all_targets.update(targets)
        for t in targets:
            dep_counts[t] = dep_counts.get(t, 0) + 1

    top_deps = sorted(dep_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    most_imports = sorted(graph.items(), key=lambda x: len(x[1]), reverse=True)[:20]

    stats = {
        "total_modules": len(graph),
        "total_edges": sum(len(v) for v in graph.values()),
        "unique_targets": len(all_targets),
        "most_depended_on": [{"module": m, "dependents": c} for m, c in top_deps],
        "most_dependencies": [{"module": m, "count": len(deps)} for m, deps in most_imports],
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "host",
        "stats": stats,
        "graph": graph,
    }


# =============================================================================
# SIGNATURES → DATABASE (shared by archmap_full + scan_sandbox)
# =============================================================================

def signatures_to_db(
    db,
    contents_data: List[Dict[str, Any]],
    base_root: str,
    source: str = "host",
) -> Optional[int]:
    """
    Extract code signatures from file contents and write to arch_code_chunks DB.

    Creates an ArchScanRun record, extracts Python/JS signatures from each file,
    and creates ArchCodeChunk entries for RAG retrieval.

    Clean-slate PER SOURCE: deletes previous ArchScanRun/ArchCodeChunk entries
    of the SAME source before writing. This avoids stale/orphaned chunks from
    renamed or deleted files, while letting host and sandbox snapshots coexist
    (2026-06-12: previously a sandbox scan wiped the host scan and vice versa).

    Args:
        db: SQLAlchemy Session
        contents_data: List of file content dicts from Phase 2
                       (each has 'path', 'content', optional 'error')
        base_root: Base path for the scan (e.g. 'D:\\Orb')
        source: Scan provenance — "host" or "sandbox" (ScanSource constants).
                Stamped on the run and every chunk for staleness labelling.

    Returns:
        rag_scan_id (int) on success, None on failure
    """
    try:
        from app.rag.models import ArchScanRun, ArchCodeChunk
        from app.rag.jobs.embedding_job import compute_content_hash
    except ImportError as ie:
        logger.warning(f"[signatures_to_db] RAG models not available: {ie}")
        return None

    try:
        # ==================================================================
        # Clean slate (same-source only): delete previous runs + chunks
        # ==================================================================
        old_runs = db.query(ArchScanRun).filter(
            ArchScanRun.source == source
        ).all()
        if old_runs:
            old_ids = [r.id for r in old_runs]
            old_chunk_ids = [
                row[0] for row in db.query(ArchCodeChunk.id).filter(
                    ArchCodeChunk.scan_id.in_(old_ids)
                ).all()
            ]
            # Delete chunks first (FK constraint)
            deleted_chunks = db.query(ArchCodeChunk).filter(
                ArchCodeChunk.scan_id.in_(old_ids)
            ).delete(synchronize_session="fetch")
            # Their embeddings are clean-slated too — otherwise stale vectors
            # keep matching and resolve to deleted chunk ids.
            deleted_embeddings = 0
            try:
                from app.embeddings.models import Embedding
                if old_chunk_ids:
                    # Chunked IN() to stay under SQLite's variable limit
                    for i in range(0, len(old_chunk_ids), 500):
                        deleted_embeddings += db.query(Embedding).filter(
                            Embedding.source_type == "arch_code_chunk",
                            Embedding.source_id.in_(old_chunk_ids[i:i + 500]),
                        ).delete(synchronize_session="fetch")
            except Exception:
                pass
            for run in old_runs:
                db.delete(run)
            db.flush()
            logger.info(
                f"[signatures_to_db] Cleaned {len(old_runs)} old {source} scan run(s), "
                f"{deleted_chunks} old chunks, {deleted_embeddings} old embeddings"
            )

        # ==================================================================
        # Create new ArchScanRun
        # ==================================================================
        rag_scan_run = ArchScanRun(
            status="running",
            signatures_file="",
            index_file="",
            source=source,
        )
        db.add(rag_scan_run)
        db.flush()  # Get the ID
        rag_scan_id = rag_scan_run.id

        logger.info(f"[signatures_to_db] Created ArchScanRun (rag_scan_id={rag_scan_id})")

        # ==================================================================
        # Extract signatures and create ArchCodeChunk entries
        # ==================================================================
        chunks_created = 0
        files_processed = 0

        for content_info in contents_data:
            path = content_info.get("path", "")
            content = content_info.get("content", "")

            if not path or not content:
                continue
            if content_info.get("error"):
                continue

            # Strip line numbers if present (archmap_full uses
            # include_line_numbers=True, scan_sandbox uses False)
            raw_content = strip_line_numbers(content)

            # Extract signatures based on file extension
            ext = os.path.splitext(path)[1].lower()
            signatures = []

            if ext in (".py", ".pyw", ".pyi"):
                signatures = extract_python_signatures(raw_content, path)
            elif ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
                signatures = extract_js_signatures(raw_content, path)

            if not signatures:
                continue

            # Create ArchCodeChunk for each signature
            for sig in signatures:
                chunk_name = sig.get("name", "") or ""
                chunk_type = map_kind_to_chunk_type(sig.get("kind", "function"))

                chunk = ArchCodeChunk(
                    scan_id=rag_scan_id,
                    file_path=path,
                    file_abs_path=path,
                    chunk_type=chunk_type,
                    chunk_name=chunk_name,
                    qualified_name=f"{path}::{chunk_name}",
                    start_line=sig.get("line"),
                    end_line=sig.get("end_line"),
                    signature=sig.get("signature"),
                    docstring=sig.get("docstring"),
                    decorators_json=(
                        json.dumps(sig.get("decorators", []))
                        if sig.get("decorators") else None
                    ),
                    parameters_json=(
                        json.dumps(sig.get("parameters", []))
                        if sig.get("parameters") else None
                    ),
                    returns=sig.get("returns"),
                    bases_json=(
                        json.dumps(sig.get("bases", []))
                        if sig.get("bases") else None
                    ),
                    embedded=False,
                    content_hash=None,  # Computed below
                    source=source,
                )

                # Compute content_hash (deterministic — no scan_id/timestamps)
                chunk.content_hash = compute_content_hash(chunk)

                db.add(chunk)
                chunks_created += 1

            files_processed += 1

            # Flush in batches to avoid memory buildup
            if files_processed % 50 == 0:
                db.flush()

        # ==================================================================
        # Finalise
        # ==================================================================
        rag_scan_run.status = "complete"
        rag_scan_run.completed_at = datetime.now(timezone.utc)
        rag_scan_run.chunks_extracted = chunks_created

        db.commit()

        logger.info(
            f"[signatures_to_db] Complete: {chunks_created} chunks from "
            f"{files_processed} files (rag_scan_id={rag_scan_id})"
        )
        return rag_scan_id

    except Exception as e:
        logger.exception(f"[signatures_to_db] Failed: {e}")
        try:
            db.rollback()
        except Exception:
            pass
        return None
