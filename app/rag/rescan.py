# FILE: app/rag/rescan.py
# Purpose: Incremental RAG Rescan.
# Called-by: app.orchestrator.refactor_loop
# Depends-on: app.memory.architecture_models, app.rag.models, app.sandbox_fs
# Last-renovated: 2026-06-11
"""
Incremental RAG Rescan.

Compares the current codebase state against what's in the database.
Identifies files that have changed, been added, or been deleted since
the last scan, and updates the RAG entries accordingly.

This is the deterministic scan that runs before every refactor file pass.
No LLM calls. Pure filesystem comparison against DB state.

Usage:
    from app.rag.rescan import rescan_codebase
    
    report = rescan_codebase(db)
    # report.added, report.modified, report.deleted, report.unchanged
"""

import ast
import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.memory.architecture_models import (
    ArchitectureFileIndex,
    ArchitectureFileContent,
)
from app.rag.models import ArchCodeChunk
from app.sandbox_fs import sandbox_read_text, sandbox_isfile, sandbox_listdir

logger = logging.getLogger(__name__)

# Only scan Python files for code chunks
CODE_EXTENSIONS = {".py"}

# Directories to skip
SKIP_DIRS = {
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".eggs",
}

# Roots to scan.
# IMPORTANT: These paths are passed to sandbox_fs functions which route
# through the sandbox controller (192.168.250.2:8765). The sandbox has
# an identical D:\Orb structure. File READS come from the sandbox,
# not the host. This ensures ASTRA's code awareness matches the
# sandbox state, not the host.
SCAN_ROOTS = [
    r"D:\Orb\app",
    r"D:\Orb\main.py",
]


@dataclass
class RescanReport:
    """Result of a rescan operation."""
    timestamp: str
    files_scanned: int = 0
    added: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)
    unchanged: int = 0
    chunks_added: int = 0
    chunks_removed: int = 0
    errors: List[str] = field(default_factory=list)


def _hash_file(path: str) -> Optional[str]:
    """SHA-256 hash of file content via sandbox."""
    content = sandbox_read_text(path)
    if content is None:
        return None
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _walk_codebase() -> Dict[str, dict]:
    """
    Walk the codebase via sandbox and return all Python files.

    Returns:
        {absolute_path: {"size": int}} — sandbox doesn't provide full stat,
        so we store a minimal dict for compatibility.
    """
    files: Dict[str, dict] = {}

    def _scan_dir(dir_path: str) -> None:
        entries = sandbox_listdir(dir_path)
        for entry in entries:
            name = entry.get("name", "")
            path = entry.get("path", os.path.join(dir_path, name))
            if not name:
                continue
            if name in SKIP_DIRS:
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext == ".py":
                size = entry.get("size_bytes", 0) or 0
                files[path] = {"size": size}
            elif not ext:
                # Likely a directory — recurse
                _scan_dir(path)

    for root_path in SCAN_ROOTS:
        if root_path.endswith(".py"):
            if sandbox_isfile(root_path):
                files[root_path] = {"size": 0}
            continue
        _scan_dir(root_path)

    return files


def _extract_symbols(file_path: str, source: str) -> List[dict]:
    """
    Extract top-level symbols from Python source using AST.
    
    Returns list of dicts with keys:
        name, kind, start_line, end_line, signature, docstring,
        decorators, parameters, returns, bases, qualified_name
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    
    symbols = []
    
    for node in ast.iter_child_nodes(tree):
        sym = None
        
        if isinstance(node, ast.FunctionDef):
            sym = _extract_function(node, "function")
        elif isinstance(node, ast.AsyncFunctionDef):
            sym = _extract_function(node, "async_function")
        elif isinstance(node, ast.ClassDef):
            sym = _extract_class(node, file_path)
        
        if sym:
            symbols.append(sym)
    
    return symbols


def _extract_function(node, kind: str) -> dict:
    """Extract function/async_function info."""
    # Build signature
    args = []
    for arg in node.args.args:
        ann = ast.unparse(arg.annotation) if arg.annotation else None
        args.append(f"{arg.arg}: {ann}" if ann else arg.arg)
    
    returns = ast.unparse(node.returns) if node.returns else None
    sig = f"def {node.name}({', '.join(args)})"
    if returns:
        sig += f" -> {returns}"
    
    if kind == "async_function":
        sig = "async " + sig
    
    # Docstring
    docstring = ast.get_docstring(node) or ""
    
    # Decorators
    decorators = [ast.unparse(d) for d in node.decorator_list]
    
    return {
        "name": node.name,
        "kind": kind,
        "start_line": node.lineno,
        "end_line": node.end_lineno or node.lineno,
        "signature": sig,
        "docstring": docstring,
        "decorators": decorators,
        "parameters": args,
        "returns": returns,
        "bases": [],
        "qualified_name": node.name,
    }


def _extract_class(node, file_path: str) -> dict:
    """Extract class info including methods."""
    bases = [ast.unparse(b) for b in node.bases]
    docstring = ast.get_docstring(node) or ""
    decorators = [ast.unparse(d) for d in node.decorator_list]
    
    return {
        "name": node.name,
        "kind": "class",
        "start_line": node.lineno,
        "end_line": node.end_lineno or node.lineno,
        "signature": f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}",
        "docstring": docstring,
        "decorators": decorators,
        "parameters": [],
        "returns": None,
        "bases": bases,
        "qualified_name": node.name,
    }


def _content_hash_for_symbol(sym: dict) -> str:
    """Generate content hash for change detection."""
    content = f"{sym['name']}|{sym['signature']}|{sym['docstring']}"
    return hashlib.sha256(content.encode()).hexdigest()


def _upsert_chunks_for_file(
    db: Session,
    scan_id: int,
    file_path: str,
    source: str,
) -> int:
    """
    Replace all active chunks for a file with fresh extractions.
    
    Deletes existing active chunks, extracts new ones from source,
    creates new chunk records. Returns count of new chunks.
    """
    import json as json_mod
    
    # Remove existing active chunks for this file
    db.query(ArchCodeChunk).filter(
        and_(
            ArchCodeChunk.file_path == file_path,
            ArchCodeChunk.status == "active",
        )
    ).delete(synchronize_session="fetch")
    
    # Extract symbols from source
    symbols = _extract_symbols(file_path, source)
    
    # Create new chunks
    created = 0
    for sym in symbols:
        chunk = ArchCodeChunk(
            scan_id=scan_id,
            file_path=file_path,
            file_abs_path=file_path,
            chunk_type=sym["kind"],
            chunk_name=sym["name"],
            qualified_name=sym["qualified_name"],
            start_line=sym["start_line"],
            end_line=sym["end_line"],
            signature=sym["signature"],
            docstring=sym["docstring"],
            decorators_json=json_mod.dumps(sym["decorators"]) if sym["decorators"] else None,
            parameters_json=json_mod.dumps(sym["parameters"]) if sym["parameters"] else None,
            returns=sym["returns"],
            bases_json=json_mod.dumps(sym["bases"]) if sym["bases"] else None,
            content_hash=_content_hash_for_symbol(sym),
            embedded=False,
            status="active",
            created_at=datetime.utcnow(),
        )
        db.add(chunk)
        created += 1
    
    return created


def rescan_codebase(
    db: Session,
    scan_id: Optional[int] = None,
) -> RescanReport:
    """
    Incrementally rescan the codebase and update RAG entries.
    
    Compares filesystem state against database. For each file:
    - New file: add to file index + extract and store chunks
    - Modified file: re-extract chunks (old chunks replaced)
    - Deleted file: mark file and chunks as quarantined
    - Unchanged: skip
    
    Args:
        db: Database session
        scan_id: Architecture scan ID to use. If None, uses latest.
        
    Returns:
        RescanReport with detailed changes
    """
    report = RescanReport(timestamp=datetime.utcnow().isoformat())
    
    # Get scan_id
    if scan_id is None:
        from app.rag.models import ArchScanRun
        latest = db.query(ArchScanRun).filter(
            ArchScanRun.status == "complete"
        ).order_by(ArchScanRun.id.desc()).first()
        
        if latest:
            scan_id = latest.id
        else:
            # Fall back to architecture_scan_runs
            from app.memory.architecture_models import ArchitectureScanRun
            latest_arch = db.query(ArchitectureScanRun).filter(
                ArchitectureScanRun.status == "completed"
            ).order_by(ArchitectureScanRun.id.desc()).first()
            scan_id = latest_arch.id if latest_arch else 1
    
    # Get current filesystem state
    disk_files = _walk_codebase()
    report.files_scanned = len(disk_files)
    
    # Get current DB state (active files only)
    db_files = {}
    for fi in db.query(ArchitectureFileIndex).filter(
        and_(
            ArchitectureFileIndex.status == "active",
            ArchitectureFileIndex.ext == ".py",
        )
    ).all():
        db_files[fi.path] = fi
    
    disk_paths = set(disk_files.keys())
    db_paths = set(db_files.keys())
    
    # New files (on disk but not in DB)
    for path in sorted(disk_paths - db_paths):
        try:
            stat = disk_files[path]
            source = sandbox_read_text(path)
            if not source:
                report.errors.append(f"Add {path}: could not read from sandbox")
                continue
            content_hash = hashlib.sha256(source.encode()).hexdigest()

            # Add to file index
            fi = ArchitectureFileIndex(
                scan_id=scan_id,
                path=path,
                name=os.path.basename(path),
                ext=os.path.splitext(path)[1],
                size_bytes=len(source.encode("utf-8")),
                mtime=datetime.utcnow().isoformat(),
                zone="backend",
                root=r"D:\Orb",
                line_count=source.count("\n") + 1,
                language="python",
                status="active",
            )
            db.add(fi)
            
            # Extract and store chunks
            chunks_added = _upsert_chunks_for_file(db, scan_id, path, source)
            report.chunks_added += chunks_added
            report.added.append(path)
            
        except Exception as e:
            report.errors.append(f"Add {path}: {e}")
    
    # Deleted files (in DB but not on disk)
    for path in sorted(db_paths - disk_paths):
        fi = db_files[path]
        fi.status = "quarantined"
        fi.quarantined_at = datetime.utcnow()
        
        # Quarantine chunks
        q_count = db.query(ArchCodeChunk).filter(
            and_(
                ArchCodeChunk.file_path == path,
                ArchCodeChunk.status == "active",
            )
        ).update({"status": "quarantined"}, synchronize_session="fetch")
        
        report.chunks_removed += q_count
        report.deleted.append(path)
    
    # Existing files — check for modifications
    for path in sorted(disk_paths & db_paths):
        fi = db_files[path]
        stat = disk_files[path]
        
        # Quick check: size (sandbox doesn't give mtime reliably)
        disk_info = disk_files[path]
        disk_size = disk_info.get("size", 0)

        if fi.size_bytes == disk_size and disk_size > 0:
            report.unchanged += 1
            continue

        # File may have changed — re-extract
        try:
            source = sandbox_read_text(path)
            if not source:
                report.errors.append(f"Modify {path}: could not read from sandbox")
                continue

            # Update file index
            actual_size = len(source.encode("utf-8"))
            fi.size_bytes = actual_size
            fi.mtime = datetime.utcnow().isoformat()
            fi.line_count = source.count("\n") + 1
            
            # Replace chunks
            old_count = db.query(ArchCodeChunk).filter(
                and_(
                    ArchCodeChunk.file_path == path,
                    ArchCodeChunk.status == "active",
                )
            ).count()
            
            new_count = _upsert_chunks_for_file(db, scan_id, path, source)
            
            report.chunks_removed += old_count
            report.chunks_added += new_count
            report.modified.append(path)
            
        except Exception as e:
            report.errors.append(f"Modify {path}: {e}")
    
    db.commit()
    
    logger.info(
        f"[rescan] Complete: {len(report.added)} added, "
        f"{len(report.modified)} modified, {len(report.deleted)} deleted, "
        f"{report.unchanged} unchanged, {len(report.errors)} errors"
    )

    # Enriched architecture cards (2026-07-02): host-read ingest, one chunk per
    # card. Deliberately AFTER the sandbox-sourced code diff — cards are host
    # artifacts (see the drift_note card) and .md, so the .py diff above can
    # never quarantine them, and an empty sandbox walk doesn't block card
    # freshness. ingest_enriched_cards never raises.
    try:
        from app.rag.enriched_cards import ingest_enriched_cards
        card_stats = ingest_enriched_cards(db, scan_id=scan_id)
        report.chunks_added += card_stats.get("added", 0) + card_stats.get("updated", 0)
        report.chunks_removed += card_stats.get("removed", 0)
    except Exception as exc:
        report.errors.append(f"Enriched cards: {exc}")

    return report
