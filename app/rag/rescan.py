# FILE: app/rag/rescan.py
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

logger = logging.getLogger(__name__)

# Only scan Python files for code chunks
CODE_EXTENSIONS = {".py"}

# Directories to skip
SKIP_DIRS = {
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".eggs",
}

# Roots to scan
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
    """SHA-256 hash of file content."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except (OSError, IOError):
        return None


def _walk_codebase() -> Dict[str, os.stat_result]:
    """
    Walk the codebase and return all Python files with their stat info.
    
    Returns:
        {absolute_path: stat_result}
    """
    files = {}
    
    for root_path in SCAN_ROOTS:
        if os.path.isfile(root_path):
            if root_path.endswith(".py"):
                try:
                    files[root_path] = os.stat(root_path)
                except OSError:
                    pass
            continue
        
        for dirpath, dirnames, filenames in os.walk(root_path):
            # Skip excluded directories
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            
            for fname in filenames:
                if not fname.endswith(".py"):
                    continue
                full_path = os.path.join(dirpath, fname)
                try:
                    files[full_path] = os.stat(full_path)
                except OSError:
                    continue
    
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
            source = open(path, "r", encoding="utf-8").read()
            content_hash = hashlib.sha256(source.encode()).hexdigest()
            
            # Add to file index
            fi = ArchitectureFileIndex(
                scan_id=scan_id,
                path=path,
                name=os.path.basename(path),
                ext=os.path.splitext(path)[1],
                size_bytes=stat.st_size,
                mtime=datetime.fromtimestamp(stat.st_mtime).isoformat(),
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
        
        # Quick check: size and mtime
        disk_size = stat.st_size
        disk_mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
        
        if fi.size_bytes == disk_size and fi.mtime == disk_mtime:
            report.unchanged += 1
            continue
        
        # File changed — re-extract
        try:
            source = open(path, "r", encoding="utf-8").read()
            
            # Update file index
            fi.size_bytes = disk_size
            fi.mtime = disk_mtime
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
    
    return report
