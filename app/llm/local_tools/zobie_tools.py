# FILE: app/llm/local_tools/zobie_tools.py
"""Streaming local-tool generators for architecture commands.

Commands:
- SCAN SANDBOX: Scan C:\\Users + D:\\ → save to DB only (scope="sandbox")
- UPDATE ARCHITECTURE: Scan D:\\Orb + D:\\orb-desktop → save to DB only (scope="code")  
- CREATE ARCHITECTURE MAP (lowercase): Load from DB → Claude Opus → map (no scan)
- CREATE ARCHITECTURE MAP (ALL CAPS): Full scan + read contents → CODEBASE.md + DB

v4.9 (2026-01): Add READ capability to FILESYSTEM_QUERY
  - Reads file contents from DB (no controller calls)
  - Case-insensitive path matching fallback
  - Preview limits: 200 lines / 16KB
  - Detects folders vs files, binary content
v4.8 (2026-01): FILESYSTEM_QUERY folder inference
v4.7 (2026-01): FILESYSTEM_QUERY handler for list/find
v4.6 (2026-01): Exclude Windows Store app caches from SCAN SANDBOX
  - Added AppData\Local\Packages to exclusion patterns
  - Added EBWebView, Cache_Data, ShaderCache, Cookies patterns
  - Prevents constant churn from Windows app cache updates
v4.5 (2026-01): Self-report incremental fetch file list
  - Phase 3 now emits incremental_fetch_files + incremental_fetch_list
  - Lists each file path, extension, and content=yes/no result
  - Capped at 50 files to prevent log spam
v4.4 (2026-01): FIX - Phase 3 truly incremental content fetch
  - _save_scan_incremental_to_db now returns new_paths/updated_paths lists
  - Phase 3 fetches ONLY intersection(changed_set, eligible_paths)
  - Adds required logging: eligible_for_content, incremental_fetch_count, batches
v4.3 (2026-01): FULL RAG INGEST for scan sandbox
  - Scan sandbox now fetches file contents (incremental: only new/changed)
  - Extracts code signatures → creates ArchCodeChunk entries
  - Queues background embedding job automatically
  - NO .architecture outputs (DB only)
v4.2 (2026-01): INCREMENTAL scan + exclusion filtering for scan sandbox
  - Added client-side exclusion patterns (from host_fs_scanner)
  - Incremental scan: only process new/changed files via mtime+size
  - No longer deletes all old scan data on every run
v4.1 (2026-01): FIX - Strip line numbers before signature extraction (was empty SIGNATURES.json)
v4.0 (2026-01): Added file content capture for CODEBASE.md and DB storage
v3.0 (2026-01): REWRITE - All scans use sandbox_controller /fs/tree, save to DB only
                Removed arch_query_service dependency
v2.2 (2026-01): FIXED - Separated update_architecture from sandbox_structure_scan
v2.1 (2026-01): Add generate_update_architecture_stream() alias for router compatibility
v2.0 (2025-12): Split architecture update from map generation
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import shutil
import contextlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from urllib import request as urllib_request, error as urllib_error

from sqlalchemy.orm import Session

from app.llm.audit_logger import RoutingTrace
from app.memory import schemas as memory_schemas
from app.memory import service as memory_service

# Architecture models for DB storage
try:
    from app.memory.architecture_models import (
        ArchitectureScanRun,
        ArchitectureFileIndex,
        ArchitectureFileContent,
        get_latest_scan,
        get_file_by_path,
        count_files_by_zone,
        detect_language,
        should_capture_content,
    )
    _ARCH_MODELS_AVAILABLE = True
except ImportError:
    _ARCH_MODELS_AVAILABLE = False
    ArchitectureScanRun = None
    ArchitectureFileIndex = None
    ArchitectureFileContent = None
    get_latest_scan = None
    count_files_by_zone = None
    detect_language = None
    should_capture_content = None

from app.llm.local_tools.archmap_helpers import (
    # Triggers
    _UPDATE_ARCH_TRIGGER_SET,
    _ARCHMAP_TRIGGER_SET,
    # Paths
    ARCHITECTURE_DIR,
    ARCHMAP_OUTPUT_DIR,
    ARCHMAP_OUTPUT_FILE,
    # Model config
    ARCHMAP_PROVIDER,
    ARCHMAP_MODEL,
    ARCHMAP_FALLBACK_PROVIDER,
    ARCHMAP_FALLBACK_MODEL,
    ARCHMAP_MAX_TOKENS,
    ARCHMAP_TEMPERATURE,
    # Scan config
    ZOBIE_MAPPER_SCRIPT,
    ZOBIE_MAPPER_TIMEOUT_SEC,
    # Functions
    get_architecture_dir,
    get_architecture_file,
    architecture_exists,
    load_architecture_manifest,
    load_architecture_files,
    load_architecture_enums,
    load_architecture_routes,
    load_architecture_imports,
    build_archmap_prompt,
    ARCHMAP_SYSTEM_PROMPT,
    default_controller_base_url,
    default_zobie_mapper_out_dir,
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Sandbox controller URL (inside Windows Sandbox)
SANDBOX_CONTROLLER_URL = os.getenv("ORB_SANDBOX_CONTROLLER_URL", "http://192.168.250.2:8765")

# Legacy zobie mapper settings
ZOBIE_CONTROLLER_URL = os.getenv("ORB_ZOBIE_CONTROLLER_URL") or default_controller_base_url(__file__)
ZOBIE_MAPPER_OUT_DIR = os.getenv("ORB_ZOBIE_MAPPER_OUT_DIR") or default_zobie_mapper_out_dir(__file__)
ZOBIE_MAPPER_ARGS_RAW = os.getenv("ORB_ZOBIE_MAPPER_ARGS", "200000 0 60 120000").strip()
ZOBIE_MAPPER_ARGS: List[str] = [a for a in ZOBIE_MAPPER_ARGS_RAW.split() if a]

# Scan roots - must match sandbox_controller ALLOWED_FS_ROOTS
# CODE: D:\Orb + D:\orb-desktop only
# SANDBOX: D:\ (full drive) + C:\Users\<user> 
CODE_SCAN_ROOTS = [r"D:\Orb", r"D:\orb-desktop"]  # Code repos only
SANDBOX_SCAN_ROOTS = ["D:\\", r"C:\Users\dizzi"]  # Full D: drive + user folder

# Output directory for CREATE ARCHITECTURE MAP (ALL CAPS)
# This is where INDEX.json, CODEBASE.md and ARCHITECTURE_MAP.md go
FULL_ARCHMAP_OUTPUT_DIR = r"D:\Orb\.architecture"
FULL_ARCHMAP_OUTPUT_FILE = "ARCHITECTURE_MAP.md"
FULL_CODEBASE_OUTPUT_FILE = "CODEBASE.md"

# Max file size for content capture (500KB)
MAX_CONTENT_FILE_SIZE = 500 * 1024

# v4.3: Sandbox scan content fetch settings
# Smaller batch size for sandbox (larger scale than code repos)
SANDBOX_CONTENT_BATCH_SIZE = int(os.getenv("ORB_SANDBOX_CONTENT_BATCH", "25"))
# Hard size cap for sandbox content fetch (1MB)
SANDBOX_MAX_CONTENT_SIZE = int(os.getenv("ORB_SANDBOX_MAX_CONTENT_SIZE", str(1_000_000)))

# Timeouts
FS_TREE_TIMEOUT_SEC = int(os.getenv("ORB_FS_TREE_TIMEOUT_SEC", "120"))


# =============================================================================
# EXCLUSION PATTERNS (v4.2 - from host_fs_scanner.py)
# =============================================================================

# Directory patterns to exclude (regex, matched against full path)
EXCLUDE_DIR_PATTERNS = [
    r"\.git$",
    r"\.git[/\\]",
    r"node_modules$",
    r"node_modules[/\\]",
    r"dist$",
    r"build$",
    r"\.next$",
    r"\.vite$",
    r"\.venv$",
    r"venv$",
    r"__pycache__$",
    r"__pycache__[/\\]",
    r"\.pytest_cache$",
    r"\.mypy_cache$",
    r"\.ruff_cache$",
    r"\.idea$",
    r"\.vscode$",
    r"\.tox$",
    r"\.nox$",
    r"\.eggs$",
    r"\.egg-info$",
    r"htmlcov$",
    r"\.coverage$",
    r"orb-electron-data$",
    # Windows caches and temp
    r"Code Cache$",
    r"GPUCache$",
    r"Cache$",
    r"CachedData$",
    r"CachedExtensions$",
    r"AppData[/\\]Local[/\\]Temp",
    r"AppData[/\\]Local[/\\]Microsoft",
    r"AppData[/\\]Local[/\\]Google[/\\]Chrome",
    r"AppData[/\\]Local[/\\]Mozilla",
    r"AppData[/\\]LocalLow",
    r"NTUSER\.DAT",
    # v4.6: Windows Store app caches
    r"AppData[/\\]Local[/\\]Packages",  # All Windows Store app data
    r"EBWebView",                        # Edge WebView2 caches
    r"Cache_Data",                       # Chromium cache folders
    r"ShaderCache",                      # GPU shader caches
    r"Cookies$",                         # Browser cookies folders
    # System folders
    r"\$Recycle\.Bin",
    r"System Volume Information",
]

# File extensions to exclude
EXCLUDE_FILE_EXTENSIONS = {
    # Logs and temp
    ".log",
    # Archives
    ".iso", ".vhd", ".vhdx", ".qcow2", ".img",
    ".zip", ".7z", ".rar", ".tar", ".gz", ".bz2", ".xz",
    # Databases (metadata ok but don't scan content)
    ".sqlite", ".sqlite3", ".db", ".wal", ".shm",
    # Binaries
    ".dll", ".exe", ".msi", ".sys", ".bin", ".dat",
    ".pdb", ".obj", ".o", ".a", ".so", ".dylib",
    ".pyc", ".pyo", ".class", ".jar", ".war",
    # Large media
    ".mp4", ".mkv", ".avi", ".mov", ".wmv",
    ".mp3", ".wav", ".flac", ".aac", ".ogg",
    # Large images
    ".psd", ".xcf", ".raw", ".cr2", ".nef",
}

# Compile exclusion patterns once
_EXCLUDE_DIR_RX = [re.compile(p, re.IGNORECASE) for p in EXCLUDE_DIR_PATTERNS]


def _is_excluded_path(path: str) -> bool:
    """Check if path should be excluded based on directory patterns."""
    path_norm = path.replace("\\", "/")
    for rx in _EXCLUDE_DIR_RX:
        if rx.search(path_norm):
            return True
    return False


def _is_excluded_extension(path: str) -> bool:
    """Check if file extension should be excluded."""
    ext = os.path.splitext(path.lower())[1]
    return ext in EXCLUDE_FILE_EXTENSIONS


def _filter_scan_results(files_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Filter scan results to exclude junk directories and file types.
    
    Returns: (filtered_files, excluded_count)
    """
    filtered = []
    excluded = 0
    
    for f in files_data:
        path = f.get("path", "")
        
        # Check directory exclusions
        if _is_excluded_path(path):
            excluded += 1
            continue
        
        # Check extension exclusions
        if _is_excluded_extension(path):
            excluded += 1
            continue
        
        # Check for hidden files (except allowed ones)
        name = os.path.basename(path)
        if name.startswith(".") and name not in {".env.example", ".gitignore", ".gitattributes", ".dockerignore"}:
            excluded += 1
            continue
        
        filtered.append(f)
    
    return filtered, excluded


# =============================================================================
# SSE HELPERS
# =============================================================================

def _sse_token(content: str) -> str:
    return "data: " + json.dumps({"type": "token", "content": content}) + "\n\n"


def _sse_error(error: str) -> str:
    return "data: " + json.dumps({"type": "error", "error": error}) + "\n\n"


def _sse_done(
    *,
    provider: str,
    model: str,
    total_length: int = 0,
    success: bool = True,
    error: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    payload: Dict[str, Any] = {
        "type": "done",
        "provider": provider,
        "model": model,
        "total_length": int(total_length or 0),
        "success": bool(success),
    }
    if error:
        payload["error"] = str(error)
    if meta:
        payload["meta"] = meta
    return "data: " + json.dumps(payload) + "\n\n"


# =============================================================================
# SANDBOX CONTROLLER CLIENT
# =============================================================================

def _call_fs_tree(
    roots: List[str],
    max_files: int = 100000,
    timeout: int = FS_TREE_TIMEOUT_SEC,
) -> Tuple[Optional[int], Optional[Dict[str, Any]], str]:
    """
    Call sandbox_controller /fs/tree endpoint.
    
    Returns: (status_code, response_data, error_message)
    """
    url = f"{SANDBOX_CONTROLLER_URL.rstrip('/')}/fs/tree"
    
    payload = json.dumps({
        "roots": roots,
        "max_files": max_files,
        "include_size": True,
        "include_mtime": True,
    }).encode("utf-8")
    
    try:
        req = urllib_request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.getcode() or 200, json.loads(body), ""
    except urllib_error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = str(e)
        return e.code, None, body
    except urllib_error.URLError as e:
        return None, None, f"Connection failed: {e.reason}"
    except json.JSONDecodeError as e:
        return None, None, f"Invalid JSON response: {e}"
    except Exception as e:
        return None, None, str(e)


def _call_fs_contents(
    paths: List[str],
    max_file_size: int = MAX_CONTENT_FILE_SIZE,
    include_line_numbers: bool = False,
    timeout: int = FS_TREE_TIMEOUT_SEC,
) -> Tuple[Optional[int], Optional[Dict[str, Any]], str]:
    """
    Call sandbox_controller /fs/contents endpoint to read file contents.
    
    Returns: (status_code, response_data, error_message)
    """
    url = f"{SANDBOX_CONTROLLER_URL.rstrip('/')}/fs/contents"
    
    payload = json.dumps({
        "paths": paths,
        "max_file_size": max_file_size,
        "include_line_numbers": include_line_numbers,
    }).encode("utf-8")
    
    try:
        req = urllib_request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.getcode() or 200, json.loads(body), ""
    except urllib_error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = str(e)
        return e.code, None, body
    except urllib_error.URLError as e:
        return None, None, f"Connection failed: {e.reason}"
    except json.JSONDecodeError as e:
        return None, None, f"Invalid JSON response: {e}"
    except Exception as e:
        return None, None, str(e)


def _save_scan_to_db(
    db: Session,
    scope: str,
    files_data: List[Dict[str, Any]],
    roots_scanned: List[str],
    scan_time_ms: int,
) -> Optional[int]:
    """
    Save scan results to database.
    
    OVERWRITES previous scans of the same scope (deletes old data first).
    
    Returns: scan_id or None if models not available
    """
    if not _ARCH_MODELS_AVAILABLE:
        logger.warning("[zobie_tools] Architecture models not available, cannot save to DB")
        return None
    
    # Delete existing scans of the same scope (overwrite behavior)
    try:
        old_scans = db.query(ArchitectureScanRun).filter(
            ArchitectureScanRun.scope == scope
        ).all()
        
        for old_scan in old_scans:
            # Cascade will delete ArchitectureFileIndex entries
            db.delete(old_scan)
        
        if old_scans:
            db.flush()
            logger.info(f"[zobie_tools] Deleted {len(old_scans)} old scan(s) for scope={scope}")
    except Exception as e:
        logger.warning(f"[zobie_tools] Failed to delete old scans: {e}")
    
    # Create scan run
    scan_run = ArchitectureScanRun(
        scope=scope,
        status="running",
        stats_json=json.dumps({
            "roots": roots_scanned,
            "scan_time_ms": scan_time_ms,
        }),
    )
    db.add(scan_run)
    db.flush()  # Get the ID
    
    # Add file entries in batches
    batch_size = 1000
    for i in range(0, len(files_data), batch_size):
        batch = files_data[i:i + batch_size]
        for f in batch:
            entry = ArchitectureFileIndex(
                scan_id=scan_run.id,
                path=f.get("path", ""),
                name=f.get("name", ""),
                ext=f.get("ext", ""),
                size_bytes=f.get("size_bytes"),
                mtime=f.get("mtime"),
                zone=f.get("zone", "other"),
                root=f.get("root"),
            )
            db.add(entry)
        db.flush()  # Commit batch
    
    # Mark complete
    scan_run.status = "completed"
    scan_run.finished_at = datetime.utcnow()
    scan_run.stats_json = json.dumps({
        "roots": roots_scanned,
        "scan_time_ms": scan_time_ms,
        "total_files": len(files_data),
    })
    db.commit()
    
    return scan_run.id


def _save_scan_incremental_to_db(
    db: Session,
    scope: str,
    files_data: List[Dict[str, Any]],
    roots_scanned: List[str],
    scan_time_ms: int,
) -> Tuple[Optional[int], Dict[str, int]]:
    """
    Save scan results INCREMENTALLY to database (v4.2).
    
    INCREMENTAL: Compares with existing scan and only:
    - Adds new files
    - Updates changed files (mtime or size differs)
    - Optionally marks deleted files (not implemented yet)
    
    Returns: (scan_id, stats_dict) where stats_dict contains:
        - new_files: count of newly added files
        - updated_files: count of files with changed mtime/size
        - unchanged_files: count of files that didn't change
        - total_files: total files in scan
    """
    if not _ARCH_MODELS_AVAILABLE:
        logger.warning("[zobie_tools] Architecture models not available, cannot save to DB")
        return None, {"error": "models_not_available"}
    
    stats = {
        "new_files": 0,
        "updated_files": 0,
        "unchanged_files": 0,
        "total_files": len(files_data),
        "new_paths": [],      # v4.4: Track paths for incremental content fetch
        "updated_paths": [],  # v4.4: Track paths for incremental content fetch
    }
    
    # Get the latest existing scan for this scope
    existing_scan = get_latest_scan(db, scope) if get_latest_scan else None
    
    # Build lookup of existing files by path for fast comparison
    existing_by_path: Dict[str, ArchitectureFileIndex] = {}
    if existing_scan:
        for entry in db.query(ArchitectureFileIndex).filter(
            ArchitectureFileIndex.scan_id == existing_scan.id
        ).all():
            existing_by_path[entry.path] = entry
        logger.info(f"[zobie_tools] Incremental scan: {len(existing_by_path)} existing files")
    
    # Create new scan run
    scan_run = ArchitectureScanRun(
        scope=scope,
        status="running",
        stats_json=json.dumps({
            "roots": roots_scanned,
            "scan_time_ms": scan_time_ms,
            "incremental": True,
            "previous_scan_id": existing_scan.id if existing_scan else None,
        }),
    )
    db.add(scan_run)
    db.flush()
    
    # Process files in batches
    batch_size = 1000
    for i in range(0, len(files_data), batch_size):
        batch = files_data[i:i + batch_size]
        for f in batch:
            path = f.get("path", "")
            name = f.get("name", "")
            ext = f.get("ext", "")
            size_bytes = f.get("size_bytes")
            mtime = f.get("mtime")
            zone = f.get("zone", "other")
            root = f.get("root")
            
            # Check if file exists in previous scan
            existing_entry = existing_by_path.get(path)
            
            if existing_entry:
                # File exists - check if changed
                changed = False
                
                # Compare mtime (string comparison, ISO format)
                if mtime and existing_entry.mtime:
                    if mtime != existing_entry.mtime:
                        changed = True
                
                # Compare size
                if size_bytes and existing_entry.size_bytes:
                    if size_bytes != existing_entry.size_bytes:
                        changed = True
                
                if changed:
                    stats["updated_files"] += 1
                    stats["updated_paths"].append(path)  # v4.4: Track for incremental fetch
                else:
                    stats["unchanged_files"] += 1
            else:
                # New file
                stats["new_files"] += 1
                stats["new_paths"].append(path)  # v4.4: Track for incremental fetch
            
            # Always create entry in new scan
            entry = ArchitectureFileIndex(
                scan_id=scan_run.id,
                path=path,
                name=name,
                ext=ext,
                size_bytes=size_bytes,
                mtime=mtime,
                zone=zone,
                root=root,
            )
            db.add(entry)
        db.flush()
    
    # Mark complete
    scan_run.status = "completed"
    scan_run.finished_at = datetime.utcnow()
    scan_run.stats_json = json.dumps({
        "roots": roots_scanned,
        "scan_time_ms": scan_time_ms,
        "total_files": stats["total_files"],
        "new_files": stats["new_files"],
        "updated_files": stats["updated_files"],
        "unchanged_files": stats["unchanged_files"],
        "incremental": True,
        "previous_scan_id": existing_scan.id if existing_scan else None,
    })
    
    # Delete old scan ONLY after new scan is complete
    # This ensures we never lose data if scan fails mid-way
    if existing_scan:
        try:
            db.delete(existing_scan)  # Cascade deletes ArchitectureFileIndex entries
            logger.info(f"[zobie_tools] Deleted old scan_id={existing_scan.id} after successful incremental update")
        except Exception as e:
            logger.warning(f"[zobie_tools] Failed to delete old scan: {e}")
    
    db.commit()
    
    return scan_run.id, stats


def _save_scan_with_contents_to_db(
    db: Session,
    scope: str,
    files_data: List[Dict[str, Any]],
    contents_data: List[Dict[str, Any]],
    roots_scanned: List[str],
    scan_time_ms: int,
) -> Optional[int]:
    """
    Save scan results WITH file contents to database.
    
    OVERWRITES previous scans of the same scope (deletes old data first).
    
    Args:
        db: Database session
        scope: "code" or "sandbox"
        files_data: List of file metadata from /fs/tree
        contents_data: List of file contents from /fs/contents
        roots_scanned: List of root directories scanned
        scan_time_ms: Total scan time
        
    Returns: scan_id or None if models not available
    """
    if not _ARCH_MODELS_AVAILABLE:
        logger.warning("[zobie_tools] Architecture models not available, cannot save to DB")
        return None
    
    # Build content lookup by path
    content_by_path: Dict[str, Dict[str, Any]] = {}
    for c in contents_data:
        path = c.get("path", "")
        if path and not c.get("error"):
            content_by_path[path] = c
    
    # Delete existing scans of the same scope (overwrite behavior)
    try:
        old_scans = db.query(ArchitectureScanRun).filter(
            ArchitectureScanRun.scope == scope
        ).all()
        
        for old_scan in old_scans:
            db.delete(old_scan)
        
        if old_scans:
            db.flush()
            logger.info(f"[zobie_tools] Deleted {len(old_scans)} old scan(s) for scope={scope}")
    except Exception as e:
        logger.warning(f"[zobie_tools] Failed to delete old scans: {e}")
    
    # Create scan run
    total_lines = sum(c.get("line_count", 0) for c in contents_data if not c.get("error"))
    
    scan_run = ArchitectureScanRun(
        scope=scope,
        status="running",
        stats_json=json.dumps({
            "roots": roots_scanned,
            "scan_time_ms": scan_time_ms,
        }),
    )
    db.add(scan_run)
    db.flush()
    
    # Add file entries with content
    files_with_content = 0
    batch_size = 100  # Smaller batch for content
    
    for i in range(0, len(files_data), batch_size):
        batch = files_data[i:i + batch_size]
        for f in batch:
            path = f.get("path", "")
            name = f.get("name", "")
            ext = f.get("ext", "")
            
            # Get content if available
            content_info = content_by_path.get(path, {})
            line_count = content_info.get("line_count")
            language = content_info.get("language")
            
            # Fallback language detection
            if not language and detect_language:
                language = detect_language(ext, name)
            
            entry = ArchitectureFileIndex(
                scan_id=scan_run.id,
                path=path,
                name=name,
                ext=ext,
                size_bytes=f.get("size_bytes"),
                mtime=f.get("mtime"),
                zone=f.get("zone", "other"),
                root=f.get("root"),
                line_count=line_count,
                language=language,
            )
            db.add(entry)
            db.flush()
            
            # Add content if available
            if content_info.get("content"):
                content_entry = ArchitectureFileContent(
                    file_index_id=entry.id,
                    content_text=content_info["content"],
                    content_hash=content_info.get("content_hash"),
                )
                db.add(content_entry)
                files_with_content += 1
        
        db.flush()
    
    # Mark complete
    scan_run.status = "completed"
    scan_run.finished_at = datetime.utcnow()
    scan_run.stats_json = json.dumps({
        "roots": roots_scanned,
        "scan_time_ms": scan_time_ms,
        "total_files": len(files_data),
        "files_with_content": files_with_content,
        "total_lines": total_lines,
    })
    db.commit()
    
    logger.info(f"[zobie_tools] Saved scan: {len(files_data)} files, {files_with_content} with content, {total_lines} lines")
    
    return scan_run.id


# =============================================================================
# SIGNATURE EXTRACTION FOR RAG (v4.1)
# =============================================================================

def _extract_python_signatures(content: str, file_path: str) -> List[Dict[str, Any]]:
    """
    Extract function and class signatures from Python source code.
    
    Returns list of signature dicts matching RAG pipeline expected format:
    {
        "name": "function_name",
        "kind": "function|async_function|class|method|async_method",
        "signature": "(arg1, arg2)",
        "line": 10,
        "end_line": 25,
        "docstring": "...",
        "decorators": ["@decorator"],
        "parameters": [],
        "returns": "str",
        "bases": ["BaseClass"]  # for classes
    }
    """
    signatures = []
    lines = content.splitlines()
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Skip empty and comment lines
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        
        # Detect decorators
        decorators = []
        while stripped.startswith("@"):
            decorators.append(stripped)
            i += 1
            if i >= len(lines):
                break
            line = lines[i]
            stripped = line.strip()
        
        # Detect class definition
        class_match = re.match(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\s*(\([^)]*\))?\s*:", stripped)
        if class_match:
            name = class_match.group(1)
            bases_str = class_match.group(2) or ""
            bases = []
            if bases_str:
                # Extract base class names
                bases_str = bases_str.strip("()")
                bases = [b.strip() for b in bases_str.split(",") if b.strip()]
            
            # Get docstring
            docstring = _extract_docstring(lines, i + 1)
            
            # Find end of class (crude: next dedented non-empty line or EOF)
            end_line = _find_block_end(lines, i)
            
            signatures.append({
                "name": name,
                "kind": "class",
                "signature": bases_str,
                "line": i + 1,
                "end_line": end_line,
                "docstring": docstring,
                "decorators": decorators,
                "parameters": [],
                "returns": None,
                "bases": bases,
            })
            i += 1
            continue
        
        # Detect function/method definition
        func_match = re.match(r"^(async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*(->\s*([^:]+))?\s*:", stripped)
        if func_match:
            is_async = bool(func_match.group(1))
            name = func_match.group(2)
            params_str = func_match.group(3) or ""
            returns = (func_match.group(5) or "").strip() if func_match.group(4) else None
            
            # Determine if method (has self/cls as first param)
            is_method = params_str.strip().startswith(("self", "cls"))
            
            if is_async:
                kind = "async_method" if is_method else "async_function"
            else:
                kind = "method" if is_method else "function"
            
            # Get docstring
            docstring = _extract_docstring(lines, i + 1)
            
            # Find end of function
            end_line = _find_block_end(lines, i)
            
            signatures.append({
                "name": name,
                "kind": kind,
                "signature": f"({params_str})",
                "line": i + 1,
                "end_line": end_line,
                "docstring": docstring,
                "decorators": decorators,
                "parameters": _parse_params(params_str),
                "returns": returns,
                "bases": [],
            })
        
        i += 1
    
    return signatures


def _extract_docstring(lines: List[str], start_idx: int) -> Optional[str]:
    """Extract docstring starting from given line index."""
    if start_idx >= len(lines):
        return None
    
    line = lines[start_idx].strip()
    
    # Single-line docstring
    if line.startswith(('"""', "'''")) and line.count('"""') >= 2:
        return line.strip('"""').strip("'''").strip()
    if line.startswith(('"""', "'''")):
        # Multi-line docstring
        quote = line[:3]
        doc_lines = [line[3:]]
        for i in range(start_idx + 1, min(start_idx + 50, len(lines))):
            doc_lines.append(lines[i])
            if quote in lines[i]:
                break
        full = "\n".join(doc_lines)
        return full.replace(quote, "").strip()[:500]  # Limit length
    
    return None


def _find_block_end(lines: List[str], start_idx: int) -> int:
    """Find end line of a Python block (class or function)."""
    if start_idx >= len(lines):
        return start_idx + 1
    
    # Get indentation of the def/class line
    start_line = lines[start_idx]
    base_indent = len(start_line) - len(start_line.lstrip())
    
    end_line = start_idx + 1
    for i in range(start_idx + 1, len(lines)):
        line = lines[i]
        if not line.strip():  # Empty line
            end_line = i + 1
            continue
        
        current_indent = len(line) - len(line.lstrip())
        if current_indent <= base_indent and line.strip():
            # Found a line at same or lower indentation - block ended
            break
        end_line = i + 1
    
    return end_line


def _parse_params(params_str: str) -> List[Dict[str, Any]]:
    """Parse parameter string into list of param dicts."""
    if not params_str.strip():
        return []
    
    params = []
    # Simple split - won't handle complex type annotations perfectly
    for p in params_str.split(","):
        p = p.strip()
        if not p:
            continue
        
        # Extract name and type annotation
        if ":" in p:
            name_part, type_part = p.split(":", 1)
            name = name_part.strip()
            if "=" in type_part:
                type_ann, default = type_part.split("=", 1)
                params.append({"name": name, "type": type_ann.strip(), "default": default.strip()})
            else:
                params.append({"name": name, "type": type_part.strip()})
        elif "=" in p:
            name, default = p.split("=", 1)
            params.append({"name": name.strip(), "default": default.strip()})
        else:
            params.append({"name": p})
    
    return params


def _extract_js_signatures(content: str, file_path: str) -> List[Dict[str, Any]]:
    """Extract function/class signatures from JavaScript/TypeScript."""
    signatures = []
    lines = content.splitlines()
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Detect class
        class_match = re.match(r"^(?:export\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+extends\s+([A-Za-z_][A-Za-z0-9_]*))?\s*\{", stripped)
        if class_match:
            name = class_match.group(1)
            base = class_match.group(2)
            signatures.append({
                "name": name,
                "kind": "class",
                "signature": "",
                "line": i + 1,
                "end_line": i + 1,
                "docstring": None,
                "decorators": [],
                "parameters": [],
                "returns": None,
                "bases": [base] if base else [],
            })
            continue
        
        # Detect function
        func_match = re.match(r"^(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)", stripped)
        if func_match:
            name = func_match.group(1)
            params = func_match.group(2)
            is_async = "async" in stripped.split("function")[0]
            signatures.append({
                "name": name,
                "kind": "async_function" if is_async else "function",
                "signature": f"({params})",
                "line": i + 1,
                "end_line": i + 1,
                "docstring": None,
                "decorators": [],
                "parameters": [],
                "returns": None,
                "bases": [],
            })
            continue
        
        # Detect const arrow function
        arrow_match = re.match(r"^(?:export\s+)?const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s+)?\(([^)]*)\)\s*(?::\s*[^=]+)?\s*=>", stripped)
        if arrow_match:
            name = arrow_match.group(1)
            params = arrow_match.group(2)
            is_async = "async" in stripped.split("=")[0] or "async" in stripped.split("const")[1].split("=")[0]
            signatures.append({
                "name": name,
                "kind": "async_function" if is_async else "function",
                "signature": f"({params})",
                "line": i + 1,
                "end_line": i + 1,
                "docstring": None,
                "decorators": [],
                "parameters": [],
                "returns": None,
                "bases": [],
            })
    
    return signatures


def _strip_line_numbers(content: str) -> str:
    """
    Strip line number prefixes from content.
    
    Handles formats like:
    - "1: code"
    - "  1: code" 
    - " 10: code"
    - "100: code"
    """
    lines = content.splitlines()
    stripped_lines = []
    
    for line in lines:
        # Match optional whitespace, digits, colon, optional single space
        match = re.match(r'^\s*\d+:\s?', line)
        if match:
            stripped_lines.append(line[match.end():])
        else:
            stripped_lines.append(line)
    
    return "\n".join(stripped_lines)


def _generate_signatures_json(
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
        raw_content = _strip_line_numbers(content)
        
        # Extract signatures based on language
        signatures = []
        ext = os.path.splitext(path)[1].lower()
        
        if ext in (".py", ".pyw", ".pyi") or language == "python":
            signatures = _extract_python_signatures(raw_content, path)
        elif ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs") or language in ("javascript", "typescript"):
            signatures = _extract_js_signatures(raw_content, path)
        
        if signatures:
            by_file[path] = signatures
    
    return {
        "scan_repo_root": repo_root,
        "by_file": by_file,
        "total_files": len(by_file),
        "total_signatures": sum(len(sigs) for sigs in by_file.values()),
    }


def _generate_index_for_rag(
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


def _generate_codebase_md(
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
    
    ## D:\Orb\main.py
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
# SCAN SANDBOX (scope="sandbox") - DB + RAG chunks + embeddings (v4.3)
# =============================================================================

# v4.3: Content extensions for sandbox scan (same as generate_full_architecture_map_stream)
# IMPORTANT: Do NOT broaden this - sandbox scans D:\ + C:\Users which is huge
_SANDBOX_CONTENT_EXTENSIONS = {
    ".py", ".pyw", ".pyi",
    ".js", ".mjs", ".cjs", ".jsx",
    ".ts", ".tsx", ".mts", ".cts",
    ".json", ".jsonc",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".sql", ".sh", ".bash", ".zsh", ".ps1", ".psm1", ".bat", ".cmd",
    ".md", ".markdown", ".rst", ".txt",
    "",  # Files without extension (like Dockerfile, Makefile)
}

# v4.3: Files to NEVER capture (secrets, credentials, keys)
_SANDBOX_SKIP_FILENAMES = {
    ".env", ".env.local", ".env.production", ".env.development",
    ".env.example",
    "secrets.json", "credentials.json", "config.secret.json",
    ".npmrc", ".pypirc",
    "id_rsa", "id_ed25519", "id_ecdsa",
    ".pem", ".key", ".crt", ".p12", ".pfx",
}

# v4.3: Patterns in filename to skip
_SANDBOX_SKIP_PATTERNS = {"secret", "credential", "password", "token", "apikey", "api_key"}


def _map_kind_to_chunk_type(kind: str) -> str:
    """
    Map signature 'kind' to ArchCodeChunk chunk_type.
    
    Expected kinds from _extract_python_signatures / _extract_js_signatures:
    - function, async_function, class, method, async_method
    """
    return kind  # Direct mapping - kinds match ChunkType values


async def generate_sandbox_structure_scan_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """
    Scan sandbox environment (C:\\Users + D:\\ areas) and save to DB.
    
    v4.3: FULL RAG INGEST
    - Scans file tree (incremental: mtime+size)
    - Fetches contents for NEW/CHANGED code files only
    - Extracts signatures → creates ArchCodeChunk entries
    - Queues background embedding job
    - NO .architecture outputs (DB only)
    """
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    
    yield _sse_token("🔍 [SCAN_SANDBOX] Scanning sandbox environment...\n")
    yield _sse_token(f"📡 Controller: {SANDBOX_CONTROLLER_URL}\n")
    yield _sse_token(f"📂 Roots: {', '.join(SANDBOX_SCAN_ROOTS)}\n\n")
    
    # Check if models available
    if not _ARCH_MODELS_AVAILABLE:
        yield _sse_error(
            "Architecture models not available. "
            "Run: alembic upgrade head to create tables."
        )
        yield _sse_done(
            provider="local",
            model="sandbox_scanner",
            success=False,
            error="models_not_available",
        )
        return
    
    # ==========================================================================
    # Phase 1: Scan file tree
    # ==========================================================================
    yield _sse_token("📊 Phase 1: Scanning file tree...\n")
    
    status, data, error_msg = await loop.run_in_executor(
        None,
        lambda: _call_fs_tree(SANDBOX_SCAN_ROOTS, max_files=200000),
    )
    
    if status != 200 or data is None:
        logger.error(f"[SCAN_SANDBOX] Failed: status={status}, error={error_msg}")
        
        if status == 404:
            yield _sse_error(
                f"Sandbox controller /fs/tree not found at {SANDBOX_CONTROLLER_URL}\n"
                f"Please update sandbox_controller to v0.3.0 or later."
            )
        elif status is None:
            yield _sse_error(
                f"Could not connect to sandbox controller at {SANDBOX_CONTROLLER_URL}\n"
                f"Error: {error_msg}\n"
                f"Is the sandbox running?"
            )
        else:
            yield _sse_error(f"Scan failed (status={status}): {error_msg}")
        
        yield _sse_done(
            provider="local",
            model="sandbox_scanner",
            success=False,
            error=f"status={status}",
        )
        return
    
    # Extract file data
    raw_files_data = data.get("files", [])
    roots_scanned = data.get("roots_scanned", SANDBOX_SCAN_ROOTS)
    scan_time_ms = data.get("scan_time_ms", 0)
    truncated = data.get("truncated", False)
    
    yield _sse_token(f"   Found {len(raw_files_data)} files in {scan_time_ms}ms\n")
    
    # v4.2: Apply exclusion filtering
    files_data, excluded_count = _filter_scan_results(raw_files_data)
    yield _sse_token(f"   Excluded {excluded_count} junk files (caches, node_modules, binaries, etc.)\n")
    yield _sse_token(f"   Keeping {len(files_data)} relevant files\n")
    
    if truncated:
        yield _sse_token("   ⚠️ Results truncated (max files limit reached)\n")
    
    # ==========================================================================
    # Phase 2: Save file metadata to DB (incremental)
    # ==========================================================================
    yield _sse_token("\n💾 Phase 2: Saving metadata to DB (incremental)...\n")
    
    scan_id = None
    stats = {}
    
    try:
        scan_id, stats = _save_scan_incremental_to_db(
            db=db,
            scope="sandbox",
            files_data=files_data,
            roots_scanned=roots_scanned,
            scan_time_ms=scan_time_ms,
        )
        
        if scan_id:
            zone_counts = count_files_by_zone(db, scan_id) if count_files_by_zone else {}
            
            yield _sse_token(f"   ✅ Metadata saved (scan_id={scan_id})\n")
            yield _sse_token(f"   New: {stats.get('new_files', 0)} | Updated: {stats.get('updated_files', 0)} | Unchanged: {stats.get('unchanged_files', 0)}\n")
            
            # Log to console for visibility
            logger.info(f"[SCAN_SANDBOX] scanned_files={len(files_data)}")
            logger.info(f"[SCAN_SANDBOX] db_upserts={scan_id}")
        else:
            yield _sse_token("   ⚠️ Could not save to DB (models not available)\n")
            yield _sse_done(
                provider="local",
                model="sandbox_scanner",
                success=False,
                error="models_not_available",
            )
            return
            
    except Exception as e:
        logger.exception(f"[SCAN_SANDBOX] DB save failed: {e}")
        yield _sse_error(f"Failed to save to DB: {e}")
        yield _sse_done(
            provider="local",
            model="sandbox_scanner",
            success=False,
            error=str(e),
        )
        return
    
    # ==========================================================================
    # Phase 3: Fetch contents for NEW/CHANGED files only (incremental)
    # ==========================================================================
    yield _sse_token("\n📖 Phase 3: Fetching contents (incremental)...\n")
    
    # Build set of paths that need content fetch (only new or changed)
    # Use incremental logic: only NEW files and UPDATED files need content fetch
    new_count = stats.get("new_files", 0)
    updated_count = stats.get("updated_files", 0)
    
    if new_count == 0 and updated_count == 0:
        yield _sse_token("   ⏭️ No new/changed files - skipping content fetch\n")
        paths_to_read = []
        eligible_paths = []  # For logging
        # Required logging for acceptance test (no changes = no fetch needed)
        yield _sse_token(f"   [SCAN_SANDBOX] eligible_for_content=0\n")
        yield _sse_token(f"   [SCAN_SANDBOX] incremental_fetch_count=0\n")
    else:
        # v4.4: TRUE INCREMENTAL content fetch
        # Build set of changed paths from Phase 2 (new + updated files only)
        changed_set = set(stats.get("new_paths", [])) | set(stats.get("updated_paths", []))
        
        yield _sse_token(f"   {new_count} new + {updated_count} changed files detected\n")
        
        # Build list of content-eligible files (apply size/extension/secret filters)
        eligible_paths = []
        for f in files_data:
            ext = (f.get("ext") or "").lower()
            size = f.get("size_bytes") or 0
            name = (f.get("name") or "").lower()
            path = f.get("path", "")
            
            # v4.3: Hard size cap (1MB) for safety
            if size > SANDBOX_MAX_CONTENT_SIZE:
                continue
            
            # Skip sensitive files
            if name in _SANDBOX_SKIP_FILENAMES:
                continue
            
            # Skip files with sensitive patterns in name
            if any(p in name for p in _SANDBOX_SKIP_PATTERNS):
                continue
            
            # Include by extension
            if ext in _SANDBOX_CONTENT_EXTENSIONS:
                eligible_paths.append(path)
                continue
            
            # Include special files without matching extension
            if name in (".gitignore", ".gitattributes", "dockerfile", "makefile"):
                eligible_paths.append(path)
        
        # v4.4: INCREMENTAL - Only fetch paths that are BOTH eligible AND changed
        paths_to_read = [p for p in eligible_paths if p in changed_set]
        
        # Required logging for acceptance test
        yield _sse_token(f"   [SCAN_SANDBOX] eligible_for_content={len(eligible_paths)}\n")
        yield _sse_token(f"   [SCAN_SANDBOX] incremental_fetch_count={len(paths_to_read)}\n")
    
    # Fetch contents in batches (v4.3: batch_size=25 for sandbox scale)
    contents_data: List[Dict[str, Any]] = []
    
    if paths_to_read:
        batch_size = SANDBOX_CONTENT_BATCH_SIZE  # Default 25
        total_batches = (len(paths_to_read) + batch_size - 1) // batch_size
        
        # Required logging for acceptance test
        yield _sse_token(f"   [SCAN_SANDBOX] batches={total_batches}\n")
        
        for i in range(0, len(paths_to_read), batch_size):
            batch_paths = paths_to_read[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            status, resp, error_msg = await loop.run_in_executor(
                None,
                lambda bp=batch_paths: _call_fs_contents(
                    bp, 
                    max_file_size=SANDBOX_MAX_CONTENT_SIZE,
                    include_line_numbers=False,  # Don't need line numbers for signatures
                ),
            )
            
            if status == 200 and resp:
                batch_files = resp.get("files", [])
                contents_data.extend(batch_files)
                
                if batch_num % 10 == 0 or batch_num == total_batches:
                    yield _sse_token(f"   Batch {batch_num}/{total_batches}: {len(contents_data)} files fetched\n")
            else:
                yield _sse_token(f"   Batch {batch_num}: Failed - {error_msg}\n")
        
        files_with_content = sum(1 for f in contents_data if f.get("content") and not f.get("error"))
        yield _sse_token(f"   ✅ Fetched {files_with_content} files with content\n")
        
        # v4.5: Self-report incremental fetch file list
        yield _sse_token(f"   [SCAN_SANDBOX] incremental_fetch_files={len(paths_to_read)}\n")
        # Build lookup of content fetch results
        content_results = {c.get("path", ""): c for c in contents_data}
        
        yield _sse_token("   [SCAN_SANDBOX] incremental_fetch_list:\n")
        max_to_print = 50
        for idx, path in enumerate(paths_to_read[:max_to_print]):
            ext = os.path.splitext(path)[1] or "(no ext)"
            result = content_results.get(path, {})
            has_content = "yes" if (result.get("content") and not result.get("error")) else "no"
            yield _sse_token(f"    - {path} ({ext}) content={has_content}\n")
        
        if len(paths_to_read) > max_to_print:
            yield _sse_token(f"    … ({len(paths_to_read) - max_to_print} more)\n")
    else:
        # No paths to fetch - log batches=0 for acceptance test
        yield _sse_token(f"   [SCAN_SANDBOX] batches=0\n")
        yield _sse_token(f"   [SCAN_SANDBOX] incremental_fetch_files=0\n")
        yield _sse_token("   [SCAN_SANDBOX] incremental_fetch_list: <none>\n")
    
    # ==========================================================================
    # Phase 4: Extract signatures → ArchCodeChunk entries
    # ==========================================================================
    yield _sse_token("\n🔗 Phase 4: Extracting signatures for RAG...\n")
    
    chunks_created = 0
    rag_scan_id = None
    
    if contents_data:
        try:
            # Import RAG models
            from app.rag.models import ArchScanRun, ArchCodeChunk
            from app.rag.jobs.embedding_job import compute_content_hash
            
            # Create ArchScanRun entry for RAG pipeline tracking
            # (This is separate from architecture_scan_runs used for file metadata)
            rag_scan_run = ArchScanRun(
                status="running",
                signatures_file="",  # No file output for sandbox scan
                index_file="",
            )
            db.add(rag_scan_run)
            db.flush()  # Get the ID
            rag_scan_id = rag_scan_run.id
            
            yield _sse_token(f"   Created ArchScanRun (rag_scan_id={rag_scan_id})\n")
            
            # Process each file with content
            for content_info in contents_data:
                path = content_info.get("path", "")
                content = content_info.get("content", "")
                
                if not path or not content:
                    continue
                if content_info.get("error"):
                    continue
                
                # Strip line numbers if present (safety)
                raw_content = _strip_line_numbers(content)
                
                # Extract signatures based on file extension
                ext = os.path.splitext(path)[1].lower()
                signatures = []
                
                if ext in (".py", ".pyw", ".pyi"):
                    signatures = _extract_python_signatures(raw_content, path)
                elif ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
                    signatures = _extract_js_signatures(raw_content, path)
                
                # Create ArchCodeChunk for each signature
                for sig in signatures:
                    chunk = ArchCodeChunk(
                        scan_id=rag_scan_id,
                        file_path=path,
                        file_abs_path=path,  # Same for sandbox paths
                        chunk_type=_map_kind_to_chunk_type(sig.get("kind", "function")),
                        chunk_name=sig.get("name", ""),
                        qualified_name=f"{path}::{sig.get('name', '')}",
                        start_line=sig.get("line"),
                        end_line=sig.get("end_line"),
                        signature=sig.get("signature"),
                        docstring=sig.get("docstring"),
                        decorators_json=json.dumps(sig.get("decorators", [])) if sig.get("decorators") else None,
                        parameters_json=json.dumps(sig.get("parameters", [])) if sig.get("parameters") else None,
                        returns=sig.get("returns"),
                        bases_json=json.dumps(sig.get("bases", [])) if sig.get("bases") else None,
                        embedded=False,  # Will be embedded by background job
                    )
                    
                    # Compute content hash for change detection
                    chunk.content_hash = compute_content_hash(chunk)
                    
                    db.add(chunk)
                    chunks_created += 1
                
                # Flush periodically to avoid memory buildup
                if chunks_created % 500 == 0:
                    db.flush()
            
            # Mark scan complete
            rag_scan_run.status = "complete"
            rag_scan_run.completed_at = datetime.utcnow()
            rag_scan_run.chunks_extracted = chunks_created
            
            db.commit()
            
            yield _sse_token(f"   ✅ Created {chunks_created} ArchCodeChunk entries\n")
            logger.info(f"[SCAN_SANDBOX] chunks_written={chunks_created}")
            
        except ImportError as ie:
            logger.warning(f"[SCAN_SANDBOX] RAG models not available: {ie}")
            yield _sse_token(f"   ⚠️ RAG models not available: {ie}\n")
        except Exception as e:
            logger.exception(f"[SCAN_SANDBOX] Signature extraction failed: {e}")
            yield _sse_token(f"   ⚠️ Signature extraction failed: {e}\n")
            # Don't fail the whole scan - continue to summary
    else:
        yield _sse_token("   ⏭️ No content to process - skipping signature extraction\n")
    
    # ==========================================================================
    # Phase 5: Queue background embedding job
    # ==========================================================================
    yield _sse_token("\n🚀 Phase 5: Queueing embedding job...\n")
    
    embedding_queued = False
    
    if chunks_created > 0:
        try:
            from app.rag.jobs.embedding_job import queue_embedding_job, EMBEDDING_AUTO_ENABLED
            from app.db import get_db_session
            
            if not EMBEDDING_AUTO_ENABLED:
                yield _sse_token("   ⚠️ Auto-embedding disabled (ORB_EMBEDDING_AUTO=false)\n")
            else:
                # Queue embedding for ALL pending chunks (not filtered by scan_id)
                # This ensures any previously unembedded chunks also get processed
                embedding_queued = queue_embedding_job(
                    db_session_factory=get_db_session,
                    scan_id=None,  # Embed all pending chunks
                )
                
                if embedding_queued:
                    yield _sse_token("   ✅ Embedding job queued (background)\n")
                    yield _sse_token("   📊 Priority: Tier1 (routers) → Tier2 (pipeline) → Tier3 (services) → ...\n")
                else:
                    yield _sse_token("   ⚠️ Embedding job not queued (may already be running)\n")
                    
        except ImportError as ie:
            logger.warning(f"[SCAN_SANDBOX] Embedding module not available: {ie}")
            yield _sse_token(f"   ⚠️ Embedding module not available: {ie}\n")
        except Exception as emb_err:
            logger.warning(f"[SCAN_SANDBOX] Failed to queue embedding job (non-fatal): {emb_err}")
            yield _sse_token(f"   ⚠️ Embedding queue failed (non-fatal): {emb_err}\n")
    else:
        yield _sse_token("   ⏭️ No chunks to embed\n")
    
    logger.info(f"[SCAN_SANDBOX] embeddings_queued={embedding_queued}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    
    # Record in memory service
    try:
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id,
                role="assistant",
                content=f"[SCAN_SANDBOX] Indexed {len(files_data)} files, {chunks_created} chunks (scan_id={scan_id}, rag_scan_id={rag_scan_id})",
                provider="local",
                model="sandbox_scanner",
            ),
        )
    except Exception:
        pass
    
    duration_ms = int(loop.time() * 1000) - started_ms
    
    if trace:
        trace.log_model_call(
            "local_tool", "local", "sandbox_scanner", "scan_sandbox",
            0, 0, duration_ms, success=True, error=None,
        )
    
    # Final summary with required logging format
    summary = (
        f"\n✅ [SCAN_SANDBOX] Complete\n"
        f"   scanned_files={len(files_data)}\n"
        f"   db_upserts={scan_id}\n"
        f"   chunks_written={chunks_created}\n"
        f"   embeddings_queued={embedding_queued}\n"
        f"   duration={duration_ms}ms\n"
    )
    yield _sse_token(summary)
    
    yield _sse_done(
        provider="local",
        model="sandbox_scanner",
        total_length=len(files_data),
        meta={
            "scan_id": scan_id,
            "rag_scan_id": rag_scan_id,
            "files": len(files_data),
            "chunks_created": chunks_created,
            "embeddings_queued": embedding_queued,
            "roots": roots_scanned,
            "scope": "sandbox",
            "new_files": stats.get("new_files", 0),
            "updated_files": stats.get("updated_files", 0),
            "unchanged_files": stats.get("unchanged_files", 0),
        },
    )


# =============================================================================
# UPDATE ARCHITECTURE (scope="code") - DB only
# =============================================================================

async def generate_update_architecture_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """
    Scan D:\\Orb + D:\\orb-desktop and save to DB.
    
    This is the code-focused scan for architecture updates.
    Does NOT save to out folder - DB only.
    """
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    
    yield _sse_token("🔍 Scanning code repositories...\n")
    yield _sse_token(f"📡 Controller: {SANDBOX_CONTROLLER_URL}\n")
    yield _sse_token(f"📂 Roots: {', '.join(CODE_SCAN_ROOTS)}\n\n")
    
    # Check if models available
    if not _ARCH_MODELS_AVAILABLE:
        yield _sse_error(
            "Architecture models not available. "
            "Run: alembic upgrade head to create tables."
        )
        yield _sse_done(
            provider="local",
            model="architecture_scanner",
            success=False,
            error="models_not_available",
        )
        return
    
    # Call sandbox_controller
    status, data, error_msg = await loop.run_in_executor(
        None,
        lambda: _call_fs_tree(CODE_SCAN_ROOTS, max_files=100000),
    )
    
    if status != 200 or data is None:
        logger.error(f"[update_arch] Failed: status={status}, error={error_msg}")
        
        if status == 404:
            yield _sse_error(
                f"Sandbox controller /fs/tree not found at {SANDBOX_CONTROLLER_URL}\n"
                f"Please update sandbox_controller to v0.3.0 or later."
            )
        elif status is None:
            yield _sse_error(
                f"Could not connect to sandbox controller at {SANDBOX_CONTROLLER_URL}\n"
                f"Error: {error_msg}\n"
                f"Is the sandbox running?"
            )
        else:
            yield _sse_error(f"Scan failed (status={status}): {error_msg}")
        
        yield _sse_done(
            provider="local",
            model="architecture_scanner",
            success=False,
            error=f"status={status}",
        )
        return
    
    # Extract file data
    files_data = data.get("files", [])
    roots_scanned = data.get("roots_scanned", CODE_SCAN_ROOTS)
    scan_time_ms = data.get("scan_time_ms", 0)
    
    yield _sse_token(f"📊 Found {len(files_data)} files in {scan_time_ms}ms\n")
    
    # Save to DB
    yield _sse_token("💾 Saving to database...\n")
    
    try:
        scan_id = _save_scan_to_db(
            db=db,
            scope="code",
            files_data=files_data,
            roots_scanned=roots_scanned,
            scan_time_ms=scan_time_ms,
        )
        
        if scan_id:
            zone_counts = count_files_by_zone(db, scan_id) if count_files_by_zone else {}
            
            yield _sse_token(f"\n✅ Architecture updated (scan_id={scan_id})\n")
            yield _sse_token(f"📁 Total files: {len(files_data)}\n")
            
            if zone_counts:
                yield _sse_token("📊 By zone:\n")
                for zone, count in sorted(zone_counts.items()):
                    yield _sse_token(f"   • {zone}: {count}\n")
        else:
            yield _sse_token("⚠️ Could not save to DB (models not available)\n")
            
    except Exception as e:
        logger.exception(f"[update_arch] DB save failed: {e}")
        yield _sse_error(f"Failed to save to DB: {e}")
        yield _sse_done(
            provider="local",
            model="architecture_scanner",
            success=False,
            error=str(e),
        )
        return
    
    # Record in memory service
    try:
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id,
                role="assistant",
                content=f"[architecture] Updated: {len(files_data)} files (scan_id={scan_id})",
                provider="local",
                model="architecture_scanner",
            ),
        )
    except Exception:
        pass
    
    duration_ms = int(loop.time() * 1000) - started_ms
    
    if trace:
        trace.log_model_call(
            "local_tool", "local", "architecture_scanner", "update_architecture",
            0, 0, duration_ms, success=True, error=None,
        )
    
    yield _sse_done(
        provider="local",
        model="architecture_scanner",
        total_length=len(files_data),
        meta={
            "scan_id": scan_id,
            "files": len(files_data),
            "roots": roots_scanned,
            "scope": "code",
        },
    )


# =============================================================================
# CREATE ARCHITECTURE MAP (from DB) - Opus generates map
# =============================================================================

async def generate_local_architecture_map_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """
    Load architecture data from DB and generate human-readable map with Claude Opus.
    
    This is the lowercase "Create architecture map" command.
    Does NOT scan - just reads from DB and generates map.
    """
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    
    yield _sse_token("📖 Loading architecture from database...\n")
    
    # Check if models available
    if not _ARCH_MODELS_AVAILABLE:
        yield _sse_error(
            "Architecture models not available. "
            "Run: alembic upgrade head to create tables."
        )
        yield _sse_done(
            provider="local",
            model="architecture_mapper",
            success=False,
            error="models_not_available",
        )
        return
    
    # Get latest code scan
    latest_scan = get_latest_scan(db, scope="code")
    
    if not latest_scan:
        yield _sse_token("⚠️ No code scan found in DB. Running scan first...\n\n")
        async for chunk in generate_update_architecture_stream(project_id, message, db, trace):
            yield chunk
        yield _sse_token("\n")
        
        # Try again
        latest_scan = get_latest_scan(db, scope="code")
        if not latest_scan:
            yield _sse_error("Failed to create architecture scan")
            yield _sse_done(
                provider="local",
                model="architecture_mapper",
                success=False,
                error="no_scan_data",
            )
            return
    
    # Load files from scan
    files = db.query(ArchitectureFileIndex).filter(
        ArchitectureFileIndex.scan_id == latest_scan.id
    ).all()
    
    yield _sse_token(f"📊 Found {len(files)} files from scan {latest_scan.id}\n")
    yield _sse_token(f"🕐 Scan timestamp: {latest_scan.finished_at}\n\n")
    
    # Group by zone
    zone_counts = count_files_by_zone(db, latest_scan.id)
    
    yield _sse_token("📊 By zone:\n")
    for zone, count in sorted(zone_counts.items()):
        yield _sse_token(f"   • {zone}: {count}\n")
    yield _sse_token("\n")
    
    # Build prompt for Opus
    yield _sse_token("🤖 Generating architecture map with Claude Opus...\n\n")
    
    # Prepare file list for prompt
    file_list = []
    for f in files:
        file_list.append({
            "path": f.path,
            "name": f.name,
            "ext": f.ext,
            "zone": f.zone,
            "size": f.size_bytes,
        })
    
    # Build structured prompt
    prompt = _build_db_archmap_prompt(file_list, zone_counts)
    
    # Call Claude Opus
    try:
        from app.llm.streaming import stream_llm
        
        messages = [{"role": "user", "content": prompt}]
        map_content = ""
        
        async for event in stream_llm(
            messages=messages,
            system_prompt=ARCHMAP_SYSTEM_PROMPT,
            provider=ARCHMAP_PROVIDER,
            model=ARCHMAP_MODEL,
        ):
            event_type = event.get("type")
            if event_type == "token":
                text = event.get("text", "")
                map_content += text
                yield _sse_token(text)
            elif event_type == "error":
                yield _sse_error(event.get("message", "Unknown error"))
                yield _sse_done(
                    provider=ARCHMAP_PROVIDER,
                    model=ARCHMAP_MODEL,
                    success=False,
                    error=event.get("message"),
                )
                return
            elif event_type == "done":
                break
        
        yield _sse_token("\n")
        
        # Save map to disk (same location as full map)
        if map_content:
            try:
                output_dir = Path(FULL_ARCHMAP_OUTPUT_DIR).resolve()
                output_dir.mkdir(parents=True, exist_ok=True)
                map_path = output_dir / FULL_ARCHMAP_OUTPUT_FILE
                with open(map_path, "w", encoding="utf-8") as f:
                    f.write(map_content)
                yield _sse_token(f"\n💾 Saved: {map_path}\n")
            except Exception as save_err:
                logger.warning(f"[archmap] Failed to save map to disk: {save_err}")
                yield _sse_token(f"\n⚠️ Could not save to disk: {save_err}\n")
        
    except Exception as e:
        logger.exception(f"[archmap] Opus call failed: {e}")
        yield _sse_error(f"Failed to generate map: {e}")
        yield _sse_done(
            provider=ARCHMAP_PROVIDER,
            model=ARCHMAP_MODEL,
            success=False,
            error=str(e),
        )
        return
    
    duration_ms = int(loop.time() * 1000) - started_ms
    
    if trace:
        trace.log_model_call(
            "local_tool", ARCHMAP_PROVIDER, ARCHMAP_MODEL, "create_architecture_map",
            len(prompt), len(map_content), duration_ms, success=True, error=None,
        )
    
    yield _sse_done(
        provider=ARCHMAP_PROVIDER,
        model=ARCHMAP_MODEL,
        total_length=len(map_content),
        meta={
            "scan_id": latest_scan.id,
            "files": len(files),
            "zones": zone_counts,
        },
    )


# =============================================================================
# CREATE ARCHITECTURE MAP - FULL (ALL CAPS) - Scan + out folder + map
# =============================================================================

async def generate_full_architecture_map_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """
    Full architecture map: Scan D:\\Orb + D:\\orb-desktop, fetch contents, save to out folder.
    
    This is the ALL CAPS "CREATE ARCHITECTURE MAP" command.
    
    Outputs to .architecture/:
    - INDEX.json: File tree metadata
    - CODEBASE.md: All source code with line numbers (for Claude context)
    - ARCHITECTURE_MAP.md: Generated overview (optional, via Opus)
    
    Also saves to DB with full file contents for future RAG.
    """
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    
    # v4.2: Debug - verify output path is correct
    logger.info(f"[full_archmap] Output directory: {FULL_ARCHMAP_OUTPUT_DIR!r}")
    print(f"[full_archmap] Output directory: {FULL_ARCHMAP_OUTPUT_DIR!r}")
    
    # Resolve to absolute path to ensure correctness
    output_dir = Path(FULL_ARCHMAP_OUTPUT_DIR).resolve()
    logger.info(f"[full_archmap] Resolved output path: {output_dir}")
    
    yield _sse_token("🔍 FULL ARCHITECTURE SCAN: Capturing codebase...\n")
    yield _sse_token(f"📡 Controller: {SANDBOX_CONTROLLER_URL}\n")
    yield _sse_token(f"📂 Roots: {', '.join(CODE_SCAN_ROOTS)}\n")
    yield _sse_token(f"📤 Output: {output_dir}\n\n")
    
    # ===========================================================================
    # Phase 1: Scan file tree
    # ===========================================================================
    yield _sse_token("📊 Phase 1: Scanning file tree...\n")
    
    status, data, error_msg = await loop.run_in_executor(
        None,
        lambda: _call_fs_tree(CODE_SCAN_ROOTS, max_files=100000),
    )
    
    if status != 200 or data is None:
        logger.error(f"[full_archmap] Scan failed: status={status}, error={error_msg}")
        yield _sse_error(f"Scan failed: {error_msg}")
        yield _sse_done(
            provider="local",
            model="architecture_scanner",
            success=False,
            error="scan_failed",
        )
        return
    
    files_data = data.get("files", [])
    scan_time_ms = data.get("scan_time_ms", 0)
    
    yield _sse_token(f"   Found {len(files_data)} files in {scan_time_ms}ms\n\n")
    
    # ===========================================================================
    # Phase 2: Fetch file contents
    # ===========================================================================
    yield _sse_token("📖 Phase 2: Reading file contents...\n")
    
    # Filter to files that should have content captured
    # Based on extension and size
    content_extensions = {
        ".py", ".pyw", ".pyi",
        ".js", ".mjs", ".cjs", ".jsx",
        ".ts", ".tsx", ".mts", ".cts",
        ".json", ".jsonc",
        ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
        ".html", ".htm", ".css", ".scss", ".sass", ".less",
        ".sql", ".sh", ".bash", ".zsh", ".ps1", ".psm1", ".bat", ".cmd",
        ".md", ".markdown", ".rst", ".txt",
        "",  # Files without extension (like Dockerfile, Makefile)
    }
    
    # Files to NEVER capture (secrets, credentials, keys)
    skip_filenames = {
        ".env", ".env.local", ".env.production", ".env.development",
        ".env.example",  # Often contains real values
        "secrets.json", "credentials.json", "config.secret.json",
        ".npmrc", ".pypirc",  # Can contain auth tokens
        "id_rsa", "id_ed25519", "id_ecdsa",  # SSH keys
        ".pem", ".key", ".crt", ".p12", ".pfx",  # Certificates
    }
    
    # Patterns to skip
    skip_patterns = {"secret", "credential", "password", "token", "apikey", "api_key"}
    
    paths_to_read = []
    for f in files_data:
        ext = (f.get("ext") or "").lower()
        size = f.get("size_bytes") or 0
        name = (f.get("name") or "").lower()
        
        # Skip large files
        if size > MAX_CONTENT_FILE_SIZE:
            continue
        
        # Skip sensitive files
        if name in skip_filenames:
            continue
        
        # Skip files with sensitive patterns in name
        if any(p in name for p in skip_patterns):
            continue
        
        # Include by extension
        if ext in content_extensions:
            paths_to_read.append(f.get("path"))
            continue
        
        # Include special files without matching extension (but not .env)
        if name in (".gitignore", ".gitattributes", "dockerfile", "makefile"):
            paths_to_read.append(f.get("path"))
    
    yield _sse_token(f"   Reading {len(paths_to_read)} source files...\n")
    
    # Fetch contents in batches to avoid timeout
    contents_data: List[Dict[str, Any]] = []
    batch_size = 100
    
    for i in range(0, len(paths_to_read), batch_size):
        batch_paths = paths_to_read[i:i + batch_size]
        
        status, resp, error_msg = await loop.run_in_executor(
            None,
            lambda bp=batch_paths: _call_fs_contents(bp, include_line_numbers=True),
        )
        
        if status == 200 and resp:
            batch_files = resp.get("files", [])
            contents_data.extend(batch_files)
            
            batch_lines = sum(f.get("line_count", 0) for f in batch_files if not f.get("error"))
            yield _sse_token(f"   Batch {i // batch_size + 1}: {len(batch_files)} files, {batch_lines:,} lines\n")
        else:
            yield _sse_token(f"   Batch {i // batch_size + 1}: Failed - {error_msg}\n")
    
    # Stats
    total_lines = sum(f.get("line_count", 0) for f in contents_data if not f.get("error"))
    total_bytes = sum(f.get("size_bytes", 0) for f in contents_data if not f.get("error"))
    files_with_content = sum(1 for f in contents_data if f.get("content") and not f.get("error"))
    
    yield _sse_token(f"\n   Total: {files_with_content} files, {total_lines:,} lines, {total_bytes / 1024 / 1024:.2f}MB\n\n")
    
    # ===========================================================================
    # Phase 3: Save to output folder
    # ===========================================================================
    yield _sse_token("💾 Phase 3: Saving to output folder...\n")
    
    try:
        # Use resolved output_dir for all file operations
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[full_archmap] Created/verified output directory: {output_dir}")
        
        # Save INDEX.json (legacy - always overwritten)
        index_path = output_dir / "INDEX.json"
        index_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "roots": CODE_SCAN_ROOTS,
            "scan_time_ms": scan_time_ms,
            "total_files": len(files_data),
            "files_with_content": files_with_content,
            "total_lines": total_lines,
            "total_bytes": total_bytes,
            "files": files_data,
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, indent=2)
        yield _sse_token(f"   Saved: INDEX.json (legacy)\n")
        
        # Generate timestamp for RAG-compatible files
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M")
        
        # Save INDEX_<timestamp>.json (RAG expected format)
        index_rag_path = output_dir / f"INDEX_{timestamp_str}.json"
        index_rag_data = _generate_index_for_rag(files_data, contents_data, CODE_SCAN_ROOTS[0])
        with open(index_rag_path, "w", encoding="utf-8") as f:
            json.dump(index_rag_data, f, indent=2)
        yield _sse_token(f"   Saved: INDEX_{timestamp_str}.json (RAG)\n")
        
        # Save SIGNATURES_<timestamp>.json (RAG required)
        signatures_path = output_dir / f"SIGNATURES_{timestamp_str}.json"
        signatures_data = _generate_signatures_json(contents_data, CODE_SCAN_ROOTS[0])
        with open(signatures_path, "w", encoding="utf-8") as f:
            json.dump(signatures_data, f, indent=2)
        total_sigs = signatures_data.get("total_signatures", 0)
        yield _sse_token(f"   Saved: SIGNATURES_{timestamp_str}.json ({total_sigs} signatures)\n")
        
        # Save CODEBASE.md (full source code)
        codebase_path = output_dir / FULL_CODEBASE_OUTPUT_FILE
        codebase_content = _generate_codebase_md(files_data, contents_data)
        with open(codebase_path, "w", encoding="utf-8") as f:
            f.write(codebase_content)
        codebase_size_mb = len(codebase_content.encode("utf-8")) / 1024 / 1024
        yield _sse_token(f"   Saved: {FULL_CODEBASE_OUTPUT_FILE} ({codebase_size_mb:.2f}MB)\n")
        
    except Exception as e:
        logger.exception(f"[full_archmap] Save failed: {e}")
        yield _sse_error(f"Failed to save: {e}")
        yield _sse_done(
            provider="local",
            model="architecture_scanner",
            success=False,
            error=str(e),
        )
        return
    
    # ===========================================================================
    # Phase 4: Save to database
    # ===========================================================================
    yield _sse_token("\n💾 Phase 4: Saving to database...\n")
    
    scan_id = None
    try:
        scan_id = _save_scan_with_contents_to_db(
            db=db,
            scope="code",
            files_data=files_data,
            contents_data=contents_data,
            roots_scanned=CODE_SCAN_ROOTS,
            scan_time_ms=scan_time_ms,
        )
        if scan_id:
            yield _sse_token(f"   Saved to DB: scan_id={scan_id}\n")
        else:
            yield _sse_token("   DB save skipped (models not available)\n")
    except Exception as e:
        logger.exception(f"[full_archmap] DB save failed: {e}")
        yield _sse_token(f"   DB save failed: {e}\n")
    
    # ===========================================================================
    # Phase 4.5: Queue background embedding job (non-blocking)
    # ===========================================================================
    # Embeddings build in background after command returns.
    # This keeps CREATE ARCHITECTURE MAP fast while enabling semantic search.
    # Wrapped in try/except so embedding failures never crash the main workflow.
    
    yield _sse_token("\n🔗 Phase 4.5: Queueing background embedding job...\n")
    
    embedding_queued = False
    try:
        from app.rag.jobs.embedding_job import queue_embedding_job, EMBEDDING_AUTO_ENABLED
        from app.db import get_db_session
        
        if not EMBEDDING_AUTO_ENABLED:
            yield _sse_token("   ⚠️ Auto-embedding disabled (ORB_EMBEDDING_AUTO=false)\n")
        else:
            # get_db_session() returns a Session directly (not a generator)
            # Pass it as the session factory callable
            # NOTE: Do NOT pass scan_id here!
            # ArchCodeChunk.scan_id references arch_scan_runs.id (RAG pipeline's scan tracking)
            # but _save_scan_with_contents_to_db creates architecture_scan_runs.id
            # These are different tables, so passing scan_id would filter out all chunks.
            # Instead, embed ALL pending ArchCodeChunk rows regardless of origin.
            embedding_queued = queue_embedding_job(
                db_session_factory=get_db_session,
                scan_id=None,  # Embed all pending chunks, not filtered by scan
            )
            
            if embedding_queued:
                yield _sse_token("   ✅ Embedding job queued (background)\n")
                yield _sse_token("   📊 Priority: Tier1 (routers) → Tier2 (pipeline) → Tier3 (services) → ...\n")
                yield _sse_token("   💡 Use `embedding status` to check progress\n")
            else:
                yield _sse_token("   ⚠️ Embedding job not queued (may already be running)\n")
                
    except ImportError as ie:
        logger.warning(f"[full_archmap] Embedding module not available: {ie}")
        yield _sse_token(f"   ⚠️ Embedding module not available: {ie}\n")
    except Exception as emb_err:
        # Never let embedding errors crash the main workflow
        logger.warning(f"[full_archmap] Failed to queue embedding job (non-fatal): {emb_err}")
        yield _sse_token(f"   ⚠️ Embedding queue failed (non-fatal): {emb_err}\n")
    
    # ===========================================================================
    # Phase 5: Generate architecture map with Opus (optional)
    # ===========================================================================
    yield _sse_token("\n🤖 Phase 5: Generating architecture overview with Claude Opus...\n\n")
    
    # Build prompt from file tree (not contents - too large)
    zone_counts: Dict[str, int] = {}
    for f in files_data:
        zone = f.get("zone", "other")
        zone_counts[zone] = zone_counts.get(zone, 0) + 1
    
    file_list = [
        {
            "path": f.get("path", ""),
            "name": f.get("name", ""),
            "ext": f.get("ext", ""),
            "zone": f.get("zone", "other"),
            "size": f.get("size_bytes"),
            "root": f.get("root", ""),
        }
        for f in files_data
    ]
    
    prompt = _build_db_archmap_prompt(file_list, zone_counts)
    map_content = ""
    
    try:
        from app.llm.streaming import stream_llm
        
        messages = [{"role": "user", "content": prompt}]
        
        async for event in stream_llm(
            messages=messages,
            system_prompt=ARCHMAP_SYSTEM_PROMPT,
            provider=ARCHMAP_PROVIDER,
            model=ARCHMAP_MODEL,
        ):
            event_type = event.get("type")
            if event_type == "token":
                text = event.get("text", "")
                map_content += text
                yield _sse_token(text)
            elif event_type == "error":
                yield _sse_token(f"\n⚠️ Opus error: {event.get('message')}\n")
                break
            elif event_type == "done":
                break
        
        # Save architecture map
        if map_content:
            map_path = output_dir / FULL_ARCHMAP_OUTPUT_FILE
            with open(map_path, "w", encoding="utf-8") as f:
                f.write(map_content)
            yield _sse_token(f"\n\n   Saved: {FULL_ARCHMAP_OUTPUT_FILE}\n")
        
    except Exception as e:
        logger.exception(f"[full_archmap] Opus call failed: {e}")
        yield _sse_token(f"\n⚠️ Opus failed: {e}\n")
    
    # ===========================================================================
    # Done
    # ===========================================================================
    
    # Record in memory
    try:
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id,
                role="assistant",
                content=f"[architecture_scan] Full scan: {len(files_data)} files, {files_with_content} with content, {total_lines:,} lines",
                provider="local",
                model="architecture_scanner",
            ),
        )
    except Exception:
        pass
    
    duration_ms = int(loop.time() * 1000) - started_ms
    
    if trace:
        trace.log_model_call(
            "local_tool", "local", "architecture_scanner", "full_architecture_map",
            len(prompt), len(map_content), duration_ms, success=True, error=None,
        )
    
    # Build embedding status for summary
    embedding_status_str = "🔗 Embeddings: queued (background)" if embedding_queued else "🔗 Embeddings: not queued"
    
    summary = (
        f"\n✅ Architecture scan complete.\n"
        f"📂 Output: {output_dir}\n"
        f"📊 Files: {len(files_data)} ({files_with_content} with content)\n"
        f"📝 Lines: {total_lines:,}\n"
        f"📦 Size: {total_bytes / 1024 / 1024:.2f}MB\n"
        f"🗺️ Outputs: INDEX.json, INDEX_{timestamp_str}.json, SIGNATURES_{timestamp_str}.json, {FULL_CODEBASE_OUTPUT_FILE}, {FULL_ARCHMAP_OUTPUT_FILE}\n"
        f"💾 DB scan_id: {scan_id}\n"
        f"{embedding_status_str}\n"
        f"⏱️ Duration: {duration_ms}ms\n"
    )
    yield _sse_token(summary)
    
    yield _sse_done(
        provider="local",
        model="architecture_scanner",
        total_length=len(codebase_content),
        meta={
            "output_dir": str(output_dir),
            "files": len(files_data),
            "files_with_content": files_with_content,
            "total_lines": total_lines,
            "total_bytes": total_bytes,
            "scan_id": scan_id,
            "zones": zone_counts,
            "rag_index": f"INDEX_{timestamp_str}.json",
            "rag_signatures": f"SIGNATURES_{timestamp_str}.json",
            "embedding_queued": embedding_queued,
        },
    )


def _build_db_archmap_prompt(files: List[Dict], zone_counts: Dict[str, int]) -> str:
    """Build architecture map prompt from file data."""
    
    # Group files by root then by relative directory
    by_root: Dict[str, Dict[str, List[Dict]]] = {}
    
    for f in files:
        root = f.get("root", "")
        path = f.get("path", "")
        name = f.get("name", "")
        ext = f.get("ext", "")
        size = f.get("size_bytes") or f.get("size") or 0
        
        if not root or not path:
            continue
        
        # Get relative path from root
        try:
            rel_path = path.replace(root, "").lstrip("\\/")
            rel_dir = os.path.dirname(rel_path).replace("\\", "/")
        except:
            rel_dir = ""
        
        if root not in by_root:
            by_root[root] = {}
        if rel_dir not in by_root[root]:
            by_root[root][rel_dir] = []
        
        by_root[root][rel_dir].append({
            "name": name,
            "ext": ext,
            "size": size,
        })
    
    # Build prompt with full tree
    lines = [
        "# Architecture Map Request",
        "",
        "Analyze this codebase and generate a comprehensive architecture map.",
        "",
        f"## Summary: {len(files)} files across {len(by_root)} root(s)",
        "",
    ]
    
    # Full file tree
    for root, dirs in sorted(by_root.items()):
        root_name = os.path.basename(root) or root
        lines.append(f"## {root_name}/")
        lines.append("```")
        
        # Sort directories for consistent output
        for dir_path in sorted(dirs.keys()):
            files_in_dir = dirs[dir_path]
            
            if dir_path:
                lines.append(f"{dir_path}/")
                prefix = "  "
            else:
                prefix = ""
            
            # Sort files by name
            for f in sorted(files_in_dir, key=lambda x: x["name"]):
                size_kb = f["size"] / 1024 if f["size"] else 0
                if size_kb > 10:
                    lines.append(f"{prefix}{f['name']} ({size_kb:.1f}KB)")
                else:
                    lines.append(f"{prefix}{f['name']}")
        
        lines.append("```")
        lines.append("")
    
    lines.extend([
        "## Instructions",
        "",
        "Create an architecture map that includes:",
        "",
        "### 1. System Overview",
        "- What is this system? (infer from file structure)",
        "- Main technology stack",
        "",
        "### 2. Component Breakdown",
        "For each major directory/module:",
        "- Purpose and responsibility",
        "- Key files and what they do",
        "",
        "### 3. Data Flow",
        "- How do requests flow through the system?",
        "- Key entry points (main.py, App.tsx, etc.)",
        "",
        "### 4. Integration Points",
        "- How do backend and frontend communicate?",
        "- External dependencies or services",
        "",
        "### 5. Observations",
        "- Architectural patterns used",
        "- Potential areas of concern (large files, complex directories)",
        "",
        "Be specific and reference actual file names from the tree above.",
    ])
    
    return "\n".join(lines)


# =============================================================================
# LEGACY: ZOBIE MAP (raw scan to out folder)
# =============================================================================

async def _run_mapper() -> Tuple[str, str, List[str]]:
    """Run zobie_map.py and return (stdout, stderr, output_paths)."""
    cmd = [sys.executable, ZOBIE_MAPPER_SCRIPT, ZOBIE_CONTROLLER_URL, ZOBIE_MAPPER_OUT_DIR] + ZOBIE_MAPPER_ARGS
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=ZOBIE_MAPPER_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        with contextlib.suppress(Exception):
            proc.kill()
        raise RuntimeError(f"Mapper timed out after {ZOBIE_MAPPER_TIMEOUT_SEC}s")

    stdout = (stdout_b or b"").decode("utf-8", errors="replace")
    stderr = (stderr_b or b"").decode("utf-8", errors="replace")

    output_paths: List[str] = []
    for line in stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        if os.path.isabs(s) and os.path.exists(s):
            output_paths.append(s)
            continue
        candidate = os.path.join(ZOBIE_MAPPER_OUT_DIR, s)
        if os.path.exists(candidate):
            output_paths.append(candidate)

    return stdout, stderr, output_paths


async def generate_local_zobie_map_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """
    Legacy zobie map command - runs zobie_map.py directly.
    
    This outputs to the out folder (legacy behavior).
    For new architecture scans, use generate_update_architecture_stream instead.
    """
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)

    yield _sse_token("🔧 Running legacy zobie_map.py...\n")
    yield _sse_token(f"📡 Controller: {ZOBIE_CONTROLLER_URL}\n")
    yield _sse_token(f"📂 Output: {ZOBIE_MAPPER_OUT_DIR}\n\n")

    try:
        stdout, stderr, output_paths = await _run_mapper()
        
        if stderr and "error" in stderr.lower():
            logger.warning(f"Mapper stderr: {stderr[:500]}")
            
    except Exception as e:
        logger.exception(f"Mapper failed: {e}")
        yield _sse_error(f"Zobie map failed: {e}")
        yield _sse_done(provider="local", model="zobie_mapper", success=False, error=str(e))
        return

    yield _sse_token(f"📦 Generated {len(output_paths)} output files:\n")
    for p in output_paths[:10]:
        yield _sse_token(f"   • {os.path.basename(p)}\n")
    if len(output_paths) > 10:
        yield _sse_token(f"   ... and {len(output_paths) - 10} more\n")

    # Record in memory
    try:
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id,
                role="assistant",
                content=f"[zobie_map] Generated {len(output_paths)} files in {ZOBIE_MAPPER_OUT_DIR}",
                provider="local",
                model="zobie_mapper",
            ),
        )
    except Exception:
        pass

    duration_ms = int(loop.time() * 1000) - started_ms
    
    if trace:
        trace.log_model_call(
            "local_tool", "local", "zobie_mapper", "zobie_map",
            0, 0, duration_ms, success=True, error=None,
        )

    summary = f"\n✅ Zobie map complete.\n📂 Output: {ZOBIE_MAPPER_OUT_DIR}\n⏱️ Duration: {duration_ms}ms\n"
    yield _sse_token(summary)
    
    yield _sse_done(
        provider="local",
        model="zobie_mapper",
        total_length=len(summary),
        meta={"outputs": output_paths, "out_dir": ZOBIE_MAPPER_OUT_DIR},
    )


# =============================================================================
# FILESYSTEM QUERY HANDLER (v4.7)
# =============================================================================
# Answers filesystem listing/find queries using scan DB index.
# NEVER runs shell commands. Only reads from architecture_file_index.
# =============================================================================

# Allowed scan roots - reject queries outside these
FILESYSTEM_QUERY_ALLOWED_ROOTS = ["D:\\", r"C:\Users\dizzi"]

# Max entries to return (hard cap)
FILESYSTEM_QUERY_MAX_ENTRIES = 200

# v4.9: Read file content limits
FILESYSTEM_READ_MAX_LINES = 200
FILESYSTEM_READ_MAX_BYTES = 16 * 1024  # 16KB preview limit

# Known folder mappings (for queries like "What's in my Desktop")
_KNOWN_FOLDER_PATHS = {
    "desktop": r"C:\Users\dizzi\Desktop",
    "onedrive": r"C:\Users\dizzi\OneDrive",
    "documents": r"C:\Users\dizzi\Documents",
    "downloads": r"C:\Users\dizzi\Downloads",
    "pictures": r"C:\Users\dizzi\Pictures",
    "videos": r"C:\Users\dizzi\Videos",
    "music": r"C:\Users\dizzi\Music",
    "appdata": r"C:\Users\dizzi\AppData",
}


def _normalize_path(path: str) -> str:
    """
    Normalize a Windows path: strip quotes, trim whitespace, convert / to \\.
    
    v4.9: Added for read file support.
    """
    if not path:
        return path
    # Strip quotes and whitespace
    path = path.strip().strip('"').strip("'").strip()
    # Convert forward slashes to backslashes
    path = path.replace('/', '\\')
    # Remove trailing backslash (unless it's a root like D:\)
    if len(path) > 3 and path.endswith('\\'):
        path = path.rstrip('\\')
    return path


def _parse_filesystem_query(message: str) -> dict:
    """
    Parse a filesystem query to extract:
    - query_type: "list", "find", or "read"
    - find_type: "folder", "file", or "any" (for find queries)
    - target_path: The directory/file path to list/search/read
    - search_term: For find queries, what to search for
    - include_full_paths: Whether to include full paths (default True)
    
    v4.9: Added "read" query_type for reading file contents from DB.
    
    Returns dict with parsed info or None if invalid.
    """
    text = message.strip()
    
    # Strip "After scan sandbox, " prefix if present
    text = re.sub(r'^[Aa]fter\s+scan\s+sandbox,?\s*', '', text).strip()
    text_lower = text.lower()
    
    result = {
        "query_type": None,
        "find_type": "any",  # v4.8: "folder", "file", or "any"
        "target_path": None,
        "search_term": None,
        "include_full_paths": "full path" in text_lower,
    }
    
    # Try to extract Windows path
    path_match = re.search(r'([A-Za-z]:[/\\][^"\',;:?!]*)', text)
    if path_match:
        result["target_path"] = _normalize_path(path_match.group(1))
    else:
        # Try to extract known folder keyword
        for folder, path in _KNOWN_FOLDER_PATHS.items():
            if folder in text_lower:
                result["target_path"] = path
                break
    
    # v4.9: Detect READ queries first (more specific patterns)
    # Patterns: "what's written in X", "read X", "show contents of X", 
    #           "open X", "cat X", "what does X contain", "display X"
    read_patterns = [
        r"what'?s\s+(?:written|inside|in)\s+",  # "what's written in", "whats inside"
        r"read\s+(?:the\s+)?(?:file\s+)?",      # "read", "read file", "read the file"
        r"show\s+(?:the\s+)?contents?\s+of\s+", # "show contents of", "show content of"
        r"(?:display|view|print|output)\s+(?:the\s+)?(?:file\s+)?",  # "display", "view file"
        r"cat\s+",                               # "cat X" (unix style)
        r"what\s+does\s+.+\s+(?:say|contain)",  # "what does X say/contain"
        r"open\s+(?:the\s+)?(?:file\s+)?",      # "open", "open file" (when path looks like file)
    ]
    
    for pattern in read_patterns:
        if re.search(pattern, text_lower):
            # Additional check: target_path should look like a file (has extension or no trailing \)
            if result["target_path"]:
                target = result["target_path"]
                # If path has an extension, it's likely a file read request
                if '.' in os.path.basename(target) or not target.endswith('\\'):
                    result["query_type"] = "read"
                    return result
    
    # Determine query type for list/find
    if re.match(r'^(?:list|show|what|contents)', text_lower):
        result["query_type"] = "list"
    elif re.match(r'^find', text_lower):
        result["query_type"] = "find"
        
        # v4.8: Detect if searching for folder specifically
        # "Find folder named Jobs" / "Find folders named X" / "Find directory called X"
        if re.search(r'find\s+(?:folder|folders|directory|directories)\s+(?:named?|called)', text_lower):
            result["find_type"] = "folder"
        elif re.search(r'find\s+(?:file|files)\s+(?:named?|called|with)', text_lower):
            result["find_type"] = "file"
        
        # Extract search term for find queries
        # "Find folder named Jobs under ..." -> "Jobs"
        # "Find MBS Fitness under OneDrive" -> "MBS Fitness"
        # "Find files with Amber in the name" -> "Amber"
        
        named_match = re.search(r'(?:named?|called)\s+["\']?([\w\s-]+)["\']?(?:\s+(?:under|in|on|inside)|$)', text, re.IGNORECASE)
        if named_match:
            result["search_term"] = named_match.group(1).strip()
        else:
            # "Find <term> under <path>"
            find_match = re.search(r'^find\s+(?:folder|file|directory)?\s*([\w\s-]+?)\s+(?:under|in|on|inside)', text, re.IGNORECASE)
            if find_match:
                term = find_match.group(1).strip()
                # Skip generic words
                if term.lower() not in {"folder", "file", "directory", "all", "everything", "files"}:
                    result["search_term"] = term
            else:
                # "Find files with X in the name"
                with_match = re.search(r'with\s+([\w\s-]+?)\s+(?:in\s+(?:the\s+)?name)', text, re.IGNORECASE)
                if with_match:
                    result["search_term"] = with_match.group(1).strip()
    
    return result


def _is_path_within_allowed_roots(path: str) -> bool:
    """Check if a path is within allowed scan roots."""
    path_lower = path.lower().replace('/', '\\')
    
    # D:\ is always allowed
    if path_lower.startswith('d:\\'):
        return True
    
    # C:\Users\dizzi is allowed
    if path_lower.startswith('c:\\users\\dizzi'):
        return True
    
    return False


async def generate_filesystem_query_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """
    Answer filesystem listing/find queries using the scan DB index.
    
    v4.7: New handler for FILESYSTEM_QUERY intent.
    
    Safety:
    - ONLY reads from architecture_file_index table (from scan sandbox)
    - NEVER runs shell commands or mentions running dir/grep
    - Hard cap of 200 entries
    - Only allows paths under D:\\ or C:\\Users\\dizzi
    
    Output format:
    - Folders first, then files
    - Full paths included
    - +N more summary if truncated
    """
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    
    yield _sse_token("📂 [FILESYSTEM_QUERY] Processing query...\n")
    
    # Check if architecture models available
    if not _ARCH_MODELS_AVAILABLE:
        yield _sse_token("⚠️ Architecture database not available.\n")
        yield _sse_token("Run `scan sandbox` first to index the filesystem.\n")
        yield _sse_done(
            provider="local",
            model="filesystem_query",
            success=False,
            error="models_not_available",
        )
        return
    
    # Parse the query
    parsed = _parse_filesystem_query(message)
    query_type = parsed.get("query_type", "list")
    target_path = parsed.get("target_path")
    search_term = parsed.get("search_term")
    
    yield _sse_token(f"   Query type: {query_type}\n")
    if target_path:
        yield _sse_token(f"   Target path: {target_path}\n")
    if search_term:
        yield _sse_token(f"   Search term: {search_term}\n")
    yield _sse_token("\n")
    
    # Validate path is within allowed roots
    if target_path and not _is_path_within_allowed_roots(target_path):
        yield _sse_token(f"❌ Path `{target_path}` is outside allowed scan roots.\n")
        yield _sse_token(f"Allowed roots: D:\\ and C:\\Users\\dizzi\n")
        yield _sse_done(
            provider="local",
            model="filesystem_query",
            success=False,
            error="path_outside_allowed_roots",
        )
        return
    
    # Get latest scan
    latest_scan = get_latest_scan(db, scope="sandbox") if get_latest_scan else None
    
    if not latest_scan:
        yield _sse_token("⚠️ No scan data found in database.\n")
        yield _sse_token("Run `scan sandbox` first to index the filesystem.\n")
        yield _sse_done(
            provider="local",
            model="filesystem_query",
            success=False,
            error="no_scan_data",
        )
        return
    
    yield _sse_token(f"📊 Using scan data (scan_id={latest_scan.id}, from {latest_scan.finished_at})\n\n")
    
    # =========================================================================
    # v4.9: READ handler - fetch file content from DB
    # =========================================================================
    if query_type == "read":
        if not target_path:
            yield _sse_token("❌ No file path specified for read operation.\n")
            yield _sse_done(
                provider="local",
                model="filesystem_query",
                success=False,
                error="no_path_specified",
            )
            return
        
        # Normalize the path
        target_path = _normalize_path(target_path)
        
        # Check if path looks like a folder (no extension, or ends with \)
        basename = os.path.basename(target_path)
        if not basename or '.' not in basename:
            yield _sse_token(f"📁 `{target_path}` looks like a folder path.\n")
            yield _sse_token("💡 Use `list {path}` to see folder contents instead.\n")
            yield _sse_done(
                provider="local",
                model="filesystem_query",
                success=False,
                error="path_is_folder",
            )
            return
        
        yield _sse_token(f"📖 Reading file: {target_path}\n\n")
        
        # Try exact path match first
        file_entry = None
        try:
            file_entry = db.query(ArchitectureFileIndex).filter(
                ArchitectureFileIndex.scan_id == latest_scan.id,
                ArchitectureFileIndex.path == target_path
            ).first()
            
            # v4.9: Case-insensitive fallback if exact match fails (Windows is case-insensitive)
            if not file_entry:
                # Use func.lower for case-insensitive comparison
                from sqlalchemy import func
                file_entry = db.query(ArchitectureFileIndex).filter(
                    ArchitectureFileIndex.scan_id == latest_scan.id,
                    func.lower(ArchitectureFileIndex.path) == target_path.lower()
                ).first()
                
                if file_entry:
                    yield _sse_token(f"   (matched case-insensitively: {file_entry.path})\n")
        except Exception as e:
            logger.exception(f"[FILESYSTEM_QUERY] Read query failed: {e}")
            yield _sse_error(f"Database query failed: {e}")
            yield _sse_done(
                provider="local",
                model="filesystem_query",
                success=False,
                error=str(e),
            )
            return
        
        if not file_entry:
            yield _sse_token(f"📭 File not found in scan index: `{target_path}`\n\n")
            yield _sse_token("Possible reasons:\n")
            yield _sse_token("  • File path may be incorrect\n")
            yield _sse_token("  • File may have been excluded from scan (binary, archive, etc.)\n")
            yield _sse_token("  • File may have been added after the last scan\n\n")
            yield _sse_token("💡 Run `scan sandbox` to refresh the index.\n")
            yield _sse_done(
                provider="local",
                model="filesystem_query",
                success=False,
                error="file_not_found",
                meta={"scan_id": latest_scan.id, "target_path": target_path},
            )
            return
        
        # File found - check if content is available
        if not file_entry.content:
            yield _sse_token(f"📄 File exists in scan index but contents were not captured.\n\n")
            yield _sse_token(f"**Path:** {file_entry.path}\n")
            if file_entry.size_bytes:
                size_kb = file_entry.size_bytes / 1024
                yield _sse_token(f"**Size:** {size_kb:.1f} KB\n")
            yield _sse_token(f"**Extension:** {file_entry.ext or '(none)'}\n\n")
            yield _sse_token("This happens for files that:\n")
            yield _sse_token("  • Were added after the last content scan\n")
            yield _sse_token("  • Exceeded size limits (>500KB)\n")
            yield _sse_token("  • Had unsupported extensions\n\n")
            yield _sse_token("💡 Run `scan sandbox` to capture contents.\n")
            yield _sse_done(
                provider="local",
                model="filesystem_query",
                success=True,  # Not an error - file exists, just no content
                meta={
                    "scan_id": latest_scan.id,
                    "target_path": target_path,
                    "file_exists": True,
                    "content_available": False,
                },
            )
            return
        
        # Content available - read and display with limits
        content_text = file_entry.content.content_text
        
        # Check if content is readable as text (shouldn't have binary chars)
        try:
            # Quick binary check: look for null bytes or high ratio of non-printable chars
            if '\x00' in content_text:
                yield _sse_token(f"📄 File appears to be binary and cannot be displayed as text.\n")
                yield _sse_token(f"**Path:** {file_entry.path}\n")
                if file_entry.size_bytes:
                    yield _sse_token(f"**Size:** {file_entry.size_bytes / 1024:.1f} KB\n")
                yield _sse_done(
                    provider="local",
                    model="filesystem_query",
                    success=True,
                    meta={"scan_id": latest_scan.id, "binary": True},
                )
                return
        except Exception:
            pass  # If check fails, continue anyway
        
        # Apply preview limits
        lines = content_text.splitlines()
        total_lines = len(lines)
        total_bytes = len(content_text.encode('utf-8', errors='replace'))
        
        truncated = False
        truncation_reason = ""
        
        # Check line limit
        if total_lines > FILESYSTEM_READ_MAX_LINES:
            lines = lines[:FILESYSTEM_READ_MAX_LINES]
            truncated = True
            truncation_reason = f"line limit ({FILESYSTEM_READ_MAX_LINES} lines)"
        
        # Check byte limit on the truncated content
        preview_text = '\n'.join(lines)
        preview_bytes = len(preview_text.encode('utf-8', errors='replace'))
        
        if preview_bytes > FILESYSTEM_READ_MAX_BYTES:
            # Truncate by bytes
            preview_text = preview_text[:FILESYSTEM_READ_MAX_BYTES]
            # Find last newline to avoid cutting mid-line
            last_nl = preview_text.rfind('\n')
            if last_nl > FILESYSTEM_READ_MAX_BYTES // 2:
                preview_text = preview_text[:last_nl]
            truncated = True
            truncation_reason = f"size limit (~{FILESYSTEM_READ_MAX_BYTES // 1024}KB)"
        
        # Output file info
        size_str = ""
        if file_entry.size_bytes:
            if file_entry.size_bytes > 1_000_000:
                size_str = f"{file_entry.size_bytes / 1_000_000:.1f} MB"
            elif file_entry.size_bytes > 1_000:
                size_str = f"{file_entry.size_bytes / 1_000:.1f} KB"
            else:
                size_str = f"{file_entry.size_bytes} bytes"
        
        yield _sse_token(f"📄 **{file_entry.name}**\n")
        yield _sse_token(f"**Path:** {file_entry.path}\n")
        if size_str:
            yield _sse_token(f"**Size:** {size_str}\n")
        yield _sse_token(f"**Lines:** {total_lines}\n")
        if file_entry.language:
            yield _sse_token(f"**Language:** {file_entry.language}\n")
        yield _sse_token("\n")
        
        # Output content
        yield _sse_token("--- Content ---\n")
        yield _sse_token(preview_text)
        if not preview_text.endswith('\n'):
            yield _sse_token("\n")
        yield _sse_token("---------------\n\n")
        
        if truncated:
            yield _sse_token(f"⚠️ Showing first {len(preview_text.splitlines())} of {total_lines} lines (truncated due to {truncation_reason})\n")
        else:
            yield _sse_token(f"✅ Showing full file ({total_lines} lines)\n")
        
        # Record in memory
        try:
            memory_service.create_message(
                db,
                memory_schemas.MessageCreate(
                    project_id=project_id,
                    role="assistant",
                    content=f"[filesystem_query] read: {file_entry.path} ({total_lines} lines)",
                    provider="local",
                    model="filesystem_query",
                ),
            )
        except Exception:
            pass
        
        duration_ms = int(loop.time() * 1000) - started_ms
        
        if trace:
            trace.log_model_call(
                "local_tool", "local", "filesystem_query", "filesystem_query",
                0, 0, duration_ms, success=True, error=None,
            )
        
        yield _sse_done(
            provider="local",
            model="filesystem_query",
            total_length=len(preview_text),
            meta={
                "scan_id": latest_scan.id,
                "query_type": "read",
                "target_path": target_path,
                "file_path": file_entry.path,
                "total_lines": total_lines,
                "total_bytes": total_bytes,
                "truncated": truncated,
            },
        )
        return  # Early exit for read queries
    
    # Build query against architecture_file_index
    try:
        query = db.query(ArchitectureFileIndex).filter(
            ArchitectureFileIndex.scan_id == latest_scan.id
        )
        
        # Filter by target path if specified
        if target_path:
            # Normalize path for LIKE query
            path_prefix = target_path.replace('/', '\\').rstrip('\\') + '\\'
            query = query.filter(
                ArchitectureFileIndex.path.like(f"{path_prefix}%")
            )
        
        # Filter by search term if specified (find queries)
        if search_term and query_type == "find":
            search_pattern = f"%{search_term}%"
            query = query.filter(
                ArchitectureFileIndex.name.ilike(search_pattern)
            )
        
        # Get results with limit + 1 to detect truncation
        max_results = FILESYSTEM_QUERY_MAX_ENTRIES + 1
        results = query.limit(max_results).all()
        
        truncated = len(results) > FILESYSTEM_QUERY_MAX_ENTRIES
        if truncated:
            results = results[:FILESYSTEM_QUERY_MAX_ENTRIES]
        
    except Exception as e:
        logger.exception(f"[FILESYSTEM_QUERY] DB query failed: {e}")
        yield _sse_error(f"Database query failed: {e}")
        yield _sse_done(
            provider="local",
            model="filesystem_query",
            success=False,
            error=str(e),
        )
        return
    
    if not results:
        if target_path:
            yield _sse_token(f"📭 No entries found under `{target_path}`\n")
        elif search_term:
            yield _sse_token(f"📭 No entries found matching '{search_term}'\n")
        else:
            yield _sse_token("📭 No entries found in scan data.\n")
        yield _sse_done(
            provider="local",
            model="filesystem_query",
            total_length=0,
            meta={"results": 0, "query_type": query_type},
        )
        return
    
    # =========================================================================
    # v4.8: Format results using path-based inference
    # DB only stores files, so we infer folders from file paths
    # =========================================================================
    
    find_type = parsed.get("find_type", "any")
    
    # For "find folder" queries: search folder segments in paths
    if query_type == "find" and find_type == "folder" and search_term:
        # Infer folders from paths that contain a segment matching search_term
        inferred_folders: set = set()
        search_lower = search_term.lower()
        
        for entry in results:
            path_normalized = entry.path.replace('/', '\\')
            # Split path into segments and check each
            segments = path_normalized.split('\\')
            for i, seg in enumerate(segments[:-1]):  # Exclude filename
                if seg.lower() == search_lower:
                    # Build full path to this folder
                    folder_path = '\\'.join(segments[:i+1])
                    inferred_folders.add(folder_path)
        
        # Output folder search results
        folders_list = sorted(inferred_folders, key=str.lower)
        total_count = len(folders_list)
        
        yield _sse_token(f"🔍 **Folder search results for '{search_term}'**\n\n")
        
        if folders_list:
            yield _sse_token(f"**Folders ({len(folders_list)}):**\n")
            for folder_path in folders_list:
                yield _sse_token(f"📁 {folder_path}\n")
            yield _sse_token("\n")
        else:
            yield _sse_token(f"📭 No folders named '{search_term}' found.\n")
        
        # Summary and done (skip normal output path)
        yield _sse_token(f"---\n")
        yield _sse_token(f"**Total:** {len(folders_list)} folders\n")
        
        # v4.8: Add note about .zip exclusion if relevant
        if search_lower.endswith('.zip') or 'zip' in search_lower:
            yield _sse_token("\n📝 Note: .zip files are excluded from scan metadata (archives not indexed).\n")
        
        # Record in memory
        try:
            memory_service.create_message(
                db,
                memory_schemas.MessageCreate(
                    project_id=project_id,
                    role="assistant",
                    content=f"[filesystem_query] find folder: {len(folders_list)} results for '{search_term}'",
                    provider="local",
                    model="filesystem_query",
                ),
            )
        except Exception:
            pass
        
        duration_ms = int(loop.time() * 1000) - started_ms
        
        if trace:
            trace.log_model_call(
                "local_tool", "local", "filesystem_query", "filesystem_query",
                0, 0, duration_ms, success=True, error=None,
            )
        
        yield _sse_done(
            provider="local",
            model="filesystem_query",
            total_length=total_count,
            meta={
                "scan_id": latest_scan.id,
                "query_type": query_type,
                "find_type": find_type,
                "target_path": target_path,
                "search_term": search_term,
                "folders": len(folders_list),
                "files": 0,
                "truncated": False,
            },
        )
        return  # Early exit for folder search
    
    # For list queries with target_path: infer immediate children
    # Folders are inferred from first-level subdirectories in file paths
    # Files are only those directly in target_path (not nested)
    inferred_folders: set = set()
    top_level_files: list = []
    
    if target_path and query_type == "list":
        target_prefix = target_path.rstrip('\\') + '\\'
        target_prefix_lower = target_prefix.lower()
        
        for entry in results:
            path_normalized = entry.path.replace('/', '\\')
            path_lower = path_normalized.lower()
            
            # Get relative path from target
            if path_lower.startswith(target_prefix_lower):
                relative = path_normalized[len(target_prefix):]
                
                if '\\' in relative:
                    # Has subdirectory - extract first folder segment
                    first_segment = relative.split('\\')[0]
                    folder_full_path = target_prefix + first_segment
                    inferred_folders.add(folder_full_path)
                else:
                    # Direct child file (no subdirectory)
                    top_level_files.append(entry)
        
        # Convert to sorted lists
        folders_list = sorted(inferred_folders, key=str.lower)
        files_list = sorted(top_level_files, key=lambda x: x.name.lower())
    else:
        # Fallback for find queries or no target_path: use old behavior
        # (search by filename, return files that match)
        folders_list = []
        files_list = sorted(results, key=lambda x: x.name.lower())
    
    total_count = len(folders_list) + len(files_list)
    
    if query_type == "list":
        yield _sse_token(f"📁 **Contents of {target_path or 'indexed filesystem'}**\n\n")
    else:
        yield _sse_token(f"🔍 **Search results for '{search_term}'**\n\n")
    
    if folders_list:
        yield _sse_token(f"**Folders ({len(folders_list)}):**\n")
        for folder_path in folders_list:
            # folder_path is a string (inferred path), not an entry object
            yield _sse_token(f"📁 {folder_path}\n")
        yield _sse_token("\n")
    
    if files_list:
        yield _sse_token(f"**Files ({len(files_list)}):**\n")
        for file in files_list:
            path_display = file.path.replace('/', '\\')
            size_str = ""
            if file.size_bytes:
                if file.size_bytes > 1_000_000:
                    size_str = f" ({file.size_bytes / 1_000_000:.1f} MB)"
                elif file.size_bytes > 1_000:
                    size_str = f" ({file.size_bytes / 1_000:.1f} KB)"
            yield _sse_token(f"📄 {path_display}{size_str}\n")
        yield _sse_token("\n")
    
    # Summary
    yield _sse_token(f"---\n")
    yield _sse_token(f"**Total:** {len(folders_list)} folders, {len(files_list)} files\n")
    
    # v4.8: Add note about .zip exclusion if listing and no zips found
    if query_type == "list" and target_path:
        # Check if user might expect to see .zip files
        target_lower = target_path.lower()
        if 'desktop' in target_lower or 'downloads' in target_lower or 'documents' in target_lower:
            yield _sse_token("\n📝 Note: .zip files are excluded from scan metadata (archives not indexed).\n")
    
    if truncated:
        # Get total count from DB for accurate "more" count
        try:
            total_in_db = query.count() if 'query' in dir() else total_count
            remaining = total_in_db - FILESYSTEM_QUERY_MAX_ENTRIES
            yield _sse_token(f"⚠️ Results truncated. +{remaining} more entries (showing first {FILESYSTEM_QUERY_MAX_ENTRIES})\n")
        except:
            yield _sse_token(f"⚠️ Results truncated to {FILESYSTEM_QUERY_MAX_ENTRIES} entries\n")
    
    # Record in memory
    try:
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id,
                role="assistant",
                content=f"[filesystem_query] {query_type}: {total_count} results for '{target_path or search_term}'",
                provider="local",
                model="filesystem_query",
            ),
        )
    except Exception:
        pass
    
    duration_ms = int(loop.time() * 1000) - started_ms
    
    if trace:
        trace.log_model_call(
            "local_tool", "local", "filesystem_query", "filesystem_query",
            0, 0, duration_ms, success=True, error=None,
        )
    
    yield _sse_done(
        provider="local",
        model="filesystem_query",
        total_length=total_count,
        meta={
            "scan_id": latest_scan.id,
            "query_type": query_type,
            "target_path": target_path,
            "search_term": search_term,
            "folders": len(folders_list),
            "files": len(files_list),
            "truncated": truncated,
        },
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "generate_sandbox_structure_scan_stream",
    "generate_update_architecture_stream",
    "generate_local_architecture_map_stream",
    "generate_full_architecture_map_stream",
    "generate_local_zobie_map_stream",
    "generate_filesystem_query_stream",
]
