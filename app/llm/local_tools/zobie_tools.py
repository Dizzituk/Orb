# FILE: app/llm/local_tools/zobie_tools.py
"""Streaming local-tool generators for architecture commands.

Commands:
- SCAN SANDBOX: Scan C:\\Users + D:\\ → save to DB only (scope="sandbox")
- UPDATE ARCHITECTURE: Scan D:\\Orb + D:\\orb-desktop → save to DB only (scope="code")  
- CREATE ARCHITECTURE MAP (lowercase): Load from DB → Claude Opus → map (no scan)
- CREATE ARCHITECTURE MAP (ALL CAPS): Full scan + read contents → CODEBASE.md + DB

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

# Timeouts
FS_TREE_TIMEOUT_SEC = int(os.getenv("ORB_FS_TREE_TIMEOUT_SEC", "120"))


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
# SCAN SANDBOX (scope="sandbox") - DB only
# =============================================================================

async def generate_sandbox_structure_scan_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """
    Scan sandbox environment (C:\\Users + D:\\ areas) and save to DB.
    
    This is a read-only scan that indexes the sandbox filesystem.
    Does NOT save to out folder - DB only.
    """
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    
    yield _sse_token("🔍 Scanning sandbox environment...\n")
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
    
    # Call sandbox_controller
    status, data, error_msg = await loop.run_in_executor(
        None,
        lambda: _call_fs_tree(SANDBOX_SCAN_ROOTS, max_files=200000),
    )
    
    if status != 200 or data is None:
        logger.error(f"[sandbox_scan] Failed: status={status}, error={error_msg}")
        
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
    files_data = data.get("files", [])
    roots_scanned = data.get("roots_scanned", SANDBOX_SCAN_ROOTS)
    scan_time_ms = data.get("scan_time_ms", 0)
    truncated = data.get("truncated", False)
    
    yield _sse_token(f"📊 Found {len(files_data)} files in {scan_time_ms}ms\n")
    
    if truncated:
        yield _sse_token("⚠️ Results truncated (max files limit reached)\n")
    
    # Save to DB
    yield _sse_token("💾 Saving to database...\n")
    
    try:
        scan_id = _save_scan_to_db(
            db=db,
            scope="sandbox",
            files_data=files_data,
            roots_scanned=roots_scanned,
            scan_time_ms=scan_time_ms,
        )
        
        if scan_id:
            # Get zone counts
            zone_counts = count_files_by_zone(db, scan_id) if count_files_by_zone else {}
            
            yield _sse_token(f"\n✅ Sandbox scan saved (scan_id={scan_id})\n")
            yield _sse_token(f"📁 Total files: {len(files_data)}\n")
            
            if zone_counts:
                yield _sse_token("📊 By zone:\n")
                for zone, count in sorted(zone_counts.items()):
                    yield _sse_token(f"   • {zone}: {count}\n")
        else:
            yield _sse_token("⚠️ Could not save to DB (models not available)\n")
            
    except Exception as e:
        logger.exception(f"[sandbox_scan] DB save failed: {e}")
        yield _sse_error(f"Failed to save to DB: {e}")
        yield _sse_done(
            provider="local",
            model="sandbox_scanner",
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
                content=f"[sandbox_scan] Indexed {len(files_data)} files (scan_id={scan_id})",
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
    
    yield _sse_done(
        provider="local",
        model="sandbox_scanner",
        total_length=len(files_data),
        meta={
            "scan_id": scan_id,
            "files": len(files_data),
            "roots": roots_scanned,
            "scope": "sandbox",
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
# EXPORTS
# =============================================================================

__all__ = [
    "generate_sandbox_structure_scan_stream",
    "generate_update_architecture_stream",
    "generate_local_architecture_map_stream",
    "generate_full_architecture_map_stream",
    "generate_local_zobie_map_stream",
]
