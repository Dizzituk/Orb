# FILE: app/llm/local_tools/zobie/db_ops.py
"""Database operations for zobie tools.

Extracted from zobie_tools.py for modularity.
No logic changes - exact same DB operations.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

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
    ARCH_MODELS_AVAILABLE = True
except ImportError:
    ARCH_MODELS_AVAILABLE = False
    ArchitectureScanRun = None
    ArchitectureFileIndex = None
    ArchitectureFileContent = None
    get_latest_scan = None
    get_file_by_path = None
    count_files_by_zone = None
    detect_language = None
    should_capture_content = None


def save_scan_to_db(
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
    if not ARCH_MODELS_AVAILABLE:
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


def save_scan_incremental_to_db(
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
    if not ARCH_MODELS_AVAILABLE:
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


def save_scan_with_contents_to_db(
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
    if not ARCH_MODELS_AVAILABLE:
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
