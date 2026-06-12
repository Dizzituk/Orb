# Purpose: codebase report utils 3
# Called-by: app.llm.local_tools.zobie.streams._codebase_report_utils_4, app.llm.local_tools.zobie.streams._codebase_report_utils_5, app.llm.local_tools.zobie.streams.codebase_report
# Depends-on: app.llm.local_tools.zobie.streams._codebase_report_utils_2, app.llm.local_tools.zobie.streams.codebase_report
# Last-renovated: 2026-06-11
from __future__ import annotations
import json
import logging
import os
from app.llm.local_tools.zobie.streams._codebase_report_utils_2 import ABSOLUTE_PATH_PATTERNS, FULL_MAX_MATCHES_PER_FILE, _count_lines_fast, _should_exclude_file, _should_exclude_folder
from pathlib import Path
from typing import Any, Dict, List, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
CODEBASE_REPORT_STATE_FILE = Path(r"D:\Orb\data\codebase_report_state.json")
FULL_MAX_BYTES_PER_FILE = 64 * 1024  # 64 KB


TOP_N_LARGEST = 30

def _load_state() -> Dict[str, Dict[str, Any]]:
    """Load previous scan state from file."""
    if not CODEBASE_REPORT_STATE_FILE.exists():
        return {}
    try:
        with open(CODEBASE_REPORT_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[CODEBASE_REPORT] Failed to load state: {e}")
        return {}

def _save_state(state: Dict[str, Dict[str, Any]]) -> None:
    """Save scan state to file."""
    try:
        CODEBASE_REPORT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CODEBASE_REPORT_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logger.error(f"[CODEBASE_REPORT] Failed to save state: {e}")

def _compute_incremental(
    current_files: Dict[str, FileEntry],
    old_state: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], List[str], List[str]]:
    """Compute incremental changes since last report.
    
    Returns: (new_files, changed_files, deleted_files)
    """
    from .codebase_report import FileEntry
    current_keys = set(current_files.keys())
    old_keys = set(old_state.keys())
    
    new_files = sorted(current_keys - old_keys)
    deleted_files = sorted(old_keys - current_keys)
    
    changed_files = []
    for key in current_keys & old_keys:
        current = current_files[key]
        old = old_state[key]
        if current.size_bytes != old.get("size_bytes") or current.mtime != old.get("mtime"):
            changed_files.append(key)
    
    return new_files, sorted(changed_files), deleted_files

def _scan_directory(
    root: Path,
    root_label: str,
    count_lines: bool = False,
) -> Tuple[List[FileEntry], List[str], Dict[str, int]]:
    """
    Scan a directory tree and return file entries.
    
    Args:
        root: Root directory to scan
        root_label: Label for this root ("Orb" or "orb-desktop")
        count_lines: Whether to count lines in text files
    
    Returns:
        (files, suspect_folders, extension_counts)
    """
    from .codebase_report import FileEntry, SUSPECT_FOLDER_KEYWORDS, _is_text_file
    files: List[FileEntry] = []
    suspect_folders: List[str] = []
    extension_counts: Dict[str, int] = {}
    
    if not root.exists():
        return files, suspect_folders, extension_counts
    
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        
        # Filter out excluded directories (modifies dirnames in-place)
        dirnames[:] = [d for d in dirnames if not _should_exclude_folder(d)]
        
        # Check for suspect folders
        for d in dirnames:
            d_lower = d.lower()
            if any(kw in d_lower for kw in SUSPECT_FOLDER_KEYWORDS):
                suspect_folders.append(str(current / d))
        
        for filename in filenames:
            file_path = current / filename
            
            # Skip excluded files
            if _should_exclude_file(file_path):
                continue
            
            try:
                stat = file_path.stat()
            except (OSError, PermissionError):
                continue
            
            # Count extension
            ext = file_path.suffix.lower() or "(no ext)"
            extension_counts[ext] = extension_counts.get(ext, 0) + 1
            
            # Count lines if requested and file is text-like
            line_count = None
            if count_lines and _is_text_file(file_path):
                line_count = _count_lines_fast(file_path)
            
            relative = str(file_path.relative_to(root))
            entry = FileEntry(
                path=file_path,
                root_label=root_label,
                relative_path=relative,
                size_bytes=stat.st_size,
                mtime=stat.st_mtime,
                line_count=line_count,
            )
            files.append(entry)
    
    return files, suspect_folders, extension_counts

def _scan_file_for_absolute_paths(
    path: Path,
    max_bytes: int = FULL_MAX_BYTES_PER_FILE,
    max_matches: int = FULL_MAX_MATCHES_PER_FILE,
) -> List[AbsolutePathFinding]:
    """Scan a single file for absolute path references."""
    from .codebase_report import AbsolutePathFinding, _is_text_file
    findings: List[AbsolutePathFinding] = []
    
    if not _is_text_file(path):
        return findings
    
    try:
        size = path.stat().st_size
        if size > max_bytes:
            return findings  # Skip large files
        
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                if len(findings) >= max_matches:
                    break
                
                for pattern in ABSOLUTE_PATH_PATTERNS:
                    for match in pattern.finditer(line):
                        snippet = line.strip()[:100]
                        finding = AbsolutePathFinding(
                            file_path=str(path),
                            line_number=line_num,
                            snippet=snippet,
                            pattern_type="windows" if ":\\" in match.group() else "unix",
                        )
                        findings.append(finding)
                        if len(findings) >= max_matches:
                            break
    except Exception:
        pass
    
    return findings

def _scan_file_for_blocked_refs(
    path: Path,
    max_bytes: int = FULL_MAX_BYTES_PER_FILE,
) -> List[Tuple[str, int, str]]:
    """Scan a file for blocked folder references."""
    from .codebase_report import BLOCKED_FOLDER_REFS, _is_text_file
    refs: List[Tuple[str, int, str]] = []
    
    if not _is_text_file(path):
        return refs
    
    try:
        size = path.stat().st_size
        if size > max_bytes:
            return refs
        
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line_num, line in enumerate(f, 1):
                line_lower = line.lower()
                for blocked in BLOCKED_FOLDER_REFS:
                    if blocked.lower() in line_lower:
                        refs.append((str(path), line_num, blocked))
    except Exception:
        pass
    
    return refs

def _format_size(size_bytes: int) -> str:
    """Format file size for display."""
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.1f} MB"
    if size_bytes >= 1_000:
        return f"{size_bytes / 1_000:.1f} KB"
    return f"{size_bytes} bytes"
