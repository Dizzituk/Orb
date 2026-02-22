from __future__ import annotations
from app.llm.local_tools.zobie.streams._codebase_report_utils_3 import TOP_N_LARGEST
from app.llm.local_tools.zobie.streams._codebase_report_utils_4 import FILE_LINES_WARN, TOP_N_LONGEST, _categorize_by_lines, _categorize_by_size
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


FILE_SIZE_WARN_BYTES = 250 * 1024      # 250 KB

FILE_SIZE_HIGH_BYTES = 1 * 1024 * 1024  # 1 MB

FILE_SIZE_CRITICAL_BYTES = 5 * 1024 * 1024  # 5 MB

FILE_LINES_HIGH = 1200

FILE_LINES_CRITICAL = 2500

class FileEntry:
    """Represents a scanned file."""
    __slots__ = ("path", "root_label", "relative_path", "size_bytes", "mtime", "line_count", "extension")
    
    def __init__(
        self,
        path: Path,
        root_label: str,
        relative_path: str,
        size_bytes: int,
        mtime: float,
        line_count: Optional[int] = None,
    ):
        self.path = path
        self.root_label = root_label
        self.relative_path = relative_path
        self.size_bytes = size_bytes
        self.mtime = mtime
        self.line_count = line_count
        self.extension = path.suffix.lower()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "root_label": self.root_label,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "mtime": self.mtime,
            "line_count": self.line_count,
            "extension": self.extension,
        }

class AbsolutePathFinding:
    """Represents an absolute path detected in content."""
    __slots__ = ("file_path", "line_number", "snippet", "pattern_type")
    
    def __init__(self, file_path: str, line_number: int, snippet: str, pattern_type: str):
        self.file_path = file_path
        self.line_number = line_number
        self.snippet = snippet[:100]  # Truncate
        self.pattern_type = pattern_type
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "snippet": self.snippet,
            "pattern_type": self.pattern_type,
        }

def _generate_json_report(
    mode: str,
    timestamp: str,
    duration_ms: int,
    files: List[FileEntry],
    suspect_folders: List[str],
    extension_counts: Dict[str, int],
    floating_orb: List[str],
    floating_desktop: List[str],
    new_files: List[str],
    changed_files: List[str],
    deleted_files: List[str],
    abs_path_findings: Optional[List[AbsolutePathFinding]] = None,
    blocked_refs: Optional[List[Tuple[str, int, str]]] = None,
    duplicate_names: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Generate JSON report content."""
    total_bytes = sum(f.size_bytes for f in files)
    
    # Top N largest/longest
    sorted_by_size = sorted(files, key=lambda f: f.size_bytes, reverse=True)[:TOP_N_LARGEST]
    files_with_lines = [f for f in files if f.line_count is not None]
    sorted_by_lines = sorted(files_with_lines, key=lambda f: f.line_count or 0, reverse=True)[:TOP_N_LONGEST]
    
    report = {
        "metadata": {
            "timestamp": timestamp,
            "mode": mode,
            "roots_scanned": ["D:\\Orb", "D:\\orb-desktop"],
            "total_files": len(files),
            "total_bytes": total_bytes,
            "duration_ms": duration_ms,
        },
        "summary": {
            "size_critical": sum(1 for f in files if _categorize_by_size(f.size_bytes) == "CRITICAL"),
            "size_high": sum(1 for f in files if _categorize_by_size(f.size_bytes) == "HIGH"),
            "size_warn": sum(1 for f in files if _categorize_by_size(f.size_bytes) == "WARN"),
            "lines_critical": sum(1 for f in files_with_lines if _categorize_by_lines(f.line_count) == "CRITICAL"),
            "lines_high": sum(1 for f in files_with_lines if _categorize_by_lines(f.line_count) == "HIGH"),
            "lines_warn": sum(1 for f in files_with_lines if _categorize_by_lines(f.line_count) == "WARN"),
            "suspect_folders_count": len(suspect_folders),
            "floating_files_orb": len(floating_orb),
            "floating_files_desktop": len(floating_desktop),
        },
        "thresholds": {
            "size_warn_bytes": FILE_SIZE_WARN_BYTES,
            "size_high_bytes": FILE_SIZE_HIGH_BYTES,
            "size_critical_bytes": FILE_SIZE_CRITICAL_BYTES,
            "lines_warn": FILE_LINES_WARN,
            "lines_high": FILE_LINES_HIGH,
            "lines_critical": FILE_LINES_CRITICAL,
        },
        "largest_files": [f.to_dict() for f in sorted_by_size],
        "longest_files": [f.to_dict() for f in sorted_by_lines],
        "floating_files": {
            "orb": floating_orb,
            "orb_desktop": floating_desktop,
        },
        "suspect_folders": suspect_folders[:50],
        "extension_counts": extension_counts,
        "incremental": {
            "new_files": new_files[:100],
            "changed_files": changed_files[:100],
            "deleted_files": deleted_files[:100],
            "new_count": len(new_files),
            "changed_count": len(changed_files),
            "deleted_count": len(deleted_files),
        },
    }
    
    if mode == "FULL":
        report["full_analysis"] = {
            "absolute_path_findings": [f.to_dict() for f in (abs_path_findings or [])[:100]],
            "blocked_refs": [
                {"file_path": fp, "line_number": ln, "blocked_folder": b}
                for fp, ln, b in (blocked_refs or [])[:50]
            ],
            "duplicate_filenames": {
                name: paths[:10]
                for name, paths in list((duplicate_names or {}).items())[:30]
            },
        }
    
    return report
