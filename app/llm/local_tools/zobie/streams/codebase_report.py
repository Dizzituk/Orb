# FILE: app/llm/local_tools/zobie/streams/codebase_report.py
"""CODEBASE REPORT stream generator.

Manual, read-only hygiene/bloat/drift report for D:\\Orb + D:\\orb-desktop.
Outputs to D:\\Orb.architecture\\ (NOT inside repo).

Commands:
- "Astra, command: codebase report fast" - Quick metadata scan
- "Astra, command: codebase report full" - Deep content scan

v1.0 (2026-01): Initial implementation
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import (
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Any,
)

from sqlalchemy.orm import Session

from app.llm.audit_logger import RoutingTrace

from ..sse import sse_token, sse_error, sse_done

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Scan roots (code repos only)
CODEBASE_REPORT_ROOTS: List[Path] = [
    Path(r"D:\Orb"),
    Path(r"D:\orb-desktop"),
]

# Output directory (OUTSIDE repos)
CODEBASE_REPORT_OUTPUT_DIR = Path(r"D:\Orb.architecture")

# State file (inside Orb data folder)
CODEBASE_REPORT_STATE_FILE = Path(r"D:\Orb\data\codebase_report_state.json")

# =============================================================================
# THRESHOLDS (Spec Section 4)
# =============================================================================

# File size thresholds
FILE_SIZE_WARN_BYTES = 250 * 1024      # 250 KB
FILE_SIZE_HIGH_BYTES = 1 * 1024 * 1024  # 1 MB
FILE_SIZE_CRITICAL_BYTES = 5 * 1024 * 1024  # 5 MB

# File length thresholds
FILE_LINES_WARN = 600
FILE_LINES_HIGH = 1200
FILE_LINES_CRITICAL = 2500

# FULL mode content caps
FULL_MAX_BYTES_PER_FILE = 64 * 1024  # 64 KB
FULL_MAX_MATCHES_PER_FILE = 50

# Top N for reports
TOP_N_LARGEST = 30
TOP_N_LONGEST = 30

# =============================================================================
# EXCLUSIONS (Spec Section 7)
# =============================================================================

EXCLUDE_FOLDER_NAMES: Set[str] = {
    ".git",
    "node_modules",
    "dist",
    "build",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".next",
    ".vite",
    ".idea",
    ".vscode",
    "orb-electron-data",
    "Code Cache",
    "GPUCache",
    "Cache",
    "CachedData",
    "CachedExtensions",
    "$RECYCLE.BIN",
    "System Volume Information",
}

EXCLUDE_FILE_EXTENSIONS: Set[str] = {
    ".db", ".sqlite", ".sqlite3",
    ".bin",
    ".zip", ".7z", ".rar", ".tar", ".gz", ".bz2", ".xz",
    ".exe", ".dll", ".msi",
    ".pdb", ".obj", ".o", ".a", ".so", ".dylib",
    ".pyc", ".pyo",
    ".mp4", ".mkv", ".avi", ".mov",
    ".mp3", ".wav", ".flac",
}

# Folders that suggest clutter/bloat
SUSPECT_FOLDER_KEYWORDS: Set[str] = {
    "tmp", "temp", "dump", "backup", "old", "archive",
    "copy", "scratch", "test_output", "export", "logs",
}

# Text-like extensions for line counting / content scanning
TEXT_EXTENSIONS: Set[str] = {
    ".py", ".pyw", ".pyi",
    ".js", ".mjs", ".cjs", ".jsx",
    ".ts", ".tsx", ".mts", ".cts",
    ".json", ".jsonc",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".sql", ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
    ".md", ".markdown", ".rst", ".txt",
    ".xml", ".xsl", ".xslt",
    ".c", ".cpp", ".h", ".hpp", ".cc",
    ".java", ".kt", ".scala",
    ".go", ".rs", ".rb",
    ".env", ".env.example",
    "",  # Files without extension (Dockerfile, Makefile)
}

# Expected root-level items (to detect floating files)
EXPECTED_ROOT_ITEMS_ORB: Set[str] = {
    ".architecture", ".env", ".gitattributes", ".gitignore",
    "app", "config", "data", "docs", "jobs", "main.py",
    "README.md", "requirements.txt", "scripts", "static",
    "tests", "tools", "_backup_before_audit", "_patches",
    "_stage1_testdata", "_stage1_testjobs", "_stage2_testdata",
    "_stage2_testjobs", "openapi.json", "orb.db",
}

EXPECTED_ROOT_ITEMS_DESKTOP: Set[str] = {
    ".git", ".gitignore", "electron.vite.config.mjs", "electron.vite.config.ts",
    "node_modules", "out", "package.json", "package-lock.json",
    "postcss.config.js", "README.md", "resources", "src",
    "tailwind.config.js", "tsconfig.json", "tsconfig.node.json",
    "tsconfig.web.json", ".prettierrc.yaml", "eslint.config.mjs",
    "components.json", ".npmrc", ".env",
}

# Absolute path patterns to detect in FULL mode
ABSOLUTE_PATH_PATTERNS = [
    re.compile(r'[A-Za-z]:\\(?:[^\s"\'<>|*?\n]+)'),  # Windows: C:\path\to\file
    re.compile(r'\\\\[A-Za-z0-9._-]+\\[^\s"\'<>|*?\n]+'),  # UNC: \\server\share
    re.compile(r'/(?:mnt|home)/[^\s"\'<>|*?\n]+'),  # Unix: /mnt/ or /home/
]

# Policy violation patterns (blocked folder references)
BLOCKED_FOLDER_REFS: Set[str] = {
    "AppData",
    "Microsoft\\Protect",
    "Microsoft\\Credentials",
    "Windows\\System32",
    "Program Files",
    "Program Files (x86)",
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

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


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

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


# =============================================================================
# SCANNING FUNCTIONS
# =============================================================================

def _should_exclude_folder(folder_name: str) -> bool:
    """Check if folder should be excluded."""
    return folder_name.lower() in {n.lower() for n in EXCLUDE_FOLDER_NAMES}


def _should_exclude_file(path: Path) -> bool:
    """Check if file should be excluded."""
    return path.suffix.lower() in EXCLUDE_FILE_EXTENSIONS


def _is_text_file(path: Path) -> bool:
    """Check if file is text-like (for line counting)."""
    ext = path.suffix.lower()
    name = path.name.lower()
    # Handle extensionless files by name
    if ext == "" and name in {"dockerfile", "makefile", "jenkinsfile", "vagrantfile"}:
        return True
    return ext in TEXT_EXTENSIONS


def _count_lines_fast(path: Path, max_bytes: int = 1_000_000) -> Optional[int]:
    """Count lines in a text file (fast, with byte limit)."""
    try:
        size = path.stat().st_size
        if size > max_bytes:
            return None  # Too large for fast counting
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


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


def _detect_floating_files(root: Path, expected: Set[str]) -> List[str]:
    """Detect unexpected files/folders at root level."""
    floating = []
    if not root.exists():
        return floating
    
    expected_lower = {e.lower() for e in expected}
    
    for item in root.iterdir():
        if item.name.lower() not in expected_lower:
            # Exclude common generated files
            if not item.name.endswith((".backup", ".bak", ".log")):
                floating.append(item.name)
    
    return floating


# =============================================================================
# FULL MODE: CONTENT SCANNING
# =============================================================================

def _scan_file_for_absolute_paths(
    path: Path,
    max_bytes: int = FULL_MAX_BYTES_PER_FILE,
    max_matches: int = FULL_MAX_MATCHES_PER_FILE,
) -> List[AbsolutePathFinding]:
    """Scan a single file for absolute path references."""
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


def _detect_duplicate_filenames(files: List[FileEntry], threshold: int = 6) -> Dict[str, List[str]]:
    """Detect filenames that appear too many times."""
    name_to_paths: Dict[str, List[str]] = {}
    
    for f in files:
        name = f.path.name.lower()
        if name not in name_to_paths:
            name_to_paths[name] = []
        name_to_paths[name].append(f.relative_path)
    
    # Filter to those over threshold
    return {name: paths for name, paths in name_to_paths.items() if len(paths) >= threshold}


# =============================================================================
# REPORT GENERATION
# =============================================================================

def _categorize_by_size(size_bytes: int) -> str:
    """Categorize file by size threshold."""
    if size_bytes >= FILE_SIZE_CRITICAL_BYTES:
        return "CRITICAL"
    if size_bytes >= FILE_SIZE_HIGH_BYTES:
        return "HIGH"
    if size_bytes >= FILE_SIZE_WARN_BYTES:
        return "WARN"
    return "OK"


def _categorize_by_lines(line_count: Optional[int]) -> str:
    """Categorize file by line count threshold."""
    if line_count is None:
        return "UNKNOWN"
    if line_count >= FILE_LINES_CRITICAL:
        return "CRITICAL"
    if line_count >= FILE_LINES_HIGH:
        return "HIGH"
    if line_count >= FILE_LINES_WARN:
        return "WARN"
    return "OK"


def _format_size(size_bytes: int) -> str:
    """Format file size for display."""
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.1f} MB"
    if size_bytes >= 1_000:
        return f"{size_bytes / 1_000:.1f} KB"
    return f"{size_bytes} bytes"


def _generate_markdown_report(
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
    # FULL-only fields
    abs_path_findings: Optional[List[AbsolutePathFinding]] = None,
    blocked_refs: Optional[List[Tuple[str, int, str]]] = None,
    duplicate_names: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Generate Markdown report content."""
    lines: List[str] = []
    
    # Header
    lines.append(f"# CODEBASE REPORT ({mode})")
    lines.append("")
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- **Timestamp:** {timestamp}")
    lines.append(f"- **Mode:** {mode}")
    lines.append(f"- **Roots Scanned:** D:\\Orb, D:\\orb-desktop")
    lines.append(f"- **Total Files:** {len(files)}")
    total_bytes = sum(f.size_bytes for f in files)
    lines.append(f"- **Total Size:** {_format_size(total_bytes)}")
    lines.append(f"- **Duration:** {duration_ms} ms")
    lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    size_critical = sum(1 for f in files if _categorize_by_size(f.size_bytes) == "CRITICAL")
    size_high = sum(1 for f in files if _categorize_by_size(f.size_bytes) == "HIGH")
    size_warn = sum(1 for f in files if _categorize_by_size(f.size_bytes) == "WARN")
    lines.append(f"- **Size issues:** {size_critical} CRITICAL, {size_high} HIGH, {size_warn} WARN")
    
    files_with_lines = [f for f in files if f.line_count is not None]
    line_critical = sum(1 for f in files_with_lines if _categorize_by_lines(f.line_count) == "CRITICAL")
    line_high = sum(1 for f in files_with_lines if _categorize_by_lines(f.line_count) == "HIGH")
    line_warn = sum(1 for f in files_with_lines if _categorize_by_lines(f.line_count) == "WARN")
    lines.append(f"- **Line count issues:** {line_critical} CRITICAL, {line_high} HIGH, {line_warn} WARN")
    lines.append(f"- **Suspect folders:** {len(suspect_folders)}")
    lines.append(f"- **Floating files (root):** Orb={len(floating_orb)}, Desktop={len(floating_desktop)}")
    lines.append("")
    
    # Largest Files
    lines.append("## Largest Files (Top 30)")
    lines.append("")
    sorted_by_size = sorted(files, key=lambda f: f.size_bytes, reverse=True)[:TOP_N_LARGEST]
    for f in sorted_by_size:
        cat = _categorize_by_size(f.size_bytes)
        lines.append(f"- [{cat}] `{f.relative_path}` ({f.root_label}) - {_format_size(f.size_bytes)}")
    lines.append("")
    
    # Longest Files
    lines.append("## Longest Files (Top 30)")
    lines.append("")
    files_with_lines_sorted = sorted(
        [f for f in files if f.line_count is not None],
        key=lambda f: f.line_count or 0,
        reverse=True,
    )[:TOP_N_LONGEST]
    for f in files_with_lines_sorted:
        cat = _categorize_by_lines(f.line_count)
        lines.append(f"- [{cat}] `{f.relative_path}` ({f.root_label}) - {f.line_count} lines")
    lines.append("")
    
    # Files Over Limits
    lines.append("## Files Over Limits")
    lines.append("")
    lines.append("### By Size")
    for cat in ["CRITICAL", "HIGH", "WARN"]:
        cat_files = [f for f in files if _categorize_by_size(f.size_bytes) == cat]
        if cat_files:
            lines.append(f"**{cat} (>{_format_size(FILE_SIZE_CRITICAL_BYTES if cat == 'CRITICAL' else FILE_SIZE_HIGH_BYTES if cat == 'HIGH' else FILE_SIZE_WARN_BYTES)}):** {len(cat_files)} files")
    lines.append("")
    lines.append("### By Line Count")
    for cat in ["CRITICAL", "HIGH", "WARN"]:
        cat_files = [f for f in files_with_lines if _categorize_by_lines(f.line_count) == cat]
        if cat_files:
            threshold = FILE_LINES_CRITICAL if cat == "CRITICAL" else FILE_LINES_HIGH if cat == "HIGH" else FILE_LINES_WARN
            lines.append(f"**{cat} (>{threshold} lines):** {len(cat_files)} files")
    lines.append("")
    
    # Floating Files
    lines.append("## Floating Files (Unexpected Root Items)")
    lines.append("")
    if floating_orb:
        lines.append("### D:\\Orb")
        for item in floating_orb[:20]:
            lines.append(f"- `{item}`")
        if len(floating_orb) > 20:
            lines.append(f"- ... and {len(floating_orb) - 20} more")
    else:
        lines.append("### D:\\Orb\n_None detected_")
    lines.append("")
    if floating_desktop:
        lines.append("### D:\\orb-desktop")
        for item in floating_desktop[:20]:
            lines.append(f"- `{item}`")
        if len(floating_desktop) > 20:
            lines.append(f"- ... and {len(floating_desktop) - 20} more")
    else:
        lines.append("### D:\\orb-desktop\n_None detected_")
    lines.append("")
    
    # Suspect Folders
    lines.append("## Suspect Folders")
    lines.append("")
    if suspect_folders:
        for folder in suspect_folders[:30]:
            lines.append(f"- `{folder}`")
        if len(suspect_folders) > 30:
            lines.append(f"- ... and {len(suspect_folders) - 30} more")
    else:
        lines.append("_None detected_")
    lines.append("")
    
    # Unusual File Types
    lines.append("## File Types by Extension")
    lines.append("")
    sorted_exts = sorted(extension_counts.items(), key=lambda x: x[1], reverse=True)
    for ext, count in sorted_exts[:30]:
        lines.append(f"- `{ext}`: {count}")
    if len(sorted_exts) > 30:
        lines.append(f"- ... and {len(sorted_exts) - 30} more extensions")
    lines.append("")
    
    # Incremental Changes
    lines.append("## Incremental Changes (Since Last Report)")
    lines.append("")
    lines.append(f"- **New files:** {len(new_files)}")
    lines.append(f"- **Changed files:** {len(changed_files)}")
    lines.append(f"- **Deleted files:** {len(deleted_files)}")
    lines.append("")
    if new_files:
        lines.append("### New Files (first 20)")
        for path in new_files[:20]:
            lines.append(f"- `{path}`")
        if len(new_files) > 20:
            lines.append(f"- ... and {len(new_files) - 20} more")
        lines.append("")
    if changed_files:
        lines.append("### Changed Files (first 20)")
        for path in changed_files[:20]:
            lines.append(f"- `{path}`")
        if len(changed_files) > 20:
            lines.append(f"- ... and {len(changed_files) - 20} more")
        lines.append("")
    if deleted_files:
        lines.append("### Deleted Files (first 20)")
        for path in deleted_files[:20]:
            lines.append(f"- `{path}`")
        if len(deleted_files) > 20:
            lines.append(f"- ... and {len(deleted_files) - 20} more")
        lines.append("")
    
    # FULL-only sections
    if mode == "FULL":
        lines.append("---")
        lines.append("## FULL MODE: Deep Analysis")
        lines.append("")
        
        # Absolute Path Findings
        lines.append("### Absolute Path Findings")
        lines.append("")
        if abs_path_findings:
            for finding in abs_path_findings[:50]:
                lines.append(f"- `{finding.file_path}` L{finding.line_number}: `{finding.snippet}`")
            if len(abs_path_findings) > 50:
                lines.append(f"- ... and {len(abs_path_findings) - 50} more")
        else:
            lines.append("_None detected_")
        lines.append("")
        
        # Blocked Folder References
        lines.append("### Blocked Folder References")
        lines.append("")
        if blocked_refs:
            for file_path, line_num, blocked in blocked_refs[:30]:
                lines.append(f"- `{file_path}` L{line_num}: references `{blocked}`")
            if len(blocked_refs) > 30:
                lines.append(f"- ... and {len(blocked_refs) - 30} more")
        else:
            lines.append("_None detected_")
        lines.append("")
        
        # Duplicate Filenames
        lines.append("### Duplicate Filenames (6+ occurrences)")
        lines.append("")
        if duplicate_names:
            for name, paths in list(duplicate_names.items())[:20]:
                lines.append(f"- `{name}` ({len(paths)} copies)")
                for p in paths[:5]:
                    lines.append(f"  - `{p}`")
                if len(paths) > 5:
                    lines.append(f"  - ... and {len(paths) - 5} more")
            if len(duplicate_names) > 20:
                lines.append(f"- ... and {len(duplicate_names) - 20} more duplicate names")
        else:
            lines.append("_None detected_")
        lines.append("")
    
    lines.append("---")
    lines.append(f"_Report generated by ASTRA Codebase Report Tool_")
    
    return "\n".join(lines)


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


# =============================================================================
# MAIN STREAM GENERATOR
# =============================================================================

async def generate_codebase_report_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """
    Generate a codebase hygiene/bloat/drift report.
    
    Commands:
    - "codebase report fast" - Quick metadata scan
    - "codebase report full" - Deep content scan
    """
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    
    # Parse mode from message
    msg_lower = message.lower()
    if "full" in msg_lower:
        mode = "FULL"
    else:
        mode = "FAST"
    
    yield sse_token(f"📊 [CODEBASE_REPORT] mode={mode} roots=2\n\n")
    logger.info(f"[CODEBASE_REPORT] mode={mode} roots=2")
    
    # Ensure output directory exists
    try:
        CODEBASE_REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        yield sse_token(f"✓ Output dir: {CODEBASE_REPORT_OUTPUT_DIR}\n")
    except Exception as e:
        yield sse_error(f"Failed to create output dir: {e}")
        yield sse_done(provider="local", model="codebase_report", success=False, error=str(e))
        return
    
    # Load previous state
    old_state = _load_state()
    
    # Scan both roots
    all_files: List[FileEntry] = []
    all_suspect_folders: List[str] = []
    all_extension_counts: Dict[str, int] = {}
    
    for root in CODEBASE_REPORT_ROOTS:
        if not root.exists():
            yield sse_token(f"⚠️ Root not found: {root}\n")
            continue
        
        root_label = root.name  # "Orb" or "orb-desktop"
        yield sse_token(f"📁 Scanning {root}...\n")
        
        # FAST always counts lines for text files; FULL does deeper analysis
        files, suspect_folders, ext_counts = _scan_directory(
            root, root_label, count_lines=True
        )
        
        all_files.extend(files)
        all_suspect_folders.extend(suspect_folders)
        for ext, count in ext_counts.items():
            all_extension_counts[ext] = all_extension_counts.get(ext, 0) + count
        
        yield sse_token(f"   ✓ {len(files)} files found\n")
    
    yield sse_token(f"\n📊 Total: {len(all_files)} files scanned\n\n")
    
    # Detect floating files
    floating_orb = _detect_floating_files(CODEBASE_REPORT_ROOTS[0], EXPECTED_ROOT_ITEMS_ORB)
    floating_desktop = _detect_floating_files(CODEBASE_REPORT_ROOTS[1], EXPECTED_ROOT_ITEMS_DESKTOP) if len(CODEBASE_REPORT_ROOTS) > 1 else []
    
    # Build state dict for incremental comparison
    # Key = root_label + "/" + relative_path
    current_state: Dict[str, FileEntry] = {}
    for f in all_files:
        key = f"{f.root_label}/{f.relative_path}"
        # Exclude the state file itself
        if CODEBASE_REPORT_STATE_FILE.name in f.relative_path:
            continue
        current_state[key] = f
    
    # Compute incremental changes
    new_files, changed_files, deleted_files = _compute_incremental(current_state, old_state)
    yield sse_token(f"📈 Incremental: new={len(new_files)} changed={len(changed_files)} deleted={len(deleted_files)}\n\n")
    logger.info(f"[CODEBASE_REPORT] incremental new={len(new_files)} changed={len(changed_files)} deleted={len(deleted_files)}")
    
    # FULL mode: Deep content scanning
    abs_path_findings: List[AbsolutePathFinding] = []
    blocked_refs: List[Tuple[str, int, str]] = []
    duplicate_names: Dict[str, List[str]] = {}
    
    if mode == "FULL":
        yield sse_token("🔍 FULL mode: Scanning file contents...\n")
        
        # Scan for absolute paths
        for i, f in enumerate(all_files):
            if i % 100 == 0 and i > 0:
                yield sse_token(f"   ... {i}/{len(all_files)} files checked\n")
            
            findings = _scan_file_for_absolute_paths(f.path)
            abs_path_findings.extend(findings)
            
            refs = _scan_file_for_blocked_refs(f.path)
            blocked_refs.extend(refs)
        
        yield sse_token(f"   ✓ Found {len(abs_path_findings)} absolute path references\n")
        yield sse_token(f"   ✓ Found {len(blocked_refs)} blocked folder references\n")
        
        # Detect duplicate filenames
        duplicate_names = _detect_duplicate_filenames(all_files)
        yield sse_token(f"   ✓ Found {len(duplicate_names)} duplicate filename patterns\n\n")
    
    # Generate timestamp for filenames
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H%M")
    timestamp_iso = now.isoformat(timespec="seconds")
    
    duration_ms = int(loop.time() * 1000) - started_ms
    
    # Generate reports
    yield sse_token("📝 Generating reports...\n")
    
    md_content = _generate_markdown_report(
        mode=mode,
        timestamp=timestamp_iso,
        duration_ms=duration_ms,
        files=all_files,
        suspect_folders=all_suspect_folders,
        extension_counts=all_extension_counts,
        floating_orb=floating_orb,
        floating_desktop=floating_desktop,
        new_files=new_files,
        changed_files=changed_files,
        deleted_files=deleted_files,
        abs_path_findings=abs_path_findings if mode == "FULL" else None,
        blocked_refs=blocked_refs if mode == "FULL" else None,
        duplicate_names=duplicate_names if mode == "FULL" else None,
    )
    
    json_content = _generate_json_report(
        mode=mode,
        timestamp=timestamp_iso,
        duration_ms=duration_ms,
        files=all_files,
        suspect_folders=all_suspect_folders,
        extension_counts=all_extension_counts,
        floating_orb=floating_orb,
        floating_desktop=floating_desktop,
        new_files=new_files,
        changed_files=changed_files,
        deleted_files=deleted_files,
        abs_path_findings=abs_path_findings if mode == "FULL" else None,
        blocked_refs=blocked_refs if mode == "FULL" else None,
        duplicate_names=duplicate_names if mode == "FULL" else None,
    )
    
    # Write files
    md_filename = f"CODEBASE_REPORT_{mode}_{timestamp_str}.md"
    json_filename = f"CODEBASE_REPORT_{mode}_{timestamp_str}.json"
    
    md_path = CODEBASE_REPORT_OUTPUT_DIR / md_filename
    json_path = CODEBASE_REPORT_OUTPUT_DIR / json_filename
    
    try:
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        yield sse_token(f"✓ Wrote: {md_path}\n")
        logger.info(f"[CODEBASE_REPORT] wrote report: {md_path}")
    except Exception as e:
        yield sse_error(f"Failed to write MD: {e}")
    
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_content, f, indent=2)
        yield sse_token(f"✓ Wrote: {json_path}\n")
        logger.info(f"[CODEBASE_REPORT] wrote report: {json_path}")
    except Exception as e:
        yield sse_error(f"Failed to write JSON: {e}")
    
    # Save state for next incremental run
    new_state = {
        key: {"size_bytes": entry.size_bytes, "mtime": entry.mtime}
        for key, entry in current_state.items()
    }
    _save_state(new_state)
    yield sse_token("✓ State saved for incremental tracking\n\n")
    
    # Final summary
    total_bytes = sum(f.size_bytes for f in all_files)
    yield sse_token(f"📊 **CODEBASE REPORT COMPLETE**\n")
    yield sse_token(f"   Mode: {mode}\n")
    yield sse_token(f"   Files: {len(all_files)}\n")
    yield sse_token(f"   Total size: {_format_size(total_bytes)}\n")
    yield sse_token(f"   Duration: {duration_ms} ms\n\n")
    yield sse_token(f"📁 Reports written to:\n")
    yield sse_token(f"   • {md_path}\n")
    yield sse_token(f"   • {json_path}\n")
    
    logger.info(f"[CODEBASE_REPORT] files_scanned={len(all_files)} bytes={total_bytes} duration_ms={duration_ms}")
    
    yield sse_done(
        provider="local",
        model="codebase_report",
        total_length=len(all_files),
        success=True,
        meta={
            "mode": mode,
            "files_scanned": len(all_files),
            "total_bytes": total_bytes,
            "duration_ms": duration_ms,
            "md_path": str(md_path),
            "json_path": str(json_path),
        },
    )
