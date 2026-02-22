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
from app.llm.local_tools.zobie.streams._codebase_report_utils_2 import ABSOLUTE_PATH_PATTERNS, CODEBASE_REPORT_OUTPUT_DIR, FULL_MAX_MATCHES_PER_FILE, _count_lines_fast, _detect_duplicate_filenames, _detect_floating_files, _should_exclude_file, _should_exclude_folder
from app.llm.local_tools.zobie.streams._codebase_report_utils_3 import TOP_N_LARGEST, _compute_incremental, _format_size, _load_state, _save_state, _scan_directory, _scan_file_for_absolute_paths, _scan_file_for_blocked_refs
from app.llm.local_tools.zobie.streams._codebase_report_utils_4 import CODEBASE_REPORT_STATE_FILE, FILE_LINES_WARN, FULL_MAX_BYTES_PER_FILE, TOP_N_LONGEST, _categorize_by_lines, _categorize_by_size, _is_text_file, generate_codebase_report_stream
from app.llm.local_tools.zobie.streams._codebase_report_utils_5 import AbsolutePathFinding, FILE_LINES_CRITICAL, FILE_LINES_HIGH, FILE_SIZE_CRITICAL_BYTES, FILE_SIZE_HIGH_BYTES, FILE_SIZE_WARN_BYTES, FileEntry, _generate_json_report

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

# State file (inside Orb data folder)

# =============================================================================
# THRESHOLDS (Spec Section 4)
# =============================================================================

# File size thresholds

# File length thresholds

# FULL mode content caps

# Top N for reports

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


# =============================================================================
# STATE MANAGEMENT
# =============================================================================


# =============================================================================
# SCANNING FUNCTIONS
# =============================================================================


# =============================================================================
# FULL MODE: CONTENT SCANNING
# =============================================================================


# =============================================================================
# REPORT GENERATION
# =============================================================================


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


# =============================================================================
# MAIN STREAM GENERATOR
# =============================================================================
