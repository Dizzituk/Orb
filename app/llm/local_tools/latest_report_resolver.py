# FILE: app/llm/local_tools/latest_report_resolver.py
r"""
Latest Architecture + Codebase Report Resolver (Read-Only)

Provides reliable resolution of the latest architecture map and codebase report
by modification time (mtime).

RULES:
- Never hardcode timestamped filenames
- Always resolve "latest" by mtime
- MD-first (ignore JSON/FAST reports unless explicitly requested later)
- Match BOTH timestamped (ARCHITECTURE_MAP_2026-01-19.md) AND non-timestamped (ARCHITECTURE_MAP.md)
- Read-only - no destructive operations

SEARCH LOCATIONS:
- Architecture map: D:\Orb\.architecture (repo internal)
- Codebase report: D:\Orb\.architecture (primary) + D:\Orb.architecture (fallback)

v1.0 (2026-01): Initial implementation
v1.1 (2026-01): Added fallback directory for codebase report resolution
"""
from __future__ import annotations

import os
import glob
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Primary location for architecture reports (inside repo)
ARCHITECTURE_DIR = Path(r"D:\Orb\.architecture")

# Fallback location for codebase reports (outside repo, at drive root)
# Some tools generate reports here instead of inside the repo
CODEBASE_REPORT_FALLBACK_DIR = Path(r"D:\Orb.architecture")

# Patterns for matching files (glob-style)
# Match BOTH timestamped and non-timestamped variants
ARCHITECTURE_MAP_PATTERNS = [
    "ARCHITECTURE_MAP.md",       # Non-timestamped
    "ARCHITECTURE_MAP_*.md",     # Timestamped variants
]

CODEBASE_REPORT_FULL_PATTERNS = [
    "CODEBASE_REPORT_FULL.md",       # Non-timestamped (if it ever exists)
    "CODEBASE_REPORT_FULL_*.md",     # Timestamped variants
]

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ResolvedReport:
    """Result of resolving the latest report file."""
    found: bool
    path: Optional[Path] = None
    filename: Optional[str] = None
    mtime: Optional[datetime] = None
    size_bytes: Optional[int] = None
    error: Optional[str] = None
    searched_dir: Optional[str] = None
    searched_patterns: Optional[List[str]] = None
    # v1.1: Track which directories were searched
    searched_dirs: Optional[List[str]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "found": self.found,
            "path": str(self.path) if self.path else None,
            "filename": self.filename,
            "mtime": self.mtime.isoformat() if self.mtime else None,
            "mtime_human": self.mtime.strftime("%Y-%m-%d %H:%M:%S") if self.mtime else None,
            "size_bytes": self.size_bytes,
            "size_human": _format_size(self.size_bytes) if self.size_bytes else None,
            "error": self.error,
            "searched_dir": self.searched_dir,
            "searched_dirs": self.searched_dirs,
            "searched_patterns": self.searched_patterns,
        }


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


# =============================================================================
# RESOLVER FUNCTIONS
# =============================================================================

def _find_latest_by_patterns(
    directory: Path,
    patterns: List[str],
) -> ResolvedReport:
    """
    Find the latest file matching any of the given patterns by mtime.
    
    Args:
        directory: Directory to search in
        patterns: List of glob patterns to match
        
    Returns:
        ResolvedReport with file info or error
    """
    # Check directory exists
    if not directory.exists():
        return ResolvedReport(
            found=False,
            error=f"Directory does not exist: {directory}",
            searched_dir=str(directory),
            searched_patterns=patterns,
        )
    
    if not directory.is_dir():
        return ResolvedReport(
            found=False,
            error=f"Path is not a directory: {directory}",
            searched_dir=str(directory),
            searched_patterns=patterns,
        )
    
    # Collect all matching files
    matching_files: List[Path] = []
    
    for pattern in patterns:
        try:
            full_pattern = str(directory / pattern)
            matches = glob.glob(full_pattern)
            for match in matches:
                p = Path(match)
                if p.is_file():
                    matching_files.append(p)
        except Exception as e:
            logger.warning(f"[latest_report_resolver] Glob error for pattern '{pattern}': {e}")
    
    # No matches found
    if not matching_files:
        return ResolvedReport(
            found=False,
            error=f"No files found matching patterns",
            searched_dir=str(directory),
            searched_patterns=patterns,
        )
    
    # Sort by mtime descending (newest first)
    try:
        matching_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception as e:
        return ResolvedReport(
            found=False,
            error=f"Error sorting files by mtime: {e}",
            searched_dir=str(directory),
            searched_patterns=patterns,
        )
    
    # Return the newest
    latest = matching_files[0]
    
    try:
        stat = latest.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime)
        size = stat.st_size
    except Exception as e:
        return ResolvedReport(
            found=False,
            error=f"Error reading file stats: {e}",
            searched_dir=str(directory),
            searched_patterns=patterns,
        )
    
    return ResolvedReport(
        found=True,
        path=latest,
        filename=latest.name,
        mtime=mtime,
        size_bytes=size,
        searched_dir=str(directory),
        searched_patterns=patterns,
    )


def _find_latest_across_dirs(
    directories: List[Path],
    patterns: List[str],
) -> ResolvedReport:
    """
    Find the latest file matching patterns across MULTIPLE directories.
    
    Collects candidates from all directories and returns the newest by mtime.
    
    Args:
        directories: List of directories to search
        patterns: List of glob patterns to match
        
    Returns:
        ResolvedReport with file info from whichever dir had the newest match
    """
    all_candidates: List[tuple[Path, datetime, int]] = []  # (path, mtime, size)
    searched_dirs: List[str] = []
    errors: List[str] = []
    
    for directory in directories:
        searched_dirs.append(str(directory))
        
        # Skip if directory doesn't exist
        if not directory.exists():
            logger.debug(f"[latest_report_resolver] Directory does not exist (skipping): {directory}")
            continue
        
        if not directory.is_dir():
            logger.debug(f"[latest_report_resolver] Path is not a directory (skipping): {directory}")
            continue
        
        # Collect matching files from this directory
        for pattern in patterns:
            try:
                full_pattern = str(directory / pattern)
                matches = glob.glob(full_pattern)
                for match in matches:
                    p = Path(match)
                    if p.is_file():
                        try:
                            stat = p.stat()
                            mtime = datetime.fromtimestamp(stat.st_mtime)
                            size = stat.st_size
                            all_candidates.append((p, mtime, size))
                            logger.debug(f"[latest_report_resolver] Found candidate: {p.name} (mtime: {mtime})")
                        except Exception as e:
                            logger.warning(f"[latest_report_resolver] Error reading stats for {p}: {e}")
            except Exception as e:
                logger.warning(f"[latest_report_resolver] Glob error for pattern '{pattern}' in {directory}: {e}")
    
    # No matches found across any directory
    if not all_candidates:
        return ResolvedReport(
            found=False,
            error=f"No files found matching patterns",
            searched_dirs=searched_dirs,
            searched_patterns=patterns,
        )
    
    # Sort by mtime descending (newest first)
    all_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Return the newest
    latest_path, latest_mtime, latest_size = all_candidates[0]
    
    logger.info(
        f"[latest_report_resolver] Selected newest from {len(all_candidates)} candidate(s): "
        f"{latest_path.name} from {latest_path.parent}"
    )
    
    return ResolvedReport(
        found=True,
        path=latest_path,
        filename=latest_path.name,
        mtime=latest_mtime,
        size_bytes=latest_size,
        searched_dir=str(latest_path.parent),  # Which dir it came from
        searched_dirs=searched_dirs,  # All dirs that were searched
        searched_patterns=patterns,
    )


def get_latest_architecture_map() -> ResolvedReport:
    """
    Resolve the latest ARCHITECTURE_MAP*.md file.
    
    Searches D:\\Orb\\.architecture\\ for:
    - ARCHITECTURE_MAP.md (non-timestamped)
    - ARCHITECTURE_MAP_*.md (timestamped variants)
    
    Returns the one with the newest mtime.
    
    Returns:
        ResolvedReport with file info or error
    """
    logger.info("[latest_report_resolver] Resolving latest architecture map...")
    
    result = _find_latest_by_patterns(
        directory=ARCHITECTURE_DIR,
        patterns=ARCHITECTURE_MAP_PATTERNS,
    )
    
    if result.found:
        logger.info(f"[latest_report_resolver] Found: {result.filename} (mtime: {result.mtime})")
    else:
        logger.warning(f"[latest_report_resolver] Not found: {result.error}")
    
    return result


def get_latest_codebase_report_full() -> ResolvedReport:
    """
    Resolve the latest CODEBASE_REPORT_FULL_*.md file.
    
    Searches MULTIPLE directories for codebase reports:
    1. D:\\Orb\\.architecture\\ (primary - inside repo)
    2. D:\\Orb.architecture\\ (fallback - outside repo at drive root)
    
    Patterns matched:
    - CODEBASE_REPORT_FULL.md (non-timestamped, if it ever exists)
    - CODEBASE_REPORT_FULL_*.md (timestamped variants)
    
    Returns the NEWEST file by mtime across all directories.
    
    IMPORTANT: This ONLY returns FULL reports (not FAST).
    MD-first: ignores JSON files.
    
    Returns:
        ResolvedReport with file info or error
    """
    logger.info("[latest_report_resolver] Resolving latest codebase report (FULL)...")
    
    # v1.1: Search both primary and fallback directories
    result = _find_latest_across_dirs(
        directories=[ARCHITECTURE_DIR, CODEBASE_REPORT_FALLBACK_DIR],
        patterns=CODEBASE_REPORT_FULL_PATTERNS,
    )
    
    if result.found:
        logger.info(
            f"[latest_report_resolver] Found: {result.filename} "
            f"(mtime: {result.mtime}, from: {result.searched_dir})"
        )
    else:
        logger.warning(f"[latest_report_resolver] Not found: {result.error}")
    
    return result


def read_report_content(
    resolved: ResolvedReport,
    max_lines: Optional[int] = None,
    max_bytes: int = 500_000,  # 500KB default limit
) -> tuple[str, bool]:
    """
    Read the content of a resolved report file.
    
    Args:
        resolved: ResolvedReport from get_latest_* functions
        max_lines: If set, return only the first N lines (for preview)
        max_bytes: Maximum bytes to read (safety limit)
        
    Returns:
        Tuple of (content, truncated)
        - content: The file content (or error message if not found)
        - truncated: True if content was truncated
    """
    if not resolved.found or not resolved.path:
        return f"Error: {resolved.error or 'File not found'}", False
    
    try:
        # Read with size limit
        with open(resolved.path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_bytes)
        
        truncated = len(content) >= max_bytes
        
        # Apply line limit if requested
        if max_lines is not None:
            lines = content.split("\n")
            if len(lines) > max_lines:
                content = "\n".join(lines[:max_lines])
                truncated = True
        
        return content, truncated
        
    except Exception as e:
        return f"Error reading file: {e}", False


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ResolvedReport",
    "get_latest_architecture_map",
    "get_latest_codebase_report_full",
    "read_report_content",
    "ARCHITECTURE_DIR",
    "CODEBASE_REPORT_FALLBACK_DIR",
]
