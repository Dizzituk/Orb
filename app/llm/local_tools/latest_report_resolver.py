# FILE: app/llm/local_tools/latest_report_resolver.py
"""
Latest Architecture + Codebase Report Resolver (Read-Only)

Provides reliable resolution of the latest architecture map and codebase report
from D:\\Orb\\.architecture\\ by modification time (mtime).

RULES:
- Never hardcode timestamped filenames
- Always resolve "latest" by mtime
- MD-first (ignore JSON/FAST reports unless explicitly requested later)
- Match BOTH timestamped (ARCHITECTURE_MAP_2026-01-19.md) AND non-timestamped (ARCHITECTURE_MAP.md)
- Read-only - no destructive operations

v1.0 (2026-01): Initial implementation
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

# Canonical location for architecture reports on host
ARCHITECTURE_DIR = Path(r"D:\Orb\.architecture")

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
    
    Searches D:\\Orb\\.architecture\\ for:
    - CODEBASE_REPORT_FULL.md (non-timestamped, if it ever exists)
    - CODEBASE_REPORT_FULL_*.md (timestamped variants)
    
    Returns the one with the newest mtime.
    
    IMPORTANT: This ONLY returns FULL reports (not FAST).
    MD-first: ignores JSON files.
    
    Returns:
        ResolvedReport with file info or error
    """
    logger.info("[latest_report_resolver] Resolving latest codebase report (FULL)...")
    
    result = _find_latest_by_patterns(
        directory=ARCHITECTURE_DIR,
        patterns=CODEBASE_REPORT_FULL_PATTERNS,
    )
    
    if result.found:
        logger.info(f"[latest_report_resolver] Found: {result.filename} (mtime: {result.mtime})")
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
]
