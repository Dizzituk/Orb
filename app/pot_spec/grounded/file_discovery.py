# FILE: app/pot_spec/grounded/file_discovery.py
"""File Discovery System (v1.0)

Discovers files matching search patterns across the codebase using PowerShell
Select-String via the sandbox controller. Used by SpecGate to build file lists
for multi-file operations.

Architecture:
    SpecGate → file_discovery.py → SandboxClient.shell_run() → PowerShell Select-String

v1.0 (2026-01-28): Initial implementation
    - discover_files(): Pattern-based search using Select-String
    - discover_files_by_extension(): Find files by extension
    - FileMatch, DiscoveryResult dataclasses
    - Configurable roots, exclusions, timeout
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_ROOTS: List[str] = [
    r"D:\Orb",
    r"D:\orb-desktop",
]

DEFAULT_EXCLUSIONS: List[str] = [
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    "*.egg-info",
    ".next",
    "coverage",
    ".coverage",
    "htmlcov",
]

DEFAULT_FILE_EXTENSIONS: List[str] = [
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".md",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".html",
    ".css",
    ".sql",
]

DEFAULT_TIMEOUT_SECONDS: int = 120
DEFAULT_MAX_RESULTS: int = 500
DEFAULT_MAX_SAMPLES_PER_FILE: int = 3


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class LineMatch:
    """Single line match within a file."""
    line_number: int
    line_content: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_number": self.line_number,
            "line_content": self.line_content,
        }


@dataclass
class FileMatch:
    """Single file with matches."""
    path: str
    occurrence_count: int
    line_matches: List[LineMatch] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "occurrence_count": self.occurrence_count,
            "line_matches": [m.to_dict() for m in self.line_matches],
        }


@dataclass
class DiscoveryResult:
    """Complete discovery results."""
    success: bool
    search_pattern: str
    total_files: int
    total_occurrences: int
    files: List[FileMatch] = field(default_factory=list)
    truncated: bool = False
    error_message: Optional[str] = None
    duration_ms: int = 0
    roots_searched: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "search_pattern": self.search_pattern,
            "total_files": self.total_files,
            "total_occurrences": self.total_occurrences,
            "files": [f.to_dict() for f in self.files],
            "truncated": self.truncated,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
            "roots_searched": self.roots_searched,
        }
    
    def get_file_preview(self, max_files: int = 10) -> str:
        """Generate human-readable preview for POT Spec review."""
        lines = [
            f"Pattern: {self.search_pattern}",
            f"Found: {self.total_files} files, {self.total_occurrences} occurrences",
            "",
        ]
        
        for i, fm in enumerate(self.files[:max_files]):
            lines.append(f"  {i+1}. {fm.path} ({fm.occurrence_count} matches)")
            for lm in fm.line_matches[:2]:
                preview = lm.line_content[:60] + "..." if len(lm.line_content) > 60 else lm.line_content
                lines.append(f"      L{lm.line_number}: {preview}")
        
        if self.total_files > max_files:
            lines.append(f"  ... and {self.total_files - max_files} more files")
        
        if self.truncated:
            lines.append("")
            lines.append("⚠️ Results truncated (max limit reached)")
        
        return "\n".join(lines)


# =============================================================================
# Discovery Functions
# =============================================================================

def discover_files(
    search_pattern: str,
    sandbox_client: Any,
    file_filter: Optional[str] = None,
    roots: Optional[List[str]] = None,
    exclusions: Optional[List[str]] = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    max_samples_per_file: int = DEFAULT_MAX_SAMPLES_PER_FILE,
    case_sensitive: bool = False,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> DiscoveryResult:
    """
    Discover files containing a search pattern.
    
    Uses PowerShell Select-String via sandbox controller for accurate,
    high-performance searching across the codebase.
    
    Args:
        search_pattern: Pattern to search for (regex supported)
        sandbox_client: SandboxClient instance (from app.overwatcher.sandbox_client)
        file_filter: Optional file extension filter (e.g., "*.py")
        roots: Directories to search (defaults to DEFAULT_ROOTS)
        exclusions: Directories to exclude (defaults to DEFAULT_EXCLUSIONS)
        max_results: Maximum files to return (default 500)
        max_samples_per_file: Max line samples per file (default 3)
        case_sensitive: Case-sensitive search (default False)
        timeout_seconds: PowerShell timeout (default 120)
    
    Returns:
        DiscoveryResult with matching files and metadata
    
    Example:
        from app.overwatcher.sandbox_client import get_sandbox_client
        
        client = get_sandbox_client()
        result = discover_files("TODO", client, file_filter="*.py")
        
        if result.success:
            for fm in result.files:
                print(f"{fm.path}: {fm.occurrence_count} matches")
    """
    roots = roots or DEFAULT_ROOTS
    exclusions = exclusions or DEFAULT_EXCLUSIONS
    
    logger.info(f"[file_discovery] Starting search: pattern={search_pattern!r}, roots={roots}")
    
    # Build PowerShell Select-String command
    ps_command = _build_select_string_command(
        pattern=search_pattern,
        roots=roots,
        exclusions=exclusions,
        file_filter=file_filter,
        case_sensitive=case_sensitive,
    )
    
    logger.debug(f"[file_discovery] PowerShell command: {ps_command[:200]}...")
    
    try:
        # Execute via sandbox
        shell_result = sandbox_client.shell_run(
            command=ps_command,
            cwd_target="REPO",
            timeout_seconds=timeout_seconds,
        )
        
        if not shell_result.ok and shell_result.exit_code != 1:
            # Exit code 1 means "no matches" in Select-String, which is valid
            logger.warning(f"[file_discovery] Shell error: exit={shell_result.exit_code}, stderr={shell_result.stderr[:200]}")
            return DiscoveryResult(
                success=False,
                search_pattern=search_pattern,
                total_files=0,
                total_occurrences=0,
                error_message=f"PowerShell error (exit {shell_result.exit_code}): {shell_result.stderr[:500]}",
                duration_ms=shell_result.duration_ms,
                roots_searched=roots,
            )
        
        # Parse Select-String output
        files, total_occurrences, truncated = _parse_select_string_output(
            stdout=shell_result.stdout,
            max_results=max_results,
            max_samples_per_file=max_samples_per_file,
        )
        
        logger.info(f"[file_discovery] Found {len(files)} files, {total_occurrences} occurrences")
        
        return DiscoveryResult(
            success=True,
            search_pattern=search_pattern,
            total_files=len(files),
            total_occurrences=total_occurrences,
            files=files,
            truncated=truncated,
            duration_ms=shell_result.duration_ms,
            roots_searched=roots,
        )
        
    except Exception as e:
        logger.error(f"[file_discovery] Exception: {e}")
        return DiscoveryResult(
            success=False,
            search_pattern=search_pattern,
            total_files=0,
            total_occurrences=0,
            error_message=str(e),
            roots_searched=roots,
        )


def discover_files_by_extension(
    extension: str,
    sandbox_client: Any,
    roots: Optional[List[str]] = None,
    exclusions: Optional[List[str]] = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> DiscoveryResult:
    """
    Discover all files with a specific extension.
    
    Simpler than discover_files() - just finds files by extension
    without pattern matching within file contents.
    
    Args:
        extension: File extension (e.g., ".py", "py", "*.py")
        sandbox_client: SandboxClient instance
        roots: Directories to search
        exclusions: Directories to exclude
        max_results: Maximum files to return
        timeout_seconds: PowerShell timeout
    
    Returns:
        DiscoveryResult with matching files (no line_matches)
    """
    roots = roots or DEFAULT_ROOTS
    exclusions = exclusions or DEFAULT_EXCLUSIONS
    
    # Normalize extension
    ext = extension.lstrip("*").lstrip(".")
    
    logger.info(f"[file_discovery] Extension search: .{ext}, roots={roots}")
    
    # Build PowerShell Get-ChildItem command
    ps_command = _build_extension_search_command(
        extension=ext,
        roots=roots,
        exclusions=exclusions,
    )
    
    try:
        shell_result = sandbox_client.shell_run(
            command=ps_command,
            cwd_target="REPO",
            timeout_seconds=timeout_seconds,
        )
        
        if not shell_result.ok and shell_result.exit_code != 0:
            return DiscoveryResult(
                success=False,
                search_pattern=f"*.{ext}",
                total_files=0,
                total_occurrences=0,
                error_message=f"PowerShell error: {shell_result.stderr[:500]}",
                duration_ms=shell_result.duration_ms,
                roots_searched=roots,
            )
        
        # Parse file list
        files, truncated = _parse_file_list_output(
            stdout=shell_result.stdout,
            max_results=max_results,
        )
        
        logger.info(f"[file_discovery] Found {len(files)} .{ext} files")
        
        return DiscoveryResult(
            success=True,
            search_pattern=f"*.{ext}",
            total_files=len(files),
            total_occurrences=len(files),  # 1 per file for extension search
            files=files,
            truncated=truncated,
            duration_ms=shell_result.duration_ms,
            roots_searched=roots,
        )
        
    except Exception as e:
        logger.error(f"[file_discovery] Exception: {e}")
        return DiscoveryResult(
            success=False,
            search_pattern=f"*.{ext}",
            total_files=0,
            total_occurrences=0,
            error_message=str(e),
            roots_searched=roots,
        )


# =============================================================================
# PowerShell Command Builders
# =============================================================================

def _build_select_string_command(
    pattern: str,
    roots: List[str],
    exclusions: List[str],
    file_filter: Optional[str],
    case_sensitive: bool,
) -> str:
    """Build PowerShell Select-String command."""
    # Escape pattern for PowerShell
    escaped_pattern = pattern.replace("'", "''")
    
    # Build exclusion regex pattern
    exclusion_parts = []
    for exc in exclusions:
        # Convert glob to regex
        exc_regex = exc.replace(".", r"\.").replace("*", ".*")
        exclusion_parts.append(exc_regex)
    exclusion_regex = "|".join(exclusion_parts) if exclusion_parts else ""
    
    # Build file filter
    include_filter = file_filter or "*.*"
    
    # Case sensitivity flag
    case_flag = "" if case_sensitive else "-CaseSensitive:$false"
    
    # Build the command
    # We use Get-ChildItem to find files, then Select-String to search
    roots_joined = "', '".join(roots)
    
    cmd_parts = [
        f"Get-ChildItem -Path '{roots_joined}' -Recurse -File -Include '{include_filter}' -ErrorAction SilentlyContinue",
    ]
    
    if exclusion_regex:
        cmd_parts.append(f"| Where-Object {{ $_.FullName -notmatch '{exclusion_regex}' }}")
    
    cmd_parts.append(f"| Select-String -Pattern '{escaped_pattern}' {case_flag} -ErrorAction SilentlyContinue")
    cmd_parts.append("| ForEach-Object { \"$($_.Path)|$($_.LineNumber)|$($_.Line)\" }")
    
    return " ".join(cmd_parts)


def _build_extension_search_command(
    extension: str,
    roots: List[str],
    exclusions: List[str],
) -> str:
    """Build PowerShell Get-ChildItem command for extension search."""
    # Build exclusion regex
    exclusion_parts = []
    for exc in exclusions:
        exc_regex = exc.replace(".", r"\.").replace("*", ".*")
        exclusion_parts.append(exc_regex)
    exclusion_regex = "|".join(exclusion_parts) if exclusion_parts else ""
    
    roots_joined = "', '".join(roots)
    
    cmd_parts = [
        f"Get-ChildItem -Path '{roots_joined}' -Recurse -File -Filter '*.{extension}' -ErrorAction SilentlyContinue",
    ]
    
    if exclusion_regex:
        cmd_parts.append(f"| Where-Object {{ $_.FullName -notmatch '{exclusion_regex}' }}")
    
    cmd_parts.append("| ForEach-Object { $_.FullName }")
    
    return " ".join(cmd_parts)


# =============================================================================
# Output Parsers
# =============================================================================

def _parse_select_string_output(
    stdout: str,
    max_results: int,
    max_samples_per_file: int,
) -> Tuple[List[FileMatch], int, bool]:
    """
    Parse Select-String output.
    
    Expected format per line: path|line_number|line_content
    
    Returns: (files, total_occurrences, truncated)
    """
    if not stdout.strip():
        return [], 0, False
    
    # Group by file path
    file_matches: Dict[str, FileMatch] = {}
    total_occurrences = 0
    truncated = False
    
    for line in stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        
        # Parse: path|line_number|line_content
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        
        path = parts[0].strip()
        try:
            line_num = int(parts[1].strip())
        except ValueError:
            continue
        line_content = parts[2] if len(parts) > 2 else ""
        
        total_occurrences += 1
        
        if path not in file_matches:
            if len(file_matches) >= max_results:
                truncated = True
                continue
            file_matches[path] = FileMatch(path=path, occurrence_count=0, line_matches=[])
        
        fm = file_matches[path]
        fm.occurrence_count += 1
        
        if len(fm.line_matches) < max_samples_per_file:
            fm.line_matches.append(LineMatch(
                line_number=line_num,
                line_content=line_content.strip(),
            ))
    
    return list(file_matches.values()), total_occurrences, truncated


def _parse_file_list_output(
    stdout: str,
    max_results: int,
) -> Tuple[List[FileMatch], bool]:
    """
    Parse Get-ChildItem file list output.
    
    Returns: (files, truncated)
    """
    if not stdout.strip():
        return [], False
    
    files: List[FileMatch] = []
    truncated = False
    
    for line in stdout.strip().split("\n"):
        path = line.strip()
        if not path:
            continue
        
        if len(files) >= max_results:
            truncated = True
            break
        
        files.append(FileMatch(
            path=path,
            occurrence_count=1,
            line_matches=[],
        ))
    
    return files, truncated


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Models
    "LineMatch",
    "FileMatch",
    "DiscoveryResult",
    # Functions
    "discover_files",
    "discover_files_by_extension",
    # Config
    "DEFAULT_ROOTS",
    "DEFAULT_EXCLUSIONS",
    "DEFAULT_FILE_EXTENSIONS",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_RESULTS",
]
