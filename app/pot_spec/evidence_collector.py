# FILE: app/pot_spec/evidence_collector.py
"""
Evidence Collector for SpecGate Contract v1.

Provides read-only evidence gathering from:
1. Latest ARCHITECTURE_MAP*.md (by mtime)
2. Latest CODEBASE_REPORT_FULL_*.md (by mtime)
3. Direct file reads (read/head/lines)
4. Repo search (find command)
5. arch_query (FALLBACK ONLY)

RULES:
- Runtime is STRICTLY READ-ONLY
- No filesystem writes
- No DB writes
- arch_query is fallback after primary sources fail
- Evidence Used must include filenames + mtimes + line ranges

v1.0 (2026-01): Initial implementation for SpecGate Contract v1
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# IMPORTS - Evidence Sources
# =============================================================================

# Primary: Latest report resolver
try:
    from app.llm.local_tools.latest_report_resolver import (
        get_latest_architecture_map,
        get_latest_codebase_report_full,
        read_report_content,
        ResolvedReport,
    )
    _REPORT_RESOLVER_AVAILABLE = True
except ImportError as e:
    logger.warning("[evidence_collector] Report resolver not available: %s", e)
    _REPORT_RESOLVER_AVAILABLE = False
    get_latest_architecture_map = None
    get_latest_codebase_report_full = None
    read_report_content = None
    ResolvedReport = None

# Secondary: File reading via zobie
try:
    from app.llm.local_tools.zobie.fs_live_ops import live_read_file
    _LIVE_READ_AVAILABLE = True
except ImportError as e:
    logger.warning("[evidence_collector] live_read_file not available: %s", e)
    _LIVE_READ_AVAILABLE = False
    live_read_file = None

# Fallback: arch_query (ONLY as last resort)
try:
    from app.llm.local_tools.arch_query import (
        search_symbols,
        get_file_signatures,
        is_service_available as arch_query_available,
    )
    _ARCH_QUERY_AVAILABLE = True
except ImportError as e:
    logger.warning("[evidence_collector] arch_query not available: %s", e)
    _ARCH_QUERY_AVAILABLE = False
    search_symbols = None
    get_file_signatures = None
    arch_query_available = None


# =============================================================================
# CONSTANTS
# =============================================================================

# Hard error for write attempts
WRITE_REFUSED_ERROR = "SpecGate runtime is read-only. Write actions are not permitted."

# Evidence priority order
EVIDENCE_PRIORITY = [
    "architecture_map",      # Highest priority
    "codebase_report",
    "file_read",
    "repo_search",
    "arch_query_fallback",  # Lowest priority - fallback only
]

# Default limits
DEFAULT_MAX_LINES = 500
DEFAULT_MAX_BYTES = 100_000


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EvidenceSource:
    """A single piece of evidence used in grounding."""
    source_type: str  # architecture_map, codebase_report, file_read, repo_search, arch_query_fallback
    filename: Optional[str] = None
    path: Optional[str] = None  # Repo-relative path
    mtime: Optional[datetime] = None
    mtime_human: Optional[str] = None
    line_range: Optional[Tuple[int, int]] = None  # (start, end) if applicable
    content_preview: Optional[str] = None  # First ~200 chars for verification
    query: Optional[str] = None  # For search operations
    found: bool = True
    error: Optional[str] = None
    is_fallback: bool = False  # True if arch_query was used

    def to_evidence_line(self) -> str:
        """Format as evidence line for POT spec."""
        parts = []
        
        if self.source_type == "architecture_map":
            parts.append(f"Architecture map: `{self.filename}`")
        elif self.source_type == "codebase_report":
            parts.append(f"Codebase report: `{self.filename}`")
        elif self.source_type == "file_read":
            parts.append(f"File read: `{self.path}`")
        elif self.source_type == "repo_search":
            parts.append(f"Repo search: query=\"{self.query}\"")
        elif self.source_type == "arch_query_fallback":
            parts.append(f"arch_query (fallback): query=\"{self.query}\"")
        
        if self.mtime_human:
            parts.append(f"(mtime: {self.mtime_human})")
        
        if self.line_range:
            parts.append(f"lines {self.line_range[0]}-{self.line_range[1]}")
        
        if self.is_fallback:
            parts.append("[FALLBACK]")
        
        if self.error:
            parts.append(f"[ERROR: {self.error}]")
        
        return " ".join(parts)


@dataclass
class EvidenceBundle:
    """Collection of evidence for a SpecGate run."""
    sources: List[EvidenceSource] = field(default_factory=list)
    arch_map_content: Optional[str] = None
    codebase_report_content: Optional[str] = None
    file_reads: Dict[str, str] = field(default_factory=dict)  # path -> content
    search_results: Dict[str, List[Dict]] = field(default_factory=dict)  # query -> results
    arch_query_used: bool = False
    loaded_at: datetime = field(default_factory=datetime.now)
    errors: List[str] = field(default_factory=list)

    def add_source(self, source: EvidenceSource) -> None:
        """Add an evidence source."""
        self.sources.append(source)
        if source.error:
            self.errors.append(f"{source.source_type}: {source.error}")

    def to_evidence_used_markdown(self) -> str:
        """Generate Evidence Used section for POT spec."""
        lines = ["## Evidence Used", ""]
        
        for source in self.sources:
            if source.found or source.error:
                lines.append(f"- {source.to_evidence_line()}")
        
        if not self.sources:
            lines.append("- No evidence sources consulted")
        
        return "\n".join(lines)


# =============================================================================
# WRITE PROTECTION
# =============================================================================

def refuse_write_operation(operation: str) -> str:
    """
    Called when any write operation is attempted.
    
    Args:
        operation: Description of the attempted operation
        
    Returns:
        Always returns the hard error message
        
    Raises:
        RuntimeError: Always raises to prevent any write
    """
    logger.error("[evidence_collector] WRITE REFUSED: %s", operation)
    raise RuntimeError(WRITE_REFUSED_ERROR)


# =============================================================================
# EVIDENCE LOADING - PRIMARY SOURCES
# =============================================================================

def load_architecture_map(
    max_lines: int = DEFAULT_MAX_LINES,
) -> Tuple[Optional[str], EvidenceSource]:
    """
    Load the latest architecture map (by mtime).
    
    Returns:
        (content, evidence_source)
    """
    if not _REPORT_RESOLVER_AVAILABLE or not get_latest_architecture_map:
        return None, EvidenceSource(
            source_type="architecture_map",
            found=False,
            error="Report resolver not available",
        )
    
    try:
        resolved = get_latest_architecture_map()
        
        if not resolved.found:
            return None, EvidenceSource(
                source_type="architecture_map",
                found=False,
                error=resolved.error or "Not found",
            )
        
        content, truncated = read_report_content(resolved, max_lines=max_lines)
        
        # Convert to repo-relative path
        repo_path = _to_repo_relative(str(resolved.path)) if resolved.path else None
        
        source = EvidenceSource(
            source_type="architecture_map",
            filename=resolved.filename,
            path=repo_path,
            mtime=resolved.mtime,
            mtime_human=resolved.mtime.strftime("%Y-%m-%d %H:%M:%S") if resolved.mtime else None,
            content_preview=content[:200] if content else None,
            found=True,
        )
        
        logger.info(
            "[evidence_collector] Loaded architecture map: %s (mtime: %s)",
            resolved.filename,
            source.mtime_human,
        )
        
        return content, source
        
    except Exception as e:
        logger.exception("[evidence_collector] Error loading architecture map: %s", e)
        return None, EvidenceSource(
            source_type="architecture_map",
            found=False,
            error=str(e),
        )


def load_codebase_report(
    max_lines: int = DEFAULT_MAX_LINES,
) -> Tuple[Optional[str], EvidenceSource]:
    """
    Load the latest codebase report FULL (by mtime).
    
    Returns:
        (content, evidence_source)
    """
    if not _REPORT_RESOLVER_AVAILABLE or not get_latest_codebase_report_full:
        return None, EvidenceSource(
            source_type="codebase_report",
            found=False,
            error="Report resolver not available",
        )
    
    try:
        resolved = get_latest_codebase_report_full()
        
        if not resolved.found:
            return None, EvidenceSource(
                source_type="codebase_report",
                found=False,
                error=resolved.error or "Not found",
            )
        
        content, truncated = read_report_content(resolved, max_lines=max_lines)
        
        repo_path = _to_repo_relative(str(resolved.path)) if resolved.path else None
        
        source = EvidenceSource(
            source_type="codebase_report",
            filename=resolved.filename,
            path=repo_path,
            mtime=resolved.mtime,
            mtime_human=resolved.mtime.strftime("%Y-%m-%d %H:%M:%S") if resolved.mtime else None,
            content_preview=content[:200] if content else None,
            found=True,
        )
        
        logger.info(
            "[evidence_collector] Loaded codebase report: %s (mtime: %s)",
            resolved.filename,
            source.mtime_human,
        )
        
        return content, source
        
    except Exception as e:
        logger.exception("[evidence_collector] Error loading codebase report: %s", e)
        return None, EvidenceSource(
            source_type="codebase_report",
            found=False,
            error=str(e),
        )


def read_file(
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    head_lines: Optional[int] = None,
) -> Tuple[Optional[str], EvidenceSource]:
    """
    Read a file from the repo (read-only).
    
    Args:
        path: File path (can be relative or absolute)
        start_line: Start line (1-indexed, inclusive)
        end_line: End line (1-indexed, inclusive)
        head_lines: Read first N lines
        
    Returns:
        (content, evidence_source)
    """
    if not _LIVE_READ_AVAILABLE or not live_read_file:
        return None, EvidenceSource(
            source_type="file_read",
            path=path,
            found=False,
            error="live_read_file not available",
        )
    
    # Normalize path
    abs_path = _to_absolute_path(path)
    repo_path = _to_repo_relative(abs_path)
    
    try:
        content, total_lines, total_bytes, truncated, error = live_read_file(
            path=abs_path,
            start_line=start_line,
            end_line=end_line,
            head_lines=head_lines,
            debug=False,
        )
        
        if error:
            return None, EvidenceSource(
                source_type="file_read",
                path=repo_path,
                found=False,
                error=error,
            )
        
        # Determine line range for evidence
        line_range = None
        if start_line and end_line:
            line_range = (start_line, end_line)
        elif head_lines:
            line_range = (1, min(head_lines, total_lines))
        
        source = EvidenceSource(
            source_type="file_read",
            path=repo_path,
            line_range=line_range,
            content_preview=content[:200] if content else None,
            found=True,
        )
        
        logger.info(
            "[evidence_collector] Read file: %s (lines %s, %d bytes)",
            repo_path,
            line_range or "all",
            len(content) if content else 0,
        )
        
        return content, source
        
    except Exception as e:
        logger.exception("[evidence_collector] Error reading file %s: %s", path, e)
        return None, EvidenceSource(
            source_type="file_read",
            path=repo_path,
            found=False,
            error=str(e),
        )


def search_repo(
    query: str,
    limit: int = 20,
) -> Tuple[List[Dict], EvidenceSource]:
    """
    Search the repo for symbols/code (uses arch_query as fallback).
    
    This is a FALLBACK operation - prefer direct file reads when possible.
    
    Args:
        query: Search query
        limit: Max results
        
    Returns:
        (results, evidence_source)
    """
    if not _ARCH_QUERY_AVAILABLE or not search_symbols:
        return [], EvidenceSource(
            source_type="arch_query_fallback",
            query=query,
            found=False,
            error="arch_query not available",
            is_fallback=True,
        )
    
    try:
        results = search_symbols(query=query, limit=limit)
        
        # Check for error response
        if results and isinstance(results, list) and len(results) > 0:
            if isinstance(results[0], dict) and "error" in results[0]:
                return [], EvidenceSource(
                    source_type="arch_query_fallback",
                    query=query,
                    found=False,
                    error=results[0]["error"],
                    is_fallback=True,
                )
        
        source = EvidenceSource(
            source_type="arch_query_fallback",
            query=query,
            found=len(results) > 0,
            content_preview=f"Found {len(results)} results",
            is_fallback=True,
        )
        
        logger.info(
            "[evidence_collector] arch_query (FALLBACK) for '%s': %d results",
            query,
            len(results),
        )
        
        return results, source
        
    except Exception as e:
        logger.exception("[evidence_collector] Error in arch_query: %s", e)
        return [], EvidenceSource(
            source_type="arch_query_fallback",
            query=query,
            found=False,
            error=str(e),
            is_fallback=True,
        )


# =============================================================================
# EVIDENCE BUNDLE LOADING
# =============================================================================

def load_evidence(
    include_arch_map: bool = True,
    include_codebase_report: bool = True,
    arch_map_max_lines: int = 500,
    codebase_report_max_lines: int = 300,
) -> EvidenceBundle:
    """
    Load all primary evidence sources.
    
    This is the main entry point for evidence collection.
    Loads architecture map and codebase report by default.
    
    Args:
        include_arch_map: Whether to load architecture map
        include_codebase_report: Whether to load codebase report
        arch_map_max_lines: Max lines to read from arch map
        codebase_report_max_lines: Max lines to read from codebase report
        
    Returns:
        EvidenceBundle with loaded content and sources
    """
    bundle = EvidenceBundle()
    
    if include_arch_map:
        content, source = load_architecture_map(max_lines=arch_map_max_lines)
        bundle.arch_map_content = content
        bundle.add_source(source)
    
    if include_codebase_report:
        content, source = load_codebase_report(max_lines=codebase_report_max_lines)
        bundle.codebase_report_content = content
        bundle.add_source(source)
    
    return bundle


def add_file_read_to_bundle(
    bundle: EvidenceBundle,
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    head_lines: Optional[int] = None,
) -> Optional[str]:
    """
    Add a file read to an existing evidence bundle.
    
    Returns:
        File content if successful, None otherwise
    """
    content, source = read_file(
        path=path,
        start_line=start_line,
        end_line=end_line,
        head_lines=head_lines,
    )
    
    bundle.add_source(source)
    
    if content:
        repo_path = source.path or path
        bundle.file_reads[repo_path] = content
    
    return content


def add_search_to_bundle(
    bundle: EvidenceBundle,
    query: str,
    limit: int = 20,
) -> List[Dict]:
    """
    Add a search result to an existing evidence bundle.
    
    NOTE: This uses arch_query as fallback. Prefer direct file reads.
    
    Returns:
        Search results
    """
    results, source = search_repo(query=query, limit=limit)
    
    bundle.add_source(source)
    bundle.arch_query_used = source.is_fallback and source.found
    
    if results:
        bundle.search_results[query] = results
    
    return results


# =============================================================================
# PATH UTILITIES
# =============================================================================

def _to_repo_relative(path: str) -> str:
    """Convert absolute path to repo-relative path."""
    # Standard repo roots
    repo_roots = [
        r"D:\Orb",
        r"D:/Orb",
        r"D:\\Orb",
    ]
    
    path_normalized = path.replace("\\", "/")
    
    for root in repo_roots:
        root_normalized = root.replace("\\", "/")
        if path_normalized.startswith(root_normalized):
            relative = path_normalized[len(root_normalized):]
            if relative.startswith("/"):
                relative = relative[1:]
            return relative
    
    # Return as-is if not in repo
    return path


def _to_absolute_path(path: str) -> str:
    """Convert relative path to absolute path in repo."""
    if Path(path).is_absolute():
        return path
    
    # Assume relative to D:\Orb
    return str(Path(r"D:\Orb") / path)


# =============================================================================
# GROUNDING UTILITIES
# =============================================================================

def find_in_evidence(
    bundle: EvidenceBundle,
    pattern: str,
    source_type: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """
    Search for a pattern in loaded evidence.
    
    Args:
        bundle: Evidence bundle to search
        pattern: Regex pattern to find
        source_type: Limit to specific source type (architecture_map, codebase_report, etc.)
        
    Returns:
        List of (source_name, matched_content) tuples
    """
    results = []
    regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    
    # Search architecture map
    if (source_type is None or source_type == "architecture_map") and bundle.arch_map_content:
        matches = regex.findall(bundle.arch_map_content)
        for match in matches:
            results.append(("architecture_map", match))
    
    # Search codebase report
    if (source_type is None or source_type == "codebase_report") and bundle.codebase_report_content:
        matches = regex.findall(bundle.codebase_report_content)
        for match in matches:
            results.append(("codebase_report", match))
    
    # Search file reads
    if source_type is None or source_type == "file_read":
        for path, content in bundle.file_reads.items():
            matches = regex.findall(content)
            for match in matches:
                results.append((f"file:{path}", match))
    
    return results


def verify_path_exists(
    bundle: EvidenceBundle,
    path: str,
) -> Tuple[bool, Optional[str]]:
    """
    Verify a path exists using evidence (no additional reads).
    
    Args:
        bundle: Evidence bundle
        path: Path to verify (repo-relative)
        
    Returns:
        (exists, source) - where source is which evidence confirmed it
    """
    # Normalize path for comparison
    path_normalized = path.replace("\\", "/").lower()
    
    # Check architecture map
    if bundle.arch_map_content:
        if path_normalized in bundle.arch_map_content.lower():
            return True, "architecture_map"
    
    # Check codebase report
    if bundle.codebase_report_content:
        if path_normalized in bundle.codebase_report_content.lower():
            return True, "codebase_report"
    
    # Check if we've already read this file
    for read_path in bundle.file_reads.keys():
        if read_path.replace("\\", "/").lower() == path_normalized:
            return True, f"file_read:{read_path}"
    
    return False, None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data structures
    "EvidenceSource",
    "EvidenceBundle",
    # Write protection
    "refuse_write_operation",
    "WRITE_REFUSED_ERROR",
    # Primary loading
    "load_architecture_map",
    "load_codebase_report",
    "read_file",
    "search_repo",
    # Bundle operations
    "load_evidence",
    "add_file_read_to_bundle",
    "add_search_to_bundle",
    # Grounding utilities
    "find_in_evidence",
    "verify_path_exists",
    # Path utilities
    "_to_repo_relative",
    "_to_absolute_path",
]
