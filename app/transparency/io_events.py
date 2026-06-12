# FILE: app/transparency/io_events.py
# Purpose: IO Event dataclasses for the Pipeline Transparency system.
# Called-by: app.transparency.collector, app.transparency.io_tracker
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
IO Event dataclasses for the Pipeline Transparency system.

Tracks every file read, write, existence check, and directory scan
performed by the pipeline. The critical field is `source` — it proves
whether the operation targeted the sandbox (correct for repo code)
or the host (correct only for pipeline operational data).

A source of "host:VIOLATION" means repo code was read from the host
filesystem instead of the sandbox. This is a bug and must be visually
flagged in the UI.

v1.0 (2026-03): Initial implementation — Pipeline Logging Overhaul.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# File extensions that belong to repo code (not pipeline operational data).
# If these are read from the host, it's a violation.
_REPO_CODE_EXTENSIONS = frozenset({
    ".py", ".tsx", ".ts", ".jsx", ".js",
    ".css", ".scss", ".less",
    ".html", ".htm",
    ".json",  # config/package.json etc.
    ".yaml", ".yml",
    ".toml",
    ".sql",
    ".graphql",
    ".prisma",
    ".svelte", ".vue",
})

# Host paths that are legitimate for pipeline operational data.
# Reads from these paths are NOT violations even for code-like extensions.
_HOST_OPERATIONAL_PREFIXES = (
    r"D:\Orb\jobs",
    r"D:\Orb\app",           # pipeline's own code, not the project being built
    r"D:\Orb\segments",
    r"D:\Orb\experience",
    r"D:\Orb\logs",
    r"D:\Orb\cost",
    r"D:\Orb\alembic",
)


def classify_source(path: str, declared_source: str) -> str:
    """Classify whether an IO operation source is valid.

    Args:
        path: The file path being accessed.
        declared_source: What the caller says the source is
                         ("sandbox" or "host").

    Returns:
        "sandbox"          — sandbox operation, always valid for repo code.
        "host"             — host operation on pipeline operational data, valid.
        "host:VIOLATION"   — host operation on what looks like repo code. Bug.
    """
    if declared_source == "sandbox":
        return "sandbox"

    if declared_source != "host":
        return declared_source

    # It's a host read — check if it's legitimate operational data
    normalised = path.replace("/", "\\")
    for prefix in _HOST_OPERATIONAL_PREFIXES:
        if normalised.startswith(prefix):
            return "host"

    # Host read, not in operational prefixes — check extension
    ext = ""
    dot_pos = path.rfind(".")
    if dot_pos != -1:
        ext = path[dot_pos:].lower()

    if ext in _REPO_CODE_EXTENSIONS:
        return "host:VIOLATION"

    # Unknown extension on host — allow but don't flag
    return "host"


@dataclass
class IOEvent:
    """A single IO operation performed by the pipeline.

    This is the atomic unit of the logging system. Every file read,
    write, existence check, and directory scan produces one IOEvent.
    """

    operation: str
    """One of: "read", "write", "exists_check", "dir_scan"."""

    path: str
    """The file or directory path."""

    source: str
    """Where the operation happened: "sandbox", "host", or "host:VIOLATION"."""

    purpose: str = ""
    """Why this operation is happening, in plain English.
    E.g. "evidence gathering — checking existing route structure"."""

    content_summary: str = ""
    """Brief description of what was found / what was written.
    E.g. "React component, 45 lines, exports DashboardView"."""

    intent: str = ""
    """For writes: what is being created and why.
    E.g. "Pydantic schemas for debug project request/response validation"."""

    bytes_count: int = 0
    """Size of the content read or written."""

    timestamp: str = field(default_factory=_utc_now)
    """ISO timestamp of when the operation occurred."""

    stage_name: str = ""
    """Which pipeline stage performed this operation."""

    is_violation: bool = False
    """True if this is a sandbox enforcement violation."""

    def to_dict(self) -> dict:
        return {
            "operation": self.operation,
            "path": self.path,
            "source": self.source,
            "purpose": self.purpose,
            "content_summary": self.content_summary,
            "intent": self.intent,
            "bytes_count": self.bytes_count,
            "timestamp": self.timestamp,
            "stage_name": self.stage_name,
            "is_violation": self.is_violation,
        }

    def to_sse_dict(self) -> dict:
        """Format for SSE emission — includes type field."""
        d = self.to_dict()
        d["type"] = "io_operation"
        return d


def make_read_event(
    path: str,
    source: str,
    purpose: str = "",
    content_summary: str = "",
    bytes_count: int = 0,
    stage_name: str = "",
) -> IOEvent:
    """Factory for file read events."""
    classified = classify_source(path, source)
    return IOEvent(
        operation="read",
        path=path,
        source=classified,
        purpose=purpose,
        content_summary=content_summary,
        bytes_count=bytes_count,
        stage_name=stage_name,
        is_violation="VIOLATION" in classified,
    )


def make_write_event(
    path: str,
    target: str,
    intent: str = "",
    content_summary: str = "",
    bytes_count: int = 0,
    stage_name: str = "",
) -> IOEvent:
    """Factory for file write events."""
    classified = classify_source(path, target)
    return IOEvent(
        operation="write",
        path=path,
        source=classified,
        intent=intent,
        content_summary=content_summary,
        bytes_count=bytes_count,
        stage_name=stage_name,
        is_violation="VIOLATION" in classified,
    )


def make_exists_event(
    path: str,
    source: str,
    result: bool,
    purpose: str = "",
    stage_name: str = "",
) -> IOEvent:
    """Factory for file/dir existence check events."""
    classified = classify_source(path, source)
    return IOEvent(
        operation="exists_check",
        path=path,
        source=classified,
        purpose=purpose,
        content_summary=f"exists={result}",
        stage_name=stage_name,
        is_violation="VIOLATION" in classified,
    )


def make_dir_scan_event(
    path: str,
    source: str,
    file_count: int = 0,
    purpose: str = "",
    stage_name: str = "",
) -> IOEvent:
    """Factory for directory listing events."""
    classified = classify_source(path, source)
    return IOEvent(
        operation="dir_scan",
        path=path,
        source=classified,
        purpose=purpose,
        content_summary=f"{file_count} files found",
        stage_name=stage_name,
        is_violation="VIOLATION" in classified,
    )


__all__ = [
    "IOEvent",
    "classify_source",
    "make_read_event",
    "make_write_event",
    "make_exists_event",
    "make_dir_scan_event",
]
