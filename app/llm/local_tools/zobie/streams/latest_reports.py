# FILE: app/llm/local_tools/zobie/streams/latest_reports.py
"""
LATEST REPORT stream generators.

Provides SSE stream output for:
- latest architecture map
- latest codebase report full

Commands:
- "Astra, command: latest architecture map"
- "Astra, command: latest codebase report full"
- "Orb, command: latest architecture map"
- "Orb, command: latest codebase report full"

v1.0 (2026-01): Initial implementation
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator, Optional

from sqlalchemy.orm import Session

from app.llm.audit_logger import RoutingTrace
from ..sse import sse_token, sse_error, sse_done

# Import resolver
from ..latest_report_resolver import (
    get_latest_architecture_map,
    get_latest_codebase_report_full,
    read_report_content,
    ResolvedReport,
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default preview line count
DEFAULT_PREVIEW_LINES = 100

# Maximum content to show in full mode (characters)
MAX_FULL_CONTENT_CHARS = 200_000


# =============================================================================
# STREAM GENERATORS
# =============================================================================

async def generate_latest_architecture_map_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream generator for latest architecture map command.
    
    Resolves and displays the latest ARCHITECTURE_MAP*.md file
    from D:\Orb\.architecture\.
    """
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    
    yield sse_token("📄 **[LATEST_REPORT]** Resolving latest architecture map...\n\n")
    
    # Resolve latest file
    resolved = get_latest_architecture_map()
    
    # Show resolution info
    yield sse_token(f"**Directory:** `{resolved.searched_dir}`\n")
    yield sse_token(f"**Patterns:** `{', '.join(resolved.searched_patterns or [])}`\n\n")
    
    if not resolved.found:
        yield sse_token(f"❌ **Not Found:** {resolved.error}\n\n")
        yield sse_token("**Possible causes:**\n")
        yield sse_token("- No architecture map has been generated yet\n")
        yield sse_token("- Run `CREATE ARCHITECTURE MAP` to generate one\n")
        
        yield sse_done(
            provider="local",
            model="latest_report_resolver",
            success=False,
            error="file_not_found",
        )
        return
    
    # Found - show file info
    yield sse_token(f"✅ **Found:** `{resolved.filename}`\n")
    yield sse_token(f"   **Modified:** {resolved.mtime.strftime('%Y-%m-%d %H:%M:%S') if resolved.mtime else 'unknown'}\n")
    yield sse_token(f"   **Size:** {_format_size(resolved.size_bytes) if resolved.size_bytes else 'unknown'}\n")
    yield sse_token(f"   **Path:** `{resolved.path}`\n\n")
    
    # Read and display content
    yield sse_token("---\n\n")
    yield sse_token(f"### Content Preview (first {DEFAULT_PREVIEW_LINES} lines)\n\n")
    
    content, truncated = read_report_content(resolved, max_lines=DEFAULT_PREVIEW_LINES)
    
    # Stream content in chunks to avoid large payloads
    chunk_size = 2000
    for i in range(0, len(content), chunk_size):
        chunk = content[i:i + chunk_size]
        yield sse_token(chunk)
        # Small delay to avoid overwhelming the client
        await asyncio.sleep(0.01)
    
    yield sse_token("\n\n---\n\n")
    
    if truncated:
        yield sse_token(f"*[Preview truncated to {DEFAULT_PREVIEW_LINES} lines. Full file available at the path above.]*\n")
    else:
        yield sse_token("*[Complete file shown above.]*\n")
    
    elapsed_ms = int(loop.time() * 1000) - started_ms
    
    yield sse_done(
        provider="local",
        model="latest_report_resolver",
        success=True,
        extra={
            "filename": resolved.filename,
            "mtime": resolved.mtime.isoformat() if resolved.mtime else None,
            "size_bytes": resolved.size_bytes,
            "elapsed_ms": elapsed_ms,
        },
    )


async def generate_latest_codebase_report_full_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream generator for latest codebase report (FULL) command.
    
    Resolves and displays the latest CODEBASE_REPORT_FULL_*.md file
    from D:\Orb\.architecture\.
    
    IMPORTANT: Only returns FULL reports (not FAST).
    """
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    
    yield sse_token("📄 **[LATEST_REPORT]** Resolving latest codebase report (FULL)...\n\n")
    
    # Resolve latest file
    resolved = get_latest_codebase_report_full()
    
    # Show resolution info
    yield sse_token(f"**Directory:** `{resolved.searched_dir}`\n")
    yield sse_token(f"**Patterns:** `{', '.join(resolved.searched_patterns or [])}`\n\n")
    
    if not resolved.found:
        yield sse_token(f"❌ **Not Found:** {resolved.error}\n\n")
        yield sse_token("**Possible causes:**\n")
        yield sse_token("- No full codebase report has been generated yet\n")
        yield sse_token("- Run `codebase report full` to generate one\n")
        yield sse_token("- Note: FAST reports are NOT matched by this command\n")
        
        yield sse_done(
            provider="local",
            model="latest_report_resolver",
            success=False,
            error="file_not_found",
        )
        return
    
    # Found - show file info
    yield sse_token(f"✅ **Found:** `{resolved.filename}`\n")
    yield sse_token(f"   **Modified:** {resolved.mtime.strftime('%Y-%m-%d %H:%M:%S') if resolved.mtime else 'unknown'}\n")
    yield sse_token(f"   **Size:** {_format_size(resolved.size_bytes) if resolved.size_bytes else 'unknown'}\n")
    yield sse_token(f"   **Path:** `{resolved.path}`\n\n")
    
    # Read and display content
    yield sse_token("---\n\n")
    yield sse_token(f"### Content Preview (first {DEFAULT_PREVIEW_LINES} lines)\n\n")
    
    content, truncated = read_report_content(resolved, max_lines=DEFAULT_PREVIEW_LINES)
    
    # Stream content in chunks
    chunk_size = 2000
    for i in range(0, len(content), chunk_size):
        chunk = content[i:i + chunk_size]
        yield sse_token(chunk)
        await asyncio.sleep(0.01)
    
    yield sse_token("\n\n---\n\n")
    
    if truncated:
        yield sse_token(f"*[Preview truncated to {DEFAULT_PREVIEW_LINES} lines. Full file available at the path above.]*\n")
    else:
        yield sse_token("*[Complete file shown above.]*\n")
    
    elapsed_ms = int(loop.time() * 1000) - started_ms
    
    yield sse_done(
        provider="local",
        model="latest_report_resolver",
        success=True,
        extra={
            "filename": resolved.filename,
            "mtime": resolved.mtime.isoformat() if resolved.mtime else None,
            "size_bytes": resolved.size_bytes,
            "elapsed_ms": elapsed_ms,
        },
    )


# =============================================================================
# HELPERS
# =============================================================================

def _format_size(size_bytes: Optional[int]) -> str:
    """Format bytes as human-readable string."""
    if size_bytes is None:
        return "unknown"
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "generate_latest_architecture_map_stream",
    "generate_latest_codebase_report_full_stream",
]
