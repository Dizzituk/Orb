# FILE: app/llm/local_tools/zobie/streams/fs_query.py
# Purpose: FILESYSTEM QUERY stream generator (v5.7 - Stage 2 Commentary Fixes).
# Called-by: app.llm.local_tools.zobie.streams, app.llm.local_tools.zobie.streams._fs_query_utils_2, app.llm.local_tools.zobie.streams._fs_query_utils_3
# Depends-on: app.llm.audit_logger, app.llm.local_tools.zobie.config, app.llm.local_tools.zobie.db_ops, app.llm.local_tools.zobie.fs_command_parser (+10 more)
# Last-renovated: 2026-06-11
"""FILESYSTEM QUERY stream generator (v5.7 - Stage 2 Commentary Fixes).

Answers filesystem listing/find/read queries AND write operations using:
1. DB index first (fast, from scan sandbox)
2. Live surgical read fallback (if DB lacks content)
3. Remote agent fallback via sandbox controller (if local read fails for OneDrive)
4. Sandbox write operations (append, overwrite, delete_area)

Supports BOTH natural language AND explicit command mode:
- Natural: "What's in my desktop?", "Show me line 45-65 of router.py"
- Command: "Astra, command: list C:\\path", "command: read \"file with spaces.txt\""

Read Commands:
- list <path>          : List files/folders in path
- find <n> [under <path>] : Search for file/folder by name
- read <path>          : Read entire file (with limits)
- head <path> <n>      : Read first N lines (default 20)
- lines <path> <start> <end> : Read specific line range

Write Commands (Stage 1):
- append <path> "content" : Append text to file
- overwrite <path> "content" : Replace entire file
- delete_area <path>   : Delete between ASTRA_BLOCK markers
- delete_lines <path> <start> <end> : Delete line range

v5.7 (2026-01): Stage 2 Fixes
  - Fixed WriteResult.source crash (WriteResult has no .source attribute)
  - Wrapped commentary rendering in try/except to prevent SSE stream crashes
  - Commentary failures now log but don't kill the stream
v5.6 (2026-01): Stage 2 - Added optional commentary rendering via Gemini 2.0 Flash
  - Enabled via ORB_COMMENTARY_ENABLED=true (default OFF)
  - Commentary provides conversational explanation of tool results
  - Deterministic tool outputs remain the driver; LLM is renderer only
v5.5 (2026-01): Added Stage 1 write operations (append, overwrite, delete_area, delete_lines)
v5.4 (2026-01): Added remote_agent fallback via sandbox controller for OneDrive paths
v5.3 (2026-01): Enhanced debug output in SSE stream for OneDrive diagnosis
v5.2 (2026-01): Fixed READ for OneDrive paths, refactored into modules
v5.1 (2026-01): Fixed quoted path parsing for paths with spaces
v5.0 (2026-01): Added surgical live read fallback, command parsing, line ranges
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import AsyncGenerator, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.llm.audit_logger import RoutingTrace
from app.memory import schemas as memory_schemas
from app.memory import service as memory_service

from ..config import (
    FILESYSTEM_QUERY_ALLOWED_ROOTS,
    FILESYSTEM_QUERY_MAX_ENTRIES,
    FILESYSTEM_READ_MAX_LINES,
    FILESYSTEM_READ_MAX_BYTES,
)
from ..sse import sse_token, sse_error, sse_done
from ..db_ops import (
    ARCH_MODELS_AVAILABLE,
    ArchitectureFileIndex,
    get_latest_scan,
)
from ..fs_path_utils import (
    normalize_path,
    is_path_allowed,
    get_basename,
    get_language_from_extension,
    looks_like_file,
)
from ..fs_live_ops import (
    live_read_file,
    live_read_file_with_remote_fallback,
    live_list_directory,
)
from ..fs_command_parser import (
    parse_filesystem_query,
)
from ..fs_write_ops import (
    sandbox_append_file,
    sandbox_overwrite_file,
    sandbox_delete_area,
    sandbox_delete_lines,
    WriteResult,
)
from app.llm.local_tools.zobie.streams._fs_query_utils_2 import _apply_line_limits, _handle_append_query, _handle_delete_area_query, _handle_delete_lines_query, _handle_overwrite_query, _process_list_results
from app.llm.local_tools.zobie.streams._fs_query_utils_3 import _format_write_result, _handle_read_query, generate_filesystem_query_stream

# Stage 2: Commentary rendering (optional)
try:
    from ..tool_commentary import (
        is_commentary_enabled,
        render_tool_commentary,
        create_read_result,
        create_write_result,
        create_list_result,
        create_find_result,
        ToolResult,
    )
    _COMMENTARY_AVAILABLE = True
except ImportError:
    _COMMENTARY_AVAILABLE = False
    is_commentary_enabled = lambda: False

logger = logging.getLogger(__name__)


# =============================================================================
# MAIN STREAM GENERATOR
# =============================================================================


# =============================================================================
# WRITE QUERY HANDLERS (Stage 1)
# =============================================================================


# =============================================================================
# READ QUERY HANDLERS
# =============================================================================

async def _handle_find_query(
    project_id: int,
    db: Session,
    target_path: Optional[str],
    search_term: Optional[str],
    trace: Optional[RoutingTrace],
    started_ms: int,
    user_message: str = "",
) -> AsyncGenerator[str, None]:
    """Handle find queries using DB index."""
    loop = asyncio.get_event_loop()
    
    if not search_term:
        yield sse_token("❌ No search term specified for find.\n")
        yield sse_done(
            provider="local",
            model="filesystem_query",
            success=False,
            error="no_search_term",
        )
        return
    
    # Check DB availability
    if not ARCH_MODELS_AVAILABLE:
        yield sse_token("⚠️ Architecture database not available.\n")
        yield sse_token("Run `scan sandbox` first to enable find.\n")
        yield sse_done(
            provider="local",
            model="filesystem_query",
            success=False,
            error="models_not_available",
        )
        return
    
    latest_scan = get_latest_scan(db, scope="sandbox") if get_latest_scan else None
    if not latest_scan:
        yield sse_token("⚠️ No scan data found. Run `scan sandbox` first.\n")
        yield sse_done(
            provider="local",
            model="filesystem_query",
            success=False,
            error="no_scan_data",
        )
        return
    
    yield sse_token(f"🔍 Searching DB for '{search_term}'...\n\n")
    
    try:
        query = db.query(ArchitectureFileIndex).filter(
            ArchitectureFileIndex.scan_id == latest_scan.id,
            ArchitectureFileIndex.name.ilike(f"%{search_term}%")
        )
        
        if target_path:
            path_prefix = normalize_path(target_path).rstrip('\\') + '\\'
            query = query.filter(
                ArchitectureFileIndex.path.like(f"{path_prefix}%")
            )
        
        results = query.limit(FILESYSTEM_QUERY_MAX_ENTRIES + 1).all()
        truncated = len(results) > FILESYSTEM_QUERY_MAX_ENTRIES
        if truncated:
            results = results[:FILESYSTEM_QUERY_MAX_ENTRIES]
        
    except Exception as e:
        logger.exception(f"[FS_QUERY] Find query failed: {e}")
        yield sse_error(f"Database query failed: {e}")
        yield sse_done(
            provider="local",
            model="filesystem_query",
            success=False,
            error=str(e),
        )
        return
    
    if not results:
        yield sse_token(f"📭 No files found matching '{search_term}'\n")
    else:
        yield sse_token(f"**Found {len(results)} match(es):**\n\n")
        for entry in results:
            size_str = _format_size(entry.size_bytes) if entry.size_bytes else ""
            yield sse_token(f"📄 {entry.path}{size_str}\n")
        
        if truncated:
            yield sse_token(f"\n⚠️ Results truncated to {FILESYSTEM_QUERY_MAX_ENTRIES} entries\n")
    
    duration_ms = int(loop.time() * 1000) - started_ms
    
    if trace:
        trace.log_model_call(
            "local_tool", "local", "filesystem_query", "find",
            0, 0, duration_ms, success=True, error=None,
        )
    
    # Stage 2: Optional commentary (wrapped in try/except to prevent stream crash)
    if _COMMENTARY_AVAILABLE and is_commentary_enabled():
        try:
            yield sse_token("\n---\n**Commentary:**\n")
            tool_result = create_find_result(
                ok=True,
                search_term=search_term,
                results_count=len(results),
                path=target_path or "",
                source="db",
                user_message=user_message,
            )
            async for commentary_chunk in render_tool_commentary(tool_result, user_message):
                yield sse_token(commentary_chunk)
            yield sse_token("\n")
        except Exception as e:
            logger.exception("[COMMENTARY] Render failed for find: %s", e)
            # Do not raise - continue silently, deterministic output already complete
    
    yield sse_done(
        provider="local",
        model="filesystem_query",
        total_length=len(results),
        meta={"query_type": "find", "search_term": search_term, "results": len(results)},
    )


async def _handle_list_query(
    project_id: int,
    db: Session,
    target_path: str,
    trace: Optional[RoutingTrace],
    started_ms: int,
    user_message: str = "",
) -> AsyncGenerator[str, None]:
    """Handle list queries using DB-first, live fallback."""
    loop = asyncio.get_event_loop()
    
    # Normalize path
    target_path = normalize_path(target_path, debug=True)
    
    # Try DB first
    db_success = False
    folders_list = []
    files_list = []
    source = "unknown"
    
    if ARCH_MODELS_AVAILABLE:
        latest_scan = get_latest_scan(db, scope="sandbox") if get_latest_scan else None
        
        if latest_scan:
            yield sse_token(f"[FS_QUERY] db_lookup=True scan_id={latest_scan.id}\n")
            
            try:
                path_prefix = target_path.rstrip('\\') + '\\'
                results = db.query(ArchitectureFileIndex).filter(
                    ArchitectureFileIndex.scan_id == latest_scan.id,
                    ArchitectureFileIndex.path.like(f"{path_prefix}%")
                ).limit(FILESYSTEM_QUERY_MAX_ENTRIES + 1).all()
                
                if results:
                    db_success = True
                    source = "db"
                    folders_list, files_list = _process_list_results(
                        results[:FILESYSTEM_QUERY_MAX_ENTRIES], path_prefix
                    )
                    
            except Exception as e:
                logger.warning(f"[FS_QUERY] DB list failed, will try live: {e}")
    
    # Fall back to live read if DB didn't work
    if not db_success:
        yield sse_token(f"[FS_QUERY] live_read=True (DB miss)\n")
        source = "local"
        
        folders, files, error = live_list_directory(target_path, debug=True)
        if error:
            yield sse_token(f"❌ {error}\n")
            yield sse_done(
                provider="local",
                model="filesystem_query",
                success=False,
                error=error,
            )
            return
        
        folders_list = folders
        files_list = [{"path": f, "name": os.path.basename(f)} for f in files]
    
    # Output results
    yield sse_token(f"\n📁 **Contents of {target_path}**\n\n")
    
    if folders_list:
        yield sse_token(f"**Folders ({len(folders_list)}):**\n")
        for folder in folders_list[:50]:
            if isinstance(folder, str):
                yield sse_token(f"📁 {folder}\n")
            else:
                yield sse_token(f"📁 {folder}\n")
        if len(folders_list) > 50:
            yield sse_token(f"   ... and {len(folders_list) - 50} more\n")
        yield sse_token("\n")
    
    if files_list:
        yield sse_token(f"**Files ({len(files_list)}):**\n")
        for f in files_list[:50]:
            if isinstance(f, dict):
                yield sse_token(f"📄 {f.get('path', f.get('name', str(f)))}\n")
            elif hasattr(f, 'path'):
                size_str = _format_size(f.size_bytes) if hasattr(f, 'size_bytes') and f.size_bytes else ""
                yield sse_token(f"📄 {f.path}{size_str}\n")
            else:
                yield sse_token(f"📄 {f}\n")
        if len(files_list) > 50:
            yield sse_token(f"   ... and {len(files_list) - 50} more\n")
        yield sse_token("\n")
    
    if not folders_list and not files_list:
        yield sse_token("📭 Directory is empty or not found in index.\n")
    
    yield sse_token(f"---\n**Total:** {len(folders_list)} folders, {len(files_list)} files\n")
    
    duration_ms = int(loop.time() * 1000) - started_ms
    
    if trace:
        trace.log_model_call(
            "local_tool", "local", "filesystem_query", "list",
            0, 0, duration_ms, success=True, error=None,
        )
    
    # Stage 2: Optional commentary (wrapped in try/except to prevent stream crash)
    if _COMMENTARY_AVAILABLE and is_commentary_enabled():
        try:
            yield sse_token("\n---\n**Commentary:**\n")
            tool_result = create_list_result(
                ok=True,
                path=target_path,
                source=source,
                folders_count=len(folders_list),
                files_count=len(files_list),
                user_message=user_message,
            )
            async for commentary_chunk in render_tool_commentary(tool_result, user_message):
                yield sse_token(commentary_chunk)
            yield sse_token("\n")
        except Exception as e:
            logger.exception("[COMMENTARY] Render failed for list: %s", e)
            # Do not raise - continue silently, deterministic output already complete
    
    yield sse_done(
        provider="local",
        model="filesystem_query",
        total_length=len(folders_list) + len(files_list),
        meta={
            "query_type": "list",
            "target_path": target_path,
            "folders": len(folders_list),
            "files": len(files_list),
            "db_hit": db_success,
        },
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _format_size(size_bytes: int) -> str:
    """Format file size for display."""
    if not size_bytes:
        return ""
    if size_bytes > 1_000_000:
        return f" ({size_bytes / 1_000_000:.1f} MB)"
    elif size_bytes > 1_000:
        return f" ({size_bytes / 1_000:.1f} KB)"
    return f" ({size_bytes} bytes)"
