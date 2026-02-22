# FILE: app/llm/local_tools/zobie/streams/fs_query.py
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
import sys
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

async def generate_filesystem_query_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """
    Answer filesystem queries using DB-first approach with live read fallback.
    Also handles Stage 1 write operations via sandbox.
    
    v5.5: Added write operations (append, overwrite, delete_area, delete_lines)
    
    Flow:
    1. Parse query (command or natural language)
    2. Validate path against allowlist
    3. For reads: Try DB lookup first, fall back to live surgical read
    4. For writes: Execute via sandbox controller
    5. Return results with appropriate formatting
    """
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    
    # Parse the query
    parsed = parse_filesystem_query(message)
    query_type = parsed.get("query_type")
    target_path = parsed.get("path")
    search_term = parsed.get("search_term")
    start_line = parsed.get("start_line")
    end_line = parsed.get("end_line")
    head_lines = parsed.get("head_lines")
    content = parsed.get("content")
    source = parsed.get("source", "natural")
    
    yield sse_token(f"📂 [FS_QUERY] Processing {source} query...\n")
    yield sse_token(f"   action={query_type}")
    if target_path:
        yield sse_token(f" path=\"{target_path}\"")
    if search_term:
        yield sse_token(f" search=\"{search_term}\"")
    if head_lines:
        yield sse_token(f" head={head_lines}")
    if start_line and end_line:
        yield sse_token(f" lines={start_line}-{end_line}")
    if content is not None:
        yield sse_token(f" content_len={len(content)}")
    yield sse_token("\n\n")
    
    # Handle missing query type
    if not query_type:
        yield sse_token("❓ Could not determine query type.\n")
        yield sse_token("Supported commands: list, find, read, head, lines, append, overwrite, delete_area, delete_lines\n")
        yield sse_done(
            provider="local",
            model="filesystem_query",
            success=False,
            error="unknown_query_type",
        )
        return
    
    # Handle missing path for path-requiring commands
    if query_type in ("list", "read", "head", "lines", "append", "overwrite", "delete_area", "delete_lines") and not target_path:
        yield sse_token("❌ No path specified.\n")
        yield sse_token(f"Usage: {query_type} <path>\n")
        yield sse_done(
            provider="local",
            model="filesystem_query",
            success=False,
            error="no_path_specified",
        )
        return
    
    # Validate path against allowlist (skip for find without path - search term only)
    if target_path:
        allowed, reason = is_path_allowed(target_path)
        if not allowed:
            yield sse_token(f"🚫 [blocked_by_allowlist] {reason}\n")
            yield sse_token(f"   Path: {target_path}\n\n")
            yield sse_token("Allowed roots:\n")
            for root in FILESYSTEM_QUERY_ALLOWED_ROOTS[:5]:
                yield sse_token(f"  ✅ {root}\n")
            yield sse_done(
                provider="local",
                model="filesystem_query",
                success=False,
                error="blocked_by_allowlist",
            )
            return
    
    # ==========================================================================
    # WRITE OPERATIONS (Stage 1)
    # ==========================================================================
    
    if query_type == "append":
        async for chunk in _handle_append_query(
            project_id, db, target_path, content, trace, started_ms, message
        ):
            yield chunk
        return
    
    if query_type == "overwrite":
        async for chunk in _handle_overwrite_query(
            project_id, db, target_path, content, trace, started_ms, message
        ):
            yield chunk
        return
    
    if query_type == "delete_area":
        async for chunk in _handle_delete_area_query(
            project_id, db, target_path, trace, started_ms, message
        ):
            yield chunk
        return
    
    if query_type == "delete_lines":
        async for chunk in _handle_delete_lines_query(
            project_id, db, target_path, start_line, end_line, trace, started_ms, message
        ):
            yield chunk
        return
    
    # ==========================================================================
    # READ OPERATIONS
    # ==========================================================================
    
    # Dispatch by query type
    if query_type == "find":
        async for chunk in _handle_find_query(
            project_id, db, target_path, search_term, trace, started_ms, message
        ):
            yield chunk
        return
    
    if query_type == "list":
        async for chunk in _handle_list_query(
            project_id, db, target_path, trace, started_ms, message
        ):
            yield chunk
        return
    
    if query_type in ("read", "head", "lines"):
        async for chunk in _handle_read_query(
            project_id, db, target_path, query_type,
            start_line, end_line, head_lines, trace, started_ms, message
        ):
            yield chunk
        return
    
    # Fallback
    yield sse_token(f"❓ Unknown query type: {query_type}\n")
    yield sse_done(
        provider="local",
        model="filesystem_query",
        success=False,
        error="unknown_query_type",
    )


# =============================================================================
# WRITE QUERY HANDLERS (Stage 1)
# =============================================================================

async def _handle_append_query(
    project_id: int,
    db: Session,
    target_path: str,
    content: Optional[str],
    trace: Optional[RoutingTrace],
    started_ms: int,
    user_message: str = "",
) -> AsyncGenerator[str, None]:
    """Handle append command."""
    loop = asyncio.get_event_loop()
    
    if content is None:
        yield sse_token("❌ No content provided for append.\n")
        yield sse_token("Usage: append <path> \"content\" OR append <path> with fenced block\n")
        yield sse_done(
            provider="local",
            model="filesystem_query",
            success=False,
            error="no_content",
        )
        return
    
    yield sse_token(f"📝 [APPEND] Appending {len(content)} chars to: {target_path}\n\n")
    
    # Execute via sandbox
    result = sandbox_append_file(target_path, content, debug=True)
    
    # Output result
    async for chunk in _format_write_result(result):
        yield chunk
    
    duration_ms = int(loop.time() * 1000) - started_ms
    
    if trace:
        trace.log_model_call(
            "local_tool", "local", "filesystem_query", "append",
            0, 0, duration_ms, success=result.status == "ok", error=result.error,
        )
    
    # Stage 2: Optional commentary (wrapped in try/except to prevent stream crash)
    if _COMMENTARY_AVAILABLE and is_commentary_enabled():
        try:
            yield sse_token("\n---\n**Commentary:**\n")
            tool_result = create_write_result(
                ok=result.status == "ok",
                action="append",
                path=target_path,
                source="remote_agent",  # Write operations go through sandbox/remote agent
                before_lines=result.lines_before,
                before_bytes=result.bytes_before,
                before_excerpt=result.preview_before or "",
                after_lines=result.lines_after,
                after_bytes=result.bytes_after,
                after_excerpt=result.preview_after or "",
                status_code=200 if result.status == "ok" else 500,
                bytes_written=len(content) if result.status == "ok" else None,
                error=result.error or "",
                user_message=user_message,
            )
            async for commentary_chunk in render_tool_commentary(tool_result, user_message):
                yield sse_token(commentary_chunk)
            yield sse_token("\n")
        except Exception as e:
            logger.exception("[COMMENTARY] Render failed for append: %s", e)
            # Do not raise - continue silently, deterministic output already complete
    
    yield sse_done(
        provider="local",
        model="filesystem_query",
        total_length=result.bytes_after,
        success=result.status == "ok",
        error=result.error,
        meta={"query_type": "append", "result": result.to_dict()},
    )


async def _handle_overwrite_query(
    project_id: int,
    db: Session,
    target_path: str,
    content: Optional[str],
    trace: Optional[RoutingTrace],
    started_ms: int,
    user_message: str = "",
) -> AsyncGenerator[str, None]:
    """Handle overwrite command."""
    loop = asyncio.get_event_loop()
    
    if content is None:
        yield sse_token("❌ No content provided for overwrite.\n")
        yield sse_token("Usage: overwrite <path> \"content\" OR overwrite <path> with fenced block\n")
        yield sse_done(
            provider="local",
            model="filesystem_query",
            success=False,
            error="no_content",
        )
        return
    
    yield sse_token(f"📝 [OVERWRITE] Replacing content of: {target_path}\n")
    yield sse_token(f"   New content: {len(content)} chars\n\n")
    
    # Execute via sandbox
    result = sandbox_overwrite_file(target_path, content, debug=True)
    
    # Output result
    async for chunk in _format_write_result(result):
        yield chunk
    
    duration_ms = int(loop.time() * 1000) - started_ms
    
    if trace:
        trace.log_model_call(
            "local_tool", "local", "filesystem_query", "overwrite",
            0, 0, duration_ms, success=result.status == "ok", error=result.error,
        )
    
    # Stage 2: Optional commentary (wrapped in try/except to prevent stream crash)
    if _COMMENTARY_AVAILABLE and is_commentary_enabled():
        try:
            yield sse_token("\n---\n**Commentary:**\n")
            tool_result = create_write_result(
                ok=result.status == "ok",
                action="overwrite",
                path=target_path,
                source="remote_agent",  # Write operations go through sandbox/remote agent
                before_lines=result.lines_before,
                before_bytes=result.bytes_before,
                before_excerpt=result.preview_before or "",
                after_lines=result.lines_after,
                after_bytes=result.bytes_after,
                after_excerpt=result.preview_after or "",
                status_code=200 if result.status == "ok" else 500,
                bytes_written=len(content) if result.status == "ok" else None,
                error=result.error or "",
                user_message=user_message,
            )
            async for commentary_chunk in render_tool_commentary(tool_result, user_message):
                yield sse_token(commentary_chunk)
            yield sse_token("\n")
        except Exception as e:
            logger.exception("[COMMENTARY] Render failed for overwrite: %s", e)
            # Do not raise - continue silently, deterministic output already complete
    
    yield sse_done(
        provider="local",
        model="filesystem_query",
        total_length=result.bytes_after,
        success=result.status == "ok",
        error=result.error,
        meta={"query_type": "overwrite", "result": result.to_dict()},
    )


async def _handle_delete_area_query(
    project_id: int,
    db: Session,
    target_path: str,
    trace: Optional[RoutingTrace],
    started_ms: int,
    user_message: str = "",
) -> AsyncGenerator[str, None]:
    """Handle delete_area command (marker-based deletion)."""
    loop = asyncio.get_event_loop()
    
    yield sse_token(f"🗑️ [DELETE_AREA] Removing content between ASTRA_BLOCK markers\n")
    yield sse_token(f"   Path: {target_path}\n")
    yield sse_token(f"   Start marker: # START ASTRA_BLOCK\n")
    yield sse_token(f"   End marker: # END ASTRA_BLOCK\n\n")
    
    # Execute via sandbox
    result = sandbox_delete_area(target_path, debug=True)
    
    # Output result
    async for chunk in _format_write_result(result):
        yield chunk
    
    duration_ms = int(loop.time() * 1000) - started_ms
    
    if trace:
        trace.log_model_call(
            "local_tool", "local", "filesystem_query", "delete_area",
            0, 0, duration_ms, success=result.status == "ok", error=result.error,
        )
    
    # Stage 2: Optional commentary (wrapped in try/except to prevent stream crash)
    if _COMMENTARY_AVAILABLE and is_commentary_enabled():
        try:
            yield sse_token("\n---\n**Commentary:**\n")
            tool_result = create_write_result(
                ok=result.status == "ok",
                action="delete_area",
                path=target_path,
                source="remote_agent",  # Write operations go through sandbox/remote agent
                before_lines=result.lines_before,
                before_bytes=result.bytes_before,
                before_excerpt=result.preview_before or "",
                after_lines=result.lines_after,
                after_bytes=result.bytes_after,
                after_excerpt=result.preview_after or "",
                status_code=200 if result.status == "ok" else 500,
                error=result.error or "",
                user_message=user_message,
            )
            async for commentary_chunk in render_tool_commentary(tool_result, user_message):
                yield sse_token(commentary_chunk)
            yield sse_token("\n")
        except Exception as e:
            logger.exception("[COMMENTARY] Render failed for delete_area: %s", e)
            # Do not raise - continue silently, deterministic output already complete
    
    yield sse_done(
        provider="local",
        model="filesystem_query",
        total_length=result.bytes_after,
        success=result.status == "ok",
        error=result.error,
        meta={"query_type": "delete_area", "result": result.to_dict()},
    )


async def _handle_delete_lines_query(
    project_id: int,
    db: Session,
    target_path: str,
    start_line: Optional[int],
    end_line: Optional[int],
    trace: Optional[RoutingTrace],
    started_ms: int,
    user_message: str = "",
) -> AsyncGenerator[str, None]:
    """Handle delete_lines command (line range deletion)."""
    loop = asyncio.get_event_loop()
    
    if start_line is None or end_line is None:
        yield sse_token("❌ Missing line range for delete_lines.\n")
        yield sse_token("Usage: delete_lines <path> <start_line> <end_line>\n")
        yield sse_done(
            provider="local",
            model="filesystem_query",
            success=False,
            error="missing_line_range",
        )
        return
    
    yield sse_token(f"🗑️ [DELETE_LINES] Removing lines {start_line}-{end_line}\n")
    yield sse_token(f"   Path: {target_path}\n\n")
    
    # Execute via sandbox
    result = sandbox_delete_lines(target_path, start_line, end_line, debug=True)
    
    # Output result
    async for chunk in _format_write_result(result):
        yield chunk
    
    duration_ms = int(loop.time() * 1000) - started_ms
    
    if trace:
        trace.log_model_call(
            "local_tool", "local", "filesystem_query", "delete_lines",
            0, 0, duration_ms, success=result.status == "ok", error=result.error,
        )
    
    # Stage 2: Optional commentary (wrapped in try/except to prevent stream crash)
    if _COMMENTARY_AVAILABLE and is_commentary_enabled():
        try:
            yield sse_token("\n---\n**Commentary:**\n")
            tool_result = create_write_result(
                ok=result.status == "ok",
                action="delete_lines",
                path=target_path,
                source="remote_agent",  # Write operations go through sandbox/remote agent
                before_lines=result.lines_before,
                before_bytes=result.bytes_before,
                before_excerpt=result.preview_before or "",
                after_lines=result.lines_after,
                after_bytes=result.bytes_after,
                after_excerpt=result.preview_after or "",
                status_code=200 if result.status == "ok" else 500,
                error=result.error or "",
                user_message=user_message,
            )
            async for commentary_chunk in render_tool_commentary(tool_result, user_message):
                yield sse_token(commentary_chunk)
            yield sse_token("\n")
        except Exception as e:
            logger.exception("[COMMENTARY] Render failed for delete_lines: %s", e)
            # Do not raise - continue silently, deterministic output already complete
    
    yield sse_done(
        provider="local",
        model="filesystem_query",
        total_length=result.bytes_after,
        success=result.status == "ok",
        error=result.error,
        meta={"query_type": "delete_lines", "result": result.to_dict()},
    )


async def _format_write_result(result: WriteResult) -> AsyncGenerator[str, None]:
    """Format write result for SSE output."""
    yield sse_token(f"\n{result.summary()}\n\n")
    
    if result.status == "ok":
        yield sse_token("**Before:**\n```\n")
        yield sse_token(result.preview_before or "(empty)")
        yield sse_token("\n```\n\n")
        
        yield sse_token("**After:**\n```\n")
        yield sse_token(result.preview_after or "(empty)")
        yield sse_token("\n```\n")


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


async def _handle_read_query(
    project_id: int,
    db: Session,
    target_path: str,
    query_type: str,
    start_line: Optional[int],
    end_line: Optional[int],
    head_lines: Optional[int],
    trace: Optional[RoutingTrace],
    started_ms: int,
    user_message: str = "",
) -> AsyncGenerator[str, None]:
    """Handle read/head/lines queries using DB-first, live fallback."""
    loop = asyncio.get_event_loop()
    
    # Normalize path (uses robust normalize_path)
    target_path = normalize_path(target_path, debug=True)
    basename = get_basename(target_path)
    
    # DEBUG: Output path diagnostics in SSE stream for visibility
    yield sse_token(f"[DEBUG] path_repr={repr(target_path)}\n")
    yield sse_token(f"[DEBUG] os.path.exists={os.path.exists(target_path)}\n")
    
    # Check if path looks like a folder
    if not looks_like_file(target_path):
        if os.path.isdir(target_path):
            yield sse_token(f"📁 `{target_path}` is a folder.\n")
            yield sse_token("💡 Use `list <path>` to see contents.\n")
            yield sse_done(
                provider="local",
                model="filesystem_query",
                success=False,
                error="path_is_folder",
            )
            return
    
    content = None
    total_lines = 0
    total_bytes = 0
    truncated = False
    source = "unknown"
    file_language = None
    
    # Try DB first
    if ARCH_MODELS_AVAILABLE:
        latest_scan = get_latest_scan(db, scope="sandbox") if get_latest_scan else None
        
        if latest_scan:
            try:
                # Try exact match
                file_entry = db.query(ArchitectureFileIndex).filter(
                    ArchitectureFileIndex.scan_id == latest_scan.id,
                    ArchitectureFileIndex.path == target_path
                ).first()
                
                # Case-insensitive fallback
                if not file_entry:
                    file_entry = db.query(ArchitectureFileIndex).filter(
                        ArchitectureFileIndex.scan_id == latest_scan.id,
                        func.lower(ArchitectureFileIndex.path) == target_path.lower()
                    ).first()
                
                if file_entry and file_entry.content and file_entry.content.content_text:
                    yield sse_token(f"[FS_QUERY] db_hit=True\n")
                    source = "db"
                    content = file_entry.content.content_text
                    total_bytes = len(content.encode('utf-8', errors='replace'))
                    file_language = file_entry.language
                    
                    content, total_lines, truncated = _apply_line_limits(
                        content, start_line, end_line, head_lines
                    )
                else:
                    yield sse_token(f"[FS_QUERY] db_miss=True (no content)\n")
                    
            except Exception as e:
                logger.warning(f"[FS_QUERY] DB read failed: {e}")
    
    # Fall back to live read (with remote fallback for OneDrive paths)
    if content is None:
        yield sse_token(f"[FS_QUERY] live_read=True\n")
        
        # Call live_read_file_with_remote_fallback (tries local first, then remote)
        content, total_lines, total_bytes, truncated, error, read_source = live_read_file_with_remote_fallback(
            target_path, start_line, end_line, head_lines, debug=True
        )
        
        # Set source from read_source (local, remote_agent, or failed)
        source = read_source
        
        if error:
            yield sse_token(f"❌ {error}\n")
            yield sse_done(
                provider="local",
                model="filesystem_query",
                success=False,
                error=error,
            )
            return
        
        if content is None:
            yield sse_token(f"❌ Could not read file: {target_path}\n")
            yield sse_done(
                provider="local",
                model="filesystem_query",
                success=False,
                error="read_failed",
            )
            return
        
        # Log the source for visibility
        yield sse_token(f"[FS_QUERY] live_read=True source={source}\n")
    
    # Detect language from extension
    if not file_language:
        file_language = get_language_from_extension(target_path)
    
    # Output
    content_lines = content.splitlines() if content else []
    shown_lines = len(content_lines)
    
    yield sse_token(f"\n📄 **{basename}**\n")
    yield sse_token(f"**Path:** {target_path}\n")
    yield sse_token(f"**Total lines:** {total_lines}\n")
    if file_language:
        yield sse_token(f"**Language:** {file_language}\n")
    yield sse_token(f"**Source:** {source}\n")
    
    if start_line and end_line:
        yield sse_token(f"**Showing:** lines {start_line}-{end_line}\n")
    elif head_lines:
        yield sse_token(f"**Showing:** first {head_lines} lines\n")
    
    yield sse_token("\n```" + (file_language or "") + "\n")
    yield sse_token(content if content else "(empty)")
    if content and not content.endswith('\n'):
        yield sse_token("\n")
    yield sse_token("```\n\n")
    
    if truncated:
        yield sse_token(f"⚠️ (truncated) Showing {shown_lines} of {total_lines} lines\n")
    else:
        yield sse_token(f"✅ Showing {shown_lines} lines\n")
    
    # Record in memory
    try:
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id,
                role="assistant",
                content=f"[filesystem_query] {query_type}: {target_path} ({shown_lines}/{total_lines} lines, source={source})",
                provider="local",
                model="filesystem_query",
            ),
        )
    except Exception:
        pass
    
    duration_ms = int(loop.time() * 1000) - started_ms
    
    if trace:
        trace.log_model_call(
            "local_tool", "local", "filesystem_query", query_type,
            0, 0, duration_ms, success=True, error=None,
        )
    
    # Stage 2: Optional commentary (wrapped in try/except to prevent stream crash)
    if _COMMENTARY_AVAILABLE and is_commentary_enabled():
        try:
            yield sse_token("\n---\n**Commentary:**\n")
            content_preview = content[:200] + "..." if content and len(content) > 200 else (content or "")
            tool_result = create_read_result(
                ok=True,
                path=target_path,
                source=source,
                total_lines=total_lines,
                shown_lines=shown_lines,
                truncated=truncated,
                content_preview=content_preview,
                user_message=user_message,
            )
            async for commentary_chunk in render_tool_commentary(tool_result, user_message):
                yield sse_token(commentary_chunk)
            yield sse_token("\n")
        except Exception as e:
            logger.exception("[COMMENTARY] Render failed for read: %s", e)
            # Do not raise - continue silently, deterministic output already complete
    
    yield sse_done(
        provider="local",
        model="filesystem_query",
        total_length=len(content) if content else 0,
        meta={
            "query_type": query_type,
            "target_path": target_path,
            "total_lines": total_lines,
            "shown_lines": shown_lines,
            "truncated": truncated,
            "source": source,
            "db_hit": source == "db",
            "live_read": source in ("local", "remote_agent"),
            "local_read": source == "local",
            "remote_agent_read": source == "remote_agent",
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


def _process_list_results(results, path_prefix):
    """Process DB list results into folders and files."""
    inferred_folders = set()
    direct_files = []
    target_prefix_lower = path_prefix.lower()
    
    for entry in results:
        path_normalized = entry.path.replace('/', '\\')
        path_lower = path_normalized.lower()
        
        if path_lower.startswith(target_prefix_lower):
            relative = path_normalized[len(path_prefix):]
            if '\\' in relative:
                first_segment = relative.split('\\')[0]
                inferred_folders.add(path_prefix + first_segment)
            else:
                direct_files.append(entry)
    
    folders_list = sorted(inferred_folders, key=str.lower)
    return folders_list, direct_files


def _apply_line_limits(content, start_line, end_line, head_lines):
    """Apply line limits to content from DB."""
    lines = content.splitlines()
    total_lines = len(lines)
    truncated = False
    
    if start_line is not None and end_line is not None:
        start_idx = max(0, start_line - 1)
        end_idx = min(total_lines, end_line)
        lines = lines[start_idx:end_idx]
        content = '\n'.join(lines)
    elif head_lines is not None:
        if head_lines < total_lines:
            lines = lines[:head_lines]
            content = '\n'.join(lines)
            truncated = True
    else:
        if total_lines > FILESYSTEM_READ_MAX_LINES:
            lines = lines[:FILESYSTEM_READ_MAX_LINES]
            content = '\n'.join(lines)
            truncated = True
        
        content_bytes = len(content.encode('utf-8', errors='replace'))
        if content_bytes > FILESYSTEM_READ_MAX_BYTES:
            content = content[:FILESYSTEM_READ_MAX_BYTES]
            last_nl = content.rfind('\n')
            if last_nl > FILESYSTEM_READ_MAX_BYTES // 2:
                content = content[:last_nl]
            truncated = True
    
    return content, total_lines, truncated
