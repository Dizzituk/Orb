# Purpose: fs query utils 2
# Called-by: app.llm.local_tools.zobie.streams._fs_query_utils_3, app.llm.local_tools.zobie.streams.fs_query
# Depends-on: app.llm.audit_logger, app.llm.local_tools.zobie.config, app.llm.local_tools.zobie.fs_write_ops, app.llm.local_tools.zobie.sse (+1 more)
# Last-renovated: 2026-06-11
from __future__ import annotations
import asyncio
import logging
from ..config import FILESYSTEM_READ_MAX_BYTES, FILESYSTEM_READ_MAX_LINES
from ..fs_write_ops import sandbox_append_file, sandbox_delete_area, sandbox_delete_lines, sandbox_overwrite_file
from ..sse import sse_done, sse_token
from app.llm.audit_logger import RoutingTrace
from sqlalchemy.orm import Session
from typing import AsyncGenerator, Optional
logger = logging.getLogger(__name__)
_COMMENTARY_AVAILABLE = True
is_commentary_enabled = lambda: False
logger = logging.getLogger(__name__)


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
    from .fs_query import _format_write_result, create_write_result, render_tool_commentary
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
    from .fs_query import _format_write_result, create_write_result, render_tool_commentary
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
    from .fs_query import _format_write_result, create_write_result, render_tool_commentary
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
    from .fs_query import _format_write_result, create_write_result, render_tool_commentary
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
