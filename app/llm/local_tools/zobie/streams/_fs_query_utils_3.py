from __future__ import annotations
import asyncio
import logging
import os
from ..config import FILESYSTEM_QUERY_ALLOWED_ROOTS
from ..db_ops import ARCH_MODELS_AVAILABLE, ArchitectureFileIndex, get_latest_scan
from ..fs_command_parser import parse_filesystem_query
from ..fs_live_ops import live_read_file_with_remote_fallback
from ..fs_path_utils import get_basename, get_language_from_extension, is_path_allowed, looks_like_file, normalize_path
from ..fs_write_ops import WriteResult
from ..sse import sse_done, sse_token
from app.llm.audit_logger import RoutingTrace
from app.llm.local_tools.zobie.streams._fs_query_utils_2 import _apply_line_limits, _handle_append_query, _handle_delete_area_query, _handle_delete_lines_query, _handle_overwrite_query
from app.memory import schemas as memory_schemas
from app.memory import service as memory_service
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import AsyncGenerator, Optional
logger = logging.getLogger(__name__)
_COMMENTARY_AVAILABLE = True
is_commentary_enabled = lambda: False
logger = logging.getLogger(__name__)


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
