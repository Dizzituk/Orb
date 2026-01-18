# FILE: app/llm/local_tools/zobie/streams/fs_query.py
"""FILESYSTEM QUERY stream generator (v5.4 - Remote agent fallback for OneDrive).

Answers filesystem listing/find/read queries using:
1. DB index first (fast, from scan sandbox)
2. Live surgical read fallback (if DB lacks content)
3. Remote agent fallback via sandbox controller (if local read fails for OneDrive)

Supports BOTH natural language AND explicit command mode:
- Natural: "What's in my desktop?", "Show me line 45-65 of router.py"
- Command: "Astra, command: list C:\\path", "command: read \"file with spaces.txt\""

Commands supported:
- list <path>          : List files/folders in path
- find <n> [under <path>] : Search for file/folder by name
- read <path>          : Read entire file (with limits)
- head <path> <n>      : Read first N lines (default 20)
- lines <path> <start> <end> : Read specific line range

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
    
    v5.3: Enhanced debug output for OneDrive path diagnosis.
    
    Flow:
    1. Parse query (command or natural language)
    2. Validate path against allowlist
    3. Try DB lookup first
    4. Fall back to live surgical read if needed
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
    yield sse_token("\n\n")
    
    # Handle missing query type
    if not query_type:
        yield sse_token("❓ Could not determine query type.\n")
        yield sse_token("Supported commands: list, find, read, head, lines\n")
        yield sse_done(
            provider="local",
            model="filesystem_query",
            success=False,
            error="unknown_query_type",
        )
        return
    
    # Handle missing path for path-requiring commands
    if query_type in ("list", "read", "head", "lines") and not target_path:
        yield sse_token("❌ No path specified.\n")
        yield sse_token(f"Usage: {query_type} <path>\n")
        yield sse_done(
            provider="local",
            model="filesystem_query",
            success=False,
            error="no_path_specified",
        )
        return
    
    # Validate path against allowlist
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
    
    # Dispatch by query type
    if query_type == "find":
        async for chunk in _handle_find_query(
            project_id, db, target_path, search_term, trace, started_ms
        ):
            yield chunk
        return
    
    if query_type == "list":
        async for chunk in _handle_list_query(
            project_id, db, target_path, trace, started_ms
        ):
            yield chunk
        return
    
    if query_type in ("read", "head", "lines"):
        async for chunk in _handle_read_query(
            project_id, db, target_path, query_type,
            start_line, end_line, head_lines, trace, started_ms
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
# QUERY HANDLERS
# =============================================================================

async def _handle_find_query(
    project_id: int,
    db: Session,
    target_path: Optional[str],
    search_term: Optional[str],
    trace: Optional[RoutingTrace],
    started_ms: int,
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
) -> AsyncGenerator[str, None]:
    """Handle list queries using DB-first, live fallback."""
    loop = asyncio.get_event_loop()
    
    # Normalize path
    target_path = normalize_path(target_path, debug=True)
    
    # Try DB first
    db_success = False
    folders_list = []
    files_list = []
    
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
                    folders_list, files_list = _process_list_results(
                        results[:FILESYSTEM_QUERY_MAX_ENTRIES], path_prefix
                    )
                    
            except Exception as e:
                logger.warning(f"[FS_QUERY] DB list failed, will try live: {e}")
    
    # Fall back to live read if DB didn't work
    if not db_success:
        yield sse_token(f"[FS_QUERY] live_read=True (DB miss)\n")
        
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
