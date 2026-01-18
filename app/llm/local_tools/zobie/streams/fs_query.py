# FILE: app/llm/local_tools/zobie/streams/fs_query.py
"""FILESYSTEM QUERY stream generator (v4.7).

Answers filesystem listing/find/read queries using scan DB index.
NEVER runs shell commands. Only reads from architecture_file_index.

Extracted from zobie_tools.py for modularity.
No logic changes - exact same behavior and SSE output format.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import AsyncGenerator, Dict, List, Optional

from sqlalchemy.orm import Session

from app.llm.audit_logger import RoutingTrace
from app.memory import schemas as memory_schemas
from app.memory import service as memory_service

from ..config import (
    FILESYSTEM_QUERY_ALLOWED_ROOTS,
    FILESYSTEM_QUERY_MAX_ENTRIES,
    FILESYSTEM_READ_MAX_LINES,
    FILESYSTEM_READ_MAX_BYTES,
    KNOWN_FOLDER_PATHS,
)
from ..sse import sse_token, sse_error, sse_done
from ..db_ops import (
    ARCH_MODELS_AVAILABLE,
    ArchitectureFileIndex,
    get_latest_scan,
)

logger = logging.getLogger(__name__)


def _normalize_path(path: str) -> str:
    """
    Normalize a Windows path: strip quotes, trim whitespace, convert / to \\.
    
    v4.9: Added for read file support.
    """
    if not path:
        return path
    # Strip quotes and whitespace
    path = path.strip().strip('"').strip("'").strip()
    # Convert forward slashes to backslashes
    path = path.replace('/', '\\')
    # Remove trailing backslash (unless it's a root like D:\)
    if len(path) > 3 and path.endswith('\\'):
        path = path.rstrip('\\')
    return path


def _parse_filesystem_query(message: str) -> dict:
    """
    Parse a filesystem query to extract:
    - query_type: "list", "find", or "read"
    - find_type: "folder", "file", or "any" (for find queries)
    - target_path: The directory/file path to list/search/read
    - search_term: For find queries, what to search for
    - include_full_paths: Whether to include full paths (default True)
    
    v4.9: Added "read" query_type for reading file contents from DB.
    
    Returns dict with parsed info or None if invalid.
    """
    text = message.strip()
    
    # Strip "After scan sandbox, " prefix if present
    text = re.sub(r'^[Aa]fter\s+scan\s+sandbox,?\s*', '', text).strip()
    text_lower = text.lower()
    
    result = {
        "query_type": None,
        "find_type": "any",  # v4.8: "folder", "file", or "any"
        "target_path": None,
        "search_term": None,
        "include_full_paths": "full path" in text_lower,
    }
    
    # Try to extract Windows path
    path_match = re.search(r'([A-Za-z]:[/\\][^"\',;:?!]*)', text)
    if path_match:
        result["target_path"] = _normalize_path(path_match.group(1))
    else:
        # Try to extract known folder keyword
        for folder, path in KNOWN_FOLDER_PATHS.items():
            if folder in text_lower:
                result["target_path"] = path
                break
    
    # v4.9: Detect READ queries first (more specific patterns)
    # Patterns: "what's written in X", "read X", "show contents of X", 
    #           "open X", "cat X", "what does X contain", "display X"
    read_patterns = [
        r"what'?s\s+(?:written|inside|in)\s+",  # "what's written in", "whats inside"
        r"read\s+(?:the\s+)?(?:file\s+)?",      # "read", "read file", "read the file"
        r"show\s+(?:the\s+)?contents?\s+of\s+", # "show contents of", "show content of"
        r"(?:display|view|print|output)\s+(?:the\s+)?(?:file\s+)?",  # "display", "view file"
        r"cat\s+",                               # "cat X" (unix style)
        r"what\s+does\s+.+\s+(?:say|contain)",  # "what does X say/contain"
        r"open\s+(?:the\s+)?(?:file\s+)?",      # "open", "open file" (when path looks like file)
    ]
    
    for pattern in read_patterns:
        if re.search(pattern, text_lower):
            # Additional check: target_path should look like a file (has extension or no trailing \)
            if result["target_path"]:
                target = result["target_path"]
                # If path has an extension, it's likely a file read request
                if '.' in os.path.basename(target) or not target.endswith('\\'):
                    result["query_type"] = "read"
                    return result
    
    # Determine query type for list/find
    if re.match(r'^(?:list|show|what|contents)', text_lower):
        result["query_type"] = "list"
    elif re.match(r'^find', text_lower):
        result["query_type"] = "find"
        
        # v4.8: Detect if searching for folder specifically
        # "Find folder named Jobs" / "Find folders named X" / "Find directory called X"
        if re.search(r'find\s+(?:folder|folders|directory|directories)\s+(?:named?|called)', text_lower):
            result["find_type"] = "folder"
        elif re.search(r'find\s+(?:file|files)\s+(?:named?|called|with)', text_lower):
            result["find_type"] = "file"
        
        # Extract search term for find queries
        # "Find folder named Jobs under ..." -> "Jobs"
        # "Find MBS Fitness under OneDrive" -> "MBS Fitness"
        # "Find files with Amber in the name" -> "Amber"
        
        named_match = re.search(r'(?:named?|called)\s+["\']?([\w\s-]+)["\']?(?:\s+(?:under|in|on|inside)|$)', text, re.IGNORECASE)
        if named_match:
            result["search_term"] = named_match.group(1).strip()
        else:
            # "Find <term> under <path>"
            find_match = re.search(r'^find\s+(?:folder|file|directory)?\s*([\w\s-]+?)\s+(?:under|in|on|inside)', text, re.IGNORECASE)
            if find_match:
                term = find_match.group(1).strip()
                # Skip generic words
                if term.lower() not in {"folder", "file", "directory", "all", "everything", "files"}:
                    result["search_term"] = term
            else:
                # "Find files with X in the name"
                with_match = re.search(r'with\s+([\w\s-]+?)\s+(?:in\s+(?:the\s+)?name)', text, re.IGNORECASE)
                if with_match:
                    result["search_term"] = with_match.group(1).strip()
    
    return result


def _is_path_within_allowed_roots(path: str) -> bool:
    """Check if a path is within allowed scan roots."""
    path_lower = path.lower().replace('/', '\\')
    
    # D:\ is always allowed
    if path_lower.startswith('d:\\'):
        return True
    
    # C:\Users\dizzi is allowed
    if path_lower.startswith('c:\\users\\dizzi'):
        return True
    
    return False


async def generate_filesystem_query_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
) -> AsyncGenerator[str, None]:
    """
    Answer filesystem listing/find queries using the scan DB index.
    
    v4.7: New handler for FILESYSTEM_QUERY intent.
    
    Safety:
    - ONLY reads from architecture_file_index table (from scan sandbox)
    - NEVER runs shell commands or mentions running dir/grep
    - Hard cap of 200 entries
    - Only allows paths under D:\\ or C:\\Users\\dizzi
    
    Output format:
    - Folders first, then files
    - Full paths included
    - +N more summary if truncated
    """
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    
    yield sse_token("📂 [FILESYSTEM_QUERY] Processing query...\n")
    
    # Check if architecture models available
    if not ARCH_MODELS_AVAILABLE:
        yield sse_token("⚠️ Architecture database not available.\n")
        yield sse_token("Run `scan sandbox` first to index the filesystem.\n")
        yield sse_done(
            provider="local",
            model="filesystem_query",
            success=False,
            error="models_not_available",
        )
        return
    
    # Parse the query
    parsed = _parse_filesystem_query(message)
    query_type = parsed.get("query_type", "list")
    target_path = parsed.get("target_path")
    search_term = parsed.get("search_term")
    
    yield sse_token(f"   Query type: {query_type}\n")
    if target_path:
        yield sse_token(f"   Target path: {target_path}\n")
    if search_term:
        yield sse_token(f"   Search term: {search_term}\n")
    yield sse_token("\n")
    
    # Validate path is within allowed roots
    if target_path and not _is_path_within_allowed_roots(target_path):
        yield sse_token(f"❌ Path `{target_path}` is outside allowed scan roots.\n")
        yield sse_token(f"Allowed roots: D:\\ and C:\\Users\\dizzi\n")
        yield sse_done(
            provider="local",
            model="filesystem_query",
            success=False,
            error="path_outside_allowed_roots",
        )
        return
    
    # Get latest scan
    latest_scan = get_latest_scan(db, scope="sandbox") if get_latest_scan else None
    
    if not latest_scan:
        yield sse_token("⚠️ No scan data found in database.\n")
        yield sse_token("Run `scan sandbox` first to index the filesystem.\n")
        yield sse_done(
            provider="local",
            model="filesystem_query",
            success=False,
            error="no_scan_data",
        )
        return
    
    yield sse_token(f"📊 Using scan data (scan_id={latest_scan.id}, from {latest_scan.finished_at})\n\n")
    
    # =========================================================================
    # v4.9: READ handler - fetch file content from DB
    # =========================================================================
    if query_type == "read":
        if not target_path:
            yield sse_token("❌ No file path specified for read operation.\n")
            yield sse_done(
                provider="local",
                model="filesystem_query",
                success=False,
                error="no_path_specified",
            )
            return
        
        # Normalize the path
        target_path = _normalize_path(target_path)
        
        # Check if path looks like a folder (no extension, or ends with \)
        basename = os.path.basename(target_path)
        if not basename or '.' not in basename:
            yield sse_token(f"📁 `{target_path}` looks like a folder path.\n")
            yield sse_token("💡 Use `list {path}` to see folder contents instead.\n")
            yield sse_done(
                provider="local",
                model="filesystem_query",
                success=False,
                error="path_is_folder",
            )
            return
        
        yield sse_token(f"📖 Reading file: {target_path}\n\n")
        
        # Try exact path match first
        file_entry = None
        try:
            file_entry = db.query(ArchitectureFileIndex).filter(
                ArchitectureFileIndex.scan_id == latest_scan.id,
                ArchitectureFileIndex.path == target_path
            ).first()
            
            # v4.9: Case-insensitive fallback if exact match fails (Windows is case-insensitive)
            if not file_entry:
                # Use func.lower for case-insensitive comparison
                from sqlalchemy import func
                file_entry = db.query(ArchitectureFileIndex).filter(
                    ArchitectureFileIndex.scan_id == latest_scan.id,
                    func.lower(ArchitectureFileIndex.path) == target_path.lower()
                ).first()
                
                if file_entry:
                    yield sse_token(f"   (matched case-insensitively: {file_entry.path})\n")
        except Exception as e:
            logger.exception(f"[FILESYSTEM_QUERY] Read query failed: {e}")
            yield sse_error(f"Database query failed: {e}")
            yield sse_done(
                provider="local",
                model="filesystem_query",
                success=False,
                error=str(e),
            )
            return
        
        if not file_entry:
            yield sse_token(f"📭 File not found in scan index: `{target_path}`\n\n")
            yield sse_token("Possible reasons:\n")
            yield sse_token("  • File path may be incorrect\n")
            yield sse_token("  • File may have been excluded from scan (binary, archive, etc.)\n")
            yield sse_token("  • File may have been added after the last scan\n\n")
            yield sse_token("💡 Run `scan sandbox` to refresh the index.\n")
            yield sse_done(
                provider="local",
                model="filesystem_query",
                success=False,
                error="file_not_found",
                meta={"scan_id": latest_scan.id, "target_path": target_path},
            )
            return
        
        # File found - check if content is available
        if not file_entry.content:
            yield sse_token(f"📄 File exists in scan index but contents were not captured.\n\n")
            yield sse_token(f"**Path:** {file_entry.path}\n")
            if file_entry.size_bytes:
                size_kb = file_entry.size_bytes / 1024
                yield sse_token(f"**Size:** {size_kb:.1f} KB\n")
            yield sse_token(f"**Extension:** {file_entry.ext or '(none)'}\n\n")
            yield sse_token("This happens for files that:\n")
            yield sse_token("  • Were added after the last content scan\n")
            yield sse_token("  • Exceeded size limits (>500KB)\n")
            yield sse_token("  • Had unsupported extensions\n\n")
            yield sse_token("💡 Run `scan sandbox` to capture contents.\n")
            yield sse_done(
                provider="local",
                model="filesystem_query",
                success=True,  # Not an error - file exists, just no content
                meta={
                    "scan_id": latest_scan.id,
                    "target_path": target_path,
                    "file_exists": True,
                    "content_available": False,
                },
            )
            return
        
        # Content available - read and display with limits
        content_text = file_entry.content.content_text
        
        # Check if content is readable as text (shouldn't have binary chars)
        try:
            # Quick binary check: look for null bytes or high ratio of non-printable chars
            if '\x00' in content_text:
                yield sse_token(f"📄 File appears to be binary and cannot be displayed as text.\n")
                yield sse_token(f"**Path:** {file_entry.path}\n")
                if file_entry.size_bytes:
                    yield sse_token(f"**Size:** {file_entry.size_bytes / 1024:.1f} KB\n")
                yield sse_done(
                    provider="local",
                    model="filesystem_query",
                    success=True,
                    meta={"scan_id": latest_scan.id, "binary": True},
                )
                return
        except Exception:
            pass  # If check fails, continue anyway
        
        # Apply preview limits
        lines = content_text.splitlines()
        total_lines = len(lines)
        total_bytes = len(content_text.encode('utf-8', errors='replace'))
        
        truncated = False
        truncation_reason = ""
        
        # Check line limit
        if total_lines > FILESYSTEM_READ_MAX_LINES:
            lines = lines[:FILESYSTEM_READ_MAX_LINES]
            truncated = True
            truncation_reason = f"line limit ({FILESYSTEM_READ_MAX_LINES} lines)"
        
        # Check byte limit on the truncated content
        preview_text = '\n'.join(lines)
        preview_bytes = len(preview_text.encode('utf-8', errors='replace'))
        
        if preview_bytes > FILESYSTEM_READ_MAX_BYTES:
            # Truncate by bytes
            preview_text = preview_text[:FILESYSTEM_READ_MAX_BYTES]
            # Find last newline to avoid cutting mid-line
            last_nl = preview_text.rfind('\n')
            if last_nl > FILESYSTEM_READ_MAX_BYTES // 2:
                preview_text = preview_text[:last_nl]
            truncated = True
            truncation_reason = f"size limit (~{FILESYSTEM_READ_MAX_BYTES // 1024}KB)"
        
        # Output file info
        size_str = ""
        if file_entry.size_bytes:
            if file_entry.size_bytes > 1_000_000:
                size_str = f"{file_entry.size_bytes / 1_000_000:.1f} MB"
            elif file_entry.size_bytes > 1_000:
                size_str = f"{file_entry.size_bytes / 1_000:.1f} KB"
            else:
                size_str = f"{file_entry.size_bytes} bytes"
        
        yield sse_token(f"📄 **{file_entry.name}**\n")
        yield sse_token(f"**Path:** {file_entry.path}\n")
        if size_str:
            yield sse_token(f"**Size:** {size_str}\n")
        yield sse_token(f"**Lines:** {total_lines}\n")
        if file_entry.language:
            yield sse_token(f"**Language:** {file_entry.language}\n")
        yield sse_token("\n")
        
        # Output content
        yield sse_token("--- Content ---\n")
        yield sse_token(preview_text)
        if not preview_text.endswith('\n'):
            yield sse_token("\n")
        yield sse_token("---------------\n\n")
        
        if truncated:
            yield sse_token(f"⚠️ Showing first {len(preview_text.splitlines())} of {total_lines} lines (truncated due to {truncation_reason})\n")
        else:
            yield sse_token(f"✅ Showing full file ({total_lines} lines)\n")
        
        # Record in memory
        try:
            memory_service.create_message(
                db,
                memory_schemas.MessageCreate(
                    project_id=project_id,
                    role="assistant",
                    content=f"[filesystem_query] read: {file_entry.path} ({total_lines} lines)",
                    provider="local",
                    model="filesystem_query",
                ),
            )
        except Exception:
            pass
        
        duration_ms = int(loop.time() * 1000) - started_ms
        
        if trace:
            trace.log_model_call(
                "local_tool", "local", "filesystem_query", "filesystem_query",
                0, 0, duration_ms, success=True, error=None,
            )
        
        yield sse_done(
            provider="local",
            model="filesystem_query",
            total_length=len(preview_text),
            meta={
                "scan_id": latest_scan.id,
                "query_type": "read",
                "target_path": target_path,
                "file_path": file_entry.path,
                "total_lines": total_lines,
                "total_bytes": total_bytes,
                "truncated": truncated,
            },
        )
        return  # Early exit for read queries
    
    # Build query against architecture_file_index
    try:
        query = db.query(ArchitectureFileIndex).filter(
            ArchitectureFileIndex.scan_id == latest_scan.id
        )
        
        # Filter by target path if specified
        if target_path:
            # Normalize path for LIKE query
            path_prefix = target_path.replace('/', '\\').rstrip('\\') + '\\'
            query = query.filter(
                ArchitectureFileIndex.path.like(f"{path_prefix}%")
            )
        
        # Filter by search term if specified (find queries)
        if search_term and query_type == "find":
            search_pattern = f"%{search_term}%"
            query = query.filter(
                ArchitectureFileIndex.name.ilike(search_pattern)
            )
        
        # Get results with limit + 1 to detect truncation
        max_results = FILESYSTEM_QUERY_MAX_ENTRIES + 1
        results = query.limit(max_results).all()
        
        truncated = len(results) > FILESYSTEM_QUERY_MAX_ENTRIES
        if truncated:
            results = results[:FILESYSTEM_QUERY_MAX_ENTRIES]
        
    except Exception as e:
        logger.exception(f"[FILESYSTEM_QUERY] DB query failed: {e}")
        yield sse_error(f"Database query failed: {e}")
        yield sse_done(
            provider="local",
            model="filesystem_query",
            success=False,
            error=str(e),
        )
        return
    
    if not results:
        if target_path:
            yield sse_token(f"📭 No entries found under `{target_path}`\n")
        elif search_term:
            yield sse_token(f"📭 No entries found matching '{search_term}'\n")
        else:
            yield sse_token("📭 No entries found in scan data.\n")
        yield sse_done(
            provider="local",
            model="filesystem_query",
            total_length=0,
            meta={"results": 0, "query_type": query_type},
        )
        return
    
    # =========================================================================
    # v4.8: Format results using path-based inference
    # DB only stores files, so we infer folders from file paths
    # =========================================================================
    
    find_type = parsed.get("find_type", "any")
    
    # For "find folder" queries: search folder segments in paths
    if query_type == "find" and find_type == "folder" and search_term:
        # Infer folders from paths that contain a segment matching search_term
        inferred_folders: set = set()
        search_lower = search_term.lower()
        
        for entry in results:
            path_normalized = entry.path.replace('/', '\\')
            # Split path into segments and check each
            segments = path_normalized.split('\\')
            for i, seg in enumerate(segments[:-1]):  # Exclude filename
                if seg.lower() == search_lower:
                    # Build full path to this folder
                    folder_path = '\\'.join(segments[:i+1])
                    inferred_folders.add(folder_path)
        
        # Output folder search results
        folders_list = sorted(inferred_folders, key=str.lower)
        total_count = len(folders_list)
        
        yield sse_token(f"🔍 **Folder search results for '{search_term}'**\n\n")
        
        if folders_list:
            yield sse_token(f"**Folders ({len(folders_list)}):**\n")
            for folder_path in folders_list:
                yield sse_token(f"📁 {folder_path}\n")
            yield sse_token("\n")
        else:
            yield sse_token(f"📭 No folders named '{search_term}' found.\n")
        
        # Summary and done (skip normal output path)
        yield sse_token(f"---\n")
        yield sse_token(f"**Total:** {len(folders_list)} folders\n")
        
        # v4.8: Add note about .zip exclusion if relevant
        if search_lower.endswith('.zip') or 'zip' in search_lower:
            yield sse_token("\n📝 Note: .zip files are excluded from scan metadata (archives not indexed).\n")
        
        # Record in memory
        try:
            memory_service.create_message(
                db,
                memory_schemas.MessageCreate(
                    project_id=project_id,
                    role="assistant",
                    content=f"[filesystem_query] find folder: {len(folders_list)} results for '{search_term}'",
                    provider="local",
                    model="filesystem_query",
                ),
            )
        except Exception:
            pass
        
        duration_ms = int(loop.time() * 1000) - started_ms
        
        if trace:
            trace.log_model_call(
                "local_tool", "local", "filesystem_query", "filesystem_query",
                0, 0, duration_ms, success=True, error=None,
            )
        
        yield sse_done(
            provider="local",
            model="filesystem_query",
            total_length=total_count,
            meta={
                "scan_id": latest_scan.id,
                "query_type": query_type,
                "find_type": find_type,
                "target_path": target_path,
                "search_term": search_term,
                "folders": len(folders_list),
                "files": 0,
                "truncated": False,
            },
        )
        return  # Early exit for folder search
    
    # For list queries with target_path: infer immediate children
    # Folders are inferred from first-level subdirectories in file paths
    # Files are only those directly in target_path (not nested)
    inferred_folders: set = set()
    top_level_files: list = []
    
    if target_path and query_type == "list":
        target_prefix = target_path.rstrip('\\') + '\\'
        target_prefix_lower = target_prefix.lower()
        
        for entry in results:
            path_normalized = entry.path.replace('/', '\\')
            path_lower = path_normalized.lower()
            
            # Get relative path from target
            if path_lower.startswith(target_prefix_lower):
                relative = path_normalized[len(target_prefix):]
                
                if '\\' in relative:
                    # Has subdirectory - extract first folder segment
                    first_segment = relative.split('\\')[0]
                    folder_full_path = target_prefix + first_segment
                    inferred_folders.add(folder_full_path)
                else:
                    # Direct child file (no subdirectory)
                    top_level_files.append(entry)
        
        # Convert to sorted lists
        folders_list = sorted(inferred_folders, key=str.lower)
        files_list = sorted(top_level_files, key=lambda x: x.name.lower())
    else:
        # Fallback for find queries or no target_path: use old behavior
        # (search by filename, return files that match)
        folders_list = []
        files_list = sorted(results, key=lambda x: x.name.lower())
    
    total_count = len(folders_list) + len(files_list)
    
    if query_type == "list":
        yield sse_token(f"📁 **Contents of {target_path or 'indexed filesystem'}**\n\n")
    else:
        yield sse_token(f"🔍 **Search results for '{search_term}'**\n\n")
    
    if folders_list:
        yield sse_token(f"**Folders ({len(folders_list)}):**\n")
        for folder_path in folders_list:
            # folder_path is a string (inferred path), not an entry object
            yield sse_token(f"📁 {folder_path}\n")
        yield sse_token("\n")
    
    if files_list:
        yield sse_token(f"**Files ({len(files_list)}):**\n")
        for file in files_list:
            path_display = file.path.replace('/', '\\')
            size_str = ""
            if file.size_bytes:
                if file.size_bytes > 1_000_000:
                    size_str = f" ({file.size_bytes / 1_000_000:.1f} MB)"
                elif file.size_bytes > 1_000:
                    size_str = f" ({file.size_bytes / 1_000:.1f} KB)"
            yield sse_token(f"📄 {path_display}{size_str}\n")
        yield sse_token("\n")
    
    # Summary
    yield sse_token(f"---\n")
    yield sse_token(f"**Total:** {len(folders_list)} folders, {len(files_list)} files\n")
    
    # v4.8: Add note about .zip exclusion if listing and no zips found
    if query_type == "list" and target_path:
        # Check if user might expect to see .zip files
        target_lower = target_path.lower()
        if 'desktop' in target_lower or 'downloads' in target_lower or 'documents' in target_lower:
            yield sse_token("\n📝 Note: .zip files are excluded from scan metadata (archives not indexed).\n")
    
    if truncated:
        # Get total count from DB for accurate "more" count
        try:
            total_in_db = query.count() if 'query' in dir() else total_count
            remaining = total_in_db - FILESYSTEM_QUERY_MAX_ENTRIES
            yield sse_token(f"⚠️ Results truncated. +{remaining} more entries (showing first {FILESYSTEM_QUERY_MAX_ENTRIES})\n")
        except:
            yield sse_token(f"⚠️ Results truncated to {FILESYSTEM_QUERY_MAX_ENTRIES} entries\n")
    
    # Record in memory
    try:
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id,
                role="assistant",
                content=f"[filesystem_query] {query_type}: {total_count} results for '{target_path or search_term}'",
                provider="local",
                model="filesystem_query",
            ),
        )
    except Exception:
        pass
    
    duration_ms = int(loop.time() * 1000) - started_ms
    
    if trace:
        trace.log_model_call(
            "local_tool", "local", "filesystem_query", "filesystem_query",
            0, 0, duration_ms, success=True, error=None,
        )
    
    yield sse_done(
        provider="local",
        model="filesystem_query",
        total_length=total_count,
        meta={
            "scan_id": latest_scan.id,
            "query_type": query_type,
            "target_path": target_path,
            "search_term": search_term,
            "folders": len(folders_list),
            "files": len(files_list),
            "truncated": truncated,
        },
    )
