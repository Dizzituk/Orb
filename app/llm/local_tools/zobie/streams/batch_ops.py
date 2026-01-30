# FILE: app/llm/local_tools/zobie/streams/batch_ops.py
"""
Batch Operations Progress Stream (v1.0 - Phase 6)

SSE streaming for multi-file operations progress updates.
Integrates with Implementer's progress_callback to stream
real-time updates to the frontend.

Version Notes:
-------------
v1.0 (2026-01-28): Initial implementation
    - BatchProgressMessage dataclass for SSE formatting
    - BatchProgressStream class for managing SSE streams
    - create_progress_callback() factory function
    - format_completion_summary() for human-readable output
    - generate_batch_operation_stream() for complete flow
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, Any, Optional, Callable, List

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BatchProgressMessage:
    """A single progress message for batch operations."""
    type: str  # "start", "progress", "complete", "error"
    current: int = 0
    total: int = 0
    file: str = ""
    status: str = ""
    replacements: int = 0
    message: str = ""
    
    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        data = {
            "type": self.type,
            "current": self.current,
            "total": self.total,
            "file": self.file,
            "status": self.status,
            "replacements": self.replacements,
            "message": self.message,
        }
        return f"data: {json.dumps(data)}\n\n"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "current": self.current,
            "total": self.total,
            "file": self.file,
            "status": self.status,
            "replacements": self.replacements,
            "message": self.message,
        }


# =============================================================================
# BATCH PROGRESS STREAM
# =============================================================================

class BatchProgressStream:
    """
    v1.0: Manages SSE streaming for batch operations.
    
    Usage:
        stream = BatchProgressStream(operation="refactor", total_files=47)
        
        # Get callback for Implementer
        callback = stream.get_progress_callback()
        
        # Generate SSE events
        async for event in stream.generate():
            yield event
        
        # From Implementer (via callback):
        callback({"type": "progress", "file": "app/main.py", ...})
    """
    
    def __init__(self, operation: str, total_files: int):
        """
        Initialize BatchProgressStream.
        
        Args:
            operation: "search" or "refactor"
            total_files: Expected number of files to process
        """
        self.operation = operation
        self.total_files = total_files
        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.completed = False
        self._started = False
        
        logger.info(
            "[batch_ops] v1.0 Stream created: operation=%s, total_files=%d",
            operation, total_files
        )
    
    def get_progress_callback(self) -> Callable[[Dict[str, Any]], None]:
        """
        Get a synchronous callback function for Implementer.
        
        The callback can be called from sync or async code.
        """
        def callback(data: Dict[str, Any]) -> None:
            try:
                self.queue.put_nowait(data)
                
                if data.get("type") == "complete":
                    self.completed = True
            except Exception as e:
                logger.error("[batch_ops] v1.0 Callback error: %s", e)
        
        return callback
    
    async def on_progress(self, data: Dict[str, Any]) -> None:
        """
        Async method to report progress.
        
        This can be used directly as an async callback.
        """
        await self.queue.put(data)
        
        if data.get("type") == "complete":
            self.completed = True
    
    async def generate(self, timeout: float = 30.0) -> AsyncGenerator[str, None]:
        """
        Generate SSE events from progress queue.
        
        Args:
            timeout: Seconds to wait between events before sending keepalive
            
        Yields:
            SSE-formatted strings
        """
        # Start message
        if not self._started:
            self._started = True
            yield BatchProgressMessage(
                type="start",
                total=self.total_files,
                message=f"Starting {self.operation} operation on {self.total_files} files",
            ).to_sse()
        
        while not self.completed:
            try:
                # Wait for next progress update with timeout
                data = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                
                msg = BatchProgressMessage(
                    type=data.get("type", "progress"),
                    current=data.get("current", 0),
                    total=data.get("total", self.total_files),
                    file=data.get("file", ""),
                    status=data.get("status", ""),
                    replacements=data.get("replacements", 0),
                    message=data.get("message", ""),
                )
                
                yield msg.to_sse()
                
                if data.get("type") == "complete":
                    break
                    
            except asyncio.TimeoutError:
                # Send keepalive
                yield ": keepalive\n\n"
        
        logger.info("[batch_ops] v1.0 Stream complete for %s operation", self.operation)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

async def create_progress_callback(
    stream: BatchProgressStream,
) -> Callable[[Dict[str, Any]], None]:
    """
    Create a progress callback that feeds the SSE stream.
    
    Returns a callback function suitable for Implementer.
    
    Args:
        stream: BatchProgressStream instance
        
    Returns:
        Callback function
    """
    # Return sync callback for compatibility
    return stream.get_progress_callback()


def create_sync_callback(
    stream: BatchProgressStream,
) -> Callable[[Dict[str, Any]], None]:
    """
    Create a synchronous progress callback.
    
    Args:
        stream: BatchProgressStream instance
        
    Returns:
        Synchronous callback function
    """
    return stream.get_progress_callback()


# =============================================================================
# FORMATTING FUNCTIONS
# =============================================================================

def format_completion_summary(result: Dict[str, Any]) -> str:
    """
    Format a human-readable summary of batch operation results.
    
    Args:
        result: The result dict from Implementer (MultiFileResult.to_dict())
        
    Returns:
        Formatted summary string
    """
    operation = result.get("operation", "unknown")
    
    if operation == "search":
        return (
            f"📊 **Search Complete**\n"
            f"   Pattern: `{result.get('search_pattern', '')}`\n"
            f"   Files: {result.get('total_files', 0)}\n"
            f"   Occurrences: {result.get('total_occurrences', result.get('total_replacements', 0))}"
        )
    
    elif operation == "refactor":
        success = result.get("success", False)
        icon = "✅" if success else "❌"
        
        lines = [
            f"{icon} **Refactor {'Complete' if success else 'Failed'}**",
            f"   Pattern: `{result.get('search_pattern', '')}` → `{result.get('replacement_pattern', '')}`",
            f"   Files modified: {result.get('files_modified', 0)}",
            f"   Files unchanged: {result.get('files_unchanged', 0)}",
            f"   Files failed: {result.get('files_failed', 0)}",
            f"   Total replacements: {result.get('total_replacements', 0)}",
        ]
        
        errors = result.get("errors", [])
        if errors:
            lines.append(f"   Errors: {len(errors)}")
            for err in errors[:5]:  # Show first 5 errors
                lines.append(f"      - {err}")
            if len(errors) > 5:
                lines.append(f"      ... and {len(errors) - 5} more")
        
        return "\n".join(lines)
    
    return f"Operation complete: {json.dumps(result, indent=2)}"


def format_progress_line(data: Dict[str, Any]) -> str:
    """
    Format a single progress update as a human-readable line.
    
    Args:
        data: Progress data dict
        
    Returns:
        Formatted progress line
    """
    current = data.get("current", 0)
    total = data.get("total", 0)
    file_path = data.get("file", "")
    status = data.get("status", "")
    replacements = data.get("replacements", 0)
    
    # Shorten path for display
    if len(file_path) > 50:
        file_path = "..." + file_path[-47:]
    
    # Status emoji
    status_emoji = {
        "processing": "⏳",
        "success": "✅",
        "unchanged": "➖",
        "error": "❌",
        "verify_failed": "⚠️",
    }.get(status, "•")
    
    # Build line
    if status == "success":
        return f"[{current}/{total}] {status_emoji} {file_path} ({replacements} replacements)"
    elif status == "unchanged":
        return f"[{current}/{total}] {status_emoji} {file_path} (no matches)"
    elif status == "processing":
        return f"[{current}/{total}] {status_emoji} {file_path}..."
    else:
        return f"[{current}/{total}] {status_emoji} {file_path}"


# =============================================================================
# STREAM GENERATOR
# =============================================================================

async def generate_batch_operation_stream(
    multi_file: Dict[str, Any],
    client: Any = None,
) -> AsyncGenerator[str, None]:
    """
    Generate SSE stream for a complete multi-file operation.
    
    This is the main entry point for streaming batch operations.
    Handles both search and refactor operations.
    
    Args:
        multi_file: Dict with multi_file data from spec
        client: Optional sandbox client
        
    Yields:
        SSE-formatted strings
    """
    from app.overwatcher.implementer import run_multi_file_operation
    
    operation_type = multi_file.get("operation_type", "search")
    total_files = multi_file.get("total_files", len(multi_file.get("target_files", [])))
    
    logger.info(
        "[batch_ops] v1.0 Starting batch stream: operation=%s, files=%d",
        operation_type, total_files
    )
    
    # Create stream and callback
    stream = BatchProgressStream(operation=operation_type, total_files=total_files)
    callback = stream.get_progress_callback()
    
    # Start operation in background task
    async def run_operation():
        try:
            result = await run_multi_file_operation(
                multi_file=multi_file,
                client=client,
                progress_callback=callback,
            )
            logger.info("[batch_ops] v1.0 Operation complete: success=%s", result.success)
            return result
        except Exception as e:
            logger.error("[batch_ops] v1.0 Operation error: %s", e)
            callback({
                "type": "error",
                "message": str(e),
            })
            callback({
                "type": "complete",
                "success": False,
                "error": str(e),
            })
            return None
    
    # Start operation
    task = asyncio.create_task(run_operation())
    
    # Yield progress events
    try:
        async for event in stream.generate():
            yield event
    except Exception as e:
        logger.error("[batch_ops] v1.0 Stream error: %s", e)
        yield BatchProgressMessage(
            type="error",
            message=str(e),
        ).to_sse()
    
    # Wait for operation to complete
    try:
        result = await task
        if result:
            # Yield summary
            summary = format_completion_summary(result.to_dict())
            yield f"data: {json.dumps({'type': 'summary', 'content': summary})}\n\n"
    except Exception as e:
        logger.error("[batch_ops] v1.0 Final error: %s", e)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "BatchProgressMessage",
    "BatchProgressStream",
    "create_progress_callback",
    "create_sync_callback",
    "format_completion_summary",
    "format_progress_line",
    "generate_batch_operation_stream",
]
