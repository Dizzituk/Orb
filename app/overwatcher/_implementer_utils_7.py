from __future__ import annotations
import logging
from app.overwatcher.sandbox_client import SandboxClient
from typing import Any, Callable, Dict, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


async def run_multi_file_search(
    *,
    multi_file: Dict[str, Any],
    client: Optional[SandboxClient] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> MultiFileResult:
    """
    v1.11: Execute multi-file search (read-only).
    
    For search operations, the discovery results are already in the spec.
    This method formats them for display and returns the summary.
    
    Args:
        multi_file: Dict with multi_file data from spec
        client: Sandbox client (optional, for verification)
        progress_callback: Optional callback for progress updates
        
    Returns:
        MultiFileResult with search results summary
    """
    from .implementer import MultiFileResult
    import time
    start_time = time.time()
    
    if not multi_file.get("is_multi_file"):
        return MultiFileResult(
            success=False,
            operation="search",
            error="Not a multi-file operation",
            duration_ms=int((time.time() - start_time) * 1000),
        )
    
    logger.info(
        "[implementer] v1.11 Multi-file SEARCH: pattern='%s', files=%d, occurrences=%d",
        multi_file.get("search_pattern", ""),
        multi_file.get("total_files", 0),
        multi_file.get("total_occurrences", 0),
    )
    
    # For search, results are already computed by SpecGate discovery
    # Just format and return
    result = MultiFileResult(
        success=True,
        operation="search",
        search_pattern=multi_file.get("search_pattern", ""),
        total_files=multi_file.get("total_files", 0),
        total_occurrences=multi_file.get("total_occurrences", 0),
        file_preview=multi_file.get("file_preview", ""),
        target_files=multi_file.get("target_files", []),
        files_processed=multi_file.get("total_files", 0),
        files_modified=0,  # Search is read-only
        files_failed=0,
        duration_ms=int((time.time() - start_time) * 1000),
    )
    
    # Send completion callback
    if progress_callback:
        try:
            callback_data = {
                "type": "complete",
                "operation": "search",
                "total_files": result.total_files,
                "total_occurrences": result.total_replacements,
                "success": True,
            }
            # Handle both sync and async callbacks
            import asyncio
            if asyncio.iscoroutinefunction(progress_callback):
                await progress_callback(callback_data)
            else:
                progress_callback(callback_data)
        except Exception as e:
            logger.warning("[implementer] v1.11 Progress callback error: %s", e)
    
    return result
