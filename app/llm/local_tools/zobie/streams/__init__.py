# FILE: app/llm/local_tools/zobie/streams/__init__.py
"""Stream generators for zobie tools.

Re-exports all stream generator functions.

v1.1 (2026-01-28): Added batch_ops for multi-file operation streaming (Phase 6)
"""

from .scan_sandbox import generate_sandbox_structure_scan_stream
from .update_arch import generate_update_architecture_stream
from .archmap_db import generate_local_architecture_map_stream
from .archmap_full import generate_full_architecture_map_stream
from .zobie_legacy import generate_local_zobie_map_stream
from .fs_query import generate_filesystem_query_stream
from .codebase_report import generate_codebase_report_stream
from .latest_reports import (
    generate_latest_architecture_map_stream,
    generate_latest_codebase_report_full_stream,
)

# v1.1: Multi-file batch operations streaming (Phase 6)
from .batch_ops import (
    BatchProgressMessage,
    BatchProgressStream,
    create_progress_callback,
    create_sync_callback,
    format_completion_summary,
    format_progress_line,
    generate_batch_operation_stream,
)

__all__ = [
    "generate_sandbox_structure_scan_stream",
    "generate_update_architecture_stream",
    "generate_local_architecture_map_stream",
    "generate_full_architecture_map_stream",
    "generate_local_zobie_map_stream",
    "generate_filesystem_query_stream",
    "generate_codebase_report_stream",
    "generate_latest_architecture_map_stream",
    "generate_latest_codebase_report_full_stream",
    # v1.1: Batch operations
    "BatchProgressMessage",
    "BatchProgressStream",
    "create_progress_callback",
    "create_sync_callback",
    "format_completion_summary",
    "format_progress_line",
    "generate_batch_operation_stream",
]
