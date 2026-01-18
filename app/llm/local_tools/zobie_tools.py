# FILE: app/llm/local_tools/zobie_tools.py
"""Streaming local-tool generators for architecture commands.

Thin orchestrator - implementation in zobie/ subpackage.
v5.0: Refactored to zobie/ subpackage for maintainability.
"""

from app.llm.local_tools.zobie.streams import (
    generate_sandbox_structure_scan_stream,
    generate_update_architecture_stream,
    generate_local_architecture_map_stream,
    generate_full_architecture_map_stream,
    generate_local_zobie_map_stream,
    generate_filesystem_query_stream,
    generate_codebase_report_stream,
)

__all__ = [
    "generate_sandbox_structure_scan_stream",
    "generate_update_architecture_stream",
    "generate_local_architecture_map_stream",
    "generate_full_architecture_map_stream",
    "generate_local_zobie_map_stream",
    "generate_filesystem_query_stream",
    "generate_codebase_report_stream",
]
