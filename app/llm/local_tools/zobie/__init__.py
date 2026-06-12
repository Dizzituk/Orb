# FILE: app/llm/local_tools/zobie/__init__.py
# Purpose: Zobie tools subpackage - internal modules for architecture scanning.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.llm.local_tools.zobie.config, app.llm.local_tools.zobie.db_ops, app.llm.local_tools.zobie.filter_utils, app.llm.local_tools.zobie.rag_helpers (+3 more)
# Last-renovated: 2026-06-11
"""Zobie tools subpackage - internal modules for architecture scanning."""

from .config import (
    SANDBOX_CONTROLLER_URL,
    CODE_SCAN_ROOTS,
    SANDBOX_SCAN_ROOTS,
    FULL_ARCHMAP_OUTPUT_DIR,
    is_excluded_path,
    is_excluded_extension,
)
from .sse import sse_token, sse_error, sse_done
from .sandbox_client import call_fs_tree, call_fs_contents
from .db_ops import (
    ARCH_MODELS_AVAILABLE,
    save_scan_to_db,
    save_scan_incremental_to_db,
    save_scan_with_contents_to_db,
)
from .signature_extract import (
    extract_python_signatures,
    extract_js_signatures,
    strip_line_numbers,
    map_kind_to_chunk_type,
)
from .rag_helpers import (
    generate_signatures_json,
    generate_index_for_rag,
    generate_codebase_md,
)
from .filter_utils import filter_scan_results

__all__ = [
    # Config
    "SANDBOX_CONTROLLER_URL",
    "CODE_SCAN_ROOTS",
    "SANDBOX_SCAN_ROOTS",
    "FULL_ARCHMAP_OUTPUT_DIR",
    "is_excluded_path",
    "is_excluded_extension",
    # SSE
    "sse_token",
    "sse_error",
    "sse_done",
    # Sandbox client
    "call_fs_tree",
    "call_fs_contents",
    # DB ops
    "ARCH_MODELS_AVAILABLE",
    "save_scan_to_db",
    "save_scan_incremental_to_db",
    "save_scan_with_contents_to_db",
    # Signatures
    "extract_python_signatures",
    "extract_js_signatures",
    "strip_line_numbers",
    "map_kind_to_chunk_type",
    # RAG helpers
    "generate_signatures_json",
    "generate_index_for_rag",
    "generate_codebase_md",
    # Filter utils
    "filter_scan_results",
]
