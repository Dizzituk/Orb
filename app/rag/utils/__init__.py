# Purpose: RAG utilities.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.rag.utils.canonical_paths
# Last-renovated: 2026-06-11
"""RAG utilities."""
from .canonical_paths import (
    canonicalize_path,
    parse_canonical_path,
    canonical_to_absolute,
    get_canonical_directory,
    is_under_canonical_prefix,
    get_path_depth,
)
