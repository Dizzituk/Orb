# Purpose: RAG indexing.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.rag.indexing.directory_indexer, app.rag.indexing.directory_summary
# Last-renovated: 2026-06-11
"""RAG indexing."""
from .directory_indexer import DirectoryIndexBuilder
from .directory_summary import generate_summaries_for_scan, generate_directory_summary
