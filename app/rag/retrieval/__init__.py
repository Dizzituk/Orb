# Purpose: RAG retrieval.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.rag.retrieval.arch_search, app.rag.retrieval.context_assembler
# Last-renovated: 2026-06-11
"""RAG retrieval."""
from .arch_search import (
    ArchitectureSearch,
    ArchSearchResult,
    ArchSearchResponse,
    search_architecture,
)
from .context_assembler import (
    ContextAssembler,
    AssembledContext,
    assemble_context,
    retrieve_architecture_context,
)
