# Purpose: RAG embeddings.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.rag.embeddings.arch_embedder
# Last-renovated: 2026-06-11
"""RAG embeddings."""
from .arch_embedder import (
    ArchitectureEmbedder,
    embed_architecture_scan,
)
