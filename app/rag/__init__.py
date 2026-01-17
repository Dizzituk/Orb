"""
ASTRA RAG System - Retrieval-Augmented Generation for codebase understanding.

This system provides semantic search and grounded Q&A over the architecture scan data.
It reads from existing ArchitectureFileContent tables (populated by scan commands),
chunks the content, generates embeddings, and enables intelligent retrieval.

Core API:
    ask_knowledge(question, scope) -> KnowledgeAnswer
    
HTTP API:
    POST /rag/query
    GET /rag/status
    POST /rag/index

Architecture:
    1. Reads FROM ArchitectureFileContent (single source of truth for file text)
    2. Chunks content intelligently (AST-based for code, window for docs)
    3. Generates embeddings via OpenAI text-embedding-3-small
    4. Stores vectors in sqlite-vec for similarity search
    5. Retrieves relevant chunks with filters (scope, type, path)
    6. Answers questions using tiered LLM routing (Mini → GPT-5.2 → Opus)

Data Flow:
    architecture_file_content (source) → RAG chunker → RAGChunk → embeddings → vector store

Phase 1: Foundation (models, schemas, vector store)
Phase 2: Indexing (chunking, embedding, orchestration)
Phase 3: Retrieval (search, ranking, filtering)
Phase 4: Grounded Q&A (prompts, answering, LLM integration)
Phase 5: API integration
Phase 6: CLI utilities

Note: This is a read-only consumer of scan data. No file modifications.
"""

# Exports will be added as modules are implemented in later phases
# Phase 1: Foundation only - no public API yet
__all__ = []

__version__ = "0.1.0"
