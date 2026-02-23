# FILE: app/memory/rag_entries_model.py
"""
SQLAlchemy model for the rag_entries table.

This table handles NEW memory domains only:
  - knowledge: domain knowledge items
  - context: ephemeral conversation context
  - decision: architectural decision records
  - ingested: items from document ingestion pipeline

Architecture data stays in arch_code_chunks (bridged via UnifiedRAGQuery).
Preferences stay in astra_preferences (bridged via MemoryRouter).

Used by: UnifiedRAGQuery, MemoryRouter, KnowledgeStore, ContextStore
"""

from datetime import datetime

from sqlalchemy import Column, Integer, String, Text, DateTime, LargeBinary, Index
from app.db import Base


class RAGEntry(Base):
    """
    Unified RAG entry for non-architecture memory domains.

    Each row is a single chunk of knowledge, context, decision, or
    ingested content. Embeddings are stored inline (BLOB) for simplicity
    since these entries are much fewer than arch_code_chunks.
    """
    __tablename__ = "rag_entries"

    id = Column(Integer, primary_key=True, index=True)

    # Project scoping — every row has a project
    project_id = Column(String(100), nullable=False, default="astra-core", index=True)

    # Memory domain
    # Values: 'knowledge', 'context', 'decision', 'ingested'
    domain = Column(String(50), nullable=False, index=True)

    # Optional file path (for file-backed entries)
    file_path = Column(String(1000), nullable=True)

    # The actual content
    chunk_text = Column(Text, nullable=False)

    # Embedding vector (nullable until embedded)
    embedding = Column(LargeBinary, nullable=True)

    # Lifecycle status
    # Values: 'ACTIVE', 'QUARANTINED', 'PURGED'
    status = Column(String(20), nullable=False, default="ACTIVE", index=True)

    # Refactor tracking (mirrors arch_code_chunks pattern)
    source_monolith = Column(String(500), nullable=True)
    refactor_job_id = Column(String(100), nullable=True)
    package_role = Column(String(500), nullable=True)

    # How this entry was created
    # Values: 'manual', 'ingest_pipeline', 'gpt_export', 'decision_log', 'context_extract'
    ingest_source = Column(String(50), nullable=True)

    # Timestamps
    indexed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    quarantined_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("ix_rag_entries_project_domain", "project_id", "domain"),
        Index("ix_rag_entries_status", "status"),
        Index("ix_rag_entries_domain", "domain"),
        Index("ix_rag_entries_file_path", "file_path"),
        Index("ix_rag_entries_ingest", "ingest_source"),
    )

    def __repr__(self):
        return (
            f"<RAGEntry(id={self.id}, domain='{self.domain}', "
            f"project='{self.project_id}', status='{self.status}')>"
        )
