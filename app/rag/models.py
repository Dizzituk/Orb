"""
RAG database models.

Tables:
- rag_files: File metadata + indexing state (references architecture_file_index)
- rag_chunks: Text chunks with metadata (embeddings stored in rag_vectors.db)
- rag_index_runs: Index job history

Design Decisions:
1. RAGFile has FK to architecture_file_index (source of truth) + denormalized path
2. RAGChunk stores embedding METADATA only (model, dims, timestamp) - bytes in rag_vectors.db
3. Reindexing = clean slate (delete old chunks, rebuild from scratch)
4. RAGIndexRun tracks which architecture scan was indexed (scan_id FK)

CRITICAL: Import in app/db.py init_db() or tables won't be created!
See: DB_WIRING_INSTRUCTIONS.md
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, 
    ForeignKey, Index, Boolean, BigInteger
)
from sqlalchemy.orm import relationship
from app.db import Base

# Import pre-existing architecture models (DO NOT redefine - causes duplicate table error)
from app.memory.architecture_models import ArchitectureScanRun, ArchitectureFileIndex


# =============================================================================
# RAG MODELS
# =============================================================================

class RAGFile(Base):
    """
    File metadata and indexing state.
    
    References architecture_file_index (source of truth) via FK.
    Tracks which files have been indexed and their state for incremental updates.
    
    Design: FK to architecture_file_index ensures we only index scanned files.
    Path is denormalized for convenience/debugging but FK is the authority.
    """
    __tablename__ = "rag_files"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Foreign key to architecture scan (source of truth)
    architecture_file_id = Column(
        Integer, 
        ForeignKey("architecture_file_index.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,  # One RAGFile per architecture file
        index=True,
    )
    
    # Denormalized path for convenience (matches architecture_file_index.path)
    # This makes queries/debugging easier without joining every time
    path = Column(String(1024), nullable=False, index=True)
    
    # File identity (denormalized from architecture scan)
    filename = Column(String(256), nullable=False, index=True)
    extension = Column(String(32), nullable=True, index=True)
    
    # Change detection (from architecture_file_content)
    content_hash = Column(String(64), nullable=False)  # SHA-256
    file_mtime = Column(DateTime, nullable=False)
    file_size = Column(Integer, nullable=False)
    
    # Classification (from architecture scan zone)
    project_tag = Column(String(32), nullable=False, default="other", index=True)
    file_type = Column(String(32), nullable=False, index=True)  # python, typescript, markdown, etc.
    
    # Indexing state
    indexed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    chunk_count = Column(Integer, default=0, nullable=False)
    index_version = Column(Integer, default=1, nullable=False)  # For re-index tracking
    
    # Error tracking
    last_error = Column(Text, nullable=True)
    error_count = Column(Integer, default=0, nullable=False)
    
    # Relationships
    architecture_file = relationship(
        "ArchitectureFileIndex",
        foreign_keys=[architecture_file_id],
        backref="rag_file",
    )
    
    chunks = relationship(
        "RAGChunk",
        back_populates="file",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    
    # Composite indexes for common queries
    __table_args__ = (
        Index("ix_rag_files_project_type", "project_tag", "file_type"),
        Index("ix_rag_files_version", "index_version"),
        Index("ix_rag_files_indexed_at", "indexed_at"),
    )
    
    def __repr__(self):
        return f"<RAGFile(id={self.id}, path={self.path}, chunks={self.chunk_count})>"


class RAGChunk(Base):
    """
    Text chunk with metadata.
    
    Stores chunk content and metadata. Embedding BYTES stored separately in rag_vectors.db.
    This table stores only embedding METADATA (model, dimensions, timestamp).
    
    Design: Single source of truth for embeddings = rag_vectors.db.
    Avoids drift from storing vectors in two places.
    """
    __tablename__ = "rag_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Foreign key to file (cascade delete - if file deleted, chunks go too)
    file_id = Column(
        Integer, 
        ForeignKey("rag_files.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    
    # Chunk identity
    chunk_index = Column(Integer, nullable=False)  # Position in file (0, 1, 2...)
    chunk_hash = Column(String(64), nullable=False, index=True)  # For deduplication
    
    # Chunk content
    content = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False)
    
    # Position in file
    start_line = Column(Integer, nullable=True)
    end_line = Column(Integer, nullable=True)
    
    # Symbol information (for code chunks)
    symbol_type = Column(String(50), nullable=True)  # function, class, method, etc.
    symbol_name = Column(String(256), nullable=True)
    parent_symbol = Column(String(256), nullable=True)  # Parent class/module
    
    # Embedding metadata (NOT the embedding bytes - those are in rag_vectors.db)
    embedding_model = Column(String(100), nullable=True)  # e.g., "text-embedding-3-small"
    embedding_dimensions = Column(Integer, nullable=True)  # e.g., 1536
    embedded_at = Column(DateTime, nullable=True)  # When embedding was generated
    
    # Relationship back to file
    file = relationship("RAGFile", back_populates="chunks")
    
    # Composite indexes for common queries
    __table_args__ = (
        Index("ix_rag_chunks_file_index", "file_id", "chunk_index"),
        Index("ix_rag_chunks_symbol", "symbol_type", "symbol_name"),
        Index("ix_rag_chunks_hash", "chunk_hash"),
        Index("ix_rag_chunks_embedded", "embedded_at"),
    )
    
    def __repr__(self):
        return f"<RAGChunk(id={self.id}, file_id={self.file_id}, chunk_index={self.chunk_index}, tokens={self.token_count})>"
    
    @property
    def has_embedding(self) -> bool:
        """Check if this chunk has been embedded."""
        return self.embedded_at is not None
    
    @property
    def line_range(self) -> Optional[str]:
        """Get line range as string (e.g., '42-78')."""
        if self.start_line is not None and self.end_line is not None:
            return f"{self.start_line}-{self.end_line}"
        return None


class RAGIndexRun(Base):
    """
    Index job history.
    
    Tracks each indexing run for debugging and incremental updates.
    Links to the architecture scan that was indexed (scan_id FK).
    
    Design: Even on clean-slate reindex, we keep run history for audit trail.
    """
    __tablename__ = "rag_index_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Foreign key to architecture scan (which scan was indexed)
    scan_id = Column(
        Integer,
        ForeignKey("architecture_scan_runs.id", ondelete="SET NULL"),
        nullable=True,  # Nullable for future non-scan-backed indexing
        index=True,
    )
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    finished_at = Column(DateTime, nullable=True)
    
    # Status (explicit index in __table_args__, don't use index=True here)
    status = Column(String(20), nullable=False, default="running")
    # Values: "running", "completed", "failed", "cancelled"
    
    # Statistics
    files_scanned = Column(Integer, default=0, nullable=False)
    files_indexed = Column(Integer, default=0, nullable=False)
    files_skipped = Column(Integer, default=0, nullable=False)
    files_errored = Column(Integer, default=0, nullable=False)
    
    chunks_created = Column(Integer, default=0, nullable=False)
    embeddings_generated = Column(Integer, default=0, nullable=False)
    
    # Error info
    error_message = Column(Text, nullable=True)
    
    # Config snapshot (JSON string of config used for this run)
    config_snapshot = Column(Text, nullable=True)
    
    # Relationship to scan
    scan_run = relationship(
        "ArchitectureScanRun",
        foreign_keys=[scan_id],
        backref="rag_index_runs",
    )
    
    # Indexes
    __table_args__ = (
        Index("ix_rag_index_runs_status", "status"),
        Index("ix_rag_index_runs_started", "started_at"),
        Index("ix_rag_index_runs_scan", "scan_id"),
    )
    
    def __repr__(self):
        return f"<RAGIndexRun(id={self.id}, status={self.status}, files={self.files_indexed}/{self.files_scanned})>"
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate run duration in seconds."""
        if self.finished_at and self.started_at:
            delta = self.finished_at - self.started_at
            return delta.total_seconds()
        return None
    
    @property
    def success_rate(self) -> Optional[float]:
        """Calculate success rate (files indexed / files scanned)."""
        if self.files_scanned > 0:
            return self.files_indexed / self.files_scanned
        return None


# =============================================================================
# WP03: ARCHITECTURE RAG MODELS (NEW - DO NOT MODIFY ABOVE)
# =============================================================================

class SourceType:
    """Source types for embeddings (extends existing)."""
    ARCH_DIRECTORY = "arch_directory"
    ARCH_CHUNK = "arch_chunk"


class ChunkType:
    """Code chunk types."""
    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    CLASS = "class"
    METHOD = "method"
    ASYNC_METHOD = "async_method"
    
    EMBEDDABLE = {FUNCTION, ASYNC_FUNCTION, CLASS, METHOD, ASYNC_METHOD}


class ArchScanRun(Base):
    """Track RAG architecture scan runs."""
    __tablename__ = "arch_scan_runs"
    
    id = Column(Integer, primary_key=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(20), default="running")  # running, complete, failed
    
    signatures_file = Column(String(500))
    index_file = Column(String(500))
    
    directories_indexed = Column(Integer, default=0)
    chunks_extracted = Column(Integer, default=0)
    embeddings_created = Column(Integer, default=0)
    
    error_message = Column(Text, nullable=True)


class ArchDirectoryIndex(Base):
    """Directory hierarchy for RAG."""
    __tablename__ = "arch_directory_index"
    
    id = Column(Integer, primary_key=True)
    scan_id = Column(Integer, ForeignKey("arch_scan_runs.id"), nullable=False, index=True)
    
    canonical_path = Column(String(500), nullable=False)
    abs_path = Column(String(1000))
    name = Column(String(200), nullable=False)
    
    root_alias = Column(String(50))
    root_kind = Column(String(20))
    zone = Column(String(50))
    
    parent_id = Column(Integer, ForeignKey("arch_directory_index.id"), nullable=True)
    depth = Column(Integer, default=0)
    
    file_count = Column(Integer, default=0)
    subdir_count = Column(Integer, default=0)
    total_lines = Column(Integer, default=0)
    total_bytes = Column(Integer, default=0)
    
    top_files_json = Column(Text)
    extensions_json = Column(Text)
    
    summary = Column(Text)
    summary_tokens = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    parent = relationship("ArchDirectoryIndex", remote_side=[id])
    
    __table_args__ = (
        Index("ix_archdir_scan_path", "scan_id", "canonical_path"),
    )


class ArchCodeChunk(Base):
    """Code chunk from SIGNATURES_*.json."""
    __tablename__ = "arch_code_chunks"
    
    id = Column(Integer, primary_key=True)
    scan_id = Column(Integer, ForeignKey("arch_scan_runs.id"), nullable=False, index=True)
    
    file_path = Column(String(500), nullable=False)
    file_abs_path = Column(String(1000))
    
    chunk_type = Column(String(20), nullable=False)
    chunk_name = Column(String(200), nullable=False)
    qualified_name = Column(String(300))
    
    start_line = Column(Integer)
    end_line = Column(Integer)
    
    signature = Column(Text)
    docstring = Column(Text)
    decorators_json = Column(Text)
    parameters_json = Column(Text)
    returns = Column(String(200))
    bases_json = Column(Text)
    
    descriptor = Column(Text)
    descriptor_tokens = Column(Integer)
    
    parent_chunk_id = Column(Integer, ForeignKey("arch_code_chunks.id"), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # ==========================================================================
    # EMBEDDING STATUS (v1.1 - incremental embedding support)
    # ==========================================================================
    # Content hash for change detection (SHA-256 of signature+docstring+name)
    content_hash = Column(String(64), nullable=True, index=True)
    
    # Embedding state
    embedded = Column(Boolean, default=False, nullable=False, index=True)
    embedding_model = Column(String(100), nullable=True)  # e.g. "text-embedding-3-small"
    embedded_at = Column(DateTime, nullable=True)
    
    # Hash that was used when embedding was generated (for staleness detection)
    embedded_content_hash = Column(String(64), nullable=True)
    
    __table_args__ = (
        Index("ix_archchunk_scan_file", "scan_id", "file_path"),
        Index("ix_archchunk_scan_name", "scan_id", "chunk_name"),
        Index("ix_archchunk_embedded", "embedded"),
        Index("ix_archchunk_content_hash", "content_hash"),
    )
    
    @property
    def needs_embedding(self) -> bool:
        """
        Check if this chunk needs (re)embedding.
        
        True if:
        - Never embedded
        - Content hash changed since last embedding
        """
        if not self.embedded:
            return True
        if self.content_hash and self.embedded_content_hash:
            return self.content_hash != self.embedded_content_hash
        return False


# =============================================================================
# QUERY HELPERS
# =============================================================================

def get_file_by_architecture_id(db, architecture_file_id: int) -> Optional[RAGFile]:
    """Get RAGFile by architecture_file_index.id."""
    return db.query(RAGFile).filter(
        RAGFile.architecture_file_id == architecture_file_id
    ).first()


def get_file_by_path(db, path: str) -> Optional[RAGFile]:
    """Get RAGFile by path."""
    return db.query(RAGFile).filter(RAGFile.path == path).first()


def get_files_by_project_tag(db, project_tag: str) -> list[RAGFile]:
    """Get all RAGFiles with a specific project tag."""
    return db.query(RAGFile).filter(RAGFile.project_tag == project_tag).all()


def get_files_by_type(db, file_type: str) -> list[RAGFile]:
    """Get all RAGFiles with a specific file type."""
    return db.query(RAGFile).filter(RAGFile.file_type == file_type).all()


def get_chunks_by_file_id(db, file_id: int) -> list[RAGChunk]:
    """Get all chunks for a file."""
    return db.query(RAGChunk).filter(RAGChunk.file_id == file_id).order_by(
        RAGChunk.chunk_index
    ).all()


def get_chunk_by_id(db, chunk_id: int) -> Optional[RAGChunk]:
    """Get a single chunk by ID."""
    return db.query(RAGChunk).filter(RAGChunk.id == chunk_id).first()


def get_embedded_chunks(db, limit: int = None) -> list[RAGChunk]:
    """Get chunks that have embeddings."""
    query = db.query(RAGChunk).filter(RAGChunk.embedded_at.isnot(None))
    if limit:
        query = query.limit(limit)
    return query.all()


def get_unembedded_chunks(db, limit: int = None) -> list[RAGChunk]:
    """Get chunks that need embeddings."""
    query = db.query(RAGChunk).filter(RAGChunk.embedded_at.is_(None))
    if limit:
        query = query.limit(limit)
    return query.all()


def get_latest_index_run(db) -> Optional[RAGIndexRun]:
    """Get the most recent index run."""
    return db.query(RAGIndexRun).order_by(RAGIndexRun.started_at.desc()).first()


def get_successful_index_runs(db, limit: int = 10) -> list[RAGIndexRun]:
    """Get recent successful index runs."""
    return db.query(RAGIndexRun).filter(
        RAGIndexRun.status == "completed"
    ).order_by(RAGIndexRun.started_at.desc()).limit(limit).all()


def count_files_by_project(db) -> dict[str, int]:
    """Count RAGFiles grouped by project tag."""
    from sqlalchemy import func
    results = db.query(
        RAGFile.project_tag,
        func.count(RAGFile.id)
    ).group_by(RAGFile.project_tag).all()
    return {tag: count for tag, count in results}


def count_chunks_by_project(db) -> dict[str, int]:
    """Count RAGChunks grouped by project tag."""
    from sqlalchemy import func
    results = db.query(
        RAGFile.project_tag,
        func.count(RAGChunk.id)
    ).join(RAGChunk).group_by(RAGFile.project_tag).all()
    return {tag: count for tag, count in results}


def delete_files_by_index_version(db, index_version: int) -> int:
    """
    Delete all files with a specific index version (clean slate reindex).
    
    Returns number of files deleted.
    Cascades to delete associated chunks.
    """
    count = db.query(RAGFile).filter(RAGFile.index_version == index_version).count()
    db.query(RAGFile).filter(RAGFile.index_version == index_version).delete()
    db.commit()
    return count


def get_index_stats(db) -> dict:
    """Get overall index statistics."""
    from sqlalchemy import func
    
    total_files = db.query(func.count(RAGFile.id)).scalar() or 0
    total_chunks = db.query(func.count(RAGChunk.id)).scalar() or 0
    
    embedded_chunks = db.query(func.count(RAGChunk.id)).filter(
        RAGChunk.embedded_at.isnot(None)
    ).scalar() or 0
    
    latest_run = get_latest_index_run(db)
    
    return {
        "total_files": total_files,
        "total_chunks": total_chunks,
        "embedded_chunks": embedded_chunks,
        "unembedded_chunks": total_chunks - embedded_chunks,
        "by_project": count_files_by_project(db),
        "by_type": count_files_by_type(db),
        "latest_run": latest_run,
    }


def count_files_by_type(db) -> dict[str, int]:
    """Count RAGFiles grouped by file type."""
    from sqlalchemy import func
    results = db.query(
        RAGFile.file_type,
        func.count(RAGFile.id)
    ).group_by(RAGFile.file_type).all()
    return {file_type: count for file_type, count in results}
