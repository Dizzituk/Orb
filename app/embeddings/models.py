# FILE: app/embeddings/models.py
"""
SQLAlchemy model for embedding storage.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Index
from app.db import Base

# Import encrypted type for content field
try:
    from app.crypto import EncryptedText
    HAS_ENCRYPTION = True
except ImportError:
    HAS_ENCRYPTION = False
    EncryptedText = Text


class Embedding(Base):
    """
    Stores vector embeddings for semantic search.
    
    source_type: "note", "message", "file" (from DocumentContent)
    source_id: ID of the source record
    content: The text that was embedded (encrypted)
    embedding: JSON-encoded float array (not encrypted - vectors are meaningless without context)
    chunk_index: For long documents split into chunks (0-based)
    """
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, nullable=False, index=True)
    source_type = Column(String(20), nullable=False)  # "note", "message", "file"
    source_id = Column(Integer, nullable=False)
    chunk_index = Column(Integer, default=0, nullable=False)  # For chunked documents
    
    # The text content that was embedded (encrypted for privacy)
    content = Column(EncryptedText, nullable=False)
    
    # JSON-encoded embedding vector (not encrypted - meaningless without key)
    embedding = Column(Text, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Composite index for efficient lookups
    __table_args__ = (
        Index('ix_embeddings_project_source', 'project_id', 'source_type', 'source_id'),
    )