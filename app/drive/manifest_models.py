# FILE: app/drive/manifest_models.py
"""
Drive file manifest — SQLAlchemy model for boot-time filesystem awareness.

Stores metadata for every file ASTRA has access to across all category
paths (Documents, Pictures, Music, Videos, Desktop, Screenshots,
ASTRA Output, Android Project).

Used by the boot scanner to detect new, modified, deleted, and moved
files between sessions. NOT for content storage — just metadata.

CRITICAL: Import in app/db.py init_db() or table won't be created!
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, BigInteger, DateTime,
    Boolean, Index, Text,
)
from app.db import Base


class DriveFileManifest(Base):
    """
    One row per file ASTRA can see on the host filesystem.

    Updated on every boot via delta scan. Previous state compared
    against current filesystem to detect changes.
    """
    __tablename__ = "drive_file_manifest"

    id = Column(Integer, primary_key=True, index=True)

    # File identity
    path = Column(String(1024), nullable=False, unique=True, index=True)
    filename = Column(String(256), nullable=False, index=True)
    extension = Column(String(32), nullable=True, index=True)

    # Category (documents, pictures, music, videos, desktop, etc.)
    category = Column(String(50), nullable=False, index=True)

    # File class (document, image, audio, video, code, other)
    file_class = Column(String(20), nullable=False, index=True)

    # Size and timestamps
    size_bytes = Column(BigInteger, nullable=False, default=0)
    mtime = Column(DateTime, nullable=False)

    # Content indexing state
    content_indexed = Column(Boolean, default=False, nullable=False, index=True)
    indexed_at = Column(DateTime, nullable=True)

    # Media metadata (populated by tier 3 cataloguing)
    media_metadata_json = Column(Text, nullable=True)

    # Scan tracking
    first_seen_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    scan_generation = Column(Integer, default=0, nullable=False, index=True)

    # Composite indexes for common queries
    __table_args__ = (
        Index("ix_dfm_category_class", "category", "file_class"),
        Index("ix_dfm_ext_class", "extension", "file_class"),
        Index("ix_dfm_last_seen", "last_seen_at"),
        Index("ix_dfm_content_indexed", "content_indexed"),
    )

    def __repr__(self):
        return f"<DriveFile({self.category}/{self.filename}, {self.file_class})>"
