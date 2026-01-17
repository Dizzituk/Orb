# FILE: app/memory/architecture_models.py
"""
SQLAlchemy ORM models for Architecture Scan storage.

These tables store structured scan data from ASTRA architecture commands.
Designed for fast querying by path, zone, scope, and scan run.
Also stores actual file contents for RAG and code analysis.

v1.0 (2026-01): Initial version with ArchitectureScanRun and ArchitectureFileIndex
v2.0 (2026-01): Added ArchitectureFileContent for source code storage

Scopes:
    - "code": D:\\Orb + D:\\orb-desktop (UPDATE ARCHITECTURE)
    - "sandbox": C:\\Users + D:\\ broader scan (SCAN SANDBOX)
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, BigInteger, Index
from sqlalchemy.orm import relationship
from app.db import Base


# =============================================================================
# LANGUAGE DETECTION
# =============================================================================

EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".pyw": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".json": "json",
    ".jsonc": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "restructuredtext",
    ".txt": "text",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".sql": "sql",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".bat": "batch",
    ".cmd": "batch",
}

# Extensions for files we should capture content from
CONTENT_EXTENSIONS = {
    # Code
    ".py", ".pyw", ".pyi",
    ".js", ".mjs", ".cjs", ".jsx",
    ".ts", ".tsx", ".mts", ".cts",
    ".json", ".jsonc",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".sql",
    ".sh", ".bash", ".zsh", ".ps1", ".psm1", ".bat", ".cmd",
    # Docs
    ".md", ".markdown", ".rst", ".txt",
    # Config (no extension handled specially)
}

# Max file size to capture content (500KB)
MAX_CONTENT_SIZE_BYTES = 500 * 1024


def detect_language(ext: str, filename: str = "") -> str:
    """Detect programming language from extension or filename."""
    ext_lower = (ext or "").lower()
    
    # Check extension map
    if ext_lower in EXTENSION_TO_LANGUAGE:
        return EXTENSION_TO_LANGUAGE[ext_lower]
    
    # Check filename for special cases
    filename_lower = (filename or "").lower()
    if filename_lower == "dockerfile":
        return "dockerfile"
    if filename_lower == "makefile":
        return "makefile"
    if filename_lower in (".env", ".env.example", ".env.local"):
        return "dotenv"
    if filename_lower in (".gitignore", ".dockerignore"):
        return "gitignore"
    
    return "text"


def should_capture_content(ext: str, size_bytes: int = 0, filename: str = "") -> bool:
    """Determine if we should capture file content."""
    if size_bytes and size_bytes > MAX_CONTENT_SIZE_BYTES:
        return False
    
    ext_lower = (ext or "").lower()
    filename_lower = (filename or "").lower()
    
    # Check extension
    if ext_lower in CONTENT_EXTENSIONS:
        return True
    
    # Check special filenames without extensions
    if filename_lower in (".env", ".gitignore", ".gitattributes", "dockerfile", "makefile"):
        return True
    
    # No extension but small file - might be config
    if not ext_lower and size_bytes and size_bytes < 50000:
        return True
    
    return False


# =============================================================================
# MODELS
# =============================================================================

class ArchitectureScanRun(Base):
    """
    A single architecture scan execution.
    
    Tracks metadata about the scan: when it ran, what scope, success/failure.
    """
    __tablename__ = "architecture_scan_runs"

    id = Column(Integer, primary_key=True, index=True)
    
    # Scope: "code" (Orb + orb-desktop) or "sandbox" (broader environment)
    scope = Column(String(50), nullable=False, index=True)
    
    # Status: "running", "completed", "failed"
    status = Column(String(20), nullable=False, default="running", index=True)
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    finished_at = Column(DateTime, nullable=True)
    
    # Stats (JSON string for flexibility)
    stats_json = Column(Text, nullable=True)
    
    # Error message if failed
    error_message = Column(Text, nullable=True)
    
    # Relationships
    files = relationship(
        "ArchitectureFileIndex",
        back_populates="scan_run",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    
    def __repr__(self):
        return f"<ArchitectureScanRun(id={self.id}, scope={self.scope}, status={self.status})>"


class ArchitectureFileIndex(Base):
    """
    A single file entry in an architecture scan.
    
    Stores path, metadata, and zone classification.
    """
    __tablename__ = "architecture_file_index"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Foreign key to scan run
    scan_id = Column(Integer, ForeignKey("architecture_scan_runs.id"), nullable=False, index=True)
    
    # Full path
    path = Column(String(1000), nullable=False)
    
    # Filename only
    name = Column(String(255), nullable=False, index=True)
    
    # Extension
    ext = Column(String(20), nullable=True, index=True)
    
    # File size in bytes
    size_bytes = Column(BigInteger, nullable=True)
    
    # Modification time (ISO format string)
    mtime = Column(String(30), nullable=True)
    
    # Zone classification: "backend", "frontend", "tools", "user", "other"
    zone = Column(String(50), nullable=False, index=True)
    
    # Root directory this file belongs to
    root = Column(String(500), nullable=True)
    
    # Code metadata
    line_count = Column(Integer, nullable=True)
    language = Column(String(50), nullable=True, index=True)
    
    # Relationships
    scan_run = relationship("ArchitectureScanRun", back_populates="files")
    content = relationship(
        "ArchitectureFileContent",
        back_populates="file_index",
        uselist=False,
        cascade="all, delete-orphan",
    )
    
    __table_args__ = (
        Index("ix_arch_file_path_prefix", "path"),
        Index("ix_arch_file_scan_zone", "scan_id", "zone"),
        Index("ix_arch_file_scan_ext", "scan_id", "ext"),
        Index("ix_arch_file_scan_lang", "scan_id", "language"),
    )
    
    def __repr__(self):
        return f"<ArchitectureFileIndex(id={self.id}, name={self.name})>"


class ArchitectureFileContent(Base):
    """
    Stores actual source code content for a file.
    
    Separate table to keep file index queries fast.
    """
    __tablename__ = "architecture_file_content"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Foreign key to file index (one-to-one)
    file_index_id = Column(
        Integer, 
        ForeignKey("architecture_file_index.id"), 
        nullable=False, 
        unique=True,
        index=True,
    )
    
    # Actual source code
    content_text = Column(Text, nullable=False)
    
    # SHA256 hash of content for dedup/change detection
    content_hash = Column(String(64), nullable=True, index=True)
    
    # Relationship back to file index
    file_index = relationship("ArchitectureFileIndex", back_populates="content")
    
    def __repr__(self):
        return f"<ArchitectureFileContent(id={self.id}, file_index_id={self.file_index_id})>"


# =============================================================================
# QUERY HELPERS
# =============================================================================

def get_latest_scan(db, scope: str = None) -> ArchitectureScanRun:
    """Get the most recent completed scan, optionally filtered by scope."""
    query = db.query(ArchitectureScanRun).filter(
        ArchitectureScanRun.status == "completed"
    )
    if scope:
        query = query.filter(ArchitectureScanRun.scope == scope)
    return query.order_by(ArchitectureScanRun.finished_at.desc()).first()


def get_files_by_zone(db, scan_id: int, zone: str) -> list:
    """Get all files in a specific zone from a scan."""
    return db.query(ArchitectureFileIndex).filter(
        ArchitectureFileIndex.scan_id == scan_id,
        ArchitectureFileIndex.zone == zone,
    ).all()


def get_files_by_extension(db, scan_id: int, ext: str) -> list:
    """Get all files with a specific extension from a scan."""
    return db.query(ArchitectureFileIndex).filter(
        ArchitectureFileIndex.scan_id == scan_id,
        ArchitectureFileIndex.ext == ext,
    ).all()


def get_files_by_language(db, scan_id: int, language: str) -> list:
    """Get all files with a specific language from a scan."""
    return db.query(ArchitectureFileIndex).filter(
        ArchitectureFileIndex.scan_id == scan_id,
        ArchitectureFileIndex.language == language,
    ).all()


def get_files_by_path_prefix(db, scan_id: int, prefix: str) -> list:
    """Get all files under a path prefix from a scan."""
    return db.query(ArchitectureFileIndex).filter(
        ArchitectureFileIndex.scan_id == scan_id,
        ArchitectureFileIndex.path.like(f"{prefix}%"),
    ).all()


def get_file_with_content(db, scan_id: int, filename: str) -> ArchitectureFileIndex:
    """Get a specific file with its content loaded."""
    return db.query(ArchitectureFileIndex).filter(
        ArchitectureFileIndex.scan_id == scan_id,
        ArchitectureFileIndex.name == filename,
    ).first()


def get_file_by_path(db, scan_id: int, path: str) -> ArchitectureFileIndex:
    """Get a specific file by exact path with content loaded."""
    return db.query(ArchitectureFileIndex).filter(
        ArchitectureFileIndex.scan_id == scan_id,
        ArchitectureFileIndex.path == path,
    ).first()


def count_files_by_zone(db, scan_id: int) -> dict:
    """Count files grouped by zone."""
    from sqlalchemy import func
    results = db.query(
        ArchitectureFileIndex.zone,
        func.count(ArchitectureFileIndex.id)
    ).filter(
        ArchitectureFileIndex.scan_id == scan_id
    ).group_by(ArchitectureFileIndex.zone).all()
    return {zone: count for zone, count in results}


def count_files_by_language(db, scan_id: int) -> dict:
    """Count files grouped by language."""
    from sqlalchemy import func
    results = db.query(
        ArchitectureFileIndex.language,
        func.count(ArchitectureFileIndex.id)
    ).filter(
        ArchitectureFileIndex.scan_id == scan_id
    ).group_by(ArchitectureFileIndex.language).all()
    return {lang or "unknown": count for lang, count in results}


def count_lines_by_zone(db, scan_id: int) -> dict:
    """Count total lines of code grouped by zone."""
    from sqlalchemy import func
    results = db.query(
        ArchitectureFileIndex.zone,
        func.sum(ArchitectureFileIndex.line_count)
    ).filter(
        ArchitectureFileIndex.scan_id == scan_id
    ).group_by(ArchitectureFileIndex.zone).all()
    return {zone: int(count or 0) for zone, count in results}


def search_files_by_name(db, scan_id: int, pattern: str) -> list:
    """Search files by name pattern (SQL LIKE)."""
    return db.query(ArchitectureFileIndex).filter(
        ArchitectureFileIndex.scan_id == scan_id,
        ArchitectureFileIndex.name.ilike(f"%{pattern}%"),
    ).all()