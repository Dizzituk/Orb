# FILE: tests/test_rag_models.py
"""
Tests for RAG models.

Tests:
- Model instantiation
- Relationships (RAGFile → chunks, RAGIndexRun → scan)
- Cascade deletes
- Query helpers
- Properties (has_embedding, line_range, duration_seconds)

Run:
    pytest tests/test_rag_models.py -v
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from datetime import datetime, timedelta

from sqlalchemy import (
    create_engine,
    Table,
    Column,
    Integer,
    text,
    event,
)
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.rag.models import (
    RAGFile, RAGChunk, RAGIndexRun,
    get_file_by_path, get_chunks_by_file_id,
    get_unembedded_chunks, get_embedded_chunks,
    count_files_by_project, get_index_stats,
)

# -------------------------------------------------------------------
# Test-only placeholder MODELS for external FK targets.
#
# In production, these tables already exist (architecture scan DB).
# In unit tests, we use sqlite:///:memory:, so we define minimal stubs
# so SQLAlchemy can resolve FK targets AND relationship() class names
# during Base.metadata.create_all().
# -------------------------------------------------------------------

# Only create if not already registered
if "ArchitectureScanRun" not in Base.registry._class_registry:
    class ArchitectureScanRun(Base):
        """Stub for architecture scan runs (FK target for RAGIndexRun)."""
        __tablename__ = "architecture_scan_runs"
        id = Column(Integer, primary_key=True)

if "ArchitectureFileIndex" not in Base.registry._class_registry:
    class ArchitectureFileIndex(Base):
        """Stub for architecture file index (FK target for RAGFile)."""
        __tablename__ = "architecture_file_index"
        id = Column(Integer, primary_key=True)


@pytest.fixture
def db_session():
    """Create in-memory database for testing."""
    engine = create_engine("sqlite:///:memory:")

    # Ensure SQLite FK enforcement is enabled (important for cascade/FK tests)
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Seed FK targets used in tests (multiple for tests that need several files)
    session.execute(text("INSERT INTO architecture_scan_runs (id) VALUES (1)"))
    for i in range(1, 11):  # IDs 1-10 for architecture files
        session.execute(text(f"INSERT INTO architecture_file_index (id) VALUES ({i})"))
    session.commit()

    yield session

    session.close()


def test_rag_file_creation(db_session):
    """Test RAGFile model creation."""
    file = RAGFile(
        architecture_file_id=1,
        path="D:\\Orb\\app\\llm\\streaming.py",
        filename="streaming.py",
        extension=".py",
        content_hash="abc123",
        file_mtime=datetime.utcnow(),
        file_size=10000,
        project_tag="backend",
        file_type="python",
        chunk_count=5,
        index_version=1,
    )

    db_session.add(file)
    db_session.commit()

    assert file.id is not None
    assert file.path == "D:\\Orb\\app\\llm\\streaming.py"
    assert file.project_tag == "backend"
    assert file.chunk_count == 5
    assert file.error_count == 0  # Default


def test_rag_chunk_creation(db_session):
    """Test RAGChunk model creation."""
    # Create file first
    file = RAGFile(
        architecture_file_id=1,
        path="D:\\Orb\\test.py",
        filename="test.py",
        extension=".py",
        content_hash="xyz789",
        file_mtime=datetime.utcnow(),
        file_size=5000,
        project_tag="backend",
        file_type="python",
    )
    db_session.add(file)
    db_session.commit()

    # Create chunk
    chunk = RAGChunk(
        file_id=file.id,
        chunk_index=0,
        chunk_hash="chunk123",
        content="def test(): pass",
        token_count=10,
        start_line=1,
        end_line=1,
        symbol_type="function",
        symbol_name="test",
    )

    db_session.add(chunk)
    db_session.commit()

    assert chunk.id is not None
    assert chunk.file_id == file.id
    assert chunk.symbol_type == "function"
    assert chunk.has_embedding is False  # No embedding yet
    assert chunk.line_range == "1-1"


def test_rag_index_run_creation(db_session):
    """Test RAGIndexRun model creation."""
    run = RAGIndexRun(
        scan_id=1,
        status="completed",
        files_scanned=100,
        files_indexed=95,
        files_skipped=3,
        files_errored=2,
        chunks_created=500,
        embeddings_generated=500,
        started_at=datetime.utcnow() - timedelta(minutes=10),
        finished_at=datetime.utcnow(),
    )

    db_session.add(run)
    db_session.commit()

    assert run.id is not None
    assert run.status == "completed"
    assert run.success_rate == 0.95
    assert run.duration_seconds is not None
    assert run.duration_seconds > 0


def test_file_chunks_relationship(db_session):
    """Test RAGFile → RAGChunk relationship."""
    file = RAGFile(
        architecture_file_id=1,
        path="D:\\Orb\\test.py",
        filename="test.py",
        extension=".py",
        content_hash="xyz",
        file_mtime=datetime.utcnow(),
        file_size=1000,
        project_tag="backend",
        file_type="python",
    )
    db_session.add(file)
    db_session.commit()

    # Add chunks
    for i in range(3):
        chunk = RAGChunk(
            file_id=file.id,
            chunk_index=i,
            chunk_hash=f"hash{i}",
            content=f"content {i}",
            token_count=10,
        )
        db_session.add(chunk)

    db_session.commit()

    # Test relationship
    assert file.chunks.count() == 3

    chunks = file.chunks.all()
    assert len(chunks) == 3
    assert all(c.file_id == file.id for c in chunks)


def test_cascade_delete(db_session):
    """Test that deleting a file cascades to delete chunks."""
    file = RAGFile(
        architecture_file_id=1,
        path="D:\\Orb\\test.py",
        filename="test.py",
        extension=".py",
        content_hash="xyz",
        file_mtime=datetime.utcnow(),
        file_size=1000,
        project_tag="backend",
        file_type="python",
    )
    db_session.add(file)
    db_session.commit()

    # Add chunk
    chunk = RAGChunk(
        file_id=file.id,
        chunk_index=0,
        chunk_hash="hash0",
        content="test content",
        token_count=5,
    )
    db_session.add(chunk)
    db_session.commit()

    chunk_id = chunk.id

    # Delete file
    db_session.delete(file)
    db_session.commit()

    # Chunk should be gone too
    deleted_chunk = db_session.query(RAGChunk).filter(RAGChunk.id == chunk_id).first()
    assert deleted_chunk is None


def test_embedding_metadata(db_session):
    """Test embedding metadata storage."""
    file = RAGFile(
        architecture_file_id=1,
        path="D:\\Orb\\test.py",
        filename="test.py",
        extension=".py",
        content_hash="xyz",
        file_mtime=datetime.utcnow(),
        file_size=1000,
        project_tag="backend",
        file_type="python",
    )
    db_session.add(file)
    db_session.commit()

    chunk = RAGChunk(
        file_id=file.id,
        chunk_index=0,
        chunk_hash="hash0",
        content="test content",
        token_count=5,
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
        embedded_at=datetime.utcnow(),
    )
    db_session.add(chunk)
    db_session.commit()

    assert chunk.has_embedding is True
    assert chunk.embedding_model == "text-embedding-3-small"
    assert chunk.embedding_dimensions == 1536


def test_query_helpers(db_session):
    """Test query helper functions."""
    # Create test data
    file = RAGFile(
        architecture_file_id=1,
        path="D:\\Orb\\test.py",
        filename="test.py",
        extension=".py",
        content_hash="xyz",
        file_mtime=datetime.utcnow(),
        file_size=1000,
        project_tag="backend",
        file_type="python",
    )
    db_session.add(file)
    db_session.commit()

    chunk1 = RAGChunk(
        file_id=file.id,
        chunk_index=0,
        chunk_hash="hash0",
        content="test",
        token_count=5,
        embedded_at=datetime.utcnow(),
    )
    chunk2 = RAGChunk(
        file_id=file.id,
        chunk_index=1,
        chunk_hash="hash1",
        content="test2",
        token_count=5,
    )
    db_session.add_all([chunk1, chunk2])
    db_session.commit()

    # Test helpers
    found_file = get_file_by_path(db_session, "D:\\Orb\\test.py")
    assert found_file.id == file.id

    chunks = get_chunks_by_file_id(db_session, file.id)
    assert len(chunks) == 2

    embedded = get_embedded_chunks(db_session)
    assert len(embedded) == 1

    unembedded = get_unembedded_chunks(db_session)
    assert len(unembedded) == 1

    stats = get_index_stats(db_session)
    assert stats["total_files"] == 1
    assert stats["total_chunks"] == 2
    assert stats["embedded_chunks"] == 1
    assert stats["unembedded_chunks"] == 1


def test_count_by_project(db_session):
    """Test counting files by project tag."""
    files = [
        RAGFile(
            architecture_file_id=i + 1,  # Use unique FK for each file (1-5)
            path=f"D:\\Orb\\test{i}.py",
            filename=f"test{i}.py",
            extension=".py",
            content_hash=f"hash{i}",
            file_mtime=datetime.utcnow(),
            file_size=1000,
            project_tag="backend" if i < 3 else "frontend",
            file_type="python",
        )
        for i in range(5)
    ]

    for f in files:
        db_session.add(f)
    db_session.commit()

    counts = count_files_by_project(db_session)
    assert counts["backend"] == 3
    assert counts["frontend"] == 2


def test_chunk_properties(db_session):
    """Test RAGChunk property methods."""
    file = RAGFile(
        architecture_file_id=1,
        path="D:\\Orb\\test.py",
        filename="test.py",
        extension=".py",
        content_hash="xyz",
        file_mtime=datetime.utcnow(),
        file_size=1000,
        project_tag="backend",
        file_type="python",
    )
    db_session.add(file)
    db_session.commit()

    chunk = RAGChunk(
        file_id=file.id,
        chunk_index=0,
        chunk_hash="hash0",
        content="test",
        token_count=5,
        start_line=10,
        end_line=25,
    )
    db_session.add(chunk)
    db_session.commit()

    assert chunk.line_range == "10-25"
    assert chunk.has_embedding is False

    # Add embedding metadata
    chunk.embedded_at = datetime.utcnow()
    db_session.commit()

    assert chunk.has_embedding is True


def test_index_run_properties(db_session):
    """Test RAGIndexRun property methods."""
    now = datetime.utcnow()
    run = RAGIndexRun(
        scan_id=1,
        status="completed",
        files_scanned=100,
        files_indexed=80,
        started_at=now - timedelta(minutes=5),
        finished_at=now,
    )
    db_session.add(run)
    db_session.commit()

    assert run.duration_seconds is not None
    assert 250 < run.duration_seconds < 350  # ~5 minutes
    assert run.success_rate == 0.8


# =============================================================================
# WP03: ARCHITECTURE RAG MODEL TESTS
# =============================================================================

from app.rag.models import (
    ArchScanRun,
    ArchDirectoryIndex,
    ArchCodeChunk,
    SourceType,
    ChunkType,
)


class TestSourceType:
    def test_arch_types(self):
        assert SourceType.ARCH_DIRECTORY == "arch_directory"
        assert SourceType.ARCH_CHUNK == "arch_chunk"


class TestChunkType:
    def test_embeddable(self):
        assert ChunkType.FUNCTION in ChunkType.EMBEDDABLE
        assert ChunkType.CLASS in ChunkType.EMBEDDABLE
        assert ChunkType.METHOD in ChunkType.EMBEDDABLE
        assert ChunkType.ASYNC_FUNCTION in ChunkType.EMBEDDABLE
        assert ChunkType.ASYNC_METHOD in ChunkType.EMBEDDABLE


def test_arch_scan_run_creation(db_session):
    """Test ArchScanRun model creation."""
    scan = ArchScanRun(
        status="running",
        signatures_file="SIGNATURES_2026-01-01_1812.json",
        index_file="INDEX_2026-01-01_1812.json",
    )
    db_session.add(scan)
    db_session.commit()
    
    assert scan.id is not None
    assert scan.status == "running"
    assert scan.started_at is not None
    assert scan.completed_at is None
    assert scan.directories_indexed == 0
    assert scan.chunks_extracted == 0


def test_arch_scan_run_complete(db_session):
    """Test ArchScanRun completion."""
    scan = ArchScanRun(status="complete")
    scan.directories_indexed = 50
    scan.chunks_extracted = 1000
    scan.embeddings_created = 1000
    scan.completed_at = datetime.utcnow()
    
    db_session.add(scan)
    db_session.commit()
    
    assert scan.status == "complete"
    assert scan.directories_indexed == 50
    assert scan.chunks_extracted == 1000


def test_arch_directory_index_creation(db_session):
    """Test ArchDirectoryIndex model creation."""
    scan = ArchScanRun(status="complete")
    db_session.add(scan)
    db_session.commit()
    
    directory = ArchDirectoryIndex(
        scan_id=scan.id,
        canonical_path="sandbox:d-drive/Orb/app",
        abs_path="D:\\Orb\\app",
        name="app",
        root_alias="d-drive",
        root_kind="sandbox",
        zone="projects",
        depth=2,
        file_count=50,
        subdir_count=10,
        total_lines=5000,
        total_bytes=150000,
    )
    db_session.add(directory)
    db_session.commit()
    
    assert directory.id is not None
    assert directory.scan_id == scan.id
    assert directory.canonical_path == "sandbox:d-drive/Orb/app"
    assert directory.file_count == 50


def test_arch_directory_index_parent(db_session):
    """Test ArchDirectoryIndex parent relationship."""
    scan = ArchScanRun(status="complete")
    db_session.add(scan)
    db_session.commit()
    
    parent_dir = ArchDirectoryIndex(
        scan_id=scan.id,
        canonical_path="sandbox:d-drive/Orb",
        name="Orb",
        depth=1,
    )
    db_session.add(parent_dir)
    db_session.commit()
    
    child_dir = ArchDirectoryIndex(
        scan_id=scan.id,
        canonical_path="sandbox:d-drive/Orb/app",
        name="app",
        parent_id=parent_dir.id,
        depth=2,
    )
    db_session.add(child_dir)
    db_session.commit()
    
    assert child_dir.parent_id == parent_dir.id
    assert child_dir.parent.name == "Orb"


def test_arch_code_chunk_creation(db_session):
    """Test ArchCodeChunk model creation."""
    scan = ArchScanRun(status="complete")
    db_session.add(scan)
    db_session.commit()
    
    chunk = ArchCodeChunk(
        scan_id=scan.id,
        file_path="sandbox:d-drive/Orb/app/main.py",
        file_abs_path="D:\\Orb\\app\\main.py",
        chunk_type=ChunkType.FUNCTION,
        chunk_name="main",
        qualified_name="main",
        start_line=10,
        end_line=50,
        signature="() -> None",
        docstring="Main entry point.",
        returns="None",
    )
    db_session.add(chunk)
    db_session.commit()
    
    assert chunk.id is not None
    assert chunk.chunk_type == "function"
    assert chunk.chunk_name == "main"
    assert chunk.start_line == 10
    assert chunk.end_line == 50


def test_arch_code_chunk_class(db_session):
    """Test ArchCodeChunk for class with method."""
    scan = ArchScanRun(status="complete")
    db_session.add(scan)
    db_session.commit()
    
    class_chunk = ArchCodeChunk(
        scan_id=scan.id,
        file_path="sandbox:d-drive/Orb/app/service.py",
        chunk_type=ChunkType.CLASS,
        chunk_name="MyService",
        qualified_name="MyService",
        start_line=1,
        end_line=100,
        bases_json='["BaseService"]',
    )
    db_session.add(class_chunk)
    db_session.commit()
    
    method_chunk = ArchCodeChunk(
        scan_id=scan.id,
        file_path="sandbox:d-drive/Orb/app/service.py",
        chunk_type=ChunkType.METHOD,
        chunk_name="process",
        qualified_name="MyService.process",
        start_line=10,
        end_line=30,
        parent_chunk_id=class_chunk.id,
        signature="(self, data: dict) -> bool",
    )
    db_session.add(method_chunk)
    db_session.commit()
    
    assert method_chunk.parent_chunk_id == class_chunk.id
    assert method_chunk.chunk_type == "method"
