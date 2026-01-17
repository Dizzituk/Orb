# tests/test_rag_directory_indexer.py
"""Tests for directory indexer."""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from sqlalchemy import create_engine, Column, Integer, text, event
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.rag.models import ArchScanRun, ArchDirectoryIndex


# Ensure stub models exist for FK resolution
if "ArchitectureScanRun" not in Base.registry._class_registry:
    class ArchitectureScanRun(Base):
        __tablename__ = "architecture_scan_runs"
        id = Column(Integer, primary_key=True)

if "ArchitectureFileIndex" not in Base.registry._class_registry:
    class ArchitectureFileIndex(Base):
        __tablename__ = "architecture_file_index"
        id = Column(Integer, primary_key=True)


@pytest.fixture
def db_session():
    """Create in-memory database for testing."""
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Seed FK targets
    session.execute(text("INSERT INTO architecture_scan_runs (id) VALUES (1)"))
    for i in range(1, 11):
        session.execute(text(f"INSERT INTO architecture_file_index (id) VALUES ({i})"))
    session.commit()

    yield session
    session.close()


class TestDirectoryIndexBuilder:
    def test_load_from_index(self, db_session):
        from app.rag.indexing.directory_indexer import DirectoryIndexBuilder
        
        # Create scan run
        scan = ArchScanRun(status="running")
        db_session.add(scan)
        db_session.commit()
        
        builder = DirectoryIndexBuilder(db_session, scan.id)
        
        # Mock INDEX.json data
        index_data = {
            "scanned_files": [
                {"path": "app/main.py", "bytes": 1000, "lines": 50},
                {"path": "app/llm/router.py", "bytes": 5000, "lines": 200},
                {"path": "tests/test_main.py", "bytes": 500, "lines": 30},
            ]
        }
        
        count = builder.load_from_index_json(index_data, r"D:\Orb")
        assert count > 0
    
    def test_save_to_db(self, db_session):
        from app.rag.indexing.directory_indexer import DirectoryIndexBuilder
        
        scan = ArchScanRun(status="running")
        db_session.add(scan)
        db_session.commit()
        
        builder = DirectoryIndexBuilder(db_session, scan.id)
        
        index_data = {
            "scanned_files": [
                {"path": "app/main.py", "bytes": 1000, "lines": 50},
                {"path": "app/llm/router.py", "bytes": 5000, "lines": 200},
            ]
        }
        
        builder.load_from_index_json(index_data, r"D:\Orb")
        stats = builder.save_to_db()
        
        assert stats["created"] > 0
        
        # Verify in database
        count = db_session.query(ArchDirectoryIndex).filter_by(
            scan_id=scan.id
        ).count()
        assert count > 0
    
    def test_idempotent(self, db_session):
        """Running twice should not duplicate."""
        from app.rag.indexing.directory_indexer import DirectoryIndexBuilder
        
        scan = ArchScanRun(status="running")
        db_session.add(scan)
        db_session.commit()
        
        index_data = {
            "scanned_files": [
                {"path": "app/main.py", "bytes": 1000, "lines": 50},
            ]
        }
        
        # First run
        builder1 = DirectoryIndexBuilder(db_session, scan.id)
        builder1.load_from_index_json(index_data, r"D:\Orb")
        stats1 = builder1.save_to_db()
        count1 = db_session.query(ArchDirectoryIndex).filter_by(scan_id=scan.id).count()
        
        # Second run
        builder2 = DirectoryIndexBuilder(db_session, scan.id)
        builder2.load_from_index_json(index_data, r"D:\Orb")
        stats2 = builder2.save_to_db()
        count2 = db_session.query(ArchDirectoryIndex).filter_by(scan_id=scan.id).count()
        
        # Should update, not create new
        assert stats2["updated"] > 0 or stats2["created"] == 0
        # Count should be same
        assert count1 == count2
    
    def test_parent_relationship(self, db_session):
        """Test parent-child directory relationships."""
        from app.rag.indexing.directory_indexer import DirectoryIndexBuilder
        
        scan = ArchScanRun(status="running")
        db_session.add(scan)
        db_session.commit()
        
        index_data = {
            "scanned_files": [
                {"path": "app/llm/router.py", "bytes": 5000, "lines": 200},
            ]
        }
        
        builder = DirectoryIndexBuilder(db_session, scan.id)
        builder.load_from_index_json(index_data, r"D:\Orb")
        builder.save_to_db()
        
        # Find the llm directory
        llm_dir = db_session.query(ArchDirectoryIndex).filter(
            ArchDirectoryIndex.scan_id == scan.id,
            ArchDirectoryIndex.name == "llm"
        ).first()
        
        assert llm_dir is not None
        assert llm_dir.parent_id is not None
        
        # Parent should be app
        parent = db_session.query(ArchDirectoryIndex).filter_by(
            id=llm_dir.parent_id
        ).first()
        assert parent.name == "app"
    
    def test_aggregates(self, db_session):
        """Test file count and byte aggregation."""
        from app.rag.indexing.directory_indexer import DirectoryIndexBuilder
        
        scan = ArchScanRun(status="running")
        db_session.add(scan)
        db_session.commit()
        
        index_data = {
            "scanned_files": [
                {"path": "app/main.py", "bytes": 1000, "lines": 50},
                {"path": "app/utils.py", "bytes": 2000, "lines": 100},
            ]
        }
        
        builder = DirectoryIndexBuilder(db_session, scan.id)
        builder.load_from_index_json(index_data, r"D:\Orb")
        builder.save_to_db()
        
        # Find app directory
        app_dir = db_session.query(ArchDirectoryIndex).filter(
            ArchDirectoryIndex.scan_id == scan.id,
            ArchDirectoryIndex.name == "app"
        ).first()
        
        assert app_dir is not None
        assert app_dir.file_count == 2
        assert app_dir.total_bytes == 3000
        assert app_dir.total_lines == 150
