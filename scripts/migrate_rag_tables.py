#!/usr/bin/env python3
"""
Create RAG architecture tables.

Non-destructive - only creates new tables.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db import engine, Base
from app.rag.models import ArchScanRun, ArchDirectoryIndex, ArchCodeChunk


def migrate():
    """Create RAG architecture tables."""
    print("Creating RAG architecture tables (additive only)...")
    
    Base.metadata.create_all(
        engine,
        tables=[
            ArchScanRun.__table__,
            ArchDirectoryIndex.__table__,
            ArchCodeChunk.__table__,
        ]
    )
    
    print("✓ arch_scan_runs")
    print("✓ arch_directory_index")
    print("✓ arch_code_chunks")
    print("\nMigration complete!")


if __name__ == "__main__":
    migrate()
