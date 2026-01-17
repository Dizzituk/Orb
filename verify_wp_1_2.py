#!/usr/bin/env python3
"""
WP-1.2 Verification Script

Verifies that RAG models are correctly deployed and tables created.

Run this AFTER:
1. Copying models.py to D:\\Orb\\app\\rag\\
2. Adding import to app/db.py
3. Restarting server

Usage:
    cd D:\\Orb
    python verify_wp_1_2.py
"""

import sys
from pathlib import Path


def verify_models_file():
    """Verify models.py exists and has content."""
    print("=" * 60)
    print("WP-1.2 Models File Verification")
    print("=" * 60)
    
    models_path = Path("app/rag/models.py")
    if not models_path.exists():
        print(f"❌ {models_path} does not exist!")
        return False
    
    content = models_path.read_text()
    
    checks = [
        ("RAGFile", "class RAGFile(Base):"),
        ("RAGChunk", "class RAGChunk(Base):"),
        ("RAGIndexRun", "class RAGIndexRun(Base):"),
        ("Query helpers", "def get_file_by_path"),
        ("FK to architecture", "architecture_file_id"),
        ("Embedding metadata", "embedding_model"),
        ("Scan FK", "scan_id"),
    ]
    
    all_found = True
    for name, pattern in checks:
        if pattern in content:
            print(f"✓ {name} found")
        else:
            print(f"❌ {name} not found")
            all_found = False
    
    return all_found


def verify_imports():
    """Verify models can be imported."""
    print("\n" + "=" * 60)
    print("Import Verification")
    print("=" * 60)
    
    try:
        from app.rag import models
        print("✓ import app.rag.models")
    except Exception as e:
        print(f"❌ import app.rag.models failed: {e}")
        return False
    
    # Check classes exist
    classes = ["RAGFile", "RAGChunk", "RAGIndexRun"]
    for cls_name in classes:
        if hasattr(models, cls_name):
            print(f"✓ {cls_name} class exists")
        else:
            print(f"❌ {cls_name} class not found")
            return False
    
    # Check helper functions
    helpers = [
        "get_file_by_path",
        "get_chunks_by_file_id",
        "get_index_stats",
        "count_files_by_project",
    ]
    
    for func_name in helpers:
        if hasattr(models, func_name):
            print(f"✓ {func_name}() helper exists")
        else:
            print(f"❌ {func_name}() helper not found")
            return False
    
    return True


def verify_tables_created():
    """Verify tables exist in database."""
    print("\n" + "=" * 60)
    print("Database Tables Verification")
    print("=" * 60)
    
    try:
        from app.db import engine
        from sqlalchemy import inspect
        
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        print(f"Total tables in database: {len(tables)}")
        
        rag_tables = ["rag_files", "rag_chunks", "rag_index_runs"]
        
        all_exist = True
        for table_name in rag_tables:
            if table_name in tables:
                print(f"✓ {table_name} table exists")
                
                # Check columns
                columns = [col["name"] for col in inspector.get_columns(table_name)]
                print(f"  Columns: {len(columns)}")
                
                # Check indexes
                indexes = inspector.get_indexes(table_name)
                print(f"  Indexes: {len(indexes)}")
            else:
                print(f"❌ {table_name} table NOT FOUND")
                all_exist = False
        
        return all_exist
    
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        print("\nPossible causes:")
        print("1. Import not added to app/db.py init_db()")
        print("2. Server not restarted after adding import")
        print("3. Database path incorrect")
        return False


def verify_relationships():
    """Verify model relationships work."""
    print("\n" + "=" * 60)
    print("Relationship Verification")
    print("=" * 60)
    
    try:
        from app.rag.models import RAGFile, RAGChunk, RAGIndexRun
        
        # Check RAGFile relationships
        if hasattr(RAGFile, "architecture_file"):
            print("✓ RAGFile → architecture_file relationship")
        else:
            print("❌ RAGFile → architecture_file relationship missing")
            return False
        
        if hasattr(RAGFile, "chunks"):
            print("✓ RAGFile → chunks relationship")
        else:
            print("❌ RAGFile → chunks relationship missing")
            return False
        
        # Check RAGChunk relationships
        if hasattr(RAGChunk, "file"):
            print("✓ RAGChunk → file relationship")
        else:
            print("❌ RAGChunk → file relationship missing")
            return False
        
        # Check RAGIndexRun relationships
        if hasattr(RAGIndexRun, "scan_run"):
            print("✓ RAGIndexRun → scan_run relationship")
        else:
            print("❌ RAGIndexRun → scan_run relationship missing")
            return False
        
        return True
    
    except Exception as e:
        print(f"❌ Relationship check failed: {e}")
        return False


def verify_properties():
    """Verify model property methods."""
    print("\n" + "=" * 60)
    print("Property Methods Verification")
    print("=" * 60)
    
    try:
        from app.rag.models import RAGChunk, RAGIndexRun
        
        # Check RAGChunk properties
        chunk_props = ["has_embedding", "line_range"]
        for prop in chunk_props:
            # Check if it's a property descriptor
            if hasattr(RAGChunk, prop):
                attr = getattr(RAGChunk, prop)
                if isinstance(attr, property):
                    print(f"✓ RAGChunk.{prop} property")
                else:
                    print(f"⚠️  RAGChunk.{prop} exists but not a property")
            else:
                print(f"❌ RAGChunk.{prop} missing")
                return False
        
        # Check RAGIndexRun properties
        run_props = ["duration_seconds", "success_rate"]
        for prop in run_props:
            if hasattr(RAGIndexRun, prop):
                attr = getattr(RAGIndexRun, prop)
                if isinstance(attr, property):
                    print(f"✓ RAGIndexRun.{prop} property")
                else:
                    print(f"⚠️  RAGIndexRun.{prop} exists but not a property")
            else:
                print(f"❌ RAGIndexRun.{prop} missing")
                return False
        
        return True
    
    except Exception as e:
        print(f"❌ Property check failed: {e}")
        return False


def main():
    """Run all verifications."""
    print("\n")
    
    results = []
    results.append(("Models File", verify_models_file()))
    results.append(("Imports", verify_imports()))
    results.append(("Tables Created", verify_tables_created()))
    results.append(("Relationships", verify_relationships()))
    results.append(("Properties", verify_properties()))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✅ WP-1.2 VERIFICATION PASSED")
        print("\nTables created:")
        print("  - rag_files (file metadata + indexing state)")
        print("  - rag_chunks (text chunks + embedding metadata)")
        print("  - rag_index_runs (index job history)")
        print("\nNext steps:")
        print("1. Run tests: pytest tests\\test_rag_models.py -v")
        print("2. Review WP-1.2_README.md")
        print("3. Proceed to WP-1.3 (Pydantic Schemas)")
        return 0
    else:
        print("\n❌ WP-1.2 VERIFICATION FAILED")
        print("\nFix the errors above before proceeding.")
        print("\nCommon issues:")
        print("1. Import not added to app/db.py init_db()")
        print("   Fix: Add 'from app.rag import models as rag_models'")
        print("2. Server not restarted")
        print("   Fix: Stop and restart the server")
        print("3. models.py not copied")
        print("   Fix: Copy app/rag/models.py from implementation")
        return 1


if __name__ == "__main__":
    sys.exit(main())
