#!/usr/bin/env python3
"""
WP-1.1 Verification Script

Verifies that the RAG module structure is correctly deployed.

Run this AFTER copying files to D:\Orb\app\rag\

Usage:
    cd D:\Orb
    python verify_wp_1_1.py
"""

import sys
from pathlib import Path


def verify_structure():
    """Verify all files exist."""
    print("=" * 60)
    print("WP-1.1 Structure Verification")
    print("=" * 60)
    
    base_path = Path("app/rag")
    if not base_path.exists():
        print(f"❌ ERROR: {base_path} does not exist!")
        print(f"   Make sure you're running from D:\\Orb directory")
        return False
    
    required_files = [
        "app/rag/__init__.py",
        "app/rag/config.py",
        "app/rag/models.py",
        "app/rag/schemas.py",
        "app/rag/vector_store.py",
        "app/rag/scanner.py",
        "app/rag/embedder.py",
        "app/rag/indexer.py",
        "app/rag/retriever.py",
        "app/rag/answerer.py",
        "app/rag/prompts.py",
        "app/rag/router.py",
        "app/rag/service.py",
        "app/rag/chunkers/__init__.py",
        "app/rag/chunkers/base.py",
        "app/rag/chunkers/window.py",
        "app/rag/chunkers/python_ast.py",
        "app/rag/chunkers/typescript_ast.py",
    ]
    
    all_exist = True
    for file_path in required_files:
        p = Path(file_path)
        if p.exists():
            print(f"✓ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_exist = False
    
    return all_exist


def verify_imports():
    """Verify imports work."""
    print("\n" + "=" * 60)
    print("Import Verification")
    print("=" * 60)
    
    try:
        import app.rag
        print("✓ import app.rag")
    except Exception as e:
        print(f"❌ import app.rag failed: {e}")
        return False
    
    try:
        import app.rag.config
        print("✓ import app.rag.config")
    except Exception as e:
        print(f"❌ import app.rag.config failed: {e}")
        return False
    
    try:
        from app.rag.config import INDEX_SCAN_SCOPE, DATA_SOURCE
        print(f"✓ INDEX_SCAN_SCOPE = {INDEX_SCAN_SCOPE}")
        print(f"✓ DATA_SOURCE = {DATA_SOURCE}")
        
        if INDEX_SCAN_SCOPE != "both":
            print(f"⚠️  Warning: INDEX_SCAN_SCOPE is '{INDEX_SCAN_SCOPE}', expected 'both'")
        
        if DATA_SOURCE != "architecture_scan":
            print(f"⚠️  Warning: DATA_SOURCE is '{DATA_SOURCE}', expected 'architecture_scan'")
    except Exception as e:
        print(f"❌ Config constants failed: {e}")
        return False
    
    # Check placeholder imports (should not fail even though they're empty)
    placeholders = [
        "app.rag.models",
        "app.rag.schemas",
        "app.rag.vector_store",
        "app.rag.chunkers.base",
    ]
    
    for module_name in placeholders:
        try:
            __import__(module_name)
            print(f"✓ import {module_name} (placeholder)")
        except Exception as e:
            print(f"❌ import {module_name} failed: {e}")
            return False
    
    return True


def verify_config():
    """Verify critical config values."""
    print("\n" + "=" * 60)
    print("Config Verification")
    print("=" * 60)
    
    try:
        from app.rag import config
        
        checks = [
            ("DATA_SOURCE", "architecture_scan", config.DATA_SOURCE),
            ("INDEX_SCAN_SCOPE", "both", config.INDEX_SCAN_SCOPE),
            ("EMBEDDING_MODEL", "text-embedding-3-small", config.EMBEDDING_MODEL),
            ("EMBEDDING_DIMENSIONS", 1536, config.EMBEDDING_DIMENSIONS),
            ("DEFAULT_TOP_K", 10, config.DEFAULT_TOP_K),
        ]
        
        all_correct = True
        for name, expected, actual in checks:
            if actual == expected:
                print(f"✓ {name} = {actual}")
            else:
                print(f"❌ {name} = {actual} (expected: {expected})")
                all_correct = False
        
        # Check that ALLOWED_USER_DIRS is NOT in config (we removed it)
        if hasattr(config, 'ALLOWED_USER_DIRS'):
            print("⚠️  Warning: ALLOWED_USER_DIRS found in config (should be removed)")
            print("   RAG should trust scan scope, not apply additional filters")
        else:
            print("✓ ALLOWED_USER_DIRS correctly removed (trusts scan scope)")
        
        return all_correct
    
    except Exception as e:
        print(f"❌ Config verification failed: {e}")
        return False


def main():
    """Run all verifications."""
    print("\n")
    
    results = []
    results.append(("Structure", verify_structure()))
    results.append(("Imports", verify_imports()))
    results.append(("Config", verify_config()))
    
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
        print("\n✅ WP-1.1 VERIFICATION PASSED")
        print("\nNext steps:")
        print("1. Review DB_WIRING_INSTRUCTIONS.md")
        print("2. Proceed to WP-1.2 (Database Models)")
        return 0
    else:
        print("\n❌ WP-1.1 VERIFICATION FAILED")
        print("\nFix the errors above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
