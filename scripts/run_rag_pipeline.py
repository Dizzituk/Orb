#!/usr/bin/env python3
"""
Run RAG indexing pipeline.

Usage:
    python scripts/run_rag_pipeline.py [scan_dir]
    python scripts/run_rag_pipeline.py  # Uses default
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Initialize encryption before any DB operations
from app.crypto import require_master_key_or_exit
require_master_key_or_exit()

from app.db import SessionLocal
from app.rag.pipeline import run_rag_pipeline

# Default output directory (matches FULL_ARCHMAP_OUTPUT_DIR in zobie_tools.py)
DEFAULT_SCAN_DIR = os.getenv(
    "ZOBIE_OUTPUT_DIR",
    r"D:\Orb\.architecture"
)


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG indexing pipeline"
    )
    parser.add_argument(
        "scan_dir",
        nargs="?",
        default=DEFAULT_SCAN_DIR,
        help="Directory with zobie output"
    )
    parser.add_argument(
        "--project-id",
        type=int,
        default=0,
        help="Project ID for embeddings"
    )
    args = parser.parse_args()
    
    if not os.path.isdir(args.scan_dir):
        print(f"Error: Directory not found: {args.scan_dir}")
        sys.exit(1)
    
    print("RAG Pipeline")
    print("=" * 50)
    print(f"Scan dir: {args.scan_dir}")
    print(f"Project ID: {args.project_id}")
    print()
    
    db = SessionLocal()
    try:
        stats = run_rag_pipeline(
            db,
            scan_dir=args.scan_dir,
            project_id=args.project_id,
        )
        
        print()
        print("Pipeline Complete!")
        print(f"  Scan ID: {stats['scan_id']}")
        print(f"  Directories: {stats['directories']}")
        print(f"  Chunks: {stats['chunks']}")
        print(f"  Descriptors: {stats['descriptors']}")
        print(f"  Embeddings: {stats['embeddings']}")
        
        if stats.get("errors"):
            print(f"  Errors: {len(stats['errors'])}")
            for err in stats["errors"]:
                print(f"    - {err}")
                
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
