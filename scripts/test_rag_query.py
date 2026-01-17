#!/usr/bin/env python3
"""
Test RAG query.

Usage:
    python scripts/test_rag_query.py "How does streaming work?"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.db import SessionLocal
from app.rag.retrieval.context_assembler import retrieve_architecture_context
from app.rag.retrieval.arch_search import search_architecture


def main():
    parser = argparse.ArgumentParser(description="Test RAG query")
    parser.add_argument("query", help="Query string")
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    
    db = SessionLocal()
    try:
        print(f"Query: {args.query}")
        print("=" * 60)
        
        if args.verbose:
            response = search_architecture(db, args.query)
            print(f"Intent: {response.intent_depth}")
            print(f"Total searched: {response.total_searched}")
            print(f"Directories: {response.directories_found}")
            print(f"Chunks: {response.chunks_found}")
            print()
            print("Top Results:")
            for i, r in enumerate(response.results[:10]):
                print(f"  {i+1}. [{r.score:.3f}] {r.source_type}: {r.name}")
                print(f"       {r.canonical_path}")
            print()
        
        context = retrieve_architecture_context(
            db,
            args.query,
            max_tokens=args.max_tokens,
        )
        
        print("Context:")
        print("-" * 60)
        print(context if context else "(No results)")
        
    finally:
        db.close()


if __name__ == "__main__":
    main()
