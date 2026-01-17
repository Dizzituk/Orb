"""
RAG system configuration.

All tunables in one place for easy adjustment.

CRITICAL DESIGN DECISIONS:
1. Source of Truth: Reads FROM architecture_file_content table (no filesystem access)
2. Scope Alignment: Indexes whatever the scan indexed (no additional filters)
3. Repo-relative Paths: All paths stored as-is from scan (no C:\ dependencies)
4. Read-only: No file modifications, no destructive operations
5. Ignore Rules: Inherits from scan - only indexes files that passed scan filters
"""

import os
from typing import Set

# ============================================================================
# SOURCE OF TRUTH
# ============================================================================

# RAG reads FROM existing architecture scan tables
# Single source of truth: architecture_file_content table
# No filesystem scanning, no file duplication
DATA_SOURCE = "architecture_scan"  # NOT filesystem

# ============================================================================
# INDEXING SCOPE
# ============================================================================

# Which scan scope to index from
# "code" = D:\Orb + D:\orb-desktop (UPDATE ARCHITECTURE)
# "sandbox" = C:\Users\dizzi + D:\ (SCAN SANDBOX)  
# "both" = Index both scopes
INDEX_SCAN_SCOPE = "both"

# Project tagging based on path patterns (zone mapping from scan)
# These map scan "zone" field to RAG "project_tag"
ZONE_TO_PROJECT_TAG = {
    "backend": "backend",
    "frontend": "frontend",
    "tools": "tools",
    "user": "sandbox",
    "repo": "other",
    "other": "other",
}

# ============================================================================
# CHUNKING STRATEGY
# ============================================================================

# AST chunking for code files
AST_MAX_CHUNK_TOKENS: int = 1500
AST_MIN_CHUNK_TOKENS: int = 50

# Window chunking (fallback for non-code or when AST fails)
WINDOW_CHUNK_TOKENS: int = 800
WINDOW_OVERLAP_TOKENS: int = 100

# Doc chunking (markdown, text files)
DOC_CHUNK_TOKENS: int = 600
DOC_OVERLAP_TOKENS: int = 80

# Files to skip even if in scan (defensive - scan should handle this)
# These are failsafes; scan should already filter these
SKIP_SENSITIVE_PATTERNS: Set[str] = {
    ".env", ".env.local", ".env.production", ".env.development",
    "secrets.json", "credentials.json", "config.secret.json",
    "id_rsa", "id_ed25519", "id_ecdsa",
}

# File types that get AST chunking (others get window chunking)
AST_CHUNKABLE_LANGUAGES: Set[str] = {
    "python",
    "typescript", 
    "javascript",
}

# ============================================================================
# EMBEDDINGS
# ============================================================================

# OpenAI embedding model (from routing_policy.json)
EMBEDDING_MODEL: str = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS: int = 1536  # For text-embedding-3-small
EMBEDDING_BATCH_SIZE: int = 100

# Cost control
MAX_EMBEDDING_REQUESTS_PER_MINUTE: int = 3000  # OpenAI tier limit

# ============================================================================
# RETRIEVAL
# ============================================================================

# Search parameters
DEFAULT_TOP_K: int = 10
MAX_TOP_K: int = 50
SIMILARITY_THRESHOLD: float = 0.3  # Minimum cosine similarity to include

# Result ranking
ENABLE_DEDUPLICATION: bool = True
DEDUP_OVERLAP_THRESHOLD: float = 0.8  # 80% line overlap = duplicate

ENABLE_SYMBOL_BOOST: bool = True
SYMBOL_BOOST_MULTIPLIER: float = 1.3  # Boost score by 30% for symbol matches

ENABLE_DIVERSITY: bool = True
MAX_CHUNKS_PER_FILE: int = 3  # Limit chunks from same file for diversity

# ============================================================================
# LLM TIERS (from routing_policy.json)
# ============================================================================

# Tier selection for grounded Q&A
LLM_TIER_SMALL: str = "gpt-4o"  # GPT-5 Mini placeholder (use GPT-4o for now)
LLM_TIER_MEDIUM: str = "gpt-4o"  # GPT-5.2 placeholder  
LLM_TIER_LARGE: str = "claude-sonnet-4-20250514"  # Claude Opus 4.5 placeholder

# Provider mapping
TIER_TO_PROVIDER = {
    "small": "openai",
    "medium": "openai",
    "large": "anthropic",
}

# Token limits per tier (context budget for chunks + answer)
TIER_TOKEN_LIMITS = {
    "small": 4000,
    "medium": 8000,
    "large": 16000,
}

# Auto-tier selection keywords
TIER_LARGE_KEYWORDS = {
    "architecture", "design", "refactor", "security", "audit",
    "migration", "performance", "scalability", "critical",
}

TIER_MEDIUM_KEYWORDS = {
    "explain", "how does", "why", "compare", "difference",
    "integration", "flow", "pipeline",
}

# ============================================================================
# DATABASE
# ============================================================================

# Vector store database path
# Separate from main orb_memory.db for performance isolation
VECTOR_DB_PATH: str = os.getenv("RAG_VECTOR_DB", "./data/rag_vectors.db")

# Main database connection (for reading architecture scan tables)
# This should match app/db.py DATABASE_URL
MAIN_DB_URL: str = os.getenv("ORB_DATABASE_URL", "sqlite:///./data/orb_memory.db")

# ============================================================================
# INDEXING BEHAVIOR
# ============================================================================

# Incremental indexing
ENABLE_INCREMENTAL_INDEX: bool = True
INDEX_VERSION: int = 1  # Bump to force full re-index

# Performance
MAX_FILES_PER_BATCH: int = 50  # Process in batches to avoid memory issues
CHUNK_COMMIT_BATCH_SIZE: int = 100  # Commit chunks in batches

# Logging
LOG_PROGRESS_EVERY_N_FILES: int = 10
LOG_EMBEDDING_PROGRESS: bool = True

# Safety limits
MAX_FILE_SIZE_TOKENS: int = 50000  # Skip files larger than this (rare)
MAX_CHUNKS_PER_FILE: int = 200  # Defensive limit

# ============================================================================
# SMOKE TEST REQUIREMENT (WP-1.4)
# ============================================================================

# sqlite-vec verification: Must pass smoke test in test_rag_vector_store.py
# Test: insert 1 vector → query it back → assert top-1 hit matches
# If this fails, vector store is misconfigured (wrong schema/query syntax)
REQUIRE_VECTOR_STORE_SMOKE_TEST: bool = True
