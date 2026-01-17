"""
Vector store abstraction.

Wraps sqlite-vec for similarity search.

CRITICAL: Must pass smoke test requirement:
- Insert 1 vector → query it back → assert top-1 hit matches
- If this fails, schema/query syntax is wrong

See: WP-1.4 for full implementation
"""

# Placeholder for Phase 1.4
# Will contain:
# - VectorSearchResult dataclass
# - SQLiteVecStore class with VERIFIED sqlite-vec integration
# - get_vector_store() singleton
# - reset_vector_store() for testing
