# FILE: app/db.py
"""
Database configuration and session management.

Uses SQLite by default with ORB_DATABASE_URL override.

v2.4: Added get_db_session() for background threads (embedding jobs)
v2.3: Added RAG tables (WP-1.2) + enabled SQLite foreign key enforcement
"""
import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.engine import Engine

# Database path: ./data/orb_memory.db relative to project root
# Override with ORB_DATABASE_URL env var if needed
DATABASE_URL = os.getenv("ORB_DATABASE_URL", "sqlite:///./data/orb_memory.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False,  # Required for SQLite
        "timeout": 30,  # v2.5: Wait up to 30s for locked DB (Python sqlite3 level)
    },
    echo=False,  # Set True to log SQL statements for debugging
)


# CRITICAL: Enable foreign key enforcement and performance pragmas for SQLite
# v2.5: Added WAL mode + busy_timeout for background job concurrency
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """
    Configure SQLite pragmas for every new connection.
    
    Sets:
    - foreign_keys=ON: Enforce FK constraints (required for CASCADE deletes)
    - journal_mode=WAL: Write-Ahead Logging for better concurrency
    - busy_timeout=30000: Wait 30s for locks before failing (SQLite level)
    - synchronous=NORMAL: Safe durability with better throughput
    
    v2.5: Added WAL + busy_timeout to fix embedding job lock contention
    """
    if "sqlite" in DATABASE_URL.lower():
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")  # 30 seconds in milliseconds
        cursor.execute("PRAGMA synchronous=NORMAL")  # Safe + faster than FULL
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Shared declarative base for all ORM models
Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session():
    """
    Non-generator DB session for use outside FastAPI request context.
    
    Use this for:
    - Background threads (embedding jobs, etc.)
    - Direct DB access in scripts
    - Code that needs a session factory callable
    
    IMPORTANT: Caller is responsible for closing the session!
    
    Usage as session factory:
        def db_session_factory():
            return get_db_session()
        
        job = EmbeddingJob(db_session_factory)
    
    Usage as context manager pattern:
        db = get_db_session()
        try:
            # ... use db ...
        finally:
            db.close()
    
    v2.4: Added for embedding_stream.py background job support.
    """
    return SessionLocal()


def init_db():
    """
    Create all tables. Call once at startup.
    
    v2.4: Added schema migration for arch_code_chunks embedding columns.
    v2.3: Added RAG tables for semantic search and Q&A.
    v2.2: Added architecture scan tables (scan sandbox, update architecture).
    v2.1: Added specs module tables (Weaver/Spec Gate).
    v2.0: Added preference and confidence system tables.
    """
    # Import models so Base.metadata knows about them
    from app.memory import models  # noqa: F401
    from app.embeddings import models as embedding_models  # noqa: F401
    from app.jobs import models as job_models  # noqa: F401
    from app.pot_spec import models as pot_spec_models  # noqa: F401
    
    # v2.0: Import ASTRA memory models (both original and preference system)
    from app.astra_memory import models as astra_models  # noqa: F401
    from app.astra_memory import preference_models  # noqa: F401
    
    # v2.1: Import specs models (Weaver/Spec Gate)
    from app.specs import models as spec_models  # noqa: F401
    
    # v2.2: Import architecture scan models (scan sandbox, update architecture)
    from app.memory import architecture_models  # noqa: F401
    
    # v2.3: Import RAG models (semantic search and Q&A)
    from app.rag import models as rag_models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    
    # v2.4: Run schema migrations for existing tables
    _migrate_arch_code_chunks_schema()


# =============================================================================
# SCHEMA MIGRATIONS (v2.4)
# =============================================================================

def _migrate_arch_code_chunks_schema():
    """
    Ensure arch_code_chunks table has all required columns for embedding support.
    
    This is a safe, idempotent migration that:
    - Only adds missing columns (never deletes)
    - Works on existing databases with data
    - Can be run multiple times safely
    
    Required columns (v1.1 embedding support):
    - content_hash TEXT (nullable) - SHA-256 of signature+docstring+name
    - embedded INTEGER DEFAULT 0 - Boolean flag
    - embedding_model TEXT (nullable) - e.g. "text-embedding-3-small"
    - embedded_at DATETIME (nullable) - When embedding was generated
    - embedded_content_hash TEXT (nullable) - Hash used when embedding was created
    
    v2.4: Added for embedding_job.py compatibility.
    """
    import logging
    from sqlalchemy import inspect, text
    
    logger = logging.getLogger(__name__)
    
    # Check if table exists
    inspector = inspect(engine)
    if "arch_code_chunks" not in inspector.get_table_names():
        # Table doesn't exist yet - will be created by create_all()
        logger.debug("[db_migrate] arch_code_chunks table doesn't exist yet, skipping migration")
        return
    
    # Get existing columns
    existing_columns = {col["name"] for col in inspector.get_columns("arch_code_chunks")}
    
    # Define columns to add (name, SQL type, default)
    # SQLite doesn't support all ALTER TABLE features, so we use simple types
    columns_to_add = [
        ("content_hash", "TEXT", None),
        ("embedded", "INTEGER", "0"),  # SQLite stores booleans as integers
        ("embedding_model", "TEXT", None),
        ("embedded_at", "DATETIME", None),
        ("embedded_content_hash", "TEXT", None),
    ]
    
    # Add missing columns
    added_columns = []
    with engine.connect() as conn:
        for col_name, col_type, default_value in columns_to_add:
            if col_name not in existing_columns:
                # Build ALTER TABLE statement
                if default_value is not None:
                    sql = f"ALTER TABLE arch_code_chunks ADD COLUMN {col_name} {col_type} DEFAULT {default_value}"
                else:
                    sql = f"ALTER TABLE arch_code_chunks ADD COLUMN {col_name} {col_type}"
                
                try:
                    conn.execute(text(sql))
                    conn.commit()
                    added_columns.append(col_name)
                    logger.info(f"[db_migrate] Added column: arch_code_chunks.{col_name}")
                except Exception as e:
                    # Column might already exist (race condition) or other error
                    logger.warning(f"[db_migrate] Failed to add column {col_name}: {e}")
    
    if added_columns:
        print(f"[db_migrate] arch_code_chunks schema updated: added {added_columns}")
    else:
        logger.debug("[db_migrate] arch_code_chunks schema is up to date")


def ensure_embedding_schema():
    """
    Public helper to ensure embedding schema is ready.
    
    Can be called before embedding operations to guarantee schema is migrated.
    Safe to call multiple times (idempotent).
    
    Usage:
        from app.db import ensure_embedding_schema
        ensure_embedding_schema()  # Before embedding operations
    """
    _migrate_arch_code_chunks_schema()
