# FILE: app/db.py
# Purpose: Database configuration and session management.
# Called-by: app.artefacts.router, app.astra_memory.decay_job, app.astra_memory.indexer, app.astra_memory.models (+154 more)
# Depends-on: app, app.astra_memory, app.astra_memory.models, app.astra_memory.preference_models (+34 more)
# Last-renovated: 2026-06-12
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

    # v3.0: Import Experience Database models (Unified Memory System)
    from app.experience import models as experience_models  # noqa: F401

    # v4.0: Import unified memory models (MemoryRouter, RAG entries, confidence)
    from app.memory import confidence_model  # noqa: F401
    from app.memory import rag_entries_model  # noqa: F401

    # v5.0: Import Content Pipeline models
    from app.content import models as content_models  # noqa: F401

    # v5.1: Import Settings models (API key management)
    from app.settings import models as settings_models  # noqa: F401

    # v6.0: Import Investments models (portfolio snapshots)
    from app.investments import models as investments_models  # noqa: F401

    # v7.0: Import Finance models (accounting, tax, budgeting)
    from app.finance import models as finance_models  # noqa: F401

    # v8.0: Import Lifestyle Engine models (weight, nutrition, workouts, goals)
    from app.lifestyle import models as lifestyle_models  # noqa: F401
    # Food overhaul Phase 6: saved-recipe tables
    from app.lifestyle import recipe_models as lifestyle_recipe_models  # noqa: F401

    # v9.0: Import Pipeline Transparency models (reasoning events, user corrections)
    from app.transparency import models as transparency_models  # noqa: F401

    # v10.0: Import Conversational Memory Layer models (sessions, summaries)
    from app.memory import conversation_models  # noqa: F401

    # v11.0: Import Project Registry models (multi-project architecture awareness)
    from app.project_registry import models as project_registry_models  # noqa: F401
    from app.project_registry import confidence_tracker  # noqa: F401

    # v12.0: Import Sentinel models (network security monitor, Phase 1)
    from app.sentinel import models as sentinel_models  # noqa: F401

    # v13.0: Import Vehicle models (OBD2 van health & mileage)
    from app.vehicle import models as vehicle_models  # noqa: F401

    # v4.0: create_all with checkfirst=True (default) handles tables.
    # SQLite may raise OperationalError for pre-existing indexes.
    # We catch and log these rather than crashing startup.
    import logging as _db_logging
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as _create_err:
        _logger = _db_logging.getLogger(__name__)
        if "already exists" in str(_create_err):
            _logger.debug(f"[init_db] Ignoring pre-existing schema element: {_create_err}")
        else:
            raise
    
    # v2.4: Run schema migrations for existing tables
    _migrate_arch_code_chunks_schema()

    # v4.0: Run memory module migrations (project columns, astra-core seed)
    from app.memory.migrations import run_memory_migrations
    run_memory_migrations()

    # v10.0: Run conversation memory migrations (session_id on messages)
    _migrate_conversation_memory_schema()

    # v11.0: Run pipeline logging overhaul migrations
    _migrate_pipeline_logging_schema()

    # Food overhaul Phase 2: per-item edit/quantity/micros columns on nutrition logs
    _migrate_nutrition_schema()

    # Food overhaul Phase 3: per-100g micronutrients on the product library
    _migrate_product_schema()

    # v4.1: Register domain stores with the MemoryRouter singleton
    from app.memory.startup import init_memory_system
    init_memory_system()


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
        # v2.5 (2026-06-12): scan provenance for staleness labelling.
        # Existing rows predate sandbox-scoped scans and came from the host
        # tree (the 2026-06-09 run contains orb-desktop chunks, which the
        # sandbox clone does not hold), so 'host' is the correct backfill.
        ("source", "TEXT", "'host'"),
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

    # arch_scan_runs.source (same provenance field at the run level)
    if "arch_scan_runs" in inspector.get_table_names():
        run_columns = {col["name"] for col in inspector.get_columns("arch_scan_runs")}
        if "source" not in run_columns:
            with engine.connect() as conn:
                try:
                    conn.execute(text(
                        "ALTER TABLE arch_scan_runs ADD COLUMN source TEXT DEFAULT 'host'"
                    ))
                    conn.commit()
                    logger.info("[db_migrate] Added column: arch_scan_runs.source")
                except Exception as e:
                    logger.warning(f"[db_migrate] Failed to add arch_scan_runs.source: {e}")


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


# =============================================================================
# SCHEMA MIGRATIONS (v10.0 — Conversational Memory Layer)
# =============================================================================

def _migrate_conversation_memory_schema():
    """
    Ensure the messages table has the session_id column.

    SQLAlchemy create_all() creates the new conversation_sessions and
    conversation_summaries tables, but won't add a column to the
    existing messages table. This migration does that.

    Safe to call multiple times (idempotent).
    v10.0: CONV-MEMORY-001 Phase 1.
    """
    import logging
    from sqlalchemy import inspect, text

    log = logging.getLogger(__name__)

    inspector = inspect(engine)
    if "messages" not in inspector.get_table_names():
        log.debug("[db_migrate] messages table doesn't exist yet, skipping")
        return

    existing = {col["name"] for col in inspector.get_columns("messages")}

    if "session_id" not in existing:
        with engine.connect() as conn:
            try:
                conn.execute(text(
                    "ALTER TABLE messages ADD COLUMN session_id INTEGER "
                    "REFERENCES conversation_sessions(id) ON DELETE SET NULL"
                ))
                conn.commit()
                log.info("[db_migrate] Added messages.session_id column")
                print("[db_migrate] Added messages.session_id column")
            except Exception as e:
                log.warning("[db_migrate] Failed to add session_id: %s", e)
    else:
        log.debug("[db_migrate] messages.session_id already exists")


# =============================================================================
# SCHEMA MIGRATIONS (v11.0 — Pipeline Logging Overhaul)
# =============================================================================

def _migrate_pipeline_logging_schema():
    """Add final_checkout_status and weaver_extraction columns to build_projects.

    Part of the Pipeline Logging & Reporting Overhaul:
    - final_checkout_status: tracks the new Final Checkout pipeline stage
    - weaver_extraction: JSON storing trimmed Weaver output (requirements,
      preferences, constraints) for the Brief tab

    Safe to call multiple times (idempotent).
    v11.0: PIPELINE-LOGGING-001.
    """
    import logging
    from sqlalchemy import inspect, text

    log = logging.getLogger(__name__)

    inspector = inspect(engine)
    if "build_projects" not in inspector.get_table_names():
        log.debug("[db_migrate] build_projects table doesn't exist yet, skipping")
        return

    existing = {col["name"] for col in inspector.get_columns("build_projects")}

    columns_to_add = [
        ("final_checkout_status", "VARCHAR(20)", "'pending'"),
        ("weaver_extraction", "TEXT", None),  # JSON stored as TEXT in SQLite
    ]

    for col_name, col_type, default_val in columns_to_add:
        if col_name not in existing:
            with engine.connect() as conn:
                try:
                    if default_val is not None:
                        sql = f"ALTER TABLE build_projects ADD COLUMN {col_name} {col_type} DEFAULT {default_val}"
                    else:
                        sql = f"ALTER TABLE build_projects ADD COLUMN {col_name} {col_type}"
                    conn.execute(text(sql))
                    conn.commit()
                    log.info("[db_migrate] Added build_projects.%s column", col_name)
                    print(f"[db_migrate] Added build_projects.{col_name} column")
                except Exception as e:
                    log.warning("[db_migrate] Failed to add %s: %s", col_name, e)
        else:
            log.debug("[db_migrate] build_projects.%s already exists", col_name)


# =============================================================================
# SCHEMA MIGRATIONS (Food overhaul Phase 2 — per-item edit / quantity / micros)
# =============================================================================

def _migrate_nutrition_schema():
    """Add the per-item edit columns to lifestyle_nutrition_logs.

    Food-module overhaul Phase 2. All columns are nullable and stored macros
    stay ABSOLUTE, so the daily-summary func.sum() aggregation is untouched.
    Safe to call multiple times (idempotent).
    """
    import logging
    from sqlalchemy import inspect, text

    log = logging.getLogger(__name__)

    inspector = inspect(engine)
    if "lifestyle_nutrition_logs" not in inspector.get_table_names():
        log.debug("[db_migrate] lifestyle_nutrition_logs table doesn't exist yet, skipping")
        return

    existing = {col["name"] for col in inspector.get_columns("lifestyle_nutrition_logs")}

    columns_to_add = [
        ("quantity_g", "FLOAT"),
        ("per_100g_json", "TEXT"),
        ("micros_json", "TEXT"),
        ("food_product_id", "INTEGER"),
        ("estimate_note", "TEXT"),
    ]

    for col_name, col_type in columns_to_add:
        if col_name not in existing:
            with engine.connect() as conn:
                try:
                    conn.execute(text(
                        f"ALTER TABLE lifestyle_nutrition_logs ADD COLUMN {col_name} {col_type}"
                    ))
                    conn.commit()
                    log.info("[db_migrate] Added lifestyle_nutrition_logs.%s column", col_name)
                    print(f"[db_migrate] Added lifestyle_nutrition_logs.{col_name} column")
                except Exception as e:
                    log.warning("[db_migrate] Failed to add %s: %s", col_name, e)
        else:
            log.debug("[db_migrate] lifestyle_nutrition_logs.%s already exists", col_name)


def _migrate_product_schema():
    """Add micros_json (per-100g micronutrients) to lifestyle_food_products.

    Food-module overhaul Phase 3. The product table is lazy-created
    (ensure_product_table), so this only runs when it already exists; a fresh
    table is created with the column by create_all. Idempotent.
    """
    import logging
    from sqlalchemy import inspect, text

    log = logging.getLogger(__name__)

    inspector = inspect(engine)
    if "lifestyle_food_products" not in inspector.get_table_names():
        log.debug("[db_migrate] lifestyle_food_products table doesn't exist yet, skipping")
        return

    existing = {col["name"] for col in inspector.get_columns("lifestyle_food_products")}
    if "micros_json" not in existing:
        with engine.connect() as conn:
            try:
                conn.execute(text(
                    "ALTER TABLE lifestyle_food_products ADD COLUMN micros_json TEXT"
                ))
                conn.commit()
                log.info("[db_migrate] Added lifestyle_food_products.micros_json column")
                print("[db_migrate] Added lifestyle_food_products.micros_json column")
            except Exception as e:
                log.warning("[db_migrate] Failed to add micros_json: %s", e)
    else:
        log.debug("[db_migrate] lifestyle_food_products.micros_json already exists")



