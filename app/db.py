# FILE: app/db.py
"""
Database configuration and session management.

Uses SQLite by default with ORB_DATABASE_URL override.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Database path: ./data/orb_memory.db relative to project root
# Override with ORB_DATABASE_URL env var if needed
DATABASE_URL = os.getenv("ORB_DATABASE_URL", "sqlite:///./data/orb_memory.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for SQLite
    echo=False,  # Set True to log SQL statements for debugging
)

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


def init_db():
    """
    Create all tables. Call once at startup.
    
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

    Base.metadata.create_all(bind=engine)
