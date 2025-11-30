# FILE: app/db.py
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

Base = declarative_base()


def get_db():
    """FastAPI dependency that yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Call once at startup."""
    # Import models so Base.metadata knows about them
    from app.memory import models  # noqa: F401
    Base.metadata.create_all(bind=engine)