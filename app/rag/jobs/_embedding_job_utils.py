from __future__ import annotations
import os
from pathlib import Path
from sqlalchemy.orm import Session
from typing import Callable, Optional


SQLITE_LOCK_MAX_RETRIES = 10

SQLITE_LOCK_INITIAL_BACKOFF = 0.25  # seconds

SQLITE_LOCK_MAX_BACKOFF = 8.0  # seconds

def _is_sqlite_lock_error(exc: Exception) -> bool:
    """Check if exception is a SQLite database lock error."""
    error_str = str(exc).lower()
    return "database is locked" in error_str or "database_is_locked" in error_str

EMBEDDING_BATCH_SIZE = int(os.getenv("ORB_EMBEDDING_BATCH_SIZE", "20"))

EMBEDDING_RATE_LIMIT_DELAY = float(os.getenv("ORB_EMBEDDING_RATE_DELAY", "0.5"))  # seconds between batches

ARCHITECTURE_OUTPUT_DIR = Path(r"D:\Orb\.architecture")

def run_embedding_job_sync(
    db_session_factory: Callable[[], Session],
    scan_id: Optional[int] = None,
) -> EmbeddingJobStatus:
    """
    Run embedding job synchronously (for manual trigger).
    
    Returns:
        Final job status
    """
    job = EmbeddingJob(db_session_factory, scan_id=scan_id)
    return job.run_sync()


# Auto-generated re-exports for symbols in numbered _utils files
_REEXPORT_MAP = {
    "EMBEDDING_AUTO_ENABLED": "_embedding_job_utils_4",
    "EMBEDDING_MODEL": "_embedding_job_utils_4",
    "STATUS_FILE": "_embedding_job_utils_4",
    "classify_chunk_priority": "_embedding_job_utils_4",
    "compute_content_hash": "_embedding_job_utils_4",
    "format_embedding_status_report": "_embedding_job_utils_4",
    "get_embedding_stats": "_embedding_job_utils_4",
    "queue_embedding_job": "_embedding_job_utils_4",
    "EmbeddingPriority": "_embedding_job_utils_5",
    "get_embedding_status": "_embedding_job_utils_5",
}

def __getattr__(name):
    if name in _REEXPORT_MAP:
        import importlib
        mod = importlib.import_module(f"app.rag.jobs.{_REEXPORT_MAP[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
