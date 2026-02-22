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
