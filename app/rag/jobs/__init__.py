# FILE: app/rag/jobs/__init__.py
"""
RAG background jobs.

- embedding_job: Incremental, priority-based embedding generation
"""

from .embedding_job import (
    # Classes
    EmbeddingJob,
    EmbeddingJobStatus,
    EmbeddingPriority,
    # Functions
    get_embedding_status,
    get_embedding_stats,
    queue_embedding_job,
    run_embedding_job_sync,
    format_embedding_status_report,
    classify_chunk_priority,
    # Config
    EMBEDDING_AUTO_ENABLED,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    PRIORITY_PATTERNS,
    ARCHITECTURE_OUTPUT_DIR,
    STATUS_FILE,
)

__all__ = [
    # Classes
    "EmbeddingJob",
    "EmbeddingJobStatus",
    "EmbeddingPriority",
    # Functions
    "get_embedding_status",
    "get_embedding_stats",
    "queue_embedding_job",
    "run_embedding_job_sync",
    "format_embedding_status_report",
    "classify_chunk_priority",
    # Config
    "EMBEDDING_AUTO_ENABLED",
    "EMBEDDING_MODEL",
    "EMBEDDING_BATCH_SIZE",
    "PRIORITY_PATTERNS",
    "ARCHITECTURE_OUTPUT_DIR",
    "STATUS_FILE",
]
