# FILE: app/rag/jobs/_embedding_batch.py
# Purpose: Embedding batch processing: embed + retry logic.
# Called-by: app.rag.jobs.embedding_job
# Depends-on: app.embeddings.service, app.llm.clients, app.rag.jobs._embedding_job_utils_6, app.rag.jobs._embedding_job_utils_7 (+2 more)
# Last-renovated: 2026-06-11
"""
Embedding batch processing: embed + retry logic.

Extracted from embedding_job.py. These methods are called by EmbeddingJob
but live here for file-size modularity.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Callable, List, TYPE_CHECKING

from sqlalchemy.orm import Session

from app.rag.jobs._embedding_job_utils_6 import (
    EMBEDDING_BATCH_SIZE,
    SQLITE_LOCK_INITIAL_BACKOFF,
    SQLITE_LOCK_MAX_BACKOFF,
    SQLITE_LOCK_MAX_RETRIES,
    _is_sqlite_lock_error,
)
from app.rag.jobs._embedding_job_utils_7 import EMBEDDING_MODEL, compute_content_hash

if TYPE_CHECKING:
    from app.rag.jobs.embedding_job import EmbeddingJobStatus

logger = logging.getLogger(__name__)


def embed_batch(
    db: Session,
    chunks: List,
    status: "EmbeddingJobStatus",
) -> int:
    """Embed a batch of chunks. Returns number successfully processed."""
    try:
        from app.llm.clients import get_embeddings
    except ImportError as e:
        logger.error("[embedding_job] Failed to import get_embeddings: %s", e)
        raise

    try:
        from app.embeddings.service import store_embedding, embedding_exists
    except ImportError as e:
        logger.error("[embedding_job] Failed to import store_embedding: %s", e)
        raise

    # Prefilter: skip chunks that already have embeddings
    chunks_to_embed = []
    skipped_existing = 0
    now = datetime.utcnow()

    for chunk in chunks:
        if embedding_exists(db=db, project_id=0, source_type="arch_code_chunk", source_id=chunk.id):
            chunk.embedded = True
            chunk.embedding_model = EMBEDDING_MODEL
            chunk.embedded_at = now
            chunk.embedded_content_hash = chunk.content_hash
            skipped_existing += 1
            if hasattr(status, 'skipped_chunks'):
                status.skipped_chunks += 1
        else:
            chunks_to_embed.append(chunk)

    if skipped_existing > 0:
        logger.info("[embedding_job] Skipped %d chunks (already exist)", skipped_existing)

    if not chunks_to_embed:
        logger.info("[embedding_job] All %d chunks already embedded", len(chunks))
        db.commit()
        return len(chunks)

    # Build embeddable texts
    texts = []
    for chunk in chunks_to_embed:
        parts = []
        if chunk.chunk_name:
            parts.append(f"{chunk.chunk_type}: {chunk.chunk_name}")
        if chunk.signature:
            parts.append(f"Signature: {chunk.signature}")
        if chunk.docstring:
            parts.append(f"Description: {chunk.docstring[:500]}")
        if chunk.file_path:
            parts.append(f"File: {chunk.file_path}")
        texts.append("\n".join(parts))

    # Call embedding API
    logger.info("[embedding_job] Calling embeddings API for %d texts...", len(texts))
    try:
        vectors = get_embeddings(texts, model=EMBEDDING_MODEL)
        logger.info("[embedding_job] Got %d vectors", len(vectors))
    except Exception as e:
        logger.error("[embedding_job] API call failed: %s", e)
        raise

    # Store embeddings
    success_count = 0
    for i, chunk in enumerate(chunks_to_embed):
        try:
            store_embedding(
                db=db, project_id=0, source_type="arch_code_chunk",
                source_id=chunk.id, content=texts[i], embedding=vectors[i],
            )
            chunk.embedded = True
            chunk.embedding_model = EMBEDDING_MODEL
            chunk.embedded_at = now
            chunk.embedded_content_hash = chunk.content_hash
            status.embedded_chunks += 1
            success_count += 1
        except Exception as e:
            logger.error("[embedding_job] Failed chunk %d: %s", chunk.id, e)
            status.failed_chunks += 1

    db.commit()
    return success_count + skipped_existing


def embed_batch_with_retry(
    chunk_ids: List[int],
    db_session_factory: Callable[[], Session],
    status: "EmbeddingJobStatus",
) -> int:
    """Embed batch with SQLite lock retry and fresh session per batch.

    v1.2: Session-per-batch prevents poisoned session cascade.
    """
    from sqlalchemy.exc import OperationalError
    from app.rag.models import ArchCodeChunk

    backoff = SQLITE_LOCK_INITIAL_BACKOFF
    last_error = None

    for attempt in range(1, SQLITE_LOCK_MAX_RETRIES + 1):
        db = None
        try:
            db = db_session_factory()
            chunks = db.query(ArchCodeChunk).filter(ArchCodeChunk.id.in_(chunk_ids)).all()

            if len(chunks) != len(chunk_ids):
                logger.warning("[embedding_job] Fetched %d/%d chunks", len(chunks), len(chunk_ids))
            if not chunks:
                return 0

            return embed_batch(db, chunks, status)

        except OperationalError as e:
            if _is_sqlite_lock_error(e):
                last_error = e
                logger.warning(
                    "[embedding_job] SQLite lock attempt %d/%d, backoff %.2fs",
                    attempt, SQLITE_LOCK_MAX_RETRIES, backoff,
                )
                if db:
                    try:
                        db.rollback()
                    except Exception:
                        pass
                    try:
                        db.close()
                    except Exception:
                        pass
                    db = None
                time.sleep(backoff)
                backoff = min(backoff * 2, SQLITE_LOCK_MAX_BACKOFF)
            else:
                raise
        except Exception:
            raise
        finally:
            if db:
                try:
                    db.close()
                except Exception:
                    pass

    error_msg = f"SQLite lock persisted after {SQLITE_LOCK_MAX_RETRIES} retries: {last_error}"
    logger.error("[embedding_job] %s", error_msg)
    status.last_error = error_msg
    raise RuntimeError(error_msg)
