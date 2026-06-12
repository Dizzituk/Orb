# FILE: app/rag/jobs/_embedding_priority.py
# Purpose: Embedding priority classification patterns and chunk grouping.
# Called-by: app.rag.jobs.embedding_job
# Depends-on: app.rag.jobs._embedding_job_utils_7, app.rag.jobs._embedding_job_utils_8, app.rag.models
# Last-renovated: 2026-06-11
"""
Embedding priority classification patterns and chunk grouping.

Extracted from embedding_job.py for file-size modularity.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from app.rag.jobs._embedding_job_utils_7 import classify_chunk_priority, compute_content_hash
from app.rag.jobs._embedding_job_utils_8 import EmbeddingPriority

logger = logging.getLogger(__name__)


# Priority patterns (regex on file_path) — order matters, first match wins
PRIORITY_PATTERNS: Dict[EmbeddingPriority, List[re.Pattern]] = {
    # TIER 1: CRITICAL — Entry points, routing, dispatch
    EmbeddingPriority.TIER1_CRITICAL: [
        re.compile(r"main\.py$", re.IGNORECASE),
        re.compile(r"stream_router", re.IGNORECASE),
        re.compile(r"translation_routing", re.IGNORECASE),
        re.compile(r"app[/\\]translation[/\\]intents\.py$", re.IGNORECASE),
        re.compile(r"app[/\\]translation[/\\]tier0_rules\.py$", re.IGNORECASE),
        re.compile(r"app[/\\]translation[/\\]modes\.py$", re.IGNORECASE),
        re.compile(r"app[/\\]llm[/\\]local_tools[/\\]", re.IGNORECASE),
        re.compile(r"app[/\\]llm[/\\]streaming\.py$", re.IGNORECASE),
        re.compile(r"uvicorn|gunicorn|startup", re.IGNORECASE),
    ],
    # TIER 2: HIGH — Pipeline core
    EmbeddingPriority.TIER2_HIGH: [
        re.compile(r"spec_gate", re.IGNORECASE),
        re.compile(r"overwatcher", re.IGNORECASE),
        re.compile(r"critical_pipeline", re.IGNORECASE),
        re.compile(r"weaver", re.IGNORECASE),
        re.compile(r"app[/\\]rag[/\\]pipeline\.py$", re.IGNORECASE),
        re.compile(r"app[/\\]rag[/\\]answerer\.py$", re.IGNORECASE),
        re.compile(r"app[/\\]memory[/\\].*architecture", re.IGNORECASE),
        re.compile(r"app[/\\]memory[/\\]service\.py$", re.IGNORECASE),
        re.compile(r"job_classifier", re.IGNORECASE),
        re.compile(r"astra_memory", re.IGNORECASE),
    ],
    # TIER 3: MEDIUM — Services, models, schemas, DB
    EmbeddingPriority.TIER3_MEDIUM: [
        re.compile(r"models\.py$", re.IGNORECASE),
        re.compile(r"schemas\.py$", re.IGNORECASE),
        re.compile(r"service\.py$", re.IGNORECASE),
        re.compile(r"services[/\\]", re.IGNORECASE),
        re.compile(r"repository", re.IGNORECASE),
        re.compile(r"app[/\\]db\.py$", re.IGNORECASE),
        re.compile(r"database", re.IGNORECASE),
        re.compile(r"app[/\\]embeddings[/\\]", re.IGNORECASE),
    ],
    # TIER 4: LOW — Handlers, utilities, clients
    EmbeddingPriority.TIER4_LOW: [
        re.compile(r"handler", re.IGNORECASE),
        re.compile(r"client", re.IGNORECASE),
        re.compile(r"util", re.IGNORECASE),
        re.compile(r"helper", re.IGNORECASE),
        re.compile(r"config\.py$", re.IGNORECASE),
    ],
    # TIER 5: NORMAL — catch-all (no patterns)
}


def get_chunks_by_priority(
    db: Session,
    scan_id: Optional[int],
) -> Dict[EmbeddingPriority, List]:
    """Get chunks needing embedding, grouped by priority.

    INCREMENTAL: Only returns chunks that have no embedding or stale embedding.
    """
    from app.rag.models import ArchCodeChunk
    from sqlalchemy import func

    total_in_db = db.query(func.count(ArchCodeChunk.id)).scalar() or 0
    logger.info("[embedding_job] Total ArchCodeChunk rows: %d", total_in_db)

    query = db.query(ArchCodeChunk)
    if scan_id:
        query = query.filter(ArchCodeChunk.scan_id == scan_id)
        logger.info("[embedding_job] Filtering by scan_id=%d", scan_id)

    all_chunks = query.all()
    logger.info("[embedding_job] Chunks after filter: %d", len(all_chunks))

    # Compute missing content hashes
    hash_updates = 0
    for chunk in all_chunks:
        if not chunk.content_hash:
            chunk.content_hash = compute_content_hash(chunk)
            hash_updates += 1
    if hash_updates > 0:
        db.commit()
        logger.info("[embedding_job] Computed %d missing content hashes", hash_updates)

    # Group by priority, filtering for those needing embedding
    result: Dict[EmbeddingPriority, List] = {p: [] for p in EmbeddingPriority}
    already_embedded = 0
    needs_reembed = 0

    for chunk in all_chunks:
        if chunk.embedded:
            if chunk.content_hash and chunk.embedded_content_hash:
                if chunk.content_hash == chunk.embedded_content_hash:
                    already_embedded += 1
                    continue
                else:
                    needs_reembed += 1
            else:
                already_embedded += 1
                continue

        priority = classify_chunk_priority(chunk.file_path or "")
        result[priority].append(chunk)

    logger.info("[embedding_job] Already embedded: %d, needs re-embed: %d", already_embedded, needs_reembed)
    return result
