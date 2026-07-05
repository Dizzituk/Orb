# FILE: app/rag/reindex.py
# Purpose: Reindex new/modified chunks after a rescan.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.embeddings.models, app.embeddings.service, app.rag.descriptors.descriptor_gen, app.rag.models
# Last-renovated: 2026-06-11
"""
Reindex new/modified chunks after a rescan.

Two-step process:
1. Generate descriptors for chunks missing them (deterministic, instant)
2. Generate embeddings for chunks missing them (API call, ~1 sec per batch)

Usage:
    from app.rag.reindex import reindex_unembedded
    
    stats = reindex_unembedded(db)
    # stats = {"descriptors_generated": N, "embeddings_created": N, "errors": N}
"""

import logging
from typing import Dict

from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.rag.models import ArchCodeChunk, ChunkType
from app.rag.descriptors.descriptor_gen import (
    generate_chunk_descriptor,
    estimate_tokens,
)
from app.embeddings.service import generate_embedding, store_embedding
from app.embeddings.provider_router import text_write_spec

logger = logging.getLogger(__name__)

ARCH_PROJECT_ID = 0
BATCH_SIZE = 50


def reindex_unembedded(
    db: Session,
    project_id: int = ARCH_PROJECT_ID,
) -> Dict[str, int]:
    """
    Generate descriptors and embeddings for all active chunks that need them.
    
    Steps:
    1. Find active, embeddable chunks without descriptors -> generate descriptors
    2. Find active, embeddable chunks without embeddings -> generate embeddings
    
    Returns:
        {"descriptors_generated": N, "embeddings_created": N, "errors": N}
    """
    stats = {
        "descriptors_generated": 0,
        "embeddings_created": 0,
        "errors": 0,
    }
    
    # Step 1: Generate descriptors for chunks missing them
    chunks_needing_descriptors = db.query(ArchCodeChunk).filter(
        and_(
            ArchCodeChunk.status == "active",
            ArchCodeChunk.chunk_type.in_(ChunkType.EMBEDDABLE),
            (ArchCodeChunk.descriptor.is_(None)) | (ArchCodeChunk.descriptor == ""),
        )
    ).all()
    
    if chunks_needing_descriptors:
        logger.info(f"[reindex] Generating descriptors for {len(chunks_needing_descriptors)} chunks")
        for chunk in chunks_needing_descriptors:
            descriptor = generate_chunk_descriptor(chunk)
            chunk.descriptor = descriptor
            chunk.descriptor_tokens = estimate_tokens(descriptor)
            stats["descriptors_generated"] += 1
        db.commit()
    
    # Step 2: Generate embeddings for active chunks that aren't embedded
    chunks_needing_embeddings = db.query(ArchCodeChunk).filter(
        and_(
            ArchCodeChunk.status == "active",
            ArchCodeChunk.chunk_type.in_(ChunkType.EMBEDDABLE),
            ArchCodeChunk.embedded == False,
            ArchCodeChunk.descriptor.isnot(None),
            ArchCodeChunk.descriptor != "",
        )
    ).all()
    
    if not chunks_needing_embeddings:
        logger.info("[reindex] All active chunks already embedded")
        return stats
    
    logger.info(f"[reindex] Embedding {len(chunks_needing_embeddings)} chunks")
    
    for i, chunk in enumerate(chunks_needing_embeddings):
        try:
            embedding = generate_embedding(chunk.descriptor)
            
            if embedding is None:
                stats["errors"] += 1
                continue
            
            store_embedding(
                db=db,
                project_id=project_id,
                source_type="arch_code_chunk",
                source_id=chunk.id,
                content=chunk.descriptor,
                embedding=embedding,
                chunk_index=0,
            )
            
            chunk.embedded = True
            # generate_embedding() above rides the router's write path — label
            # with the router's spec so the stamp can't drift from the actual
            # model (LANE E).
            chunk.embedding_model = text_write_spec().label
            chunk.embedded_at = __import__("datetime").datetime.utcnow()
            chunk.embedded_content_hash = chunk.content_hash
            
            stats["embeddings_created"] += 1
            
            if (i + 1) % BATCH_SIZE == 0:
                db.commit()
                logger.info(f"[reindex] Embedded {i + 1}/{len(chunks_needing_embeddings)}")
                
        except Exception as e:
            logger.error(f"[reindex] Error embedding chunk {chunk.id} ({chunk.chunk_name}): {e}")
            stats["errors"] += 1
    
    db.commit()
    
    logger.info(
        f"[reindex] Complete: {stats['descriptors_generated']} descriptors, "
        f"{stats['embeddings_created']} embeddings, {stats['errors']} errors"
    )
    return stats


def cleanup_orphaned_embeddings(db: Session) -> int:
    """
    Remove embeddings that point to non-active chunks.
    
    Called after a rescan to clean up stale embedding vectors.
    
    Returns:
        Number of orphaned embeddings removed
    """
    from app.embeddings.models import Embedding
    
    # Get all active chunk IDs
    active_ids = {
        r[0] for r in
        db.query(ArchCodeChunk.id).filter(ArchCodeChunk.status == "active").all()
    }
    
    # Find orphaned embeddings
    arch_embeddings = db.query(Embedding).filter(
        Embedding.source_type == "arch_code_chunk"
    ).all()
    
    orphaned = [e for e in arch_embeddings if e.source_id not in active_ids]
    
    for e in orphaned:
        db.delete(e)
    
    db.commit()
    
    if orphaned:
        logger.info(f"[reindex] Removed {len(orphaned)} orphaned embeddings")
    
    return len(orphaned)
