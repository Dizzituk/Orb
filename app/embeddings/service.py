# FILE: app/embeddings/service.py
# Purpose: Core embedding service: generation, storage, and search.
# Called-by: app.content.video_pipeline.asset_library, app.drive.content_indexer, app.embeddings, app.embeddings.router (+13 more)
# Depends-on: app.embeddings.provider_router, app.embeddings.models, app.embeddings.schemas, app.memory (+1 more)
# Last-renovated: 2026-07-02 (LANE E: provider router + model-stamped rows + model-filtered dual-read search)
"""
Core embedding service: generation, storage, and search.
"""

import json
import math
import time
from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from .models import Embedding
from .schemas import SearchResult


# ============ CONFIGURATION ============

# Model & dimensions constants stay importable for legacy callers; actual
# provider selection is env-driven via provider_router (LANE E, 2026-07-02).
from app.embeddings.gemini_provider import (
    EMBEDDING_DIMENSIONS,
    GEMINI_EMBEDDING_MODEL as EMBEDDING_MODEL,
    TaskType,
)
from app.embeddings import provider_router

CHUNK_SIZE = 400  # tokens (approximate)
CHUNK_OVERLAP = 50  # tokens overlap between chunks


# ============ EMBEDDING GENERATION ============

def generate_embedding(
    text: str,
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> Optional[List[float]]:
    """
    Generate an embedding vector for text.

    LANE E: routes via provider_router — document-side tasks embed with the
    role's WRITE provider (gemini or local per env, effective immediately,
    rows stamped); query-side tasks respect the parity-gated cutover flag
    (EMBEDDINGS_TEXT_QUERY_CUTOVER).

    Args:
        text: Content to embed.
        task_type: Task hint. Use "RETRIEVAL_DOCUMENT" when indexing,
                   "RETRIEVAL_QUERY" when searching.

    Returns:
        List of floats (dims per active model), or None on failure.
    """
    return provider_router.embed_text(text, task_type=task_type)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks.
    Uses word boundaries for cleaner splits.
    """
    if not text:
        return []
    
    # Rough token estimate: 1 token ≈ 4 chars, so chunk_size tokens ≈ chunk_size * 4 chars
    char_chunk_size = chunk_size * 4
    char_overlap = overlap * 4
    
    # If text is short enough, return as single chunk
    if len(text) <= char_chunk_size:
        return [text]
    
    chunks = []
    words = text.split()
    
    # Estimate words per chunk (average word length ~5 chars + space)
    words_per_chunk = char_chunk_size // 6
    words_overlap = char_overlap // 6
    
    start = 0
    while start < len(words):
        end = min(start + words_per_chunk, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        
        # Move start, accounting for overlap
        start = end - words_overlap
        if start >= len(words) - words_overlap:
            break
    
    return chunks


# ============ STORAGE ============

def store_embedding(
    db: Session,
    project_id: int,
    source_type: str,
    source_id: int,
    content: str,
    embedding: List[float],
    chunk_index: int = 0,
    embedding_model: Optional[str] = None,
    dims: Optional[int] = None,
) -> Embedding:
    """Store an embedding in the database.

    LANE E: every row is stamped with its producing model + dims. Default
    stamp is the active text WRITE spec (all current writers embed via the
    router in the same process, so spec and vector cannot drift); dims is
    always the actual vector length.
    """
    if embedding_model is None:
        embedding_model = provider_router.text_write_spec().label
    record = Embedding(
        project_id=project_id,
        source_type=source_type,
        source_id=source_id,
        chunk_index=chunk_index,
        content=content,
        embedding=json.dumps(embedding),
        embedding_model=embedding_model,
        dims=dims if dims is not None else len(embedding),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def delete_embeddings_for_source(
    db: Session,
    project_id: int,
    source_type: str,
    source_id: int,
) -> int:
    """Delete all embeddings for a specific source."""
    count = db.query(Embedding).filter(
        and_(
            Embedding.project_id == project_id,
            Embedding.source_type == source_type,
            Embedding.source_id == source_id,
        )
    ).delete()
    db.commit()
    return count


def _model_space_criterion(embedding_model: str):
    """SQLAlchemy criterion for 'row belongs to this model-space'.

    NULL-stamped rows count as LEGACY gemini rows: the boot backfill skips
    id-windows that sit on the embeddings table's damaged btree pages (see
    app/embeddings/migrations.py), and those rows must stay searchable
    exactly as they were pre-LANE-E. reembed_batch restamps them per-row."""
    from sqlalchemy import or_
    from app.embeddings.provider_router import LEGACY_TEXT_LABEL

    if embedding_model == LEGACY_TEXT_LABEL:
        return or_(
            Embedding.embedding_model == embedding_model,
            Embedding.embedding_model.is_(None),
        )
    return Embedding.embedding_model == embedding_model


def get_embeddings_for_project(
    db: Session,
    project_id: int,
    source_types: Optional[List[str]] = None,
    embedding_model: Optional[str] = None,
) -> List[Embedding]:
    """Get all embeddings for a project, optionally filtered by source type
    and (LANE E) by producing model — search paths ALWAYS pass the model so
    scoring never mixes model spaces."""
    query = db.query(Embedding).filter(Embedding.project_id == project_id)
    if source_types:
        query = query.filter(Embedding.source_type.in_(source_types))
    if embedding_model is not None:
        query = query.filter(_model_space_criterion(embedding_model))
    return query.all()


def embedding_exists(
    db: Session,
    project_id: int,
    source_type: str,
    source_id: int,
) -> bool:
    """Check if embeddings exist for a source."""
    return db.query(Embedding).filter(
        and_(
            Embedding.project_id == project_id,
            Embedding.source_type == source_type,
            Embedding.source_id == source_id,
        )
    ).first() is not None


# ============ SEARCH ============

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(vec_a) != len(vec_b):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


# LANE E: cached per-model row counts drive the dual-read drop-out — once the
# legacy space is empty its query-embedding provider is never called again.
_MODEL_COUNT_TTL_SECONDS = 60.0
_model_count_cache: Dict[str, Tuple[int, float]] = {}


def _model_row_count(db: Session, embedding_model: str) -> int:
    """COUNT of rows in `embedding_model`'s space (cached ~60s). NULL rows
    count toward the legacy space — see _model_space_criterion."""
    now = time.monotonic()
    hit = _model_count_cache.get(embedding_model)
    if hit is not None and (now - hit[1]) < _MODEL_COUNT_TTL_SECONDS:
        return hit[0]
    count = (
        db.query(func.count(Embedding.id))
        .filter(_model_space_criterion(embedding_model))
        .scalar()
        or 0
    )
    _model_count_cache[embedding_model] = (int(count), now)
    return int(count)


def reset_model_count_cache() -> None:
    """Testing/migration hook: forget cached per-model row counts."""
    _model_count_cache.clear()


def search_embeddings(
    db: Session,
    project_id: int,
    query: str,
    top_k: int = 5,
    source_types: Optional[List[str]] = None,
) -> Tuple[List[SearchResult], int]:
    """
    Search embeddings by semantic similarity.
    Returns (results, total_searched).

    LANE E hard rule: each stored model-space is scored ONLY with a query
    vector from its own model — rows are filtered by embedding_model BEFORE
    scoring, so mixed-model cosine is structurally impossible. With a single
    active space, scores are raw cosine (bit-identical to the pre-LANE-E
    behaviour). During migration (dual-read) each space is max-normalised
    before merging; the legacy space drops out at zero rows.
    """
    spaces = provider_router.text_query_spaces()
    per_space: List[List[Tuple[Embedding, float]]] = []
    total_searched = 0

    for i, spec in enumerate(spaces):
        # Secondary (legacy) space: skip once emptied by reembed_batch.
        if i > 0 and _model_row_count(db, spec.label) == 0:
            continue

        rows = get_embeddings_for_project(
            db, project_id, source_types, embedding_model=spec.label
        )
        if not rows:
            continue

        query_embedding = provider_router.embed_text(
            query, task_type="RETRIEVAL_QUERY", spec=spec
        )
        if not query_embedding:
            continue

        scored: List[Tuple[Embedding, float]] = []
        for emb in rows:
            try:
                stored_embedding = json.loads(emb.embedding)
            except (json.JSONDecodeError, TypeError):
                continue
            similarity = cosine_similarity(query_embedding, stored_embedding)
            scored.append((emb, similarity))
        total_searched += len(rows)
        if scored:
            per_space.append(scored)

    if not per_space:
        return [], total_searched

    if len(per_space) == 1:
        merged = per_space[0]
    else:
        # Dual-read merge: normalise per space so neither model's score
        # scale dominates, then interleave by normalised score.
        merged = []
        for scored in per_space:
            max_abs = max(abs(s) for _, s in scored) or 1.0
            merged.extend((emb, s / max_abs) for emb, s in scored)

    merged.sort(key=lambda x: x[1], reverse=True)
    top_results = merged[:top_k]

    results = [
        SearchResult(
            source_type=emb.source_type,
            source_id=emb.source_id,
            chunk_index=emb.chunk_index,
            content=emb.content,
            similarity=round(sim, 4),
        )
        for emb, sim in top_results
    ]

    return results, total_searched


# ============ INDEXING ============

def index_note(db: Session, note, force: bool = False) -> int:
    """Index a single note. Returns number of embeddings created."""
    from app.memory.models import Note
    
    if not force and embedding_exists(db, note.project_id, "note", note.id):
        return 0
    
    # Delete existing embeddings if re-indexing
    if force:
        delete_embeddings_for_source(db, note.project_id, "note", note.id)
    
    # Combine title and content for embedding
    text = f"{note.title}\n\n{note.content}" if note.content else note.title
    
    chunks = chunk_text(text)
    count = 0
    
    for i, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk)
        if embedding:
            store_embedding(db, note.project_id, "note", note.id, chunk, embedding, i)
            count += 1
    
    return count


def index_message(db: Session, message, force: bool = False) -> int:
    """Index a single message. Returns number of embeddings created."""
    from app.memory.models import Message
    
    # Only index assistant messages (user messages are typically short)
    if message.role != "assistant":
        return 0
    
    if not force and embedding_exists(db, message.project_id, "message", message.id):
        return 0
    
    if force:
        delete_embeddings_for_source(db, message.project_id, "message", message.id)
    
    chunks = chunk_text(message.content)
    count = 0
    
    for i, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk)
        if embedding:
            store_embedding(db, message.project_id, "message", message.id, chunk, embedding, i)
            count += 1
    
    return count


def index_document(db: Session, doc, force: bool = False) -> int:
    """Index a DocumentContent record. Returns number of embeddings created."""
    from app.memory.models import DocumentContent
    
    if not force and embedding_exists(db, doc.project_id, "file", doc.file_id):
        return 0
    
    if force:
        delete_embeddings_for_source(db, doc.project_id, "file", doc.file_id)
    
    # Build text: filename + summary + structured data + raw text
    parts = [f"Document: {doc.filename}"]
    
    if doc.summary:
        parts.append(f"Summary: {doc.summary}")
    
    if doc.structured_data:
        try:
            data = json.loads(doc.structured_data)
            parts.append(f"Structured Data: {json.dumps(data)}")
        except json.JSONDecodeError:
            pass
    
    if doc.raw_text:
        parts.append(doc.raw_text)
    
    text = "\n\n".join(parts)
    chunks = chunk_text(text)
    count = 0
    
    for i, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk)
        if embedding:
            store_embedding(db, doc.project_id, "file", doc.file_id, chunk, embedding, i)
            count += 1
    
    return count


def index_project(
    db: Session,
    project_id: int,
    source_types: Optional[List[str]] = None,
    force: bool = False,
) -> Dict[str, int]:
    """
    Index all content for a project.
    If force=True, re-indexes everything. Otherwise only indexes new items.
    Returns counts per source type.
    """
    from app.memory import models as memory_models
    
    counts = {"notes": 0, "messages": 0, "files": 0, "errors": 0}
    types_to_index = source_types or ["note", "message", "file"]
    
    # Index notes
    if "note" in types_to_index:
        notes = db.query(memory_models.Note).filter(
            memory_models.Note.project_id == project_id
        ).all()
        for note in notes:
            try:
                counts["notes"] += index_note(db, note, force)
            except Exception as e:
                print(f"[embeddings] Error indexing note {note.id}: {e}")
                counts["errors"] += 1
    
    # Index messages (assistant only)
    if "message" in types_to_index:
        messages = db.query(memory_models.Message).filter(
            memory_models.Message.project_id == project_id,
            memory_models.Message.role == "assistant",
        ).all()
        for msg in messages:
            try:
                counts["messages"] += index_message(db, msg, force)
            except Exception as e:
                print(f"[embeddings] Error indexing message {msg.id}: {e}")
                counts["errors"] += 1
    
    # Index documents
    if "file" in types_to_index:
        docs = db.query(memory_models.DocumentContent).filter(
            memory_models.DocumentContent.project_id == project_id
        ).all()
        for doc in docs:
            try:
                counts["files"] += index_document(db, doc, force)
            except Exception as e:
                print(f"[embeddings] Error indexing document {doc.id}: {e}")
                counts["errors"] += 1
    
    return counts


def reindex_project(
    db: Session,
    project_id: int,
    source_types: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Re-index all content for a project (deletes and recreates embeddings)."""
    return index_project(db, project_id, source_types, force=True)

# ============ COMPATIBILITY ALIAS ============

def search(
    db: Session,
    project_id: int,
    query: str,
    top_k: int = 5,
    source_types: Optional[List[str]] = None,
) -> List[SearchResult]:
    """
    Compatibility alias for search_embeddings.
    Returns only the results list (not the tuple).
    
    This function exists for backward compatibility with code that calls
    embeddings_service.search() instead of embeddings_service.search_embeddings().
    """
    results, _ = search_embeddings(db, project_id, query, top_k, source_types)
    return results