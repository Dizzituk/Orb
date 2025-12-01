# FILE: app/embeddings/service.py
"""
Core embedding service: generation, storage, and search.
"""

import os
import json
import math
from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_

from .models import Embedding
from .schemas import SearchResult


# ============ CONFIGURATION ============

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
CHUNK_SIZE = 400  # tokens (approximate)
CHUNK_OVERLAP = 50  # tokens overlap between chunks


# ============ EMBEDDING GENERATION ============

def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Generate embedding vector for text using OpenAI.
    Returns None if generation fails.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("[embeddings] OpenAI package not installed")
        return None
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[embeddings] OPENAI_API_KEY not set")
        return None
    
    if not text or not text.strip():
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Truncate very long text (model limit is ~8191 tokens)
        # Rough estimate: 1 token ≈ 4 chars
        max_chars = 30000
        if len(text) > max_chars:
            text = text[:max_chars]
        
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        
        return response.data[0].embedding
        
    except Exception as e:
        print(f"[embeddings] Generation error: {e}")
        return None


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
) -> Embedding:
    """Store an embedding in the database."""
    record = Embedding(
        project_id=project_id,
        source_type=source_type,
        source_id=source_id,
        chunk_index=chunk_index,
        content=content,
        embedding=json.dumps(embedding),
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


def get_embeddings_for_project(
    db: Session,
    project_id: int,
    source_types: Optional[List[str]] = None,
) -> List[Embedding]:
    """Get all embeddings for a project, optionally filtered by source type."""
    query = db.query(Embedding).filter(Embedding.project_id == project_id)
    if source_types:
        query = query.filter(Embedding.source_type.in_(source_types))
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
    """
    # Generate query embedding
    query_embedding = generate_embedding(query)
    if not query_embedding:
        return [], 0
    
    # Get all embeddings for project
    embeddings = get_embeddings_for_project(db, project_id, source_types)
    
    if not embeddings:
        return [], 0
    
    # Compute similarities
    scored_results = []
    for emb in embeddings:
        try:
            stored_embedding = json.loads(emb.embedding)
            similarity = cosine_similarity(query_embedding, stored_embedding)
            scored_results.append((emb, similarity))
        except (json.JSONDecodeError, TypeError):
            continue
    
    # Sort by similarity (descending)
    scored_results.sort(key=lambda x: x[1], reverse=True)
    
    # Take top_k
    top_results = scored_results[:top_k]
    
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
    
    return results, len(embeddings)


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