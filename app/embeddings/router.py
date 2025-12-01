# FILE: app/embeddings/router.py
"""
FastAPI routes for embedding operations.
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth
from app.memory import service as memory_service

from . import service
from .schemas import (
    SearchRequest,
    SearchResponse,
    IndexRequest,
    IndexResponse,
)

router = APIRouter(
    prefix="/embeddings",
    tags=["embeddings"],
    dependencies=[Depends(require_auth)],
)


# Also create a search endpoint under /memory for consistency
search_router = APIRouter(
    prefix="/memory",
    tags=["memory"],
    dependencies=[Depends(require_auth)],
)


@router.post("/index", response_model=IndexResponse)
def index_project_embeddings(
    req: IndexRequest,
    db: Session = Depends(get_db),
):
    """
    Index new content for a project.
    Only creates embeddings for items that don't have them yet.
    """
    project = memory_service.get_project(db, req.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    counts = service.index_project(
        db,
        req.project_id,
        source_types=req.source_types,
        force=False,
    )
    
    total_indexed = counts["notes"] + counts["messages"] + counts["files"]
    
    return IndexResponse(
        project_id=req.project_id,
        indexed_count=total_indexed,
        skipped_count=0,  # Not tracked in simple mode
        error_count=counts["errors"],
        details=f"Notes: {counts['notes']}, Messages: {counts['messages']}, Files: {counts['files']}",
    )


@router.post("/reindex", response_model=IndexResponse)
def reindex_project_embeddings(
    req: IndexRequest,
    db: Session = Depends(get_db),
):
    """
    Re-index all content for a project.
    Deletes existing embeddings and creates new ones.
    """
    project = memory_service.get_project(db, req.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    counts = service.reindex_project(
        db,
        req.project_id,
        source_types=req.source_types,
    )
    
    total_indexed = counts["notes"] + counts["messages"] + counts["files"]
    
    return IndexResponse(
        project_id=req.project_id,
        indexed_count=total_indexed,
        skipped_count=0,
        error_count=counts["errors"],
        details=f"Notes: {counts['notes']}, Messages: {counts['messages']}, Files: {counts['files']}",
    )


@router.get("/status/{project_id}")
def get_embedding_status(
    project_id: int,
    db: Session = Depends(get_db),
):
    """Get embedding statistics for a project."""
    project = memory_service.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    from .models import Embedding
    from sqlalchemy import func
    
    # Count embeddings by source type
    counts = db.query(
        Embedding.source_type,
        func.count(Embedding.id).label("count"),
    ).filter(
        Embedding.project_id == project_id
    ).group_by(
        Embedding.source_type
    ).all()
    
    result = {
        "project_id": project_id,
        "total": 0,
        "by_type": {},
    }
    
    for source_type, count in counts:
        result["by_type"][source_type] = count
        result["total"] += count
    
    return result


@search_router.post("/search", response_model=SearchResponse)
def semantic_search(
    req: SearchRequest,
    db: Session = Depends(get_db),
):
    """
    Perform semantic search across project content.
    Returns most relevant notes, messages, and files.
    """
    project = memory_service.get_project(db, req.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    results, total_searched = service.search_embeddings(
        db,
        req.project_id,
        req.query,
        top_k=req.top_k,
        source_types=req.source_types,
    )
    
    return SearchResponse(
        query=req.query,
        results=results,
        total_searched=total_searched,
    )


# Convenience endpoint: index a single note immediately after creation
@router.post("/index/note/{note_id}")
def index_single_note(
    note_id: int,
    db: Session = Depends(get_db),
):
    """Index a single note by ID."""
    from app.memory.models import Note
    
    note = db.query(Note).filter(Note.id == note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    
    count = service.index_note(db, note, force=True)
    
    return {"indexed": count, "note_id": note_id}


@router.post("/index/message/{message_id}")
def index_single_message(
    message_id: int,
    db: Session = Depends(get_db),
):
    """Index a single message by ID."""
    from app.memory.models import Message
    
    message = db.query(Message).filter(Message.id == message_id).first()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    
    count = service.index_message(db, message, force=True)
    
    return {"indexed": count, "message_id": message_id}


@router.post("/index/file/{file_id}")
def index_single_file(
    file_id: int,
    db: Session = Depends(get_db),
):
    """Index a single file (DocumentContent) by file ID."""
    from app.memory.models import DocumentContent
    
    doc = db.query(DocumentContent).filter(DocumentContent.file_id == file_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document content not found for this file")
    
    count = service.index_document(db, doc, force=True)
    
    return {"indexed": count, "file_id": file_id}