# Purpose: FastAPI endpoints for RAG system.
# Called-by: main
# Depends-on: app.db, app.embeddings.models, app.rag.answerer, app.rag.models (+1 more)
# Last-renovated: 2026-06-11
"""
FastAPI endpoints for RAG system.

POST /rag/index - Trigger RAG indexing pipeline
GET /rag/status - Check RAG system status

v1.0 (2026-01): Initial implementation
"""

import os
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.rag.pipeline import run_rag_pipeline
from app.rag.models import ArchCodeChunk, ArchDirectoryIndex
from app.rag.answerer import ask_architecture, ArchAnswer

router = APIRouter(prefix="/rag", tags=["rag"])

# Default scan directory
DEFAULT_SCAN_DIR = os.getenv("ZOBIE_OUTPUT_DIR", r"D:\Orb\.architecture")


def _allowed_scan_roots() -> list[Path]:
    """Roots /rag/index may scan: the default dir + ASTRA_RAG_SCAN_ROOTS extras.

    scan_dir is a caller-supplied path that drives a file-scan + LLM pipeline —
    without this check any authenticated caller could point it anywhere on disk
    (security hardening 2026-07-02).
    """
    roots = [Path(DEFAULT_SCAN_DIR)]
    extra = os.getenv("ASTRA_RAG_SCAN_ROOTS", "")
    roots += [Path(p.strip()) for p in extra.split(";") if p.strip()]
    return roots


def _validate_scan_dir(scan_dir: str) -> str:
    requested = Path(scan_dir).resolve()
    for root in _allowed_scan_roots():
        root_resolved = root.resolve()
        if requested == root_resolved or \
                str(requested).startswith(str(root_resolved) + os.sep):
            return str(requested)
    raise HTTPException(
        status_code=403,
        detail="scan_dir outside allowed roots (default scan dir or "
               "ASTRA_RAG_SCAN_ROOTS)")


class IndexRequest(BaseModel):
    scan_dir: Optional[str] = None
    project_id: int = 0


class IndexResponse(BaseModel):
    success: bool
    message: str
    scan_id: Optional[int] = None
    chunks: Optional[int] = None
    directories: Optional[int] = None
    embeddings: Optional[int] = None


class StatusResponse(BaseModel):
    chunks: int
    directories: int
    embeddings: int
    last_scan_dir: Optional[str] = None


@router.post("/index", response_model=IndexResponse)
def trigger_rag_index(
    request: IndexRequest,
    db: Session = Depends(get_db),
):
    """
    Trigger RAG indexing pipeline.
    
    Reads SIGNATURES_*.json from scan_dir and creates embeddings.
    """
    scan_dir = _validate_scan_dir(request.scan_dir or DEFAULT_SCAN_DIR)

    if not os.path.isdir(scan_dir):
        raise HTTPException(status_code=400, detail=f"Directory not found: {scan_dir}")
    
    try:
        result = run_rag_pipeline(
            db=db,
            scan_dir=scan_dir,
            project_id=request.project_id,
        )
        
        return IndexResponse(
            success=True,
            message="RAG indexing complete",
            scan_id=result.get("scan_id"),
            chunks=result.get("chunks", 0),
            directories=result.get("directories", 0),
            embeddings=result.get("embeddings", 0),
        )
    except Exception as e:
        return IndexResponse(
            success=False,
            message=f"Pipeline failed: {str(e)}",
        )


@router.get("/status", response_model=StatusResponse)
def get_rag_status(db: Session = Depends(get_db)):
    """Get current RAG index status."""
    chunks = db.query(ArchCodeChunk).count()
    directories = db.query(ArchDirectoryIndex).count()
    
    # Count embeddings (approximate - embeddings table may have multiple sources)
    try:
        from app.embeddings.models import Embedding
        embeddings = db.query(Embedding).filter(
            Embedding.source_type.in_(["arch_code_chunk", "arch_directory"])
        ).count()
    except Exception:
        embeddings = 0
    
    return StatusResponse(
        chunks=chunks,
        directories=directories,
        embeddings=embeddings,
        last_scan_dir=DEFAULT_SCAN_DIR,
    )


class QueryRequest(BaseModel):
    question: str
    use_embeddings: bool = True


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list
    chunks_searched: int
    model_used: str


@router.post("/query", response_model=QueryResponse)
def query_architecture(
    request: QueryRequest,
    db: Session = Depends(get_db),
):
    """
    Ask a question about the codebase.
    
    Uses RAG to find relevant code and answer with LLM.
    """
    result = ask_architecture(
        db=db,
        question=request.question,
        use_embeddings=request.use_embeddings,
    )
    
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=result.sources,
        chunks_searched=result.chunks_searched,
        model_used=result.model_used,
    )
