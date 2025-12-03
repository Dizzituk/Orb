# FILE: app/artefacts/router.py
"""
Phase 4 Artefacts Router - HTTP API Endpoints

Provides REST API for artefact management:
- GET /artefacts/{artefact_id} - Read artefact
- GET /artefacts/list - List artefacts with filters
- POST /artefacts/write - Write/update artefact
- DELETE /artefacts/{artefact_id} - Delete artefact
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from app.db import get_db
from app.jobs.models import Artefact
from app.artefacts.service import (
    ArtefactService,
    ArtefactNotFoundError,
    ArtefactConflictError,
)

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class WriteArtefactRequest(BaseModel):
    """Request to write an artefact."""
    project_id: int
    artefact_type: str = Field(..., description="Type: research_dossier, architecture_doc, code_patch_proposal, etc.")
    name: str = Field(..., description="Human-readable name")
    content: str = Field(..., description="Artefact content")
    metadata: Optional[dict] = None
    created_by_job_id: Optional[str] = None
    
    # For updates (optional)
    artefact_id: Optional[str] = Field(None, description="Existing artefact ID for updates")
    etag: Optional[str] = Field(None, description="Expected etag for concurrency control")


class WriteArtefactResponse(BaseModel):
    """Response after writing an artefact."""
    artefact_id: str
    version: int
    etag: str
    status: str


class ArtefactResponse(BaseModel):
    """Single artefact response."""
    artefact_id: str
    project_id: int
    artefact_type: str
    name: str
    version: int
    etag: str
    status: str
    content: str
    metadata: Optional[dict] = None
    created_by_job_id: Optional[str] = None
    supersedes_id: Optional[str] = None
    created_at: str
    updated_at: str


class ListArtefactsResponse(BaseModel):
    """List of artefacts."""
    artefacts: list[ArtefactResponse]
    total: int


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("/{artefact_id}", response_model=ArtefactResponse)
async def get_artefact(
    artefact_id: str,
    project_id: int = Query(..., description="Project ID for security"),
    db: Session = Depends(get_db),
):
    """
    Get an artefact by ID.
    
    Path parameters:
    - artefact_id: Artefact identifier
    
    Query parameters:
    - project_id: Project identifier (required for security)
    
    Returns:
    - Complete artefact with content and metadata
    """
    artefact = ArtefactService.read_artefact(
        db=db,
        artefact_id=artefact_id,
        project_id=project_id,
    )
    
    if not artefact:
        raise HTTPException(
            status_code=404,
            detail=f"Artefact not found: {artefact_id}"
        )
    
    return ArtefactResponse(
        artefact_id=artefact.id,
        project_id=artefact.project_id,
        artefact_type=artefact.artefact_type,
        name=artefact.name,
        version=artefact.version,
        etag=artefact.etag,
        status=artefact.status,
        content=artefact.content,
        metadata=artefact.metadata_json,
        created_by_job_id=artefact.created_by_job_id,
        supersedes_id=artefact.supersedes_id,
        created_at=artefact.created_at.isoformat(),
        updated_at=artefact.updated_at.isoformat(),
    )


@router.get("/list", response_model=ListArtefactsResponse)
async def list_artefacts(
    project_id: int = Query(..., description="Project ID"),
    artefact_type: Optional[str] = Query(None, description="Filter by type"),
    status: Optional[str] = Query(None, description="Filter by status (current, superseded, draft)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """
    List artefacts with filters.
    
    Query parameters:
    - project_id: Project identifier (required)
    - artefact_type: Optional type filter
    - status: Optional status filter (current, superseded, draft)
    - limit: Maximum results (1-200, default 50)
    - offset: Pagination offset (default 0)
    
    Returns:
    - artefacts: List of artefacts
    - total: Total count matching filters (before pagination)
    """
    # Build base query with filters
    query = db.query(Artefact).filter(Artefact.project_id == project_id)
    
    if artefact_type:
        query = query.filter(Artefact.artefact_type == artefact_type)
    
    if status:
        query = query.filter(Artefact.status == status)
    
    # Get total count BEFORE applying pagination
    total = query.count()
    
    # Apply ordering and pagination
    query = query.order_by(Artefact.created_at.desc())
    query = query.limit(limit).offset(offset)
    
    artefacts_list = query.all()
    
    # Convert to response models
    artefacts_response = [
        ArtefactResponse(
            artefact_id=a.id,
            project_id=a.project_id,
            artefact_type=a.artefact_type,
            name=a.name,
            version=a.version,
            etag=a.etag,
            status=a.status,
            content=a.content,
            metadata=a.metadata_json,
            created_by_job_id=a.created_by_job_id,
            supersedes_id=a.supersedes_id,
            created_at=a.created_at.isoformat(),
            updated_at=a.updated_at.isoformat(),
        )
        for a in artefacts_list
    ]
    
    return ListArtefactsResponse(
        artefacts=artefacts_response,
        total=total,  # Now returns actual total count
    )


@router.post("/write", response_model=WriteArtefactResponse)
async def write_artefact(
    request: WriteArtefactRequest,
    db: Session = Depends(get_db),
):
    """
    Write or update an artefact.
    
    For new artefacts:
    - Omit artefact_id and etag
    - Creates new artefact with version 1
    
    For updates:
    - Provide artefact_id and etag
    - Creates new version if etag matches
    - Returns 409 if etag doesn't match (concurrent modification)
    
    Request body:
    - project_id: Project identifier
    - artefact_type: Type (research_dossier, architecture_doc, etc.)
    - name: Human-readable name
    - content: Artefact content
    - metadata: Optional metadata dict
    - artefact_id: Optional (for updates)
    - etag: Optional (for concurrency control)
    
    Returns:
    - artefact_id: Created/updated artefact ID
    - version: Version number
    - etag: New etag
    - status: Status (current)
    """
    try:
        artefact = ArtefactService.write_artefact(
            db=db,
            project_id=request.project_id,
            artefact_type=request.artefact_type,
            name=request.name,
            content=request.content,
            metadata=request.metadata,
            created_by_job_id=request.created_by_job_id,
            etag=request.etag,
            artefact_id=request.artefact_id,
        )
        
        logger.info(
            f"[artefacts] Written artefact {artefact.id} "
            f"type={artefact.artefact_type} v{artefact.version}"
        )
        
        return WriteArtefactResponse(
            artefact_id=artefact.id,
            version=artefact.version,
            etag=artefact.etag,
            status=artefact.status,
        )
    
    except ArtefactNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    except ArtefactConflictError as e:
        # Concurrent modification detected
        raise HTTPException(
            status_code=409,
            detail=str(e),
        )
    
    except Exception as e:
        logger.exception(f"[artefacts] Error writing artefact")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.delete("/{artefact_id}")
async def delete_artefact(
    artefact_id: str,
    project_id: int = Query(..., description="Project ID for security"),
    db: Session = Depends(get_db),
):
    """
    Delete an artefact.
    
    Note: This is a hard delete. Consider marking as superseded instead.
    
    Path parameters:
    - artefact_id: Artefact identifier
    
    Query parameters:
    - project_id: Project identifier (required for security)
    
    Returns:
    - success: True if deleted
    """
    success = ArtefactService.delete_artefact(
        db=db,
        artefact_id=artefact_id,
        project_id=project_id,
    )
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Artefact not found: {artefact_id}"
        )
    
    logger.info(f"[artefacts] Deleted artefact {artefact_id}")
    
    return {"success": True}


__all__ = ["router"]