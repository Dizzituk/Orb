# FILE: app/debug/project_router.py
"""
FastAPI router for Debug Project CRUD.

Endpoints:
    GET    /api/debug/projects          — List all projects
    POST   /api/debug/projects          — Create a project
    GET    /api/debug/projects/{id}     — Get one project
    PUT    /api/debug/projects/{id}     — Update a project
    DELETE /api/debug/projects/{id}     — Delete a project

v1.0 (2026-03-07): Initial implementation.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.debug.project_service import (
    list_projects,
    get_project,
    create_project,
    update_project,
    delete_project,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/debug/projects", tags=["debug-projects"])


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

class ProjectCreate(BaseModel):
    title: str
    description: str = ""
    error_summary: str = ""


class ProjectUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    error_summary: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("")
async def list_projects_endpoint(status: Optional[str] = None):
    """List all debug projects."""
    try:
        projects = list_projects(status=status)
        return {"projects": projects}
    except Exception as e:
        logger.exception("[debug_projects] List failed")
        raise HTTPException(500, detail=str(e))


@router.post("", status_code=201)
async def create_project_endpoint(payload: ProjectCreate):
    """Create a new debug project."""
    try:
        project = create_project(
            title=payload.title,
            description=payload.description,
            error_summary=payload.error_summary,
        )
        return project
    except Exception as e:
        logger.exception("[debug_projects] Create failed")
        raise HTTPException(500, detail=str(e))


@router.get("/{project_id}")
async def get_project_endpoint(project_id: str):
    """Get a single debug project."""
    project = get_project(project_id)
    if not project:
        raise HTTPException(404, detail=f"Project {project_id} not found")
    return project


@router.put("/{project_id}")
async def update_project_endpoint(project_id: str, payload: ProjectUpdate):
    """Update a debug project."""
    existing = get_project(project_id)
    if not existing:
        raise HTTPException(404, detail=f"Project {project_id} not found")

    fields = payload.model_dump(exclude_none=True)
    updated = update_project(project_id, **fields)
    return updated


@router.delete("/{project_id}", status_code=204)
async def delete_project_endpoint(project_id: str):
    """Delete a debug project."""
    ok = delete_project(project_id)
    if not ok:
        raise HTTPException(404, detail=f"Project {project_id} not found")
    return None
