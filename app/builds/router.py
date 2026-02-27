# FILE: app/builds/router.py
"""
Build project router — CRUD + pipeline state endpoints.
Prefix: /builds/projects
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth
from app.builds import service as build_service
from app.builds.schemas import (
    CreateBuildRequest,
    UpdateBuildRequest,
    UpdateStageRequest,
    BuildProjectResponse,
    BuildProjectSummary,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/builds/projects",
    tags=["Project Builds"],
    dependencies=[Depends(require_auth)],
)


@router.get("", response_model=List[BuildProjectSummary])
def list_projects(db: Session = Depends(get_db)):
    """List all build projects (summary view for grid)."""
    projects = build_service.list_projects(db)
    return [build_service.to_summary(p) for p in projects]


@router.get("/{project_id}", response_model=BuildProjectResponse)
def get_project(project_id: str, db: Session = Depends(get_db)):
    """Get full project detail including pipeline stages and log."""
    p = build_service.get_project(db, project_id)
    if not p:
        raise HTTPException(404, "Build project not found")
    return build_service.to_response(p)


@router.post("", response_model=BuildProjectResponse, status_code=201)
def create_project(req: CreateBuildRequest, db: Session = Depends(get_db)):
    """Create a new build project."""
    p = build_service.create_project(
        db,
        name=req.name,
        description=req.description,
        original_brief=req.original_brief,
    )
    return build_service.to_response(p)


@router.patch("/{project_id}", response_model=BuildProjectResponse)
def update_project(project_id: str, req: UpdateBuildRequest, db: Session = Depends(get_db)):
    """Update project metadata."""
    p = build_service.update_project(
        db, project_id, name=req.name, description=req.description, status=req.status,
    )
    if not p:
        raise HTTPException(404, "Build project not found")
    return build_service.to_response(p)


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str, db: Session = Depends(get_db)):
    """Delete a build project."""
    if not build_service.delete_project(db, project_id):
        raise HTTPException(404, "Build project not found")


@router.post("/{project_id}/stage", response_model=BuildProjectResponse)
def update_stage(project_id: str, req: UpdateStageRequest, db: Session = Depends(get_db)):
    """Update pipeline stage status (called by pipeline handlers)."""
    p = build_service.advance_stage(
        db, project_id,
        stage=req.stage,
        stage_status=req.stage_status,
        detail=req.detail,
    )
    if not p:
        raise HTTPException(404, "Build project not found")
    return build_service.to_response(p)


@router.patch("/{project_id}/spec", response_model=BuildProjectResponse)
def link_spec(project_id: str, spec_id: str, db: Session = Depends(get_db)):
    """Link a validated spec to this build project."""
    p = build_service.link_spec(db, project_id, spec_id)
    if not p:
        raise HTTPException(404, "Build project not found")
    return build_service.to_response(p)


@router.patch("/{project_id}/job", response_model=BuildProjectResponse)
def link_job(project_id: str, job_id: str, db: Session = Depends(get_db)):
    """Link a pipeline job to this build project."""
    p = build_service.link_job(db, project_id, job_id)
    if not p:
        raise HTTPException(404, "Build project not found")
    return build_service.to_response(p)
