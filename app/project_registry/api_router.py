# FILE: app/project_registry/api_router.py
# Purpose: FastAPI router for the Project Registry API.
# Called-by: main
# Depends-on: app.auth.middleware, app.db, app.project_registry, app.project_registry.confidence_tracker (+1 more)
# Last-renovated: 2026-06-11
"""
FastAPI router for the Project Registry API.

Endpoints:
  GET  /registry/projects       — List registered projects
  POST /registry/projects       — Register a new project
  GET  /registry/projects/{slug} — Get project details + staleness
  DELETE /registry/projects/{slug} — Deregister a project
  POST /registry/feedback       — Record command confidence feedback

v1.0 (2026-03): Initial implementation.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth.middleware import require_auth

from . import service as registry_service
from . import confidence_tracker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/registry", tags=["project_registry"])


# ── Request/Response models ─────────────────────────────────────────────

class RegisterProjectRequest(BaseModel):
    name: str
    root_paths: List[str]
    project_type: Optional[str] = None
    primary_language: Optional[str] = None
    description: Optional[str] = None


class ProjectResponse(BaseModel):
    id: int
    name: str
    slug: str
    project_type: Optional[str]
    primary_language: Optional[str]
    description: Optional[str]
    root_paths: List[str]
    last_scan_at: Optional[str]
    last_scan_file_count: Optional[int]
    has_arch_map: bool
    is_stale: bool
    staleness_message: str

    class Config:
        from_attributes = True


class FeedbackRequest(BaseModel):
    feedback_id: int
    confirmed: bool


# ── Endpoints ───────────────────────────────────────────────────────────

@router.get("/projects", response_model=List[ProjectResponse])
def list_projects(db: Session = Depends(get_db)):
    """List all registered projects with staleness status."""
    projects = registry_service.list_projects(db)
    result = []
    for p in projects:
        staleness = registry_service.check_staleness(p)
        result.append(ProjectResponse(
            id=p.id,
            name=p.name,
            slug=p.slug,
            project_type=p.project_type,
            primary_language=p.primary_language,
            description=p.description,
            root_paths=registry_service.get_root_paths(p),
            last_scan_at=p.last_scan_at.isoformat() if p.last_scan_at else None,
            last_scan_file_count=p.last_scan_file_count,
            has_arch_map=p.has_arch_map,
            is_stale=staleness["is_stale"],
            staleness_message=staleness["message"],
        ))
    return result


@router.post("/projects", response_model=ProjectResponse)
def register_project(
    req: RegisterProjectRequest,
    db: Session = Depends(get_db),
):
    """Register a new external project."""
    try:
        p = registry_service.register_project(
            db=db,
            name=req.name,
            root_paths=req.root_paths,
            project_type=req.project_type,
            primary_language=req.primary_language,
            description=req.description,
        )
        staleness = registry_service.check_staleness(p)
        return ProjectResponse(
            id=p.id,
            name=p.name,
            slug=p.slug,
            project_type=p.project_type,
            primary_language=p.primary_language,
            description=p.description,
            root_paths=registry_service.get_root_paths(p),
            last_scan_at=None,
            last_scan_file_count=0,
            has_arch_map=False,
            is_stale=True,
            staleness_message=staleness["message"],
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/projects/{slug}", response_model=ProjectResponse)
def get_project(slug: str, db: Session = Depends(get_db)):
    """Get a project's details and staleness status."""
    p = registry_service.get_project_by_slug(db, slug)
    if not p:
        raise HTTPException(status_code=404, detail="Project not found")
    staleness = registry_service.check_staleness(p)
    return ProjectResponse(
        id=p.id,
        name=p.name,
        slug=p.slug,
        project_type=p.project_type,
        primary_language=p.primary_language,
        description=p.description,
        root_paths=registry_service.get_root_paths(p),
        last_scan_at=p.last_scan_at.isoformat() if p.last_scan_at else None,
        last_scan_file_count=p.last_scan_file_count,
        has_arch_map=p.has_arch_map,
        is_stale=staleness["is_stale"],
        staleness_message=staleness["message"],
    )


@router.delete("/projects/{slug}")
def deregister_project(slug: str, db: Session = Depends(get_db)):
    """Soft-delete a registered project."""
    success = registry_service.deregister_project(db, slug)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"deleted": slug}


@router.post("/feedback")
def record_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    """Record yes/no feedback from a command confirmation button."""
    confidence_tracker.record_feedback(db, req.feedback_id, req.confirmed)
    return {"recorded": True, "feedback_id": req.feedback_id}