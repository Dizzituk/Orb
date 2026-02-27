# FILE: app/content/project_router.py
"""
Content project router — CRUD for content hub projects.
Prefix: /content/projects
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth
from app.content import project_service
from app.content.project_schemas import CreateProjectRequest, UpdateProjectRequest, ProjectResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/content/projects",
    tags=["Content Hub"],
    dependencies=[Depends(require_auth)],
)


@router.get("", response_model=List[ProjectResponse])
def list_projects(db: Session = Depends(get_db)):
    projects = project_service.list_projects(db)
    return [project_service.to_response(p, db) for p in projects]


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: str, db: Session = Depends(get_db)):
    p = project_service.get_project(db, project_id)
    if not p:
        raise HTTPException(404, "Project not found")
    return project_service.to_response(p, db)


@router.post("", response_model=ProjectResponse, status_code=201)
def create_project(req: CreateProjectRequest, db: Session = Depends(get_db)):
    p = project_service.create_project(db, req.name)
    return project_service.to_response(p, db)


@router.patch("/{project_id}", response_model=ProjectResponse)
def update_project(project_id: str, req: UpdateProjectRequest, db: Session = Depends(get_db)):
    p = project_service.update_project(db, project_id, name=req.name, status=req.status)
    if not p:
        raise HTTPException(404, "Project not found")
    return project_service.to_response(p, db)


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str, db: Session = Depends(get_db)):
    if not project_service.delete_project(db, project_id):
        raise HTTPException(404, "Project not found")
