# FILE: app/content/project_service.py
"""
Content project service — CRUD for projects + content counts.
"""

import logging
from typing import Optional, List

from sqlalchemy.orm import Session
from sqlalchemy import func as sa_func

from app.content.project_models import (
    ContentProject, StyleReference, ProjectContentItem, ProjectStatus,
)
from app.content.project_schemas import ContentCounts, ProjectResponse

logger = logging.getLogger(__name__)


def list_projects(db: Session) -> List[ContentProject]:
    return db.query(ContentProject).order_by(ContentProject.updated_at.desc()).all()


def get_project(db: Session, project_id: str) -> Optional[ContentProject]:
    return db.query(ContentProject).filter(ContentProject.id == project_id).first()


def create_project(db: Session, name: str) -> ContentProject:
    project = ContentProject(name=name, status=ProjectStatus.active)
    db.add(project)
    db.commit()
    db.refresh(project)
    logger.info(f"[content-hub] Created project '{name}' ({project.id})")
    return project


def update_project(db: Session, project_id: str, name: Optional[str] = None, status: Optional[str] = None) -> Optional[ContentProject]:
    project = get_project(db, project_id)
    if not project:
        return None
    if name is not None:
        project.name = name
    if status is not None:
        project.status = status
    db.commit()
    db.refresh(project)
    return project


def delete_project(db: Session, project_id: str) -> bool:
    project = get_project(db, project_id)
    if not project:
        return False
    db.query(StyleReference).filter(StyleReference.project_id == project_id).delete()
    db.query(ProjectContentItem).filter(ProjectContentItem.project_id == project_id).delete()
    db.delete(project)
    db.commit()
    logger.info(f"[content-hub] Deleted project {project_id}")
    return True


def get_content_counts(db: Session, project_id: str) -> ContentCounts:
    rows = (
        db.query(ProjectContentItem.content_type, sa_func.count(ProjectContentItem.id))
        .filter(ProjectContentItem.project_id == project_id)
        .group_by(ProjectContentItem.content_type)
        .all()
    )
    counts = {r[0]: r[1] for r in rows}
    return ContentCounts(
        videos=counts.get("video", 0),
        images=counts.get("image", 0),
        blogs=counts.get("blog", 0),
        total=sum(counts.values()),
    )


def to_response(project: ContentProject, db: Session) -> ProjectResponse:
    counts = get_content_counts(db, project.id)
    return ProjectResponse(
        id=project.id,
        name=project.name,
        status=project.status,
        created_at=project.created_at,
        updated_at=project.updated_at,
        thumbnail_url=project.thumbnail_path,
        content_counts=counts,
        style_profile_id=project.style_profile_id,
    )
