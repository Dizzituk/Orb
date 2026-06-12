# Purpose: service utils 2
# Called-by: app.bridge.missed_replies, app.bridge.router, app.memory.service
# Depends-on: app.memory, app.memory.models, app.memory.schemas
# Last-renovated: 2026-06-11
from __future__ import annotations
from app.memory import models, schemas
from sqlalchemy.orm import Session
from typing import List, Optional


def create_project(db: Session, data: schemas.ProjectCreate) -> models.Project:
    project = models.Project(
        name=data.name,
        description=data.description,
        type=data.type or "development",
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return project

def get_project_by_name(db: Session, name: str) -> Optional[models.Project]:
    return db.query(models.Project).filter(models.Project.name == name).first()

def list_projects(db: Session) -> List[models.Project]:
    # Pin-aware ordering: pinned projects first (so they survive any LIMIT
    # the caller applies), then by recency. .desc() on a Boolean sorts True
    # (1) before False (0). The bridge endpoint depends on this so the phone
    # always sees pinned chats even if they're older than the recency cutoff.
    return (
        db.query(models.Project)
        .order_by(models.Project.pinned.desc(), models.Project.created_at.desc())
        .all()
    )

def get_file_by_name(db: Session, project_id: int, filename: str) -> Optional[models.File]:
    return (
        db.query(models.File)
        .filter(models.File.project_id == project_id)
        .filter(models.File.original_name.ilike(f"%{filename}%"))
        .order_by(models.File.created_at.desc())
        .first()
    )

def list_files(db: Session, project_id: int) -> List[models.File]:
    return (
        db.query(models.File)
        .filter(models.File.project_id == project_id)
        .order_by(models.File.created_at.desc())
        .all()
    )

def list_messages(db: Session, project_id: int, limit: int = 100) -> List[models.Message]:
    # v2026-06-10 FIX: was .order_by(created_at.asc()).limit(limit), which
    # returns the FIRST N messages ever -- freezing conversation history at
    # the start of any chat longer than `limit`. Fetch the most recent N
    # (by id, monotonic and tie-safe) then reverse so callers still get
    # chronological order.
    rows = (
        db.query(models.Message)
        .filter(models.Message.project_id == project_id)
        .order_by(models.Message.id.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(rows))

def get_document_content_by_file_id(
    db: Session, file_id: int
) -> Optional[models.DocumentContent]:
    return (
        db.query(models.DocumentContent)
        .filter(models.DocumentContent.file_id == file_id)
        .first()
    )

def get_latest_document_content(
    db: Session, project_id: int
) -> Optional[models.DocumentContent]:
    return (
        db.query(models.DocumentContent)
        .filter(models.DocumentContent.project_id == project_id)
        .order_by(models.DocumentContent.created_at.desc())
        .first()
    )
