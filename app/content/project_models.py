# FILE: app/content/project_models.py
"""
Content Hub project models.
Adds project grouping and style reference tracking
on top of the existing content pipeline models.
"""

import uuid
import enum
from datetime import datetime, timezone

from sqlalchemy import Column, String, Integer, Text, DateTime, Enum as SAEnum
from app.db import Base


def _uuid():
    return str(uuid.uuid4())


def _now():
    return datetime.now(timezone.utc)


class ProjectStatus(str, enum.Enum):
    draft = "draft"
    active = "active"
    archived = "archived"


class AnalysisStatus(str, enum.Enum):
    pending = "pending"
    analysing = "analysing"
    done = "done"
    failed = "failed"


class StyleCategory(str, enum.Enum):
    video = "video"
    image = "image"
    blog = "blog"
    brand = "brand"


class ContentProject(Base):
    """A content project — groups style refs, source material, and outputs."""
    __tablename__ = "content_projects"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String(120), nullable=False)
    status = Column(SAEnum(ProjectStatus), default=ProjectStatus.active, nullable=False)
    thumbnail_path = Column(String, nullable=True)
    style_profile_id = Column(String, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_now)
    updated_at = Column(DateTime, nullable=False, default=_now, onupdate=_now)


class StyleReference(Base):
    """An uploaded style reference — video, image, blog, or brand asset."""
    __tablename__ = "content_style_references"

    id = Column(String, primary_key=True, default=_uuid)
    project_id = Column(String, nullable=False, index=True)
    category = Column(SAEnum(StyleCategory), nullable=False)
    filename = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String, nullable=True)
    upload_path = Column(String, nullable=False)
    analysis_status = Column(SAEnum(AnalysisStatus), default=AnalysisStatus.pending, nullable=False)
    style_notes = Column(Text, nullable=True)
    extracted_preferences = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_now)


class ProjectContentItem(Base):
    """A source file or output within a project."""
    __tablename__ = "content_project_items"

    id = Column(String, primary_key=True, default=_uuid)
    project_id = Column(String, nullable=False, index=True)
    content_type = Column(String, nullable=False)
    title = Column(String, nullable=False)
    status = Column(String, default="uploaded", nullable=False)
    source_path = Column(String, nullable=True)
    output_path = Column(String, nullable=True)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_now)
    updated_at = Column(DateTime, nullable=False, default=_now, onupdate=_now)
