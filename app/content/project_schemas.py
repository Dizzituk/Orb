# FILE: app/content/project_schemas.py
# Purpose: Pydantic schemas for the Content Hub project endpoints.
# Called-by: app.content.item_router, app.content.project_router, app.content.project_service, app.content.style_router
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Pydantic schemas for the Content Hub project endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class ProjectStatusEnum(str, Enum):
    draft = "draft"
    active = "active"
    archived = "archived"


class StyleCategoryEnum(str, Enum):
    video = "video"
    image = "image"
    blog = "blog"
    brand = "brand"


class AnalysisStatusEnum(str, Enum):
    pending = "pending"
    analysing = "analysing"
    done = "done"
    failed = "failed"


# ── Requests ──

class CreateProjectRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)


class UpdateProjectRequest(BaseModel):
    name: Optional[str] = None
    status: Optional[ProjectStatusEnum] = None


class ChatMessageRequest(BaseModel):
    message: str = Field(..., min_length=1)
    attachments: List[str] = Field(default_factory=list)


# ── Responses ──

class ContentCounts(BaseModel):
    videos: int = 0
    images: int = 0
    blogs: int = 0
    total: int = 0


class ProjectResponse(BaseModel):
    id: str
    name: str
    status: ProjectStatusEnum
    created_at: datetime
    updated_at: datetime
    thumbnail_url: Optional[str] = None
    content_counts: ContentCounts = ContentCounts()
    style_profile_id: Optional[str] = None

    class Config:
        from_attributes = True


class StyleReferenceResponse(BaseModel):
    id: str
    project_id: str
    category: StyleCategoryEnum
    filename: str
    file_size: int
    mime_type: Optional[str] = None
    upload_path: str
    analysis_status: AnalysisStatusEnum
    style_notes: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ContentItemResponse(BaseModel):
    id: str
    project_id: str
    content_type: str
    title: str
    status: str
    source_path: Optional[str] = None
    output_path: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
