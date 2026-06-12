# Purpose: schemas
# Called-by: app.education.router, app.education.scraper, app.education.service
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CourseStatusEnum(str, Enum):
    active = "active"
    completed = "completed"
    archived = "archived"


class ModuleStatusEnum(str, Enum):
    not_started = "not_started"
    in_progress = "in_progress"
    completed = "completed"


class CreateCourseRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


class SubmitCourseUrlRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=1000)


class UpdateCourseRequest(BaseModel):
    status: Optional[CourseStatusEnum] = None
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)


class CourseModuleResponse(BaseModel):
    id: str
    course_id: str
    title: str
    description: Optional[str] = None
    order_index: int
    status: ModuleStatusEnum
    sub_modules: List[Dict[str, Any]] = []
    created_at: datetime

    class Config:
        from_attributes = True


class CourseSummary(BaseModel):
    id: str
    name: str
    url: Optional[str] = None
    status: CourseStatusEnum
    module_count: int = 0
    skills_gained: List[str] = []
    tools_learned: List[str] = []
    course_details: Dict[str, Any] = {}
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CourseResponse(BaseModel):
    id: str
    name: str
    url: Optional[str] = None
    status: CourseStatusEnum
    skills_gained: List[str] = []
    tools_learned: List[str] = []
    course_details: Dict[str, Any] = {}
    modules: List[CourseModuleResponse] = []
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ScrapedModule(BaseModel):
    title: str
    description: Optional[str] = None
    order_index: int
    sub_modules: List[Dict[str, Any]] = []


class ScrapeResultResponse(BaseModel):
    course: CourseResponse
    modules: List[CourseModuleResponse]
    scraped_count: int
    source_url: str
    provider: str = "coursera"
    skills_gained: List[str] = []
    tools_learned: List[str] = []
    course_details: Dict[str, Any] = {}
