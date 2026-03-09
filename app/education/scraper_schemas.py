from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ScrapedModuleItem(BaseModel):
    title: str = Field(..., min_length=1)
    description: Optional[str] = None


class ScrapedSubCourse(BaseModel):
    title: str = Field(..., min_length=1)
    description: Optional[str] = None
    modules: List[ScrapedModuleItem] = Field(default_factory=list)


class ScrapedCourseData(BaseModel):
    title: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)
    sub_courses: List[ScrapedSubCourse] = Field(default_factory=list)


__all__ = [
    "ScrapedModuleItem",
    "ScrapedSubCourse",
    "ScrapedCourseData",
]
