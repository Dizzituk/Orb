# Purpose: models
# Called-by: app.education.service, app.llm.routing.domain_context, main
# Depends-on: app.db
# Last-renovated: 2026-06-11
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Enum as SAEnum, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import relationship

from app.db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class CourseStatus(str, enum.Enum):
    active = "active"
    completed = "completed"
    archived = "archived"


class ModuleStatus(str, enum.Enum):
    not_started = "not_started"
    in_progress = "in_progress"
    completed = "completed"


class Course(Base):
    __tablename__ = "education_courses"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String(200), nullable=False)
    url = Column(String(1000), nullable=True)
    status = Column(SAEnum(CourseStatus), default=CourseStatus.active, nullable=False)
    skills_gained = Column(JSON, nullable=True, default=list)
    tools_learned = Column(JSON, nullable=True, default=list)
    course_details = Column(JSON, nullable=True, default=dict)
    created_at = Column(DateTime, nullable=False, default=_now)
    updated_at = Column(DateTime, nullable=False, default=_now, onupdate=_now)

    modules = relationship(
        "CourseModule",
        back_populates="course",
        cascade="all, delete-orphan",
        order_by="CourseModule.order_index",
    )


class CourseModule(Base):
    __tablename__ = "education_course_modules"

    id = Column(String, primary_key=True, default=_uuid)
    course_id = Column(String, ForeignKey("education_courses.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(300), nullable=False)
    description = Column(Text, nullable=True)
    order_index = Column(Integer, nullable=False, default=0)
    status = Column(SAEnum(ModuleStatus), default=ModuleStatus.not_started, nullable=False)
    sub_modules = Column(JSON, nullable=True, default=list)
    created_at = Column(DateTime, nullable=False, default=_now)

    course = relationship("Course", back_populates="modules")
