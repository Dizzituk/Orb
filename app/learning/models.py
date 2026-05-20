# FILE: app/learning/models.py
"""
Data model for the learning / course-content subsystem.

Three-level hierarchy:

    CourseEnrollment  1 ── * CourseLesson  1 ── * CourseLessonContent

- CourseEnrollment    — a course the user is signed up for
- CourseLesson        — a single unit (video, reading, quiz) within a course
- CourseLessonContent — the actual text (transcript, reading body, notes)
                        + pointers to rag_entries for retrieval-time use

All three are timestamped so the scraper can decide what needs
refreshing and RAG can decide what's stale.

Kept deliberately simple at this stage. The scraper and RAG ingestion
layers will refine the schema as we learn what Coursera actually
surfaces. If a field feels missing, add it - don't work around.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import (
    Column, String, Integer, Float, DateTime, JSON, Boolean, ForeignKey, Enum as SAEnum,
)
from sqlalchemy.orm import relationship

from app.db import Base


# ─── Enums ──────────────────────────────────────────────────────────


class CourseEnrollmentStatus(str, Enum):
    """Where the user is in a course."""
    ACTIVE = "active"             # actively working through it
    COMPLETED = "completed"       # finished (may or may not have cert)
    ABANDONED = "abandoned"       # started but moved on
    ARCHIVED = "archived"         # user de-prioritised


class LessonType(str, Enum):
    """Rough categorisation of lesson content — drives scraper strategy."""
    VIDEO = "video"               # has a transcript endpoint
    READING = "reading"           # text-heavy, may have embedded media
    QUIZ = "quiz"                 # interactive, skip for RAG ingestion
    ASSIGNMENT = "assignment"     # user-submitted, skip for ingestion
    DISCUSSION = "discussion"     # forum-like, skip
    UNKNOWN = "unknown"           # scraper couldn't classify


# ─── Models ─────────────────────────────────────────────────────────


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CourseEnrollment(Base):
    """
    A course the user has signed up for on some learning platform.

    One row per (platform, external_id). The scraper upserts on this
    pair so re-scans don't duplicate.
    """
    __tablename__ = "course_enrollments"

    id = Column(String, primary_key=True, default=_new_uuid)

    # Platform identity
    platform = Column(String(64), nullable=False, index=True)
    # e.g. "coursera", "udemy" — matches the WebSession.platform key.
    external_id = Column(String(512), nullable=False, index=True)
    # Coursera's course slug, e.g. "machine-learning-intro".

    # Display / navigation
    title = Column(String(512), nullable=False)
    url = Column(String(1024), nullable=False)
    provider = Column(String(256), nullable=True)
    # e.g. "DeepLearning.AI", "Stanford University".

    # Progress
    status = Column(
        SAEnum(CourseEnrollmentStatus, values_callable=lambda x: [e.value for e in x]),
        default=CourseEnrollmentStatus.ACTIVE,
        nullable=False,
    )
    progress_percent = Column(Float, nullable=True)
    # 0-100 if the platform surfaces it.

    # Timestamps
    enrolled_at = Column(DateTime(timezone=True), nullable=True)
    # When the user signed up (if the platform shows it).
    first_seen_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    # When ASTRA first saw this enrollment.
    last_scraped_at = Column(DateTime(timezone=True), nullable=True)
    # When the scraper last walked this course's lessons.

    # Free-form metadata (instructor names, certificate info, etc.)
    meta_json = Column(JSON, nullable=True)

    lessons = relationship(
        "CourseLesson",
        back_populates="enrollment",
        cascade="all, delete-orphan",
    )


class CourseLesson(Base):
    """
    A single lesson unit within a course.

    Lesson order is captured in `sequence` (global across the course) and
    `module_name` (the module/week the lesson belongs to). Together these
    let us reconstruct the course outline on demand.
    """
    __tablename__ = "course_lessons"

    id = Column(String, primary_key=True, default=_new_uuid)
    enrollment_id = Column(
        String,
        ForeignKey("course_enrollments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Platform identity
    external_id = Column(String(512), nullable=False)
    # Coursera's item_id - unique within an enrollment.
    url = Column(String(1024), nullable=False)

    # Display / structure
    title = Column(String(512), nullable=False)
    module_name = Column(String(256), nullable=True)
    # "Week 1: Introduction" or similar.
    sequence = Column(Integer, nullable=True)
    # Global position within the course.

    # Classification
    lesson_type = Column(
        SAEnum(LessonType, values_callable=lambda x: [e.value for e in x]),
        default=LessonType.UNKNOWN,
        nullable=False,
    )
    duration_seconds = Column(Integer, nullable=True)
    # For videos. Quizzes/readings leave this null.

    # Progress
    completed = Column(Boolean, default=False, nullable=False)

    # Timestamps
    first_seen_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    last_scraped_at = Column(DateTime(timezone=True), nullable=True)

    enrollment = relationship("CourseEnrollment", back_populates="lessons")
    contents = relationship(
        "CourseLessonContent",
        back_populates="lesson",
        cascade="all, delete-orphan",
    )


class CourseLessonContent(Base):
    """
    The actual text (transcript / reading body) extracted from a lesson.

    One lesson can have multiple content rows if the scraper extracts
    different artefacts (e.g. both a video transcript AND a reading
    companion). content_type distinguishes them.

    rag_entry_ids is a JSON array pointing into the existing rag_entries
    table - the link that lets voice queries retrieve this content
    through the normal RAG pipeline without needing to know about
    CourseLessonContent at all.
    """
    __tablename__ = "course_lesson_contents"

    id = Column(String, primary_key=True, default=_new_uuid)
    lesson_id = Column(
        String,
        ForeignKey("course_lessons.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # What kind of text: "transcript", "reading", "summary", "notes"
    content_type = Column(String(64), nullable=False, index=True)

    # The text itself. Not indexed - retrieval goes through rag_entries.
    text = Column(String, nullable=False)
    char_count = Column(Integer, nullable=False, default=0)

    # Where it came from on the platform, for audit + reproduction.
    source_url = Column(String(1024), nullable=True)

    # Links to rag_entries - populated by rag_ingest when we chunk and
    # embed this content. Empty/null until ingestion runs.
    rag_entry_ids = Column(JSON, nullable=True)

    scraped_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    lesson = relationship("CourseLesson", back_populates="contents")
