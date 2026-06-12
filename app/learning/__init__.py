# FILE: app/learning/__init__.py
# Purpose: Course content scraping + RAG ingestion for Coursera (and future platforms).
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.learning.models
# Last-renovated: 2026-06-11
"""
Course content scraping + RAG ingestion for Coursera (and future platforms).

Purpose: let ASTRA answer questions about courses the user is enrolled
in, without needing live browser access. A nightly scraper walks the
user's enrolled courses, extracts transcripts / reading text, and
writes chunked content into the existing RAG system. The Bridge app
(driving use) can then query "what did that lesson say about X" and
get sub-2-second answers from retrieval rather than a 30s live browse.

Layers:
    models.py      - SQLAlchemy data models (enrollments, lessons, content)
    scraper.py     - [TO BUILD] walks Coursera via web_automation primitives
    rag_ingest.py  - [TO BUILD] chunks text and writes into rag_entries
    scheduler.py   - [TO BUILD] nightly run hook
    router.py      - [TO BUILD] thin HTTP API for the dashboard

The chat-side live demo path (where the user says "open my Coursera
and tell me what's there") uses the web_* LLM tools directly - it
doesn't need this module at all. This module exists for the driving
use case: pre-cached content that voice queries hit fast.
"""
from app.learning.models import (
    CourseEnrollment,
    CourseLesson,
    CourseLessonContent,
    CourseEnrollmentStatus,
    LessonType,
)

__all__ = [
    "CourseEnrollment",
    "CourseLesson",
    "CourseLessonContent",
    "CourseEnrollmentStatus",
    "LessonType",
]
