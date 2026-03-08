from __future__ import annotations

import logging
from typing import List, Optional

from sqlalchemy.orm import Session, selectinload

from app.education.models import Course, CourseModule, CourseStatus, ModuleStatus
from app.education.schemas import CourseResponse, CourseSummary, CourseModuleResponse, ScrapeResultResponse
from app.education.scraper import EducationScrapeError, scrape_coursera_course

logger = logging.getLogger(__name__)


def create_course(db: Session, name: str) -> Course:
    course = Course(name=name.strip())
    db.add(course)
    db.commit()
    db.refresh(course)
    return course


def list_courses(db: Session) -> List[Course]:
    return db.query(Course).options(selectinload(Course.modules)).order_by(Course.updated_at.desc()).all()


def get_course_with_modules(db: Session, course_id: str) -> Optional[Course]:
    return (
        db.query(Course)
        .options(selectinload(Course.modules))
        .filter(Course.id == course_id)
        .first()
    )


def submit_url_and_scrape(db: Session, course_id: str, url: str) -> Optional[ScrapeResultResponse]:
    course = get_course_with_modules(db, course_id)
    if not course:
        return None

    scraped_modules = scrape_coursera_course(url)
    course.url = url.strip()

    for existing in list(course.modules):
        db.delete(existing)
    db.flush()

    for module in scraped_modules:
        db.add(
            CourseModule(
                course_id=course.id,
                title=module.title,
                description=module.description,
                order_index=module.order_index,
                status=ModuleStatus.not_started,
            )
        )

    db.commit()
    db.refresh(course)
    course = get_course_with_modules(db, course_id)
    assert course is not None

    return ScrapeResultResponse(
        course=to_response(course),
        modules=[CourseModuleResponse.model_validate(module) for module in course.modules],
        scraped_count=len(course.modules),
        source_url=url.strip(),
    )


def update_course_status(db: Session, course_id: str, *, status: Optional[str] = None, name: Optional[str] = None) -> Optional[Course]:
    course = get_course_with_modules(db, course_id)
    if not course:
        return None
    if status is not None:
        course.status = CourseStatus(status)
    if name is not None:
        course.name = name.strip()
    db.commit()
    db.refresh(course)
    return get_course_with_modules(db, course_id)


def delete_course(db: Session, course_id: str) -> bool:
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        return False
    db.delete(course)
    db.commit()
    return True


def to_summary(course: Course) -> CourseSummary:
    return CourseSummary(
        id=course.id,
        name=course.name,
        url=course.url,
        status=course.status,
        module_count=len(course.modules or []),
        created_at=course.created_at,
        updated_at=course.updated_at,
    )


def to_response(course: Course) -> CourseResponse:
    return CourseResponse(
        id=course.id,
        name=course.name,
        url=course.url,
        status=course.status,
        modules=[CourseModuleResponse.model_validate(module) for module in (course.modules or [])],
        created_at=course.created_at,
        updated_at=course.updated_at,
    )


__all__ = [
    "EducationScrapeError",
    "create_course",
    "list_courses",
    "get_course_with_modules",
    "submit_url_and_scrape",
    "update_course_status",
    "delete_course",
    "to_summary",
    "to_response",
]
