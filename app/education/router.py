# Purpose: router
# Called-by: main
# Depends-on: app.auth, app.db, app.education, app.education.schemas (+1 more)
# Last-renovated: 2026-06-11
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth import require_auth
from app.db import get_db
from app.education import service
from app.education.schemas import (
    CourseModuleResponse,
    CourseResponse,
    CourseSummary,
    CreateCourseRequest,
    ScrapeResultResponse,
    SubmitCourseUrlRequest,
    UpdateCourseRequest,
)

router = APIRouter(
    prefix="/education/courses",
    tags=["Education"],
    dependencies=[Depends(require_auth)],
)


@router.post("", response_model=CourseResponse, status_code=201)
def create_course(req: CreateCourseRequest, db: Session = Depends(get_db)):
    return service.to_response(service.create_course(db, req.name))


@router.get("", response_model=List[CourseSummary])
def list_courses(db: Session = Depends(get_db)):
    return [service.to_summary(course) for course in service.list_courses(db)]


@router.get("/{course_id}", response_model=CourseResponse)
def get_course(course_id: str, db: Session = Depends(get_db)):
    course = service.get_course_with_modules(db, course_id)
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    return service.to_response(course)


@router.post("/{course_id}/scrape", response_model=ScrapeResultResponse)
def scrape_course(course_id: str, req: SubmitCourseUrlRequest, db: Session = Depends(get_db)):
    try:
        result = service.submit_url_and_scrape(db, course_id, req.url)
    except service.EducationScrapeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not result:
        raise HTTPException(status_code=404, detail="Course not found")
    return result


@router.patch("/{course_id}", response_model=CourseResponse)
def update_course(course_id: str, req: UpdateCourseRequest, db: Session = Depends(get_db)):
    course = service.update_course_status(
        db,
        course_id,
        status=req.status.value if req.status else None,
        name=req.name,
    )
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    return service.to_response(course)


@router.post("/modules/{module_id}/enroll", response_model=CourseModuleResponse)
def enroll_module(module_id: str, db: Session = Depends(get_db)):
    module = service.update_module_status(db, module_id, "in_progress")
    if not module:
        raise HTTPException(status_code=404, detail="Module not found")
    return CourseModuleResponse.model_validate(module)


@router.delete("/{course_id}", status_code=204)
def delete_course(course_id: str, db: Session = Depends(get_db)):
    if not service.delete_course(db, course_id):
        raise HTTPException(status_code=404, detail="Course not found")
