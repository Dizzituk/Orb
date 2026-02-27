# FILE: app/content/style_router.py
"""
Style reference router — upload, list, analyse.
Prefix: /content (nested under projects and standalone for refs)
"""

import asyncio
import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth
from app.content import style_service
from app.content.style_analyser import analyse_reference, analyse_all
from app.content.project_schemas import StyleReferenceResponse, StyleCategoryEnum
from app.content.project_models import StyleCategory
from app.content.sse_manager import sse_manager

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/content",
    tags=["Content Style"],
    dependencies=[Depends(require_auth)],
)


@router.get("/projects/{project_id}/style-references", response_model=List[StyleReferenceResponse])
def list_refs(project_id: str, category: Optional[StyleCategoryEnum] = None, db: Session = Depends(get_db)):
    return style_service.list_references(db, project_id, category)


@router.post("/projects/{project_id}/style-references", response_model=StyleReferenceResponse, status_code=201)
async def upload_ref(
    project_id: str,
    file: UploadFile = File(...),
    category: StyleCategoryEnum = Form(...),
    db: Session = Depends(get_db),
):
    return await style_service.upload_reference(db, project_id, StyleCategory(category.value), file)


@router.delete("/style-references/{reference_id}", status_code=204)
def delete_ref(reference_id: str, db: Session = Depends(get_db)):
    if not style_service.delete_reference(db, reference_id):
        raise HTTPException(404, "Style reference not found")


@router.post("/style-references/{reference_id}/analyse")
async def trigger_analyse(reference_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    ref = style_service.get_reference(db, reference_id)
    if not ref:
        raise HTTPException(404, "Style reference not found")

    async def _run():
        await analyse_reference(db, reference_id, sse_callback=lambda e: sse_manager.publish(ref.project_id, e))

    background_tasks.add_task(asyncio.ensure_future, _run())
    return {"status": "queued", "reference_id": reference_id}


@router.post("/projects/{project_id}/style-references/analyse-all")
async def trigger_analyse_all(
    project_id: str,
    category: Optional[StyleCategoryEnum] = None,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db),
):
    async def _run():
        await analyse_all(db, project_id, category=category, sse_callback=lambda e: sse_manager.publish(project_id, e))

    background_tasks.add_task(asyncio.ensure_future, _run())
    return {"status": "queued", "project_id": project_id}
