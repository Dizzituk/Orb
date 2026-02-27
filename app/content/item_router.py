# FILE: app/content/item_router.py
"""
Content item router — upload source material, list items.
Prefix: /content
"""

from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth
from app.content import item_service
from app.content.project_schemas import ContentItemResponse

router = APIRouter(
    prefix="/content",
    tags=["Content Items"],
    dependencies=[Depends(require_auth)],
)


@router.get("/projects/{project_id}/items", response_model=List[ContentItemResponse])
def list_items(project_id: str, content_type: Optional[str] = None, db: Session = Depends(get_db)):
    return item_service.list_items(db, project_id, content_type)


@router.post("/projects/{project_id}/items/upload", response_model=ContentItemResponse, status_code=201)
async def upload_source(project_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    return await item_service.upload_source(db, project_id, file)


@router.delete("/items/{item_id}", status_code=204)
def delete_item(item_id: str, db: Session = Depends(get_db)):
    if not item_service.delete_item(db, item_id):
        raise HTTPException(404, "Item not found")
