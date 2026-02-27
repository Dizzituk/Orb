# FILE: app/content/item_service.py
"""
Content item service — upload source material, list items.
"""

import os
import logging
from typing import Optional, List

from sqlalchemy.orm import Session
from fastapi import UploadFile

from app.content.project_models import ProjectContentItem

logger = logging.getLogger(__name__)
ITEM_UPLOAD_DIR = os.path.join("data", "content", "items")


def _ensure_dir(project_id: str) -> str:
    path = os.path.join(ITEM_UPLOAD_DIR, project_id)
    os.makedirs(path, exist_ok=True)
    return path


def list_items(db: Session, project_id: str, content_type: Optional[str] = None) -> List[ProjectContentItem]:
    q = db.query(ProjectContentItem).filter(ProjectContentItem.project_id == project_id)
    if content_type:
        q = q.filter(ProjectContentItem.content_type == content_type)
    return q.order_by(ProjectContentItem.created_at.desc()).all()


async def upload_source(db: Session, project_id: str, file: UploadFile) -> ProjectContentItem:
    upload_dir = _ensure_dir(project_id)
    file_path = os.path.join(upload_dir, file.filename)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    ct = _infer_type(file.content_type, file.filename)
    item = ProjectContentItem(
        project_id=project_id,
        content_type=ct,
        title=file.filename,
        status="uploaded",
        source_path=file_path,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def delete_item(db: Session, item_id: str) -> bool:
    item = db.query(ProjectContentItem).filter(ProjectContentItem.id == item_id).first()
    if not item:
        return False
    for p in [item.source_path, item.output_path]:
        if p and os.path.exists(p):
            os.remove(p)
    db.delete(item)
    db.commit()
    return True


EXT_MAP = {
    "mp4": "video", "mov": "video", "avi": "video", "webm": "video",
    "png": "image", "jpg": "image", "jpeg": "image", "gif": "image", "webp": "image",
    "md": "blog", "txt": "blog", "html": "blog", "pdf": "blog", "docx": "blog",
}


def _infer_type(mime: Optional[str], filename: str) -> str:
    if mime:
        if mime.startswith("video/"):
            return "video"
        if mime.startswith("image/"):
            return "image"
        if mime.startswith("text/"):
            return "blog"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return EXT_MAP.get(ext, "video")
