# FILE: app/content/style_service.py
"""
Style reference service — upload, list, analysis lifecycle.
"""

import os
import json
import logging
from typing import Optional, List

from sqlalchemy.orm import Session
from fastapi import UploadFile

from app.content.project_models import StyleReference, AnalysisStatus, StyleCategory

logger = logging.getLogger(__name__)
STYLE_UPLOAD_DIR = os.path.join("data", "content", "style_references")


def _ensure_dir(project_id: str) -> str:
    path = os.path.join(STYLE_UPLOAD_DIR, project_id)
    os.makedirs(path, exist_ok=True)
    return path


def list_references(db: Session, project_id: str, category: Optional[StyleCategory] = None) -> List[StyleReference]:
    q = db.query(StyleReference).filter(StyleReference.project_id == project_id)
    if category:
        q = q.filter(StyleReference.category == category)
    return q.order_by(StyleReference.created_at.desc()).all()


def get_reference(db: Session, reference_id: str) -> Optional[StyleReference]:
    return db.query(StyleReference).filter(StyleReference.id == reference_id).first()


async def upload_reference(db: Session, project_id: str, category: StyleCategory, file: UploadFile) -> StyleReference:
    upload_dir = _ensure_dir(project_id)
    file_path = os.path.join(upload_dir, file.filename)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    ref = StyleReference(
        project_id=project_id,
        category=category,
        filename=file.filename,
        file_size=len(content),
        mime_type=file.content_type,
        upload_path=file_path,
        analysis_status=AnalysisStatus.pending,
    )
    db.add(ref)
    db.commit()
    db.refresh(ref)
    logger.info(f"[style] Uploaded {file.filename} -> {ref.id}")
    return ref


def delete_reference(db: Session, reference_id: str) -> bool:
    ref = get_reference(db, reference_id)
    if not ref:
        return False
    if ref.upload_path and os.path.exists(ref.upload_path):
        os.remove(ref.upload_path)
    db.delete(ref)
    db.commit()
    return True


def mark_analysing(db: Session, reference_id: str) -> Optional[StyleReference]:
    ref = get_reference(db, reference_id)
    if not ref:
        return None
    ref.analysis_status = AnalysisStatus.analysing
    db.commit()
    db.refresh(ref)
    return ref


def save_analysis_result(db: Session, reference_id: str, notes: str, preferences: Optional[dict] = None) -> Optional[StyleReference]:
    ref = get_reference(db, reference_id)
    if not ref:
        return None
    ref.analysis_status = AnalysisStatus.done
    ref.style_notes = notes
    if preferences:
        ref.extracted_preferences = json.dumps(preferences)
    db.commit()
    db.refresh(ref)
    return ref


def mark_failed(db: Session, reference_id: str, error: str = "") -> Optional[StyleReference]:
    ref = get_reference(db, reference_id)
    if not ref:
        return None
    ref.analysis_status = AnalysisStatus.failed
    ref.style_notes = f"Analysis failed: {error}" if error else "Analysis failed"
    db.commit()
    db.refresh(ref)
    return ref
