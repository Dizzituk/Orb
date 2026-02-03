# FILE: app/specs/service.py
"""
Service layer for ASTRA Spec operations.

Provides CRUD operations for specs, plus helper functions for the Weaver.
"""
from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc

from .models import Spec, SpecQuestion, SpecHistory
from .schema import (
    Spec as SpecSchema,
    SpecProvenance,
    SpecStatus,
    validate_spec,
    spec_to_markdown,
    SPEC_SCHEMA_VERSION,
)


def utcnow():
    """Timezone-aware UTC now."""
    return datetime.now(timezone.utc)


def create_spec(
    db: Session,
    project_id: int,
    spec_schema: SpecSchema,
    generator_model: str = "weaver-v1",
) -> Spec:
    """
    Create a new spec in the database.
    
    Args:
        db: Database session
        project_id: Project this spec belongs to
        spec_schema: The canonical spec object
        generator_model: Model that generated this spec
    
    Returns:
        Created Spec database record
    """
    # Compute hash
    spec_hash = spec_schema.compute_hash()
    
    # Generate markdown
    markdown = spec_to_markdown(spec_schema)
    
    # Create database record
    db_spec = Spec(
        spec_id=spec_schema.spec_id,
        spec_hash=spec_hash,
        project_id=project_id,
        status=SpecStatus.DRAFT.value,
        spec_version=spec_schema.spec_version,
        content_json=spec_schema.to_json(),
        content_markdown=markdown,
        title=spec_schema.title,
        summary=spec_schema.summary,
        source_message_ids=spec_schema.provenance.source_message_ids,
        summary_ids=spec_schema.provenance.summary_ids,
        token_count=spec_schema.provenance.token_count,
        commit_hash=spec_schema.provenance.commit_hash,
        generator_model=generator_model,
    )
    
    # Parse timestamps if present
    if spec_schema.provenance.timestamp_start:
        try:
            db_spec.timestamp_start = datetime.fromisoformat(spec_schema.provenance.timestamp_start)
        except (ValueError, TypeError):
            pass
    
    if spec_schema.provenance.timestamp_end:
        try:
            db_spec.timestamp_end = datetime.fromisoformat(spec_schema.provenance.timestamp_end)
        except (ValueError, TypeError):
            pass
    
    db.add(db_spec)
    db.commit()
    db.refresh(db_spec)
    
    # Record history
    add_spec_history(db, db_spec.id, "created", None, SpecStatus.DRAFT.value, 
                     triggered_by=generator_model)
    
    return db_spec


def get_spec(db: Session, spec_id: str) -> Optional[Spec]:
    """Get a spec by its UUID."""
    return db.query(Spec).filter(Spec.spec_id == spec_id).first()


def get_spec_by_db_id(db: Session, db_id: int) -> Optional[Spec]:
    """Get a spec by its database ID."""
    return db.query(Spec).filter(Spec.id == db_id).first()


def get_latest_spec(db: Session, project_id: int) -> Optional[Spec]:
    """Get the most recent spec for a project."""
    return db.query(Spec).filter(
        Spec.project_id == project_id
    ).order_by(desc(Spec.created_at)).first()


def get_latest_draft_spec(db: Session, project_id: int) -> Optional[Spec]:
    """Get the most recent draft spec for a project."""
    return db.query(Spec).filter(
        Spec.project_id == project_id,
        Spec.status == SpecStatus.DRAFT.value,
    ).order_by(desc(Spec.created_at)).first()


def get_latest_validated_spec(db: Session, project_id: int) -> Optional[Spec]:
    """Get the most recent validated spec for a project."""
    return db.query(Spec).filter(
        Spec.project_id == project_id,
        Spec.status == SpecStatus.VALIDATED.value,
    ).order_by(desc(Spec.created_at)).first()


def list_specs(
    db: Session,
    project_id: int,
    status: Optional[str] = None,
    limit: int = 20,
) -> List[Spec]:
    """List specs for a project, optionally filtered by status."""
    query = db.query(Spec).filter(Spec.project_id == project_id)
    
    if status:
        query = query.filter(Spec.status == status)
    
    return query.order_by(desc(Spec.created_at)).limit(limit).all()


def update_spec_status(
    db: Session,
    spec_id: str,
    new_status: str,
    validation_result: Optional[dict] = None,
    triggered_by: str = "system",
) -> Optional[Spec]:
    """Update a spec's status."""
    db_spec = get_spec(db, spec_id)
    if not db_spec:
        return None
    
    old_status = db_spec.status
    db_spec.status = new_status
    
    if validation_result:
        db_spec.validation_result = validation_result
    
    db.commit()
    db.refresh(db_spec)
    
    # Record history
    add_spec_history(db, db_spec.id, "status_changed", old_status, new_status,
                     details={"validation_result": validation_result} if validation_result else None,
                     triggered_by=triggered_by)
    
    return db_spec


def add_spec_history(
    db: Session,
    spec_db_id: int,
    action: str,
    old_status: Optional[str],
    new_status: Optional[str],
    details: Optional[dict] = None,
    triggered_by: str = "system",
) -> SpecHistory:
    """Add a history record for a spec."""
    history = SpecHistory(
        spec_id=spec_db_id,
        action=action,
        old_status=old_status,
        new_status=new_status,
        details=details,
        triggered_by=triggered_by,
    )
    db.add(history)
    db.commit()
    return history


def create_spec_question(
    db: Session,
    spec_db_id: int,
    category: str,
    question_text: str,
    context: Optional[str] = None,
) -> SpecQuestion:
    """Create a question for a spec (from Spec Gate)."""
    question = SpecQuestion(
        spec_id=spec_db_id,
        question_id=str(uuid.uuid4()),
        category=category,
        question_text=question_text,
        context=context,
        status="pending",
    )
    db.add(question)
    db.commit()
    db.refresh(question)
    return question


def answer_spec_question(
    db: Session,
    question_id: str,
    answer_text: str,
) -> Optional[SpecQuestion]:
    """Answer a spec question."""
    question = db.query(SpecQuestion).filter(
        SpecQuestion.question_id == question_id
    ).first()
    
    if not question:
        return None
    
    question.answer_text = answer_text
    question.status = "answered"
    question.answered_at = utcnow()
    
    db.commit()
    db.refresh(question)
    return question


def get_pending_questions(db: Session, spec_id: str) -> List[SpecQuestion]:
    """Get all pending questions for a spec."""
    db_spec = get_spec(db, spec_id)
    if not db_spec:
        return []
    
    return db.query(SpecQuestion).filter(
        SpecQuestion.spec_id == db_spec.id,
        SpecQuestion.status == "pending",
    ).all()


def get_spec_schema(db_spec: Spec) -> SpecSchema:
    """Convert database Spec to SpecSchema object."""
    return SpecSchema.from_json(db_spec.content_json)


def update_spec_content_markdown(
    db: Session,
    spec_id: str,
    content_markdown: str,
) -> Optional[Spec]:
    """
    Update a spec's content_markdown field.
    
    v2.4 (2026-02-03): Used to overwrite the generic spec_to_markdown() output
    with actual POT spec markdown (containing ## Change / ## Skip sections).
    This enables Overwatcher POT detection and Critical Pipeline architecture
    generation to receive the full grounded spec content.
    """
    db_spec = get_spec(db, spec_id)
    if not db_spec:
        return None
    db_spec.content_markdown = content_markdown
    db.commit()
    db.refresh(db_spec)
    return db_spec


def check_duplicate_spec(db: Session, project_id: int, spec_hash: str) -> Optional[Spec]:
    """Check if a spec with the same hash already exists."""
    return db.query(Spec).filter(
        Spec.project_id == project_id,
        Spec.spec_hash == spec_hash,
    ).first()
