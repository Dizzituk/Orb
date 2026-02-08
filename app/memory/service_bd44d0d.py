# FILE: app/memory/service.py
"""
Memory service layer for Orb.

v0.12.4: Fixed create_message() to properly save provider, model, and reasoning fields.
         Fixed get_message_history() to return actual provider/model/reasoning values.
"""
from typing import List, Optional
from sqlalchemy.orm import Session
from app.memory import models, schemas


# ============== PROJECT ==============

def create_project(db: Session, data: schemas.ProjectCreate) -> models.Project:
    project = models.Project(name=data.name, description=data.description)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def get_project(db: Session, project_id: int) -> Optional[models.Project]:
    return db.query(models.Project).filter(models.Project.id == project_id).first()


def get_project_by_name(db: Session, name: str) -> Optional[models.Project]:
    return db.query(models.Project).filter(models.Project.name == name).first()


def list_projects(db: Session) -> List[models.Project]:
    return db.query(models.Project).order_by(models.Project.created_at.desc()).all()


def update_project(db: Session, project_id: int, data: schemas.ProjectUpdate) -> Optional[models.Project]:
    project = get_project(db, project_id)
    if not project:
        return None
    if data.name is not None:
        project.name = data.name
    if data.description is not None:
        project.description = data.description
    db.commit()
    db.refresh(project)
    return project


def delete_project(db: Session, project_id: int) -> bool:
    project = get_project(db, project_id)
    if not project:
        return False
    db.delete(project)
    db.commit()
    return True


# ============== NOTE ==============

def _index_note_if_enabled(db: Session, note: models.Note, force: bool = False) -> None:
    """Helper to index a note if auto-indexing is enabled."""
    try:
        from app.embeddings import auto_index_enabled, index_note
        if auto_index_enabled():
            count = index_note(db, note, force=force)
            print(f"[memory.service] Auto-indexed note {note.id}: {count} embeddings")
    except Exception as e:
        # Don't fail the main operation if indexing fails
        print(f"[memory.service] Failed to auto-index note {note.id}: {e}")


def create_note(db: Session, data: schemas.NoteCreate) -> models.Note:
    note = models.Note(
        project_id=data.project_id,
        title=data.title,
        content=data.content,
        tags=data.tags,
        source=data.source,
    )
    db.add(note)
    db.commit()
    db.refresh(note)
    
    _index_note_if_enabled(db, note)
    
    return note


def create_note_for_project(
    db: Session, project_id: int, data: schemas.NoteCreateForProject
) -> models.Note:
    note = models.Note(
        project_id=project_id,
        title=data.title,
        content=data.content,
        tags=data.tags,
        source=data.source,
    )
    db.add(note)
    db.commit()
    db.refresh(note)
    
    _index_note_if_enabled(db, note)
    
    return note


def get_note(db: Session, note_id: int) -> Optional[models.Note]:
    return db.query(models.Note).filter(models.Note.id == note_id).first()


def list_notes(
    db: Session,
    project_id: int,
    tag_filter: Optional[str] = None,
    search: Optional[str] = None,
) -> List[models.Note]:
    query = db.query(models.Note).filter(models.Note.project_id == project_id)
    if tag_filter:
        query = query.filter(models.Note.tags.ilike(f"%{tag_filter}%"))
    if search:
        pattern = f"%{search}%"
        query = query.filter(
            (models.Note.title.ilike(pattern)) | (models.Note.content.ilike(pattern))
        )
    return query.order_by(models.Note.created_at.desc()).all()


def update_note(db: Session, note_id: int, data: schemas.NoteUpdate) -> Optional[models.Note]:
    note = get_note(db, note_id)
    if not note:
        return None
    
    content_changed = False
    
    if data.title is not None:
        note.title = data.title
        content_changed = True
    if data.content is not None:
        note.content = data.content
        content_changed = True
    if data.tags is not None:
        note.tags = data.tags
    if data.source is not None:
        note.source = data.source
    
    db.commit()
    db.refresh(note)
    
    # Re-index if title or content changed
    if content_changed:
        _index_note_if_enabled(db, note, force=True)
    
    return note


def delete_note(db: Session, note_id: int) -> bool:
    note = get_note(db, note_id)
    if not note:
        return False
    
    # Delete associated embeddings
    try:
        from app.embeddings import delete_embeddings_for_source
        delete_embeddings_for_source(db, note.project_id, "note", note.id)
    except Exception as e:
        print(f"[memory.service] Failed to delete embeddings for note {note_id}: {e}")
    
    db.delete(note)
    db.commit()
    return True


# ============== TASK ==============

def create_task(db: Session, data: schemas.TaskCreate) -> models.Task:
    task = models.Task(
        project_id=data.project_id,
        title=data.title,
        description=data.description,
        status=data.status,
        priority=data.priority,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def create_task_for_project(
    db: Session, project_id: int, data: schemas.TaskCreateForProject
) -> models.Task:
    task = models.Task(
        project_id=project_id,
        title=data.title,
        description=data.description,
        status=data.status,
        priority=data.priority,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


def get_task(db: Session, task_id: int) -> Optional[models.Task]:
    return db.query(models.Task).filter(models.Task.id == task_id).first()


def list_tasks(
    db: Session,
    project_id: int,
    status: Optional[str] = None,
) -> List[models.Task]:
    query = db.query(models.Task).filter(models.Task.project_id == project_id)
    if status:
        query = query.filter(models.Task.status == status)
    return query.order_by(models.Task.created_at.desc()).all()


def update_task(db: Session, task_id: int, data: schemas.TaskUpdate) -> Optional[models.Task]:
    task = get_task(db, task_id)
    if not task:
        return None
    if data.title is not None:
        task.title = data.title
    if data.description is not None:
        task.description = data.description
    if data.status is not None:
        task.status = data.status
    if data.priority is not None:
        task.priority = data.priority
    db.commit()
    db.refresh(task)
    return task


def delete_task(db: Session, task_id: int) -> bool:
    task = get_task(db, task_id)
    if not task:
        return False
    db.delete(task)
    db.commit()
    return True


# ============== FILE ==============

def create_file(db: Session, data: schemas.FileCreate) -> models.File:
    file_record = models.File(
        project_id=data.project_id,
        path=data.path,
        original_name=data.original_name,
        file_type=data.file_type,
        description=data.description,
    )
    db.add(file_record)
    db.commit()
    db.refresh(file_record)
    return file_record


def create_file_for_project(
    db: Session,
    project_id: int,
    file_data: schemas.FileCreate,
) -> models.File:
    file_record = models.File(
        project_id=project_id,
        path=file_data.path,
        original_name=file_data.original_name,
        file_type=file_data.file_type,
        description=file_data.description,
    )
    db.add(file_record)
    db.commit()
    db.refresh(file_record)
    return file_record


def get_file(db: Session, file_id: int) -> Optional[models.File]:
    return db.query(models.File).filter(models.File.id == file_id).first()


def get_file_by_name(db: Session, project_id: int, filename: str) -> Optional[models.File]:
    return (
        db.query(models.File)
        .filter(models.File.project_id == project_id)
        .filter(models.File.original_name.ilike(f"%{filename}%"))
        .order_by(models.File.created_at.desc())
        .first()
    )


def list_files(db: Session, project_id: int) -> List[models.File]:
    return (
        db.query(models.File)
        .filter(models.File.project_id == project_id)
        .order_by(models.File.created_at.desc())
        .all()
    )


def delete_file(db: Session, file_id: int) -> bool:
    file_record = get_file(db, file_id)
    if not file_record:
        return False
    
    # Delete associated embeddings
    try:
        from app.embeddings import delete_embeddings_for_source
        delete_embeddings_for_source(db, file_record.project_id, "file", file_id)
    except Exception as e:
        print(f"[memory.service] Failed to delete embeddings for file {file_id}: {e}")
    
    db.delete(file_record)
    db.commit()
    return True


# ============== MESSAGE ==============

def _index_message_if_enabled(db: Session, message: models.Message) -> None:
    """Helper to index a message if auto-indexing is enabled."""
    try:
        from app.embeddings import auto_index_enabled, index_message
        if auto_index_enabled():
            count = index_message(db, message, force=False)
            if count > 0:
                print(f"[memory.service] Auto-indexed message {message.id}: {count} embeddings")
    except Exception as e:
        # Don't fail the main operation if indexing fails
        print(f"[memory.service] Failed to auto-index message {message.id}: {e}")


def _sanitize_utf8(text: str) -> str:
    """Sanitize string to valid UTF-8, replacing surrogate pairs.
    
    Some emoji characters (e.g. \u2699\ufe0f) can produce surrogate pairs
    that cause 'surrogates not allowed' errors when SQLite/Python tries
    to encode them. This replaces any problematic characters with the
    Unicode replacement character.
    """
    if not text:
        return text
    try:
        # Round-trip through bytes to flush out surrogates
        return text.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='replace')
    except Exception:
        return text


def create_message(db: Session, data: schemas.MessageCreate) -> models.Message:
    """
    Create a new message in the database.
    
    v0.12.4: Now properly saves provider, model, and reasoning fields.
    v0.12.5: Sanitizes content to prevent UTF-8 surrogate encoding errors.
    """
    msg = models.Message(
        project_id=data.project_id,
        role=data.role,
        content=_sanitize_utf8(data.content),  # v0.12.5: Sanitize surrogates
        provider=data.provider,  # v0.12.4: Was missing
        model=data.model,  # v0.12.4: Was missing
        reasoning=data.reasoning,  # v0.12.4: New field
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    
    _index_message_if_enabled(db, msg)
    
    return msg


def list_messages(db: Session, project_id: int, limit: int = 100) -> List[models.Message]:
    return (
        db.query(models.Message)
        .filter(models.Message.project_id == project_id)
        .order_by(models.Message.created_at.asc())
        .limit(limit)
        .all()
    )


def delete_messages_for_project(db: Session, project_id: int) -> int:
    # Delete associated embeddings first
    try:
        from app.embeddings.models import Embedding
        db.query(Embedding).filter(
            Embedding.project_id == project_id,
            Embedding.source_type == "message",
        ).delete()
    except Exception as e:
        print(f"[memory.service] Failed to delete message embeddings for project {project_id}: {e}")
    
    count = db.query(models.Message).filter(models.Message.project_id == project_id).delete()
    db.commit()
    return count


# ============== MESSAGE HISTORY (NEW) ==============

def get_message_history(
    db: Session,
    project_id: int,
    limit: int = 50,
    before_id: Optional[int] = None,
) -> schemas.MessageHistoryResponse:
    """
    Get paginated message history for a project.
    
    v0.12.4: Now returns actual provider, model, and reasoning values.
    """
    limit = max(1, min(limit, 200))
    
    query = db.query(models.Message).filter(models.Message.project_id == project_id)
    
    if before_id is not None:
        query = query.filter(models.Message.id < before_id)
    
    messages_desc = (
        query
        .order_by(models.Message.id.desc())
        .limit(limit)
        .all()
    )
    
    messages = list(reversed(messages_desc))
    
    oldest_id = messages[0].id if messages else None
    
    has_older = False
    if oldest_id is not None:
        older_exists = (
            db.query(models.Message.id)
            .filter(models.Message.project_id == project_id)
            .filter(models.Message.id < oldest_id)
            .first()
        )
        has_older = older_exists is not None
    
    # v0.12.4: Include actual provider, model, and reasoning values
    message_items = [
        schemas.MessageHistoryItem(
            id=msg.id,
            project_id=msg.project_id,
            role=msg.role,
            content=msg.content,
            provider=msg.provider,  # v0.12.4: Was hardcoded to None
            model=msg.model,  # v0.12.4: New field
            reasoning=msg.reasoning,  # v0.12.4: New field
            created_at=msg.created_at,
        )
        for msg in messages
    ]
    
    return schemas.MessageHistoryResponse(
        messages=message_items,
        has_older=has_older,
        oldest_id=oldest_id,
    )


# ============== DOCUMENT CONTENT ==============

def create_document_content(
    db: Session, data: schemas.DocumentContentCreate
) -> models.DocumentContent:
    doc = models.DocumentContent(
        project_id=data.project_id,
        file_id=data.file_id,
        filename=data.filename,
        doc_type=data.doc_type,
        raw_text=data.raw_text,
        summary=data.summary,
        structured_data=data.structured_data,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def get_document_content_by_file_id(
    db: Session, file_id: int
) -> Optional[models.DocumentContent]:
    return (
        db.query(models.DocumentContent)
        .filter(models.DocumentContent.file_id == file_id)
        .first()
    )


def get_document_content_by_filename(
    db: Session, project_id: int, filename: str
) -> Optional[models.DocumentContent]:
    return (
        db.query(models.DocumentContent)
        .filter(models.DocumentContent.project_id == project_id)
        .filter(models.DocumentContent.filename.ilike(f"%{filename}%"))
        .order_by(models.DocumentContent.created_at.desc())
        .first()
    )


def list_document_contents(
    db: Session, project_id: int, doc_type: Optional[str] = None
) -> List[models.DocumentContent]:
    query = db.query(models.DocumentContent).filter(
        models.DocumentContent.project_id == project_id
    )
    if doc_type:
        query = query.filter(models.DocumentContent.doc_type == doc_type)
    return query.order_by(models.DocumentContent.created_at.desc()).all()


def search_document_contents(
    db: Session, project_id: int, search_term: str
) -> List[models.DocumentContent]:
    pattern = f"%{search_term}%"
    return (
        db.query(models.DocumentContent)
        .filter(models.DocumentContent.project_id == project_id)
        .filter(
            (models.DocumentContent.raw_text.ilike(pattern)) |
            (models.DocumentContent.summary.ilike(pattern)) |
            (models.DocumentContent.filename.ilike(pattern))
        )
        .order_by(models.DocumentContent.created_at.desc())
        .all()
    )


def get_latest_document_content(
    db: Session, project_id: int
) -> Optional[models.DocumentContent]:
    return (
        db.query(models.DocumentContent)
        .filter(models.DocumentContent.project_id == project_id)
        .order_by(models.DocumentContent.created_at.desc())
        .first()
    )


def delete_document_content(db: Session, doc_id: int) -> bool:
    doc = db.query(models.DocumentContent).filter(models.DocumentContent.id == doc_id).first()
    if not doc:
        return False
    
    # Delete associated embeddings
    try:
        from app.embeddings import delete_embeddings_for_source
        delete_embeddings_for_source(db, doc.project_id, "file", doc.file_id)
    except Exception as e:
        print(f"[memory.service] Failed to delete embeddings for doc {doc_id}: {e}")
    
    db.delete(doc)
    db.commit()
    return True
