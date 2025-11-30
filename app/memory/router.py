# file: app/memory/router.py
from typing import List, Optional
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File as FastAPIFile, Form
from sqlalchemy.orm import Session

from app.db import get_db
from app.memory import service, schemas
from app.auth import require_auth

router = APIRouter(
    prefix="/memory",
    tags=["memory"],
    dependencies=[Depends(require_auth)],
)


# ============== PROJECTS ==============

@router.post("/projects", response_model=schemas.ProjectOut, status_code=201)
def create_project(data: schemas.ProjectCreate, db: Session = Depends(get_db)):
    existing = service.get_project_by_name(db, data.name)
    if existing:
        raise HTTPException(status_code=400, detail="Project name already exists")
    return service.create_project(db, data)


@router.get("/projects", response_model=List[schemas.ProjectOut])
def list_projects(db: Session = Depends(get_db)):
    return service.list_projects(db)


@router.get("/projects/{project_id}", response_model=schemas.ProjectOut)
def get_project(project_id: int, db: Session = Depends(get_db)):
    project = service.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.patch("/projects/{project_id}", response_model=schemas.ProjectOut)
def update_project(project_id: int, data: schemas.ProjectUpdate, db: Session = Depends(get_db)):
    project = service.update_project(db, project_id, data)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.delete("/projects/{project_id}", status_code=204)
def delete_project(project_id: int, db: Session = Depends(get_db)):
    success = service.delete_project(db, project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    return None


# ============== NOTES (FLAT) ==============

@router.post("/notes", response_model=schemas.NoteOut, status_code=201)
def create_note(data: schemas.NoteCreate, db: Session = Depends(get_db)):
    project = service.get_project(db, data.project_id)
    if not project:
        raise HTTPException(status_code=400, detail="Project not found")
    return service.create_note(db, data)


@router.get("/notes", response_model=List[schemas.NoteOut])
def list_notes(
    project_id: int,
    tag: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db),
):
    return service.list_notes(db, project_id, tag_filter=tag, search=search)


@router.get("/notes/{note_id}", response_model=schemas.NoteOut)
def get_note(note_id: int, db: Session = Depends(get_db)):
    note = service.get_note(db, note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


@router.patch("/notes/{note_id}", response_model=schemas.NoteOut)
def update_note(note_id: int, data: schemas.NoteUpdate, db: Session = Depends(get_db)):
    note = service.update_note(db, note_id, data)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


@router.delete("/notes/{note_id}", status_code=204)
def delete_note(note_id: int, db: Session = Depends(get_db)):
    success = service.delete_note(db, note_id)
    if not success:
        raise HTTPException(status_code=404, detail="Note not found")
    return None


# ============== NOTES (NESTED UNDER PROJECT) ==============

@router.post("/projects/{project_id}/notes", response_model=schemas.NoteOut, status_code=201)
def create_note_for_project(
    project_id: int,
    data: schemas.NoteCreateForProject,
    db: Session = Depends(get_db),
):
    """Create a note for a specific project (project_id in URL, not body)."""
    project = service.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return service.create_note_for_project(db, project_id, data)


@router.get("/projects/{project_id}/notes", response_model=List[schemas.NoteOut])
def list_notes_for_project(
    project_id: int,
    tag: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """List all notes for a specific project."""
    project = service.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return service.list_notes(db, project_id, tag_filter=tag, search=search)


# ============== TASKS (FLAT) ==============

@router.post("/tasks", response_model=schemas.TaskOut, status_code=201)
def create_task(data: schemas.TaskCreate, db: Session = Depends(get_db)):
    project = service.get_project(db, data.project_id)
    if not project:
        raise HTTPException(status_code=400, detail="Project not found")
    return service.create_task(db, data)


@router.get("/tasks", response_model=List[schemas.TaskOut])
def list_tasks(
    project_id: int,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
):
    return service.list_tasks(db, project_id, status=status)


@router.get("/tasks/{task_id}", response_model=schemas.TaskOut)
def get_task(task_id: int, db: Session = Depends(get_db)):
    task = service.get_task(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.patch("/tasks/{task_id}", response_model=schemas.TaskOut)
def update_task(task_id: int, data: schemas.TaskUpdate, db: Session = Depends(get_db)):
    task = service.update_task(db, task_id, data)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.delete("/tasks/{task_id}", status_code=204)
def delete_task(task_id: int, db: Session = Depends(get_db)):
    success = service.delete_task(db, task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found")
    return None


# ============== TASKS (NESTED UNDER PROJECT) ==============

@router.post("/projects/{project_id}/tasks", response_model=schemas.TaskOut, status_code=201)
def create_task_for_project(
    project_id: int,
    data: schemas.TaskCreateForProject,
    db: Session = Depends(get_db),
):
    """Create a task for a specific project (project_id in URL, not body)."""
    project = service.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return service.create_task_for_project(db, project_id, data)


@router.get("/projects/{project_id}/tasks", response_model=List[schemas.TaskOut])
def list_tasks_for_project(
    project_id: int,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """List all tasks for a specific project."""
    project = service.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return service.list_tasks(db, project_id, status=status)


# ============== FILES ==============

@router.post(
    "/projects/{project_id}/files/upload",
    response_model=schemas.FileOut,
    status_code=201,
)
async def upload_file_for_project(
    project_id: int,
    file: UploadFile = FastAPIFile(...),
    description: Optional[str] = Form(None),
    file_type: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """
    Upload a file for a specific project.

    - Validates that the project exists.
    - Saves the file under data/files/{project_id}/ with a unique filename.
    - Creates a File row via the service layer.
    """
    project = service.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    data_root = Path("data")
    project_dir = data_root / "files" / str(project_id)
    project_dir.mkdir(parents=True, exist_ok=True)

    original_name = file.filename or "uploaded_file"
    suffix = Path(original_name).suffix
    unique_name = f"{uuid4().hex}{suffix}"
    file_path = project_dir / unique_name

    # Read and write file contents
    contents = await file.read()
    file_path.write_bytes(contents)

    # Relative path from data/ as required by schema
    relative_path = file_path.relative_to(data_root)
    normalized_relative_path = str(relative_path).replace("\\", "/")

    # Infer file_type if not provided
    inferred_type: Optional[str] = file_type or file.content_type
    if not inferred_type and suffix:
        inferred_type = suffix.lstrip(".")  # e.g. "md", "txt"

    file_create = schemas.FileCreate(
        project_id=project_id,
        path=normalized_relative_path,
        original_name=original_name,
        file_type=inferred_type,
        description=description or "",
    )

    file_record = service.create_file_for_project(db, project_id, file_create)
    return file_record


@router.post("/files", response_model=schemas.FileOut, status_code=201)
def create_file(data: schemas.FileCreate, db: Session = Depends(get_db)):
    project = service.get_project(db, data.project_id)
    if not project:
        raise HTTPException(status_code=400, detail="Project not found")
    return service.create_file(db, data)


@router.get("/files", response_model=List[schemas.FileOut])
def list_files(project_id: int, db: Session = Depends(get_db)):
    return service.list_files(db, project_id)


@router.get("/files/{file_id}", response_model=schemas.FileOut)
def get_file(file_id: int, db: Session = Depends(get_db)):
    file_record = service.get_file(db, file_id)
    if not file_record:
        raise HTTPException(status_code=404, detail="File not found")
    return file_record


@router.delete("/files/{file_id}", status_code=204)
def delete_file(file_id: int, db: Session = Depends(get_db)):
    success = service.delete_file(db, file_id)
    if not success:
        raise HTTPException(status_code=404, detail="File not found")
    return None


# ============== MESSAGES ==============

@router.post("/messages", response_model=schemas.MessageOut, status_code=201)
def create_message(data: schemas.MessageCreate, db: Session = Depends(get_db)):
    project = service.get_project(db, data.project_id)
    if not project:
        raise HTTPException(status_code=400, detail="Project not found")
    return service.create_message(db, data)


@router.get("/messages", response_model=List[schemas.MessageOut])
def list_messages(project_id: int, limit: int = 100, db: Session = Depends(get_db)):
    return service.list_messages(db, project_id, limit=limit)


@router.get("/projects/{project_id}/messages", response_model=List[schemas.MessageOut])
def list_messages_for_project(project_id: int, limit: int = 100, db: Session = Depends(get_db)):
    """List all messages for a specific project (nested endpoint)."""
    project = service.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return service.list_messages(db, project_id, limit=limit)


@router.delete("/projects/{project_id}/messages", status_code=200)
def clear_messages_for_project(project_id: int, db: Session = Depends(get_db)):
    """Delete all messages for a project (useful for clearing chat history)."""
    project = service.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    count = service.delete_messages_for_project(db, project_id)
    return {"deleted": count}