# FILE: app/memory/schemas.py
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict


# ============== PROJECT ==============

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime


# ============== NOTE ==============

class NoteCreate(BaseModel):
    project_id: int
    title: str
    content: str
    tags: Optional[str] = None
    source: Optional[str] = None


class NoteCreateForProject(BaseModel):
    title: str
    content: str
    tags: Optional[str] = None
    source: Optional[str] = None


class NoteUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[str] = None
    source: Optional[str] = None


class NoteOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    title: str
    content: str
    tags: Optional[str]
    source: Optional[str]
    created_at: datetime
    updated_at: datetime


# ============== TASK ==============

class TaskCreate(BaseModel):
    project_id: int
    title: str
    description: Optional[str] = None
    status: str = "todo"
    priority: Optional[str] = None


class TaskCreateForProject(BaseModel):
    title: str
    description: Optional[str] = None
    status: str = "todo"
    priority: Optional[str] = None


class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None


class TaskOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    title: str
    description: Optional[str]
    status: str
    priority: Optional[str]
    created_at: datetime
    updated_at: datetime


# ============== FILE ==============

class FileCreate(BaseModel):
    project_id: int
    path: str
    original_name: str
    file_type: Optional[str] = None
    description: Optional[str] = None


class FileOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    path: str
    original_name: str
    file_type: Optional[str]
    description: Optional[str]
    created_at: datetime


# ============== MESSAGE ==============

class MessageCreate(BaseModel):
    project_id: int
    role: str
    content: str


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    role: str
    content: str
    created_at: datetime


# ============== DOCUMENT CONTENT ==============

class DocumentContentCreate(BaseModel):
    project_id: int
    file_id: int
    filename: str
    doc_type: Optional[str] = None
    raw_text: str
    summary: Optional[str] = None
    structured_data: Optional[str] = None


class DocumentContentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    file_id: int
    filename: str
    doc_type: Optional[str]
    raw_text: str
    summary: Optional[str]
    structured_data: Optional[str]
    created_at: datetime