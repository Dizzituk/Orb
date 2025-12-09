# FILE: app/memory/schemas.py
"""
Memory module Pydantic schemas.

v0.12.4: Added model and reasoning fields to MessageCreate and MessageHistoryItem.
"""
from datetime import datetime
from typing import Optional, List, Literal
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
    """
    Schema for creating a new message.
    
    v0.12.4: Added model and reasoning fields.
    - model: Which specific model generated the response (e.g., "gpt-4.1-mini")
    - reasoning: Chain-of-thought reasoning from <THINKING> tags (hidden from UI)
    """
    project_id: int
    role: str
    content: str
    provider: Optional[str] = None  # v0.16.0: Track which LLM provider generated the message
    model: Optional[str] = None  # v0.12.4: Track which model generated the response
    reasoning: Optional[str] = None  # v0.12.4: Store reasoning separately from content


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    role: str
    content: str
    created_at: datetime


# ============== MESSAGE HISTORY (v0.12.4 Updated) ==============

class MessageHistoryItem(BaseModel):
    """
    Single message item for history response.
    
    v0.12.4: Added model and reasoning fields for frontend display.
    """
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    role: Literal["user", "assistant", "system"]
    content: str
    provider: Optional[str] = None
    model: Optional[str] = None  # v0.12.4: Model that generated the response
    reasoning: Optional[str] = None  # v0.12.4: Chain-of-thought reasoning (hidden from UI)
    created_at: datetime


class MessageHistoryResponse(BaseModel):
    """Response schema for message history endpoint."""
    messages: List[MessageHistoryItem]
    has_older: bool
    oldest_id: Optional[int] = None


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