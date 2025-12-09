# app/memory/models.py
"""
SQLAlchemy ORM models for Orb memory system.

Security Level 4: Sensitive fields use EncryptedText/EncryptedJSON types.
These require the master key to be initialized before any data operations.

v0.12.4: Added `reasoning` column to Message table to store chain-of-thought
from <THINKING> tags separately from the visible response content.

v0.12.1 FIX: Added `model` column to Message table to track which LLM model
generated each response. This was missing, causing model info to be lost.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from app.db import Base

# Import encrypted column types - REQUIRED for Security Level 4
from app.crypto import EncryptedText, EncryptedJSON

print("[models] Encryption types loaded successfully")


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)  # Not encrypted (metadata)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    notes = relationship("Note", back_populates="project", cascade="all, delete-orphan")
    tasks = relationship("Task", back_populates="project", cascade="all, delete-orphan")
    files = relationship("File", back_populates="project", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="project", cascade="all, delete-orphan")
    document_contents = relationship("DocumentContent", back_populates="project", cascade="all, delete-orphan")


class Note(Base):
    __tablename__ = "notes"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False)  # Not encrypted (for search/display)
    
    # ENCRYPTED: Note content is sensitive
    content = Column(EncryptedText, nullable=False)
    
    tags = Column(String(500), nullable=True)  # Not encrypted (for filtering)
    source = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    project = relationship("Project", back_populates="notes")


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False)  # Not encrypted (for listing)
    
    # ENCRYPTED: Task description may contain sensitive details
    description = Column(EncryptedText, nullable=True)
    
    status = Column(String(20), default="todo", nullable=False)  # Not encrypted (for filtering)
    priority = Column(String(10), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    project = relationship("Project", back_populates="tasks")


class File(Base):
    __tablename__ = "files"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    path = Column(String(500), nullable=False)  # Not encrypted (file path)
    original_name = Column(String(255), nullable=False)  # Not encrypted (for display)
    file_type = Column(String(100), nullable=True)  # Not encrypted (metadata)
    
    # ENCRYPTED: File description may be sensitive
    description = Column(EncryptedText, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    project = relationship("Project", back_populates="files")
    document_content = relationship("DocumentContent", back_populates="file", uselist=False, cascade="all, delete-orphan")


class Message(Base):
    """
    Chat message storage.
    
    v0.12.4: Added `reasoning` column to store chain-of-thought from <THINKING>
    tags separately from the visible response content. This allows the UI to
    optionally display reasoning in a collapsible panel without polluting the
    main message content.
    
    v0.12.1: Added `model` column to track which specific LLM model
    generated each assistant response (e.g., "gpt-4o-mini", "claude-sonnet-4-20250514").
    """
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # Not encrypted (metadata)
    
    # ENCRYPTED: Message content is sensitive (chat history)
    content = Column(EncryptedText, nullable=False)
    
    # v0.16.0: Track which LLM provider generated the response
    # For user messages: "local"
    # For assistant messages: "openai", "anthropic", "google"
    provider = Column(String(50), nullable=True)
    
    # v0.12.1 FIX: Track which specific model generated the response
    # For user messages: None
    # For assistant messages: "gpt-4o-mini", "claude-sonnet-4-20250514", "gemini-2.0-flash", etc.
    model = Column(String(100), nullable=True)
    
    # v0.12.4: Store chain-of-thought reasoning from <THINKING> tags
    # ENCRYPTED: Reasoning content is sensitive (internal model reasoning)
    # For user messages: None
    # For assistant messages: Content from <THINKING>...</THINKING> block (if present)
    reasoning = Column(EncryptedText, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    project = relationship("Project", back_populates="messages")


class DocumentContent(Base):
    """
    Stores extracted text content from uploaded files.
    Used for RAG/retrieval when user asks questions about their documents.
    """
    __tablename__ = "document_contents"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    file_id = Column(Integer, ForeignKey("files.id"), nullable=False, unique=True, index=True)
    
    # Original filename for easy lookup (not encrypted - for display)
    filename = Column(String(255), nullable=False)
    
    # Document type: cv, report, notes, code, etc. (not encrypted - for filtering)
    doc_type = Column(String(50), nullable=True)
    
    # ENCRYPTED: Raw extracted text is highly sensitive
    raw_text = Column(EncryptedText, nullable=False)
    
    # ENCRYPTED: LLM-generated summary
    summary = Column(EncryptedText, nullable=True)
    
    # ENCRYPTED: Structured data as JSON string (for CVs: roles, skills, etc.)
    structured_data = Column(EncryptedText, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    project = relationship("Project", back_populates="document_contents")
    file = relationship("File", back_populates="document_content")