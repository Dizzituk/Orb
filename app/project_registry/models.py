# FILE: app/project_registry/models.py
# Purpose: SQLAlchemy models for the ASTRA Project Registry.
# Called-by: app.db, app.project_registry, app.project_registry.scanner, app.project_registry.service (+1 more)
# Depends-on: app.db
# Last-renovated: 2026-06-11
"""
SQLAlchemy models for the ASTRA Project Registry.

Tracks external projects (apps ASTRA builds or manages) with their
scan metadata, root paths, and architecture state. Enables multi-project
architecture awareness — each project gets the same RAG treatment
as ASTRA's own codebase.

v1.0 (2026-03): Initial implementation.
"""
from __future__ import annotations

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, Index,
)
from app.db import Base


class RegisteredProject(Base):
    """
    A project ASTRA knows about and can scan/query architecture for.

    Each registered project gets scoped RAG entries and architecture
    scans, separate from the ASTRA core codebase.
    """
    __tablename__ = "registered_projects"

    id = Column(Integer, primary_key=True, index=True)

    # Human-readable name (e.g. "Driver CoPilot", "Bar Stock Manager")
    name = Column(String(200), nullable=False, unique=True)

    # Short slug for scoping (e.g. "driver-copilot", "bar-stock")
    slug = Column(String(100), nullable=False, unique=True, index=True)

    # Root path(s) on disk — JSON array of strings
    # e.g. ["D:\\Astra Android Folder\\AndroidDriverCopilot"]
    root_paths_json = Column(Text, nullable=False)

    # Project type for context (android, electron, python, web, etc.)
    project_type = Column(String(50), nullable=True)

    # Primary language (kotlin, typescript, python, etc.)
    primary_language = Column(String(50), nullable=True)

    # Brief description for context injection
    description = Column(Text, nullable=True)

    # ── Scan metadata ───────────────────────────────────────────────

    # Last successful scan timestamp
    last_scan_at = Column(DateTime, nullable=True)

    # Last scan ID (FK to architecture_scan_runs)
    last_scan_id = Column(Integer, nullable=True)

    # File count at last scan
    last_scan_file_count = Column(Integer, nullable=True, default=0)

    # Whether an architecture map file exists in the project folder
    has_arch_map = Column(Boolean, default=False, nullable=False)

    # Path to the architecture map file within the project
    arch_map_path = Column(String(500), nullable=True)

    # ── Lifecycle ───────────────────────────────────────────────────

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Soft delete
    is_active = Column(Boolean, default=True, nullable=False)

    __table_args__ = (
        Index("ix_regproj_active", "is_active"),
        Index("ix_regproj_last_scan", "last_scan_at"),
    )

    def __repr__(self) -> str:
        return f"<RegisteredProject(id={self.id}, name={self.name!r}, slug={self.slug!r})>"