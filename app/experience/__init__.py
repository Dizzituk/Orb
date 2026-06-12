# FILE: app/experience/__init__.py
# Purpose: ASTRA Experience Database — Unified Memory System Layer 1.
# Called-by: app.db, app.orchestrator.final_checkout
# Depends-on: app.experience.journal_writer, app.experience.schemas
# Last-renovated: 2026-06-11
"""
ASTRA Experience Database — Unified Memory System Layer 1.

Capture, structure, retrieve, and inject lessons learned from pipeline runs.

Components:
- schemas: Data structures for journal entries, patterns, and events
- journal_writer: NDJSON Build Journal writer with append hooks
- models: SQLAlchemy models for ExperiencePattern storage
- experience_store: CRUD operations for patterns
- distillation: Post-job journal → pattern extraction
- retrieval: Two-stage retrieval (indexed filter → semantic ranking)
"""

from .schemas import (
    BuildJournalEntry,
    EventSeverity,
    JournalEventType,
)
from .journal_writer import (
    JournalWriter,
    get_journal_writer,
    emit_journal_entry,
)

__all__ = [
    "BuildJournalEntry",
    "EventSeverity",
    "JournalEventType",
    "JournalWriter",
    "get_journal_writer",
    "emit_journal_entry",
]
