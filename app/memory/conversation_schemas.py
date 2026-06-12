# FILE: app/memory/conversation_schemas.py
# Purpose: Pydantic schemas for the Conversational Memory Layer.
# Called-by: app.memory.conversation_service, app.memory.summary_generator, tests.test_conversation_memory
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Pydantic schemas for the Conversational Memory Layer.

Covers CRUD operations for ConversationSession and ConversationSummary.

Part of CONV-MEMORY-001: Phase 1 (Session Model + Storage).
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict


# =========================================================================
# ConversationSession schemas
# =========================================================================

class SessionCreate(BaseModel):
    """Create a new conversation session for a project."""
    project_id: int


class SessionOut(BaseModel):
    """Read-only view of a conversation session."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    status: str
    started_at: datetime
    last_activity_at: datetime
    message_count: int
    models_used: str


class SessionUpdate(BaseModel):
    """Partial update for a session (status changes, activity bump)."""
    status: Optional[str] = None
    message_count: Optional[int] = None
    models_used: Optional[str] = None


# =========================================================================
# ConversationSummary schemas
# =========================================================================

class SummaryCreate(BaseModel):
    """Create a new summary version for a session."""
    session_id: int
    project_id: int
    summary_json: Dict[str, Any]
    summarised_through_message_id: Optional[int] = None
    token_estimate: int = 0


class SummaryOut(BaseModel):
    """Read-only view of a conversation summary."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: int
    project_id: int
    version: int
    summarised_through_message_id: Optional[int]
    summary_json: Optional[Dict[str, Any]]
    token_estimate: int
    created_at: datetime
    updated_at: datetime


# =========================================================================
# Summary JSON structure (used by Phase 2 generator, defined here for typing)
# =========================================================================

class SummaryContent(BaseModel):
    """
    Structured content inside summary_json.

    Phase 2 (summary_generator.py) produces this; Phase 3 injects it
    into the LLM context window. Defined here so both phases share
    the same schema.
    """
    current_topic: str = ""
    key_decisions: List[str] = []
    open_items: List[str] = []
    technical_context: str = ""
    attachments_described: List[str] = []
    user_preferences_expressed: List[str] = []
    # Phase 5: dedicated slot for durable biographical statements made
    # during the session (DOB, birthplace, location changes, job changes,
    # family, etc). Preserves hard facts through summarisation so the
    # knowledge extractor has something to work with.
    biographical_facts: List[str] = []
    models_involved: List[str] = []
    message_range: Optional[Dict[str, int]] = None  # {"from_id": N, "to_id": M}


# =========================================================================
# Composite views (for debug endpoints / testing)
# =========================================================================

class SessionWithSummary(BaseModel):
    """Session plus its latest summary, for inspection."""
    model_config = ConfigDict(from_attributes=True)

    session: SessionOut
    latest_summary: Optional[SummaryOut] = None
