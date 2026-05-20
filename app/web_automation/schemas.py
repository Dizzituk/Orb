# FILE: app/web_automation/schemas.py
"""
Pydantic schemas for the Web Automation REST surface.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Sessions ─────────────────────────────────────────────────────────

class WebSessionOut(BaseModel):
    id: str
    platform: str
    label: str
    partition: str
    landing_url: str
    purpose: Optional[str] = None
    status: str
    current_url: Optional[str] = None
    current_title: Optional[str] = None
    last_error: Optional[str] = None
    last_used_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class WebSessionCreate(BaseModel):
    platform: str = Field(..., min_length=1, max_length=64)
    label: str = Field(..., min_length=1, max_length=128)
    partition: str = Field(..., min_length=1, max_length=128)
    landing_url: str = Field(..., min_length=1, max_length=2048)
    purpose: Optional[str] = None


class WebSessionPatch(BaseModel):
    label: Optional[str] = None
    landing_url: Optional[str] = None
    purpose: Optional[str] = None


# ── Actions ──────────────────────────────────────────────────────────

class ActionRequest(BaseModel):
    session_id: str
    action_type: str
    payload: dict = Field(default_factory=dict)
    timeout_seconds: float = 30.0


class ActionOut(BaseModel):
    id: str
    session_id: str
    action_type: str
    payload: dict
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: datetime
    delivered_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ActionResultIn(BaseModel):
    """Posted by Electron when an action completes."""
    ok: bool
    result: Optional[dict] = None
    error: Optional[str] = None
    current_url: Optional[str] = None
    current_title: Optional[str] = None


# ── Electron polling ─────────────────────────────────────────────────

class PendingActionOut(BaseModel):
    """Shape returned to Electron's long-poll. All None => no pending action."""
    action_id: Optional[str] = None
    session_id: Optional[str] = None
    action_type: Optional[str] = None
    payload: Optional[dict] = None


# ── Session state from Electron ──────────────────────────────────────

class SessionStatusIn(BaseModel):
    """Electron reports live state back (session opened / closed / crashed)."""
    status: str                              # idle | opening | live | error
    current_url: Optional[str] = None
    current_title: Optional[str] = None
    error: Optional[str] = None
