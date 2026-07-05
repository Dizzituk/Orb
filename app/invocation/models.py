# FILE: app/invocation/models.py
# Purpose: Invocation models - Pydantic schemas for the ambient invocation pipeline.
# Called-by: app.invocation, app.invocation.classifier, app.invocation.router, app.invocation.tier_gate
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Invocation models - Pydantic schemas for the ambient invocation pipeline.

Kept intentionally small (~3KB). All business logic lives in the
classifier/tier_gate/router modules; this file defines shapes only.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class Intent(str, Enum):
    """What the user is trying to do when they invoke Astra."""
    CONVERSE      = "converse"
    LOOK_READ     = "look_read"
    LOOK_DESCRIBE = "look_describe"
    LOOK_EXPLAIN  = "look_explain"
    LOOK_DIAGNOSE = "look_diagnose"
    TAKEOVER      = "takeover"
    # WEB_ACTION (2026-07-03): an actionable web/study request from ambient voice
    # ("open my Coursera course", "next lesson", "resume studying"). Unlike the
    # tool-less CONVERSE/LOOK paths, the router runs the REAL chat tool loop for
    # this intent so voice can actually open + drive the browser, and reports
    # which session to surface via InvocationResponse.surface_session.
    WEB_ACTION    = "web_action"
    NOTIFY        = "notify"
    UNKNOWN       = "unknown"


class Tier(str, Enum):
    """Model tier for an invocation."""
    T1_MINI   = "t1_mini"
    T2_STD    = "t2_std"
    T3_MULTI  = "t3_multi"
    T4_OPUS   = "t4_opus"


class PromotionReason(str, Enum):
    """Why the tier gate promoted or declined to promote."""
    DEFAULT               = "default"
    CONTEXT_VOLUME        = "context_volume"
    CROSS_SOURCE          = "cross_source"
    ARCHITECTURAL_KEYWORD = "architectural_keyword"
    TOOL_ROUNDS_ESTIMATE  = "tool_rounds_estimate"
    EXPLICIT_PROMOTION    = "explicit_promotion"
    DEMOTED_SMALL_CONTEXT = "demoted_small_context"


class TierDecision(BaseModel):
    """Result of the tier promotion gate."""
    tier: Tier
    starting_tier: Tier
    reasons: List[PromotionReason] = Field(default_factory=list)
    cost_estimate_usd_low: float = 0.0
    cost_estimate_usd_high: float = 0.0
    notes: Optional[str] = None


class ClassifiedIntent(BaseModel):
    """Result of the intent classifier."""
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    matched_rule: Optional[str] = None
    needs_screenshot: bool = False


class InvocationRequest(BaseModel):
    """A single ambient invocation, from trigger to result."""
    transcript: str
    screenshot_b64: Optional[str] = None
    screenshot_mime: str = "image/png"
    active_window_title: Optional[str] = None
    active_window_app: Optional[str] = None
    context_bytes: int = 0
    source_count: int = 0
    explicit_tier: Optional[Tier] = None
    session_id: Optional[str] = None
    # Chat-UI mirroring (2026-07-03): id of the conversation the desktop has
    # OPEN when the wake-word turn fires. When set, the turn persists into
    # that conversation (so the open panel's live poll renders it) instead of
    # the fallback "Voice" project. None = old behaviour (phone, no chat open).
    chat_project_id: Optional[int] = None


class InvocationResponse(BaseModel):
    """What the router returns after processing an invocation."""
    intent: ClassifiedIntent
    tier_decision: TierDecision
    answer: str
    provider: Optional[str] = None
    model: Optional[str] = None
    error: Optional[str] = None
    elapsed_ms: int = 0
    # WEB_ACTION (2026-07-03): platform key or session id the frontend should
    # pull onscreen (e.g. "coursera") after a voice-driven web action. None for
    # the tool-less CONVERSE/LOOK paths. AmbientVoicePipeline calls
    # surfaceBrowserSession() with this so the wake-word path can bring the Web
    # tab forward — the ambient counterpart of the chat SURFACE_ON_TOOL list.
    surface_session: Optional[str] = None