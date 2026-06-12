# FILE: app/content/schemas.py
# Purpose: Pydantic schemas for Content Pipeline API validation.
# Called-by: app.content.router
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Pydantic schemas for Content Pipeline API validation.

Request/response models for all content pipeline endpoints.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


# ─── CONVERSATION SCHEMAS ───

class ConversationCreate(BaseModel):
    """Start a new content-tracked conversation."""
    linked_video_path: Optional[str] = None


class ConversationEnd(BaseModel):
    """End a conversation and trigger deep analysis."""
    conversation_id: str
    transcript_raw: Optional[str] = None


class ConversationSummary(BaseModel):
    """Lightweight conversation listing."""
    id: str
    timestamp_start: datetime
    timestamp_end: Optional[datetime]
    duration_seconds: Optional[int]
    tag_count: int = 0
    scout_processed: bool = False
    deep_analysis_done: bool = False


# ─── CONTENT TAG SCHEMAS ───

class ContentTagOut(BaseModel):
    """A tagged content moment."""
    id: str
    conversation_id: str
    tag_type: str
    content_category: Optional[str]
    excerpt: str
    strength_score: float
    topic_name: Optional[str] = None
    converted: bool = False


# ─── TOPIC SCHEMAS ───

class TopicCreate(BaseModel):
    """Create or update a topic."""
    name: str
    description: Optional[str] = None


class TopicOut(BaseModel):
    """Topic with tracking metadata."""
    id: str
    name: str
    description: Optional[str]
    first_discussed: datetime
    last_discussed: datetime
    discussion_count: int
    maturity_score: float
    published_piece_count: int = 0


class TopicPositionUpdate(BaseModel):
    """Record an evolution in the user's position on a topic."""
    topic_id: str
    summary: str
    key_arguments: List[str] = []
    evidence_cited: List[str] = []


# ─── CONTENT PIECE SCHEMAS ───

class ContentPieceProposal(BaseModel):
    """A content opportunity proposed by the Content Scout."""
    id: str
    title: str
    description: Optional[str]
    content_category: str
    topic_name: Optional[str] = None
    series_name: Optional[str] = None
    overall_score: Optional[float]
    recommended_formats: List[str] = []
    suggested_hooks: List[str] = []
    key_excerpts: List[str] = []
    previously_covered: bool = False
    last_published_on_topic: Optional[datetime] = None


class ContentPieceApproval(BaseModel):
    """User decision on a content piece."""
    piece_id: str
    decision: str = Field(..., pattern="^(approved|rejected|deferred|merged)$")
    modifications: Optional[Dict[str, Any]] = None
    rejection_reason: Optional[str] = None
    merge_with_piece_id: Optional[str] = None


class ContentPieceOut(BaseModel):
    """Full content piece with current status."""
    id: str
    title: str
    description: Optional[str]
    content_category: str
    status: str
    topic_name: Optional[str] = None
    series_name: Optional[str] = None
    overall_score: Optional[float]
    recommended_formats: List[str] = []
    output_count: int = 0
    created_at: datetime
    updated_at: datetime
    published_at: Optional[datetime] = None


# ─── SERIES SCHEMAS ───

class SeriesCreate(BaseModel):
    """Create a content series."""
    name: str
    description: Optional[str] = None
    categories: List[str] = []
    target_formats: List[str] = []
    target_platforms: List[str] = []
    posting_cadence: Optional[str] = None


class SeriesOut(BaseModel):
    """Series with metadata."""
    id: str
    name: str
    description: Optional[str]
    categories: List[str]
    target_formats: List[str]
    target_platforms: List[str]
    posting_cadence: Optional[str]
    active: bool
    piece_count: int = 0


# ─── END-OF-DAY REVIEW SCHEMAS ───

class DailyReviewOut(BaseModel):
    """End-of-day content review summary."""
    date: str
    conversations_count: int
    total_duration_minutes: int
    topics_discussed: List[str]
    proposals: List[ContentPieceProposal]
    deferred_count: int = 0


# ─── ANALYTICS SCHEMAS ───

class AnalyticsSnapshot(BaseModel):
    """Platform analytics for a published output."""
    output_id: str
    views: int = 0
    likes: int = 0
    comments: int = 0
    shares: int = 0
    saves: int = 0
    avg_watch_time_seconds: Optional[float] = None
    completion_rate: Optional[float] = None
    engagement_rate: Optional[float] = None
    follower_delta: int = 0


# ─── STYLE PROFILE SCHEMAS ───

class StyleProfileOut(BaseModel):
    """Current active style profile."""
    id: str
    name: str
    video_params: Dict[str, Any]
    voice_profile: Dict[str, Any]
    reference_video_count: int = 0
    active: bool
