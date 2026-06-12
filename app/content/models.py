# FILE: app/content/models.py
# Purpose: Content Memory Database Models (Spec Section 5)
# Called-by: app.content.distribution.analytics, app.content.distribution.posting_time_learner, app.content.distribution.publisher, app.content.distribution.scheduler (+16 more)
# Depends-on: app.db
# Last-renovated: 2026-06-11
"""
Content Memory Database Models (Spec Section 5)

SQLAlchemy ORM models for the Content Creation Pipeline.
Handles: conversations, content tags, topics, content pieces,
series, style profiles, and analytics feedback.

Part of ASTRA Content Creation Pipeline v1.0
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean,
    DateTime, ForeignKey, JSON, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from app.db import Base


# ─── HELPERS ───

def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ─── ENUMS ───

CONTENT_CATEGORIES = ("opinion", "educational", "documentary", "tutorial")
CONTENT_STATUSES = (
    "identified",    # Tagged by content scout
    "proposed",      # Presented in end-of-day review
    "approved",      # User approved for production
    "deferred",      # Saved for later
    "rejected",      # User rejected
    "in_production", # Being produced
    "review",        # Awaiting final human gate review
    "published",     # Live on platform(s)
    "archived",      # Removed from active rotation
)
APPROVAL_TYPES = ("cutaway", "draft", "final", "schedule")
OUTPUT_FORMATS = (
    "instagram_reel", "instagram_carousel", "youtube_short",
    "youtube_longform", "tiktok", "facebook_video",
    "blog_post", "twitter_thread",
)


# ═══════════════════════════════════════════════════
# CONVERSATION TRACKING (Spec Section 3)
# ═══════════════════════════════════════════════════

class ContentConversation(Base):
    """
    A recorded conversation session with content potential.
    Maps to Spec Section 3.2 — Conversation Transcript Schema.
    """
    __tablename__ = "content_conversations"

    id = Column(String, primary_key=True, default=_uuid)
    timestamp_start = Column(DateTime, nullable=False, default=_now)
    timestamp_end = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)

    # Raw transcript from voice-to-text
    transcript_raw = Column(Text, nullable=True)

    # Structured segments as JSON array
    # [{timestamp, speaker, text, segment_index}]
    transcript_segments = Column(JSON, nullable=True)

    # Mood/energy arc as JSON array
    # [{timestamp, energy_score, mood_label}]
    mood_arc = Column(JSON, nullable=True)

    # Reference to companion video file (if recorded simultaneously)
    linked_video_path = Column(String, nullable=True)

    # Processing state
    scout_processed = Column(Boolean, default=False)
    deep_analysis_done = Column(Boolean, default=False)

    created_at = Column(DateTime, default=_now)

    # Relationships
    content_tags = relationship(
        "ContentTag", back_populates="conversation",
        cascade="all, delete-orphan"
    )


# ═══════════════════════════════════════════════════
# CONTENT SCOUT TAGS (Spec Section 4)
# ═══════════════════════════════════════════════════

class ContentTag(Base):
    """
    A moment tagged by the Content Scout during or after conversation.
    Maps to Spec Section 4.1 — Real-Time Tagging.
    """
    __tablename__ = "content_tags"

    id = Column(String, primary_key=True, default=_uuid)
    conversation_id = Column(
        String, ForeignKey("content_conversations.id", ondelete="CASCADE"),
        nullable=False
    )

    # When in the conversation this moment occurred
    timestamp_offset_seconds = Column(Float, nullable=True)

    # Tag classification
    # novel_argument | clear_explanation | strong_opinion |
    # emotional_resonance | data_point | quotable_moment
    tag_type = Column(String, nullable=False)

    # Content category: opinion | educational | documentary | tutorial
    content_category = Column(String, nullable=True)

    # The tagged transcript excerpt
    excerpt = Column(Text, nullable=False)

    # Strength score (0.0 to 1.0)
    strength_score = Column(Float, default=0.5)

    # Topic assignment (linked after deep analysis)
    topic_id = Column(
        String, ForeignKey("content_topics.id", ondelete="SET NULL"),
        nullable=True
    )

    # Whether this tag has been converted into a content piece
    converted = Column(Boolean, default=False)
    content_piece_id = Column(
        String, ForeignKey("content_pieces.id", ondelete="SET NULL"),
        nullable=True
    )

    created_at = Column(DateTime, default=_now)

    # Relationships
    conversation = relationship("ContentConversation", back_populates="content_tags")
    topic = relationship("ContentTopic", back_populates="tags")


# ═══════════════════════════════════════════════════
# TOPIC TRACKING (Spec Section 5.2)
# ═══════════════════════════════════════════════════

class ContentTopic(Base):
    """
    An evolving topic the user discusses across conversations.
    Maps to Spec Section 5.2 — Topic Tracking.
    """
    __tablename__ = "content_topics"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False, index=True)

    # Topic description / summary
    description = Column(Text, nullable=True)

    # Temporal tracking
    first_discussed = Column(DateTime, nullable=False, default=_now)
    last_discussed = Column(DateTime, nullable=False, default=_now)
    discussion_count = Column(Integer, default=1)

    # Position history as JSON array
    # [{date, summary, key_arguments, evidence_cited}]
    position_history = Column(JSON, default=list)

    # Related topics as JSON array of topic_ids
    related_topic_ids = Column(JSON, default=list)

    # How refined the user's thinking is (0.0 to 1.0)
    maturity_score = Column(Float, default=0.1)

    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)

    # Relationships
    tags = relationship("ContentTag", back_populates="topic")
    pieces = relationship("ContentPiece", back_populates="topic")


# ═══════════════════════════════════════════════════
# CONTENT PIECES (Spec Section 6 & 7)
# ═══════════════════════════════════════════════════

class ContentPiece(Base):
    """
    A single content piece from identification through to publication.
    Tracks the full lifecycle: identified → proposed → approved →
    in_production → review → published.
    """
    __tablename__ = "content_pieces"

    id = Column(String, primary_key=True, default=_uuid)

    # What this piece is about
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)

    # Classification
    content_category = Column(String, nullable=False)  # opinion/educational/etc
    series_id = Column(
        String, ForeignKey("content_series.id", ondelete="SET NULL"),
        nullable=True
    )
    topic_id = Column(
        String, ForeignKey("content_topics.id", ondelete="SET NULL"),
        nullable=True
    )

    # Source conversation(s) as JSON array of conversation_ids
    source_conversation_ids = Column(JSON, default=list)

    # Source tags as JSON array of tag_ids
    source_tag_ids = Column(JSON, default=list)

    # Lifecycle status
    status = Column(String, default="identified", index=True)

    # Scores from content scout
    originality_score = Column(Float, nullable=True)
    audience_relevance_score = Column(Float, nullable=True)
    emotional_impact_score = Column(Float, nullable=True)
    educational_value_score = Column(Float, nullable=True)
    overall_score = Column(Float, nullable=True)

    # Recommended output formats as JSON array
    recommended_formats = Column(JSON, default=list)

    # Suggested hooks as JSON array
    suggested_hooks = Column(JSON, default=list)

    # Key transcript excerpts as JSON array
    key_excerpts = Column(JSON, default=list)

    # Production data
    draft_text = Column(Text, nullable=True)
    edit_decision_list = Column(JSON, nullable=True)

    # Cutaway concepts as JSON array
    # [{type, description, placement, status, asset_path}]
    cutaway_concepts = Column(JSON, default=list)

    # User feedback / modifications
    user_notes = Column(Text, nullable=True)
    rejection_reason = Column(Text, nullable=True)

    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)
    approved_at = Column(DateTime, nullable=True)
    published_at = Column(DateTime, nullable=True)

    # Relationships
    topic = relationship("ContentTopic", back_populates="pieces")
    series = relationship("ContentSeries", back_populates="pieces")
    outputs = relationship(
        "ContentOutput", back_populates="piece",
        cascade="all, delete-orphan"
    )
    approvals = relationship(
        "ContentApproval", back_populates="piece",
        cascade="all, delete-orphan"
    )


# ═══════════════════════════════════════════════════
# CONTENT OUTPUTS (Spec Section 9)
# ═══════════════════════════════════════════════════

class ContentOutput(Base):
    """
    A platform-specific output generated from a content piece.
    One piece can produce multiple outputs (multi-format).
    """
    __tablename__ = "content_outputs"

    id = Column(String, primary_key=True, default=_uuid)
    piece_id = Column(
        String, ForeignKey("content_pieces.id", ondelete="CASCADE"),
        nullable=False
    )

    # Output format: instagram_reel, youtube_short, blog_post, etc.
    output_format = Column(String, nullable=False)

    # Platform target
    platform = Column(String, nullable=False)

    # File paths for generated assets
    # Primary output (video file, HTML file, image set, etc.)
    primary_asset_path = Column(String, nullable=True)
    # Thumbnail
    thumbnail_path = Column(String, nullable=True)
    # Caption / description text
    caption_text = Column(Text, nullable=True)

    # Platform-specific metadata
    # {title, description, tags, hashtags, scheduled_time, etc.}
    platform_metadata = Column(JSON, default=dict)

    # Publishing state
    scheduled_at = Column(DateTime, nullable=True)
    published_at = Column(DateTime, nullable=True)
    platform_post_id = Column(String, nullable=True)  # ID returned by platform API

    # Device routing: phone | desktop
    publish_device = Column(String, default="desktop")

    created_at = Column(DateTime, default=_now)

    # Relationships
    piece = relationship("ContentPiece", back_populates="outputs")
    analytics = relationship(
        "ContentAnalytics", back_populates="output",
        cascade="all, delete-orphan"
    )


# ═══════════════════════════════════════════════════
# ANALYTICS FEEDBACK (Spec Section 9.3)
# ═══════════════════════════════════════════════════

class ContentAnalytics(Base):
    """
    Engagement metrics pulled from platform APIs after publishing.
    Feeds back into content scoring and style profile optimisation.
    """
    __tablename__ = "content_analytics"

    id = Column(String, primary_key=True, default=_uuid)
    output_id = Column(
        String, ForeignKey("content_outputs.id", ondelete="CASCADE"),
        nullable=False
    )

    # Snapshot timestamp (metrics are pulled periodically)
    snapshot_at = Column(DateTime, default=_now)

    # Core metrics
    views = Column(Integer, default=0)
    likes = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    shares = Column(Integer, default=0)
    saves = Column(Integer, default=0)

    # Retention / watch metrics (video only)
    avg_watch_time_seconds = Column(Float, nullable=True)
    completion_rate = Column(Float, nullable=True)  # 0.0 to 1.0
    replay_rate = Column(Float, nullable=True)

    # Growth metrics
    follower_delta = Column(Integer, default=0)  # New followers from this post
    click_through_rate = Column(Float, nullable=True)

    # Engagement rate (computed)
    engagement_rate = Column(Float, nullable=True)

    # Traffic sources as JSON
    # {home_feed: 0.4, search: 0.2, hashtags: 0.1, ...}
    traffic_sources = Column(JSON, nullable=True)

    # Retention graph data as JSON array (video only)
    # [{second, retention_pct}]
    retention_graph = Column(JSON, nullable=True)

    # Relationships
    output = relationship("ContentOutput", back_populates="analytics")


# ═══════════════════════════════════════════════════
# SERIES MANAGEMENT (Spec Section 10)
# ═══════════════════════════════════════════════════

class ContentSeries(Base):
    """
    A named content series with its own visual identity and scope.
    Maps to Spec Section 10.
    """
    __tablename__ = "content_series"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False, unique=True)
    description = Column(Text, nullable=True)

    # Which content categories this series covers
    # JSON array: ["opinion", "educational"]
    categories = Column(JSON, default=list)

    # Which output formats this series targets
    # JSON array: ["youtube_short", "instagram_reel"]
    target_formats = Column(JSON, default=list)

    # Which platforms this series publishes to
    # JSON array: ["youtube", "instagram", "tiktok"]
    target_platforms = Column(JSON, default=list)

    # Visual identity config as JSON
    # {intro_template, colour_palette, fonts, thumbnail_style,
    #  audio_profile, caption_style}
    visual_config = Column(JSON, default=dict)

    # Posting cadence target (e.g., "3_per_week")
    posting_cadence = Column(String, nullable=True)

    # Active flag
    active = Column(Boolean, default=True)

    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)

    # Relationships
    pieces = relationship("ContentPiece", back_populates="series")


# ═══════════════════════════════════════════════════
# APPROVAL LOG (Spec Section 8)
# ═══════════════════════════════════════════════════

class ContentApproval(Base):
    """
    Tracks every approval/rejection decision for preference learning.
    Maps to Spec Section 8.3 — Preference Learning Model.
    """
    __tablename__ = "content_approvals"

    id = Column(String, primary_key=True, default=_uuid)
    piece_id = Column(
        String, ForeignKey("content_pieces.id", ondelete="CASCADE"),
        nullable=False
    )

    # What was being approved: cutaway | draft | final | schedule
    approval_type = Column(String, nullable=False)

    # The decision: approved | rejected | modified
    decision = Column(String, nullable=False)

    # What was proposed (JSON snapshot of the item)
    proposed = Column(JSON, nullable=True)

    # What was modified (if decision == "modified")
    modifications = Column(JSON, nullable=True)

    # Reason for rejection (if decision == "rejected")
    reason = Column(Text, nullable=True)

    created_at = Column(DateTime, default=_now)

    # Relationships
    piece = relationship("ContentPiece", back_populates="approvals")


# ═══════════════════════════════════════════════════
# STYLE PROFILE (Spec Section 11)
# ═══════════════════════════════════════════════════

class StyleProfile(Base):
    """
    Production style parameters — both defaults and learned preferences.
    Maps to Spec Section 11.
    """
    __tablename__ = "content_style_profiles"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False, default="default")

    # Video production parameters as JSON
    # {cut_frequency_shorts, cut_frequency_longform,
    #  max_same_shot_duration, caption_font, caption_colour,
    #  colour_grading_lut, background_music_style, etc.}
    video_params = Column(JSON, default=dict)

    # Voice profile for draft writing as JSON
    # {vocabulary_preferences, sentence_structure, tone,
    #  analogy_sources, signature_phrases}
    voice_profile = Column(JSON, default=dict)

    # Reference video learnings as JSON array
    # [{source_url, extracted_params, weight, extracted_at}]
    reference_videos = Column(JSON, default=list)

    # Active flag (only one profile active at a time)
    active = Column(Boolean, default=True)

    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)
