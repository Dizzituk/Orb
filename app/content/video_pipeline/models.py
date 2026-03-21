# FILE: app/content/video_pipeline/models.py
"""
Pydantic models for the Script-to-Video Pipeline.

All data structures that flow between pipeline stages.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


# ═══════════════════════════════════════════════════
# SHARED PIPELINE MODEL CONFIG
# ═══════════════════════════════════════════════════

PIPELINE_GEMINI_MODEL = "gemini-3.1-pro-preview"
"""Single model used across the entire video pipeline.
All Gemini calls (script analysis, style extraction, future QA)
use this model to maintain consistent context and quality."""


class SegmentType(str, Enum):
    INTRO = "intro"
    BODY = "body"
    CUTAWAY = "cutaway"
    AVATAR = "avatar"
    TRANSITION = "transition"
    OUTRO = "outro"


class AssetTier(str, Enum):
    LOCAL = "local"
    FREE_STOCK = "free_stock"
    AI_GENERATED = "ai_generated"
    AVATAR = "avatar"


class AssetSource(str, Enum):
    LOCAL_INDEX = "local_index"
    PEXELS = "pexels"
    PIXABAY = "pixabay"
    FAL_AI = "fal_ai"
    HEYGEN = "heygen"
    TTS = "tts"


# ═══════════════════════════════════════════════════
# SCRIPT ANALYSIS OUTPUT
# ═══════════════════════════════════════════════════

class AvatarFraming(str, Enum):
    FULL_FRAME = "full_frame"  # Avatar takes full screen, background behind
    PIP = "pip"  # Small avatar in corner, main video underneath
    NONE = "none"  # No avatar for this segment


class ClipSpec(BaseModel):
    """A specific clip chosen by the director for a segment."""
    shot_description: str = Field(
        default="", description="What this clip shows"
    )
    search_query: str = Field(
        default="", description="2-4 word Pexels search query"
    )
    duration_weight: float = Field(
        default=0.5, description="Fraction of segment time (0.5 = half)"
    )


class SceneSegment(BaseModel):
    """A single segment in the scene plan."""
    segment_id: str = Field(..., description="e.g. seg_001")
    segment_type: SegmentType
    script_text: str = Field(..., description="Narration text for this segment")
    visual_description: str = Field(..., description="What should be shown visually")
    clips: List[ClipSpec] = Field(
        default_factory=list,
        description="1-2 director-chosen clips with search queries",
    )
    search_keywords: List[str] = Field(default_factory=list)
    mood_tags: List[str] = Field(default_factory=list)
    estimated_duration_s: float = Field(default=5.0)
    requires_avatar: bool = Field(default=False)
    avatar_framing: AvatarFraming = Field(default=AvatarFraming.NONE)
    priority_tier: AssetTier = Field(default=AssetTier.LOCAL)


class ScenePlan(BaseModel):
    """Complete scene plan from the script analyzer."""
    title: str
    total_segments: int
    estimated_total_duration_s: float
    segments: List[SceneSegment]
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════
# STYLE PROFILE
# ═══════════════════════════════════════════════════

class StyleProfile(BaseModel):
    """Quantified style parameters extracted from reference videos."""
    profile_id: str = ""
    # Pacing
    avg_cut_duration_s: float = 3.5
    intro_length_s: float = 5.0
    outro_length_s: float = 8.0
    segment_rhythm: str = "moderate"  # slow | moderate | fast
    # Transitions
    primary_transition: str = "cut"
    secondary_transition: str = "dissolve"
    transition_frequency: str = "every_scene"
    # Colour
    colour_temperature: str = "neutral"  # warm | neutral | cool
    saturation_level: str = "medium"
    contrast_level: str = "medium"
    lut_reference: Optional[str] = None
    # Captions
    caption_style: str = "bold_overlay"
    font_family: str = "Montserrat"
    font_size_px: int = 48
    caption_position: str = "bottom_center"
    caption_animation: str = "pop_in"
    # Audio
    music_volume_ratio: float = 0.15
    voice_volume: float = 1.0
    sfx_frequency: str = "low"
    music_genre_preference: str = "ambient_electronic"
    # Composition
    aspect_ratio_preference: str = "16:9"
    zoom_usage: str = "subtle_slow"
    b_roll_density: str = "high"
    avatar_frequency: str = "intro_outro_only"
    # Mood
    overall_tone: str = "educational"
    energy_level: str = "medium"
    # Visual production mood (NOT topic) — e.g. clean, gritty, cinematic, raw
    visual_mood_keywords: List[str] = Field(
        default_factory=lambda: ["clean", "modern", "minimal"]
    )


# ═══════════════════════════════════════════════════
# RESOLVED ASSETS
# ═══════════════════════════════════════════════════

class ResolvedAsset(BaseModel):
    """A matched asset for a single scene segment."""
    segment_id: str
    source: AssetSource
    tier: AssetTier
    file_path: str = ""
    url: str = ""
    duration_s: float = 0.0
    cost_usd: float = 0.0
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResolvedPlan(BaseModel):
    """All resolved assets for the full scene plan."""
    assets: List[ResolvedAsset]
    total_cost_usd: float = 0.0
    unresolved_segments: List[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════
# PIPELINE JOB
# ═══════════════════════════════════════════════════

class PipelineJobRequest(BaseModel):
    """Request to start the video pipeline."""
    script_text: str
    title: str
    target_platform: str = "youtube_longform"
    target_duration_s: Optional[int] = None
    style_profile_id: Optional[str] = None
    max_budget_usd: float = 10.0


class PipelineStageUpdate(BaseModel):
    """SSE event payload for pipeline progress."""
    job_id: str
    stage: str
    status: str  # started | progress | complete | error
    message: str = ""
    progress_pct: float = 0.0
    data: Dict[str, Any] = Field(default_factory=dict)
