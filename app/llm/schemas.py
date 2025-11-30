# FILE: app/llm/schemas.py
"""
LLM routing schemas: job types, task definitions, and result models.
"""
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel


class JobType(str, Enum):
    """
    All recognized job types for LLM routing.
    Each job type maps to one or more providers.
    """
    # Low-stakes text jobs → GPT only
    CASUAL_CHAT = "casual_chat"
    NOTE_CLEANUP = "note_cleanup"
    COPYWRITING = "copywriting"
    PROMPT_SHAPING = "prompt_shaping"
    SUMMARY = "summary"
    EXPLANATION = "explanation"
    
    # Medium development work → GPT or Claude (configurable)
    SIMPLE_CODE_CHANGE = "simple_code_change"
    SMALL_BUGFIX = "small_bugfix"
    
    # Heavy development / architecture → Claude (primary)
    COMPLEX_CODE_CHANGE = "complex_code_change"
    CODEGEN_FULL_FILE = "codegen_full_file"
    ARCHITECTURE_DESIGN = "architecture_design"
    CODE_REVIEW = "code_review"
    SPEC_REVIEW = "spec_review"
    REFACTOR = "refactor"
    IMPLEMENTATION_PLAN = "implementation_plan"
    
    # High-stakes / critical work → Claude + Gemini review
    HIGH_STAKES_INFRA = "high_stakes_infra"
    SECURITY_SENSITIVE_CHANGE = "security_sensitive_change"
    PRIVACY_SENSITIVE_CHANGE = "privacy_sensitive_change"
    PUBLIC_APP_PACKAGING = "public_app_packaging"
    
    # Vision/analysis tasks → Gemini
    IMAGE_ANALYSIS = "image_analysis"
    SCREENSHOT_ANALYSIS = "screenshot_analysis"
    VIDEO_ANALYSIS = "video_analysis"
    
    # Fallback
    UNKNOWN = "unknown"


class Provider(str, Enum):
    """LLM provider identifiers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class LLMTask(BaseModel):
    """
    A structured task to be routed to the appropriate LLM.
    """
    job_type: JobType
    messages: list[dict]  # [{"role": "user", "content": "..."}]
    
    # Optional context
    system_prompt: Optional[str] = None
    project_context: Optional[str] = None  # Notes + tasks context block
    
    # Optional metadata
    metadata: Optional[dict[str, Any]] = None
    
    # Force a specific provider (overrides routing rules)
    force_provider: Optional[Provider] = None


class LLMResult(BaseModel):
    """
    Result from an LLM call, potentially including critic review.
    """
    # Primary response
    provider: Provider
    content: str
    
    # For two-step flows (high-stakes): critic review
    critic_provider: Optional[Provider] = None
    critic_review: Optional[str] = None
    
    # Metadata
    job_type: JobType
    was_reviewed: bool = False
    
    # Token usage (if available)
    usage: Optional[dict[str, int]] = None


# ============== ROUTING CONFIGURATION ==============

class RoutingConfig:
    """
    Static routing rules: job_type → provider(s).
    Modify these to change model selection behavior.
    """
    
    # Jobs that go to GPT only
    GPT_ONLY_JOBS: set[JobType] = {
        JobType.CASUAL_CHAT,
        JobType.NOTE_CLEANUP,
        JobType.COPYWRITING,
        JobType.PROMPT_SHAPING,
        JobType.SUMMARY,
        JobType.EXPLANATION,
    }
    
    # Jobs that can go to GPT or Claude (default: Claude for quality)
    MEDIUM_DEV_JOBS: set[JobType] = {
        JobType.SIMPLE_CODE_CHANGE,
        JobType.SMALL_BUGFIX,
    }
    
    # Jobs that go to Claude (heavy dev work)
    CLAUDE_PRIMARY_JOBS: set[JobType] = {
        JobType.COMPLEX_CODE_CHANGE,
        JobType.CODEGEN_FULL_FILE,
        JobType.ARCHITECTURE_DESIGN,
        JobType.CODE_REVIEW,
        JobType.SPEC_REVIEW,
        JobType.REFACTOR,
        JobType.IMPLEMENTATION_PLAN,
    }
    
    # Jobs that require Claude + Gemini review (two-step)
    HIGH_STAKES_JOBS: set[JobType] = {
        JobType.HIGH_STAKES_INFRA,
        JobType.SECURITY_SENSITIVE_CHANGE,
        JobType.PRIVACY_SENSITIVE_CHANGE,
        JobType.PUBLIC_APP_PACKAGING,
    }
    
    # Jobs that go to Gemini (vision/analysis)
    GEMINI_JOBS: set[JobType] = {
        JobType.IMAGE_ANALYSIS,
        JobType.SCREENSHOT_ANALYSIS,
        JobType.VIDEO_ANALYSIS,
    }
    
    # For medium dev jobs: use "smart" provider (configurable)
    # Set to ANTHROPIC for quality, OPENAI for speed
    SMART_PROVIDER: Provider = Provider.ANTHROPIC