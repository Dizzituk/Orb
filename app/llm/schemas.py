# FILE: app/llm/schemas.py
"""
LLM routing schemas: job types, task definitions, and result models.

Backward compatible with original enum-based routing.
Also supports new string-based policy routing.

PHASE 4 FIXES:
- Added provider, model, routing fields to LLMTask for backward compat
- Added RoutingOptions model (was missing - RoutingConfig only had static attrs)
- Fixed LLMResult to include all fields used by router.py
"""
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel


class JobType(str, Enum):
    """
    All recognized job types for LLM routing.
    Each job type maps to one or more providers.
    
    ORIGINAL values preserved for backward compatibility.
    NEW values added for policy-based routing.
    """
    # =========== ORIGINAL VALUES (do not remove) ===========
    
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
    
    # =========== NEW VALUES (policy-based routing) ===========
    
    # GPT-primary (additional)
    QUICK_QUESTION = "quick_question"
    THOUGHT_ORGANIZATION = "thought_organization"
    SUMMARIZATION = "summarization"
    REWRITING = "rewriting"
    DOCUMENTATION = "documentation"
    RESEARCH = "research"
    LINGUISTICS = "linguistics"
    VOICE_INPUT = "voice_input"
    EMBEDDINGS = "embeddings"
    
    # Claude-primary (additional)
    ARCHITECTURE = "architecture"
    DEEP_PLANNING = "deep_planning"
    SECURITY_REVIEW = "security_review"
    COMPLEX_CODE = "complex_code"
    REFACTORING = "refactoring"
    MIGRATION = "migration"
    BUG_ANALYSIS = "bug_analysis"
    BUG_FIX = "bug_fix"
    
    # Gemini-primary (additional)
    VISION = "vision"
    UI_ANALYSIS = "ui_analysis"
    DOCUMENT_ANALYSIS = "document_analysis"
    OCR = "ocr"
    WEB_SEARCH = "web_search"
    CRITIQUE = "critique"
    ANALYSIS = "analysis"
    CV_PARSING = "cv_parsing"


class Provider(str, Enum):
    """LLM provider identifiers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class RoutingOptions(BaseModel):
    """
    Per-task routing options (budget, limits, etc).
    
    This is an INSTANCE model for LLMTask.routing field.
    Not to be confused with RoutingConfig which holds static routing rules.
    """
    max_calls: int = 3
    max_tokens: int = 8000
    max_cost_usd: float = 1.0
    timeout_seconds: int = 60
    
    # Optional overrides
    force_provider: Optional[Provider] = None
    force_model: Optional[str] = None
    allow_fallback: bool = True
    
    class Config:
        # Allow extra fields for forward compatibility
        extra = "ignore"


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
    
    # NEW: Attachments for vision tasks
    attachments: Optional[list[dict]] = None
    
    # =========== RESTORED FIELDS (for backward compat with router.py) ===========
    
    # Provider hint (used by router to select provider)
    provider: Optional[Provider] = None
    
    # Model hint (used by router to select specific model)
    model: Optional[str] = None
    
    # Routing options (budget, limits, etc)
    routing: Optional[RoutingOptions] = None
    
    class Config:
        # Allow extra fields for forward compatibility
        extra = "ignore"


class LLMResult(BaseModel):
    """
    Result from an LLM call, potentially including critic review.
    
    This model supports BOTH:
    - Legacy fields (provider as enum, job_type, was_reviewed, critic_*)
    - New fields (provider as str, model, finish_reason, token counts, etc)
    
    The router.py constructs results with the new fields.
    Other code may still use legacy fields.
    """
    # =========== PRIMARY RESPONSE ===========
    
    # Content is always present (may be empty on error)
    content: str
    
    # Provider can be either enum or string for compatibility
    # Router passes string, legacy code may pass enum
    provider: str  # Changed from Provider enum to str for flexibility
    
    # =========== NEW FIELDS (used by router.py) ===========
    
    # Model that was used
    model: Optional[str] = None
    
    # How the response finished
    finish_reason: Optional[str] = None  # "stop", "error", "validation_error", etc
    
    # Error details (if any)
    error_message: Optional[str] = None
    
    # Token usage (flat fields for router.py compatibility)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    
    # Raw response from provider (for debugging)
    raw_response: Optional[Any] = None
    
    # =========== LEGACY FIELDS (for backward compat) ===========
    
    # For two-step flows (high-stakes): critic review
    critic_provider: Optional[str] = None
    critic_review: Optional[str] = None
    
    # Metadata
    job_type: Optional[JobType] = None  # Made optional for router.py compat
    was_reviewed: bool = False
    
    # Token usage (dict form for legacy code)
    usage: Optional[dict[str, int]] = None
    
    # Additional metadata from policy routing
    resolved_job_type: Optional[str] = None
    routing_decision: Optional[dict] = None
    critic_usage: Optional[dict] = None
    error: Optional[str] = None  # Alias for error_message in legacy code
    critic_error: Optional[str] = None
    
    class Config:
        # Allow extra fields for forward compatibility
        extra = "ignore"
    
    def has_error(self) -> bool:
        """Check if there was an error."""
        return self.error_message is not None or self.error is not None
    
    def is_success(self) -> bool:
        """Check if the call succeeded."""
        return not self.has_error() and self.finish_reason in (None, "stop")
    
    def get_full_response(self, include_review: bool = True) -> str:
        """Get full response, optionally including critic review."""
        if not include_review or not self.was_reviewed or not self.critic_review:
            return self.content
        return f"{self.content}\n\n---\n\n## Critic Review\n\n{self.critic_review}"


# ============== ROUTING CONFIGURATION (Static Rules) ==============

class RoutingConfig:
    """
    Static routing rules: job_type → provider(s).
    Modify these to change model selection behavior.
    
    NOTE: When policy.py is available, policy-based routing takes precedence.
    These sets are kept for backward compatibility.
    
    NOTE: This is a static configuration class, NOT a Pydantic model.
    For per-task routing options, use RoutingOptions instead.
    """
    
    # Jobs that go to GPT only
    GPT_ONLY_JOBS: set[JobType] = {
        JobType.CASUAL_CHAT,
        JobType.NOTE_CLEANUP,
        JobType.COPYWRITING,
        JobType.PROMPT_SHAPING,
        JobType.SUMMARY,
        JobType.EXPLANATION,
        # New policy-based
        JobType.QUICK_QUESTION,
        JobType.THOUGHT_ORGANIZATION,
        JobType.SUMMARIZATION,
        JobType.REWRITING,
        JobType.DOCUMENTATION,
        JobType.RESEARCH,
        JobType.LINGUISTICS,
        JobType.VOICE_INPUT,
    }
    
    # Jobs that can go to GPT or Claude (default: Claude for quality)
    MEDIUM_DEV_JOBS: set[JobType] = {
        JobType.SIMPLE_CODE_CHANGE,
        JobType.SMALL_BUGFIX,
        JobType.BUG_FIX,
        JobType.REFACTORING,
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
        # New policy-based
        JobType.COMPLEX_CODE,
        JobType.BUG_ANALYSIS,
    }
    
    # Jobs that require Claude + Gemini review (two-step)
    HIGH_STAKES_JOBS: set[JobType] = {
        JobType.HIGH_STAKES_INFRA,
        JobType.SECURITY_SENSITIVE_CHANGE,
        JobType.PRIVACY_SENSITIVE_CHANGE,
        JobType.PUBLIC_APP_PACKAGING,
        # New policy-based
        JobType.ARCHITECTURE,
        JobType.DEEP_PLANNING,
        JobType.SECURITY_REVIEW,
        JobType.MIGRATION,
    }
    
    # Jobs that go to Gemini (vision/analysis)
    GEMINI_JOBS: set[JobType] = {
        JobType.IMAGE_ANALYSIS,
        JobType.SCREENSHOT_ANALYSIS,
        JobType.VIDEO_ANALYSIS,
        # New policy-based
        JobType.VISION,
        JobType.UI_ANALYSIS,
        JobType.DOCUMENT_ANALYSIS,
        JobType.OCR,
        JobType.WEB_SEARCH,
        JobType.CRITIQUE,
        JobType.ANALYSIS,
        JobType.CV_PARSING,
    }
    
    # For medium dev jobs: use "smart" provider (configurable)
    # Set to ANTHROPIC for quality, OPENAI for speed
    SMART_PROVIDER: Provider = Provider.ANTHROPIC


# ============== API REQUEST/RESPONSE MODELS ==============

class ChatRequest(BaseModel):
    """Request model for /chat endpoint."""
    message: str
    project_id: int
    job_type: str = "casual_chat"
    include_context: bool = True


class ChatResponse(BaseModel):
    """Response model for /chat endpoint."""
    response: str
    provider: str
    job_type: str
    was_reviewed: bool = False
    critic_review: Optional[str] = None
    context_used: bool = False
    error: Optional[str] = None


class StreamRequest(BaseModel):
    """Request model for /stream/chat endpoint."""
    message: str
    project_id: int
    job_type: str = "casual_chat"
    provider: Optional[str] = None


class ProviderInfo(BaseModel):
    """Info about an available provider."""
    name: str
    display_name: str
    available: bool
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_web_search: bool = False


class ProvidersResponse(BaseModel):
    """Response for /providers endpoint."""
    providers: list[ProviderInfo]
    default_provider: str