# FILE: app/llm/schemas.py
"""
LLM routing schemas: job types, task definitions, and result models.

Version: 0.13.0 - Phase 4 Routing Fix

8-ROUTE CLASSIFICATION SYSTEM:
- CHAT_LIGHT: OpenAI (gpt-4.1-mini) - casual chat
- TEXT_HEAVY: OpenAI (gpt-4.1) - heavy text work, text-only PDFs
- CODE_MEDIUM: Anthropic Sonnet - 1-3 files, scoped patches
- ORCHESTRATOR: Anthropic Opus - multi-file, architecture
- IMAGE_SIMPLE: Gemini Flash - simple screenshots
- IMAGE_COMPLEX: Gemini 2.5 Pro - PDFs with images, complex vision
- VIDEO_HEAVY: Gemini 3.0 Pro - video >10MB
- OPUS_CRITIC: Gemini 3.0 Pro - explicit Opus review only

HARD RULES:
1. Never send images/video to Claude (Sonnet or Opus)
2. Never send PDFs to Claude
3. PDFs with no images → GPT text.heavy
4. PDFs with images → Gemini image.complex
5. If ambiguous between CODE_MEDIUM and ORCHESTRATOR → choose ORCHESTRATOR
6. opus.critic is explicit/opt-in only (no fuzzy matching)
"""
from enum import Enum
from typing import Optional, Any, Set, Dict, List
from pydantic import BaseModel
import os


# =============================================================================
# 8-ROUTE JOB CLASSIFICATION
# =============================================================================

class JobType(str, Enum):
    """
    Primary job type classification (8 routes).
    These map directly to provider/model combinations.
    """
    # =========== PRIMARY 8 ROUTES ===========
    
    # OpenAI routes
    CHAT_LIGHT = "chat.light"
    TEXT_HEAVY = "text.heavy"
    
    # Anthropic routes
    CODE_MEDIUM = "code.medium"
    ORCHESTRATOR = "orchestrator"
    
    # Google routes
    IMAGE_SIMPLE = "image.simple"
    IMAGE_COMPLEX = "image.complex"
    VIDEO_HEAVY = "video.heavy"
    OPUS_CRITIC = "opus.critic"
    
    # =========== DOCUMENT-SPECIFIC ROUTES ===========
    DOCUMENT_PDF_TEXT = "document.pdf_text"
    DOCUMENT_PDF_VISION = "document.pdf_vision"
    
    # =========== LEGACY ALIASES (backward compat) ===========
    TEXT_ADMIN = "text_admin"
    CASUAL_CHAT = "casual_chat"
    QUICK_QUESTION = "quick_question"
    NOTE_CLEANUP = "note_cleanup"
    COPYWRITING = "copywriting"
    PROMPT_SHAPING = "prompt_shaping"
    VOICE_INPUT = "voice_input"
    SUMMARY = "summary"
    EXPLANATION = "explanation"
    THOUGHT_ORGANIZATION = "thought_organization"
    SUMMARIZATION = "summarization"
    REWRITING = "rewriting"
    DOCUMENTATION = "documentation"
    RESEARCH = "research"
    LINGUISTICS = "linguistics"
    SMALL_CODE = "small_code"
    SIMPLE_CODE_CHANGE = "simple_code_change"
    SMALL_BUGFIX = "small_bugfix"
    BUG_FIX = "bug_fix"
    BIG_ARCHITECTURE = "big_architecture"
    COMPLEX_CODE_CHANGE = "complex_code_change"
    CODEGEN_FULL_FILE = "codegen_full_file"
    ARCHITECTURE_DESIGN = "architecture_design"
    CODE_REVIEW = "code_review"
    SPEC_REVIEW = "spec_review"
    REFACTOR = "refactor"
    IMPLEMENTATION_PLAN = "implementation_plan"
    HIGH_STAKES_INFRA = "high_stakes_infra"
    SECURITY_SENSITIVE_CHANGE = "security_sensitive_change"
    PRIVACY_SENSITIVE_CHANGE = "privacy_sensitive_change"
    PUBLIC_APP_PACKAGING = "public_app_packaging"
    ARCHITECTURE = "architecture"
    DEEP_PLANNING = "deep_planning"
    SECURITY_REVIEW = "security_review"
    COMPLEX_CODE = "complex_code"
    REFACTORING = "refactoring"
    MIGRATION = "migration"
    BUG_ANALYSIS = "bug_analysis"
    SIMPLE_VISION = "simple_vision"
    IMAGE_ANALYSIS = "image_analysis"
    SCREENSHOT_ANALYSIS = "screenshot_analysis"
    OCR = "ocr"
    HEAVY_MULTIMODAL_CRITIQUE = "heavy_multimodal_critique"
    VIDEO_ANALYSIS = "video_analysis"
    VISION = "vision"
    UI_ANALYSIS = "ui_analysis"
    DOCUMENT_ANALYSIS = "document_analysis"
    WEB_SEARCH = "web_search"
    CRITIQUE = "critique"
    ANALYSIS = "analysis"
    CV_PARSING = "cv_parsing"
    UNKNOWN = "unknown"
    EMBEDDINGS = "embeddings"


class Provider(str, Enum):
    """LLM provider identifiers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


# =============================================================================
# ROUTING DECISION MODEL
# =============================================================================

class RoutingDecision(BaseModel):
    """Result of job classification and routing."""
    job_type: JobType
    provider: Provider
    model: str
    reason: str
    user_override: bool = False
    pdf_image_count: Optional[int] = None
    pdf_text_chars: Optional[int] = None


# =============================================================================
# ROUTING OPTIONS (per-task)
# =============================================================================

class RoutingOptions(BaseModel):
    """Per-task routing options (budget, limits, etc)."""
    max_calls: int = 3
    max_tokens: int = 8000
    max_cost_usd: float = 1.0
    timeout_seconds: int = 60
    force_provider: Optional[Provider] = None
    force_model: Optional[str] = None
    allow_fallback: bool = True
    
    class Config:
        extra = "ignore"


# =============================================================================
# ATTACHMENT INFO
# =============================================================================

class AttachmentInfo(BaseModel):
    """Information about an attachment for classification."""
    filename: str
    mime_type: Optional[str] = None
    size_bytes: int = 0
    pdf_image_count: Optional[int] = None
    pdf_text_chars: Optional[int] = None
    pdf_page_count: Optional[int] = None
    
    @property
    def extension(self) -> str:
        return os.path.splitext(self.filename)[1].lower()
    
    @property
    def is_image(self) -> bool:
        return self.extension in {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".svg"}
    
    @property
    def is_video(self) -> bool:
        return self.extension in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv", ".flv"}
    
    @property
    def is_document(self) -> bool:
        return self.extension in {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".txt", ".md"}
    
    @property
    def is_pdf(self) -> bool:
        return self.extension == ".pdf"
    
    @property
    def is_code(self) -> bool:
        return self.extension in {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala", ".sql", ".sh", ".bash", ".ps1", ".html", ".css", ".json", ".yaml", ".yml", ".xml", ".toml"}


# =============================================================================
# ROUTING CONFIGURATION (Static Rules)
# =============================================================================

class RoutingConfig:
    """Static routing configuration for the 8-route system."""
    
    # === JOB TYPE → PROVIDER/MODEL MAPPING ===
    JOB_TYPE_ROUTING: Dict[JobType, tuple] = {
        # OpenAI routes
        JobType.CHAT_LIGHT: (Provider.OPENAI, "OPENAI_MODEL_LIGHT_CHAT", "gpt-4.1-mini"),
        JobType.TEXT_HEAVY: (Provider.OPENAI, "OPENAI_MODEL_HEAVY_TEXT", "gpt-4.1"),
        
        # Anthropic routes
        JobType.CODE_MEDIUM: (Provider.ANTHROPIC, "ANTHROPIC_SONNET_MODEL", "claude-sonnet-4-5-20250929"),
        JobType.ORCHESTRATOR: (Provider.ANTHROPIC, "ANTHROPIC_OPUS_MODEL", "claude-opus-4-5-20250514"),
        
        # Google routes
        JobType.IMAGE_SIMPLE: (Provider.GOOGLE, "GEMINI_VISION_MODEL_FAST", "gemini-2.0-flash"),
        JobType.IMAGE_COMPLEX: (Provider.GOOGLE, "GEMINI_VISION_MODEL_COMPLEX", "gemini-2.5-pro"),
        JobType.VIDEO_HEAVY: (Provider.GOOGLE, "GEMINI_VIDEO_HEAVY_MODEL", "gemini-3.0-pro-preview"),
        JobType.OPUS_CRITIC: (Provider.GOOGLE, "GEMINI_OPUS_CRITIC_MODEL", "gemini-3.0-pro-preview"),
        
        # Document-specific
        JobType.DOCUMENT_PDF_TEXT: (Provider.OPENAI, "OPENAI_MODEL_HEAVY_TEXT", "gpt-4.1"),
        JobType.DOCUMENT_PDF_VISION: (Provider.GOOGLE, "GEMINI_VISION_MODEL_COMPLEX", "gemini-2.5-pro"),
    }
    
    # === LEGACY JOB TYPE → PRIMARY TYPE MAPPING ===
    LEGACY_TO_PRIMARY: Dict[JobType, JobType] = {
        # CHAT_LIGHT aliases
        JobType.TEXT_ADMIN: JobType.CHAT_LIGHT,
        JobType.CASUAL_CHAT: JobType.CHAT_LIGHT,
        JobType.QUICK_QUESTION: JobType.CHAT_LIGHT,
        JobType.NOTE_CLEANUP: JobType.CHAT_LIGHT,
        JobType.COPYWRITING: JobType.CHAT_LIGHT,
        JobType.PROMPT_SHAPING: JobType.CHAT_LIGHT,
        JobType.VOICE_INPUT: JobType.CHAT_LIGHT,
        JobType.EMBEDDINGS: JobType.CHAT_LIGHT,
        JobType.WEB_SEARCH: JobType.CHAT_LIGHT,
        
        # TEXT_HEAVY aliases
        JobType.SUMMARY: JobType.TEXT_HEAVY,
        JobType.EXPLANATION: JobType.TEXT_HEAVY,
        JobType.THOUGHT_ORGANIZATION: JobType.TEXT_HEAVY,
        JobType.SUMMARIZATION: JobType.TEXT_HEAVY,
        JobType.REWRITING: JobType.TEXT_HEAVY,
        JobType.DOCUMENTATION: JobType.TEXT_HEAVY,
        JobType.RESEARCH: JobType.TEXT_HEAVY,
        JobType.LINGUISTICS: JobType.TEXT_HEAVY,
        JobType.CRITIQUE: JobType.TEXT_HEAVY,
        JobType.ANALYSIS: JobType.TEXT_HEAVY,
        
        # CODE_MEDIUM aliases
        JobType.SMALL_CODE: JobType.CODE_MEDIUM,
        JobType.SIMPLE_CODE_CHANGE: JobType.CODE_MEDIUM,
        JobType.SMALL_BUGFIX: JobType.CODE_MEDIUM,
        JobType.BUG_FIX: JobType.CODE_MEDIUM,
        
        # ORCHESTRATOR aliases
        JobType.BIG_ARCHITECTURE: JobType.ORCHESTRATOR,
        JobType.COMPLEX_CODE_CHANGE: JobType.ORCHESTRATOR,
        JobType.CODEGEN_FULL_FILE: JobType.ORCHESTRATOR,
        JobType.ARCHITECTURE_DESIGN: JobType.ORCHESTRATOR,
        JobType.CODE_REVIEW: JobType.ORCHESTRATOR,
        JobType.SPEC_REVIEW: JobType.ORCHESTRATOR,
        JobType.REFACTOR: JobType.ORCHESTRATOR,
        JobType.IMPLEMENTATION_PLAN: JobType.ORCHESTRATOR,
        JobType.HIGH_STAKES_INFRA: JobType.ORCHESTRATOR,
        JobType.SECURITY_SENSITIVE_CHANGE: JobType.ORCHESTRATOR,
        JobType.PRIVACY_SENSITIVE_CHANGE: JobType.ORCHESTRATOR,
        JobType.PUBLIC_APP_PACKAGING: JobType.ORCHESTRATOR,
        JobType.ARCHITECTURE: JobType.ORCHESTRATOR,
        JobType.DEEP_PLANNING: JobType.ORCHESTRATOR,
        JobType.SECURITY_REVIEW: JobType.ORCHESTRATOR,
        JobType.COMPLEX_CODE: JobType.ORCHESTRATOR,
        JobType.REFACTORING: JobType.ORCHESTRATOR,
        JobType.MIGRATION: JobType.ORCHESTRATOR,
        JobType.BUG_ANALYSIS: JobType.ORCHESTRATOR,
        
        # IMAGE_SIMPLE aliases
        JobType.SIMPLE_VISION: JobType.IMAGE_SIMPLE,
        JobType.IMAGE_ANALYSIS: JobType.IMAGE_SIMPLE,
        JobType.SCREENSHOT_ANALYSIS: JobType.IMAGE_SIMPLE,
        JobType.OCR: JobType.IMAGE_SIMPLE,
        JobType.VISION: JobType.IMAGE_SIMPLE,
        JobType.UI_ANALYSIS: JobType.IMAGE_SIMPLE,
        
        # IMAGE_COMPLEX aliases
        JobType.HEAVY_MULTIMODAL_CRITIQUE: JobType.IMAGE_COMPLEX,
        JobType.DOCUMENT_ANALYSIS: JobType.IMAGE_COMPLEX,
        JobType.CV_PARSING: JobType.IMAGE_COMPLEX,
        
        # VIDEO_HEAVY aliases
        JobType.VIDEO_ANALYSIS: JobType.VIDEO_HEAVY,
        
        # Primary types map to themselves
        JobType.CHAT_LIGHT: JobType.CHAT_LIGHT,
        JobType.TEXT_HEAVY: JobType.TEXT_HEAVY,
        JobType.CODE_MEDIUM: JobType.CODE_MEDIUM,
        JobType.ORCHESTRATOR: JobType.ORCHESTRATOR,
        JobType.IMAGE_SIMPLE: JobType.IMAGE_SIMPLE,
        JobType.IMAGE_COMPLEX: JobType.IMAGE_COMPLEX,
        JobType.VIDEO_HEAVY: JobType.VIDEO_HEAVY,
        JobType.OPUS_CRITIC: JobType.OPUS_CRITIC,
        JobType.DOCUMENT_PDF_TEXT: JobType.DOCUMENT_PDF_TEXT,
        JobType.DOCUMENT_PDF_VISION: JobType.DOCUMENT_PDF_VISION,
        JobType.UNKNOWN: JobType.CHAT_LIGHT,
    }
    
    # === KEYWORD LISTS FOR CLASSIFICATION ===
    
    CHAT_LIGHT_KEYWORDS: Set[str] = {
        "hello", "hi", "hey", "thanks", "thank you",
        "what's up", "how are you", "good morning",
        "quick question", "simple question",
        "can you help", "just wondering",
    }
    
    TEXT_HEAVY_KEYWORDS: Set[str] = {
        "email", "letter", "report", "blog", "readme", "documentation",
        "plan", "brainstorm", "summary", "summarize", "explain",
        "write a", "draft", "compose", "rewrite", "proofread",
        "meeting notes", "agenda", "memo", "announcement",
        "spreadsheet", "presentation", "slide", "outline",
        "translate", "transcribe", "format", "grammar",
        "essay", "article", "content", "copy",
    }
    
    CODE_MEDIUM_KEYWORDS: Set[str] = {
        "fix this bug", "small fix", "quick fix", "simple fix",
        "add a function", "helper function", "utility function",
        "tweak", "minor change", "small change",
        "one file", "single file", "this file",
        "lint", "format code", "clean up",
        "add logging", "add error handling",
        "rename", "extract method",
    }
    
    ORCHESTRATOR_KEYWORDS: Set[str] = {
        "architect", "architecture", "multi-file", "multiple files",
        "refactor", "routing", "memory", "database", "schema",
        "security", "encryption", "authentication", "authorization",
        "system design", "design pattern", "infrastructure",
        "migration", "upgrade", "overhaul", "rewrite",
        "microservice", "api design", "data model",
        "full implementation", "complete system", "end to end",
        "production", "deployment", "ci/cd", "pipeline",
        "phase", "v0.", "version",
    }
    
    IMAGE_SIMPLE_KEYWORDS: Set[str] = {
        "screenshot", "what is this", "ocr", "read this",
        "small image", "this image", "the image",
        "what do you see", "describe this",
        "logo", "icon", "diagram",
    }
    
    IMAGE_COMPLEX_KEYWORDS: Set[str] = {
        "complex pdf", "multiple pages", "long document",
        "analyze in detail", "thorough review",
        "cv", "resume", "portfolio",
        "multiple images", "compare images",
    }
    
    VIDEO_HEAVY_KEYWORDS: Set[str] = {
        "video", "youtube", "reel", "lecture", "recording",
        "watch this", "analyze video", "video analysis",
        "clip", "footage", "movie",
    }
    
    # === BACKWARD COMPAT SETS ===
    
    GPT_ONLY_JOBS: Set[JobType] = {
        JobType.CASUAL_CHAT, JobType.NOTE_CLEANUP, JobType.COPYWRITING,
        JobType.PROMPT_SHAPING, JobType.SUMMARY, JobType.EXPLANATION,
        JobType.QUICK_QUESTION, JobType.THOUGHT_ORGANIZATION,
        JobType.SUMMARIZATION, JobType.REWRITING, JobType.DOCUMENTATION,
        JobType.RESEARCH, JobType.LINGUISTICS, JobType.VOICE_INPUT,
        JobType.TEXT_ADMIN, JobType.CHAT_LIGHT, JobType.TEXT_HEAVY,
    }
    
    MEDIUM_DEV_JOBS: Set[JobType] = {
        JobType.SIMPLE_CODE_CHANGE, JobType.SMALL_BUGFIX, JobType.BUG_FIX,
        JobType.SMALL_CODE, JobType.CODE_MEDIUM,
    }
    
    CLAUDE_PRIMARY_JOBS: Set[JobType] = {
        JobType.COMPLEX_CODE_CHANGE, JobType.CODEGEN_FULL_FILE,
        JobType.ARCHITECTURE_DESIGN, JobType.CODE_REVIEW, JobType.SPEC_REVIEW,
        JobType.REFACTOR, JobType.IMPLEMENTATION_PLAN, JobType.COMPLEX_CODE,
        JobType.BUG_ANALYSIS, JobType.BIG_ARCHITECTURE, JobType.ORCHESTRATOR,
    }
    
    HIGH_STAKES_JOBS: Set[JobType] = {
        JobType.HIGH_STAKES_INFRA, JobType.SECURITY_SENSITIVE_CHANGE,
        JobType.PRIVACY_SENSITIVE_CHANGE, JobType.PUBLIC_APP_PACKAGING,
        JobType.ARCHITECTURE, JobType.DEEP_PLANNING, JobType.SECURITY_REVIEW,
        JobType.MIGRATION,
        # v0.13.10: Added missing high-stakes types for critique pipeline
        JobType.ARCHITECTURE_DESIGN,   # Critical: was routing to Sonnet instead of Opus
        JobType.BIG_ARCHITECTURE,      # Critical: multi-file architecture
        JobType.ORCHESTRATOR,          # Needed for infra tasks
        JobType.COMPLEX_CODE_CHANGE,   # High-risk code changes
        JobType.IMPLEMENTATION_PLAN,   # Architecture planning
        JobType.SPEC_REVIEW,           # Design review
    }
    
    GEMINI_JOBS: Set[JobType] = {
        JobType.IMAGE_ANALYSIS, JobType.SCREENSHOT_ANALYSIS, JobType.VIDEO_ANALYSIS,
        JobType.VISION, JobType.UI_ANALYSIS, JobType.DOCUMENT_ANALYSIS,
        JobType.OCR, JobType.WEB_SEARCH, JobType.CV_PARSING,
        JobType.SIMPLE_VISION, JobType.HEAVY_MULTIMODAL_CRITIQUE,
        JobType.IMAGE_SIMPLE, JobType.IMAGE_COMPLEX, JobType.VIDEO_HEAVY,
        JobType.OPUS_CRITIC,
    }
    
    SMART_PROVIDER: Provider = Provider.ANTHROPIC
    
    @classmethod
    def normalize_job_type(cls, job_type: JobType) -> JobType:
        """Map any job type to one of the 8 primary types."""
        return cls.LEGACY_TO_PRIMARY.get(job_type, JobType.CHAT_LIGHT)
    
    @classmethod
    def get_routing(cls, job_type: JobType) -> tuple:
        """Get (provider, model) for a job type."""
        primary = cls.normalize_job_type(job_type)
        config = cls.JOB_TYPE_ROUTING.get(primary)
        if not config:
            config = cls.JOB_TYPE_ROUTING[JobType.CHAT_LIGHT]
        provider, env_var, default_model = config
        model = os.getenv(env_var, default_model)
        return (provider, model)


# =============================================================================
# TASK AND RESULT MODELS
# =============================================================================

class LLMTask(BaseModel):
    """A structured task to be routed to the appropriate LLM."""
    job_type: JobType
    messages: list[dict]
    system_prompt: Optional[str] = None
    project_context: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    force_provider: Optional[Provider] = None
    attachments: Optional[list[dict]] = None
    provider: Optional[Provider] = None
    model: Optional[str] = None
    routing: Optional[RoutingOptions] = None
    
    class Config:
        extra = "ignore"


class LLMResult(BaseModel):
    """Result from an LLM call."""
    content: str
    provider: str
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    error_message: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    raw_response: Optional[Any] = None
    critic_provider: Optional[str] = None
    critic_review: Optional[str] = None
    job_type: Optional[JobType] = None
    was_reviewed: bool = False
    usage: Optional[dict[str, int]] = None
    resolved_job_type: Optional[str] = None
    routing_decision: Optional[dict] = None
    critic_usage: Optional[dict] = None
    error: Optional[str] = None
    critic_error: Optional[str] = None
    reasoning: Optional[str] = None
    
    class Config:
        extra = "ignore"
    
    def has_error(self) -> bool:
        return self.error_message is not None or self.error is not None
    
    def is_success(self) -> bool:
        return not self.has_error() and self.finish_reason in (None, "stop")


# =============================================================================
# API REQUEST/RESPONSE MODELS
# =============================================================================

class ChatRequest(BaseModel):
    project_id: int
    message: str
    job_type: str = "casual_chat"
    include_context: bool = True


class ChatResponse(BaseModel):
    response: str
    provider: str
    job_type: str
    was_reviewed: bool = False
    critic_review: Optional[str] = None
    context_used: bool = False
    error: Optional[str] = None


class StreamRequest(BaseModel):
    message: str
    project_id: int
    job_type: str = "casual_chat"
    provider: Optional[str] = None


class ProviderInfo(BaseModel):
    name: str
    display_name: str
    available: bool
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_web_search: bool = False


class ProvidersResponse(BaseModel):
    providers: list[ProviderInfo]
    default_provider: str