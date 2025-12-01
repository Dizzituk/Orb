"""
Orb LLM Routing Policy Module

Provides deterministic routing decisions and data-handling enforcement
for the multi-LLM orchestration system.

Usage:
    from app.llm.policy import load_routing_policy, get_policy_for_job, validate_task_data

    policy = load_routing_policy()
    job_policy = get_policy_for_job("architecture")
    validate_task_data(job_policy, attachments)
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# ENUMS
# =============================================================================

class Provider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class AttachmentMode(str, Enum):
    """How attachments should be handled."""
    NONE = "none"
    TEXT_ONLY = "text_only"
    ALL = "all"


class DataType(str, Enum):
    """Content types for routing decisions."""
    TEXT = "text"
    CODE = "code"
    LOGS = "logs"
    STRUCTURED_JSON = "structured_json"
    IMAGES = "images"
    VIDEOS = "videos"
    AUDIO = "audio"
    PDFS = "pdfs"
    SCREENSHOTS = "screenshots"
    LARGE_CODEBASE = "large_codebase"
    DESIGN_DOCS = "design_docs"
    EMBEDDINGS_REQUEST = "embeddings_request"


class JobType(str, Enum):
    """All supported job types for routing."""
    # GPT-primary jobs
    CASUAL_CHAT = "casual_chat"
    QUICK_QUESTION = "quick_question"
    THOUGHT_ORGANIZATION = "thought_organization"
    SUMMARIZATION = "summarization"
    REWRITING = "rewriting"
    DOCUMENTATION = "documentation"
    RESEARCH = "research"
    LINGUISTICS = "linguistics"
    VOICE_INPUT = "voice_input"
    EMBEDDINGS = "embeddings"
    
    # Claude-primary jobs
    ARCHITECTURE = "architecture"
    DEEP_PLANNING = "deep_planning"
    SECURITY_REVIEW = "security_review"
    COMPLEX_CODE = "complex_code"
    REFACTORING = "refactoring"
    MIGRATION = "migration"
    BUG_ANALYSIS = "bug_analysis"
    CODE_REVIEW = "code_review"
    IMPLEMENTATION_PLAN = "implementation_plan"
    BUG_FIX = "bug_fix"
    
    # Gemini-primary jobs
    VISION = "vision"
    UI_ANALYSIS = "ui_analysis"
    DOCUMENT_ANALYSIS = "document_analysis"
    OCR = "ocr"
    VIDEO_ANALYSIS = "video_analysis"
    WEB_SEARCH = "web_search"
    CRITIQUE = "critique"
    ANALYSIS = "analysis"
    CV_PARSING = "cv_parsing"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    display_name: str
    model: str
    streaming_model: str
    embedding_model: Optional[str] = None
    vision_model: Optional[str] = None
    max_context_tokens: int
    max_output_tokens: int
    supports_streaming: bool = True
    supports_vision: bool = False
    supports_web_search: bool = False
    allowed_data: list[str]
    forbidden_data: list[str]
    max_chars_input: int


class DataTypeConfig(BaseModel):
    """Configuration for a data type."""
    description: str
    mime_prefixes: Optional[list[str]] = None
    mime_types: Optional[list[str]] = None
    extensions: Optional[list[str]] = None
    patterns: Optional[list[str]] = None
    keywords: Optional[list[str]] = None
    min_bytes: Optional[int] = None
    internal: bool = False


class PolicyDefaults(BaseModel):
    """Default values for job policies."""
    primary_provider: Provider = Provider.OPENAI
    review_provider: Optional[Provider] = None
    include_semantic_context: bool = True
    include_web_search: bool = False
    include_attachments: AttachmentMode = AttachmentMode.TEXT_ONLY
    max_chars_to_provider: int = 100000
    timeout_seconds: int = 120
    retry_count: int = 2
    temperature: float = 0.7


class JobPolicy(BaseModel):
    """Routing policy for a specific job type."""
    job_type: str
    description: str = ""
    primary_provider: Provider
    review_provider: Optional[Provider] = None
    include_semantic_context: bool = True
    include_web_search: bool = False
    include_attachments: AttachmentMode = AttachmentMode.TEXT_ONLY
    allowed_data: list[str]
    forbidden_data: list[str]
    max_chars_to_provider: int = 100000
    temperature: float = 0.7
    timeout_seconds: int = 120
    retry_count: int = 2
    requires_review: bool = False
    requires_vision: bool = False
    special_endpoint: Optional[str] = None

    @field_validator("primary_provider", "review_provider", mode="before")
    @classmethod
    def normalize_provider(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            return Provider(v)
        return v

    @field_validator("include_attachments", mode="before")
    @classmethod
    def normalize_attachment_mode(cls, v):
        if isinstance(v, str):
            return AttachmentMode(v)
        return v


class ValidationRules(BaseModel):
    """Rules for validating routing decisions."""
    require_vision_for: list[str] = Field(default_factory=list)
    require_web_search_for: list[str] = Field(default_factory=list)
    high_stakes_jobs: list[str] = Field(default_factory=list)
    vision_capable_providers: list[str] = Field(default_factory=list)
    web_search_capable_providers: list[str] = Field(default_factory=list)
    embedding_capable_providers: list[str] = Field(default_factory=list)


class RoutingPolicy(BaseModel):
    """Complete routing policy configuration."""
    policy_version: str
    schema_version: str
    description: str = ""
    providers: dict[str, ProviderConfig]
    data_types: dict[str, DataTypeConfig]
    defaults: PolicyDefaults
    entries: list[JobPolicy]
    job_type_aliases: dict[str, str] = Field(default_factory=dict)
    validation_rules: ValidationRules = Field(default_factory=ValidationRules)

    # Internal lookup cache
    _job_lookup: dict[str, JobPolicy] = {}

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def build_lookup(self):
        """Build job type lookup cache after validation."""
        object.__setattr__(self, "_job_lookup", {})
        for entry in self.entries:
            self._job_lookup[entry.job_type] = entry
        return self

    def get_job_policy(self, job_type: str) -> JobPolicy:
        """Get policy for a job type, resolving aliases."""
        # Resolve alias if present
        resolved = self.job_type_aliases.get(job_type, job_type)
        
        if resolved in self._job_lookup:
            return self._job_lookup[resolved]
        
        raise PolicyError(f"Unknown job type: '{job_type}' (resolved: '{resolved}')")

    def get_provider_config(self, provider: Provider | str) -> ProviderConfig:
        """Get configuration for a provider."""
        key = provider.value if isinstance(provider, Provider) else provider
        if key not in self.providers:
            raise PolicyError(f"Unknown provider: '{key}'")
        return self.providers[key]

    def list_job_types(self) -> list[str]:
        """List all valid job types (excluding aliases)."""
        return [e.job_type for e in self.entries]

    def list_aliases(self) -> dict[str, str]:
        """List all job type aliases."""
        return dict(self.job_type_aliases)

    def is_high_stakes(self, job_type: str) -> bool:
        """Check if a job type requires review."""
        resolved = self.job_type_aliases.get(job_type, job_type)
        return resolved in self.validation_rules.high_stakes_jobs

    def requires_vision(self, job_type: str) -> bool:
        """Check if a job type requires vision capability."""
        resolved = self.job_type_aliases.get(job_type, job_type)
        return resolved in self.validation_rules.require_vision_for


# =============================================================================
# EXCEPTIONS
# =============================================================================

class PolicyError(Exception):
    """Base exception for policy errors."""
    pass


class UnknownJobTypeError(PolicyError):
    """Raised when an unknown job type is requested."""
    def __init__(self, job_type: str, available: list[str]):
        self.job_type = job_type
        self.available = available
        super().__init__(
            f"Unknown job type: '{job_type}'. "
            f"Available: {', '.join(sorted(available)[:10])}..."
        )


class UnknownProviderError(PolicyError):
    """Raised when an unknown provider is specified."""
    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(
            f"Unknown provider: '{provider}'. "
            f"Available: {', '.join(p.value for p in Provider)}"
        )


class DataValidationError(PolicyError):
    """Raised when data violates policy rules."""
    def __init__(self, job_type: str, forbidden: list[str], found: list[str]):
        self.job_type = job_type
        self.forbidden = forbidden
        self.found = found
        super().__init__(
            f"Job '{job_type}' forbids data types {forbidden}, but found: {found}"
        )


class ProviderCapabilityError(PolicyError):
    """Raised when provider lacks required capability."""
    def __init__(self, provider: str, capability: str, job_type: str):
        self.provider = provider
        self.capability = capability
        self.job_type = job_type
        super().__init__(
            f"Provider '{provider}' does not support '{capability}' required for job '{job_type}'"
        )


# =============================================================================
# POLICY LOADING
# =============================================================================

_cached_policy: Optional[RoutingPolicy] = None
_policy_path: Optional[Path] = None


def load_routing_policy(
    path: Optional[str | Path] = None,
    force_reload: bool = False
) -> RoutingPolicy:
    """
    Load the routing policy from JSON file.
    
    Args:
        path: Path to policy JSON. Defaults to data/routing_policy.json
        force_reload: If True, reload even if cached
    
    Returns:
        RoutingPolicy instance
    
    Raises:
        PolicyError: If file not found or invalid JSON/schema
    """
    global _cached_policy, _policy_path
    
    if path is None:
        # Default path relative to this file's parent (app/llm/)
        # Policy lives in data/routing_policy.json
        base = Path(__file__).parent.parent.parent  # Orb/
        path = base / "data" / "routing_policy.json"
    else:
        path = Path(path)
    
    # Return cached if same path and not forcing reload
    if not force_reload and _cached_policy is not None and _policy_path == path:
        return _cached_policy
    
    if not path.exists():
        raise PolicyError(f"Policy file not found: {path}")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise PolicyError(f"Invalid JSON in policy file: {e}")
    
    try:
        policy = RoutingPolicy(**data)
    except Exception as e:
        raise PolicyError(f"Policy schema validation failed: {e}")
    
    _cached_policy = policy
    _policy_path = path
    
    return policy


def get_policy_for_job(
    job_type: str,
    policy: Optional[RoutingPolicy] = None
) -> JobPolicy:
    """
    Get the routing policy for a specific job type.
    
    Args:
        job_type: The job type or alias
        policy: Optional pre-loaded policy (loads default if None)
    
    Returns:
        JobPolicy for the job type
    
    Raises:
        UnknownJobTypeError: If job type not found
    """
    if policy is None:
        policy = load_routing_policy()
    
    try:
        return policy.get_job_policy(job_type)
    except PolicyError:
        available = policy.list_job_types() + list(policy.job_type_aliases.keys())
        raise UnknownJobTypeError(job_type, available)


def get_provider_for_job(
    job_type: str,
    policy: Optional[RoutingPolicy] = None
) -> tuple[Provider, Optional[Provider]]:
    """
    Get primary and review providers for a job type.
    
    Args:
        job_type: The job type or alias
        policy: Optional pre-loaded policy
    
    Returns:
        Tuple of (primary_provider, review_provider)
    """
    job_policy = get_policy_for_job(job_type, policy)
    return job_policy.primary_provider, job_policy.review_provider


def resolve_job_type(
    job_type: str,
    policy: Optional[RoutingPolicy] = None
) -> str:
    """
    Resolve a job type alias to its canonical form.
    
    Args:
        job_type: The job type or alias
        policy: Optional pre-loaded policy
    
    Returns:
        Canonical job type string
    """
    if policy is None:
        policy = load_routing_policy()
    
    return policy.job_type_aliases.get(job_type, job_type)


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_task_data(
    job_policy: JobPolicy,
    data_types: list[str],
    raise_on_error: bool = True
) -> tuple[bool, list[str]]:
    """
    Validate that data types are allowed for a job.
    
    Args:
        job_policy: The job policy to validate against
        data_types: List of data type strings present in the task
        raise_on_error: If True, raises DataValidationError on failure
    
    Returns:
        Tuple of (is_valid, list_of_violations)
    
    Raises:
        DataValidationError: If raise_on_error and violations found
    """
    violations = []
    
    for dt in data_types:
        if dt in job_policy.forbidden_data:
            violations.append(dt)
    
    is_valid = len(violations) == 0
    
    if not is_valid and raise_on_error:
        raise DataValidationError(
            job_type=job_policy.job_type,
            forbidden=job_policy.forbidden_data,
            found=violations
        )
    
    return is_valid, violations


def validate_provider_capability(
    job_policy: JobPolicy,
    policy: Optional[RoutingPolicy] = None,
    raise_on_error: bool = True
) -> tuple[bool, list[str]]:
    """
    Validate that the assigned provider has required capabilities.
    
    Args:
        job_policy: The job policy to validate
        policy: Optional pre-loaded routing policy
        raise_on_error: If True, raises on capability mismatch
    
    Returns:
        Tuple of (is_valid, list_of_missing_capabilities)
    
    Raises:
        ProviderCapabilityError: If raise_on_error and capability missing
    """
    if policy is None:
        policy = load_routing_policy()
    
    provider = job_policy.primary_provider.value
    missing = []
    
    # Check vision capability
    if job_policy.requires_vision:
        if provider not in policy.validation_rules.vision_capable_providers:
            missing.append("vision")
    
    # Check web search capability
    if job_policy.include_web_search:
        if provider not in policy.validation_rules.web_search_capable_providers:
            missing.append("web_search")
    
    # Check embeddings capability
    if job_policy.special_endpoint == "embeddings":
        if provider not in policy.validation_rules.embedding_capable_providers:
            missing.append("embeddings")
    
    is_valid = len(missing) == 0
    
    if not is_valid and raise_on_error:
        raise ProviderCapabilityError(
            provider=provider,
            capability=missing[0],
            job_type=job_policy.job_type
        )
    
    return is_valid, missing


def detect_data_types(
    content: str,
    attachments: Optional[list[dict]] = None,
    policy: Optional[RoutingPolicy] = None
) -> list[str]:
    """
    Detect data types present in content and attachments.
    
    Args:
        content: The text content
        attachments: List of attachment dicts with 'mime_type', 'filename', 'size'
        policy: Optional pre-loaded routing policy
    
    Returns:
        List of detected data type strings
    """
    if policy is None:
        policy = load_routing_policy()
    
    detected = set()
    detected.add("text")  # Always have text
    
    # Check content characteristics
    content_lower = content.lower()
    
    # Code detection (simple heuristics)
    code_indicators = ["def ", "function ", "class ", "import ", "const ", "let ", "var "]
    if any(ind in content for ind in code_indicators):
        detected.add("code")
    
    # Log detection
    log_indicators = ["traceback", "exception", "error:", "warning:", "[error]", "[warn]"]
    if any(ind in content_lower for ind in log_indicators):
        detected.add("logs")
    
    # JSON detection
    if content.strip().startswith("{") or content.strip().startswith("["):
        detected.add("structured_json")
    
    # Design doc detection
    design_indicators = ["architecture", "design doc", "specification", "rfc", "proposal"]
    if any(ind in content_lower for ind in design_indicators):
        detected.add("design_docs")
    
    # Large codebase detection (>50KB of code-like content)
    if len(content) > 51200 and "code" in detected:
        detected.add("large_codebase")
    
    # Attachment detection
    if attachments:
        for att in attachments:
            mime = att.get("mime_type", "")
            filename = att.get("filename", "")
            
            if mime.startswith("image/"):
                detected.add("images")
                detected.add("screenshots")  # Could refine this
            elif mime.startswith("video/"):
                detected.add("videos")
            elif mime.startswith("audio/"):
                detected.add("audio")
            elif mime == "application/pdf":
                detected.add("pdfs")
            elif filename.endswith((".json", ".yaml", ".yml", ".toml")):
                detected.add("structured_json")
            elif filename.endswith((".py", ".js", ".ts", ".java", ".c", ".cpp", ".go", ".rs")):
                detected.add("code")
    
    return list(detected)


# =============================================================================
# ROUTING DECISION
# =============================================================================

class RoutingDecision(BaseModel):
    """Complete routing decision for a task."""
    job_type: str
    resolved_job_type: str
    primary_provider: Provider
    primary_model: str
    review_provider: Optional[Provider] = None
    review_model: Optional[str] = None
    include_semantic_context: bool
    include_web_search: bool
    include_attachments: AttachmentMode
    max_chars: int
    temperature: float
    timeout_seconds: int
    retry_count: int
    requires_vision: bool
    special_endpoint: Optional[str] = None
    detected_data_types: list[str] = Field(default_factory=list)
    is_high_stakes: bool = False


def make_routing_decision(
    job_type: str,
    content: str = "",
    attachments: Optional[list[dict]] = None,
    policy: Optional[RoutingPolicy] = None
) -> RoutingDecision:
    """
    Make a complete routing decision for a task.
    
    This is the main entry point for the router. It:
    1. Resolves job type aliases
    2. Gets the job policy
    3. Validates data types
    4. Validates provider capabilities
    5. Returns a complete routing decision
    
    Args:
        job_type: The job type or alias
        content: The text content of the task
        attachments: Optional list of attachment metadata
        policy: Optional pre-loaded routing policy
    
    Returns:
        RoutingDecision with all routing parameters
    
    Raises:
        UnknownJobTypeError: If job type not found
        DataValidationError: If forbidden data present
        ProviderCapabilityError: If provider lacks capability
    """
    if policy is None:
        policy = load_routing_policy()
    
    # Resolve and get policy
    resolved = resolve_job_type(job_type, policy)
    job_policy = get_policy_for_job(resolved, policy)
    
    # Detect data types
    detected = detect_data_types(content, attachments, policy)
    
    # Validate data
    validate_task_data(job_policy, detected, raise_on_error=True)
    
    # Validate provider capabilities
    validate_provider_capability(job_policy, policy, raise_on_error=True)
    
    # Get provider configs
    primary_config = policy.get_provider_config(job_policy.primary_provider)
    
    review_model = None
    if job_policy.review_provider:
        review_config = policy.get_provider_config(job_policy.review_provider)
        review_model = review_config.model
    
    return RoutingDecision(
        job_type=job_type,
        resolved_job_type=resolved,
        primary_provider=job_policy.primary_provider,
        primary_model=primary_config.model,
        review_provider=job_policy.review_provider,
        review_model=review_model,
        include_semantic_context=job_policy.include_semantic_context,
        include_web_search=job_policy.include_web_search,
        include_attachments=job_policy.include_attachments,
        max_chars=job_policy.max_chars_to_provider,
        temperature=job_policy.temperature,
        timeout_seconds=job_policy.timeout_seconds,
        retry_count=job_policy.retry_count,
        requires_vision=job_policy.requires_vision,
        special_endpoint=job_policy.special_endpoint,
        detected_data_types=detected,
        is_high_stakes=policy.is_high_stakes(resolved)
    )


# =============================================================================
# CONVENIENCE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "Provider",
    "AttachmentMode",
    "DataType",
    "JobType",
    # Models
    "ProviderConfig",
    "DataTypeConfig",
    "PolicyDefaults",
    "JobPolicy",
    "ValidationRules",
    "RoutingPolicy",
    "RoutingDecision",
    # Exceptions
    "PolicyError",
    "UnknownJobTypeError",
    "UnknownProviderError",
    "DataValidationError",
    "ProviderCapabilityError",
    # Functions
    "load_routing_policy",
    "get_policy_for_job",
    "get_provider_for_job",
    "resolve_job_type",
    "validate_task_data",
    "validate_provider_capability",
    "detect_data_types",
    "make_routing_decision",
]