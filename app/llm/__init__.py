# FILE: app/llm/__init__.py
"""
LLM module exports.

v0.12.7: Added video analysis and is_video_mime_type exports.
v0.12.4: Added new vision functions for direct image Q&A.
"""

# ============== ROUTER EXPORTS ==============

from app.llm.router import call_llm, quick_chat, request_code, review_work
from app.llm.router import (
    analyze_with_vision,
    web_search_query,
    list_job_types,
    get_routing_info,
    is_policy_routing_enabled,
    enable_policy_routing,
)

# ============== SCHEMA EXPORTS ==============

from app.llm.schemas import LLMTask, LLMResult, JobType, Provider

# ============== VISION EXPORTS ==============

from app.llm.gemini_vision import (
    analyze_image,
    is_image_mime_type,
    ask_about_image,
    check_vision_available,
    get_vision_model_for_complexity,
    analyze_video,
)

# ============== FILE ANALYZER EXPORTS ==============

from app.llm.file_analyzer import (
    extract_text_content,
    detect_document_type,
    parse_cv_with_llm,
    generate_document_summary,
    is_video_mime_type,
    is_audio_mime_type,
)

# ============== CLIENT EXPORTS ==============

from app.llm.clients import (
    check_provider_availability,
    list_available_providers,
    get_embeddings,
)

# ============== OPTIONAL POLICY EXPORTS ==============

try:
    from app.llm.policy import (
        load_routing_policy,
        get_policy_for_job,
        make_routing_decision,
        RoutingPolicy,
        RoutingDecision,
        JobPolicy,
        PolicyError,
        UnknownJobTypeError,
        DataValidationError,
        ProviderCapabilityError,
    )
    _policy_exports = [
        "load_routing_policy",
        "get_policy_for_job",
        "make_routing_decision",
        "RoutingPolicy",
        "RoutingDecision",
        "JobPolicy",
        "PolicyError",
        "UnknownJobTypeError",
        "DataValidationError",
        "ProviderCapabilityError",
    ]
except ImportError:
    _policy_exports = []

# ============== ALL EXPORTS ==============

__all__ = [
    # Router
    "call_llm",
    "quick_chat",
    "request_code",
    "review_work",
    "analyze_with_vision",
    "web_search_query",
    "list_job_types",
    "get_routing_info",
    "is_policy_routing_enabled",
    "enable_policy_routing",
    # Schemas
    "LLMTask",
    "LLMResult",
    "JobType",
    "Provider",
    # Vision
    "analyze_image",
    "is_image_mime_type",
    "ask_about_image",
    "check_vision_available",
    "get_vision_model_for_complexity",
    "analyze_video",
    # File analyzer
    "extract_text_content",
    "detect_document_type",
    "parse_cv_with_llm",
    "generate_document_summary",
    "is_video_mime_type",
    "is_audio_mime_type",
    # Clients
    "check_provider_availability",
    "list_available_providers",
    "get_embeddings",
] + _policy_exports